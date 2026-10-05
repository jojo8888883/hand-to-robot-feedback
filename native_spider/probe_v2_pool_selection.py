#!/usr/bin/env python3
"""Frozen single-pool local versus persistent initial-uncertainty selection."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from probe_v2_live_branch import COPY_FIELDS, read_inputs
from probe_native_v2_position import make_grid, summarize_world

REMOTE_ROOT = Path(os.environ.get('SPIDER_ROOT', '/mnt/sdc/dty_user/hand_to_robot_20261002'))
REVISION = '44717007de41cbef7565dff7ff9f4453557a2d3d'
GROUPS = ('base', 'base_duplicate', 'local_selected', 'persistent_selected')
SCENARIOS = ('center', 'local_minus', 'local_plus', 'persistent_minus', 'persistent_plus')
GRID = make_grid()
WORLDS, PREFIX, HORIZON, TOTAL = 1024, 150, 300, 450


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def validate_inputs(args):
    spec, controls, reference, dt = read_inputs(args.spec, args.controls, args.reference)
    if spec.get('source_revision') != REVISION or int(spec.get('seed', args.seed)) != args.seed:
        raise ValueError('Source revision or seed differs from frozen inputs')
    if args.seed not in (0, 1, 2):
        raise ValueError('Only frozen seeds 0/1/2 are supported')
    if controls.shape != (450, 18) or not np.isfinite(controls).all():
        raise ValueError('Expected finite saved controls (450,18)')
    if reference.ndim != 2 or reference.shape[1] != 25 or len(reference) <= TOTAL:
        raise ValueError('Reference must cover post-step rows 1..450')
    if not np.isfinite(reference[:451]).all() or not np.isclose(dt, .01, rtol=0, atol=1e-12):
        raise ValueError('Nonfinite reference or unexpected sim_dt')
    return spec, controls, reference, dt


def preflight(args):
    spec, controls, reference, dt = validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {'status': 'PASS_NO_SIMULATION', 'gpu_used': False, 'simulation_steps': 0,
              'source_revision': REVISION, 'seed': args.seed, 'job_id': args.job_id,
              'controls_source': str(args.controls), 'controls_shape': list(controls.shape),
              'reference_source': str(args.reference), 'reference_shape': list(reference.shape),
              'sim_dt_s': dt, 'num_worlds': WORLDS, 'copy_fields': COPY_FIELDS,
              'group_order': GROUPS, 'scenario_order': SCENARIOS,
              'xy_offsets_m': GRID.tolist(), 'pair_margin_m': -.005,
              'candidate_center_slice': [150, 450], 'candidate_count': WORLDS,
              'reference_scoring_rows': [151, 451],
              'preflight_note': 'CPU array/spec validation only; no simulator import or physics.'}
    write_json(args.output_dir / 'preflight.json', record)
    print(json.dumps(record))
    return 0


def load_config(spec, seed):
    import mujoco
    from hydra import compose, initialize_config_dir
    from spider.config import Config, filter_config_fields, process_config
    import spider.simulators.mjwp as sim
    repo = REMOTE_ROOT / 'repos/spider-native'
    overrides = [x for x in spec['author_cli_arguments'] if not x.startswith('seed=')]
    overrides.append(f'seed={seed}')
    with initialize_config_dir(config_dir=str(repo / 'examples/config'), version_base=None):
        cfg = Config(**filter_config_fields(dict(compose(config_name='default', overrides=overrides))))
    cfg.num_samples = WORLDS
    cfg.seed = seed
    cfg = process_config(cfg)
    if cfg.horizon_steps != HORIZON or cfg.ctrl_steps != PREFIX or cfg.terminate_resample or cfg.contact_guidance:
        raise ValueError('Timing/resampling/contact-guidance differs from frozen preset')
    if cfg.perturb_force != 0 or cfg.perturb_torque != 0:
        raise ValueError('Frozen slice expects zero author external perturbation')
    original = sim.setup_mj_model
    def compatible(config):
        model = original(config)
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        return model
    sim.setup_mj_model = compatible
    return cfg


def build_record(args, spec, reference, dt):
    """Recover all summaries from saved arrays, including after render failure."""
    with np.load(args.output_dir / 'trajectories.npz') as data:
        qpos = data['qpos']
        initial = data['initial_qpos']
    with np.load(args.output_dir / 'scores.npz') as data:
        selected = {'local': int(data['local_index']), 'persistent': int(data['persistent_index'])}
    boundaries = json.loads((args.output_dir / 'bank_fields_exact.json').read_text())
    groups = []
    for group, name in enumerate(GROUPS):
        worlds = []
        for world in range(9):
            metrics = summarize_world(qpos[:, group, world], initial[world], reference)
            worlds.append({'world_index': world, 'xy_offset_m': GRID[world].tolist(),
                           'participated_in_scoring': world in (0, 4, 8), **metrics})
        groups.append({'name': name, 'worlds': worlds})
    return {'status': 'TRAJECTORY_SAVED', 'protocol': 'NATIVE_V2_POOL_SELECTION_PROTOCOL.md',
            'seed': args.seed, 'job_id': args.job_id, 'source_revision': spec['source_revision'],
            'controls_source': str(args.controls), 'reference_source': str(args.reference),
            'trajectory': str(args.output_dir / 'trajectories.npz'),
            'pool': str(args.output_dir / 'candidate_pool.npz'),
            'scores': str(args.output_dir / 'scores.npz'),
            'boundary_state': str(args.output_dir / 'boundary_state.npz'),
            'group_order': list(GROUPS), 'scenario_order': list(SCENARIOS),
            'selected_indices': selected, 'same_selection': selected['local'] == selected['persistent'],
            'groups': groups, 'bank_fields_exact': boundaries,
            'videos': {}, 'outputvideo': {}, 'video_status': 'PENDING_RENDER',
            'execution': {'prefix_steps': PREFIX, 'suffix_steps': HORIZON, 'steps': TOTAL,
                          'candidate_count': WORLDS, 'scoring_rollouts': 5,
                          'candidate_sampling_calls': 1, 'optimizer_iterations': 0,
                          'sim_dt_s': dt, 'pair_margin_m': -.005,
                          'prefix_history_shared_across_groups': True,
                          'actual_suffix_worlds': 36, 'last_scoring_step_uses_terminal_reward': True},
            'claim_boundary': 'Single fixed-pool simulated method slice; no complete MPC/recovery or task-success claim.'}


def render_saved(args, spec):
    """Render stored qpos only; never step physics or sample candidates."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    import spider.simulators.mjwp as sim
    cfg = load_config(spec, args.seed)
    model = sim.setup_mj_model(cfg)
    model.vis.global_.offwidth, model.vis.global_.offheight = 480, 320
    renderer = mujoco.Renderer(model, width=480, height=320)
    data = mujoco.MjData(model)
    with np.load(args.output_dir / 'trajectories.npz') as saved:
        qpos, qvel, controls = saved['qpos'], saved['qvel'], saved['ctrl']
        times = saved['relative_time'] + saved['initial_time'][None, None, :]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.median(qpos[:, :, :, 18:21].reshape(-1, 3), axis=0)
    camera.lookat[2] += .04
    camera.distance, camera.azimuth, camera.elevation = .65, 140, -25
    videos = {}
    try:
        for group, name in enumerate(GROUPS):
            video = args.output_dir / f'{name}_9grid.mp4'
            with imageio.get_writer(str(video), fps=50, codec='libx264', quality=8, macro_block_size=16) as writer:
                for step in range(1, TOTAL, 2):
                    canvas = Image.new('RGB', (1440, 960))
                    for world in range(9):
                        data.qpos[:] = qpos[step, group, world]
                        data.qvel[:] = qvel[step, group, world]
                        data.ctrl[:] = controls[group, step]
                        data.time = float(times[step, group, world])
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera)
                        tile = Image.fromarray(renderer.render().copy())
                        draw = ImageDraw.Draw(tile)
                        draw.rectangle((0, 0, 480, 38), fill='black')
                        xy = (GRID[world] * 1000).astype(int)
                        draw.text((7, 5), f'{name} | grid{world} xy={xy.tolist()}mm', fill='white')
                        draw.text((7, 21), f'poststep {step+1} | t={(step+1)*.01:.2f}s | ' +
                                  ('prefix' if step < PREFIX else 'actual selected suffix'), fill='white')
                        canvas.paste(tile, ((world % 3) * 480, (world // 3) * 320))
                    writer.append_data(np.asarray(canvas))
            videos[name] = str(video)
    finally:
        renderer.close()
    return videos


def finish(args, spec, reference, dt, started):
    record = build_record(args, spec, reference, dt)
    write_json(args.output_dir / 'condition_result.json', record)
    try:
        record['videos'] = render_saved(args, spec)
        record['outputvideo'] = record['videos']
        record['video_status'] = 'RENDERED_FROM_SAVED_STATES_NO_PHYSICS_REPLAY'
        record['status'] = 'COMPLETE'
    except Exception as exc:
        record['video_status'] = 'RENDER_FAILED_ARRAYS_PRESERVED'
        record['render_error'] = f'{type(exc).__name__}: {exc}'
    record['postprocess_elapsed_seconds'] = time.time() - started
    write_json(args.output_dir / 'condition_result.json', record)
    write_json(args.output_dir / 'progress.json', {'stage': record['status'], 'completed_steps': TOTAL})
    if record['status'] == 'COMPLETE':
        write_json(args.output_dir / 'result.json', {'status': 'COMPLETE', 'job_id': args.job_id,
                   'condition_result': str(args.output_dir / 'condition_result.json')})
    print(json.dumps({'status': record['status'], 'selected_indices': record['selected_indices']}))
    return 0 if record['status'] == 'COMPLETE' else 1


def run(args):
    started = time.time()
    spec, controls, reference, dt = validate_inputs(args)
    if not args.render_only and any((args.output_dir / name).exists() for name in
                                   ('prefix.npz', 'candidate_pool.npz', 'scores.npz', 'trajectories.npz')):
        raise FileExistsError('Physics/candidate arrays already exist; do not rerun. Use --render-only after complete trajectory save.')
    repo = REMOTE_ROOT / 'repos/spider-native'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if args.render_only:
        return finish(args, spec, reference, dt, started)

    import torch
    import warp as wp
    import spider.simulators.mjwp as sim
    from spider.io import load_data
    from spider.optimizers.sampling import sample_ctrls
    cfg = load_config(spec, args.seed)
    refs = load_data(cfg, cfg.data_path)
    if not np.array_equal(refs[0][:451].detach().cpu().numpy(), reference[:451].astype(np.float32)):
        raise ValueError('Actual loader reference differs from saved reference')
    env = sim.setup_env(cfg, refs)
    if env.model_cpu.neq or cfg.nq != 25 or cfg.nv != 24 or cfg.nu != 18:
        raise ValueError('Current 31-field bank expects the frozen 25/24/18 scene without equality')
    sim.load_env_params(cfg, env, {'pair_margin': -.005})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def progress(stage, count):
        write_json(args.output_dir / 'progress.json', {'stage': stage, 'completed_steps': count})
    def tensor(field):
        return wp.to_torch(getattr(env.data_wp, field))
    def cpu(value):
        return value.detach().cpu().numpy().copy()
    def snapshot(indices):
        return {field: tensor(field)[indices].clone() for field in COPY_FIELDS}
    def restore(state):
        wp.synchronize()
        exact = {}
        for field in COPY_FIELDS:
            value = tensor(field)
            expected = state[field].expand_as(value)
            value.copy_(expected)
        torch.cuda.synchronize(torch.device(env.device))
        for field in COPY_FIELDS:
            exact[field] = bool(torch.equal(tensor(field), state[field].expand_as(tensor(field))))
        if not all(exact.values()):
            raise RuntimeError('31-field bank restore mismatch: ' + str(exact))
        return exact
    def reward_at(step, terminal=False):
        fn = sim.get_terminal_reward if terminal else sim.get_reward
        reward, _ = fn(cfg, env, [ref[step] for ref in refs])
        if not bool(torch.isfinite(reward).all()):
            raise RuntimeError(f'Nonfinite reward at reference step {step}')
        return reward
    def collect(indices):
        arrays = tuple(cpu(tensor(field)[indices]) for field in ('qpos', 'qvel', 'time'))
        if not all(np.isfinite(a).all() for a in arrays):
            raise RuntimeError('Nonfinite physical trajectory')
        return arrays

    grid_indices = np.arange(WORLDS) % 9
    q_initial = tensor('qpos').clone()
    before_initial = snapshot(slice(0, 9))
    q_initial[:, 18:20] += torch.as_tensor(GRID[grid_indices], device=cfg.device, dtype=q_initial.dtype)
    tensor('qpos').copy_(q_initial)
    torch.cuda.synchronize(torch.device(env.device))
    after_initial = snapshot(slice(0, 9))
    expected_initial = before_initial['qpos'].clone()
    expected_initial[:, 18:20] += torch.as_tensor(GRID, device=cfg.device, dtype=q_initial.dtype)
    if not torch.equal(after_initial['qpos'], expected_initial):
        raise RuntimeError('Initial XY grid mismatch')
    for field in COPY_FIELDS:
        if field != 'qpos' and not torch.equal(before_initial[field], after_initial[field]):
            raise RuntimeError('Initialization changed non-XY state: ' + field)
    prefix_q, prefix_v, prefix_t, prefix_r = [], [], [], []
    for step in range(PREFIX):
        sim.step_env(cfg, env, torch.as_tensor(controls[step], device=cfg.device))
        q, v, t = collect(slice(0, 9))
        prefix_q.append(q); prefix_v.append(v); prefix_t.append(t)
        prefix_r.append(cpu(reward_at(step + 1)[:9]))
        if (step + 1) % 25 == 0:
            progress('PREFIX', step + 1)
    bank = snapshot(slice(0, 9))
    np.savez_compressed(args.output_dir / 'prefix.npz', qpos=np.stack(prefix_q), qvel=np.stack(prefix_v),
                        time=np.stack(prefix_t), reward=np.stack(prefix_r), initial_qpos=cpu(after_initial['qpos']),
                        initial_time=cpu(after_initial['time']), xy_offsets_m=GRID)
    boundary_arrays = {'initial_' + f: cpu(v) for f, v in after_initial.items()}
    boundary_arrays.update({'bank_' + f: cpu(v) for f, v in bank.items()})
    np.savez_compressed(args.output_dir / 'boundary_state.npz', **boundary_arrays)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    center_controls = torch.as_tensor(controls[PREFIX:TOTAL], device=cfg.device)
    pool = sample_ctrls(cfg, center_controls, {'global_noise_scale': 1.0})
    pool[0].copy_(center_controls)
    if tuple(pool.shape) != (1024, 300, 18) or not bool(torch.isfinite(pool).all()):
        raise RuntimeError('Invalid official candidate pool')
    pool_np = cpu(pool)
    np.savez_compressed(args.output_dir / 'candidate_pool.npz', ctrl=pool_np, center_ctrl=controls[PREFIX:TOTAL],
                        seed=np.asarray(args.seed), noise_scale=cpu(cfg.noise_scale), knot_steps=np.asarray(cfg.knot_steps))
    exact_checks, mean_rewards, step_rewards, final_qpos = {}, [], [], []
    for scenario, name in enumerate(SCENARIOS):
        source = (4, 4, 4, 0, 8)[scenario]
        state = {f: value[source:source+1].clone() for f, value in bank.items()}
        if scenario in (1, 2):
            state['qpos'][:, 18:20] += -.005 if scenario == 1 else .005
        exact_checks[name] = restore(state)
        write_json(args.output_dir / 'bank_fields_exact.json', exact_checks)
        total = torch.zeros(WORLDS, device=cfg.device)
        rows = []
        for step in range(HORIZON):
            sim.step_env(cfg, env, pool[:, step])
            rew = reward_at(PREFIX + step + 1, terminal=step == HORIZON - 1)
            total += rew
            rows.append(cpu(rew))
            if (step + 1) % 50 == 0:
                progress('SCORING_' + name, step + 1)
        mean_rewards.append(cpu(total / HORIZON))
        step_rewards.append(np.stack(rows))
        final_qpos.append(cpu(tensor('qpos')))
        if not np.isfinite(final_qpos[-1]).all():
            raise RuntimeError('Nonfinite scoring final qpos')
    if not np.array_equal(cpu(pool), pool_np):
        raise RuntimeError('Shared candidate pool was modified during scoring')
    raw_rewards = np.stack(mean_rewards)
    local_score = np.min(raw_rewards[[0, 1, 2]], axis=0)
    persistent_score = np.min(raw_rewards[[0, 3, 4]], axis=0)
    local_index, persistent_index = int(np.argmax(local_score)), int(np.argmax(persistent_score))
    np.savez_compressed(args.output_dir / 'scores.npz', reward=raw_rewards, step_reward=np.stack(step_rewards),
                        final_qpos=np.stack(final_qpos), local_score=local_score, persistent_score=persistent_score,
                        local_order=np.argsort(-local_score, kind='stable'),
                        persistent_order=np.argsort(-persistent_score, kind='stable'),
                        local_index=np.asarray(local_index), persistent_index=np.asarray(persistent_index),
                        selected_ctrl=pool_np[[local_index, persistent_index]], scenario_order=np.asarray(SCENARIOS))
    selected_controls = np.stack([controls[PREFIX:TOTAL], controls[PREFIX:TOTAL],
                                  pool_np[local_index], pool_np[persistent_index]])
    eval_indices = np.arange(WORLDS) % 9
    eval_state = {f: value[eval_indices].clone() for f, value in bank.items()}
    exact_checks['evaluation_all1024'] = restore(eval_state)
    write_json(args.output_dir / 'bank_fields_exact.json', exact_checks)
    qrows, vrows, trows, rrows = [], [], [], []
    for step in range(HORIZON):
        ctrl = np.repeat(selected_controls[0, step][None], WORLDS, axis=0)
        ctrl[:36] = np.repeat(selected_controls[:, step], 9, axis=0)
        sim.step_env(cfg, env, torch.as_tensor(ctrl, device=cfg.device))
        q, v, t = collect(slice(0, 36))
        qrows.append(q.reshape(4, 9, 25)); vrows.append(v.reshape(4, 9, 24)); trows.append(t.reshape(4, 9))
        rrows.append(cpu(reward_at(PREFIX + step + 1, terminal=step == HORIZON - 1)[:36]).reshape(4, 9))
        if (step + 1) % 25 == 0:
            progress('EVALUATION', step + 1)
    prefix = lambda rows: np.repeat(np.stack(rows)[:, None], 4, axis=1)
    all_q = np.concatenate([prefix(prefix_q), np.stack(qrows)])
    all_v = np.concatenate([prefix(prefix_v), np.stack(vrows)])
    all_t = np.concatenate([prefix(prefix_t), np.stack(trows)])
    all_r = np.concatenate([prefix(prefix_r), np.stack(rrows)])
    all_controls = np.concatenate([np.repeat(controls[:PREFIX][None], 4, axis=0), selected_controls], axis=1)
    initial_time = cpu(after_initial['time'])
    increments = np.diff(np.concatenate([np.broadcast_to(initial_time, (1, 4, 9)), all_t]), axis=0)
    if not np.allclose(increments, dt, rtol=1e-3, atol=1e-6):
        raise RuntimeError('Physical time reset or wrong step duration')
    write_json(args.output_dir / 'bank_fields_exact.json', exact_checks)
    np.savez_compressed(args.output_dir / 'trajectories.npz', qpos=all_q, qvel=all_v,
                        relative_time=all_t - initial_time[None, None], initial_time=initial_time,
                        ctrl=all_controls, reward=all_r, initial_qpos=cpu(after_initial['qpos']),
                        xy_offsets_m=GRID, group_order=np.asarray(GROUPS), sim_dt_s=np.asarray(dt),
                        physical_reference_steps=np.arange(1, TOTAL+1))
    progress('TRAJECTORY_SAVED', TOTAL)
    write_json(args.output_dir / 'physics_execution.json', {'seed': args.seed, 'source_revision': REVISION,
               'simulation_steps': PREFIX + 5*HORIZON + HORIZON, 'world_step_count': WORLDS*(PREFIX+6*HORIZON),
               'sampling_calls': 1, 'physics_elapsed_seconds': time.time()-started,
               'config': {'knot_steps': cfg.knot_steps, 'horizon_steps': cfg.horizon_steps,
                          'num_samples': cfg.num_samples, 'use_torch_compile': cfg.use_torch_compile,
                          'terminal_rew_scale': cfg.terminal_rew_scale}, 'margin_m': -.005})
    del env, bank, pool
    torch.cuda.empty_cache()
    return finish(args, spec, reference, dt, time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    # Paths remain valid after changing cwd into the official repository.
    for name in ('spec', 'controls', 'reference', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return preflight(args) if args.preflight else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
