#!/usr/bin/env python3
"""Compare initial XY offsets with the same offsets injected before physical step 151."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
REMOTE_ROOT = Path(os.environ.get('SPIDER_ROOT', '/mnt/sdc/dty_user/hand_to_robot_20261002'))
if SCRIPT.parent.name == 'scripts' and (SCRIPT.parent.parent / 'native_spider').is_dir():
    PROJECT_ROOT = SCRIPT.parent.parent
    SPIDER_DIR = PROJECT_ROOT / 'native_spider'
else:
    SPIDER_DIR = SCRIPT.parent
    PROJECT_ROOT = SPIDER_DIR.parent
PROTOCOL = PROJECT_ROOT / 'NATIVE_V2_LIVE_BRANCH_PROTOCOL.md'
TOTAL_STEPS = 300
SHIFT_STEP = 150
NUM_WORLDS = 6
PAIR_MARGIN = -0.005
WORLD_NAMES = ('N', 'P-', 'P+', 'R-', 'R+', 'R0')
OFFSETS = np.asarray(((0, 0), (-0.005, -0.005), (0.005, 0.005),
                      (-0.005, -0.005), (0.005, 0.005), (0, 0)), dtype=np.float64)
INITIAL_OFFSETS = OFFSETS.copy()
INITIAL_OFFSETS[3:] = 0
COPY_FIELDS = ('qpos qvel qacc time ctrl act act_dot qacc_warmstart qfrc_applied xfrc_applied energy mocap_pos mocap_quat xpos xquat xmat xipos ximat geom_xpos geom_xmat site_xpos site_xmat cacc cfrc_int cfrc_ext sensordata actuator_length actuator_velocity actuator_force ten_length ten_velocity').split()


def quat_angle_rad(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    """SO(3) geodesic angle for MuJoCo wxyz quaternions."""
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def read_inputs(spec_path: Path, controls_path: Path, reference_path: Path):
    spec = json.loads(spec_path.read_text())
    with np.load(controls_path) as archive:
        controls = np.asarray(archive['ctrl']).reshape(-1, 18).astype(np.float32)
    with np.load(reference_path) as archive:
        reference = np.asarray(archive['qpos'], dtype=np.float64)
        sim_dt = float(np.asarray(archive['sim_dt']).item()) if 'sim_dt' in archive else 0.01
    return spec, controls, reference, sim_dt


def preflight(args: argparse.Namespace) -> int:
    """CPU-only file/config checks; does not import simulator, torch, Warp, or renderers."""
    spec, controls, reference, sim_dt = read_inputs(args.spec, args.controls, args.reference)
    if spec.get('source_revision') != '44717007de41cbef7565dff7ff9f4453557a2d3d':
        raise ValueError('Unexpected SPIDER source revision')
    if int(spec.get('seed', args.seed)) != args.seed:
        raise ValueError('CLI seed must match the frozen spec seed')
    if controls.shape != (450, 18) or not np.isfinite(controls).all():
        raise ValueError(f'Expected finite saved controls (450,18), got {controls.shape}')
    if reference.ndim != 2 or reference.shape[1] != 25 or reference.shape[0] <= TOTAL_STEPS:
        raise ValueError(f'Reference must contain post-step rows 1..300, got {reference.shape}')
    if not np.isfinite(reference[: TOTAL_STEPS + 1]).all() or sim_dt <= 0:
        raise ValueError('Reference data or sim_dt is invalid')
    expected_offsets = np.asarray(((0, 0), (-.005, -.005), (.005, .005),
                                   (-.005, -.005), (.005, .005), (0, 0)), dtype=np.float64)
    if OFFSETS.shape != (NUM_WORLDS, 2) or not np.array_equal(OFFSETS, expected_offsets):
        raise ValueError('World order/XY offsets do not match the frozen protocol')
    record = {
        'status': 'PASS_NO_SIMULATION',
        'gpu_used': False,
        'simulation_steps': 0,
        'source_revision': spec['source_revision'],
        'seed': args.seed,
        'job_id': args.job_id,
        'controls_source': str(args.controls),
        'controls_shape': list(controls.shape),
        'controls_finite': True,
        'reference_source': str(args.reference),
        'reference_shape': list(reference.shape),
        'reference_finite_first_301_rows': True,
        'sim_dt_s': sim_dt,
        'world_order': list(WORLD_NAMES),
        'xy_offsets_m': OFFSETS.tolist(),
        'pair_margin_m': PAIR_MARGIN,
        'preflight_note': 'Array/config checks only. Does not execute physics or imply result readiness beyond these inputs.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


def validate_initial_boundary(before_q, after_q, before_v, after_v, before_t, after_t):
    changed = np.zeros(25, dtype=bool)
    changed[18:20] = True
    if not np.array_equal(before_q[:, ~changed], after_q[:, ~changed]):
        raise RuntimeError('Initial perturbation changed non-XY qpos')
    if not np.array_equal(before_v, after_v) or not np.array_equal(before_t, after_t):
        raise RuntimeError('Initial perturbation changed qvel or time')
    expected = before_q.copy()
    expected[:, 18:20] += INITIAL_OFFSETS.astype(expected.dtype)
    if not np.allclose(after_q[:, 18:20], expected[:, 18:20], rtol=0, atol=1e-7):
        raise RuntimeError('Initial XY positions differ from frozen offsets')


def validate_midshift_boundary(before_q, after_q, before_v, after_v, before_t, after_t):
    changed = np.zeros((NUM_WORLDS, 25), dtype=bool)
    changed[3:5, 18:20] = True
    if not np.array_equal(before_q[~changed], after_q[~changed]):
        raise RuntimeError('Mid-run intervention changed non-target qpos')
    if not np.array_equal(before_v, after_v) or not np.array_equal(before_t, after_t):
        raise RuntimeError('Mid-run intervention changed qvel or time')
    expected = before_q.copy()
    expected[3:5, 18:20] += OFFSETS[3:5].astype(expected.dtype)
    if not np.allclose(after_q[:, 18:20], expected[:, 18:20], rtol=0, atol=1e-7):
        raise RuntimeError('Mid-run XY positions differ from frozen offsets')


def compare_metrics(qpos, qvel, time_rows, rewards, initial_qpos, reference, sim_dt, seed, branch_qvel):
    # qpos row k is post-step k+1. The second block is physical/reference steps 151..300.
    sl = slice(150, 300)
    target = reference[151:301]
    actual = qpos[sl]
    pos_error = np.linalg.norm(actual[:, :, 18:21] - target[:, None, 18:21], axis=-1)
    rot_error = quat_angle_rad(actual[:, :, 21:25], target[None, :, 21:25].transpose(1, 0, 2))
    # The target orientation above broadcasts one reference qpos across worlds.
    if rot_error.shape != (150, NUM_WORLDS):
        raise RuntimeError(f'Unexpected orientation metric shape {rot_error.shape}')
    reward_window = rewards[sl]
    metrics = []
    for world, name in enumerate(WORLD_NAMES):
        metrics.append({
            'world': name,
            'world_index': world,
            'xy_offset_mm': (OFFSETS[world] * 1000).tolist(),
            'window_physical_steps': [151, 300],
            'samples': 150,
            'root_position_error_mean_m': float(pos_error[:, world].mean()),
            'root_position_error_max_m': float(pos_error[:, world].max()),
            'so3_error_mean_rad': float(rot_error[:, world].mean()),
            'so3_error_max_rad': float(rot_error[:, world].max()),
            'mean_step_reward': float(reward_window[:, world].mean()),
            'end_object_root_z_m': float(qpos[299, world, 20]),
            'end_height_delta_from_initial_m': float(qpos[299, world, 20] - initial_qpos[world, 20]),
            'step151_qvel_delta_l2': float(np.linalg.norm(qvel[150, world] - branch_qvel[world])),
            'step151_time_s': float(time_rows[150, world]),
        })
    pair_differences = []
    for direction, p_idx, r_idx in (('minus', 1, 3), ('plus', 2, 4)):
        p, r = metrics[p_idx], metrics[r_idx]
        pair_differences.append({
            'direction': direction,
            'P_world': p['world'], 'R_world': r['world'],
            'P': p, 'R': r,
            'R_minus_P': {
                'root_position_error_mean_m': r['root_position_error_mean_m'] - p['root_position_error_mean_m'],
                'so3_error_mean_rad': r['so3_error_mean_rad'] - p['so3_error_mean_rad'],
                'mean_step_reward': r['mean_step_reward'] - p['mean_step_reward'],
                'step151_qvel_delta_l2': r['step151_qvel_delta_l2'] - p['step151_qvel_delta_l2'],
            },
        })
    return metrics, pair_differences


def render_saved(args, spec, reference, trajectory_path):
    """Render saved states only; never replays controls through physics."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    from hydra import compose, initialize_config_dir
    from spider.config import Config, filter_config_fields, process_config
    from spider.io import load_data
    import spider.simulators.mjwp as sim
    from spider.viewers import render_image, setup_renderer

    repo = REMOTE_ROOT / 'repos/spider-native'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    overrides = [x for x in spec['author_cli_arguments'] if not x.startswith('seed=')]
    overrides.append(f'seed={args.seed}')
    with initialize_config_dir(config_dir=str(repo / 'examples/config'), version_base=None):
        cfg = Config(**filter_config_fields(dict(compose(config_name='default', overrides=overrides))))
    cfg = process_config(cfg)
    cfg.num_samples = NUM_WORLDS
    cfg.seed = args.seed
    original_setup = sim.setup_mj_model
    def compatible_setup(config):
        model = original_setup(config)
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        return model
    sim.setup_mj_model = compatible_setup
    model = sim.setup_mj_model(cfg)
    refs = load_data(cfg, cfg.data_path)
    renderer = setup_renderer(cfg, model)

    with np.load(trajectory_path) as archive:
        qpos = np.asarray(archive['qpos'])
        qvel = np.asarray(archive['qvel'])
        saved_ctrl = np.asarray(archive['ctrl'])
        relative_time = np.asarray(archive['relative_time'])
        initial_time = np.asarray(archive['initial_time'])
    render_data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    tile_w, tile_h = 480, 320
    video_path = args.output_dir / 'live_branch_6grid.mp4'
    writer = imageio.get_writer(str(video_path), fps=50, codec='libx264', quality=8, macro_block_size=16)
    try:
        for step in range(TOTAL_STEPS):
            if (step + 1) % 2:
                continue
            canvas = Image.new('RGB', (3 * tile_w, 2 * tile_h), (0, 0, 0))
            for world in range(NUM_WORLDS):
                if world >= 3 and step < SHIFT_STEP:
                    tile = Image.new('RGB', (tile_w, tile_h), (35, 35, 35))
                    ImageDraw.Draw(tile).text((12, 30), WORLD_NAMES[world] + ' inactive until step150', fill='white')
                    canvas.paste(tile, ((world % 3) * tile_w, (world // 3) * tile_h))
                    continue
                render_data.qpos[:] = qpos[step, world]
                render_data.qvel[:] = qvel[step, world]
                render_data.ctrl[:] = saved_ctrl[step]
                render_data.time = float(initial_time[world] + relative_time[step, world])
                mujoco.mj_forward(model, render_data)
                ref_row = refs[0][min(step + 1, len(refs[0]) - 1)]
                reference_data.qpos[:] = ref_row.detach().cpu().numpy()
                image = np.asarray(render_image(cfg, renderer, model, render_data, reference_data))
                sim_half = image[:, image.shape[1] // 2:]
                tile = Image.fromarray(sim_half).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                labels = ('N | no shift', 'P- | initial -5,-5 mm', 'P+ | initial +5,+5 mm',
                          'R- | before step 151: -5,-5 mm', 'R+ | at branch: +5,+5 mm', 'R0 | branch: zero shift')
                draw = ImageDraw.Draw(tile)
                draw.rectangle((4, 4, tile_w - 4, 24), fill=(0, 0, 0))
                draw.text((8, 7), labels[world], fill=(255, 255, 255))
                canvas.paste(tile, ((world % 3) * tile_w, (world // 3) * tile_h))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        if hasattr(renderer, 'close'):
            renderer.close()
    return video_path


def run_probe(args: argparse.Namespace) -> int:
    import mujoco
    import torch
    import warp as wp
    from hydra import compose, initialize_config_dir
    from spider.config import Config, filter_config_fields, process_config
    from spider.io import load_data
    import spider.simulators.mjwp as sim

    started = time.time()
    spec, controls, reference_np, sim_dt = read_inputs(args.spec, args.controls, args.reference)
    if spec.get('source_revision') != '44717007de41cbef7565dff7ff9f4453557a2d3d':
        raise ValueError('Unexpected SPIDER source revision')
    if int(spec.get('seed', args.seed)) != args.seed:
        raise ValueError('Seed differs from the source control spec')
    if controls.shape[0] < TOTAL_STEPS or controls.shape[1] != 18 or not np.isfinite(controls).all():
        raise ValueError('Need at least 300 finite 18D saved controls')
    if reference_np.shape[0] <= TOTAL_STEPS or reference_np.shape[1] != 25:
        raise ValueError('Reference must cover post-step rows 1..300')

    if args.render_only:
        trajectory_path = args.output_dir / 'trajectories.npz'
        if not trajectory_path.is_file():
            raise FileNotFoundError(f'No saved trajectory available for render-only: {trajectory_path}')
        video = render_saved(args, spec, reference_np, trajectory_path)
        result_path = args.output_dir / 'condition_result.json'
        record = json.loads(result_path.read_text()) if result_path.is_file() else {'status': 'TRAJECTORY_SAVED'}
        record['video'] = str(video)
        record['video_status'] = 'RENDERED_FROM_SAVED_STATES_NO_PHYSICS_REPLAY'
        record['render_repair_seconds'] = time.time() - started
        record['status'] = 'COMPLETE'
        result_path.write_text(json.dumps(record, indent=2) + '\n')
        (args.output_dir / 'result.json').write_text(json.dumps({'status': 'COMPLETE', 'job_id': args.job_id, 'condition_result': str(result_path)}) + '\n')
        (args.output_dir / 'progress.json').write_text(json.dumps({'stage': 'COMPLETE', 'completed_steps': TOTAL_STEPS}) + '\n')
        return 0

    if (args.output_dir / 'trajectories.npz').exists():
        raise FileExistsError('Saved trajectory already exists; use --render-only to repair video, not rerun physics.')

    repo = REMOTE_ROOT / 'repos/spider-native'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    overrides = [x for x in spec['author_cli_arguments'] if not x.startswith('seed=')]
    overrides.append(f'seed={args.seed}')
    with initialize_config_dir(config_dir=str(repo / 'examples/config'), version_base=None):
        cfg = Config(**filter_config_fields(dict(compose(config_name='default', overrides=overrides))))
    cfg = process_config(cfg)
    cfg.num_samples = NUM_WORLDS
    cfg.seed = args.seed
    original_setup_model = sim.setup_mj_model
    def compatible_setup_model(config):
        model = original_setup_model(config)
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        return model
    sim.setup_mj_model = compatible_setup_model
    reference_tensors = load_data(cfg, cfg.data_path)
    env = sim.setup_env(cfg, reference_tensors)
    sim.load_env_params(cfg, env, {'pair_margin': PAIR_MARGIN})

    initial_qpos_before = sim.get_qpos(cfg, env).detach().cpu().numpy().copy()
    initial_qvel_before = sim.get_qvel(cfg, env).detach().cpu().numpy().copy()
    initial_time_before = wp.to_torch(env.data_wp.time).detach().cpu().numpy().copy()
    q_initial_t = sim.get_qpos(cfg, env).detach().clone()
    offsets_t = torch.as_tensor(INITIAL_OFFSETS, dtype=q_initial_t.dtype, device=q_initial_t.device)
    q_initial_t[:, 18:20] += offsets_t
    wp.copy(env.data_wp.qpos, wp.from_torch(q_initial_t.contiguous()))
    initial_qpos_after = sim.get_qpos(cfg, env).detach().cpu().numpy().copy()
    initial_qvel_after = sim.get_qvel(cfg, env).detach().cpu().numpy().copy()
    initial_time_after = wp.to_torch(env.data_wp.time).detach().cpu().numpy().copy()
    validate_initial_boundary(initial_qpos_before, initial_qpos_after,
                              initial_qvel_before, initial_qvel_after,
                              initial_time_before, initial_time_after)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'progress.json').write_text(json.dumps({'stage': 'RUNNING', 'completed_steps': 0}) + '\n')
    qpos_rows = np.empty((TOTAL_STEPS, NUM_WORLDS, 25), dtype=np.float32)
    qvel_rows = np.empty((TOTAL_STEPS, NUM_WORLDS, 24), dtype=np.float32)
    time_rows = np.empty((TOTAL_STEPS, NUM_WORLDS), dtype=np.float64)
    rewards = np.empty((TOTAL_STEPS, NUM_WORLDS), dtype=np.float32)
    clone_arrays = {}
    clone_equality = {}
    midshift_qpos_before = midshift_qvel_before = midshift_time_before = None
    midshift_qpos_after = midshift_qvel_after = midshift_time_after = None
    for step in range(TOTAL_STEPS):
        if step == SHIFT_STEP:
            midshift_qpos_before = sim.get_qpos(cfg, env).detach().cpu().numpy().copy()
            midshift_qvel_before = sim.get_qvel(cfg, env).detach().cpu().numpy().copy()
            midshift_time_before = wp.to_torch(env.data_wp.time).detach().cpu().numpy().copy()
            for field in COPY_FIELDS:
                clone_arrays['before_clone_' + field] = wp.to_torch(getattr(env.data_wp, field)).detach().cpu().numpy().copy()
            sim.copy_sample_state(cfg, env, torch.tensor([0, 0, 0], device=cfg.device),
                                  torch.tensor([3, 4, 5], device=cfg.device))
            for field in COPY_FIELDS:
                value = wp.to_torch(getattr(env.data_wp, field)).detach().cpu().numpy().copy()
                clone_arrays['after_clone_' + field] = value
                clone_equality[field] = all(np.array_equal(value[0], value[w]) for w in (3, 4, 5))
            if not all(clone_equality.values()):
                raise RuntimeError('Official live copy field mismatch: ' + str(clone_equality))
            clone_qpos = sim.get_qpos(cfg, env).detach().cpu().numpy().copy()
            clone_qvel = sim.get_qvel(cfg, env).detach().cpu().numpy().copy()
            clone_time = wp.to_torch(env.data_wp.time).detach().cpu().numpy().copy()
            if not all(np.array_equal(clone_qpos[0], clone_qpos[w]) and
                       np.array_equal(clone_qvel[0], clone_qvel[w]) and
                       np.array_equal(clone_time[0], clone_time[w]) for w in (3, 4, 5)):
                raise RuntimeError('Live state branch did not copy qpos/qvel/time exactly')
            midshift_qpos_before = clone_qpos
            midshift_qvel_before = clone_qvel
            midshift_time_before = clone_time
            qshift = sim.get_qpos(cfg, env).detach().clone()
            shift_offsets = torch.as_tensor(OFFSETS[3:5], dtype=qshift.dtype, device=qshift.device)
            qshift[3:5, 18:20] += shift_offsets
            wp.copy(env.data_wp.qpos, wp.from_torch(qshift.contiguous()))
            midshift_qpos_after = sim.get_qpos(cfg, env).detach().cpu().numpy().copy()
            midshift_qvel_after = sim.get_qvel(cfg, env).detach().cpu().numpy().copy()
            midshift_time_after = wp.to_torch(env.data_wp.time).detach().cpu().numpy().copy()
            validate_midshift_boundary(midshift_qpos_before, midshift_qpos_after,
                                       midshift_qvel_before, midshift_qvel_after,
                                       midshift_time_before, midshift_time_after)

        control = torch.from_numpy(controls[step]).to(cfg.device)
        sim.step_env(cfg, env, control)
        q = sim.get_qpos(cfg, env).detach().cpu().numpy().copy()
        v = sim.get_qvel(cfg, env).detach().cpu().numpy().copy()
        t = wp.to_torch(env.data_wp.time).detach().cpu().numpy().copy()
        reward, _ = sim.get_reward(cfg, env, [row[step + 1] for row in reference_tensors])
        reward_np = reward.detach().cpu().numpy().copy()
        if not all(np.isfinite(a).all() for a in (q, v, t, reward_np)):
            raise RuntimeError(f'Nonfinite state/reward at physical step {step + 1}')
        qpos_rows[step] = q
        qvel_rows[step] = v
        time_rows[step] = t - initial_time_before
        rewards[step] = reward_np
        if (step + 1) % 25 == 0:
            (args.output_dir / 'progress.json').write_text(json.dumps({'stage': 'RUNNING', 'completed_steps': step + 1}) + '\n')

    trajectory_path = args.output_dir / 'trajectories.npz'
    np.savez_compressed(
        trajectory_path,
        qpos=qpos_rows, qvel=qvel_rows, relative_time=time_rows,
        ctrl=controls[:TOTAL_STEPS].astype(np.float32), reward=rewards,
        initial_qpos=initial_qpos_after.astype(np.float32), sim_dt_s=np.asarray(sim_dt),
        initial_time=initial_time_before,
    )
    np.savez_compressed(
        args.output_dir / 'boundary_state.npz',
        initial_qpos_before=initial_qpos_before, initial_qpos_after=initial_qpos_after,
        initial_qvel_before=initial_qvel_before, initial_qvel_after=initial_qvel_after,
        initial_time_before=initial_time_before, initial_time_after=initial_time_after,
        midshift_qpos_before=midshift_qpos_before, midshift_qpos_after=midshift_qpos_after,
        midshift_qvel_before=midshift_qvel_before, midshift_qvel_after=midshift_qvel_after,
        midshift_time_before=midshift_time_before, midshift_time_after=midshift_time_after,
        **clone_arrays,
    )
    (args.output_dir / 'progress.json').write_text(json.dumps({'stage': 'TRAJECTORY_SAVED', 'completed_steps': TOTAL_STEPS}) + '\n')

    metrics, pair_differences = compare_metrics(
        qpos_rows, qvel_rows, time_rows, rewards, initial_qpos_after,
        reference_np, sim_dt, args.seed, midshift_qvel_after)
    record = {
        'status': 'TRAJECTORY_SAVED',
        'protocol': 'NATIVE_V2_LIVE_BRANCH_PROTOCOL.md',
        'job_id': args.job_id,
        'seed': args.seed,
        'source_revision': spec['source_revision'],
        'controls_source': str(args.controls),
        'reference_source': str(args.reference),
        'trajectory': str(trajectory_path),
        'video': None,
        'video_status': 'PENDING_RENDER',
        'world_order': list(WORLD_NAMES),
        'initial_offsets_m': INITIAL_OFFSETS.tolist(),
        'step150_offsets_m': np.concatenate([np.zeros((3, 2)), OFFSETS[3:5], np.zeros((1, 2))]).tolist(),
        'pair_margin_m': PAIR_MARGIN,
        'execution': {'steps': TOTAL_STEPS, 'prefix_steps_before_R_shift': SHIFT_STEP,
                      'R_shift_applied_after_step': SHIFT_STEP, 'same_controls_all_worlds': True,
                      'optimizer_called': False, 'sim_dt_s': sim_dt},
        'initial_boundary_only_xy': True,
        'midshift_boundary_only_R_xy': True,
        'live_branch_qpos_qvel_time_exact': True,
        'official_copy_fields_exact': clone_equality,
        'reserved_prefix_is_not_used_for_comparison': True,
        'boundary_state': str(args.output_dir / 'boundary_state.npz'),
        'metrics_window': {'output_rows_zero_based': [150, 300], 'physical_steps': [151, 300],
                           'reference_rows': [151, 301], 'samples': 150},
        'worlds': metrics,
        'paired_deltas': pair_differences,
        'claim_boundary': 'Fixed nominal saved controls; descriptive simulated timing diagnostic only, not a causal explanation of robust3 planning failure or a task-success rate.',
        'elapsed_seconds': time.time() - started,
    }
    (args.output_dir / 'condition_result.json').write_text(json.dumps(record, indent=2) + '\n')
    csv_path = args.output_dir / 'direction_metrics.csv'
    with csv_path.open('w', newline='') as f:
        fields = ['direction', 'P_world', 'R_world', 'P_root_mean_m', 'R_root_mean_m', 'R_minus_P_root_mean_m',
                  'P_so3_mean_rad', 'R_so3_mean_rad', 'R_minus_P_so3_mean_rad',
                  'P_mean_step_reward', 'R_mean_step_reward', 'R_minus_P_mean_step_reward',
                  'P_step151_qvel_delta_l2', 'R_step151_qvel_delta_l2', 'R_minus_P_step151_qvel_delta_l2']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair in pair_differences:
            p, r, delta = pair['P'], pair['R'], pair['R_minus_P']
            writer.writerow({'direction': pair['direction'], 'P_world': pair['P_world'], 'R_world': pair['R_world'],
                'P_root_mean_m': p['root_position_error_mean_m'], 'R_root_mean_m': r['root_position_error_mean_m'],
                'R_minus_P_root_mean_m': delta['root_position_error_mean_m'],
                'P_so3_mean_rad': p['so3_error_mean_rad'], 'R_so3_mean_rad': r['so3_error_mean_rad'],
                'R_minus_P_so3_mean_rad': delta['so3_error_mean_rad'],
                'P_mean_step_reward': p['mean_step_reward'], 'R_mean_step_reward': r['mean_step_reward'],
                'R_minus_P_mean_step_reward': delta['mean_step_reward'],
                'P_step151_qvel_delta_l2': p['step151_qvel_delta_l2'], 'R_step151_qvel_delta_l2': r['step151_qvel_delta_l2'],
                'R_minus_P_step151_qvel_delta_l2': delta['step151_qvel_delta_l2']})

    # Close simulation resources before initializing a renderer for the saved trajectory.
    if hasattr(env, 'close'):
        env.close()
    try:
        video = render_saved(args, spec, reference_np, trajectory_path)
        record['video'] = str(video)
        record['video_status'] = 'RENDERED'
        record['status'] = 'COMPLETE'
    except Exception as exc:
        record['video_status'] = 'RENDER_FAILED_TRAJECTORY_PRESERVED'
        record['render_error'] = f'{type(exc).__name__}: {exc}'
    record['elapsed_seconds'] = time.time() - started
    (args.output_dir / 'condition_result.json').write_text(json.dumps(record, indent=2) + '\n')
    if record['status'] == 'COMPLETE':
        (args.output_dir / 'result.json').write_text(json.dumps({'status': 'COMPLETE', 'job_id': args.job_id, 'condition_result': str(args.output_dir / 'condition_result.json')}) + '\n')
    (args.output_dir / 'progress.json').write_text(json.dumps({'stage': record['video_status'], 'completed_steps': TOTAL_STEPS}) + '\n')
    print(json.dumps({'status': record['status'], 'video_status': record['video_status'],
                      'trajectory': str(trajectory_path), 'steps': TOTAL_STEPS}, indent=2))
    return 0 if record['status'] == 'COMPLETE' else 1


def main() -> int:
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
    if args.preflight:
        return preflight(args)
    return run_probe(args)


if __name__ == '__main__':
    raise SystemExit(main())
