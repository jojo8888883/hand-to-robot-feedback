#!/usr/bin/env python3
"""Frozen observed-object feedback coverage with two matched official searches."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from probe_v2_pool_selection import REMOTE_ROOT, REVISION, GRID, validate_inputs, load_config, write_json
from probe_v2_live_branch import COPY_FIELDS, quat_angle_rad
from probe_native_v2_position import summarize_world

GROUPS = ('base', 'base_duplicate', 'nominal_object', 'observed_object')
ARMS = ('nominal_object', 'observed_object')
ACTUAL_BANKS = {'pour_tube': 5, 'wipe_board': 1}
ESTIMATE_INDICES = (147, 149)
PREFIX, HORIZON, TOTAL, WORLDS = 150, 300, 450, 1024


BASIC_FIELDS = ('qpos qvel time ctrl act qacc_warmstart qfrc_applied xfrc_applied mocap_pos mocap_quat').split()


def quat_rotation_wxyz(quaternion):
    """Body-to-world rotation for the scene's free-joint wxyz quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError('Expected a finite nonzero wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def task_and_bank(spec):
    if int(spec.get('original_control_seed', 0)) != 0:
        raise ValueError('Frozen coverage recovery reuses only the seed0 original controls')
    tasks = [x.split('=', 1)[1] for x in spec['author_cli_arguments'] if x.startswith('task=')]
    if len(tasks) != 1 or tasks[0] not in ACTUAL_BANKS:
        raise ValueError('Expected exactly one frozen pour_tube or wipe_board task in author_cli_arguments')
    return tasks[0], ACTUAL_BANKS[tasks[0]]


def make_planning_inputs(nominal_qpos, nominal_qvel, actual_qpos, actual_robot_qvel, estimated_object_qvel):
    """Shared actual robot, with either nominal object or causal observed object."""
    nq, nv, aq, rv, ev = [np.asarray(x, dtype=np.float32) for x in
                         (nominal_qpos, nominal_qvel, actual_qpos, actual_robot_qvel, estimated_object_qvel)]
    if (nq.shape, nv.shape, aq.shape, rv.shape, ev.shape) != ((25,), (24,), (25,), (18,), (6,)):
        raise ValueError('Expected nominal qpos25/qvel24, actual qpos25/robot qvel18, estimated object qvel6')
    if not all(np.isfinite(x).all() for x in (nq, nv, aq, rv, ev)):
        raise ValueError('Nonfinite planning inputs')
    nominal_pose = aq.copy()
    nominal_pose[18:] = nq[18:]
    # Nominal object pose and body-local angular velocity stay in the same frame.
    nominal_velocity = np.concatenate([rv, nv[18:]])
    observed_velocity = np.concatenate([rv, ev])
    return {'nominal_object': {'qpos': nominal_pose, 'qvel': nominal_velocity},
            'observed_object': {'qpos': aq.copy(), 'qvel': observed_velocity}}


def estimate_object_velocity(object_poses, timestamps):
    """Causal secant/log estimate; accepts only two object poses and their times."""
    poses, times = np.asarray(object_poses, dtype=np.float64), np.asarray(timestamps, dtype=np.float64)
    if poses.shape != (2, 7) or times.shape != (2,) or not np.isfinite(poses).all() or not np.isfinite(times).all():
        raise ValueError('Expected finite two xyz/wxyz poses and two timestamps')
    delta_t = float(times[1] - times[0])
    if delta_t <= 0:
        raise ValueError('Estimator timestamps must increase')
    qs = poses[:, 3:].copy()
    norms = np.linalg.norm(qs, axis=1)
    if np.any(norms == 0):
        raise ValueError('Estimator quaternion must be nonzero')
    q0, q1 = qs / norms[:, None]
    # q1 * conjugate(q0) represents R150 R148^-1 in the world frame.
    w0, v0 = q0[0], -q0[1:]
    w1, v1 = q1[0], q1[1:]
    relative = np.concatenate([[w1*w0 - np.dot(v1, v0)],
                               w1*v0 + w0*v1 + np.cross(v1, v0)])
    relative /= np.linalg.norm(relative)
    if relative[0] < 0:
        relative *= -1
    sine = float(np.linalg.norm(relative[1:]))
    if sine < 1e-12:
        rotvec_world = 2*relative[1:]
    else:
        rotvec_world = (2*np.arctan2(sine, relative[0])/sine)*relative[1:]
    linear_world = (poses[1, :3] - poses[0, :3])/delta_t
    angular_world = rotvec_world/delta_t
    angular_body = quat_rotation_wxyz(q1).T @ angular_world
    return np.concatenate([linear_world, angular_body]), linear_world, angular_world, delta_t


def accepted_or_original(candidate, original, succeeded):
    """A failed official search never changes the executed control sequence."""
    candidate = np.asarray(candidate, dtype=np.float32)
    original = np.asarray(original, dtype=np.float32)
    if candidate.shape != (300, 18) or original.shape != (300, 18):
        raise ValueError('Expected candidate and original controls (300,18)')
    if not np.isfinite(candidate).all() or not np.isfinite(original).all():
        raise ValueError('Nonfinite search/original controls are an engineering failure')
    return (candidate if bool(succeeded) else original).copy()


def preflight(args):
    spec, controls, reference, dt = validate_inputs(args)
    task, actual_bank = task_and_bank(spec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {'status': 'PASS_NO_SIMULATION', 'gpu_used': False, 'simulation_steps': 0,
              'source_revision': REVISION, 'task': task, 'seed': args.seed, 'base_controls_seed': 0, 'job_id': args.job_id,
              'controls_source': str(args.controls), 'controls_shape': list(controls.shape),
              'reference_source': str(args.reference), 'reference_shape': list(reference.shape),
              'sim_dt_s': dt, 'num_worlds': WORLDS, 'group_order': list(GROUPS),
              'copy_fields': COPY_FIELDS, 'actual_bank_index': actual_bank,
              'object_pose_bank_indices': {'nominal_object': 4, 'observed_object': actual_bank},
              'object_velocity_sources': {'nominal_object': 'BANK4_BODY_LOCAL_UNCHANGED', 'observed_object': 'ACTUAL_PREFIX148/150_POSE_TIME_ONLY'},
              'estimator_prefix_indices': list(ESTIMATE_INDICES), 'estimator_physical_steps': [148, 150],
              'robot_velocity_bank_index': actual_bank, 'planning_warmstart': 'ZERO_ALL_ARMS',
              'reconstruction_api': 'mujoco_warp.reset_data/write_basic_inputs/forward',
              'execution_bank_index': actual_bank, 'warmstart_slice': [150, 450],
              'optimizer_reference_slice': [151, 451], 'actual_suffix_steps': [151, 450],
              'failed_search_execution': 'FALLBACK_ORIGINAL', 'outer_revert': False,
              'preflight_note': 'CPU input/array checks only; no simulator imported or stepped.'}
    write_json(args.output_dir / 'preflight.json', record)
    print(json.dumps(record))
    return 0


def build_record(args, spec, reference, dt):
    with np.load(args.output_dir / 'trajectories.npz') as saved:
        qpos, initial = saved['qpos'], saved['initial_qpos']
    task, actual_bank = task_and_bank(spec)
    optimizations = json.loads((args.output_dir / 'optimizations.json').read_text())
    exact = json.loads((args.output_dir / 'bank_fields_exact.json').read_text())
    groups = []
    for index, name in enumerate(GROUPS):
        actual = qpos[:, index, 0]
        metrics = summarize_world(actual, initial[0], reference)
        local, target = actual[150:391].astype(np.float64), reference[151:392]
        pos = np.linalg.norm(local[:, 18:21] - target[:, 18:21], axis=1)
        rot = quat_angle_rad(local[:, 21:25], target[:, 21:25])
        metrics['local_reference_window'] = {
            'physical_steps_start_end_inclusive': [151, 391], 'samples': 241,
            'root_position_error_mean_m': float(pos.mean()),
            'root_position_error_mean_cm': float(pos.mean()*100),
            'root_position_error_max_m': float(pos.max()),
            'so3_error_mean_rad': float(rot.mean()), 'so3_error_max_rad': float(rot.max())}
        groups.append({'name': name, 'worlds': [{'world_index': 0, 'source_bank_index': actual_bank,
                                               'xy_offset_m': GRID[actual_bank].tolist(), **metrics}]})
    return {'status': 'TRAJECTORY_SAVED', 'protocol': 'NATIVE_V2_FEEDBACK_COVERAGE_PROTOCOL.md',
            'seed': args.seed, 'base_controls_seed': 0, 'task': task, 'job_id': args.job_id, 'source_revision': spec['source_revision'],
            'controls_source': str(args.controls), 'reference_source': str(args.reference),
            'spec_source': str(args.spec), 'run_spec': spec,
            'trajectory': str(args.output_dir / 'trajectories.npz'),
            'boundary_state': str(args.output_dir / 'boundary_state.npz'),
            'group_order': list(GROUPS), 'groups': groups, 'optimizations': optimizations,
            'bank_fields_exact': exact,
            'reconstruction': json.loads((args.output_dir / 'reconstruction_checks.json').read_text()),
            'planning_inputs': {arm: str(args.output_dir / f'reconstruction_{arm}.npz') for arm in ARMS},
            'velocity_estimation': json.loads((args.output_dir / 'causal_velocity_estimate.json').read_text()),
            'videos': {}, 'outputvideo': {}, 'video_status': 'PENDING_RENDER',
            'execution': {'steps': 450, 'shared_prefix_steps': 150, 'actual_suffix_steps': 300,
                          'saved_worlds': len(GROUPS), 'num_worlds': WORLDS, 'execution_bank_index': actual_bank,
                          'sim_dt_s': dt, 'pair_margin_m': -.005, 'outer_revert': False,
                          'predicted_snapshot_used_for_execution': False},
            'primary_metric': 'LAST50_REAL_REFERENCE_TRACKING_ALL_STEPS',
            'lift_metric_interpretation': 'HEIGHT_DESCRIPTION_ONLY_NOT_RECOVERY_GATE',
            'claim_boundary': 'Actual robot proprioception in both arms, nominal versus observed object pose/causal velocity; extra search computation on two development clips, not visual estimation or full task success.'}


def render_saved(args, spec):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    import spider.simulators.mjwp as sim
    cfg = load_config(spec, args.seed)
    task, actual_bank = task_and_bank(spec)
    model = sim.setup_mj_model(cfg)
    model.vis.global_.offwidth, model.vis.global_.offheight = 720, 480
    renderer = mujoco.Renderer(model, width=720, height=480)
    data = mujoco.MjData(model)
    with np.load(args.output_dir / 'trajectories.npz') as saved:
        qpos, qvel, controls = saved['qpos'], saved['qvel'], saved['ctrl']
        times = saved['relative_time'] + saved['initial_time'][None, None, :]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.median(qpos[:, :, 0, 18:21].reshape(-1, 3), axis=0)
    camera.lookat[2] += .04
    camera.distance, camera.azimuth, camera.elevation = .65, 140, -25
    video = args.output_dir / 'feedback_coverage_4grid.mp4'
    try:
        with imageio.get_writer(str(video), fps=50, codec='libx264', quality=8, macro_block_size=16) as writer:
            for step in range(1, TOTAL, 2):
                canvas = Image.new('RGB', (1440, 960))
                for group, name in enumerate(GROUPS):
                    data.qpos[:] = qpos[step, group, 0]
                    data.qvel[:] = qvel[step, group, 0]
                    data.ctrl[:] = controls[group, step]
                    data.time = float(times[step, group, 0])
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera)
                    tile = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(tile)
                    draw.rectangle((0, 0, 720, 40), fill='black')
                    draw.text((8, 5), f'{task} | {name} | initial xy={GRID[actual_bank,0]*1000:+.0f},{GRID[actual_bank,1]*1000:+.0f}mm', fill='white')
                    phase = 'shared actual prefix' if step < PREFIX else 'actual suffix'
                    if step >= 391:
                        phase += ' | reference padding'
                    draw.text((8, 22), f'poststep {step+1} | t={(step+1)*.01:.2f}s | {phase}', fill='white')
                    canvas.paste(tile, ((group % 2)*720, (group // 2)*480))
                writer.append_data(np.asarray(canvas))
    finally:
        renderer.close()
    return str(video)


def finish(args, spec, reference, dt):
    record = build_record(args, spec, reference, dt)
    write_json(args.output_dir / 'condition_result.json', record)
    try:
        video = render_saved(args, spec)
        record['video'] = video
        record['videos'] = {'comparison_2x2': video}
        record['outputvideo'] = {group: video for group in GROUPS}
        record['video_status'] = 'RENDERED_FROM_SAVED_STATES_NO_PHYSICS_REPLAY'
        record['status'] = 'COMPLETE'
    except Exception as exc:
        record['video_status'] = 'RENDER_FAILED_ARRAYS_PRESERVED'
        record['render_error'] = f'{type(exc).__name__}: {exc}'
    write_json(args.output_dir / 'condition_result.json', record)
    write_json(args.output_dir / 'progress.json', {'stage': record['status'], 'completed_steps': 450})
    if record['status'] == 'COMPLETE':
        write_json(args.output_dir / 'result.json', {'status': 'COMPLETE', 'job_id': args.job_id,
                   'condition_result': str(args.output_dir / 'condition_result.json')})
    print(json.dumps({'status': record['status'], 'optimizations': record['optimizations']}))
    return 0 if record['status'] == 'COMPLETE' else 1


def run(args):
    started = time.perf_counter()
    spec, controls, reference, dt = validate_inputs(args)
    task, actual_bank = task_and_bank(spec)
    if not args.render_only and any((args.output_dir / name).exists() for name in
                                   ('prefix.npz', 'optimizations.json', 'trajectories.npz')):
        raise FileExistsError('Existing physical/search artifacts must be preserved; complete arrays allow --render-only.')
    repo = REMOTE_ROOT / 'repos/spider-native'
    os.environ['MUJOCO_GL'], os.environ['PYOPENGL_PLATFORM'] = 'egl', 'egl'
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if args.render_only:
        return finish(args, spec, reference, dt)

    import torch
    import warp as wp
    import mujoco
    import spider.simulators.mjwp as sim
    from spider.io import load_data
    from spider.optimizers.sampling_fast import make_rollout_fn_fast, make_optimize_once_fn_fast, make_optimize_fn_fast
    cfg = load_config(spec, args.seed)
    if cfg.num_dr != 1 or tuple(cfg.xy_offset_range) != (0., 0.) or cfg.max_num_iterations != 4:
        raise ValueError('Expected the frozen nominal DR1/offset0/4-iteration preset')
    cfg.env_params_list = [[{'xy_offset': 0., 'pair_margin': -.005}] for _ in range(4)]
    refs = load_data(cfg, cfg.data_path)
    if not np.array_equal(refs[0][:451].detach().cpu().numpy(), reference[:451].astype(np.float32)):
        raise ValueError('Loaded reference differs from frozen saved reference')
    env = sim.setup_env(cfg, refs)
    if env.model_cpu.neq or (cfg.nq, cfg.nv, cfg.nu) != (25, 24, 18):
        raise ValueError('31-field bank expects the current 25/24/18 scene without equality')
    object_sites, robot_sites = [], []
    for sid in range(env.model_cpu.nsite):
        name = mujoco.mj_id2name(env.model_cpu, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name and name.startswith('trace'):
            (object_sites if 'object' in name else robot_sites).append(sid)
    cfg.trace_site_ids = object_sites + robot_sites
    sim.load_env_params(cfg, env, {'pair_margin': -.005})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact_checks = {}
    def tensor(field):
        return wp.to_torch(getattr(env.data_wp, field))
    def cpu(value):
        return value.detach().cpu().numpy().copy()
    def snapshot(indices):
        wp.synchronize()
        return {field: tensor(field)[indices].clone() for field in COPY_FIELDS}
    def restore(state, name):
        wp.synchronize()
        for field in COPY_FIELDS:
            tensor(field).copy_(state[field].expand_as(tensor(field)))
        torch.cuda.synchronize(torch.device(env.device))
        exact_checks[name] = {f: bool(torch.equal(tensor(f), state[f].expand_as(tensor(f)))) for f in COPY_FIELDS}
        write_json(args.output_dir / 'bank_fields_exact.json', exact_checks)
        if not all(exact_checks[name].values()):
            raise RuntimeError('Actual-state restore mismatch: ' + name)
    def reward_at(step):
        reward, _ = sim.get_reward(cfg, env, [ref[step] for ref in refs])
        if not bool(torch.isfinite(reward).all()):
            raise RuntimeError('Nonfinite execution reward')
        return reward
    def collect(indices):
        arrays = tuple(cpu(tensor(f)[indices]) for f in ('qpos', 'qvel', 'time'))
        if not all(np.isfinite(a).all() for a in arrays):
            raise RuntimeError('Nonfinite actual state')
        return arrays
    def progress(stage, step):
        write_json(args.output_dir / 'progress.json', {'stage': stage, 'completed_steps': step})

    before_initial = snapshot(slice(0, 9))
    q = tensor('qpos').clone()
    q[:, 18:20] += torch.as_tensor(GRID[np.arange(WORLDS)%9], device=cfg.device, dtype=q.dtype)
    tensor('qpos').copy_(q)
    torch.cuda.synchronize(torch.device(env.device))
    initial = snapshot(slice(0, 9))
    expected = before_initial['qpos'].clone()
    expected[:, 18:20] += torch.as_tensor(GRID, device=cfg.device, dtype=q.dtype)
    if not torch.equal(initial['qpos'], expected):
        raise RuntimeError('Initial XY grid mismatch')
    if any(not torch.equal(initial[f], before_initial[f]) for f in COPY_FIELDS if f != 'qpos'):
        raise RuntimeError('Initial offset changed another state field')
    prefix_q, prefix_v, prefix_t, prefix_r = [], [], [], []
    for step in range(PREFIX):
        sim.step_env(cfg, env, torch.as_tensor(controls[step], device=cfg.device))
        qr, vr, tr = collect(slice(0, 9))
        prefix_q.append(qr); prefix_v.append(vr); prefix_t.append(tr)
        prefix_r.append(cpu(reward_at(step+1)[:9]))
        if (step+1)%25 == 0:
            progress('PREFIX', step+1)
    bank = snapshot(slice(0, 9))
    np.savez_compressed(args.output_dir / 'prefix.npz', qpos=np.stack(prefix_q), qvel=np.stack(prefix_v),
                        time=np.stack(prefix_t), reward=np.stack(prefix_r), initial_qpos=cpu(initial['qpos']),
                        initial_time=cpu(initial['time']), xy_offsets_m=GRID)
    boundary = {'initial_' + f: cpu(v) for f,v in initial.items()}
    boundary.update({'bank_' + f: cpu(v) for f,v in bank.items()})
    np.savez_compressed(args.output_dir / 'boundary_state.npz', **boundary)

    # Estimated input has no velocity/reference argument: only past poses/timestamps.
    estimator_poses = np.stack([prefix_q[i][actual_bank, 18:25] for i in ESTIMATE_INDICES])
    estimator_times = np.asarray([prefix_t[i][actual_bank] for i in ESTIMATE_INDICES], dtype=np.float64).reshape(2)
    estimate, estimate_linear, estimate_angular_world, estimate_dt = estimate_object_velocity(estimator_poses, estimator_times)
    if not np.isclose(estimate_dt, 2*dt, rtol=1e-4, atol=1e-6):
        raise RuntimeError('Frozen causal estimator must use the 148/150 poststeps, 20ms apart')
    actual_object_pose = cpu(bank['qpos'][actual_bank,18:25])
    if not np.array_equal(estimator_poses[1], actual_object_pose):
        raise RuntimeError('Last estimator pose differs from the actual planning boundary')
    planning_inputs = make_planning_inputs(cpu(bank['qpos'][4]), cpu(bank['qvel'][4]),
                                          cpu(bank['qpos'][actual_bank]), cpu(bank['qvel'][actual_bank,:18]), estimate)
    # Truth is accessed here only after the estimator output is already fixed.
    truth = cpu(bank['qvel'][actual_bank,18:24]).astype(np.float64)
    truth_angular_world = quat_rotation_wxyz(actual_object_pose[3:]) @ truth[3:]
    estimate_record = {'method': 'WORLD_POSITION_SECANT_AND_SHORTEST_WORLD_SO3_LOG_THEN_CURRENT_BODY',
                       'prefix_array_indices': list(ESTIMATE_INDICES), 'physical_steps': [148,150],
                       'source_world_index': actual_bank, 'timestamps_s': estimator_times.tolist(),
                       'window_seconds_actual': estimate_dt, 'latest_input_step': 150,
                       'estimator_inputs': 'OBJECT_POSE_AND_TIMESTAMP_ONLY',
                       'future_or_reference_or_true_velocity_used_by_estimator': False,
                       'estimated_linear_world_m_s': estimate_linear.tolist(),
                       'estimated_angular_world_rad_s': estimate_angular_world.tolist(),
                       'estimated_object_qvel_actual_body': estimate.tolist(),
                       'posthoc_truth_linear_world_m_s': truth[:3].tolist(),
                       'posthoc_truth_angular_world_rad_s': truth_angular_world.tolist(),
                       'posthoc_linear_error_norm_m_s': float(np.linalg.norm(estimate_linear-truth[:3])),
                       'posthoc_angular_error_norm_rad_s': float(np.linalg.norm(estimate_angular_world-truth_angular_world)),
                       'output': str(args.output_dir / 'causal_velocity_estimate.npz')}
    np.savez_compressed(args.output_dir / 'causal_velocity_estimate.npz',
                        input_object_qpos=estimator_poses, input_timestamps=estimator_times,
                        prefix_array_indices=np.asarray(ESTIMATE_INDICES), physical_steps=np.asarray([148,150]),
                        source_world_index=np.asarray(actual_bank), window_seconds=np.asarray(estimate_dt),
                        estimated_object_qvel_actual_body=estimate, estimated_linear_world=estimate_linear,
                        estimated_angular_world=estimate_angular_world,
                        posthoc_truth_object_qvel_actual_body=truth, posthoc_truth_angular_world=truth_angular_world)
    write_json(args.output_dir / 'causal_velocity_estimate.json', estimate_record)
    requested = {}
    for arm in ARMS:
        requested[arm] = {f: value[actual_bank:actual_bank+1].clone() for f, value in bank.items()
                          if f in BASIC_FIELDS and f not in ('qpos', 'qvel')}
        for field in ('qpos', 'qvel'):
            requested[arm][field] = torch.as_tensor(planning_inputs[arm][field][None],
                                                   device=cfg.device, dtype=bank[field].dtype)
        requested[arm]['qacc_warmstart'].zero_()
    common_checks = {f: bool(torch.equal(requested[ARMS[0]][f], requested[ARMS[1]][f]))
                     for f in BASIC_FIELDS if f not in ('qpos', 'qvel')}
    for field in ('qpos', 'qvel'):
        common_checks['robot_' + field + '_0_18_identical'] = bool(torch.equal(requested[ARMS[0]][field][:,:18],
                                                                              requested[ARMS[1]][field][:,:18]))
        common_checks['robot_' + field + '_is_actual'] = all(bool(torch.equal(requested[a][field][:,:18],
                                                                             bank[field][actual_bank:actual_bank+1,:18])) for a in ARMS)
    common_checks['nominal_object_qpos_is_bank4'] = bool(torch.equal(requested['nominal_object']['qpos'][:,18:], bank['qpos'][4:5,18:]))
    common_checks['nominal_object_qvel_is_bank4_body_local_unchanged'] = bool(torch.equal(requested['nominal_object']['qvel'][:,18:], bank['qvel'][4:5,18:]))
    common_checks['observed_object_qpos_is_actual'] = bool(torch.equal(requested['observed_object']['qpos'][:,18:], bank['qpos'][actual_bank:actual_bank+1,18:]))
    common_checks['observed_object_qvel_is_pose_only_estimate'] = bool(torch.equal(requested['observed_object']['qvel'][:,18:],
                                                                                 torch.as_tensor(estimate[None],device=cfg.device,dtype=bank['qvel'].dtype)))
    common_checks['zero_external_forces'] = all(bool(torch.count_nonzero(requested[a][f]) == 0)
                                               for a in ARMS for f in ('qfrc_applied', 'xfrc_applied'))
    if not all(common_checks.values()):
        raise RuntimeError('Planning state construction differs from frozen robot/object sources')
    object_velocity_sources = {'nominal_object': 'BANK4_BODY_LOCAL_UNCHANGED',
                               'observed_object': 'ACTUAL_PREFIX148/150_POSE_TIME_ONLY'}
    reconstructions = {'api': 'mujoco_warp.reset_data/write_basic_inputs/forward',
                       'common_inputs': common_checks, 'arms': {}, 'gpu_dynamic_checks': True}
    def reconstruct(arm):
        wp.synchronize()
        with wp.ScopedDevice(env.device):
            sim.mjwarp.reset_data(env.model_wp, env.data_wp)
        wp.synchronize()
        # Public reset/forward preserves static world transforms. Clearing all
        # derived arrays destroys the floor transform, which forward skips.
        # Contact and dynamic caches are rebuilt by the public API.
        for field in BASIC_FIELDS:
            tensor(field).copy_(requested[arm][field].expand_as(tensor(field)))
        torch.cuda.synchronize(torch.device(env.device))
        sim.load_env_params(cfg, env, {'pair_margin': -.005})
        before = {f: tensor(f)[:1].clone() for f in BASIC_FIELDS}
        before_exact = {f: bool(torch.equal(tensor(f), requested[arm][f].expand_as(tensor(f)))) for f in BASIC_FIELDS}
        with wp.ScopedDevice(env.device):
            sim.mjwarp.forward(env.model_wp, env.data_wp)
        wp.synchronize()
        after_exact = {f: bool(torch.equal(tensor(f), requested[arm][f].expand_as(tensor(f)))) for f in BASIC_FIELDS}
        static_world_exact = {
            'floor_geom0_xpos': bool(torch.equal(tensor('geom_xpos')[:, 0], bank['geom_xpos'][actual_bank:actual_bank+1, 0].expand_as(tensor('geom_xpos')[:, 0]))),
            'floor_geom0_xmat': bool(torch.equal(tensor('geom_xmat')[:, 0], bank['geom_xmat'][actual_bank:actual_bank+1, 0].expand_as(tensor('geom_xmat')[:, 0]))),
            'world_body0_xquat': bool(torch.equal(tensor('xquat')[:, 0], bank['xquat'][actual_bank:actual_bank+1, 0].expand_as(tensor('xquat')[:, 0]))),
        }
        finite = {f: bool(torch.isfinite(tensor(f)).all()) for f in COPY_FIELDS}
        arrays = {'requested_' + f: cpu(value) for f, value in requested[arm].items()}
        arrays.update({'before_forward_' + f: cpu(value) for f, value in before.items()})
        arrays.update({'after_forward_' + f: cpu(tensor(f)[:1]) for f in COPY_FIELDS})
        arrays.update({'nominal_object_qpos': cpu(bank['qpos'][4,18:25]),
                       'nominal_object_qvel': cpu(bank['qvel'][4,18:24]),
                       'actual_object_qpos': cpu(bank['qpos'][actual_bank,18:25]),
                       'actual_object_qvel': cpu(bank['qvel'][actual_bank,18:24])})
        np.savez_compressed(args.output_dir / f'reconstruction_{arm}.npz', **arrays)
        reconstructions['arms'][arm] = {'requested_basic_all1024_exact_before_forward': before_exact,
                                      'requested_basic_all1024_exact_after_forward': after_exact,
                                      'all31_fields_finite_after_forward': finite,
                                      'static_world_matches_actual_prefix': static_world_exact,
                                      'qacc_warmstart_zero': bool(torch.count_nonzero(tensor('qacc_warmstart')) == 0),
                                      'planning_robot_qpos_bank_index': actual_bank, 'robot_qvel_bank_index': actual_bank,
                                      'object_qpos_bank_index': 4 if arm == 'nominal_object' else actual_bank,
                                      'object_qvel_source': object_velocity_sources[arm],
                                      'angular_velocity_frame': 'NOMINAL_BODY_LOCAL' if arm == 'nominal_object' else 'ACTUAL_BODY_LOCAL',
                                      'derived_state_source': 'GPU_FORWARD',
                                      'output': str(args.output_dir / f'reconstruction_{arm}.npz')}
        write_json(args.output_dir / 'reconstruction_checks.json', reconstructions)
        if not all(before_exact.values()) or not all(after_exact.values()) or not all(finite.values()) or not all(static_world_exact.values()):
            raise RuntimeError('GPU reconstruction changed basic inputs or produced nonfinite state: ' + arm)
        if not reconstructions['arms'][arm]['qacc_warmstart_zero']:
            raise RuntimeError('GPU reconstruction retained solver warmstart: ' + arm)

    rollout = make_rollout_fn_fast(sim.step_env, sim.save_state, sim.load_state, sim.get_reward,
                  sim.get_terminal_reward, sim.get_terminate, sim.get_trace, sim.save_env_params,
                  sim.load_env_params, sim.copy_sample_state, sim.get_qpos, sim.get_qvel)
    official_once = make_optimize_once_fn_fast(rollout)
    optimizations, iteration_records, effective = {}, {}, []
    current_arm = None
    def recorded_once(*call_args, **kwargs):
        begin = time.perf_counter()
        result = official_once(*call_args, **kwargs)
        torch.cuda.synchronize(torch.device(env.device))
        _, _, info = result
        iteration_records[current_arm].append({'iteration': len(iteration_records[current_arm])+1,
             'wall_seconds': time.perf_counter()-begin, 'rew_max': float(info['rew_max']),
             'rew_mean': float(info['rew_mean']), 'candidate_world_steps': WORLDS*HORIZON})
        write_json(args.output_dir / 'optimization_iterations.json', iteration_records)
        return result
    optimize = make_optimize_fn_fast(recorded_once)
    warmstart = controls[150:450].copy()
    ref_slice = tuple(ref[151:451] for ref in refs)
    rng_cpu_start = None
    for arm in ARMS:
        source = object_velocity_sources[arm]
        current_arm = arm
        reconstruct(arm)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        rng_start = cpu(torch.get_rng_state())
        cuda_rng_start = cpu(torch.cuda.get_rng_state(torch.device(env.device)))
        if rng_cpu_start is None:
            rng_cpu_start, rng_cuda_start = rng_start.copy(), cuda_rng_start.copy()
        elif not np.array_equal(rng_start, rng_cpu_start) or not np.array_equal(cuda_rng_start, rng_cuda_start):
            raise RuntimeError('Planning arms do not have identical starting RNG')
        iteration_records[arm] = []
        torch.cuda.synchronize(torch.device(env.device))
        begin = time.perf_counter()
        returned, info = optimize(cfg, env, torch.as_tensor(warmstart, device=cfg.device).clone(), ref_slice)
        torch.cuda.synchronize(torch.device(env.device))
        wall_seconds = time.perf_counter()-begin
        returned_np = cpu(returned)
        succeeded = bool(np.asarray(info['succeeded']).reshape(-1)[0])
        chosen = accepted_or_original(returned_np, warmstart, succeeded)
        opt_steps = int(np.asarray(info['opt_steps']).reshape(-1)[0])
        attempts = int(np.asarray(info['num_attempts']).reshape(-1)[0])
        if opt_steps != len(iteration_records[arm]) or attempts not in (1,2,3,4) or opt_steps != 4*attempts:
            raise RuntimeError('Official optimization iteration/attempt accounting mismatch')
        arrays = {'returned_ctrl': returned_np, 'effective_ctrl': chosen, 'warmstart_ctrl': warmstart,
                  'succeeded': np.asarray(succeeded), 'reference_indices': np.arange(151,451),
                  'planning_robot_qpos_bank_index': np.asarray(actual_bank),
                  'object_qpos_bank_index': np.asarray(4 if arm == 'nominal_object' else actual_bank), 'object_velocity_source': np.asarray(source), 'rng_cpu_start': rng_start, 'rng_cuda_start': cuda_rng_start}
        # Predicted state snapshots are not used to advance execution.
        info.pop('state_snapshot', None)
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                value = cpu(value)
            array = np.asarray(value)
            if array.dtype.kind != 'O':
                arrays['info_' + key] = array
        np.savez_compressed(args.output_dir / f'optimization_{arm}.npz', **arrays)
        optimizations[arm] = {'succeeded': succeeded, 'status': 'ACCEPTED' if succeeded else 'SEARCH_NOT_ACCEPTED/FALLBACK_ORIGINAL',
                  'wall_seconds': wall_seconds, 'opt_steps': opt_steps, 'attempts': attempts,
                  'candidate_world_steps': opt_steps*WORLDS*HORIZON,
                  'nominal_config': {'num_dr': 1, 'xy_offset': 0., 'margin': -.005, 'iterations_per_attempt': 4},
                  'planning_robot_source_bank_index': actual_bank,
                  'planning_object_qpos_bank_index': 4 if arm == 'nominal_object' else actual_bank, 'object_velocity_source': source,
                  'planning_robot_velocity_bank_index': actual_bank, 'planning_solver_warmstart': 'ZERO',
                  'reconstruction': str(args.output_dir / f'reconstruction_{arm}.npz'), 'warmstart_slice': [150,450],
                  'reference_slice': [151,451], 'starting_rng_identical': True,
                  'output': str(args.output_dir / f'optimization_{arm}.npz')}
        write_json(args.output_dir / 'optimizations.json', optimizations)
        effective.append(chosen)
        del info, returned

    actual_state = {f: value[actual_bank:actual_bank+1].clone() for f,value in bank.items()}
    restore(actual_state, 'execution_all1024')
    sim.load_env_params(cfg, env, {'pair_margin': -.005})
    group_suffix = np.stack([warmstart, warmstart, *effective])
    suffix_q, suffix_v, suffix_t, suffix_r = [], [], [], []
    for step in range(HORIZON):
        ctrl = np.repeat(warmstart[step][None], WORLDS, axis=0)
        ctrl[:len(GROUPS)] = group_suffix[:,step]
        sim.step_env(cfg, env, torch.as_tensor(ctrl, device=cfg.device))
        qr,vr,tr = collect(slice(0,len(GROUPS)))
        suffix_q.append(qr[:,None]); suffix_v.append(vr[:,None]); suffix_t.append(tr[:,None])
        suffix_r.append(cpu(reward_at(151+step)[:len(GROUPS)])[:,None])
        if (step+1)%25 == 0:
            progress('ACTUAL_SUFFIX', step+1)
    shared_prefix = lambda rows: np.repeat(np.stack(rows)[:,None,actual_bank:actual_bank+1], len(GROUPS), axis=1)
    all_q = np.concatenate([shared_prefix(prefix_q), np.stack(suffix_q)])
    all_v = np.concatenate([shared_prefix(prefix_v), np.stack(suffix_v)])
    all_t = np.concatenate([shared_prefix(prefix_t), np.stack(suffix_t)])
    all_r = np.concatenate([shared_prefix(prefix_r), np.stack(suffix_r)])
    initial_time = cpu(initial['time'][actual_bank:actual_bank+1])
    increments = np.diff(np.concatenate([np.broadcast_to(initial_time,(1,len(GROUPS),1)),all_t]),axis=0)
    if not np.allclose(increments,dt,rtol=1e-3,atol=1e-6):
        raise RuntimeError('Actual execution time reset or wrong step duration')
    all_ctrl = np.concatenate([np.repeat(controls[:150][None],len(GROUPS),axis=0),group_suffix],axis=1)
    np.savez_compressed(args.output_dir / 'trajectories.npz', qpos=all_q,qvel=all_v,
                        relative_time=all_t-initial_time[None,None],initial_time=initial_time,
                        reward=all_r,ctrl=all_ctrl,initial_qpos=cpu(initial['qpos'][actual_bank:actual_bank+1]),
                        sim_dt_s=np.asarray(dt),xy_offsets_m=GRID[actual_bank:actual_bank+1],group_order=np.asarray(GROUPS),
                        physical_reference_steps=np.arange(1,451))
    write_json(args.output_dir / 'physics_execution.json', {'execution_steps': 450,
               'execution_world_steps': 450*WORLDS,
               'candidate_world_steps': sum(x['candidate_world_steps'] for x in optimizations.values()),
               'elapsed_seconds_before_render': time.perf_counter()-started,
               'task': task, 'base_controls_seed': 0, 'actual_bank_index': actual_bank,
               'planning_actual_robot_pose_and_velocity': True, 'planning_solver_warmstart': 'ZERO_ALL_ARMS',
               'execution_restored_original_actual_bank31': True, 'outer_revert': False})
    progress('TRAJECTORY_SAVED',450)
    del env, bank
    torch.cuda.empty_cache()
    return finish(args,spec,reference,dt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec','controls','reference','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--render-only',action='store_true')
    args = parser.parse_args()
    for name in ('spec','controls','reference','output_dir'):
        setattr(args,name,getattr(args,name).resolve())
    return preflight(args) if args.preflight else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
