#!/usr/bin/env python3
"""Frozen causal two-pose object-velocity diagnostic with three matched official searches."""
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

GROUPS = ('base', 'base_duplicate', 'predicted', 'estimated', 'oracle')
ARMS = ('predicted', 'estimated', 'oracle')
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


def predicted_object_velocity(nominal_object_pose, nominal_object_velocity, actual_object_pose):
    """Transport nominal body angular velocity to the current actual body frame."""
    nq, nv, aq = [np.asarray(x) for x in (nominal_object_pose, nominal_object_velocity, actual_object_pose)]
    if nq.shape != (7,) or aq.shape != (7,) or nv.shape != (6,):
        raise ValueError('Expected object xyz/wxyz7 and velocity6')
    if not all(np.isfinite(x).all() for x in (nq, nv, aq)):
        raise ValueError('Nonfinite predicted input')
    result = nv.copy()
    result[3:] = quat_rotation_wxyz(aq[3:]).T @ quat_rotation_wxyz(nq[3:]) @ nv[3:]
    return result


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {'status': 'PASS_NO_SIMULATION', 'gpu_used': False, 'simulation_steps': 0,
              'source_revision': REVISION, 'seed': args.seed, 'job_id': args.job_id,
              'controls_source': str(args.controls), 'controls_shape': list(controls.shape),
              'reference_source': str(args.reference), 'reference_shape': list(reference.shape),
              'sim_dt_s': dt, 'num_worlds': WORLDS, 'group_order': list(GROUPS),
              'copy_fields': COPY_FIELDS, 'planning_qpos_bank_index': 8, 'object_velocity_sources': {'predicted': 'BANK4', 'estimated': 'PREFIX148/150_POSE_TIME_ONLY', 'oracle': 'BANK8'},
              'estimator_prefix_indices': list(ESTIMATE_INDICES), 'estimator_physical_steps': [148, 150],
              'robot_velocity_bank_index': 8, 'planning_warmstart': 'ZERO_ALL_ARMS',
              'reconstruction_api': 'mujoco_warp.reset_data/write_basic_inputs/forward',
              'execution_bank_index': 8, 'warmstart_slice': [150, 450],
              'optimizer_reference_slice': [151, 451], 'actual_suffix_steps': [151, 450],
              'failed_search_execution': 'FALLBACK_ORIGINAL', 'outer_revert': False,
              'preflight_note': 'CPU input/array checks only; no simulator imported or stepped.'}
    write_json(args.output_dir / 'preflight.json', record)
    print(json.dumps(record))
    return 0


def build_record(args, spec, reference, dt):
    with np.load(args.output_dir / 'trajectories.npz') as saved:
        qpos, initial = saved['qpos'], saved['initial_qpos']
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
        groups.append({'name': name, 'worlds': [{'world_index': 0, 'source_bank_index': 8,
                                               'xy_offset_m': [.005, .005], **metrics}]})
    return {'status': 'TRAJECTORY_SAVED', 'protocol': 'NATIVE_V2_CAUSAL_VELOCITY_PROTOCOL.md',
            'seed': args.seed, 'job_id': args.job_id, 'source_revision': spec['source_revision'],
            'controls_source': str(args.controls), 'reference_source': str(args.reference),
            'trajectory': str(args.output_dir / 'trajectories.npz'),
            'boundary_state': str(args.output_dir / 'boundary_state.npz'),
            'group_order': list(GROUPS), 'groups': groups, 'optimizations': optimizations,
            'bank_fields_exact': exact,
            'reconstruction': json.loads((args.output_dir / 'reconstruction_checks.json').read_text()),
            'planning_inputs': {arm: str(args.output_dir / f'reconstruction_{arm}.npz') for arm in ARMS},
            'velocity_estimation': json.loads((args.output_dir / 'causal_velocity_estimate.json').read_text()),
            'videos': {}, 'outputvideo': {}, 'video_status': 'PENDING_RENDER',
            'execution': {'steps': 450, 'shared_prefix_steps': 150, 'actual_suffix_steps': 300,
                          'saved_worlds': len(GROUPS), 'num_worlds': WORLDS, 'execution_bank_index': 8,
                          'sim_dt_s': dt, 'pair_margin_m': -.005, 'outer_revert': False,
                          'predicted_snapshot_used_for_execution': False},
            'claim_boundary': 'Accurate simulator pose and robot velocity, with nominal prediction versus causal two-pose velocity versus actual object velocity; extra search computation and local recovery only.'}


def render_saved(args, spec):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    import spider.simulators.mjwp as sim
    cfg = load_config(spec, args.seed)
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
    video = args.output_dir / 'causal_velocity_5grid.mp4'
    try:
        with imageio.get_writer(str(video), fps=50, codec='libx264', quality=8, macro_block_size=16) as writer:
            for step in range(1, TOTAL, 2):
                canvas = Image.new('RGB', (2160, 960))
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
                    draw.text((8, 5), f'{name} | initial xy=+5,+5mm', fill='white')
                    phase = 'shared actual prefix' if step < PREFIX else 'actual suffix'
                    if step >= 391:
                        phase += ' | reference padding'
                    draw.text((8, 22), f'poststep {step+1} | t={(step+1)*.01:.2f}s | {phase}', fill='white')
                    canvas.paste(tile, ((group % 3)*720, (group // 3)*480))
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
        record['videos'] = {'comparison_3x2': video}
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
    estimator_poses = np.stack([prefix_q[i][8, 18:25] for i in ESTIMATE_INDICES])
    estimator_times = np.asarray([prefix_t[i][8] for i in ESTIMATE_INDICES], dtype=np.float64).reshape(2)
    estimate, estimate_linear, estimate_angular_world, estimate_dt = estimate_object_velocity(estimator_poses, estimator_times)
    if not np.isclose(estimate_dt, 2*dt, rtol=1e-4, atol=1e-6):
        raise RuntimeError('Frozen causal estimator must use the 148/150 poststeps, 20ms apart')
    actual_object_pose = cpu(bank['qpos'][8,18:25])
    if not np.array_equal(estimator_poses[1], actual_object_pose):
        raise RuntimeError('Last estimator pose differs from the actual planning boundary')
    requested_velocities = {}
    for arm in ARMS:
        requested_velocities[arm] = np.empty(24, dtype=np.float32)
        requested_velocities[arm][:18] = cpu(bank['qvel'][8,:18])
    requested_velocities['estimated'][18:] = estimate
    requested_velocities['predicted'][18:] = predicted_object_velocity(
        cpu(bank['qpos'][4,18:25]), cpu(bank['qvel'][4,18:24]), actual_object_pose)
    requested_velocities['oracle'][18:] = cpu(bank['qvel'][8,18:24])
    # Truth is accessed here only after the estimator output is already fixed.
    truth = cpu(bank['qvel'][8,18:24]).astype(np.float64)
    truth_angular_world = quat_rotation_wxyz(actual_object_pose[3:]) @ truth[3:]
    estimate_record = {'method': 'WORLD_POSITION_SECANT_AND_SHORTEST_WORLD_SO3_LOG_THEN_CURRENT_BODY',
                       'prefix_array_indices': list(ESTIMATE_INDICES), 'physical_steps': [148,150],
                       'source_world_index': 8, 'timestamps_s': estimator_times.tolist(),
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
                        source_world_index=np.asarray(8), window_seconds=np.asarray(estimate_dt),
                        estimated_object_qvel_actual_body=estimate, estimated_linear_world=estimate_linear,
                        estimated_angular_world=estimate_angular_world,
                        posthoc_truth_object_qvel_actual_body=truth, posthoc_truth_angular_world=truth_angular_world)
    write_json(args.output_dir / 'causal_velocity_estimate.json', estimate_record)
    requested = {}
    for arm in ARMS:
        requested[arm] = {f: value[8:9].clone() for f, value in bank.items() if f in BASIC_FIELDS and f != 'qvel'}
        requested[arm]['qvel'] = torch.as_tensor(requested_velocities[arm][None],
                                                 device=cfg.device, dtype=bank['qvel'].dtype)
        requested[arm]['qacc_warmstart'].zero_()
    common_checks = {f: all(bool(torch.equal(requested['predicted'][f], requested[a][f])) for a in ARMS)
                     for f in BASIC_FIELDS if f != 'qvel'}
    common_checks['robot_qvel_0_18_identical'] = all(bool(torch.equal(requested['predicted']['qvel'][:,:18],
                                                                  requested[a]['qvel'][:,:18])) for a in ARMS)
    common_checks['robot_qvel_is_actual'] = all(bool(torch.equal(requested[a]['qvel'][:,:18], bank['qvel'][8:9,:18]))
                                               for a in ARMS)
    common_checks['zero_external_forces'] = all(bool(torch.count_nonzero(requested[a][f]) == 0)
                                               for a in ARMS for f in ('qfrc_applied', 'xfrc_applied'))
    if not all(common_checks.values()):
        raise RuntimeError('Planning inputs differ beyond the frozen object velocity intervention')
    object_velocity_sources = {'predicted': 'BANK4_WORLD_TRANSPORT', 'estimated': 'PREFIX148/150_POSE_TIME_ONLY',
                               'oracle': 'BANK8_ACTUAL'}
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
            'floor_geom0_xpos': bool(torch.equal(tensor('geom_xpos')[:, 0], bank['geom_xpos'][8:9, 0].expand_as(tensor('geom_xpos')[:, 0]))),
            'floor_geom0_xmat': bool(torch.equal(tensor('geom_xmat')[:, 0], bank['geom_xmat'][8:9, 0].expand_as(tensor('geom_xmat')[:, 0]))),
            'world_body0_xquat': bool(torch.equal(tensor('xquat')[:, 0], bank['xquat'][8:9, 0].expand_as(tensor('xquat')[:, 0]))),
        }
        finite = {f: bool(torch.isfinite(tensor(f)).all()) for f in COPY_FIELDS}
        arrays = {'requested_' + f: cpu(value) for f, value in requested[arm].items()}
        arrays.update({'before_forward_' + f: cpu(value) for f, value in before.items()})
        arrays.update({'after_forward_' + f: cpu(tensor(f)[:1]) for f in COPY_FIELDS})
        arrays.update({'nominal_object_qpos': cpu(bank['qpos'][4,18:25]),
                       'nominal_object_qvel': cpu(bank['qvel'][4,18:24]),
                       'actual_object_qpos': cpu(bank['qpos'][8,18:25]),
                       'actual_object_qvel': cpu(bank['qvel'][8,18:24])})
        np.savez_compressed(args.output_dir / f'reconstruction_{arm}.npz', **arrays)
        reconstructions['arms'][arm] = {'requested_basic_all1024_exact_before_forward': before_exact,
                                      'requested_basic_all1024_exact_after_forward': after_exact,
                                      'all31_fields_finite_after_forward': finite,
                                      'static_world_matches_actual_prefix': static_world_exact,
                                      'qacc_warmstart_zero': bool(torch.count_nonzero(tensor('qacc_warmstart')) == 0),
                                      'planning_qpos_bank_index': 8, 'robot_qvel_bank_index': 8,
                                      'object_qvel_source': object_velocity_sources[arm],
                                      'angular_velocity_frame': 'ACTUAL_BODY_LOCAL',
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
                  'planning_qpos_bank_index': np.asarray(8), 'object_velocity_source': np.asarray(source), 'rng_cpu_start': rng_start, 'rng_cuda_start': cuda_rng_start}
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
                  'planning_source_bank_index': 8, 'object_velocity_source': source,
                  'planning_robot_velocity_bank_index': 8, 'planning_solver_warmstart': 'ZERO',
                  'reconstruction': str(args.output_dir / f'reconstruction_{arm}.npz'), 'warmstart_slice': [150,450],
                  'reference_slice': [151,451], 'starting_rng_identical': True,
                  'output': str(args.output_dir / f'optimization_{arm}.npz')}
        write_json(args.output_dir / 'optimizations.json', optimizations)
        effective.append(chosen)
        del info, returned

    actual_state = {f: value[8:9].clone() for f,value in bank.items()}
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
    shared_prefix = lambda rows: np.repeat(np.stack(rows)[:,None,8:9], len(GROUPS), axis=1)
    all_q = np.concatenate([shared_prefix(prefix_q), np.stack(suffix_q)])
    all_v = np.concatenate([shared_prefix(prefix_v), np.stack(suffix_v)])
    all_t = np.concatenate([shared_prefix(prefix_t), np.stack(suffix_t)])
    all_r = np.concatenate([shared_prefix(prefix_r), np.stack(suffix_r)])
    initial_time = cpu(initial['time'][8:9])
    increments = np.diff(np.concatenate([np.broadcast_to(initial_time,(1,len(GROUPS),1)),all_t]),axis=0)
    if not np.allclose(increments,dt,rtol=1e-3,atol=1e-6):
        raise RuntimeError('Actual execution time reset or wrong step duration')
    all_ctrl = np.concatenate([np.repeat(controls[:150][None],len(GROUPS),axis=0),group_suffix],axis=1)
    np.savez_compressed(args.output_dir / 'trajectories.npz', qpos=all_q,qvel=all_v,
                        relative_time=all_t-initial_time[None,None],initial_time=initial_time,
                        reward=all_r,ctrl=all_ctrl,initial_qpos=cpu(initial['qpos'][8:9]),
                        sim_dt_s=np.asarray(dt),xy_offsets_m=GRID[8:9],group_order=np.asarray(GROUPS),
                        physical_reference_steps=np.arange(1,451))
    write_json(args.output_dir / 'physics_execution.json', {'execution_steps': 450,
               'execution_world_steps': 450*WORLDS,
               'candidate_world_steps': sum(x['candidate_world_steps'] for x in optimizations.values()),
               'elapsed_seconds_before_render': time.perf_counter()-started,
               'planning_actual_pose_and_robot_velocity': True, 'planning_solver_warmstart': 'ZERO_ALL_ARMS',
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
