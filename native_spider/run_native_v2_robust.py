"""Run the official fast optimizer with accepted-block physical execution."""
import ast
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path('/mnt/sdc/dty_user/hand_to_robot_20261002')
REPO = ROOT / 'repos/spider-native'
HOOK_PREFIX = '    def _replay_and_advance(sim_step_val, inf):\n        nonlocal ctrls\n'
HOOK_INSERT = (
    '        nonlocal last_snap, last_snap_idx\n'
    '        _continuous_execute(config, env, ctrls, inf, sim_step_val)\n'
    '        last_snap = inf["state_snapshot"]\n'
    '        last_snap_idx = inf["best_idx"]\n'
)
SEMANTICS = (
    'Official fast optimizer/retry/revert with a common accepted-block continuous '
    'MuJoCo-Warp execution adapter; actual world0 post-step states and full '
    'snapshot advance subsequent planning. Fixed margin -0.005m; no execution '
    'xy injection. Not the unchanged official fast state-copy entrypoint.'
)


def patch_source(source):
    """Insert once at the acceptance boundary, leaving optimizer code intact."""
    if source.count(HOOK_PREFIX) != 1:
        raise RuntimeError('Expected exactly one official _replay_and_advance prefix')
    restore = '_restore_best_state_to_env(env, inf["state_snapshot"], inf["best_idx"])'
    if source.count(restore) != 1:
        raise RuntimeError('Expected one successor restore consuming inf snapshot/index')
    patched = source.replace(HOOK_PREFIX, HOOK_PREFIX + HOOK_INSERT, 1)
    tree = ast.parse(patched)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    replay = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == '_replay_and_advance')
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == '_continuous_execute']
    if len(calls) != 1 or calls[0] not in list(ast.walk(replay)):
        raise RuntimeError('Physical execution hook must occur only inside accepted replay')
    return patched, {'source_prefix_matches': 1, 'execution_hook_calls': 1,
                     'original_prefix_line': source[:source.index(HOOK_PREFIX)].count('\n') + 1,
                     'successor_restore_consumes_actual_inf': True,
                     'checkpoint_cache_updated_to_actual_snapshot': True}


def continuous_execute(config, env, controls, info, sim_step, sim, sampling, records, output):
    """Execute only a control block accepted by the author's outer state machine."""
    import numpy as np
    import torch

    count = config.ctrl_steps
    if controls.shape[0] < count or controls.shape[1] != config.nu:
        raise RuntimeError('Accepted control block has incompatible dimensions')
    if not bool(torch.isfinite(controls[:count]).all().item()):
        raise RuntimeError('Accepted control block is nonfinite')
    # Do not pass xy_offset: the loader adds it to qpos, even at this boundary.
    sim.load_env_params(config, env, {'pair_margin': -0.005})
    initial_qpos = sim.get_qpos(config, env)[0].clone()
    initial_qvel = sim.get_qvel(config, env)[0].clone()
    initial_time = float(sim.wp.to_torch(env.data_wp.time)[0].item())
    # The candidate snapshot is no longer needed once this block is accepted.
    info.pop('state_snapshot', None)
    qpos, qvel, physical_times = [], [], []
    for index in range(count):
        # Official step_env broadcasts this 1-D control to all candidate worlds.
        sim.step_env(config, env, controls[index])
        qpos.append(sim.get_qpos(config, env)[0].clone())
        qvel.append(sim.get_qvel(config, env)[0].clone())
        physical_times.append(sim.wp.to_torch(env.data_wp.time)[0].clone())
    actual_qpos = torch.stack(qpos)
    actual_qvel = torch.stack(qvel)
    actual_times = torch.stack(physical_times).detach().cpu().numpy()
    time_increments = np.diff(np.r_[initial_time, actual_times])
    if not np.isfinite(actual_times).all() or not np.allclose(time_increments, config.sim_dt, rtol=1e-3, atol=1e-6):
        raise RuntimeError('Accepted-block physical time reset or incorrect step duration')
    if not bool(torch.isfinite(actual_qpos).all().item()) or not bool(torch.isfinite(actual_qvel).all().item()):
        raise RuntimeError('Nonfinite accepted-block physical execution')
    actual_snapshot = sampling._snapshot_env_state(env)
    if not torch.equal(actual_snapshot['qpos'][0], actual_qpos[-1]) or not torch.equal(actual_snapshot['qvel'][0], actual_qvel[-1]):
        raise RuntimeError('Successor snapshot does not match last actual post-step state')
    info['best_qpos'] = actual_qpos.detach().cpu().numpy()
    info['best_qvel'] = actual_qvel.detach().cpu().numpy()
    info['state_snapshot'] = actual_snapshot
    info['best_idx'] = 0
    info['execution_reference_steps'] = np.arange(sim_step + 1, sim_step + count + 1)
    info['execution_physical_times'] = actual_times
    info['execution_qpos_start'] = initial_qpos.detach().cpu().numpy()
    info['execution_qvel_start'] = initial_qvel.detach().cpu().numpy()
    record = {'execution_call': len(records), 'sim_step_start': int(sim_step),
              'reference_first_poststep': int(sim_step + 1),
              'reference_last_poststep': int(sim_step + count),
              'logical_time_start': float(sim_step * config.sim_dt),
              'logical_time_end': float((sim_step + count) * config.sim_dt),
              'physical_time_start': initial_time,
              'physical_time_end': float(actual_times[-1]),
              'physical_step_calls': int(count), 'worlds': int(env.num_worlds),
              'world_step_count': int(count * env.num_worlds),
              'pair_margin_m': -0.005, 'xy_injected': False, 'snapshot_world': 0,
              'snapshot_matches_final_qpos_qvel': True,
              'qpos_start': info['execution_qpos_start'].tolist(),
              'qvel_start': info['execution_qvel_start'].tolist(),
              'qpos_end': info['best_qpos'][-1].tolist(),
              'qvel_end': info['best_qvel'][-1].tolist()}
    records.append(record)
    (output / 'continuous_execution.json').write_text(json.dumps({
        'hook_executed': True, 'execution_semantics': SEMANTICS,
        'blocks': records, 'includes_blocks_later_discarded_by_author_revert': True,
        'physical_step_calls': sum(item['physical_step_calls'] for item in records),
        'world_step_count': sum(item['world_step_count'] for item in records)}, indent=2) + '\n')


def validate_condition(config, condition):
    mode = Path(condition).name.split('_s')[0]
    expected = {'nominal': (1, (0.0, 0.0)), 'robust3': (3, (-0.005, 0.005))}
    if mode not in expected:
        raise RuntimeError('Condition must be robust01/nominal_sN or robust01/robust3_sN')
    num_dr, offsets = expected[mode]
    if config.num_dr != num_dr or tuple(config.xy_offset_range) != offsets:
        raise RuntimeError('DR config does not match the frozen condition')
    if tuple(config.pair_margin_range) != (-0.005, -0.005):
        raise RuntimeError('Both conditions must preserve fixed -0.005 margin')
    if config.num_samples != 1024 or config.ctrl_steps != 150 or config.horizon_steps != 300:
        raise RuntimeError('Candidate count or timing does not match the frozen protocol')
    if config.max_num_iterations != 4 or config.contact_guidance or config.terminate_resample:
        raise RuntimeError('Initial author search preset differs from the frozen protocol')
    return mode


def main():
    spec_path = Path(os.environ['SPIDER_V2_SPEC'])
    spec = json.loads(spec_path.read_text())
    output = ROOT / 'results/native_v2_01' / spec['condition']
    output.mkdir(parents=True, exist_ok=True)
    preflight = '--preflight' in sys.argv
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    entry = importlib.import_module('examples.run_mjwp_fast')
    source_path = Path(entry.__file__)
    source, hook = patch_source(source_path.read_text())
    # Keep the official module filename/package and Hydra config resolution.
    exec(compile(source, str(source_path), 'exec'), entry.__dict__)
    from spider.simulators import mjwp as sim
    from spider.optimizers import sampling_fast as sampling

    for function, parameters in [(sim.step_env, ('config', 'env', 'ctrl_mujoco')),
                                 (sim.get_qpos, ('config', 'env')),
                                 (sim.get_qvel, ('config', 'env')),
                                 (sampling._snapshot_env_state, ('env',))]:
        if tuple(inspect.signature(function).parameters) != parameters:
            raise RuntimeError('Official execution/snapshot interface changed')
    records = []
    entry._continuous_execute = lambda config, env, controls, info, sim_step: continuous_execute(
        config, env, controls, info, sim_step, sim, sampling, records, output)
    original_process = entry.process_config

    def process_config(config):
        config = original_process(config)
        validate_condition(config, spec['condition'])
        return config

    entry.process_config = process_config
    official_run_main = entry.run_main
    hook.update({'installed': True, 'source': str(source_path),
                 'source_revision': spec['source_revision'],
                 'execution_semantics': SEMANTICS, 'retry_boost_revert_preserved': True})

    def run_main_with_output():
        instrumented_main = entry.main

        def main_with_output(config):
            result = instrumented_main(config)
            if preflight:
                path = ROOT / 'native_spider' / spec.get('preflight_filename', 'V2_CPU_PREFLIGHT.json')
                record = json.loads(path.read_text())
                record['continuous_execution_adapter'] = {**hook, 'hook_executed': False,
                                                           'simulation_steps': 0}
                path.write_text(json.dumps(record, indent=2) + '\n')
                print(json.dumps(record))
                return result
            path = output / 'condition_result.json'
            record = json.loads(path.read_text())
            if record['status'] == 'FULL_OFFICIAL_OUTPUT':
                record['status'] = 'FULL_CONTINUOUS_ADAPTER_OUTPUT'
            elif record['status'] == 'OFFICIAL_TERMINATED_PARTIAL':
                record['status'] = 'CONTINUOUS_ADAPTER_TERMINATED_PARTIAL'
            record['execution_semantics'] = SEMANTICS
            record['continuous_execution_adapter'] = {**hook, 'hook_executed': bool(records),
                                                      'execution_calls': len(records),
                                                      'audit': str(output / 'continuous_execution.json')}
            record['official_returned_metrics_usage'] = (
                'Preserved author diagnostic only; official final computation uses '
                'reference[:T], not the required post-step reference[1:T+1]. '
                'New primary metrics come from the independent fixed-control probe.'
            )
            path.write_text(json.dumps(record, indent=2) + '\n')
            return result

        entry.main = main_with_output
        return official_run_main()

    entry.run_main = run_main_with_output
    # Reuse its exact compatibility flag, CPU loader, search instrumentation,
    # startup-bypass record, standard artifacts, and condition_result fields.
    runpy.run_path(str(Path(__file__).with_name('run_native_v2.py')), run_name='__main__')


if __name__ == '__main__':
    main()
