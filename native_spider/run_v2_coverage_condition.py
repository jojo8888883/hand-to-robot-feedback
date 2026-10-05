"""Run one nominal continuous baseline and fixed-grid evaluation on an OakInk-v2 task clip."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np

ROOT = Path('/mnt/sdc/dty_user/hand_to_robot_20261002')


def cli_values(arguments):
    values = {}
    for argument in arguments:
        if '=' in argument:
            key, value = argument.split('=', 1)
            key = key.lstrip('+')
            if key in {'task', 'data_id', 'robot_type', 'embodiment_type', 'dataset_dir'}:
                values[key] = value.strip("'\"")
    required = {'task', 'data_id', 'robot_type', 'embodiment_type', 'dataset_dir'}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"author_cli_arguments missing required entries: {sorted(missing)}")
    return values


def source_trajectory(dataset_dir, arguments):
    values = cli_values(arguments)
    return (Path(dataset_dir) / 'processed/oakinkv2' / values['robot_type']
            / values['embodiment_type'] / values['task'] / values['data_id']
            / 'trajectory_mjwp_fast.npz')


def main():
    spec_path = Path(os.environ['SPIDER_V2_SPEC'])
    spec = json.loads(spec_path.read_text())
    mode = spec['comparison_mode']
    if mode != 'nominal':
        raise RuntimeError('This launcher is only for the frozen nominal clip baseline')
    values = cli_values(spec['author_cli_arguments'])
    out = ROOT / 'results/native_v2_01' / spec['condition']
    out.mkdir(parents=True, exist_ok=True)
    condition_path = out / 'condition_result.json'
    trajectory = source_trajectory(values['dataset_dir'], spec['author_cli_arguments'])

    if not condition_path.exists():
        if trajectory.exists():
            with np.load(trajectory) as saved:
                complete = saved['ctrl'].reshape(-1, 18).shape[0] >= 450
            if complete:
                raise RuntimeError('Controls already complete; repair missing planning metadata/video without rerunning optimization.')
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_native_v2_robust.py')], check=True)

    planning = json.loads(condition_path.read_text())
    for key in ('trajectory', 'video', 'config'):
        if planning.get(key):
            source = Path(planning[key])
            if source.is_file() and source.parent != out:
                shutil.copy2(source, out / source.name)

    shared = {
        'job_id': spec['job_id'],
        'mode': mode,
        'seed': spec['seed'],
        'protocol': spec['protocol'],
        'planning': str(condition_path),
        'claim_boundary': (
            'Descriptive simulated diagnostic on an independent official OakInk-v2 task clip; '
            'no method comparison, real-task success, or cross-task generalization claim.'
        ),
    }
    if planning.get('steps', 0) < 450:
        result = {
            'status': 'PLANNING_INCOMPLETE',
            **shared,
            'lift_count': 0,
            'tracking_count': 0,
            'evaluation_denominator': 9,
            'root_and_so3_errors': None,
            'reason': 'Official planning/recovery ended without the required full control sequence.',
        }
    else:
        evaluation = out / 'evaluation'
        eval_path = evaluation / 'condition_result.json'
        if not eval_path.exists():
            subprocess.run([
                sys.executable, str(ROOT / 'scripts/probe_native_v2_position.py'),
                '--spec', str(spec_path), '--controls', planning['trajectory'],
                '--reference', str(out / 'official_loaded_reference.npz'),
                '--output-dir', str(evaluation), '--job-id', spec['job_id'],
                '--protocol', spec['protocol'],
            ], check=True)
        evaluation_data = json.loads(eval_path.read_text())
        worlds = evaluation_data['worlds']
        lift_count = sum(
            bool(w['last_0p5s_lift_window']['lift_proxy_all_steps_at_least_0p10m'])
            for w in worlds
        )
        tracking_count = sum(
            bool(w['last_0p5s_lift_window']['track_reference_proxy_all_steps'])
            for w in worlds
        )
        result = {
            'status': 'COMPLETE',
            **shared,
            'evaluation': str(eval_path),
            'lift_count': lift_count,
            'tracking_count': tracking_count,
            'evaluation_denominator': 9,
            'center_lift': worlds[4]['last_0p5s_lift_window']['lift_proxy_all_steps_at_least_0p10m'],
            'center_tracking': worlds[4]['last_0p5s_lift_window']['track_reference_proxy_all_steps'],
        }
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
