#!/usr/bin/env python3
"""Task-matched summaries for the frozen two-task observed-object diagnostic."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prepare_v2_robust_visuals import contact_sheets
from probe_v2_causal_velocity import estimate_object_velocity

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/native_v2_01/feedback_coverage01'
GROUPS = ('base', 'base_duplicate', 'nominal_object', 'observed_object')
TASK_BANKS = {'pour_tube': 5, 'wipe_board': 1}


def check_collected_interface(task, seed):
    out = RESULTS / task / f's{seed}'
    actual = TASK_BANKS[task]
    d = np.load(out/'trajectories.npz')
    prefix = np.load(out/'prefix.npz')
    boundary = np.load(out/'boundary_state.npz')
    est = np.load(out/'causal_velocity_estimate.npz')
    checks = json.loads((out/'reconstruction_checks.json').read_text())
    assert checks['gpu_dynamic_checks'] and all(checks['common_inputs'].values())
    assert d['qpos'].shape == (450,4,1,25) and d['qvel'].shape == (450,4,1,24)
    assert d['ctrl'].shape == (4,450,18) and tuple(d['group_order']) == GROUPS
    assert np.array_equal(est['prefix_array_indices'], [147,149])
    assert np.array_equal(est['input_object_qpos'], prefix['qpos'][[147,149],actual,18:25])
    assert np.array_equal(est['input_timestamps'], prefix['time'][[147,149],actual])
    estimate = estimate_object_velocity(est['input_object_qpos'], est['input_timestamps'])[0]
    assert np.allclose(estimate, est['estimated_object_qvel_actual_body'], rtol=0, atol=1e-12)
    assert np.allclose(np.diff(d['relative_time'],axis=0),.01,rtol=1e-3,atol=1e-6)
    opts = {a:np.load(out/f'optimization_{a}.npz') for a in GROUPS[2:]}
    recs = {a:np.load(out/f'reconstruction_{a}.npz') for a in GROUPS[2:]}
    for i,a in enumerate(GROUPS):
        assert np.array_equal(d['qpos'][:150,i,0],prefix['qpos'][:,actual])
        assert np.array_equal(d['qvel'][:150,i,0],prefix['qvel'][:,actual])
        assert np.isfinite(d['qpos'][:,i]).all() and np.isfinite(d['qvel'][:,i]).all()
        if i < 2:
            continue
        opt,rec,check = opts[a],recs[a],checks['arms'][a]
        for field in ('requested_basic_all1024_exact_before_forward','requested_basic_all1024_exact_after_forward',
                      'all31_fields_finite_after_forward','static_world_matches_actual_prefix'):
            assert all(check[field].values())
        assert check['qacc_warmstart_zero']
        assert np.array_equal(rec['requested_qpos'][0,:18],boundary['bank_qpos'][actual,:18])
        assert np.array_equal(rec['requested_qvel'][0,:18],boundary['bank_qvel'][actual,:18])
        assert np.array_equal(opt['reference_indices'],np.arange(151,451))
        for k in ('rng_cpu_start','rng_cuda_start','warmstart_ctrl'):
            assert np.array_equal(opt[k],opts['nominal_object'][k])
        expected = opt['returned_ctrl'] if bool(opt['succeeded']) else opt['warmstart_ctrl']
        assert np.array_equal(opt['effective_ctrl'],expected)
        assert np.array_equal(d['ctrl'][i,150:],expected)
    assert np.array_equal(recs['nominal_object']['requested_qpos'][0,18:],boundary['bank_qpos'][4,18:])
    assert np.array_equal(recs['nominal_object']['requested_qvel'][0,18:],boundary['bank_qvel'][4,18:])
    assert np.array_equal(recs['observed_object']['requested_qpos'][0,18:],boundary['bank_qpos'][actual,18:])
    assert np.array_equal(recs['observed_object']['requested_qvel'][0,18:],estimate.astype(np.float32))
    assert np.array_equal(d['ctrl'][0],d['ctrl'][1])
    assert np.array_equal(d['ctrl'][0,150:],opts['nominal_object']['warmstart_ctrl'])
    restore = json.loads((out/'bank_fields_exact.json').read_text())['execution_all1024']
    assert len(restore)==31 and all(restore.values())
    result = {'status':'PASS','task':task,'seed':seed,'actual_bank':actual,'causal_steps':[148,150],
              'common_robot_state_and_rng':True,'task_specific_object_state_sources':True,
              'same_actual_execution_prefix_restore':True,'acceptance_fallback_matches':True}
    (out/'collected_interface_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def analyze_seed(task,seed):
    out = RESULTS / task / f's{seed}'
    if not (out/'result.json').is_file():
        return {'task':task,'seed':seed,'status':'PENDING'}
    check_collected_interface(task,seed)
    c = json.loads((out/'condition_result.json').read_text())
    optimizations = json.loads((out/'optimizations.json').read_text())
    d = np.load(out/'trajectories.npz')
    q = d['qpos'][:,:,0]
    ref = np.load(ROOT/f'results/native_v2_01/coverage01/{task}/nominal_s0/official_loaded_reference.npz')['qpos']
    pos = np.linalg.norm(q[:391,:,18:21]-ref[1:392,None,18:21],axis=-1)
    a,b = q[:391,:,21:25],ref[1:392,None,21:25]
    angle = 2*np.arccos(np.clip(np.abs((a*b).sum(-1))/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)),0,1))
    groups = {}
    for i,g in enumerate(GROUPS):
        post,tail = slice(150,391),slice(341,391)
        groups[g] = {
            'last50_real_steps':[342,391],
            'terminal_tracking_all_steps':bool(np.all((pos[tail,i]<.1)&(angle[tail,i]<.5))),
            'terminal_position_mean_cm':float(pos[tail,i].mean()*100),
            'terminal_position_max_cm':float(pos[tail,i].max()*100),
            'terminal_so3_mean_rad':float(angle[tail,i].mean()),
            'terminal_so3_max_rad':float(angle[tail,i].max()),
            'post_feedback_real_steps':[151,391],
            'post_feedback_position_mean_cm':float(pos[post,i].mean()*100),
            'post_feedback_so3_mean_deg':float(np.rad2deg(angle[post,i]).mean()),
            'lift_not_used_as_task_metric':True,
        }
    vis = out/'visuals'
    vis.mkdir(exist_ok=True)
    fig,axes = plt.subplots(3,1,figsize=(11,10))
    time = np.arange(1,451)*.01
    axes[0].plot(time[:391],ref[1:392,20],'k--',label='reference')
    for i,g in enumerate(GROUPS):
        axes[0].plot(time,q[:,i,20],label=g)
        axes[1].plot(time[:391],pos[:,i]*100,label=g)
        axes[2].plot(time[:391],np.rad2deg(angle[:,i]),label=g)
    for ax,title in zip(axes,('Object root height (m)','Root position error (cm)','SO3 error (degrees)')):
        ax.set_ylabel(title);ax.axvline(1.5,color='black',alpha=.4);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].axvspan(3.91,4.5,color='grey',alpha=.1)
    axes[2].set_xlabel('Physical time (seconds)')
    fig.suptitle(f'{task} / search seed{seed}; actual initial bank{TASK_BANKS[task]}')
    fig.tight_layout();plot=vis/'feedback_coverage_errors.png';fig.savefig(plot,dpi=130);plt.close(fig)
    sheets={}
    for label,path in c.get('videos',{}).items():
        video=out/Path(path).name
        if video.is_file():sheets[label]=contact_sheets(video,vis,label)
    return {'task':task,'seed':seed,'status':'COMPLETE','groups':groups,'optimizations':optimizations,
            'plot':str(plot),'contact_sheets':sheets}


def frozen_task_outcome(records):
    if len(records)!=3 or not all(r['status']=='COMPLETE' for r in records):
        return {'outcome':'PENDING'}
    count={g:sum(r['groups'][g]['terminal_tracking_all_steps'] for r in records) for g in GROUPS}
    accepted=sum(r['optimizations']['observed_object']['succeeded'] and
                 r['groups']['observed_object']['terminal_tracking_all_steps'] for r in records)
    noise=abs(count['base']-count['base_duplicate'])
    gain_nom=count['observed_object']-count['nominal_object']
    gain_base=count['observed_object']-count['base']
    signal=accepted>=2 and gain_nom>noise and gain_base>noise
    return {'outcome':'OBSERVED_OBJECT_RECOVERY_SIGNAL' if signal else 'NO_FROZEN_TASK_RECOVERY_SIGNAL',
            'tracking_counts':count,'observed_accepted_tracking_count':accepted,
            'observed_minus_nominal':gain_nom,'observed_minus_base':gain_base,
            'null_tracking_count_difference':noise}


if __name__=='__main__':
    tasks={}
    for task in TASK_BANKS:
        records=[analyze_seed(task,s) for s in range(3)]
        tasks[task]={**frozen_task_outcome(records),'seeds':records}
    complete=all(t['outcome']!='PENDING' for t in tasks.values())
    summary={'status':'COMPLETE' if complete else 'PARTIAL','tasks':tasks,
             'both_development_tasks_show_frozen_signal':complete and all(t['outcome']=='OBSERVED_OBJECT_RECOVERY_SIGNAL' for t in tasks.values()),
             'scope':'Two preselected development failures with existing seed0 controls and three recovery seeds each; accurate simulator poses; no liquid/wiping/real-time/generalization/novelty claim.'}
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({**summary,'tasks':{k:{a:b for a,b in v.items() if a!='seeds'} for k,v in tasks.items()}},indent=2))
