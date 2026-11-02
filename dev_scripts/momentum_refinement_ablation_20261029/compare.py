"""Audit the completed 0/1-layer AdapterOnly test and make its comparison artifacts."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dev_scripts.backbone_downstream_20261029.compare import read,save,csv_save,history_read,column,vector_metrics,figure_save

LABELS={'without':'No Mamba refinement','with':'One Mamba2 refinement','cvt':'CVT / COATJAVA'}
COLORS={'without':'#6b7280','with':'#35805c','cvt':'#8c4b96'}


def frame(path):
    data={'keys':[],'true':[],'adapter':[],'cvt_entrance':[]}
    with gzip.open(path,'rt') as f:
        for row in csv.DictReader(f):
            data['keys'].append([int(row['source_event_index']),int(row['segment_label'])])
            for kind in ('true','adapter','cvt_entrance'):
                data[kind].append([float(row[f'{kind}_{c}_gev']) if row[f'{kind}_{c}_gev'] else np.nan for c in ('px','py','pz')])
    data={k:np.asarray(v,dtype=np.int64 if k=='keys' else np.float64) for k,v in data.items()}
    assert len(data['keys'])==50000 and len(np.unique(data['keys'],axis=0))==50000
    return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();newroot=root/'refinement/a518';out=root/'refinement/comparison';out.mkdir(parents=True,exist_ok=True)
    assert read(newroot/'runner_result.json')['exit_code']==0 and read(newroot/'completed.json')['complete']
    pair=read(newroot/'comparison.json');assert pair['complete']
    pilot=read(newroot/'pilot_complete.json');assert pilot['complete'] and pilot['training']['finite_gradient_updates']==5
    assert pilot['training']['strict_head_reload'] and pilot['training']['prediction_equivalence']
    assert pilot['evaluation']['actual_samples']==64
    paths={'without':root/'a518/runs/momentum_adapteronly','with':newroot/'with_mamba2'}
    records={}
    for name,path in paths.items():
        t,e,s,cfg=(read(path/f) for f in ['training_complete.json','evaluation_complete.json','evaluation/summary.json','resolved.json'])
        assert t['complete'] and t['finite_gradient_updates']==t['final_step']==30000
        assert t['strict_head_reload'] and t['prediction_equivalence'] and t['optimizer_scheduler_restore'] and t['backbone_unchanged']
        assert t['head_mamba_blocks']==(0 if name=='without' else 1)
        assert t['head_parameters']==(232720 if name=='without' else 664744)
        assert e['actual_samples']==e['adapter_valid_samples']==50000
        assert cfg['use_amp'] is False and cfg['adapter_version']=='mamba2' and cfg['num_embedder_layers']==0
        h=history_read(path/'checkpoints/training.log',delimiter='\t')
        assert len(h)==30 and len(set(column(h,'step')))==30 and h[-1]['step']==30000
        records[name]={'training':t,'evaluation':e,'summary':s,'config':cfg,'history':h}
    a,b=records['without'],records['with']
    assert b['training']['shared_initial_head_sha256']==a['training']['initial_head_sha256']
    for key in ('train_keys_sha256','validation_keys_sha256','train_samples','validation_samples','final_step'):
        assert a['training'][key]==b['training'][key],key
    for key in ('evaluation_keys_sha256','truth_sha256','actual_samples','adapter_valid_samples'):
        assert a['evaluation'][key]==b['evaluation'][key],key
    for key in ('batch_size','valid_batch_size','limit_size','limit_test_size','seed','max_optimizer_steps','max_lr','min_lr',
                'warmup_steps','scheduler_first_cycle_steps','adapter_weight_decay','grad_clip_value','dropout','regression_loss','pooling','embed_dim'):
        assert a['config'][key]==b['config'][key],key
    data={name:frame(path/'evaluation/predictions.csv.gz') for name,path in paths.items()}
    for key in ('keys','true','cvt_entrance'):
        assert np.array_equal(data['without'][key],data['with'][key],equal_nan=True),key
    truth=data['without']['true'];cvt=data['without']['cvt_entrance'];pred={n:d['adapter'] for n,d in data.items()}
    assert np.isfinite(truth).all() and all(np.isfinite(x).all() for x in pred.values())
    mask=np.isfinite(cvt).all(axis=1);assert mask.sum()==49803
    metrics={n:vector_metrics(truth,p) for n,p in pred.items()}
    common={n:vector_metrics(truth[mask],p[mask]) for n,p in pred.items()};common['cvt']=vector_metrics(truth[mask],cvt[mask])
    keys=['component_mae_mev','p_mae_mev','relative_width68_pct','theta_mae_deg','phi_mae_deg','opening_angle_mean_deg','tail10_pct']
    for name,m in metrics.items():
        s=records[name]['summary']['methods']['adapter']
        assert np.isclose(m['p_mae_mev'],1000*s['momentum']['mae_gev'])
        assert np.isclose(m['relative_width68_pct'],100*s['momentum']['relative_resolution_68'])
    gains={k:100*(1-metrics['with'][k]/metrics['without'][k]) for k in keys}
    gains['validation_mae']=100*(1-b['training']['best_loss']/a['training']['best_loss'])
    rows=[]
    for name in ('without','with'):
        t=records[name]['training']
        rows.append({'model':name,'total_parameters':t['head_parameters'],'declared_trainable_parameters':200464 if name=='without' else 632488,
            'best_validation_mae':t['best_loss'],'best_step':t['best_step'],'finite_updates':t['finite_gradient_updates'],
            'training_seconds':t['elapsed_seconds'],'peak_cuda_gib':t['peak_cuda_allocated_bytes']/2**30,**metrics[name]})
    csv_save(out/'metrics.csv',rows);csv_save(out/'common_cvt_population.csv',[{'model':n,**m} for n,m in common.items()])
    csv_save(out/'relative_improvements.csv',[{'metric':k,'reduction_percent':v} for k,v in gains.items()])
    csv_save(out/'validation_histories.csv',[{'model':n,**r} for n in records for r in records[n]['history']])
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for ax,loglog in zip(axs,(False,True)):
        for name,r in records.items():
            h=r['history'];ax.plot(column(h,'step'),column(h,'val_loss'),color=COLORS[name],label=LABELS[name],lw=1.8)
            t=r['training'];ax.scatter(t['best_step'],t['best_loss'],marker='*',s=70,color=COLORS[name])
        ax.set(title='Validation MAE'+(' — log–log' if loglog else ''),xlabel='Optimizer updates',yscale='log',xscale='log' if loglog else 'linear')
        ax.grid(alpha=.2,which='both');ax.legend(fontsize=9)
    figure_save(fig,out,'validation_learning_curves','Raw standardized validation MAE; selected checkpoints marked. Same 50k training / 10k validation tracks, seed 42, FP32.')
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for ax,key,title in zip(axs.flat,['component_mae_mev','p_mae_mev','relative_width68_pct','theta_mae_deg','phi_mae_deg','tail10_pct'],
        ['Mean component MAE [MeV] ↓','Magnitude MAE [MeV] ↓','Relative momentum width68 [%] ↓','Theta MAE [degrees] ↓','Phi MAE [degrees] ↓','|Δp/p| >10% [% of tracks] ↓']):
        for i,n in enumerate(('without','with','cvt')):
            y=common[n][key];ax.scatter(i,y,color=COLORS[n],s=65);ax.annotate(f'{y:.2f}',(i,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        ax.set(title=title,xticks=range(3),xticklabels=['No Mamba','One Mamba2','CVT'],xlim=(-.5,2.5));ax.margins(y=.25);ax.grid(alpha=.2,axis='y')
    figure_save(fig,out,'quality_common_population','Same 49,803 tracks with valid CVT vectors. Both learned models are AdapterOnly; single seed. Lower is better.')
    bins=np.array([.25,.5,.75,1.,1.5,2.,3.,5.,10.]);p=np.linalg.norm(truth[mask],axis=1);binned=[]
    for name,prediction in {**{n:x[mask] for n,x in pred.items()},'cvt':cvt[mask]}.items():
        for low,high in zip(bins[:-1],bins[1:]):
            sel=(p>=low)&(p<high)
            if sel.sum()<20:continue
            binned.append({'model':name,'bin_low_gev':low,'bin_high_gev':high,'bin_center_gev':(low+high)/2,
                **vector_metrics(truth[mask][sel],prediction[sel])})
    csv_save(out/'binned_common_population.csv',binned)
    fig,axs=plt.subplots(1,3,figsize=(12,4.4))
    for ax,key,title in zip(axs,['p_mae_mev','relative_width68_pct','opening_angle_mean_deg'],['Magnitude MAE [MeV]','Relative momentum width68 [%]','Mean opening angle [degrees]']):
        for name in ('without','with','cvt'):
            selected=[r for r in binned if r['model']==name]
            ax.plot([r['bin_center_gev'] for r in selected],[r[key] for r in selected],'.-',color=COLORS[name],label=LABELS[name])
        ax.set(title=title,xlabel='True entrance momentum [GeV]');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    figure_save(fig,out,'momentum_binned','Identical CVT-valid population within every bin. Width68 = half(q84−q16), not a Gaussian sigma.')
    fig,axs=plt.subplots(1,3,figsize=(12,4.2))
    for ax,key,title,scale in zip(axs,['declared_trainable_parameters','training_seconds','peak_cuda_gib'],['Declared trainable parameters [thousands]','Training / validation / audit time [min]','Peak PyTorch CUDA allocation [GiB]'],[.001,1/60,1]):
        vals=[r[key]*scale for r in rows]
        ax.bar([0,1],vals,color=[COLORS['without'],COLORS['with']],width=.6)
        for i,y in enumerate(vals):ax.text(i,y,f'{y:.2f}',ha='center',va='bottom',fontsize=9)
        ax.set(title=title,xticks=[0,1],xticklabels=['No Mamba','One Mamba2'],ylim=(0,max(vals)*1.2));ax.grid(alpha=.2,axis='y')
    figure_save(fig,out,'capacity_runtime_memory','Same a518 GPU and FP32 execution. Adds 432,024 parameters; recorded runtime includes data/validation/audit work.')
    old=read(root/'comparison/summary.json');m3=next(r for r in old['runs'] if r['task']=='momentum' and r['model']=='mamba3')
    report=['# AdapterOnly momentum: does Mamba refinement matter?','',
        'Both arms completed 30,000 finite optimizer updates and evaluated 50,000 tracks on a518. The new arm passed its GPU pilot, strict checkpoint/prediction reloads and optimizer/scheduler restoration. The supervisor exited with code 0.','',
        'Shared initial head weights, train/validation/evaluation identities, target statistics, complete truth vectors, optimization settings and FP32 execution match. Predictions were checked row by row for 50,000 unique event/segment identities. Both heads use zero input-embedder Mamba layers; the treatment adds one Mamba2 refinement block at width 256, state size 64, convolution width 4 and expansion 2, plus its normalization.','',
        '| Metric | No refinement | One Mamba2 layer | Reduction |', '|---|---:|---:|---:|']
    report.append(f"| Best standardized validation MAE | {a['training']['best_loss']:.6f} | {b['training']['best_loss']:.6f} | {gains['validation_mae']:.1f}% |")
    for key,label in [('component_mae_mev','Mean component MAE [MeV]'),('p_mae_mev','Magnitude MAE [MeV]'),('relative_width68_pct','Relative momentum width68 [%]'),('theta_mae_deg','Theta MAE [degrees]'),('phi_mae_deg','Phi MAE [degrees]'),('opening_angle_mean_deg','Mean opening angle [degrees]'),('tail10_pct','Relative-error tail above10% [% of tracks]')]:
        report.append(f"| {label} | {metrics['without'][key]:.3f} | {metrics['with'][key]:.3f} | {gains[key]:.1f}% |")
    report.extend(['',f"The refinement layer makes a large measured difference in this seed: removing it raises component MAE by {metrics['without']['component_mae_mev']/metrics['with']['component_mae_mev']:.2f}× and magnitude MAE by {metrics['without']['p_mae_mev']/metrics['with']['p_mae_mev']:.2f}×. The layer allows hit representations to incorporate sequence context before pooling, but this experiment changes capacity as well as processing structure, so it does not isolate that mechanism at equal parameter count.",'',
        '## Capacity and measured cost','', '| Quantity | No refinement | One Mamba2 layer |','|---|---:|---:|',
        '| Total head parameters |232,720|664,744|','| Declared trainable head parameters |200,464|632,488|'])
    report.extend([f"| Recorded training/validation/audit time [min] | {rows[0]['training_seconds']/60:.2f} | {rows[1]['training_seconds']/60:.2f} |",
        f"| Peak PyTorch CUDA allocation [GiB] | {rows[0]['peak_cuda_gib']:.3f} | {rows[1]['peak_cuda_gib']:.3f} |",'',
        'The added block plus normalization contributes 432,024 parameters. Both heads include 32,256 frozen coordinate-encoding parameters and 12 declared-trainable depth-mixing coefficients unused in AdapterOnly mode. Timings exclude standalone evaluation; they include data, validation and audit overhead and are not isolated kernel benchmarks.','',
        '## Matched CVT reference','',
        'CVT has finite entrance vectors for 49,803 tracks; its 197 missing cases are excluded from every method in this table. All three methods use exactly the same remaining population.','',
        '| Method | Mean component MAE [MeV] | Magnitude MAE [MeV] | Relative width68 [%] | Mean opening angle [degrees] |', '|---|---:|---:|---:|'])
    for name in ('without','with','cvt'):
        report.append('| '+LABELS[name]+' | '+' | '.join(f'{common[name][k]:.3f}' for k in ['component_mae_mev','p_mae_mev','relative_width68_pct','opening_angle_mean_deg'])+' |')
    report.extend(['','The refined AdapterOnly model beats this CVT reference on component/magnitude MAE and relative momentum width68. CVT retains slightly better mean opening angle and substantially better phi MAE; theta MAE is close. See the common-population plots and CSV for the full metric set.','',
        '## What this changes about the earlier backbone comparison','',
        f"The refined AdapterOnly component MAE is {metrics['with']['component_mae_mev']:.2f} MeV, compared with {m3['component_mae_mev']:.2f} MeV for frozen Mamba3 plus the zero-Mamba head. Its magnitude MAE is {metrics['with']['p_mae_mev']:.2f} versus {m3['p_mae_mev']:.2f} MeV. That is an unequal-head observation, not a controlled test of pretraining: the AdapterOnly head now has more trainable capacity and sequence refinement. The previously reported 52.0% component-MAE gain remains correct against the zero-layer control; it is not evidence of an advantage over a stronger refined AdapterOnly baseline. A comparison of pretraining against this stronger baseline must give every arm the same one-layer head. No additional training was launched by this analysis.",'',
        '## Fixed recipe and limits','',
        '50k training truth tracks, 10k validation tracks, 50k final evaluation tracks on disjoint source events; seed 42; batch 128; Cartesian entrance momentum targets; train-only target normalization; standardized MAE; attention pooling; input width 512 and latent width 256; AdamW LR 1e-5→3e-4→1e-5 over 30k updates with 1k warmup; weight decay 1e-4; norm clipping 0.1; dropout 0.1. Same source image and a518 GPU. Validation-loss checkpoint selection chooses 29k steps without refinement and 30k with refinement. Single-seed evidence; no family-specific tuning and no parameter-matched alternative. This is a matched structural ablation, not a numerical reproduction of older campaigns. Width68 denotes half(q84−q16), not a Gaussian fit.','',
        '## Artifacts','',
        '- `validation_learning_curves.{png,pdf}`: linear-x and log-log raw validation MAE.',
        '- `quality_common_population.{png,pdf}` and `momentum_binned.{png,pdf}`: matched CVT population.',
        '- `capacity_runtime_memory.{png,pdf}`; `metrics.csv`, `common_cvt_population.csv`, `binned_common_population.csv`, `relative_improvements.csv`, `validation_histories.csv`.',
        '- `audit.json`, `summary.json` and the original treatment artifacts in the sibling `a518` folder; the original control is preserved under the main report a518 folder.'])
    (out/'report.md').write_text('\n'.join(report)+'\n')
    save(out/'audit.json',{'complete':True,'pilot_finite_updates':5,'full_finite_updates_per_arm':30000,'evaluation_tracks_per_arm':50000,
        'identities_unique_and_equal':True,'truth_and_cvt_vectors_equal':True,'shared_initial_head_weights_equal':True,'training_recipe_matched':True,
        'heads_fp32':True,'common_cvt_valid_tracks':int(mask.sum()),'added_parameters':432024})
    save(out/'summary.json',{'complete':True,'models':rows,'reductions_percent':gains,'common_population':common,'single_seed':True,'added_parameters':432024})
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for base in paths.values() for p in [base/'training_complete.json',base/'evaluation_complete.json',base/'evaluation/summary.json',base/'resolved.json']}
    save(out/'source_artifact_sha256.json',sources)
    print(json.dumps({'complete':True,'report':str(out/'report.md'),'metrics':rows,'reductions':gains},indent=2))


if __name__=='__main__':main()
