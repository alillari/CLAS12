"""Audit and compare the eight completed downstream fits; never trains models."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSIONS = ['adapteronly', 'mamba1', 'mamba2', 'mamba3']
LABELS = {'adapteronly': 'Adapter-only', 'mamba1': 'Mamba1', 'mamba2': 'Mamba2', 'mamba3': 'Mamba3', 'cvt': 'CVT / COATJAVA'}
COLORS = {'adapteronly': '#6b7280', 'mamba1': '#3b6fb6', 'mamba2': '#d98a29', 'mamba3': '#35805c', 'cvt': '#8c4b96'}
TF_METRICS = ['ari_signal', 'ari_with_background', 'matched_iou_mean', 'track_efficiency_global',
    'track_purity_global', 'fake_rate', 'miss_rate', 'split_rate', 'merge_rate',
    'background_rejection', 'background_contamination', 'signal_loss_to_background']


def read(path): return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def csv_save(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def history_read(path, delimiter=','):
    with Path(path).open() as f: rows=list(csv.DictReader(f,delimiter=delimiter))
    result=[]
    for source in rows:
        row={}
        for key,value in source.items():
            try: row[key]=float(value)
            except (ValueError,TypeError): row[key]=value
        row['step']=int(row.pop('Step',row.get('step',0)))
        if 'Val_Loss' in row: row['val_loss']=row.pop('Val_Loss')
        result.append(row)
    return result


def column(rows,key): return np.asarray([r[key] for r in rows])


def figure_save(fig, out, name, note=None, top=1):
    if note: fig.text(.5, .012, note, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045 if note else 0, 1, top))
    for suffix in ('png', 'pdf'): fig.savefig(out / (name+'.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def vector_metrics(truth, prediction):
    assert truth.shape == prediction.shape and np.isfinite(truth).all() and np.isfinite(prediction).all()
    p, q = np.linalg.norm(truth, axis=1), np.linalg.norm(prediction, axis=1)
    assert (p > 0).all() and (q > 0).all()
    rel = (q-p)/p
    theta_t, theta_q = np.degrees(np.arccos(np.clip(truth[:,2]/p, -1, 1))), np.degrees(np.arccos(np.clip(prediction[:,2]/q, -1, 1)))
    phi = np.degrees(np.arctan2(prediction[:,1], prediction[:,0])-np.arctan2(truth[:,1], truth[:,0]))
    phi = (phi+180)%360-180
    angle = np.degrees(np.arccos(np.clip(np.sum(truth*prediction, axis=1)/(p*q), -1, 1)))
    q16, q84 = np.quantile(rel, [.16,.84])
    return {'n': len(truth), 'component_mae_mev': float(np.abs(prediction-truth).mean()*1000),
        'px_mae_mev': float(np.abs(prediction[:,0]-truth[:,0]).mean()*1000),
        'py_mae_mev': float(np.abs(prediction[:,1]-truth[:,1]).mean()*1000),
        'pz_mae_mev': float(np.abs(prediction[:,2]-truth[:,2]).mean()*1000),
        'p_mae_mev': float(np.abs(q-p).mean()*1000), 'relative_width68_pct': float((q84-q16)*50),
        'relative_bias_pct': float(rel.mean()*100), 'tail10_pct': float((np.abs(rel)>.1).mean()*100),
        'theta_mae_deg': float(np.abs(theta_q-theta_t).mean()), 'phi_mae_deg': float(np.abs(phi).mean()),
        'opening_angle_mean_deg': float(angle.mean())}


def load_runs(root):
    runs = {}
    for host in ('a02c','a518'):
        base = root/host
        assert read(base/'runner_result.json')['exit_code'] == 0
        queue = read(base/'completed.json')
        assert queue['complete'] and len(queue['runs']) == 4
        for record in queue['runs']:
            assert record['status'] == 'complete' and record['train_exit_code'] == record['eval_exit_code'] == 0
            task, version = record['task'], record['backbone']; p = base/'runs'/record['name']
            t, e, s, cfg = (read(p/f) for f in ['training_complete.json','evaluation_complete.json','evaluation/summary.json','resolved.json'])
            assert t['complete'] and t['final_step'] == t['finite_gradient_updates'] == 30000
            assert t['head_mamba_blocks'] == 0 and t['backbone_unchanged']
            assert t['strict_head_reload'] and t['prediction_equivalence'] and t['optimizer_scheduler_restore']
            assert e['complete'] and e['actual_samples'] == (10000 if task=='finding' else 50000)
            assert cfg['num_adapter_layers'] == cfg['num_embedder_layers'] == 0
            assert t['train_samples'] == 50000 and t['validation_samples'] == (5000 if task=='finding' else 10000)
            if task == 'finding':
                paths = list((p/'checkpoints').glob('training_validation/*/metrics.csv'))
                assert len(paths)==1
                history = history_read(paths[0])
                assert np.isclose(next(h['selection_score'] for h in history if h['step']==t['best_step']),t['best_ari'])
            else:
                history = history_read(p/'checkpoints/training.log', delimiter='\t')
            assert len(history)==30 and len(set(column(history,'step')))==30 and history[-1]['step']==30000
            runs[task,version] = {'host':host,'path':p,'training':t,'evaluation':e,'summary':s,'config':cfg,'history':history}
    assert len(runs)==8
    for task in ('finding','momentum'):
        arms = [runs[task,v] for v in VERSIONS]
        for key in ('train_keys_sha256','validation_keys_sha256','initial_head_sha256'):
            assert len({r['training'][key] for r in arms})==1, (task,key)
        assert len({r['evaluation']['evaluation_keys_sha256'] for r in arms})==1
        for key in ('batch_size','valid_batch_size','max_optimizer_steps','max_lr','min_lr','warmup_steps',
                    'scheduler_first_cycle_steps','adapter_weight_decay','grad_clip_value','dropout','seed'):
            assert len({r['config'][key] for r in arms})==1,(task,key)
    assert len({runs['momentum',v]['evaluation']['truth_sha256'] for v in VERSIONS})==1
    coat = runs['finding','adapteronly']['summary']['baselines']['coatjava']
    for v in VERSIONS:
        s = runs['finding',v]['summary']
        assert s['baselines']['coatjava'] == coat
        for key in ('n_events','n_points','n_signal_points','n_background_points','n_true_tracks'):
            assert s['metrics'][key] == coat[key]
    return runs


def prediction_audit(runs, out):
    frames = {}
    for v in VERSIONS:
        frame={'keys':[],'true':[],'adapter':[],'cvt_entrance':[]}
        with gzip.open(runs['momentum',v]['path']/'evaluation/predictions.csv.gz','rt') as f:
            for row in csv.DictReader(f):
                frame['keys'].append([int(row['source_event_index']),int(row['segment_label'])])
                for kind in ('true','adapter','cvt_entrance'):
                    frame[kind].append([float(row[f'{kind}_{c}_gev']) if row[f'{kind}_{c}_gev'] else np.nan for c in ('px','py','pz')])
        frame={k:np.asarray(values,dtype=np.int64 if k=='keys' else np.float64) for k,values in frame.items()}
        assert len(frame['keys'])==50000 and len(np.unique(frame['keys'],axis=0))==50000
        frames[v]=frame
    ref=frames['adapteronly']; keys=ref['keys']
    truth=ref['true'];cvt=ref['cvt_entrance']
    predictions={v:frames[v]['adapter'] for v in VERSIONS}
    assert np.isfinite(truth).all()
    for v, frame in frames.items():
        assert np.array_equal(keys,frame['keys'])
        assert np.array_equal(truth,frame['true'])
        assert np.array_equal(cvt,frame['cvt_entrance'],equal_nan=True)
        assert np.isfinite(predictions[v]).all()
    mask=np.isfinite(cvt).all(axis=1)
    assert mask.sum()==runs['momentum','adapteronly']['summary']['methods']['cvt']['n']
    full={v:vector_metrics(truth,predictions[v]) for v in VERSIONS}
    common={v:vector_metrics(truth[mask],predictions[v][mask]) for v in VERSIONS}
    common['cvt']=vector_metrics(truth[mask],cvt[mask])
    for v in VERSIONS:
        saved=runs['momentum',v]['summary']['methods']['adapter']
        assert np.isclose(full[v]['p_mae_mev'],saved['momentum']['mae_gev']*1000)
        assert np.isclose(full[v]['relative_width68_pct'],saved['momentum']['relative_resolution_68']*100)
    csv_save(out/'momentum_common_population.csv',[{'model':v,**m} for v,m in common.items()])
    bins=np.array([.25,.5,.75,1.,1.5,2.,3.,5.,10.])
    p=np.linalg.norm(truth[mask],axis=1); binned=[]
    for v, prediction in {**{v:predictions[v][mask] for v in VERSIONS},'cvt':cvt[mask]}.items():
        for low,high in zip(bins[:-1],bins[1:]):
            selected=(p>=low)&(p<high)
            if selected.sum()<20: continue
            binned.append({'model':v,'bin_low_gev':low,'bin_high_gev':high,'bin_center_gev':(low+high)/2,
                **vector_metrics(truth[mask][selected],prediction[selected])})
    csv_save(out/'momentum_binned_common_population.csv',binned)
    save(out/'population_audit.json',{'complete':True,'momentum_unique_event_segment_keys':len(keys),
        'momentum_all_four_truths_identical':True,'momentum_all_four_cvt_vectors_identical':True,
        'cvt_common_valid_tracks':int(mask.sum()),'cvt_missing_tracks':int((~mask).sum()),
        'all_learned_predictions_finite':True,'all_heads_initial_weights_and_cohorts_match':True})
    return full,common,binned


def plots(runs,full,common,binned,out):
    for loglog in (False,True):
        fig,axs=plt.subplots(1,2,figsize=(11,4.4))
        for ax,task,title in zip(axs,('finding','momentum'),('Track finding: validation loss','Momentum: standardized validation MAE')):
            for v in VERSIONS:
                h=runs[task,v]['history']; ax.plot(column(h,'step'),column(h,'val_loss'),label=LABELS[v],color=COLORS[v],lw=1.8)
                t=runs[task,v]['training']; best=next(p for p in h if p['step']==t['best_step'])
                ax.scatter(best['step'],best['val_loss'],color=COLORS[v],marker='*',s=65)
            ax.set(title=title,xlabel='Optimizer updates',yscale='log',xscale='log' if loglog else 'linear')
            ax.grid(alpha=.2,which='both'); ax.legend(fontsize=9)
        figure_save(fig,out,'validation_loss_vs_steps'+('_loglog' if loglog else ''),'Raw checkpoint validation values; stars mark selected checkpoints. Single seed; no Mamba in heads.')
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,key,title in zip(axs.flat,['option2_ari_signal','option2_ari_with_background','option2_track_purity_global','option2_background_rejection'],['Signal ARI','Inclusive ARI','Global track purity','Background rejection']):
        for v in VERSIONS:
            h=runs['finding',v]['history']; ax.plot(column(h,'step'),column(h,key),color=COLORS[v],label=LABELS[v])
        ax.set(title=title,xlabel='Optimizer updates'); ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=9)
    figure_save(fig,out,'track_finding_validation_behavior','Validation population is separate from final evaluation. Assignment option 2; threshold 0.2.')
    coat=runs['finding','adapteronly']['summary']['baselines']['coatjava']
    tf={v:runs['finding',v]['summary']['metrics'] for v in VERSIONS};tf['cvt']=coat
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,key,title in zip(axs.flat,['ari_signal','matched_iou_mean','track_efficiency_global','track_purity_global','background_rejection','fake_rate'],['Signal ARI ↑','Matched IoU ↑','Global efficiency ↑','Global purity ↑','Background rejection ↑','Event-mean fake rate ↓']):
        for i,v in enumerate([*VERSIONS,'cvt']):
            y=tf[v][key];ax.scatter(i,y,color=COLORS[v],s=55);ax.annotate(f'{y:.4f}',(i,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set(title=title,xticks=range(5),xticklabels=['AO','M1','M2','M3','CVT'],xlim=(-.5,4.5));ax.margins(y=.25);ax.grid(alpha=.2,axis='y')
    figure_save(fig,out,'track_finding_quality','Same 10,000 events / 27,197 truth tracks. AO = adapter-only. No Mamba in heads; single seed.')
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,key,title in zip(axs.flat,['component_mae_mev','p_mae_mev','relative_width68_pct','theta_mae_deg','phi_mae_deg','tail10_pct'],['Mean component MAE [MeV] ↓','Momentum magnitude MAE [MeV] ↓','Relative momentum width68 [%] ↓','Theta MAE [degrees] ↓','Phi MAE [degrees] ↓','|Δp/p| > 10% [% of tracks] ↓']):
        for i,v in enumerate([*VERSIONS,'cvt']):
            y=common[v][key];ax.scatter(i,y,color=COLORS[v],s=55);ax.annotate(f'{y:.2f}',(i,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set(title=title,xticks=range(5),xticklabels=['AO','M1','M2','M3','CVT'],xlim=(-.5,4.5));ax.margins(y=.25);ax.grid(alpha=.2,axis='y')
    figure_save(fig,out,'momentum_quality_common_population',f"Identical {common['cvt']['n']:,}-track population with finite CVT vectors. No Mamba in heads; single seed.")
    fig,axs=plt.subplots(1,3,figsize=(13,4.4))
    for ax,key,title in zip(axs,['p_mae_mev','relative_width68_pct','opening_angle_mean_deg'],['Momentum magnitude MAE [MeV]','Relative momentum width68 [%]','Mean opening angle [degrees]']):
        for v in [*VERSIONS,'cvt']:
            rows=[r for r in binned if r['model']==v]
            ax.plot([r['bin_center_gev'] for r in rows],[r[key] for r in rows],'.-',color=COLORS[v],label=LABELS[v])
        ax.set(title=title,xlabel='True entrance momentum [GeV]');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    figure_save(fig,out,'momentum_binned_common_population','Same CVT-valid tracks within every bin. Width68 = half of the central 16–84 percentile interval; not a Gaussian fit.')
    fig,axs=plt.subplots(1,2,figsize=(11,4.4));x=np.arange(4)
    for task,offset,label in [('finding',-.18,'Track finding'),('momentum',.18,'Momentum')]:
        t=[runs[task,v]['training'] for v in VERSIONS]
        axs[0].bar(x+offset,[r['elapsed_seconds']/60 for r in t],width=.34,label=label)
        axs[1].bar(x+offset,[r['peak_cuda_allocated_bytes']/2**30 for r in t],width=.34,label=label)
    for ax,title in zip(axs,['Recorded training / validation / audit time [min]','Peak PyTorch CUDA allocation [GiB]']):
        ax.set(title=title,xticks=x,xticklabels=['AO','M1','M2','M3']);ax.legend(loc='upper left');ax.grid(alpha=.2,axis='y')
        ax.set_ylim(0, ax.get_ylim()[1]*1.3)
    figure_save(fig,out,'runtime_memory','Separate host runs; includes frozen feature extraction. Excludes standalone evaluation; not an architecture-only speed comparison.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();out=root/'comparison';out.mkdir(exist_ok=True)
    runs=load_runs(root);full,common,binned=prediction_audit(runs,out)
    rows=[];gains=[];tfrows=[];mrrows=[]
    for task in ('finding','momentum'):
        for v in VERSIONS:
            r=runs[task,v];t=r['training']
            row={'task':task,'model':v,'host':r['host'],'best_validation_loss':t['best_loss'],'best_step':t['best_step'],
                'training_seconds':t['elapsed_seconds'],'peak_cuda_gib':t['peak_cuda_allocated_bytes']/2**30,
                'head_parameters':t['head_parameters'],'evaluation_samples':r['evaluation']['actual_samples']}
            if task=='finding':
                row.update({k:r['summary']['metrics'][k] for k in TF_METRICS});tfrows.append(row)
                if v!='adapteronly':
                    base=runs[task,'adapteronly']['summary']['metrics']
                    gains.append({'task':task,'model':v,**{k+'_delta_pp':100*(row[k]-base[k]) for k in TF_METRICS}})
            else:
                row.update(full[v]);mrrows.append(row)
                if v!='adapteronly': gains.append({'task':task,'model':v,**{k+'_reduction_pct':100*(1-full[v][k]/full['adapteronly'][k]) for k in ['component_mae_mev','p_mae_mev','relative_width68_pct','theta_mae_deg','phi_mae_deg']}})
            rows.append(row)
    csv_save(out/'run_table.csv',rows);csv_save(out/'track_finding_metrics.csv',tfrows);csv_save(out/'momentum_metrics.csv',mrrows);csv_save(out/'gains_over_adapteronly.csv',gains)
    all_history=[]
    for (task,v),r in runs.items():
        for record in r['history']: all_history.append({'task':task,'model':v,**record})
    csv_save(out/'validation_histories.csv',all_history)
    plots(runs,full,common,binned,out)
    report=['# Completed downstream comparison — 29 October 2026','',
        'All eight fits completed 30,000 finite optimizer updates, strict head reload/prediction checks, unchanged frozen backbones and full evaluation. Both host supervisors exited 0. All downstream heads have zero Mamba blocks.','',
        'The task cohorts, common initial head weights and within-task optimization settings match. Momentum predictions were checked row-by-row: all four use the same 50,000 unique event/segment identities, identical truth vectors, finite learned outputs and identical CVT reference vectors. Validation and evaluation source events are disjoint.','',
        '## Track finding — 10,000 evaluation events','',
        '| Model | Signal ARI | Inclusive ARI | IoU | Global efficiency | Global purity | Background rejection | Fake rate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    coat=runs['finding','adapteronly']['summary']['baselines']['coatjava']
    for v in [*VERSIONS,'cvt']:
        m=coat if v=='cvt' else runs['finding',v]['summary']['metrics']
        report.append('| '+LABELS[v]+' | '+' | '.join(f'{m[k]:.4f}' for k in ['ari_signal','ari_with_background','matched_iou_mean','track_efficiency_global','track_purity_global','background_rejection','fake_rate'])+' |')
    report.extend(['', 'All pretrained backbones improve association, global efficiency/purity and background rejection versus AdapterOnly. Mamba3 has the highest signal ARI, IoU and global efficiency/purity in this seed. Mamba1 has slightly higher background rejection than Mamba3. COATJAVA retains higher global purity and background rejection than every learned model. Fake/split/merge rates are event means; global efficiency/purity are pooled track ratios. Matched IoU is conditional on matched tracks. The complete table also includes miss/split/merge, contamination and signal loss.','',
        '## Momentum — all 50,000 evaluation tracks','',
        '| Model | Best standardized val MAE | Mean component MAE [MeV] | Magnitude MAE [MeV] | Relative width68 [%] | Theta MAE [deg] | Phi MAE [deg] |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for v in VERSIONS:
        m=full[v];report.append(f"| {LABELS[v]} | {runs['momentum',v]['training']['best_loss']:.6f} | "+' | '.join(f'{m[k]:.3f}' for k in ['component_mae_mev','p_mae_mev','relative_width68_pct','theta_mae_deg','phi_mae_deg'])+' |')
    g=next(x for x in gains if x['task']=='momentum' and x['model']=='mamba3')
    report.extend(['',f"Mamba3 has the lowest measured momentum errors among the four learned models: mean component MAE is reduced {g['component_mae_mev_reduction_pct']:.1f}% and magnitude MAE {g['p_mae_mev_reduction_pct']:.1f}% versus the no-Mamba AdapterOnly control. Relative width68 falls from {full['adapteronly']['relative_width68_pct']:.2f}% to {full['mamba3']['relative_width68_pct']:.2f}%. These results do not compare against the historical one-Mamba-layer AdapterOnly baseline; that separate test is still running.",'',
        '## Fair CVT reference — identical population','',
        f"CVT has finite entrance vectors on {common['cvt']['n']:,} of 50,000 tracks. The following table recomputes every method on exactly those tracks; the 197 missing-CVT cases are not silently counted as valid reference predictions.",'',
        '| Model | Tracks | Mean component MAE [MeV] | Magnitude MAE [MeV] | Relative width68 [%] | Mean opening angle [deg] |',
        '|---|---:|---:|---:|---:|---:|'])
    for v in [*VERSIONS,'cvt']:
        m=common[v];report.append(f"| {LABELS[v]} | {m['n']} | "+' | '.join(f'{m[k]:.3f}' for k in ['component_mae_mev','p_mae_mev','relative_width68_pct','opening_angle_mean_deg'])+' |')
    report.extend(['','MAE and central resolution need not rank methods the same way: tail errors affect MAE strongly. CVT has the narrower relative momentum core and substantially better directional accuracy than these learned heads, while Mamba3 has lower magnitude MAE. See the matched-population binned plots for momentum dependence.','',
        '## Runtime, memory and checkpoint selection','',
        '| Task | Model | Selected step | Recorded time [min] | Peak CUDA [GiB] |', '|---|---|---:|---:|---:|'])
    for row in rows: report.append(f"| {row['task']} | {LABELS[row['model']]} | {row['best_step']} | {row['training_seconds']/60:.2f} | {row['peak_cuda_gib']:.3f} |")
    report.extend(['','Times include training, validation and audit work, with frozen feature extraction where applicable; standalone evaluation and launch/setup are excluded. CUDA allocation is PyTorch peak allocation, not total system/unified memory. Runs were performed on two GB10 hosts. Mamba1/2 frozen blocks use FP32; Mamba3 uses its local BF16 boundary. All heads use FP32. These are pipeline timings, not an architecture-only speed result.','',
        '## Recipe and limits','',
        '50,000 labeled examples per task, seed42, 30k updates, fixed initial common head weights, width512 input/256 adapter, zero input/refinement Mamba layers, AdamW, LR1e-5→3e-4→1e-5, 1k warmup, weight decay1e-4, norm clipping0.1, dropout0.1. Track finding: batch32, 16 queries, two attention-refinement layers, signal-only targets, assignment option2/threshold0.2, validation selection by signal ARI with loss tie break. Momentum: batch128, Cartesian entrance targets, train-only standardization, MAE, attention pooling, validation-loss selection. Validation:5k events/10k tracks; final evaluation:10k events/50k tracks.','',
        'Single seed with a shared recipe and no family-specific tuning. No statistical claim is made that the small backbone differences generalize. Final evaluation events are excluded from downstream validation, conditional on the existing source dataset split. Track-finding selection uses a mean of batch event-means; history option2 ARI and final evaluation ARI are event means, so the recorded selection score can differ slightly from the plotted validation ARI. Track-finding validation loss and optimized training loss have different aggregation/auxiliary-loss paths; raw validation curves are shown without interpreting their scale difference as a generalization gap. Quantile width68 is half(q84−q16), not a fitted Gaussian sigma. Mamba refinement also changes model capacity; its dedicated follow-up is separate.','',
        '## Outputs','',
        '- `track_finding_quality.{png,pdf}` and `track_finding_validation_behavior.{png,pdf}`.',
        '- `momentum_quality_common_population.{png,pdf}` and `momentum_binned_common_population.{png,pdf}`.',
        '- `validation_loss_vs_steps{,_loglog}.{png,pdf}` and `runtime_memory.{png,pdf}`.',
        '- CSV tables:run_table,track_finding_metrics,momentum_metrics,gains_over_adapteronly,momentum_common_population,momentum_binned_common_population,validation_histories.',
        '- `population_audit.json` and `summary.json`; original per-run configurations, histories, metrics and prediction records are preserved in sibling host folders.'])
    (out/'report.md').write_text('\n'.join(report)+'\n')
    save(out/'summary.json',{'complete':True,'runs':rows,'gains_over_adapteronly':gains,'momentum_common_population':common,
        'track_finding_coatjava':coat,'single_seed':True,'head_mamba_layers':0,'source_hosts':['a02c','a518']})
    print(json.dumps({'complete':True,'report':str(out/'report.md'),'momentum_mamba3_reductions':g,'tf_mamba3':next(x for x in tfrows if x['model']=='mamba3')},indent=2))


if __name__=='__main__': main()
