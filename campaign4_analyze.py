"""Descriptive analysis of the fixed evaluation outputs; no model execution."""
import argparse, collections, csv, json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/out'));ap.add_argument('--plot',action='store_true');args=ap.parse_args()
p=args.root/'main';results=json.loads((p/'results.json').read_text());rows=json.loads((p/'token_annotations.json').read_text())
cats=['same_valid_track','different_valid_track','unknown_noise_ambiguous'];cat=np.array([r['category'] for r in rows]);layer=np.array([r['layer'] for r in rows]);distance=np.array([r['distance'] for r in rows]);occupancy=np.array([r['occupancy'] for r in rows]);event=np.array([r['event'] for r in rows])
ids=json.loads((p/'sample_selection.json').read_text())['indices'];emap={e:i for i,e in enumerate(ids)};ei=np.array([emap[e] for e in event]);n=len(ids)
group=layer*100+np.digitize(distance,[.03,.06,.1,.15,.2,.3])*10+np.digitize(occupancy,[25,35,45])
cells=[]
for g in np.unique(group):
    ns=np.array([np.sum((group==g)&(cat==c)) for c in range(2)])
    if ns.min()>=20:cells.append((g,ns))
weights=np.array([ns.sum() for g,ns in cells]);weights=weights/weights.sum()
analysis=dict(standardization=dict(cells=len(cells),minimum_per_category_per_cell=20,layer='source calibrated band 0..11',distance_edges=[.03,.06,.1,.15,.2,.3],occupancy_edges=[25,35,45],
    weighting='pooled same+different counts across common-support cells',coverage=[sum(ns[c] for g,ns in cells)/np.sum(cat==c) for c in range(2)]),models={},geometry={},validation={})
for c,name in enumerate(cats):
    m=cat==c
    analysis['geometry'][name]=dict(count=int(m.sum()),normalized_distance_quantiles=np.quantile(distance[m],[.1,.5,.9,.99]).tolist(),
        query_copy_mse=float((distance[m]**2/3).mean()),occupancy_mean=float(occupancy[m].mean()),
        reasons=collections.Counter(f"{r['source_reason']},{r['target_reason']}" for r in rows if r['category']==c))
analysis['validation']['matching_negative_labels_not_same']=sum(r['source_truth']<0 and r['source_truth']==r['target_truth'] for r in rows)
assert all(r['source_truth']>=0 and r['target_truth']>=0 and r['source_truth']==r['target_truth'] and r['source_reason']==r['target_reason']==0 for r in rows if r['category']==0)
assert all(r['source_truth']>=0 and r['target_truth']>=0 and r['source_truth']!=r['target_truth'] and r['source_reason']==r['target_reason']==0 for r in rows if r['category']==1)
arrays=np.load(p/'sample_inputs.npz');points=np.concatenate([arrays[k] for k in arrays.files]);analysis['validation']['real_tokens']=len(points);analysis['validation']['no_valid_outward_target_tokens']=len(points)-len(rows)
analysis['validation']['both_hilbert_axes_saturated_fraction']=float(np.mean(np.all(points[:,:2]*1e4>=1023,axis=1)))
losses={key:np.load(p/(key+'_losses.npz'))['loss'] for key in results}
for key,z in results.items():
    loss=losses[key];mu=np.array([[loss[(group==g)&(cat==c)].mean() for c in range(2)] for g,ns in cells]);std=(mu*weights[:,None]).sum(0)
    analysis['models'][key]=dict(standardized_same_mean=float(std[0]),standardized_different_mean=float(std[1]),standardized_gap=float(std[1]-std[0]),
        original_conditional_gap=z['categories'][cats[1]]['conditional_mean']-z['categories'][cats[0]]['conditional_mean'])
for width in [64,128,256,512,1024,1536]:
    diff=losses[f'w{width}_last']-losses[f'w{width}_best'];ec=np.bincount(ei,minlength=n);es=np.bincount(ei,weights=diff,minlength=n)
    rng=np.random.default_rng(8722);boot=[]
    for _ in range(400):
        idx=rng.integers(n,size=n);boot.append(es[idx].sum()/ec[idx].sum())
    analysis['models'][f'w{width}_last']['paired_last_minus_best']=dict(mean=float(diff.mean()),relative_percent=float(diff.mean()/losses[f'w{width}_best'].mean()*100),ci95=np.quantile(boot,[.025,.975]).tolist())
analysis['validation'].update(max_partition_error=max(abs(z['partition_pooled_error']) for z in results.values()),max_original_batch2_partition_error=max(abs(z['partition_original_batch2_error']) for z in results.values()),max_direct_original_reduction_error=max(abs(z['original_reduction_error']) for z in results.values()),max_padding_error=max(z['padding_invariance_max_abs'] for z in results.values()),max_causality_error=max(z['causality_max_abs'] for z in results.values()))
(args.root/'analysis.json').write_text(json.dumps(analysis,indent=2))
with (args.root/'summary.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['checkpoint','step','nominal_event_presentations','n_targets','pooled_mse','original_batch2_mse','same_mean','different_mean','unknown_mean','same_contribution','different_contribution','unknown_contribution'])
    for key,z in results.items():w.writerow([key,z['step'],z['nominal_single_process_event_presentations'],z['n_targets'],z['pooled_loss'],z['original_batch2_loss']]+[z['categories'][c]['conditional_mean'] for c in cats]+[z['categories'][c]['pooled_contribution'] for c in cats])
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#157a6e','#b64e32','#7261a3'];labels=['Same valid track','Different valid track','Unknown / noise / ambiguous']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    widths=[64,128,256,512,1024,1536]
    for c,name in enumerate(cats):
        axes[0].plot(widths,[results[f'w{w}_best']['categories'][name]['conditional_mean'] for w in widths],marker='o',color=colors[c],label=labels[c])
        for width,style in [(64,'--'),(1536,'-')]:
            v=np.sort(losses[f'w{width}_best'][cat==c]);axes[1].plot(v,np.arange(1,len(v)+1)/len(v),style,color=colors[c],label=f'{labels[c]}, w{width}')
    axes[0].set(xscale='log',yscale='log',xlabel='Backbone width (exposure also varies)',ylabel='Conditional normalized coordinate MSE',title='All categories improve across campaign models');axes[0].set_xticks(widths,[str(w) for w in widths]);axes[0].legend(fontsize=8)
    axes[1].set(xscale='log',xlim=(1e-7,.1),ylim=(0,1),xlabel='Per-target MSE',ylabel='Empirical cumulative fraction',title='Heavy tails: w64 dashed, w1536 solid');axes[1].legend(fontsize=6,loc='lower right')
    z=results['w1536_best'];vals=np.array([z['categories'][c]['pooled_contribution'] for c in cats]);axes[2].bar(labels,vals*1e3,color=colors);axes[2].tick_params(axis='x',labelrotation=18);axes[2].set(ylabel='Contribution to pooled MSE (×10⁻³)',title='w1536: minority categories carry 64.7% of loss')
    fig.suptitle('Campaign 4 diagnostic • 4,096 common v6.02 test events • 128,991 k=1 targets\nOriginal campaign split overlap is unverified',fontsize=12)
    fig.savefig(args.root/'loss_diagnostic.png',dpi=170);plt.close(fig)
print(json.dumps(dict(geometry=analysis['geometry'],standardization=analysis['standardization'],late_changes={k:v['paired_last_minus_best'] for k,v in analysis['models'].items() if 'paired_last_minus_best' in v},validation=analysis['validation']),indent=2))
