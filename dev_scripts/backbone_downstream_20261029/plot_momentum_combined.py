"""Show both AdapterOnly heads and the three frozen-backbone momentum fits."""
import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dev_scripts.backbone_downstream_20261029.compare import read,save,csv_save,history_read,column,figure_save

ORDER=['ao0','ao1','mamba1','mamba2','mamba3']
COLORS={'ao0':'#6b7280','ao1':'#8c4b96','mamba1':'#3b6fb6','mamba2':'#d98a29','mamba3':'#35805c'}
LABELS={'ao0':'Adapter-only · no refinement','ao1':'Adapter-only · one Mamba2 layer',
    'mamba1':'Frozen Mamba1 · no refinement','mamba2':'Frozen Mamba2 · no refinement','mamba3':'Frozen Mamba3 · no refinement'}
SHORT=['AO\n0 layers','AO\n1 layer','M1\n0 layers','M2\n0 layers','M3\n0 layers']


def csv_read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();out=root/'momentum_combined';out.mkdir(exist_ok=True)
    main=read(root/'comparison/summary.json');ablation=read(root/'refinement/comparison/summary.json')
    assert main['complete'] and ablation['complete']
    paths={'ao0':root/'a518/runs/momentum_adapteronly','ao1':root/'refinement/a518/with_mamba2',
        'mamba1':root/'a02c/runs/momentum_mamba1','mamba2':root/'a518/runs/momentum_mamba2','mamba3':root/'a02c/runs/momentum_mamba3'}
    audits={k:read(p/'training_complete.json') for k,p in paths.items()}
    evaluations={k:read(p/'evaluation_complete.json') for k,p in paths.items()}
    for k in ORDER:
        t,e=audits[k],evaluations[k]
        assert t['complete'] and t['final_step']==t['finite_gradient_updates']==30000
        assert t['head_mamba_blocks']==(1 if k=='ao1' else 0)
        assert e['actual_samples']==50000
    for field in ('train_keys_sha256','validation_keys_sha256'):
        assert len({t[field] for t in audits.values()})==1
    assert len({t.get('shared_initial_head_sha256',t['initial_head_sha256']) for t in audits.values()})==1
    for field in ('evaluation_keys_sha256','truth_sha256'):
        assert len({e[field] for e in evaluations.values()})==1
    common={k:main['momentum_common_population']['adapteronly' if k=='ao0' else k] for k in ORDER if k!='ao1'}
    common['ao1']=ablation['common_population']['with']
    cvt=main['momentum_common_population']['cvt']
    for key,value in ablation['common_population']['without'].items():
        assert np.isclose(value,common['ao0'][key]),key
    for key,value in ablation['common_population']['cvt'].items():assert np.isclose(value,cvt[key]),key
    assert all(m['n']==49803 for m in common.values())
    history={k:history_read(p/'checkpoints/training.log',delimiter='\t') for k,p in paths.items()}
    assert all(len(h)==30 and h[-1]['step']==30000 for h in history.values())
    rows=[{'model':k,'label':LABELS[k],'head_mamba_layers':audits[k]['head_mamba_blocks'],
        'head_total_parameters':audits[k]['head_parameters'],**common[k]} for k in ORDER]
    csv_save(out/'metrics_common_population.csv',rows+[{'model':'cvt','label':'CVT reference',**cvt}])
    full={('ao0' if r['model']=='adapteronly' else r['model']):r for r in main['runs'] if r['task']=='momentum'}
    full['ao1']=next(r for r in ablation['models'] if r['model']=='with')
    csv_save(out/'metrics_all_50000_tracks.csv',[{'model':k,'label':LABELS[k],**{a:b for a,b in full[k].items() if a!='model'}} for k in ORDER])
    csv_save(out/'validation_histories.csv',[{'model':k,'label':LABELS[k],**r} for k in ORDER for r in history[k]])
    fig,axs=plt.subplots(1,2,figsize=(12,5.4))
    for ax,loglog in zip(axs,(False,True)):
        for k in ORDER:
            h=history[k];line='--' if k=='ao1' else '-'
            ax.plot(column(h,'step'),column(h,'val_loss'),label=LABELS[k],color=COLORS[k],ls=line,lw=2 if k=='ao1' else 1.7)
            t=audits[k];ax.scatter(t['best_step'],t['best_loss'],marker='*',s=65,color=COLORS[k],zorder=4)
        ax.set(title='Momentum validation MAE'+(' — log–log' if loglog else ''),xlabel='Optimizer updates',yscale='log',xscale='log' if loglog else 'linear')
        ax.grid(alpha=.2,which='both')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=8.5,frameon=False)
    figure_save(fig,out,'validation_learning_curves','Only the purple adapter-only model has Mamba refinement. Shared recipe, 30k updates, single seed; heads have unequal capacity.',top=.83)
    specs=[('component_mae_mev','Mean component MAE [MeV] ↓'),('p_mae_mev','Momentum magnitude MAE [MeV] ↓'),
        ('relative_width68_pct','Relative momentum width68 [%] ↓'),('theta_mae_deg','Theta MAE [degrees] ↓'),
        ('phi_mae_deg','Phi MAE [degrees] ↓'),('tail10_pct','|Δp/p| >10% [% of tracks] ↓')]
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for ax,(key,title) in zip(axs.flat,specs):
        for i,k in enumerate(ORDER):
            y=common[k][key];ax.scatter(i,y,color=COLORS[k],s=65);ax.annotate(f'{y:.2f}',(i,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(cvt[key],ls='--',color='#252525',alpha=.7,lw=1.2,label='CVT reference')
        ax.set(title=title,xticks=range(5),xticklabels=SHORT,xlim=(-.5,4.5));ax.margins(y=.25);ax.grid(alpha=.2,axis='y');ax.tick_params(axis='x',labelsize=8)
    axs.flat[0].legend(fontsize=8,loc='upper right')
    figure_save(fig,out,'momentum_quality','Same 49,803 CVT-valid tracks. AO = adapter-only; M1/M2/M3 = frozen backbones. Layer counts refer to downstream refinement.')
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for ax,key,label in zip(axs,['px_mae_mev','py_mae_mev','pz_mae_mev'],['px','py','pz']):
        for i,k in enumerate(ORDER):
            y=common[k][key];ax.scatter(i,y,color=COLORS[k],s=65);ax.annotate(f'{y:.2f}',(i,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(cvt[key],ls='--',color='#252525',alpha=.7,lw=1.2,label='CVT reference')
        ax.set(title=label+' MAE [MeV] ↓',xticks=range(5),xticklabels=SHORT,xlim=(-.5,4.5));ax.margins(y=.25);ax.grid(alpha=.2,axis='y');ax.tick_params(axis='x',labelsize=8)
    axs[0].legend(fontsize=8)
    figure_save(fig,out,'component_errors','Same 49,803 tracks. AO = adapter-only; M1/M2/M3 = frozen backbones. Only purple has Mamba refinement and extra head capacity.')
    bins=csv_read(root/'comparison/momentum_binned_common_population.csv')
    bins=[{**r,'model':'ao0' if r['model']=='adapteronly' else r['model']} for r in bins]
    extra=csv_read(root/'refinement/comparison/binned_common_population.csv')
    bins.extend({**r,'model':'ao1'} for r in extra if r['model']=='with')
    for r in extra:
        if r['model']!='without':continue
        original=next(b for b in bins if b['model']=='ao0' and b['bin_low_gev']==r['bin_low_gev'])
        assert int(original['n'])==int(r['n'])
        assert np.isclose(float(original['p_mae_mev']),float(r['p_mae_mev']))
    for low in set(r['bin_low_gev'] for r in bins):
        assert len({int(r['n']) for r in bins if r['bin_low_gev']==low})==1
    csv_save(out/'binned_metrics_common_population.csv',bins)
    fig,axs=plt.subplots(1,3,figsize=(14,4.9))
    for ax,key,title in zip(axs,['p_mae_mev','relative_width68_pct','opening_angle_mean_deg'],['Magnitude MAE [MeV]','Relative momentum width68 [%]','Mean opening angle [degrees]']):
        for k in [*ORDER,'cvt']:
            selected=sorted([r for r in bins if r['model']==k],key=lambda r:float(r['bin_center_gev']))
            ax.plot([float(r['bin_center_gev']) for r in selected],[float(r[key]) for r in selected],
                color=COLORS.get(k,'#252525'),ls='--' if k in ('ao1','cvt') else '-',marker='.',
                lw=2 if k=='ao1' else 1.5,label=LABELS.get(k,'CVT reference'))
        ax.set(title=title,xlabel='True entrance momentum [GeV]');ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=8,frameon=False)
    figure_save(fig,out,'momentum_binned','Identical CVT-valid population in every bin. Purple alone has refinement; width68 = half(q84−q16), not a Gaussian sigma.',top=.82)
    save(out/'audit.json',{'complete':True,'learned_models':5,'evaluation_tracks_each':50000,'common_cvt_population':49803,
        'train_validation_evaluation_identities_match':True,'truth_checksums_match':True,'shared_initial_head_weights_match':True,
        'head_refinement_counts':{k:audits[k]['head_mamba_blocks'] for k in ORDER},
        'warning':'Only adapter-only + Mamba2 has refinement and greater head capacity; this is not a matched-head test of pretraining.'})
    (out/'README.md').write_text('# Momentum: five-model comparison\n\n'
        'Both AdapterOnly models and all three frozen-backbone models are shown together. '
        'Only the purple AdapterOnly model has one Mamba2 refinement layer; all other downstream heads have zero. '
        'These are single-seed results with shared cohorts and common starting head weights. The differing head capacities prevent an architecture-matched claim about pretraining.\n\n'
        '`validation_learning_curves` shows raw validation MAE versus optimizer step with linear-x and log-log axes. '
        '`momentum_quality`, `component_errors` and `momentum_binned` use the same 49,803 tracks with valid CVT reference vectors. '
        'Metrics on all 50,000 tracks are also saved separately. Every figure is exported to PNG and PDF; CSV tables and the audit accompany them.\n')
    with zipfile.ZipFile(root/'momentum_combined_plots.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,'momentum_combined/'+p.name)
    print(json.dumps({'output':str(out),'series':LABELS,'audited_models':5,'figures':4},indent=2))


if __name__=='__main__':main()
