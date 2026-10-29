"""Pretraining-only pair comparison and five-model comparison."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from mmap_ninja import RaggedMmap

from dev_scripts.backbone_pretraining_comparison.compare import curve_from_log
from dev_scripts.backbone_pretraining_comparison.tokens import replay

HERE=Path(__file__).resolve().parent
LABELS={'mamba1':'Mamba 1','mamba2':'Mamba 2','mamba3':'Mamba 3',
        'transformer_nope':'Transformer NoPE','transformer_rope':'Transformer RoPE'}
COLORS={'mamba1':'#3974b2','mamba2':'#dc922d','mamba3':'#3a956b',
        'transformer_nope':'#9560b3','transformer_rope':'#d84e67'}


def attention_training_flops(padded_tokens, width=512, layers=12, batch=64):
    # QK and AV: 4*B*L^2*D forward, then x3 for forward+backward.
    # Charge full dense products, including masked causal entries. This is an
    # algorithmic convention, not an assertion about the selected SDPA kernel.
    return 12 * layers * width * (np.asarray(padded_tokens,dtype=np.int64)**2 // batch)


def render(models, output, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,suffix,label,divisor in [('step','steps','Optimizer updates',1),
         ('valid_tokens','tokens','Valid input tokens presented',1e6),
         ('estimated_training_flops','compute','Estimated training arithmetic (PFLOP)',1e15),
         ('dense_training_flops','dense_compute','Projection-only training arithmetic (PFLOP)',1e15)]:
        fig,ax=plt.subplots(figsize=(8.8,5.2),layout='constrained')
        for m in models:
            curve=m['curve']; version=m['version']
            x=[p[key]/divisor for p in curve]; y=[p['validation_mse'] for p in curve]
            ax.plot(x,y,label=LABELS[version],color=COLORS[version],lw=1.6)
            best=int(np.argmin(y)); ax.scatter(x[best],y[best],marker='*',s=85,color=COLORS[version],zorder=3)
        if key=='valid_tokens': label+=' (millions)'
        ax.set(xlabel=label,ylabel='Raw checkpoint validation MSE',yscale='log',
               title='Matched pretraining: '+('NoPE versus RoPE' if len(models)==2 else 'Transformers versus Mamba'))
        ax.grid(alpha=.2,which='both'); ax.legend(frameon=False)
        fig.text(.5,-.015,'Seed 42; same training exposure. Validation selects checkpoints. Precision differs across Mambas.',
                 ha='center',fontsize=8)
        for ext in ['png','pdf']: fig.savefig(output/f'{stem}_{suffix}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True)
    args=parser.parse_args(); root=Path(args.root); out=root/'comparison'; out.mkdir(exist_ok=True)
    reference=json.loads((HERE/'mamba_reference.json').read_text())
    models=reference['models']
    for m in models:
        assert m['completed']['final_step']==200000
    recipe=__import__('ruamel.yaml',fromlist=['YAML']).YAML(typ='safe').load((HERE/'recipe.yaml').read_text())
    data=Path(next(iter(recipe.values()))['data_root'])
    features=RaggedMmap(str(data/'features_pretrain'))
    lengths=(np.asarray(features.ends)-np.asarray(features.starts))//3
    selected=np.flatnonzero((lengths>=1)&(lengths<=50))[:10000000]
    buf=io.BytesIO(); np.save(buf,selected)
    assert hashlib.sha256(buf.getvalue()).hexdigest()==models[0]['cohorts']['training']['indices_sha256']
    padded,valid=replay(lengths[selected],64,200000)
    cpad,cvalid,cattention=np.cumsum(padded),np.cumsum(valid),np.cumsum(attention_training_flops(padded))
    assert int(cpad[-1])==reference['total_padded_training_tokens']
    assert int(cvalid[-1])==reference['total_valid_training_tokens']
    transformers=[]
    for encoding in ['nope','rope']:
        arm=root/encoding
        result=json.loads((arm/'completed.json').read_text())
        assert result['final_step']==result['finite_gradient_updates']==200000
        cohorts=json.loads((arm/'cohorts.json').read_text())
        for split in ['training','validation']:
            assert cohorts[split]['indices_sha256']==models[0]['cohorts'][split]['indices_sha256']
        curve=curve_from_log(arm/f'training/transformer_{encoding}_w512_d12_k1/seed42/train.log')
        assert [p['step'] for p in curve]==list(range(1000,200001,1000))
        dense=result['compute']['dense_training_flops_per_padded_token']
        for p in curve:
            i=p['step']-1
            p.update(padded_tokens=int(cpad[i]),valid_tokens=int(cvalid[i]),
                     dense_training_flops=int(cpad[i])*dense,
                     estimated_training_flops=int(cpad[i])*dense+int(cattention[i]))
        best=min(curve,key=lambda p:p['validation_mse'])
        m={'version':'transformer_'+encoding,'complete':True,'completed':result,'cohorts':cohorts,'curve':curve,
           'summary':{'parameters':result['parameter_count'],'best_validation_mse':best['validation_mse'],
            'best_step':best['step'],'final_validation_mse':curve[-1]['validation_mse'],
            'estimated_training_pflop':curve[-1]['estimated_training_flops']/1e15,
            'dense_training_pflop':curve[-1]['dense_training_flops']/1e15,
            'recorded_elapsed_seconds':result['elapsed_seconds']}}
        transformers.append(m)
    for key in ['initial_state_sha256','embedding_sha256','sampler_generator_sha256']:
        assert transformers[0]['completed'][key]==transformers[1]['completed'][key]
    models+=transformers
    models.sort(key=lambda m:m['version'])
    budget=min(m['curve'][-1]['estimated_training_flops'] for m in models)
    common={m['version']:max((p for p in m['curve'] if p['estimated_training_flops']<=budget),key=lambda p:p['step']) for m in models}
    limitations=reference['limitations']+[
      'Transformer pair uses identical initial tensors, FP32 and FFN width544; total parameters19.34M.',
      'Mamba1/2 FP32; Mamba3 local BF16. Recorded end-to-end times span hosts and sessions.',
      'Transformer estimate adds full dense QK/AV arithmetic (including masked entries), x3 for forward/backward.',
      'RoPE rotations, attention softmax, norms, nonlinearities, optimizer and kernel-specific work are excluded.',
      'Common-compute comparison uses the last recorded checkpoint within the budget, without interpolation or schedule retuning.']
    summary={'complete':True,'scope':'pretraining only','models':models,'common_compute_budget_flops':budget,
             'at_common_compute_budget':common,'total_valid_training_tokens':int(cvalid[-1]),
             'total_padded_training_tokens':int(cpad[-1]),'limitations':limitations}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    compact={m['version']:m['summary'] for m in models}
    (out/'metrics.json').write_text(json.dumps({'models':compact,'common_compute':common,'limitations':limitations},indent=2)+'\n')
    rows=[{'model':m['version'],**p} for m in models for p in m['curve']]
    with (out/'learning_curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    render(transformers,out,'transformer_pair'); render(models,out,'all_backbones')
    lines=['# Matched pretraining comparison','',
      '| Model | Parameters | Best validation MSE | Best step | Final MSE | Estimated PFLOP |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in models:
        s=m['summary']; lines.append(f"| {LABELS[m['version']]} | {s['parameters']:,} | {s['best_validation_mse']:.8f} | {s['best_step']} | {s['final_validation_mse']:.8f} | {s['estimated_training_pflop']:.3f} |")
    n,r=[m['summary'] for m in transformers]
    lines+=['',f"RoPE changes best validation MSE by {(r['best_validation_mse']/n['best_validation_mse']-1)*100:+.2f}% relative to NoPE.",
            '', '## Last recorded checkpoint within the common compute budget','',
            '| Model | Update | Validation MSE |','| --- | ---: | ---: |']
    for m in models:
        p=common[m['version']]; lines.append(f"| {LABELS[m['version']]} | {p['step']} | {p['validation_mse']:.8f} |")
    lines+=['',*['- '+x for x in limitations]]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(compact),flush=True)


if __name__=='__main__': main()
