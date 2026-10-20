"""Evaluation-only campaign 4 diagnostic. Historical implementation stays untouched."""
import argparse, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from mmap_ninja import RaggedMmap
from ruamel.yaml import YAML
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset, MyCollator, knn_later_indices_batch
from fm4npp.hilbert import assign_clas12_layer, clas12_band_hilbert_order
from fm4npp.utils import cartesian_to_polar_batched
from fm4npp.models.mambagpt import Mamba1GPT

CATS=['same_valid_track','different_valid_track','unknown_noise_ambiguous']
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(data,n,seed,out):
    readers={k:RaggedMmap(data/(k+'_test')) for k in ['features','seg_target','noise_target','cluster_reason_target','cluster_meta_target','coatjava_seg_pred']}
    # Bypass only full-dataset indexing scan; __getitem__ and transforms are original.
    ds=TPCBatchDataset.__new__(TPCBatchDataset)
    ds.memmap_feature=readers['features']; ds.memmap_seg_target=readers['seg_target']
    ds.train=False; ds.chunk_training=False; ds.normalize=True; ds.num_pred_points=1
    ds.eta_lim={'min':-2.5,'max':1.5};ds.phi_lim={'min':-torch.pi,'max':torch.pi};ds.r_lim={'min':6.,'max':23.}
    ds.band_classification=False;ds.data_scaler=1
    rng=np.random.default_rng(seed); chosen=[]; seen=set(); skipped={}; rawrows=[]; events=[]
    # Uniform random event indices, without replacement, rejection only for original size cut.
    while len(chosen)<n:
        i=int(rng.integers(len(readers['features'])))
        if i in seen:continue
        seen.add(i); xyz=np.copy(readers['features'][i]); size=len(xyz)
        if not 1<=size<=50:
            key='outside_original_1_50';skipped[key]=skipped.get(key,0)+1;continue
        if size==1:
            # Original squeeze() path cannot handle one-point events. No valid target exists.
            skipped['singleton_no_targets_loader_squeeze']=skipped.get('singleton_no_targets_loader_squeeze',0)+1;continue
        chosen.append(i)
        a={k:np.copy(v[i]) for k,v in readers.items()}
        assert all(len(v)==size for v in a.values())
        x=torch.from_numpy(xyz).unsqueeze(0)
        nf=ds.apply_norm(cartesian_to_polar_batched(x))
        assert torch.isfinite(nf).all()
        rperm=nf[...,-1].argsort(dim=1).squeeze(0); nf=nf[:,rperm]
        band=assign_clas12_layer(nf[...,2])[0]
        dist=torch.norm(nf.unsqueeze(2)-nf.unsqueeze(1),dim=-1)[0]
        eligible=band[None,:]>band[:,None]
        dv,ni=dist.masked_fill(~eligible,float('inf')).min(dim=1)
        valid=torch.isfinite(dv)
        independent=torch.full_like(nf[0],-100); independent[valid]=nf[0,ni[valid]]
        assert torch.equal(independent,knn_later_indices_batch(nf,1)[0])
        hp=clas12_band_hilbert_order(phi=nf[0,:,1],eta=nf[0,:,0],r=nf[0,:,2])
        ds.idxlist=[i]; original=ds[0]
        raw_order=rperm[hp].numpy(); target_raw=rperm[ni[hp]].numpy()
        assert torch.equal(original[0],nf[0,hp])
        assert torch.equal(original[1],torch.from_numpy(a['seg_target'][raw_order]))
        assert torch.equal(original[2],independent[hp])
        # Labels/reasons are annotations only; no reconstructed IDs participate.
        labels=a['seg_target']; reason=a['cluster_reason_target']; noise=a['noise_target']
        # This exporter uses event-local zero-based truth segment IDs: 0 is valid.
        # -1 and all other negative/sentinel or rejected/ambiguous rows are unknown.
        label_valid=(labels>=0)&(labels!=2147483647)&(reason==0)&(noise==0)
        inv=np.empty(size,int);inv[raw_order]=np.arange(size)
        for q,raw in enumerate(raw_order):
            if not valid[hp[q]]:continue
            tr=int(target_raw[q]); assert inv[tr]>q  # target strictly in causal future
            qv=bool(label_valid[raw]);tv=bool(label_valid[tr])
            cat=(0 if labels[raw]==labels[tr] else 1) if qv and tv else 2
            rawrows.append(dict(event=i,token=q,source_row=int(raw),target_row=tr,category=cat,
                source_truth=int(labels[raw]),target_truth=int(labels[tr]),source_reason=int(reason[raw]),target_reason=int(reason[tr]),
                layer=int(band[hp[q]]),target_layer=int(band[ni[hp[q]]]),occupancy=size,
                distance=float(dv[hp[q]]),raw_distance_cm=float(np.linalg.norm(xyz[raw]-xyz[tr]))))
        events.append(original)
    dump(out/'sample_selection.json',dict(seed=seed,n_events=n,raw_event_count=len(readers['features']),indices=chosen,skipped=skipped,
        split='test',dataset=str(data),criterion='uniform raw test event indices without replacement; original 1..50 cut; singleton has no valid targets',
        independence='Original mmap_cluster_full and split membership unavailable; no claim of campaign-training disjointness.'))
    dump(out/'token_annotations.json',rawrows)
    np.savez_compressed(out/'sample_inputs.npz',**{f'event_{i}':x[0].numpy() for i,x in zip(chosen,events)})
    return events,rawrows

def summarize(loss,rows,nbatches,event_ids,bootstrap=400):
    cat=np.array([r['category'] for r in rows]); eid=np.array([r['event'] for r in rows])
    # Recover original validator mean-of-batch means with local_valid_batch_size=2.
    emap={e:i for i,e in enumerate(event_ids)};bi=np.array([emap[e]//2 for e in eid]); cnt=np.bincount(bi,minlength=nbatches)
    assert (cnt>0).all()
    w=1/(cnt[bi]*nbatches)
    original=float(np.sum(loss*w)); pooled=float(loss.mean())
    out=dict(n_targets=len(loss),pooled_loss=pooled,original_batch2_loss=original,categories={})
    evcount=np.zeros((len(event_ids),3));evsum=evcount.copy()
    ei=np.array([emap[e] for e in eid]); np.add.at(evcount,(ei,cat),1);np.add.at(evsum,(ei,cat),loss)
    rng=np.random.default_rng(8722);boot=[]
    for _ in range(bootstrap):
        ids=rng.integers(len(event_ids),size=len(event_ids)); s=evsum[ids].sum(0);c=evcount[ids].sum(0)
        boot.append(np.r_[s/c,s.sum()/c.sum(),s[1]/c[1]-s[0]/c[0]])
    boot=np.array(boot)
    out['pooled_loss_ci95_event_bootstrap']=np.quantile(boot[:,3],[.025,.975]).tolist()
    out['different_minus_same_ci95_event_bootstrap']=np.quantile(boot[:,4],[.025,.975]).tolist()
    for c,name in enumerate(CATS):
        m=cat==c;v=loss[m]
        out['categories'][name]=dict(count=int(m.sum()),fraction=float(m.mean()),events=int(len(set(eid[m]))),
            conditional_mean=float(v.mean()) if len(v) else None,pooled_contribution=float(v.sum()/len(loss)),
            original_batch2_contribution=float((loss[m]*w[m]).sum()),
            original_effective_fraction=float(w[m].sum()),
            conditional_mean_ci95_event_bootstrap=np.quantile(boot[:,c],[.025,.975]).tolist(),
            quantiles=dict(zip(['q01','q10','q25','q50','q75','q90','q95','q99'],np.quantile(v,[.01,.1,.25,.5,.75,.9,.95,.99]).tolist())),
            distance_mean=float(np.mean([r['distance'] for r in np.array(rows,dtype=object)[m]])))
    out['partition_pooled_error']=sum(x['pooled_contribution'] for x in out['categories'].values())-pooled
    out['partition_original_batch2_error']=sum(x['original_batch2_contribution'] for x in out['categories'].values())-original
    assert abs(out['partition_pooled_error'])<1e-12 and abs(out['partition_original_batch2_error'])<1e-12
    strata=[]
    for field,edges in [('layer',None),('occupancy',[0,20,30,40,51]),('distance',[0,.02,.05,.1,.2,float('inf')])]:
        values=np.array([r[field] for r in rows]); groups=values if edges is None else np.digitize(values,edges)-1
        for g in np.unique(groups):
            for c,name in enumerate(CATS):
                m=(groups==g)&(cat==c)
                if m.any():strata.append(dict(field=field,bin=int(g),category=name,count=int(m.sum()),loss=float(loss[m].mean()),distance=float(np.mean([r['distance'] for r in np.array(rows,dtype=object)[m]]))))
    out['strata']=strata
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--events',type=int,default=128);ap.add_argument('--widths',type=int,nargs='+',default=[64]);ap.add_argument('--last',action='store_true');ap.add_argument('--seed',type=int,default=4204);ap.add_argument('--out',default='/out/pilot');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(4204)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=YAML(typ='safe').load(Path('campaign4_recovered.yaml'))
    events,rows=prepare(Path('/data'),a.events,a.seed,out);dump(out/'validation.json',dict(original_loader_equal=True,target_gather_equal=True,truth_reorder_equal=True,same_layer_excluded=True,target_strict_future=True,labels_not_model_inputs=True,n_events=len(events),n_targets=len(rows)))
    print('PREPARED',len(events),len(rows),flush=True)
    eid=list(dict.fromkeys(r['event'] for r in rows)); assert len(eid)==len(events)
    collate=MyCollator();outputs={}
    for width in a.widths:
      for kind in (['best','last'] if a.last else ['best']):
        start=time.time();cfg=config[f'scale_w{width}_d12_chinchilla']
        assert cfg['klen']==1 and not cfg['rep_aaai'] and not cfg['loss_reweight'] and not cfg['ablate_loss_scale'] and not cfg['chunk_training']
        path=Path(f'/campaign/scale_w{width}_d12_n39553933')/('ckpt_best.tar' if kind=='best' else 'sweep/training_checkpoints/ckpt.tar')
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        model=Mamba1GPT(embed_dim=width,num_layers=cfg['num_layers_backbone'],d_state=cfg['d_state'],klen=1,dropout=cfg['dropout'],embed_method=cfg['embed_method'],pe_method=cfg['pe_method'])
        model.load_state_dict(ckpt['model_state'],strict=True);step=ckpt['iters'];epoch=ckpt['epoch'];del ckpt
        model=model.cuda().eval(); assert not model.training
        all_loss=[];all_feature=[];direct_batches=[];padded_max=0.;causal_max=0.
        with torch.no_grad():
          for b in range(0,len(events),32):
            x,_,t=collate(events[b:b+32]);pred=model(x.cuda()).cpu();mask=t!=-100
            # Independent original validator expression, calculated on original 2-event batches.
            for j in range(0,len(x),2):
                m=mask[j:j+2];direct_batches.append(float(torch.nn.functional.mse_loss(pred[j:j+2][m],t[j:j+2][m])))
            err=(pred-t).square();valid=mask.all(-1)
            all_loss.extend(err[valid].mean(-1).double().tolist());all_feature.extend(err[valid].double().tolist())
            if b==0:
                # Causal padding invariance and no future-token leakage (no changes to objective).
                xx=events[0][0].unsqueeze(0).cuda();one=model(xx).cpu();padded_max=float((one[0]-pred[0,:xx.shape[1]]).abs().max())
                cut=max(1,xx.shape[1]//2);changed=xx.clone();changed[:,cut:]=torch.flip(changed[:,cut:],[1])+.013
                causal_max=float((model(changed)[:,:cut]-one[:,:cut].cuda()).abs().max())
                assert padded_max<3e-5 and causal_max<3e-5
        loss=np.array(all_loss); feature=np.array(all_feature);assert len(loss)==len(rows)
        s=summarize(loss,rows,len(direct_batches),eid)
        s.update(width=width,checkpoint=str(path),step=int(step),epoch=int(epoch),nominal_single_process_event_presentations=int(step*cfg['local_batch_size']),
            train_batch_size=cfg['local_batch_size'],checkpoint_sha256=sha(path),seconds=time.time()-start,
            direct_original_batch2_loss=float(np.mean(direct_batches)),padding_invariance_max_abs=padded_max,causality_max_abs=causal_max,
            feature_mse_mean=dict(zip(['normalized_eta','normalized_phi','normalized_radius'],feature.mean(0).tolist())))
        s['original_reduction_error']=s['original_batch2_loss']-s['direct_original_batch2_loss'];assert abs(s['original_reduction_error'])<1e-8
        for c,name in enumerate(CATS):
            m=np.array([r['category']==c for r in rows]);s['categories'][name]['feature_mse_mean']=dict(zip(['normalized_eta','normalized_phi','normalized_radius'],feature[m].mean(0).tolist()))
        name=f'w{width}_{kind}';np.savez_compressed(out/(name+'_losses.npz'),loss=loss,feature_loss=feature)
        outputs[name]=s;dump(out/(name+'.json'),s);print('RESULT',name,'step',step,'loss',s['pooled_loss'],'categories',[(k,v['count'],v['conditional_mean']) for k,v in s['categories'].items()],'seconds',s['seconds'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    dump(out/'results.json',outputs)

if __name__=='__main__':main()
