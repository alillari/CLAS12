"""Select query and assignment cutoffs on validation, then evaluate the frozen pair on test."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from fm4npp.datasets.dataset import get_val_loader
from fm4npp.pretrained import configure_pretrained_backbone
from fm4npp.utils import YParams
from train.downstream.eval.evaluate_track_finding import load_checkpoint
from train.downstream.loss import assign_points_to_masks
from train.downstream.track_finding_metrics import MCReconstructionAccumulator, MatchConfig, event_mc_reconstruction_metrics
from train.downstream.track_finding_trainer import DownstreamTrainer

OUT = Path(os.environ.get('M4_FINDING_SCORE', '/results'))
FIT = Path(os.environ.get('M4_FINDING_FIT', '/fit/finding'))
REF = (.5, .2)
MATCH = MatchConfig(background_label=-1, iou_threshold=.5, min_purity=.5, min_efficiency=.5)
FIELDS = ('mc_track_dice_sum','n_true_tracks','n_pred_tracks','n_matched_tracks','n_signal_points',
          'n_mc_associated_pred_points','n_mc_correct_points')


def write(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False, indent=2) + '\n')


@dataclass
class Event:
    key: tuple
    truth: np.ndarray
    codes: np.ndarray
    nt: int
    confidence: np.ndarray
    scores: np.ndarray


def winners(e, q):
    eligible = np.flatnonzero(e.confidence > q)
    if len(eligible) == 0:
        return np.full(len(e.truth), -1, dtype=int), np.full(len(e.truth), -1., dtype='f4')
    chosen = eligible[np.argmax(e.scores[:,eligible], axis=1)]
    return chosen, e.scores[np.arange(len(chosen)), chosen]


def final(e, q, a):
    chosen, scores = winners(e,q)
    return np.where(scores > a, chosen, -1)


def dice_sum(e, pred):
    if not e.nt:
        return 0.
    nq = len(e.confidence)
    groups = np.where(pred >= 0, pred, nq) * (e.nt+1) + e.codes
    counts = np.bincount(groups, minlength=(nq+1)*(e.nt+1)).reshape(nq+1,e.nt+1)
    hits = counts[:-1,:-1]
    active = hits.sum(axis=1)>0
    if not active.any():
        return 0.
    dice = 2*hits[active] / (counts[:-1].sum(axis=1)[active,None] + counts[:,:-1].sum(axis=0))
    r,c = linear_sum_assignment(-dice)
    return float(dice[r,c].sum())


def scan(events, q_grid, a_grid, rows):
    total_truth = sum(e.nt for e in events)
    for q in q_grid:
        thresholds = np.array([a for a in a_grid if (q,a) not in rows])
        if not len(thresholds):
            continue
        sums = np.zeros(len(thresholds))
        for e in events:
            ids,scores = winners(e,q)
            # Equal surviving sets yield exactly the same assignment and Dice.
            n_rejected = np.searchsorted(np.sort(scores),thresholds.astype(scores.dtype),side='right')
            _, first, bins = np.unique(n_rejected,return_index=True,return_inverse=True)
            values = [dice_sum(e,np.where(scores>thresholds[t],ids,-1)) for t in first]
            sums += np.array(values)[bins]
        rows.update({(float(q),float(a)):float(v/total_truth) for a,v in zip(thresholds,sums)})


def grid(center, width, step):
    n=round(width/step)
    return sorted({round(float(np.clip(center+i*step,0,1)),6) for i in range(-n,n+1)})


def best(rows):
    return min(rows,key=lambda x:(-rows[x],(x[0]-REF[0])**2+(x[1]-REF[1])**2,x))


@torch.inference_mode()
def events(trainer, loader):
    dataset = loader.dataset
    for batch in loader:
        x, labels, _, _ = trainer._unpack_batch(batch)
        x = x.to(trainer.device)
        valid = x[...,0] != -100
        inp = trainer._representation_input(x,batch)
        frozen = bool(trainer.params.pretrained_ckpt)
        features = None
        if frozen:
            _, embeddings, _ = trainer.model(inp, return_z=True)
            features = torch.stack(embeddings)
        out = trainer.down_model(inp,feature=features,pretrain=frozen,padding_mask=valid)
        cls,mask = out['class_probs'],out['mask_probs']
        native = assign_points_to_masks(dict(pred_probs=cls,pred_masks=mask.permute(0,2,1)),option=2,threshold=REF[1])
        assert cls.shape[-1] == 2
        scores=(mask*cls[:,None,:,1]).cpu().numpy()
        confidences=cls[:,:,1].cpu().numpy()
        assert np.isfinite(scores).all() and np.isfinite(confidences).all()
        for i in range(len(x)):
            key=tuple(int(batch[k][i]) for k in ('source_id','original_ordinal','source_event_index'))
            raw=key[2]
            xyz=torch.from_numpy(dataset.memmap_feature[raw]).float().unsqueeze(0)
            tokens=dataset.apply_norm(dataset._to_model_features(xyz))
            radial=tokens[0,:,-1].argsort()
            ids=radial[dataset._compute_sorter(tokens[:,radial])].numpy()
            order=np.argsort(ids)
            keep=valid[i].cpu().numpy()
            lab=labels[i].cpu().numpy()[keep][order]
            np.testing.assert_array_equal(lab,dataset.memmap_seg_target[raw])
            tids=np.unique(lab[lab>=0])
            codes=np.where(lab>=0,np.searchsorted(tids,lab),len(tids))
            e=Event(key,lab,codes,len(tids),confidences[i],scores[i][keep][order])
            ref=native['assignments'][i].cpu().numpy()[keep][order]
            np.testing.assert_array_equal(final(e,*REF),ref)
            yield e


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    params=YParams(str(FIT/'model.yaml'),'finding')
    receipt=json.loads((FIT/'training_run.json').read_text())
    assert receipt['final_step']==30000 and receipt['selection_metric']=='mc_track_dice_mean'
    frozen = bool(receipt['usepretrain'])
    assert frozen == bool(params.pretrained_ckpt)
    ckpt=FIT/'checkpoints'/Path(receipt['checkpoint']).name
    # Reuse the fitted head and its training model config in the standard held-out evaluator.
    # Validation choices below consume only the validation loader.
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    model = yaml.load(FIT/'model.yaml')['finding']
    model.update(checkpoint_dir=str(OUT/'checkpoints'), downstream_dir=str(OUT/'runtime'))
    model_file = OUT/'model.yaml'
    yaml.dump({'finding': model}, model_file)
    analysis = yaml.load(Path(__file__).parent/'recipes/finding_analysis.yaml')['analysis']
    analysis.update(model_config='finding', model_yaml=str(model_file), pretrained_checkpoint=params.pretrained_ckpt,
                    checkpoint=str(ckpt), training_log=str(FIT/'checkpoints/training.log'), output_dir=str(OUT/'production'),
                    evaluation_split='test', use_pretrained_backbone=frozen,
                    run_name=params.pretrained_backbone['name'] if frozen else 'adapter_only', run_num='42')
    yaml.dump({'analysis': analysis}, OUT/'analysis.yaml')
    params.update_params(dict(limit_size=1,limit_test_size=5000,limit_test_data=True,
        return_dict=True,return_reg_test=True,validation_split='validation',drop_last_test=False,
        checkpoint_dir=str(OUT/'checkpoints'),downstream_dir=str(OUT/'runtime'),log_file_name='evaluate.log',
        return_coatjava_seg_pred_test=False,require_coatjava_seg_pred_test=False))
    configure_pretrained_backbone(params,enabled=frozen,checkpoint=params.pretrained_ckpt)
    trainer=DownstreamTrainer(params,SimpleNamespace(root_dir=str(OUT/'runtime'),global_log_dir='logs',config='finding',run_num='42'))
    try:
        trainer.launch()
        load_checkpoint(trainer,ckpt,frozen)
        validation=list(events(trainer,trainer.val_data_loader))
        val_ids={e.key[:2] for e in validation}
        assert trainer.val_data_loader.dataset.split=='validation' and len(validation)==len(val_ids)==5000
        rows={}
        scan(validation,sorted(set(grid(.5,.5,.1)+[.95,.975,.99])),grid(.5,.5,.05),rows)
        for qw,qs,aw,ats in ((.1,.02,.05,.01),(.02,.005,.01,.0025)):
            q,a=best(rows)
            scan(validation,grid(q,qw,qs),grid(a,aw,ats),rows)
        chosen=best(rows)
        # Verify the search metric against the independent full scorer for both operating points.
        for op in {chosen,REF}:
            full=MCReconstructionAccumulator()
            for e in validation:
                full.add(event_mc_reconstruction_metrics(e.truth,final(e,*op),config=MATCH))
            np.testing.assert_allclose(full.summary()['mc_track_dice_mean'],rows[op],rtol=0,atol=1e-10)
        if abs(rows[REF]-receipt['best_selection_score'])>1e-5:
            raise RuntimeError(f"Fixed-point validation replay differs from selected training checkpoint: {rows[REF]} vs {receipt['best_selection_score']}")
        selected=dict(validation_events=len(val_ids),checkpoint_step=receipt['best_step'],
            adapter_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),backbone_sha256=params.pretrained_backbone['sha256'] if frozen else None,
            objective='mean track Dice',tie_break='closest to (.5,.2), then lower query and assignment cutoffs',
            selection=dict(query_confidence=chosen[0],assignment=chosen[1],validation_dice=rows[chosen]),
            historical=dict(query_confidence=REF[0],assignment=REF[1],validation_dice=rows[REF]),
            grid=[dict(query_confidence=q,assignment=a,validation_dice=v) for (q,a),v in sorted(rows.items())])
        selection_file=OUT/'selection.json'
        write(selection_file,selected)
        digest=hashlib.sha256(selection_file.read_bytes()).hexdigest()
        print('Frozen validation selection',selected['selection'],digest,flush=True)
        del validation

        params.update_params(dict(validation_split='test',limit_test_size=10000))
        loader=get_val_loader(params,False)
        assert loader.dataset.split=='test'
        sums={name:MCReconstructionAccumulator() for name in ('fixed','tuned')}
        test_rows=defaultdict(list)
        test_ids=set()
        for e in events(trainer,loader):
            assert e.key[:2] not in val_ids and e.key[:2] not in test_ids
            test_ids.add(e.key[:2])
            for name,op in [('fixed',REF),('tuned',chosen)]:
                scored=event_mc_reconstruction_metrics(e.truth,final(e,*op),config=MATCH)
                sums[name].add(scored)
                record=dict(method='adapter',source_id=e.key[0],original_ordinal=e.key[1],source_event_index=e.key[2])
                record.update({k:scored[k] for k in FIELDS})
                test_rows[name].append(record)
        assert len(test_ids)==10000 and hashlib.sha256(selection_file.read_bytes()).hexdigest()==digest
        native_csv=FIT/'evaluation/per_event_metrics.csv'
        if not native_csv.is_file():
            subprocess.run([sys.executable, '-u', 'train/downstream/eval/evaluate_track_finding.py',
                            '--analysis-config', str(OUT/'analysis.yaml')], check=True)
            native_csv=OUT/'production/per_event_metrics.csv'
        with native_csv.open() as stream:
            old=[(int(r['source_id']),int(r['original_ordinal']),int(r['n_matched_tracks']))
                 for r in csv.DictReader(stream) if r['method']=='adapter']
        assert [(r['source_id'],r['original_ordinal'],r['n_matched_tracks']) for r in test_rows['fixed']]==old
        for name,acc in sums.items():
            path=OUT/name
            path.mkdir()
            rows=test_rows[name]
            with (path/'per_event_metrics.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write(path/'summary.json',acc.summary())
        write(OUT/'receipt.json',dict(selection_sha256=digest,validation_events=len(val_ids),
             test_events=len(test_ids),disjoint=True,fixed_matches_replayed=True))
    finally:
        trainer.cleanup()


if __name__=='__main__':
    main()
