import gc, hashlib, json, re
from pathlib import Path
import numpy as np
import torch
from mmap_ninja import RaggedMmap

OUT=Path('/out'); ROOT=Path('/campaign')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
rows=[]
for d in sorted(ROOT.glob('scale_*'),key=lambda x:int(re.search(r'w(\d+)',x.name)[1])):
    for p in [d/'ckpt_best.tar',d/'sweep/training_checkpoints/ckpt_best.tar',d/'sweep/training_checkpoints/ckpt.tar']:
        c=torch.load(p,map_location='cpu',weights_only=False)
        s=c['model_state']; width=s['output_layer.weight'].shape[1]
        row=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),step=c['iters'],epoch=c['epoch'],width=width,
                 shapes={k:list(v.shape) for k,v in s.items()},optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in c['optimizer_state_dict']['param_groups']])
        rows.append(row)
        print(str(p),row['step'],row['epoch'],row['sha256'],flush=True)
        del c,s;gc.collect()
OUT.joinpath('checkpoint_inventory.json').write_text(json.dumps(rows,indent=2))
data=Path('/data'); readers={k:RaggedMmap(data/(k+'_test')) for k in ['features','seg_target','noise_target','cluster_reason_target','cluster_meta_target','coatjava_seg_pred']}
print('reader',readers['features'].__dict__.keys(),flush=True)
for i in [0,1,10,100,1000]:
    print('event',i,{k:np.asarray(v[i]).tolist()[:6] for k,v in readers.items()},flush=True)
