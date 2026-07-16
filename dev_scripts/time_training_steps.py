"""
Time actual training steps using the REAL get_data_loader pipeline (MyCollator,
local_batch_size, 8 workers, prefetch) so the timing reflects production exactly.
Loads a real config from the generated yaml.

Run inside container:  python3 dev_scripts/time_training_steps.py [config_name]
If no config given, uses the first scale_* config in _grid_generated.yaml.
"""
import sys, time, glob, re
sys.path.insert(0, "/workspace/PP_collision")
import torch
from fm4npp.utils import YParams
from fm4npp.datasets.dataset_pretrain import get_data_loader
from fm4npp.models.mambagpt import Mamba1GPT

YAML = "/workspace/PP_collision/scripts/configs/_grid_generated.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pick a config
if len(sys.argv) > 1:
    cfg_name = sys.argv[1]
else:
    txt = open(YAML).read()
    m = re.search(r'^(scale_w\d+_d\d+_n\d+):', txt, re.M)
    cfg_name = m.group(1) if m else None
    if cfg_name is None:
        raise SystemExit("No scale_* config found in generated yaml; pass one explicitly.")
print(f"Using config: {cfg_name}")

params = YParams(YAML, cfg_name)
print(f"local_batch_size (per step): {params.local_batch_size}")
print(f"num_data_workers: {params.num_data_workers}")

train_loader, _, _, _ = get_data_loader(params, distributed=False)

model = Mamba1GPT(embed_dim=params.embed_dim, num_layers=params.num_layers_backbone,
                  d_state=params.d_state, klen=params.klen, dropout=params.dropout,
                  embed_method=params.embed_method, pe_method=params.pe_method,
                  band_classification=getattr(params,'band_classification',False),
                  n_bands=getattr(params,'n_bands',6)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()

print("\nTiming steps (first 5 are warmup: worker spin-up)...")
it = iter(train_loader)
data_t, compute_t = 0.0, 0.0
NSTEP = 40
t_prev = time.time()
for step in range(NSTEP):
    t0 = time.time()
    batch = next(it)
    grouped = batch[0]; knn = batch[2]
    t_data = time.time() - t0
    grouped = grouped.to(device); b = grouped.size(0)
    klabel = knn.reshape(b, -1, params.klen*3).to(device)
    t1 = time.time()
    pred = model(grouped)
    # crude loss just to exercise backward; not the real loss path
    n = min(pred.shape[-1], klabel.shape[-1])
    loss = torch.nn.functional.mse_loss(pred[..., :n].float(), klabel[..., :n].float())
    loss.backward(); opt.step(); opt.zero_grad()
    if device.type == "cuda": torch.cuda.synchronize()
    t_compute = time.time() - t1
    total = time.time() - t_prev; t_prev = time.time()
    if step >= 5:
        data_t += t_data; compute_t += t_compute
    print(f"  step {step:2d}: total {total*1000:7.1f}ms  (data {t_data*1000:7.1f}ms, compute {t_compute*1000:6.1f}ms)")

n = NSTEP - 5
avg_step = (data_t + compute_t) / n
print(f"\nSteady-state: data {1000*data_t/n:.1f}ms/step, compute {1000*compute_t/n:.1f}ms/step, total {1000*avg_step:.1f}ms/step")
print(f"Projected per run (100k steps): {avg_step*100000/3600:.2f} hours")
print(f"Projected for 36 runs: {avg_step*100000*36/3600:.1f} hours = {avg_step*100000*36/86400:.1f} days")