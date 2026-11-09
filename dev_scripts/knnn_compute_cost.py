"""
Compute cost of kNNN, measured on the code the trainer actually runs.

kNNN touches two places:
  1. CPU, data loader: knn_later_indices_batch (fm4npp/datasets/dataset_pretrain.py)
     builds the k target slots for every event. Timed per event on real
     background-merged events, single-threaded like a DataLoader worker.
  2. GPU, training step: the backbone is the same for every k; only the output
     layer grows (klen * 4 outputs with sin/cos). Timed as forward + backward +
     AdamW step for the m3 Mamba-3 model B config.

Then: events/s from the loader (x num_data_workers) vs events/s from the GPU,
whichever is slower sets the step time, and the hours for a 100k-step run.

Run from the repo root on the mamba3-sincos branch:
  python3 dev_scripts/knnn_compute_cost.py data/merged_recon.hipo
Options: [n_events for the loader timing, default 500] [batch, default 64] [workers, default 8]
"""
import sys, time
import numpy as np
import torch
import hipopy.hipopy as hp

sys.path.insert(0, ".")
from fm4npp.datasets.dataset_pretrain import knn_later_indices_batch
from fm4npp.models.mamba_config import build_backbone

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/PP_collision/data/merged_recon.hipo"
N_EVENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 64
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
KS = list(range(1, 12))
STEPS = 100_000
ETA_LIM, R_LIM = (-2.5, 1.5), (6.0, 23.0)


# ---------- real events: normalized (eta, phi, r), same convention as the loader ----------
def load_events(path, n):
    f = hp.open(path, mode="r")
    for b in ("BST::Clusters", "BMT::Clusters"):
        f.readBank(b)
    out = []
    for i, ev in enumerate(f):
        if len(out) >= n:
            break
        xyz = []
        for d in ("BST", "BMT"):
            c = [np.asarray(ev.get(f"{d}::Clusters_{k}", [])).ravel() for k in ("cx", "cy", "cz")]
            if len(c[0]):
                xyz.append(np.stack(c, -1))
        if not xyz:
            continue
        x, y, z = np.concatenate(xyz).T
        rho = np.hypot(x, y)
        eta = np.arcsinh(z / np.where(rho > 0, rho, 1e-9))
        phi = np.arctan2(y, x)
        f3 = np.stack([(eta - ETA_LIM[0]) / (ETA_LIM[1] - ETA_LIM[0]),
                       (phi + np.pi) / (2 * np.pi),
                       (rho - R_LIM[0]) / (R_LIM[1] - R_LIM[0])], -1)
        out.append(torch.tensor(f3, dtype=torch.float32).unsqueeze(0))  # (1, N, 3)
    return out


events = load_events(PATH, N_EVENTS)
lens = np.array([e.shape[1] for e in events])
print(f"{len(events)} events, clusters/event mean {lens.mean():.0f}, max {lens.max()}")

# ---------- 1. CPU: kNNN target construction ----------
torch.set_num_threads(1)  # DataLoader workers run single-threaded
aux = [torch.rand(1, e.shape[1], 4) for e in events]
cpu_ms = {}
for k in KS:
    for e, a in zip(events[:20], aux[:20]):  # warmup
        knn_later_indices_batch(e, k=k, aux_A=a)
    t0 = time.perf_counter()
    for e, a in zip(events, aux):
        knn_later_indices_batch(e, k=k, aux_A=a)
    cpu_ms[k] = 1000 * (time.perf_counter() - t0) / len(events)
torch.set_num_threads(max(1, torch.get_num_threads()))

# ---------- 2. GPU: m3 Mamba-3 training step ----------
dev = "cuda"
L = int(np.percentile(lens, 99))  # batches are padded to their longest event
gpu_ms = {}
for k in KS:
    params = dict(backbone_version="mamba3", embed_dim=256, num_layers_backbone=12, klen=k,
                  dropout=0.0, embed_method="pos_plus_aux", pe_method="nerf", pos_dim=3, aux_dim=4,
                  d_state=128, expand=2, headdim=64, use_sincos_target=True)
    model = build_backbone(params).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.rand(BATCH, L, 7, device=dev)
    for i in range(25):
        if i == 5:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x)
        out.float().pow(2).mean().backward()
        opt.step()
    torch.cuda.synchronize()
    gpu_ms[k] = 1000 * (time.perf_counter() - t0) / 20
    del model, opt, x
    torch.cuda.empty_cache()

# ---------- table ----------
print(f"\nm3 Mamba-3, batch {BATCH} events padded to {L} clusters, {WORKERS} loader workers, {STEPS:,} steps\n")
print(f"{'kNNN':>4} {'loader ms/event':>16} {'loader events/s':>16} {'GPU ms/step':>12} "
      f"{'GPU events/s':>13} {'bottleneck':>11} {'hours/100k steps':>17}")
for k in KS:
    load_eps = WORKERS * 1000 / cpu_ms[k]
    gpu_eps = BATCH * 1000 / gpu_ms[k]
    eps = min(load_eps, gpu_eps)
    hours = STEPS * BATCH / eps / 3600
    print(f"{k:>4} {cpu_ms[k]:>16.2f} {load_eps:>16,.0f} {gpu_ms[k]:>12.1f} {gpu_eps:>13,.0f} "
          f"{'loader' if load_eps < gpu_eps else 'GPU':>11} {hours:>17.1f}")