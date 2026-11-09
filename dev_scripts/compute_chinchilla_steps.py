"""
Compute Chinchilla-optimal (compute-optimal) training step budgets for each
width in the m1-m6 sweep, instead of the flat 100k steps every width shared.

Chinchilla ratio (Hoffmann et al. 2022): D_opt ~= 20 x N_params, where D is
training tokens and N is parameter count. In this sequence-modeling setting,
one "token" = one point/hit the backbone processes -- directly analogous to
a word token in an LLM, since the Mamba backbone consumes the point cloud as
a sequence. Converts that into a step count using the actual average points
per event (measured, not assumed) and the real batch size used.

Run (from /workspace/PP_collision):
    python3 dev_scripts/compute_chinchilla_steps.py
"""
import sys
sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT

# ============================================================
DEPTH = 12
BATCH_SIZE = 64          # the tuned k=1 recipe's local_batch_size
AVG_POINTS_PER_EVENT = 35.2   # measured mean, dataset_baseline_hits_per_event.py on mmap_cluster_full
CHINCHILLA_RATIO = 20    # tokens per parameter, Hoffmann et al. 2022
CURRENT_FLAT_STEPS = 100_000
WARMUP_FRACTION = 0.10   # keep warmup at ~10% of each width's own total_steps
WIDTHS = [64, 128, 256, 512, 1024, 1536]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}
# ============================================================


def param_count(embed_dim, depth):
    m = Mamba1GPT(embed_dim=embed_dim, num_layers=depth, d_state=16, klen=1,
                  dropout=0.1, embed_method='pos_only', pe_method='nerf')
    return sum(p.numel() for p in m.parameters())


tokens_per_step = BATCH_SIZE * AVG_POINTS_PER_EVENT

print(f"Settings: batch_size={BATCH_SIZE}, avg_points/event={AVG_POINTS_PER_EVENT}, "
      f"tokens/step={tokens_per_step:.0f}, chinchilla_ratio={CHINCHILLA_RATIO}")
print()
print(f"{'width':>6} {'m':>3} {'params':>14} {'D_opt (tokens)':>16} {'steps_needed':>14} "
      f"{'current_steps':>14} {'current/optimal':>16}  status")
print("-" * 100)

rows = []
for w in WIDTHS:
    n_params = param_count(w, DEPTH)
    d_opt = CHINCHILLA_RATIO * n_params
    steps_needed = d_opt / tokens_per_step
    ratio = CURRENT_FLAT_STEPS / steps_needed
    status = "UNDERTRAINED" if ratio < 0.8 else ("over-provisioned" if ratio > 1.5 else "roughly adequate")
    rows.append(dict(width=w, m=M_LABEL[w], n_params=n_params, d_opt=d_opt,
                     steps_needed=steps_needed, ratio=ratio, status=status))
    print(f"{w:>6} {M_LABEL[w]:>3} {n_params:>14,} {d_opt:>16,.0f} {steps_needed:>14,.0f} "
          f"{CURRENT_FLAT_STEPS:>14,} {ratio:>16.2f}  {status}")

print()
print("=" * 100)
print("RECOMMENDED per-width step budgets (rounded to nearest 1000, min 5000):")
print("=" * 100)
for r in rows:
    steps = max(5000, round(r["steps_needed"] / 1000) * 1000)
    warmup = max(100, round(steps * WARMUP_FRACTION / 100) * 100)
    est_hours_at_w256_rate = None  # left as a placeholder; real per-width step time varies
    print(f"  {r['m']} (w{r['width']:>4}): total_steps={steps:>7,}  warmup_steps={warmup:>6,}")

print()
print("NOTE: this does not account for actual wall-clock time per step, which")
print("grows with width (e.g. w1536 measured earlier at ~0.47s/step vs w256's")
print("~0.11s/step). Recommended step counts should be paired with per-width")
print("timing to estimate real sweep duration before committing to a rerun.")