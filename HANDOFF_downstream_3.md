# CLAS12 Refit — Handoff Notes (Pretraining → Downstream)

I adapted **pretraining** from TPC to CLAS12 CVT. Below: what changed, the
shared-code items that affect you, and the data contract.

## Core differences from TPC
- **No energy** — hits are pure `(x,y,z)`, 3 columns not 4.
- **6 fixed layer radii** — `6.52944, 9.28923, 12.03261` (BST), `14.76460,
  19.26460, 22.26460` (BMT), ±0.3. Discrete, not continuous.
- **Few hits/event** — ~17 typical (3 particles × ~6 layers), range ~5–40.

## Files I changed
- `fm4npp/datasets/dataset_pretrain.py` — rewritten for CLAS12.
- `fm4npp/hilbert.py` — added `clas12_band_hilbert_order` (new ordering).
- `fm4npp/models/embed.py` — fixed `EmbedderPosOnly` (shared, see below).
- `fm4npp/models/mambagpt.py` — added `embed_method='pos_only'` (shared).
- `train/pretrain/nppmamba/train_multi_gpu_mamba1.py` — fixed 4-col reshape.
- `scripts/configs/mamba_pretrain.yaml` — added `clas12_pretrain` block.

## Shared-code items that affect downstream
These live in files you also use — you'll hit the same issues.

- **`embed.py` / `EmbedderPosOnly`** — fixed two bugs: indexing assumed a leading
  energy column (`neighborhood[..., 1:4]` → `neighborhood`), and it returned one
  value where `Mamba1GPT.forward` expects a 2-tuple (`return out, out`).
  `EmbedderAdd`/`EmbedderConcat` still strip energy and will break on 3-col input
  — use `pos_only` for CLAS12.
- **`mambagpt.py`** — added `embed_method='pos_only'` branch to both `MambaGPT`
  and `Mamba1GPT` (originally only `concat`/`add`). Required for no-energy models.
- **Trainer reshape** — `reshape(b, -1, 4)[:, :, 1:]` (hardcoded 4 cols, strips
  energy) appears twice in `train_multi_gpu_mamba1.py`; fixed to
  `reshape(b, -1, 3)`. **Your downstream trainer likely has the same line.**
- **Missing yaml attrs** — trainer references `rep_aaai`, `nexttoken`,
  `ablate_loss_scale`, absent from the original default block. Add them (`false`)
  or it `AttributeError`s.
- **Pickle loads** — trainer unconditionally loads `loss_bin_pp.pkl` /
  `loss_weight_pp.pkl` from `stat_dir` (only used if `loss_reweight=true`, which
  is false). Needs dummy files to exist.
- **`dataset.py` / `dataset_eval.py`** — structurally identical to
  `dataset_pretrain.py`; the CLAS12 changes (3-col, norm constants, ordering,
  filter) likely need mirroring there.

## Key dataset_pretrain.py changes (for mirroring)
- 3-column `[eta, phi, r]` throughout; `cartesian_to_polar_batched(features)` on
  all 3 cols (was `[..., 1:]`); `E_mean`/`E_std` deleted.
- Norm constants: `eta_lim={-2.5,1.5}`, `phi_lim={-π,π}`, `r_lim={6.0,23.0}`.
- `Voxelizer`/HRS and `rescale_serialize_Rlast` removed; ordering via
  `clas12_band_hilbert_order`.
- `klen`/`num_pred_points` reduced (only resizes final output layer).
- Filter `low_thr=5, high_thr=40` (was 50/3200). **`low_thr` is also used in a
  `torch.randint(0, N - low_thr + 1, ...)` — crashes if any sequence is shorter
  than `low_thr`, so keep `low_thr` ≤ shortest allowed length.**
- Deleted confirmed-dead funcs: `strip_masked`, `serialize_neighbors`, `Group`,
  module-level `apply_norm`, `group_points`, `set_simpler`, both
  `rescale_serialize_Rlast` defs.

## Ordering scheme (`clas12_band_hilbert_order`, in `hilbert.py`)
Snaps `r` to one of 6 fixed bands (asserts on out-of-band) → sorts by band →
2D Hilbert over `(phi, eta)` within each band. Constants
`CLAS12_LAYER_RADII_RAW`, `CLAS12_LAYER_DELTA_RAW=0.3` in the file; converted
internally to normalized space (called on post-`apply_norm` `r`).

## DATA CONTRACT

### Format
Output is **3 `RaggedMmap` directories per split** (`pretrain` and `test`), six
total. RaggedMmap stores one variable-length array per event/sequence; mechanism
is unchanged from TPC, only the contents differ. Read with
`RaggedMmap(path)[i]` → the i-th event's array.

```
<data_root>/
  features_pretrain/     reg_target_pretrain/     seg_target_pretrain/
  features_test/         reg_target_test/         seg_target_test/
```

### Per-event arrays — exact shapes, dtypes, columns

**features_{split}** — model input
- dtype `float32`, shape `(n_hits, 3)`
- columns: `[x0, y0, z0]` — Cartesian hit position, **raw detector units (cm)**
- **NO energy column.** This is the single biggest break from TPC's
  `[E, x, y, z]`. Anything indexing column 0 as energy is wrong.
- `r`, `phi`, `eta` are NOT stored — derived downstream from `(x,y,z)` via the
  standard `cartesian_to_polar_batched`, which returns `[eta, phi, r]`.

**seg_target_{split}** — per-hit track label (segmentation)
- dtype `float32`, shape `(n_hits,)` — one value per hit, **row-aligned with
  features** (hit `i` in features ↔ label `i` here)
- value = `trkID`. Integer-valued but stored as float; cast as needed.
- `trkID = -1` means a **noise hit** (not part of any real particle).
- **trkID is reconstruction output, not pure truth:** one real particle can be
  split across multiple trkIDs, and trkID values (1,2,3,…) **repeat across
  different events** (event 0's trkID 1 ≠ event 5's trkID 1). To recover
  individual tracks, **group by (event index, trkID) together — never trkID
  alone.** Physical truth is always 3 particles/event regardless of distinct
  trkID count.

**reg_target_{split}** — regression target (STUB)
- dtype `float32`, shape `(n_hits, 7)`, row-aligned with features
- **all zeros — placeholder only.** Exists solely because the loader loads it
  unconditionally; pretraining never reads it.
- Nominal column meaning (from TPC) `[px, py, pz, vtx_x, vtx_y, vtx_z, energy]`,
  **but every value is 0.0.** If a downstream regression task needs real
  momentum/vertex targets, **they are not in this data** — source them from MC
  truth separately and build your own `reg_target`.

### Per-event characteristics (standard event-grouped data)
- One array index = one collision event = up to 3 particles' hits, interleaved.
- `n_hits` per event: ~17 typical, range ~5–40 after the `low_thr=5/high_thr=40`
  filter. (~12.5% of all hits are noise, `trkID=-1`.)
- Hits within an event arrive in arbitrary order; the dataset re-sorts by radius
  and applies the Hilbert band ordering at load time — so **do not assume any
  input ordering** in the stored arrays.
- Every hit's `r = sqrt(x0²+y0²)` falls within ±0.3 of one of the 6 layer radii.

### Alignment guarantee
For event `i`: `features[i]`, `seg_target[i]`, `reg_target[i]` all have the same
`n_hits` (first dim) and are **row-aligned** — row `j` across all three refers
to the same physical hit.

### Source / regeneration
Built from two CSVs (BST + BMT). I keep only `event, x0, y0, z0, trkID`, merge
per-event, write the mmap. Dropped: `sector` (inconsistent width/location per
layer — not a usable geometric key), `region`/`c_layer`/`z_layer` (bookkeeping),
`theta` (redundant), BMT `c_*`/`z_*` variant points, `err_*`. Conversion script
available from me.

## Files to take from my repo (if adopting my changes)
Drop-in or merge these into your repo:

**Shared model/util files — safe to take as-is (fixes are CLAS12-compatible and
don't break the energy path except where noted):**
- `fm4npp/models/embed.py` — `EmbedderPosOnly` fixes.
- `fm4npp/models/mambagpt.py` — adds `pos_only` branch (keeps `concat`/`add`).
- `fm4npp/hilbert.py` — adds `clas12_band_hilbert_order` + constants (purely
  additive; original `encode`/`decode` untouched).

**Reference only — adapt into your own task files, don't blind-copy:**
- `fm4npp/datasets/dataset_pretrain.py` — my CLAS12 dataset; use as the template
  for the same edits in your `dataset.py` / `dataset_eval.py`.
- `train/pretrain/nppmamba/train_multi_gpu_mamba1.py` — shows the reshape fix +
  the yaml-attr / pickle handling; apply the equivalents to your downstream
  trainer.
- `scripts/configs/mamba_pretrain.yaml` (`clas12_pretrain` block) — template for
  your downstream config (`embed_method: pos_only`, 3-col assumptions,
  `rep_aaai`/`nexttoken`/`ablate_loss_scale: false`, CLAS12 paths).

**Need from me separately (not in repo):**
- CSV→RaggedMmap conversion script (produces the data described above).
- Dummy `loss_bin_pp.pkl` / `loss_weight_pp.pkl` (or just generate your own).

**Minimum to run a CLAS12 model at all:** `embed.py` + `mambagpt.py` (model
must support 3-col `pos_only`), a 3-col dataset class, the trainer reshape fix,
and the data in the contract format.
