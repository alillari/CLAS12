# CLAS12 Refit Tracking — `dataset_pretrain.py`

Working document. Original repo: `~/projects/PP_collision` (untouched TPC reference).
CLAS12 fork: `~/projects/PP_collision_clas12`.

Status legend: 🔲 not started · 🟡 in progress · ✅ decided/done

---

## Confirmed decisions going in (from prior discussion)

1. **No energy** — drop entirely. Input is `(eta, phi, r)` only, 3 channels, not 4.
2. **Ordering** — replace Voxelizer/HRS box-partition with a Hilbert space-filling
   curve over `(radius, phi, eta)`, radius given dominant priority in the recursive
   ordering. (Note: the file already has an *unused* `space_filling_order` path —
   see Section 2 below, this is not what we'll use as-is, but it's relevant prior art.)
3. **kNNN** — `klen`/`num_pred_points` reduced from 30 → 1 (starting point, trivially
   adjustable later, only resizes the final linear layer).
4. **No track-ID leakage** — ordering function must use only geometric fields
   available at pretraining time, never `trkID`.
5. **Fixed/deterministic** — no statistically-fit bins, no per-event adaptive logic.

---

## Section-by-section walkthrough

### 1. Imports (lines 1-17) — 🔲

```python
import numpy as np
import torch
from torch.utils.data import Dataset
from mmap_ninja import RaggedMmap
from pathlib import Path
import os
import glob
import torch.nn as nn

import torch
from fm4npp.utils import *
from .voxelizer import *

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(42)
```

**Plan:** `from .voxelizer import *` will likely be removable once Voxelizer is
fully replaced — TBD once we confirm nothing else in this file needs it besides
the `self.voxelizer` object itself. Need to add the Hilbert encoding function
(from `fm4npp/hilbert.py`'s `encode`, already imported transitively via
`from fm4npp.utils import *` since `utils.py` defines `encode`/`hilbert_encode`
itself — confirmed in our earlier trace). **No new import needed for Hilbert** —
it's already reachable through the existing `utils` import.

---

### 2. `rescale_serialize_Rlast` (lines 19-38, duplicated again at 136-155) — 🔲

```python
def rescale_serialize_Rlast(centers, scaler = 1e4, order='z'):
    """
    Reorder centroids based on a designated order.
    Rlast indicates that R will be the last global order.
    arr: (N x 3) -> should be integer location
    """
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if len(centers.shape) > 2:
        centers = centers.squeeze(0)
    arr = centers[..., 1:]
    
    arr = swap_dim(arr)
    toserial = (arr * scaler).long() # Making the floating points to integer.
    ordered = encode(toserial, batch=None, depth=16, order=order)
    sorter = torch.argsort(ordered)
    out = arr[sorter]
    out = swap_dim(out)
    out = torch.cat([centers[..., 0:1], out], dim=-1)
    
    return out.unsqueeze(0), sorter
```

**IMPORTANT DISCOVERY:** this function is defined **twice**, identically (lines
19-38 and again 136-155) — looks like a copy-paste artifact in the original repo,
not something we introduced. Worth flagging but not fixing in the original;
in our CLAS12 fork we'll only keep one copy.

**What it actually does, traced:**
- Takes `centers` shape `(N, 4)` — note: hardcoded to expect a 4-column input
  (`centers[..., 0:1]` assumed to be E, `centers[..., 1:]` assumed to be the
  3 spatial coords it reorders).
- `swap_dim` (see Section 3) swaps columns 1 and 2 of the 3-column spatial part
  — i.e., swaps eta and phi order, putting `(phi, eta, r)` → `(eta, phi, r)` or
  vice versa depending which order they started in. **Need to check call site
  to know which.**
- Scales by `1e4` and casts to integer — this is the float→int quantization
  step Hilbert/Z-order encoding requires (confirmed from `utils.py`'s `encode`
  needing integer grid coords).
- Calls `encode(..., order=order)` — this is the **already-implemented**
  Hilbert/Z-order call we were planning to add ourselves. `order='hilbert'` is
  already a valid, supported option here.
- Sorts by the resulting 1D code, returns reordered array + the sorter permutation.

**This function is already 90% of what we need for our Hilbert ordering change.**
Caveats / required changes for CLAS12:
- Hardcoded to 4 columns (`centers[..., 0:1]` for E). **Must change to 3 columns**
  once energy is dropped — `centers` becomes pure `(eta, phi, r)` with no E
  column to split off.
- Uses a **single global `scaler=1e4`** for all 3 dims equally — this treats all
  axes with equal resolution/priority. **Must change to give radius dominant
  priority** (per our decision) — likely needs per-axis scaling/bit-depth rather
  than one uniform scaler, or a different axis ordering passed into `encode`.
  Need to check `fm4npp/hilbert.py`'s `encode` signature for how axis priority
  could be controlled (e.g., does axis order in the input array matter to the
  recursive construction, or only `depth`?). **Open question — need to check
  hilbert.py directly before finalizing this.**
- `depth=16` hardcoded — fine to keep as-is unless we find a reason to change it.

**Status:** Strong candidate to become the core of our new ordering function with
modification, not a from-scratch build. Need `fm4npp/hilbert.py` to resolve the
axis-priority question before finalizing.

---

### 3. `knn_later_indices_batch` (lines 41-122) — 🔲

Already fully traced earlier in this conversation. Mechanism: for each point `i`,
search all points `j` with strictly higher value in the last input column (R, in
the current TPC usage), take the `k` closest by Euclidean distance among them,
gather their raw coordinates, pad with `-100` if fewer than `k` exist.

**Required changes for CLAS12:**
- `assert D == 3` (line 53) — **already expects 3 columns**, not 4! Confirms this
  function already operates on `(eta, phi, r)` only — i.e. it's *already* called
  with `norm_features[..., 1:]` (E stripped) in the current `__getitem__`. **No
  change needed here for the "no energy" requirement** — this function never saw
  energy in the first place.
- `k` parameter — change call site to pass `1` instead of `self.num_pred_points`
  (or just set `num_pred_points=1` at the `TPCBatchDataset.__init__` call site /
  config level — **no code change needed inside this function itself**, it's
  already fully general over `k`).

**Status:** Needs zero internal modification. Only the `k` value passed in changes,
and only at the config/call-site level.

---

### 4. `swap_dim` (lines 124-128) — 🔲

```python
def swap_dim(arr, dims = [1,2]):
    c = arr.clone()
    c[..., 1] = arr[..., 2]
    c[..., 2] = arr[..., 1]
    return c
```

Swaps columns 1 and 2 of a 3-column array. Takes a `dims` parameter that's
**defined but never used** — the body hardcodes indices 1 and 2 regardless of
what `dims` is set to. Minor dead-parameter bug in the original, not ours to fix
unless it affects us.

**Open question:** need to confirm exactly what column order `rescale_serialize_Rlast`
expects vs. what we'll be feeding it (eta,phi,r vs phi,eta,r) — relevant once we
finalize the radius-priority Hilbert ordering. Revisit after Section 2's open
question is resolved.

---

### 5. `strip_masked` (lines 130-134) — 🔲

```python
def strip_masked(g, maskval = -100):
    """input: 1 x N x group_size x 4"""
    assert g.size(0) == 1, 'only for batch_size of 1'
    masker = g[..., 1:].mean(-1).mean(-1) != -100
    return g[masker].unsqueeze(0)
```

Hardcoded to 4-column input (`g[..., 1:]` assumes column 0 is something to skip,
i.e. E). **Need to check if this function is even called anywhere in the active
pretraining path** — not seen yet in `__getitem__` or the Group class forward.
Possibly dead code / only used by a different (downstream?) path. Flag for
later — don't change yet, confirm usage first.

---

### 6. `serialize_neighbors` (lines 157-176) — 🔲

Calls `rescale_serialize_Rlast` per-group in a loop. **Need to confirm if this
is called anywhere in the active pretrain path** — not seen yet in
`TPCBatchDataset.__getitem__`. Possibly part of the `Group`/FPS-KNN downstream
mechanism (see Section 7), not pretraining. Flag for later, same as Section 5.

---

### 7. `Group` class (lines 178-204) — 🔲

FPS (farthest point sampling) + KNN grouping — uses `sample_farthest_points`,
`knn_points` (likely from `pytorch3d` or similar, not yet confirmed import
source). **Need to confirm this isn't used in the pretrain path** — looks like
it may be leftover/shared code from a different task (possibly the downstream
Group-based tokenization mentioned in earlier SETUP.md docs, distinct from the
Voxelizer-based pretrain path we've been tracing all along). Flag, don't touch
yet.

---

### 8. `rescale_polar_radius`, `minmax_normalize`, `apply_norm` (module-level, lines 206-219) — 🔲

```python
def apply_norm(features):
    """Dim 2 and 3 are the same to preserve absolute distance"""
    fnorm = features.clone()
    for i in range(4):
        fnorm[..., i] = minmax_normalize(fnorm[..., i], features[..., i].max(), features[..., i].min())
    return fnorm
```

**IMPORTANT:** this is a **second, different, module-level `apply_norm`** —
NOT the one we traced earlier (`TPCBatchDataset.apply_norm`, an instance method,
lines 375-381, using hardcoded constants). This module-level version:
- Hardcoded to 4 columns (`range(4)`)
- Uses **per-batch min/max** (`features[..., i].max()`), not fixed constants —
  totally different normalization strategy than the instance method.
- **Need to confirm this is unused / dead code in the pretrain path** — the
  actual `__getitem__` calls `self.apply_norm(...)`, the instance method, not
  this module-level function. Possibly leftover from an earlier version or used
  by a different script entirely. Flag, don't touch yet, but note for CLAS12:
  if this WERE used, dropping energy would require changing `range(4)` to
  `range(3)` — keep in mind if we discover it's actually live somewhere.

---

### 9. `group_points` (lines 221-235) — 🔲

Hardcoded 4-column pad value tensor (`torch.ones(..., c) * pad_val` — actually
general over `c`, fine). **Need to confirm usage in pretrain path** — not seen
in `__getitem__` yet. Flag, don't touch yet.

---

### 10. `set_simpler` (lines 237-280) — 🔲

Filters down to the `nleave` largest trajectories per event by point count.
**Referenced in `__getitem__` but commented out** (line 481:
`# features, target = set_simpler(...)`) — confirms this is currently inactive
in the live pretrain path, consistent with `self.nleave` defaulting to `1e6`
(i.e., never actually triggers filtering in practice). Not a concern for CLAS12
unless we want to re-enable it — flag as available-but-unused.

---

### 11. `TPCBatchDataset.__init__` (lines 282-359) — 🟡 KEY SECTION

```python
def __init__(self, 
             data_root, 
             version = 'pp_100k',
             train = True,
             split = 'pretrain',
             nleave = 1e6,
             npoint_lower_thr = 5,                  
             group_size = 32, 
             normalize_by_center = False, 
             normalize = True,
             order = 'REP', 
             num_pred_points = 10, 
             klen = 5,
             len_chunk = 512,
             chunk_training = False,
             limit_data = False,
             limit_size = 8000, 
             voxelize = True,
             space_filling_order = None,
             space_filling_curve = 'z',
             bin_dir = ''):
    
    split = split
    self.memmap_feature = RaggedMmap(os.path.join(data_root, 'features_{}'.format(split)))
    self.memmap_seg_target = RaggedMmap(os.path.join(data_root, 'seg_target_{}'.format(split)))
    self.memmap_reg_target = RaggedMmap(os.path.join(data_root, 'reg_target_{}'.format(split)))
    

    self.reco_cols = ['E', 'x', 'y', 'z']
    self.particle_reg_cols = ['px', 'py', 'pz', 'vtx_x', 'vtx_y', 'vtx_z', 'energy']
    self.particle_seg_col = 'track_id'
    
    # filtering out some trajectories
    self.nleave = nleave
    self.order = order
    self.npoint_lower_thr = npoint_lower_thr
    self.num_pred_points = num_pred_points
    
    # voxelization ablation
    self.voxelize = voxelize
    self.space_filling_order = space_filling_order
    self.space_filling_curve = space_filling_curve
    
    # for normalization
    self.eta_lim = {'min':-2, 'max':2}
    self.phi_lim = {'min':-torch.pi, 'max':torch.pi}
    self.r_lim = {'min': 31.371997833251953, 'max': 75.38493347167969}
    self.E_mean, self.E_std = 253.0982, 268.7093
    # (E)ta / (P)hi / (R)adius
    self.orderdict = {
        'EPR': {'dim_sweep_order':[2,1,0], 'revert_order':[2,1,0]},
        'RPE': {'dim_sweep_order':[0,1,2], 'revert_order':[0,1,2]},
        'REP': {'dim_sweep_order':[1,0,2], 'revert_order':[1,0,2]},
        'PER': {'dim_sweep_order':[2,0,1], 'revert_order':[1,2,0]},
             }

    dim_sweep_order = self.orderdict[self.order]['dim_sweep_order']
    revert_order = self.orderdict[self.order]['revert_order']
    
    self.low_thr = 50
    self.normalize = normalize
    
    # Tokenizer
    self.group_size = group_size
    self.normalize_by_center = normalize_by_center
    self.voxelizer = Voxelizer(bin_dir = bin_dir, bin_version = 'v3', n_bins = (8, 8, 6), dim_sweep_order=dim_sweep_order, revert_order=revert_order)
    self.dim_sweep_order = dim_sweep_order
    self.revert_order = revert_order
    self.limit_data = limit_data
    self.limit_size = limit_size
    self.len_chunk = len_chunk
    
    self.train = train
    self.chunk_training = chunk_training
    self.filter_data(high_thr = 3200)
    import math
    self.data_scaler = 1 # [TOGGLE][TEMPORARY] SCALER
```

**Confirmed required changes:**

1. **`self.particle_reg_cols`** — list of 7 momentum/vertex names, the
   `reg_target` columns we already know are absent from public data (separate
   issue from this conversation's TPC work, but same underlying gap — CLAS12
   has its own version of "what would reg_target even be," **TBD, separate
   decision, not blocking pretraining since reg_target is unused in the loss**.
2. **`self.eta_lim`, `self.phi_lim`, `self.r_lim`, `self.E_mean/E_std`** — ALL
   of these are **TPC-specific hardcoded constants**. `r_lim` in particular
   (31.37 to 75.38) reflects TPC's continuous radius range — **completely wrong
   for CLAS12's 6 discrete radii** (~6.5 to ~22.3, per the real header samples
   you showed earlier). **Must be recomputed from real CLAS12 data.**
   `E_mean`/`E_std` — **delete entirely**, no energy channel.
3. **`self.voxelizer = Voxelizer(...)`** — **DELETE.** This is the line that
   unconditionally builds the Voxelizer regardless of `voxelize`/
   `space_filling_order` flags — confirms our earlier suspicion that even using
   the existing space-filling path today would still hit this line and require
   a bin-edges pickle for no reason. For CLAS12, since we're dropping
   Voxelizer/HRS entirely, this line and the `self.dim_sweep_order`/
   `self.revert_order`/`self.orderdict` machinery feeding it can all be deleted.
4. **`group_size`, `normalize_by_center`** — appear to feed the `Group` class
   (Section 7), which we suspect is unused in the pretrain path. Leave for now,
   revisit once Section 7's usage question is resolved.
5. **`space_filling_order`, `space_filling_curve` params already exist** — we
   will likely repurpose these rather than invent new ones, once we finalize
   exactly how our Hilbert call differs from the existing `rescale_serialize_Rlast`
   path (Section 2's open question).

**Status:** Clear picture of what to delete (Voxelizer construction, TPC norm
constants) and what to recompute (real CLAS12 eta/phi/r min/max). Blocked on
Section 2's axis-priority question before finalizing the ordering-related
constructor args.

---

### 12. Normalization instance methods (lines 361-389) — 🔲

`znormalize`/`z_unnormalize`/`minmax_normalize`/`minmax_unnormalize`/`apply_norm`/
`apply_unnorm` — all general-purpose, reference `self.E_mean` etc. set in
`__init__`. **No internal logic changes needed** — once energy is dropped and
`__init__`'s constants are fixed (Section 11), `apply_norm`/`apply_unnorm` need
their 4-column loop (`fnorm[..., 0]` through `[..., 3]`) reduced to 3 columns
(`eta, phi, r` only, no `E` line).

---

### 13. `filter_data` (lines 391-436) — 🔲

```python
def filter_data(self, low_thr = -1, high_thr = 10e10):
    ...
```
Called as `self.filter_data(high_thr = 3200)` in `__init__` (line 357) —
**TPC-specific threshold.** `low_thr=50` is also set as a hardcoded instance
attribute (line 342, inside `__init__`, not as a `filter_data` default).

**Required change:** Both thresholds need CLAS12-appropriate values. Given your
own description — typical events ~18 points, noise events 100+ — `high_thr=3200`
is meaningless (would never filter anything CLAS12-scale) and `low_thr=50` would
**incorrectly discard almost every real CLAS12 event** (since most have far
fewer than 50 points!). **This is a real, concrete bug we must fix** — using
the TPC defaults unchanged would filter out nearly all valid CLAS12 data.
Need real thresholds — e.g., `low_thr` maybe 3-4 (per your "could be 3 or 4
hits" comment), `high_thr` maybe 100-150 (to exclude the heavy-noise outlier
events, if that's desired) — **TBD, your call on the actual cutoff numbers.**

---

### 14. `cut_chunk`, `__len__` (lines 438-463) — 🔲

General-purpose, reference `self.low_thr`/`self.len_chunk`/`self.chunk_training`.
No direct changes needed beyond what Section 13 already fixes.

---

### 15. `__getitem__` (lines 465-520) — 🟡 KEY SECTION, ALREADY PARTIALLY TRACED

```python
features = torch.from_numpy(np.copy(self.memmap_feature[real_idx])).unsqueeze(0)
target = torch.from_numpy(np.copy(self.memmap_seg_target[real_idx])).unsqueeze(0)

## To polar representation
polar_coord = cartesian_to_polar_batched(features[..., 1:])
E = features[..., 0:1]
polar_features = torch.cat([E, polar_coord], dim=-1)

## Normalize the polar representation
if self.normalize:
    norm_features = self.apply_norm(polar_features)
else:
    norm_features = polar_features

# Sort by R
ind = norm_features[...,-1].argsort(dim=1)
norm_features = norm_features[:, ind.squeeze()]
knearest_points = knn_later_indices_batch(norm_features[..., 1:], k=self.num_pred_points)
norm_target = target[:, ind.squeeze()]

if self.space_filling_order:
    _, zsorter = rescale_serialize_Rlast(norm_features, scaler = 1e4, order=self.space_filling_curve)
    serialized_points = norm_features[:, zsorter.squeeze()].squeeze(0)
    knearest_points = knearest_points[:, zsorter.squeeze()].squeeze(0)
    serialized_target = norm_target[:, zsorter.squeeze()].squeeze(0)

else:

    if self.voxelize:
        quantized = self.voxelizer.tokenize(norm_features, start_idx = 1)
        grouped = self.voxelizer.grouping(quantized)
        gsort, sorter = grouped.sort(dim=-1, stable=True)
        serialized_points = norm_features[:, sorter.squeeze()].squeeze(0)
        knearest_points = knearest_points[:, sorter.squeeze()].squeeze(0)
        serialized_target = norm_target[:, sorter.squeeze()].squeeze(0)
    else:
        serialized_points = norm_features.squeeze(0)
        knearest_points = knearest_points.squeeze(0)
        serialized_target = norm_target.squeeze(0)

return serialized_points * self.data_scaler, serialized_target, knearest_points * self.data_scaler
```

**This is the big one. Confirmed required changes:**

1. **`features = ...`** — raw load. **No change to the load mechanism itself**,
   but the *source* data (RaggedMmap built from CLAS12 x,y,z, not TPC E,x,y,z)
   means `features[..., 0]` is no longer "E" — **need to decide: does our
   CLAS12 RaggedMmap store raw `(x,y,z)` as a 3-column array (no leading energy
   column at all), or do we keep a 4-column array with a dummy/zero leading
   column for compatibility?** Given our "remove energy completely, don't fake
   it with a placeholder" decision earlier, **the correct approach is a genuine
   3-column array, no placeholder column at all.** This cascades through every
   line below.
2. **`polar_coord = cartesian_to_polar_batched(features[..., 1:])`** — currently
   strips off column 0 (E) before converting. **Change to
   `cartesian_to_polar_batched(features)`** directly on all 3 columns, no
   stripping needed, once features is genuinely 3-column `(x,y,z)`.
3. **`E = features[..., 0:1]; polar_features = torch.cat([E, polar_coord], ...)`**
   — **DELETE both lines entirely.** `polar_features = polar_coord` directly,
   3 columns `(eta, phi, r)`.
4. **`norm_features = self.apply_norm(polar_features)`** — works once
   `apply_norm` itself is reduced to 3 columns (Section 12) and `__init__`'s
   eta/phi/r constants are recomputed for CLAS12 (Section 11).
5. **`ind = norm_features[...,-1].argsort(dim=1)`** — sorts by last column,
   which is still `r` in a 3-column layout (`eta, phi, r` → index -1 is still
   r). **No change needed**, indexing already works correctly for 3 columns.
6. **`knearest_points = knn_later_indices_batch(norm_features[..., 1:], k=...)`**
   — currently strips column 0 expecting to drop E and keep
   `(eta,phi,r)`→3 cols for the `assert D==3` check. **Once `norm_features` is
   ALREADY 3 columns total, `[..., 1:]` would incorrectly strip off `eta` too,
   leaving only `(phi, r)` — 2 columns, failing the `assert D==3`.** **Must
   change to `knn_later_indices_batch(norm_features, k=...)`** — no stripping,
   pass all 3 columns directly.
7. **`if self.space_filling_order: ... rescale_serialize_Rlast(norm_features, ...)`**
   — this existing branch is close to what we want, but
   `rescale_serialize_Rlast` itself is hardcoded for 4-column input
   (`centers[..., 0:1]` for E, Section 2) — **must update that function for
   3 columns too**, and resolve the radius-priority question before using this
   branch as our actual path.
8. **`if self.voxelize: ...` branch and the Voxelizer object itself** — **DELETE
   ENTIRELY.** No box-partition, no `self.voxelizer.tokenize/grouping` calls.
9. **Final `return serialized_points * self.data_scaler, ...`** — no change,
   `data_scaler=1` is a no-op regardless of column count.

**Status:** Fully mapped out. Real blocking dependency: Section 2's open
question (how to make Hilbert prioritize radius) needs an answer before this
section's `space_filling_order` branch can be finalized and adopted as our
actual path (rather than written from scratch).

---

### 16. `MyCollator` (lines 523-554) — 🔲

Pads to longest event in batch with `-100`, stacks. **General over column count**
— `g.size(0)` is the point-count dimension, padding logic doesn't hardcode 4
anywhere. **No changes needed.**

---

### 17. `get_data_loader`, `get_val_loader` (lines 558-653) — 🔲

Constructs `TPCBatchDataset` with config-driven params. **Required changes:**
- `bin_dir = params.stat_dir` — **becomes irrelevant/removable** once Voxelizer
  is deleted from `__init__` (Section 11).
- `voxelize = params.voxelize` — should be hardcoded `False` (or the param
  removed from the constructor signature entirely) for CLAS12.
- `space_filling_order = params.space_filling_order` — should be hardcoded
  `True` for CLAS12 (this is now our primary/only path, not an ablation toggle).
- `params.klen` — set to `1` in the CLAS12 config file (not a code change here,
  a config value change — flagging for completeness).
- **BUG NOTED, line 630:** `get_val_loader` references `**self.orderdict[params.order]`
  but `self` is not defined inside this free function (`orderdict` is an
  instance attribute of `TPCBatchDataset`, not available here) — **this looks
  like a pre-existing bug in the original repo**, would raise `NameError` if
  this code path is ever actually called. Not introduced by us; flag and avoid
  triggering this path, or fix it while we're in here regardless since it's
  trivially broken as written.

---

## Resolved since initial walkthrough

### Hilbert axis-priority — RESOLVED (Option B)

Checked `fm4npp/hilbert.py` and `fm4npp/z_order.py` directly: **neither curve
has a built-in axis-priority mechanism.** Both treat all input dimensions with
identical structural weight (Z-order interleaves one bit per axis per depth
level, symmetric by construction; Hilbert's `encode` loops over all dims
identically, no per-axis weighting parameter exists). "Give radius more bits
than phi/eta" (unequal bit allocation) was considered but rejected as only a
soft statistical bias, not a guarantee — and CLAS12's radius is genuinely,
near-exactly discrete (6 known bands, confirmed below), so a guarantee is both
available and preferable to a bias.

**Decided approach (Option B):**
1. Snap every point's measured `r` into exactly one of 6 known fixed bands
   (see radii below) — this is a hard assertion, not a nearest-match/fallback,
   per the physical reasoning that CLAS12's detector hardware only exists at
   these 6 radii (signal AND noise hits are both constrained to real sensor
   locations — there is no "in-between" space the detector can register a hit
   in). If a point's `r` doesn't fall in any band, that indicates corrupted
   input data, not a normal case to handle gracefully — assert/fail loudly.
2. Sort all points by band index (1-6) first — this is the global, exact,
   zero-ambiguity radius ordering. **Not a re-introduction of HRS-style
   binning**: the bands are physical constants (detector geometry), not
   statistically-fit edges over a continuous distribution — same category of
   "fixed constant" as `E_mean`/`E_std` already was in the original code, not
   the same category as `Voxelizer`'s `equalObs`.
3. Within each band, run a **2D** Hilbert curve (`encode(locs, num_dims=2,
   num_bits=...)`) over `(phi, eta)` only — radius has zero remaining variance
   within a band by definition, so only 2 real coordinates are left to order.
4. Concatenate band-groups in band-index order. This guarantees zero
   cross-layer interleaving (impossible by construction) while still using
   Hilbert's locality guarantee for the angular tie-break within a layer.

**Real layer radii, measured directly from data** (script:
`r_histogram_check.py`, run against full BST/BMT CSVs):

| Layer | Source | Peak r | Min | Max | Notes |
|---|---|---|---|---|---|
| 1 | BST region 1 | 6.52944 | 6.52870 | 6.82496 | real measurement scatter |
| 2 | BST region 2 | 9.28923 | 9.28870 | 9.49927 | real measurement scatter |
| 3 | BST region 3 | 12.03261 | 12.03220 | 12.19549 | real measurement scatter |
| 4 | BMT region 1 | 14.76460 | 14.76460 | 14.76460 | exact — software-snapped, zero spread |
| 5 | BMT region 2 | 19.26460 | 19.26460 | 19.26460 | exact — software-snapped, zero spread |
| 6 | BMT region 3 | 22.26460 | 22.26460 | 22.26460 | exact — software-snapped, zero spread |

**Band half-width δ = 0.3** — single shared value across all 6 layers, chosen
as the worst-case max deviation observed (BST region 1: peak-to-max gap of
0.29552), rounded up slightly for safety margin. BMT needs no tolerance at all
(exact values) but uses the same shared δ for simplicity/consistency — since
BMT hits are exact, a δ=0.3 band around them will never erroneously capture a
neighboring layer's point (inter-layer spacing is ~2.5-5.0, far larger than
2×δ=0.6).

**These radii are hardcoded constants** — not derived per-event, not refit if
the dataset changes — consistent with the "fixed, deterministic" requirement
held throughout this design process.

### kNNN k value — tentatively k=1, flagged for later sweep

Per "reduced space → next nearest prediction" from researcher notes, `klen`/
`num_pred_points` set to 1 as a starting point. Confirmed this only affects
`Mamba1GPT`'s final `nn.Linear(embed_dim, klen*3)` output layer — learned
latent representations are completely unaffected by this value, so it is safe
to treat as a cheap, sweepable hyperparameter later (try 2, compare downstream
performance) rather than a value that needs to be "correct" upfront.

### No-energy — confirmed full removal, not a sentinel value

Considered and rejected setting energy to `-100` (gets silently zeroed by
`Mamba1GPT.change_maskval` before reaching the embedder, with no exclusion
mechanism downstream the way real padding has via the loss mask) and rejected
setting it to any other constant (still flows through `EmbedderAdd`'s learnable
`self.proj` and into every loss computation, since no padding-style mask exists
for "real point with meaningless channel"). **Decision: remove the energy
pathway from the embedder architecturally** — use something built on
`EmbedderPosOnly` (already present, unused, in `embed.py`) rather than
`EmbedardAdd`, feeding only `(eta, phi, r)`.

---

### Normalization bounds (eta_lim/phi_lim/r_lim) — RESOLVED

Measured directly from data (`minmax_check.py`, combined BST+BMT):
`r: [6.52870, 22.26460]`, `phi: [-3.14153, 3.14154]`, `eta: [-2.08875, 1.20874]`.

**Locked-in constants, with padding:**
- `r_lim = {'min': 6.0, 'max': 23.0}` — matches the existing constant already
  hardcoded in the plotting script (`R_NORM_MIN, R_NORM_MAX = 6.0, 23.0`),
  independent confirmation this is a sensible round bound.
- `phi_lim = {'min': -torch.pi, 'max': torch.pi}` — unchanged from the
  original TPC constant. No padding needed/meaningful: phi is mathematically
  bounded by `arctan2`, cannot exceed `±π` regardless of dataset.
- `eta_lim = {'min': -2.5, 'max': 1.5}` — asymmetric padding around the
  observed `[-2.089, 1.209]` range, rounded to a half-integer with headroom
  on both sides.
- `E_mean`, `E_std` — **deleted entirely**, no energy channel.

These replace `TPCBatchDataset.__init__`'s TPC-specific `eta_lim`/`phi_lim`/
`r_lim`/`E_mean`/`E_std` block (Section 11).

### Filter thresholds (low_thr/high_thr) — RESOLVED

Based on a real histogram of hit-count-per-event across the dataset: the
distribution sits well within 5 to 40 hits per event. **`low_thr = 5`,
`high_thr = 40`** — replaces TPC's `low_thr=50`, `high_thr=3200` (Section 13).
Note the much narrower span here (8x, vs TPC's 64x) reflects CLAS12's far
more uniform event size (small fixed particle/layer count) compared to TPC's
wide variation in points-per-event.

### Sections 5-10 dead code status — RESOLVED

Repo-wide `grep` confirms: `strip_masked`, `serialize_neighbors`, `class Group`,
module-level `apply_norm`, `group_points`, `set_simpler` are each defined
identically in all three dataset files (`dataset.py`, `dataset_pretrain.py`,
`dataset_eval.py`) but **called nowhere in the entire repo** — not just unused
in pretraining, genuinely dead in every file they appear in. `set_simpler`'s
only reference anywhere is the pre-existing commented-out line. Module-level
`apply_norm` is shadowed/superseded by the instance method `self.apply_norm`,
which is the only version any call site ever actually invokes.

**Decision: safe to delete all 6 from our CLAS12 fork's `dataset_pretrain.py`**
with zero risk of breaking anything — confirmed dead, not just locally unused.

---

## Open questions blocking finalization

1. ~~Hilbert axis-priority mechanism~~ — **RESOLVED**, see above.
2. ~~Real CLAS12 eta/phi/r min/max for normalization~~ — **RESOLVED**, see above.
3. ~~`low_thr`/`high_thr` filter values~~ — **RESOLVED**, see above.
4. ~~Sections 5-10 dead code status~~ — **RESOLVED**, see above. Confirmed
   safe to delete.
5. **Write the actual band-snapping + 2-level ordering function** — code
   drafted and bug-fixed (`hilbert_clas12_addition.py`), raw-vs-normalized r
   issue resolved by converting layer radii/delta into normalized space using
   the locked r_lim. Not yet smoke-tested or wired into `__getitem__`.
6. ~~num_bits for 2D Hilbert encoding~~ — **RESOLVED**, kept at 10 bits.

## ALL DESIGN QUESTIONS RESOLVED — ready to write the actual CLAS12
## `dataset_pretrain.py` in full, then smoke-test.

---

## Section B: `train_multi_gpu_mamba1.py` + `mamba_pretrain.yaml`

Checked for hardcoded dimension assumptions outside `dataset_pretrain.py`,
since the dataset file alone doesn't guarantee the training loop adapts
correctly to a 3-column (no-energy) input.

### CRITICAL BUG FOUND — hardcoded column count, must fix

**File:** `train/pretrain/nppmamba/train_multi_gpu_mamba1.py`
**Lines 287 and 377 (training loop AND validation loop, both occurrences)**

```python
targets = grouped.reshape(b, -1, 4)[:, :, 1:].to(self.device)
```

This hardcodes `4` as the column count and then slices `[:, :, 1:]` to strip
column 0 (E), keeping `(eta, phi, r)`. **Once `grouped` is genuinely 3 columns
(no energy), this line is wrong in two ways at once:**
1. `reshape(b, -1, 4)` will fail (or silently misbehave if element count
   happens to divide evenly by 4 by coincidence) since the real last
   dimension is 3, not 4.
2. `[:, :, 1:]` assumes there's an E column at index 0 to strip — with no E
   column at all, this slicing is not just unnecessary, it would incorrectly
   drop a real coordinate (whichever ends up at index 0 — eta, in our layout).

**Required fix, both occurrences:**
```python
targets = grouped.reshape(b, -1, 3).to(self.device)
```
No slicing — `targets` should just be the reshaped 3-column tensor directly,
no column to strip.

Note: `b, c = grouped.size(0), grouped.size(-1)` (lines 284, 376) already
reads the real column count dynamically into `c` — the bug is that the very
next lines ignore `c` and hardcode `4` instead. `c` itself needs no change.

### `klabel` reshape — confirmed correct as-is, no code change needed

```python
klabel = knearest.reshape(b, -1, self.klen * 3).to(self.device)
```
`self.klen` is read from `self.params.klen` at `__init__` time. Once the
CLAS12 yaml config sets `klen: 1`, this becomes `reshape(b, -1, 3)`
automatically via the config value — no source change required here.

### `loss_bin`/`loss_weight` pickle files — same unconditional-load pattern as TPC voxelizer stats

```python
self.loss_bin = pickle_load('{}/loss_bin_pp.pkl'.format(self.params.stat_dir))
self.loss_weight = pickle_load('{}/loss_weight_pp.pkl'.format(self.params.stat_dir))
```
Loaded unconditionally in `__init__`, regardless of whether they're ever used.
Confirmed only actually read when `self.params.loss_reweight` is `True`
(line 314) — and the yaml's default block sets `loss_reweight: false`. Same
situation as the TPC `Voxelizer` bin-edges pickle from earlier in this
project: **we need dummy/placeholder pickle files at
`{stat_dir}/loss_bin_pp.pkl` and `{stat_dir}/loss_weight_pp.pkl`** just to
satisfy the unconditional load — contents irrelevant since `loss_reweight`
stays `false`.

### Missing yaml attributes — pre-existing fragility, must add for CLAS12 config

Training loop references `self.params.rep_aaai`, `self.params.nexttoken`,
and `self.params.ablate_loss_scale` (lines 297-298, 326), but **none of these
three appear in `mamba_pretrain.yaml`'s `default` block.** This looks like a
pre-existing gap in the original repo (not introduced by us) that would throw
an `AttributeError` the moment training actually runs, unless `YParams`
silently tolerates missing keys (unconfirmed — TBD if worth checking, but
safer to just add them explicitly regardless).

**Required addition to our CLAS12 yaml config:**
```yaml
rep_aaai: false
nexttoken: false
ablate_loss_scale: false
```
(`nexttoken` only matters if `rep_aaai` is true, given the code's branching —
safe to set either way as long as `rep_aaai: false`.)

### YAML parameters requiring new CLAS12 values

| Parameter | TPC default | CLAS12 value | Rationale |
|---|---|---|---|
| `data_root` | NERSC path | `~/projects/PP_collision_clas12/data/mmap` (or wherever CLAS12 mmap data lives) | path update |
| `stat_dir` | NERSC path | local stats dir (for dummy loss_bin/loss_weight pickles) | path update |
| `checkpoint_dir` | NERSC path | local checkpoint dir | path update |
| `klen` | 30 | **1** | per "reduced space → next nearest prediction" decision |
| `order` | EPR | *(irrelevant — Voxelizer deleted)* | dead param once Voxelizer/orderdict removed |
| `voxelize` | true | **false** | HRS/box-partition removed entirely for CLAS12 |
| `space_filling_order` | false | **true** | now the only/primary ordering path |
| `space_filling_curve` | z | *(irrelevant — replaced by `clas12_band_hilbert_order`)* | our new function replaces this mechanism entirely, not a drop-in swap of the existing `z`/`hilbert` string flag |
| `len_chunk` | 512 | irrelevant either way | `chunk_training: false` already, and CLAS12 events (max ~40 pts) would never trigger chunking regardless |
| `batch_size` / `local_batch_size` | 256 / 16 | **TBD, likely much larger** | CLAS12 events are far smaller (≤40 pts vs TPC's thousands) — same GPU memory likely affords a much bigger batch. Not yet decided, flagged as a tuning opportunity. |

### Other observations, not requiring action

- `nleave: 1e10` in the real yaml (not `1e6`, the `TPCBatchDataset.__init__`
  default) — irrelevant either way since `set_simpler` (which uses `nleave`)
  is confirmed dead code (see Section A above).
- Three separate model-size presets exist (`mamba_5m`, `mamba2_5m`,
  `mamba_small`) — `mamba_small` (embed_dim=128, 6 layers, ~1.5M params) may
  be a more proportionate starting point for CLAS12's much simpler/smaller
  data than the default `mamba_5m`, though this is a tuning choice, not a
  correctness requirement.

---

## Section C: Repo-wide usage audit (`embed.py`, `mambagpt.py`, `Voxelizer`, z_order/hilbert)

Repo-wide `grep` to confirm exactly what needs changing vs. what's confirmed
safe, beyond the dataset/training files already covered.

### `EmbedderPosOnly` — confirmed unused anywhere, requires real source change to use

```
grep -rn "EmbedderPosOnly" --include="*.py" .
```
Only matches its own class definition in `embed.py` — **no trainer, model, or
script anywhere instantiates it.** Using it for CLAS12 is not a config flag
flip; it requires modifying `Mamba1GPT.__init__` (`fm4npp/models/mambagpt.py`,
lines ~13-18 and the duplicate `Mamba1GPT` class definition at ~60-64) to add
a third branch:

```python
assert embed_method in ['concat', 'add']  # CURRENT
if embed_method == 'concat':
    Embedder = EmbedderConcat
else:
    Embedder = EmbedderAdd
```
**Required change:**
```python
assert embed_method in ['concat', 'add', 'pos_only']  # add third option
if embed_method == 'concat':
    Embedder = EmbedderConcat
elif embed_method == 'pos_only':
    Embedder = EmbedderPosOnly
else:
    Embedder = EmbedderAdd
```
Then set `embed_method: pos_only` in the CLAS12 yaml config (currently
`embed_method: add` in the TPC default).

**Also need to verify `EmbedderPosOnly.forward`'s exact column indexing**
(`embed.py`, not yet re-checked against a genuine 3-column no-E input — see
remaining gap below).

### `EmbedderPosOnly` — CONFIRMED BUGS, two fixes needed

**File:** `fm4npp/models/embed.py`, lines 68-76

```python
class EmbedderPosOnly(nn.Module):
    def __init__(self, pe_method, embed_dim, learnable_projection = False):
        super(EmbedderPosOnly, self).__init__()
        assert pe_method in ['none', 'ff', 'nerf', 'cpe']
        self.embed = CoordinateEmbedder(method = pe_method, n_continuous_dim = 3, target_dim = embed_dim, learnable_projection = learnable_projection)

    def forward(self, neighborhood):
        out = self.embed(neighborhood[..., 1:4])       
        return out
```

**Bug 1 — hardcoded column indexing assumes a 4th column exists.**
`neighborhood[..., 1:4]` assumes index 0 is energy to skip past, taking
indices 1,2,3 as the 3 spatial coords — same class of bug as the
`train_multi_gpu_mamba1.py` hardcoded `reshape(..., 4)` issue. With our
genuine 3-column `(eta, phi, r)` input, `neighborhood` only has indices 0,1,2
— `[..., 1:4]` would either index-error or silently return fewer columns than
`CoordinateEmbedder`'s `n_continuous_dim=3` expects, causing a downstream
shape mismatch.

**Required fix:**
```python
def forward(self, neighborhood):
    out = self.embed(neighborhood)  # all 3 columns directly, no skip
    return out
```

**Bug 2 — return signature mismatch.** `EmbedderAdd`/`EmbedderConcat` both
`return out, pos_embed` (a 2-tuple) — confirmed `Mamba1GPT.forward` consumes
this as `x, pos = self.embedder(x)` (two values expected). `EmbedderPosOnly`
as written `return out` (single value) — **this would break the unpacking in
`Mamba1GPT.forward`.**

**Required fix:**
```python
def forward(self, neighborhood):
    out = self.embed(neighborhood)
    return out, out  # pos_embed == out here, since position is the only signal
```
(`pos_embed` and `out` are identical for this embedder, since there's no
separate energy component to add/concat against — returning `out` twice
satisfies the 2-tuple contract `Mamba1GPT.forward` expects without changing
that calling code.)



```
grep -rn "embed_method" --include="*.py" .
```
Real usages found in: `point_classification_trainer.py`,
`track_finding_trainer.py`, `trackinghead.py`, `model.py` (all
`train/downstream/`), plus `linformer_gpt.py`/`longformer_gpt.py` (alternative
architectures, confirmed not core/included per REPO_SUMMARY.md). **None of
these are reachable from pretraining** — only `train_multi_gpu_mamba1.py`'s
construction of `Mamba1GPT` (`fm4npp/models/mambagpt.py`) matters for our
purposes. **Conclusion: only `mambagpt.py`'s `Mamba1GPT` class needs the
`pos_only` branch added — no downstream file needs touching.**

### `Voxelizer` — confirmed exactly 3 usages, all already known, all safe to delete

```
grep -rn "Voxelizer" --include="*.py" .
```
Only 3 real usages, one per dataset file (`dataset.py`, `dataset_pretrain.py`,
`dataset_eval.py`), each in the identical unconditional
`self.voxelizer = Voxelizer(...)` pattern already documented in Section 11.
**No hidden caller anywhere else in the repo.** Deletion from our CLAS12
`dataset_pretrain.py` confirmed complete and safe.

### Z-order / Hilbert key functions — confirmed no changes needed in `z_order.py` itself

```
grep -rln "z_order\|xyz2key\|key2xyz" --include="*.py" .
```
Only `z_order.py` (own definitions) and `utils.py` (imports to build
`encode`/`z_order_encode`/`hilbert_encode`) reference these — exactly as
expected, no hidden usage. **`z_order.py` requires zero changes** — our
CLAS12 approach (Option B) only ever calls `hilbert.py`'s `encode`, never
`z_order.py`'s functions directly.

---

## Section D: `fm4npp/utils.py` audit + exact replacement point confirmed

Read `utils.py` in full against the current checklist.

### `parse_mean_E` — confirmed fully dead, no action needed

```
grep -rn "parse_mean_E" --include="*.py" .
```
Only its own definition. Zero callers anywhere in the repo. Energy-specific
by name/purpose but irrelevant regardless since it's never invoked.

### `rescale_serialize_Rlast` / `utils.encode` — CONFIRMED exact replacement point in `__getitem__`

```
grep -rn "rescale_serialize_Rlast" --include="*.py" .
```
Confirms the precise call site we need to change:

**File:** `fm4npp/datasets/dataset_pretrain.py`, **line 501**, inside
`__getitem__`'s `if self.space_filling_order:` branch:
```python
_, zsorter = rescale_serialize_Rlast(norm_features, scaler = 1e4, order=self.space_filling_curve)
```

**This is the exact line to replace** with a call to our new
`clas12_band_hilbert_order` function (from `hilbert_clas12_addition.py`,
appended to `fm4npp/hilbert.py`). Required change:
```python
# Need to import clas12_band_hilbert_order at the top of dataset_pretrain.py
zsorter = clas12_band_hilbert_order(
    phi=norm_features[..., 1].squeeze(0),  # confirm exact column index once final column order is fixed
    eta=norm_features[..., 0].squeeze(0),  # confirm exact column index once final column order is fixed
    r=norm_features[..., 2].squeeze(0),
)
```
(Column indices above are placeholders pending final confirmation of
`norm_features`'s exact column order after the energy column is removed —
see Section 15's open sub-item on this.)

**Consequence: `rescale_serialize_Rlast` (and by extension `utils.py`'s
`encode`/`decode` wrapper functions it depends on) become dead code in our
CLAS12 fork once this swap is made** — not because they were always unused
(they were genuinely reachable in the original, just inactive at runtime
since `space_filling_order` defaults to `False`), but because we are actively
replacing their only call site. Confirmed via the same grep: `dataset.py` and
`dataset_eval.py` also define+call `rescale_serialize_Rlast` independently
(their own copies) — **those two files are NOT part of our CLAS12 pretrain
fork's scope** (they serve downstream/eval tasks we're not touching), so no
action needed there.

**Decision: safe to delete `rescale_serialize_Rlast` (both duplicate
definitions, lines 19 and 136) from our CLAS12 `dataset_pretrain.py`**, since
its only call site (line 501) is being replaced, not preserved.

### `reg_target` for CLAS12 — RESOLVED

Same approach as the earlier TPC work: zero-stub, since `reg_target` is
loaded unconditionally by `TPCBatchDataset.__init__` via `RaggedMmap` but
never actually read by the pretraining loss (confirmed both for TPC and
CLAS12 — `train_multi_gpu_mamba1.py`'s loss only ever uses `knearest`/
`klabel`, never `reg_target`). CLAS12's conversion script (see below) will
generate a `reg_target_pretrain`/`reg_target_test` RaggedMmap of zeros, same
per-event length as `features`/`seg_target`, shape `(n_points, 7)` matching
`particle_reg_cols`'s 7 entries — purely to satisfy the unconditional load,
contents irrelevant.

### CLAS12 ideal intermediate format — RESOLVED

Rather than have every script read raw BST/BMT CSVs directly (with their
inconsistent `sector` field, unused `region`/`c_layer`/`z_layer`/`err_*`
columns, and the awkward two-file split), convert once into a single ideal
intermediate file, then have everything downstream (RaggedMmap conversion,
any future validation/visualization) read from that one clean file.

**Format: single `.npz`, matching the existing Zenodo TPC convention exactly**
(same structure our own `convert_to_mmap.py` already consumes), chosen over
a simpler pickle-of-per-event-arrays alternative specifically for future-proofing:
event counts could scale into the millions, where npz's flat-array +
memory-mappable structure matters (avoids per-event Python object overhead
and full-deserialization cost a pickle of many small arrays would incur at
that scale).

- `data`: flat array, shape `(total_hits, 4)`, columns `[x0, y0, z0, trkID]`
- `size`: shape `(n_events,)`, hit-count per event (cumulative-sum gives
  per-event start/end indices into `data`, identical mechanism to
  `event_slices()` in the original TPC `convert_to_mmap.py`)

**Columns deliberately dropped** (present in raw CSVs, not needed anywhere
in our pipeline): `sector` (confirmed inconsistent width/location per layer,
unusable as a fixed geometric key), `region`/`c_layer`/`z_layer` (detector
bookkeeping — `assign_clas12_layer` recomputes layer membership from real
measured `r` directly, doesn't need these labels), `theta` (redundant — `eta`
and `phi` both derivable from `x,y,z` via `cartesian_to_polar_batched`,
same as TPC), BMT's `c_x0/c_y0/c_z0` and `z_x0/z_y0/z_z0` variants (only
plain `x0,y0,z0` used, per earlier decision), `err_x0/err_y0/err_z0`
(measurement uncertainty, unused anywhere downstream).

**`trkID` is kept** even though pretraining itself never reads it — valuable
for future validation (e.g., empirically checking real track-coherence of the
Hilbert ordering scheme, a test we discussed wanting to eventually run).

**One conversion script merges raw BST + BMT CSVs into this single npz once**
(handles the per-file region numbering, x0/y0/z0 extraction, event grouping);
everything else (the RaggedMmap converter, equivalent to TPC's
`convert_to_mmap.py`) reads only from this ideal npz, never touching the raw
CSVs again. Original raw CSVs remain untouched on disk throughout — nothing
is lost, this is purely an additive, reusable conversion step.

---