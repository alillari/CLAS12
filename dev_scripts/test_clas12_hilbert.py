"""
Smoke test for clas12_band_hilbert_order — run INSIDE the EdgeXpert container
(which has torch). From the repo root inside the container:

    python3 test_clas12_hilbert.py

Confirms:
  1. assign_clas12_layer correctly bins 18 points (3 particles x 6 layers)
     into layer indices 0..5.
  2. clas12_band_hilbert_order produces a permutation that groups points
     strictly by layer band (0,0,0,1,1,1,...,5,5,5 after sorting).
  3. The out-of-band assertion fires when a point has an impossible radius.
"""

import torch
from fm4npp.hilbert import (
    clas12_band_hilbert_order,
    assign_clas12_layer,
    CLAS12_LAYER_RADII_RAW,
    CLAS12_LAYER_RADII,
    CLAS12_LAYER_DELTA,
    _normalize_r,
)

print("=== Constants ===")
print("Raw layer radii:      ", CLAS12_LAYER_RADII_RAW)
print("Normalized radii:     ", [round(r, 5) for r in CLAS12_LAYER_RADII])
print("Normalized delta:     ", round(CLAS12_LAYER_DELTA, 6))
print()

# --- Build a simulated event: 3 particles x 6 layers = 18 points ---
# Each particle contributes one hit at each of the 6 layer radii.
raw_radii = CLAS12_LAYER_RADII_RAW * 3                      # 18 raw radii
r_norm = torch.tensor([_normalize_r(r) for r in raw_radii], dtype=torch.float32)

torch.manual_seed(0)
phi_norm = torch.rand(18)   # simulate already-normalized [0,1] phi
eta_norm = torch.rand(18)   # simulate already-normalized [0,1] eta

# --- Test 1: layer assignment ---
print("=== Test 1: assign_clas12_layer ===")
layers = assign_clas12_layer(r_norm)
print("Assigned layers:", layers.tolist())
expected_layers = sorted(list(range(6)) * 3)
ok1 = sorted(layers.tolist()) == expected_layers
print("PASS" if ok1 else "FAIL", "- 18 points map to layers 0-5, 3 each")
print()

# --- Test 2: full ordering groups by band ---
print("=== Test 2: clas12_band_hilbert_order ===")
sorter = clas12_band_hilbert_order(phi=phi_norm, eta=eta_norm, r=r_norm)
ordered_layers = layers[sorter].tolist()
print("Layer order after sort:", ordered_layers)
ok2 = ordered_layers == sorted(ordered_layers)
print("PASS" if ok2 else "FAIL", "- sorted output is grouped/monotonic by layer band")
print()

# --- Test 3: out-of-band assertion fires ---
print("=== Test 3: out-of-band assertion ===")
bad_r = r_norm.clone()
bad_r[0] = 0.5  # a normalized radius far from any real layer band
try:
    assign_clas12_layer(bad_r)
    print("FAIL - expected an assertion for an out-of-band radius, none raised")
except AssertionError as e:
    print("PASS - assertion correctly fired for out-of-band radius")
print()

print("=== Summary ===")
print("All passed!" if (ok1 and ok2) else "SOMETHING FAILED — see above")