"""
Hilbert Order
Modified from https://github.com/PrincetonLIPS/numpy-hilbert-curve

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Kaixin Xu
Please cite our work if the code is helpful to you.
"""

import torch


def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     k: The number of bits to shift. Default 1.

     axis: The axis along which to shift.  Default -1.

    Returns:
    --------
     Returns an ndarray with zero prepended and the ends truncated, along
     whatever axis was specified."""

    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # Determine the padding pattern.
    # padding = [(0,0)] * len(binary.shape)
    # padding[axis] = (k,0)

    # Determine the slicing pattern to eliminate just the last one.
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def binary2gray(binary, axis=-1):
    """Convert an array of binary values into Gray codes.

    This uses the classic X ^ (X >> 1) trick to compute the Gray code.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     axis: The axis along which to compute the gray code. Default=-1.

    Returns:
    --------
     Returns an ndarray of Gray codes.
    """
    shifted = right_shift(binary, axis=axis)

    # Do the X ^ (X >> 1) trick.
    gray = torch.logical_xor(binary, shifted)

    return gray


def gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values.

    Parameters:
    -----------
     gray: An ndarray of gray codes.

     axis: The axis along which to perform Gray decoding. Default=-1.

    Returns:
    --------
     Returns an ndarray of binary values.
    """

    # Loop the log2(bits) number of times necessary, with shift and xor.
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def encode(locs, num_dims, num_bits):
    """Decode an array of locations in a hypercube into a Hilbert integer.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     locs - An ndarray of locations in a hypercube of num_dims dimensions, in
            which each dimension runs from 0 to 2**num_bits-1.  The shape can
            be arbitrary, as long as the last dimension of the same has size
            num_dims.

     num_dims - The dimensionality of the hypercube. Integer.

     num_bits - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of uint64 integers with the same shape as the
     input, excluding the last dimension, which needs to be num_dims.
    """

    # Keep around the original shape for later.
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            """
      The shape of locs was surprising in that the last dimension was of size
      %d, but num_dims=%d.  These need to be equal.
      """
            % (orig_shape[-1], num_dims)
        )

    if num_dims * num_bits > 63:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a int64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits, num_dims * num_bits)
        )

    # Treat the location integers as 64-bit unsigned and then split them up into
    # a sequence of uint8s.  Preserve the association by dimension.
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    # Now turn these into bits and truncate to num_bits.
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    for bit in range(0, num_bits):
        # Iterate forwards through the dimensions.
        for dim in range(0, num_dims):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Now flatten out.
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    # Convert Gray back to binary.
    hh_bin = gray2binary(gray)

    # Pad back out to 64 bits.
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

    # Convert binary values into uint8s.
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )

    # Convert uint8s into uint64s.
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()

    return hh_uint64


def decode(hilberts, num_dims, num_bits):
    """Decode an array of Hilbert integers into locations in a hypercube.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     hilberts - An ndarray of Hilbert integers.  Must be an integer dtype and
                cannot have fewer bits than num_dims * num_bits.

     num_dims - The dimensionality of the hypercube. Integer.

     num_bits - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of unsigned integers with the same shape as hilberts
     but with an additional dimension of size num_dims.
    """

    if num_dims * num_bits > 64:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits)
        )

    # Handle the case where we got handed a naked integer.
    hilberts = torch.atleast_1d(hilberts)

    # Keep around the shape for later.
    orig_shape = hilberts.shape
    bitpack_mask = 2 ** torch.arange(0, 8).to(hilberts.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    # Treat each of the hilberts as a s equence of eight uint8.
    # This treats all of the inputs as uint64 and makes things uniform.
    hh_uint8 = (
        hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    )

    # Turn these lists of uints into lists of bits and then truncate to the size
    # we actually need for using Skilling's procedure.
    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[:, -num_dims * num_bits :]
    )

    # Take the sequence of bits and Gray-code it.
    gray = binary2gray(hh_bits)

    # There has got to be a better way to do this.
    # I could index them differently, but the eventual packbits likes it this way.
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

    # Iterate backwards through the bits.
    for bit in range(num_bits - 1, -1, -1):
        # Iterate backwards through the dimensions.
        for dim in range(num_dims - 1, -1, -1):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Pad back out to 64 bits.
    extra_dims = 64 - num_bits
    padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)

    # Now chop these up into blocks of 8.
    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))

    # Take those blocks and turn them unto uint8s.
    # from IPython import embed; embed()
    locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)

    # Finally, treat these as uint64s.
    flat_locs = locs_uint8.view(torch.int64)

    # Return them in the expected shape.
    return flat_locs.reshape((*orig_shape, num_dims))
# ============================================================
# CLAS12-specific addition: radius-band grouping + 2D Hilbert ordering.
#
# The default policy is the calibrated 12-band boundary table used by Mike's
# current pretraining. The older 6-band exact-center policy is retained under
# band_config="legacy6" for older backbones/configs.
# ============================================================

import json as _json
import os as _os


CLAS12_LEGACY6_LAYER_RADII_RAW = [
    6.52944,    # BST region 1
    9.28923,    # BST region 2
    12.03261,   # BST region 3
    14.76460,   # BMT region 1
    19.26460,   # BMT region 2
    22.26460,   # BMT region 3
]

# Single shared band half-width, chosen as the worst-case observed
# deviation across all 6 layers (BST region 1), rounded up for margin.
# Given in RAW (unnormalized) units.
CLAS12_LEGACY6_LAYER_DELTA_RAW = 0.3

# r_lim used by TPCBatchDataset.apply_norm for CLAS12 (see
# CLAS12_CHANGES.md "Normalization bounds" section) -- must match exactly,
# since assign_clas12_layer is called on r AFTER apply_norm has already run
# (confirmed: apply_norm executes before the space-filling-order branch in
# __getitem__). Keeping this constant here, alongside the layer radii, so
# the raw->normalized conversion below always stays consistent with
# whatever r_lim TPCBatchDataset actually uses.
CLAS12_R_LIM_MIN = 6.0
CLAS12_R_LIM_MAX = 23.0
CLAS12_DEFAULT_BAND_CONFIG = "calibrated12"


def _normalize_r(r_raw, r_min=CLAS12_R_LIM_MIN, r_max=CLAS12_R_LIM_MAX):
    """Same min-max formula as TPCBatchDataset.minmax_normalize."""
    return (r_raw - r_min) / (r_max - r_min)



CLAS12_LEGACY6_LAYER_RADII = [_normalize_r(r) for r in CLAS12_LEGACY6_LAYER_RADII_RAW]
CLAS12_LEGACY6_LAYER_DELTA = CLAS12_LEGACY6_LAYER_DELTA_RAW / (CLAS12_R_LIM_MAX - CLAS12_R_LIM_MIN)


def _calibration_path():
    return _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "..",
        "dev_scripts",
        "clas12_band_calibration.json",
    )


def load_clas12_band_calibration(path=None):
    path = path or _calibration_path()
    with open(path) as stream:
        cal = _json.load(stream)
    centers_raw = cal["band_centers_raw_cm"]
    boundaries_raw = cal["band_boundaries_raw_cm"]
    r_threshold_raw = cal["suggested_r_threshold_raw_cm"]
    return {
        "layer_radii_raw": centers_raw,
        "layer_radii": [_normalize_r(r) for r in centers_raw],
        "boundaries_raw": boundaries_raw,
        "boundaries": [_normalize_r(b) for b in boundaries_raw],
        "r_threshold_raw": r_threshold_raw,
        "r_threshold": r_threshold_raw / (CLAS12_R_LIM_MAX - CLAS12_R_LIM_MIN),
        "n_bands": len(centers_raw),
    }


_CLAS12_CALIBRATION = load_clas12_band_calibration()
CLAS12_LAYER_RADII_RAW = _CLAS12_CALIBRATION["layer_radii_raw"]
CLAS12_LAYER_RADII = _CLAS12_CALIBRATION["layer_radii"]
CLAS12_BAND_BOUNDARIES = _CLAS12_CALIBRATION["boundaries"]
CLAS12_KNNN_R_THRESHOLD = _CLAS12_CALIBRATION["r_threshold"]


def assign_clas12_layer_legacy6(
    r,
    layer_radii=CLAS12_LEGACY6_LAYER_RADII,
    delta=CLAS12_LEGACY6_LAYER_DELTA,
):
    layer_radii_t = torch.tensor(layer_radii, dtype=r.dtype, device=r.device)  # (L,)
    diffs = torch.abs(r.unsqueeze(-1) - layer_radii_t.unsqueeze(0))  # (N, L)
    in_band = diffs <= delta  # (N, L)

    n_bands_hit = in_band.sum(dim=-1)  # (N,)
    assert torch.all(n_bands_hit == 1), (
        "Found {} point(s) not matching exactly one CLAS12 layer band "
        "(0 or 2+ matches). This indicates corrupted input data -- "
        "check raw r values.".format((n_bands_hit != 1).sum().item())
    )

    layer_idx = in_band.float().argmax(dim=-1)  # (N,)
    return layer_idx


def assign_clas12_layer_calibrated12(r, boundaries=CLAS12_BAND_BOUNDARIES):
    boundaries_t = torch.tensor(boundaries, dtype=r.dtype, device=r.device)
    return torch.bucketize(r.contiguous(), boundaries_t)


def assign_clas12_layer(r, band_config=CLAS12_DEFAULT_BAND_CONFIG):
    if band_config in {"calibrated12", "12", "default"}:
        return assign_clas12_layer_calibrated12(r)
    if band_config in {"legacy6", "6"}:
        return assign_clas12_layer_legacy6(r)
    raise ValueError(f"Unsupported CLAS12 band_config={band_config!r}")


def clas12_band_hilbert_order(
    phi,
    eta,
    r,
    num_bits=10,
    scaler=1e4,
    band_config=CLAS12_DEFAULT_BAND_CONFIG,
):
    """
    Order CLAS12 points by: (1) exact layer band [guaranteed, zero
    cross-layer interleaving], then (2) 2D Hilbert curve over (phi, eta)
    within each band [locality-preserving angular tie-break].

    Params:
    -------
     phi, eta, r: 1D tensors of shape (N,), already in whatever normalized
                  range apply_norm produces (expected ~[0,1] each, but this
                  function only requires phi/eta to be non-negative after
                  scaling -- see note below).
     num_bits: bits of precision per axis for the 2D Hilbert encoding.
               10 bits = 1024 distinct values per axis, comfortably more
               resolution than needed for typical per-layer point counts.
     scaler: float-to-integer scaling factor applied to phi/eta before
             quantizing to the Hilbert hypercube. Must be large enough
             that distinct phi/eta values don't collapse to the same
             integer, but produce values < 2**num_bits after scaling.
     band_config: "calibrated12" (default) or "legacy6".

    Returns:
    --------
     sorter: 1D long tensor of shape (N,), the permutation that orders
             the input points as described above. Apply via
             tensor[sorter] to any (N, ...) array aligned with phi/eta/r.

    Note on phi/eta sign: phi and eta can be negative (e.g. phi in
    [-pi, pi]). Since encode() requires non-negative integer coordinates,
    this function shifts both by a fixed offset before scaling. If phi/eta
    are passed in already normalized to [0, 1] (i.e. apply_norm has already
    run), no offset is needed -- this is the expected calling convention.
    """
    N = phi.shape[0]
    assert eta.shape[0] == N and r.shape[0] == N

    layer_idx = assign_clas12_layer(r, band_config=band_config)  # (N,)

    # Quantize phi/eta to non-negative integers for the 2D Hilbert encode.
    # Expects phi, eta already normalized to [0, 1] (apply_norm convention) --
    # if not, shift to non-negative range before calling this function.
    locs = torch.stack([phi, eta], dim=-1)  # (N, 2)
    locs_int = (locs * scaler).long()
    assert torch.all(locs_int >= 0), (
        "phi/eta produced negative integers after scaling -- pass "
        "already-normalized [0,1] values, or add an offset before scaling."
    )
    max_val = 2 ** num_bits - 1
    locs_int = torch.clamp(locs_int, max=max_val)

    hilbert_codes = encode(locs_int, num_dims=2, num_bits=num_bits)  # (N,)

    # Composite sort key: layer dominates (exact, integer), Hilbert code
    # breaks ties within a layer. Pack into one key so a single argsort
    # produces the full two-level order in one pass.
    composite_key = layer_idx.long() * (2 ** 48) + hilbert_codes.long()
    sorter = torch.argsort(composite_key)
    return sorter
