"""
Mike Frank's single-neighbor geometric pretraining target for CLAS12 CVT.

For cluster i, among clusters in later layers: the nearest layer that has a
compatible cluster, and in it the cluster with the smallest
    chi2 = dphi^2 / (s_phi(i)^2 + s_phi(j)^2 + s_bend^2)
         + dtheta^2 / (s_theta(i)^2 + s_theta(j)^2 + s_vtx^2).
Compatible: chi2 < CHI2_CUT (3 sigma, 2 degrees of freedom). No compatible
cluster anywhere: the smallest chi2 overall. No later cluster: -1.

The layer dimensions below are pinned to CCDB run 11, rga_fall2018_mc.
The bend and vertex terms are geometric allowances. The score is a ranking
heuristic, not a fitted probability or a track association.
  SVT  /geometry/cvt/svt/svt: pitch 0.156 mm, 256 strips graded 0-3 deg, 3 sensors of
       109.955 mm, dead zone 0.835 mm -> strip length 333.4 mm (SVTConstants.STRIPLENMAX).
       A single layer does not measure z; z unknown along the strip also smears phi
       through the strip angle.
  BMT  /geometry/cvt/mvt/bmt_layer_noshim, bmt_strip_L1..L6: Z layers measure phi
       (pitch / sqrt 12), z unknown over the tile; C layers measure z (per strip group),
       phi unknown over the tile.
  s_bend  a track reaching the outer BMT layer has curvature radius >= r_max / 2.
  s_vtx   vertex anywhere along the target (RGA LH2 cell, 5 cm centred at z = -3 cm;
          not from the CVT tables).
Positions in cm.
"""
import math

import numpy as np
import torch

from fm4npp.hilbert import assign_clas12_layer

MM = 0.1
CHI2_CUT = 11.83
SVT_STRIP_LEN = (3 * 109.955 + 4 * 0.835 + 2 * 0.112) * MM
SVT_STEREO = math.radians(3.0)
SVT_PITCH = 0.156 * MM
BMT = {7: ("C", 146.146, -134.07, 238.25, 34.856, 145.144),
       8: ("Z", 161.146, -135.77, 236.93, 34.402, 145.598),
       9: ("Z", 176.146, -156.69, 264.25, 34.027, 145.973),
       10: ("C", 191.146, -156.23, 264.25, 33.710, 146.290),
       11: ("Z", 206.146, -177.60, 267.25, 33.439, 146.561),
       12: ("C", 221.146, -177.55, 267.25, 33.206, 146.794)}
BMT_STRIPS = {
    7: [(32, .505), (32, .44), (32, .385), (32, .335), (512, .33), (32, .37), (32, .42), (32, .47), (32, .53),
        (32, .6), (32, .675), (32, .765), (32, .86)],
    8: [(640, .487)], 9: [(640, .536)],
    10: [(32, .52), (32, .475), (32, .42), (32, .37), (608, .34), (32, .36), (32, .395), (32, .435), (32, .48),
         (32, .53), (32, .58), (32, .64), (32, .705), (32, .77)],
    11: [(768, .529)],
    12: [(32, .54), (32, .48), (32, .43), (32, .39), (704, .33), (64, .34), (32, .38), (32, .41), (32, .45),
         (32, .49), (32, .53), (32, .57), (32, .62), (32, .67)]}
R_MIN_CURV = BMT[12][1] * MM / 2
TARGET_CENTER, TARGET_LEN = -3.0, 5.0


def layer_sigmas():
    """Per 0-based layer: (sigma along z, sigma of r*phi), cm."""
    sz, srphi = np.zeros(12), np.zeros(12)
    rms_tan = math.tan(SVT_STEREO) / math.sqrt(3)
    for L in range(6):
        sz[L] = SVT_STRIP_LEN / math.sqrt(12)
        srphi[L] = math.hypot(SVT_PITCH / math.sqrt(12), sz[L] * rms_tan)
    for L1, (axis, r, zmin, zmax, pmin, pmax) in BMT.items():
        p = np.array([q for n, q in BMT_STRIPS[L1] for _ in range(n)]) * MM
        meas = math.sqrt(float(np.mean(p ** 2)) / 12)
        if axis == "Z":
            srphi[L1 - 1], sz[L1 - 1] = meas, (zmax - zmin) * MM / math.sqrt(12)
        else:
            sz[L1 - 1], srphi[L1 - 1] = meas, r * MM * math.radians(pmax - pmin) / math.sqrt(12)
    return sz, srphi


SZ, SRPHI = (torch.tensor(a, dtype=torch.float64) for a in layer_sigmas())


def geometric_target_indices(xyz, layer=None):
    """xyz (N, 3) in cm -> (N,) index of each cluster's target, -1 if no later cluster.
    layer: (N,) 0-based layers; computed from r if not given."""
    if xyz.size(0) == 0:
        return xyz.new_empty((0,), dtype=torch.long)
    xyz = xyz.to(torch.float64)
    r = torch.hypot(xyz[:, 0], xyz[:, 1])
    if layer is None:
        layer = assign_clas12_layer((r - 6.0) / 17.0)
    layer = layer.long()
    zc = xyz[:, 2] - TARGET_CENTER
    phi = torch.atan2(xyz[:, 1], xyz[:, 0])
    th = torch.atan2(r, zc)
    s2 = torch.sin(th) ** 2
    sphi = SRPHI.to(xyz.device)[layer] / r
    sth = SZ.to(xyz.device)[layer] * s2 / r
    d = phi[None, :] - phi[:, None]
    dphi = torch.atan2(torch.sin(d), torch.cos(d))
    dth = th[None, :] - th[:, None]
    dr = (r[None, :] - r[:, None]).abs()
    s_bend = dr / (2 * R_MIN_CURV) / math.sqrt(3)
    s_vtx = (TARGET_LEN / math.sqrt(12)) * s2[:, None] * (1 / r[:, None] - 1 / r[None, :]).abs()
    var_phi = sphi[:, None].square() + sphi[None, :].square() + s_bend.square()
    var_theta = sth[:, None].square() + sth[None, :].square() + s_vtx.square()
    chi2 = dphi.square() / var_phi + dth.square() / var_theta
    later = layer[None, :] > layer[:, None]
    chi2 = chi2.masked_fill(~later, float("inf"))
    compatible = chi2 < CHI2_CUT
    no_layer = len(SZ)
    first = torch.where(compatible, layer[None, :], no_layer).amin(dim=1)
    in_first = compatible & (layer[None, :] == first[:, None])
    nearest_compatible = chi2.masked_fill(~in_first, float("inf")).argmin(dim=1)
    selected = torch.where(first < no_layer, nearest_compatible, chi2.argmin(dim=1))
    return torch.where(later.any(dim=1), selected, -1)
