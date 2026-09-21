import numpy as np
import torch
from torch.utils.data import Dataset
from mmap_ninja import RaggedMmap
from dev_scripts.ragged_npy_reader import RaggedNpyReader
from pathlib import Path
import os
import glob
import torch.nn as nn

import torch
from fm4npp.utils import *
# NOTE (CLAS12): Voxelizer/HRS removed entirely — replaced by Hilbert band ordering.
# from .voxelizer import *   # (no longer used)
from fm4npp.hilbert import clas12_band_hilbert_order, assign_clas12_layer

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(42)

# NOTE (CLAS12): rescale_serialize_Rlast deleted — replaced by
# clas12_band_hilbert_order (imported from fm4npp.hilbert). It was the original
# TPC space-filling ordering, hardcoded to 4 columns and equal-axis priority.


def knn_later_indices_batch(A, k, r_col=2):
    """
    A: Tensor (B, N, 3) = (batch, points, [eta, phi, r]). r is column r_col (=2),
       in NORMALIZED units matching apply_norm's r_lim.
    k: number of prediction slots per point.

    [PHYSICS NEXT-BAND RULE, band-index based] Each point's radius is mapped
    to a discrete, CALIBRATED CVT layer band via assign_clas12_layer (loaded
    from dev_scripts/clas12_band_calibration.json -- see calibrate_clas12_bands.py).
    For prediction slot m (1-indexed, m=1..k), a neighbor j is a valid target
    for query i iff:
        band(j) >= band(i) + m
    i.e. slot 1 must be at least one real band further out, slot 2 at least
    two bands further out, etc. -- so successive slots are guaranteed to
    progress to strictly higher bands as m increases, not just be "the next
    k nearest points" from one shared threshold.

    Among the points satisfying slot m's band condition, the single nearest
    (full 3D distance) is kept for that slot. No valid candidate (e.g. query
    is within m-1 bands of the outermost layer) -> that slot is padded with
    -100, masked out by the loss the same way as before.

    [WHY band-index, not a radius threshold] A fixed radius threshold breaks
    down when within-band spread (e.g. SVT's stereo-angle-driven radius
    variation across one physical layer, up to ~0.15 cm) is comparable to or
    larger than the smallest real inter-band gap (e.g. the SVT U/V gap within
    one region, ~0.20 cm) -- a threshold can't distinguish "moved to a
    genuinely different layer" from "spread within my own layer". Comparing
    discrete band INDICES (from the calibrated boundaries) is immune to this,
    since band assignment already collapses intra-band spread away.

    Returns: (B, N, 3*k).
    """
    B, N, D = A.shape
    assert D == 3, "A must have shape (B, N, 3)"

    # 1) pairwise full-3D distances (B, N, N)
    A_expanded = A.unsqueeze(2)  # (B, N, 1, 3)
    A_tiled = A.unsqueeze(1)     # (B, 1, N, 3)
    pairwise_distances = torch.norm(A_expanded - A_tiled, dim=-1)  # (B, N, N)

    # 2) band index per point (torch.bucketize supports any input shape
    # against the 1D calibrated boundaries, so this runs on the full (B,N)
    # tensor directly -- no need to loop over the batch).
    r = A[..., r_col]                # (B, N)
    band = assign_clas12_layer(r)    # (B, N), long
    band_i = band.unsqueeze(2)       # (B, N, 1)
    band_j = band.unsqueeze(1)       # (B, 1, N)

    # 3) for each slot m=1..k, the nearest point with band_j >= band_i + m
    slot_idx = torch.full((B, N, k), -1, device=A.device, dtype=torch.long)
    for m in range(1, k + 1):
        valid_m = band_j >= (band_i + m)                    # (B, N, N)
        dist_m = pairwise_distances.masked_fill(~valid_m, float('inf'))
        nearest_val, nearest_idx = dist_m.min(dim=2)         # (B, N) each
        has_valid = ~torch.isinf(nearest_val)
        slot_idx[..., m - 1] = torch.where(
            has_valid, nearest_idx, torch.full_like(nearest_idx, -1)
        )

    # 4) gather coords, padding invalid slots with -100
    knn_neighbors = torch.full((B, N, k, D), -100, device=A.device, dtype=A.dtype)
    safe_idx = slot_idx.clone()
    safe_idx[safe_idx < 0] = 0
    valid_mask = (slot_idx >= 0)
    b_idx = torch.arange(B, device=A.device).view(B, 1, 1).expand(B, N, k)
    knn_neighbors[valid_mask] = A[b_idx[valid_mask], safe_idx[valid_mask], :]

    # 5) reshape to (B, N, 3*k)
    knn_neighbors = knn_neighbors.view(B, N, 3 * k)
    return knn_neighbors


# NOTE (CLAS12): The following confirmed-dead functions were deleted here
# (repo-wide grep showed zero callers anywhere): swap_dim, strip_masked,
# rescale_serialize_Rlast (2nd duplicate), serialize_neighbors, Group,
# rescale_polar_radius, module-level minmax_normalize, module-level apply_norm,
# group_points, set_simpler.

class TPCBatchDataset(Dataset):
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
                 data_fraction = 1.0,
                 reader_type = 'ragged_mmap',
                 voxelize = True,
                 space_filling_order = None,
                 space_filling_curve = 'z',
                 band_classification = False,
                 bin_dir = '',
                 use_aux_features = False,
                 aux_extra_feature = 'length'):
        
        split = split
        self.band_classification = band_classification
        # [READER TOGGLE] 'ragged_mmap' = real mmap_ninja.RaggedMmap (existing data/mmap).
        # 'ragged_npy' = Alessio's values.npy+offsets.npy format (e.g. data/mmap_v4),
        # which is NOT a real RaggedMmap despite similar structure -- see each
        # folder's WARNING_NOT_RAGGEDMMAP.txt. Default preserves current behavior.
        reader_cls = RaggedNpyReader if reader_type == 'ragged_npy' else RaggedMmap
        self.memmap_feature = reader_cls(os.path.join(data_root, 'features_{}'.format(split)))
        self.memmap_seg_target = reader_cls(os.path.join(data_root, 'seg_target_{}'.format(split)))
        self.memmap_reg_target = reader_cls(os.path.join(data_root, 'reg_target_{}'.format(split)))

        # [NEW] Optional auxiliary per-point geometric features (measurement
        # direction sx,sy,sz + strip length), from cluster_geometry_context_target.
        # Off by default -- existing behavior/callers are completely unaffected.
        # Layout of that bank: [x1,y1,z1,x2,y2,z2,sx,sy,sz,physical_length_cm,pitch_cm].
        # We only use sx,sy,sz (cols 6:9) and physical_length_cm (col 9); endpoints
        # and pitch are intentionally not used (see team design discussion).
        self.use_aux_features = use_aux_features
        # [NEW] which scalar(s) beyond direction to include: 'length' (along-
        # strip physical length, the unmeasured-axis uncertainty bound),
        # 'pitch' (strip pitch, the measured-axis precision -- what sigma
        # was meant to be, sidestepping the fact that the bank's own `e`
        # field is confirmed always-zero for SVT), or 'both'.
        assert aux_extra_feature in ('length', 'pitch', 'both')
        self.aux_extra_feature = aux_extra_feature
        if self.use_aux_features:
            self.memmap_aux_geom = reader_cls(os.path.join(data_root, 'cluster_geometry_context_target_{}'.format(split)))
        

        self.reco_cols = ['x', 'y', 'z']   # CLAS12: no energy
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
        
        # for normalization (CLAS12 values — see CLAS12_CHANGES.md)
        self.eta_lim = {'min':-2.5, 'max':1.5}
        self.phi_lim = {'min':-torch.pi, 'max':torch.pi}
        self.r_lim = {'min': 6.0, 'max': 23.0}
        # [NEW] physical_length_cm normalization range -- from real measured
        # data on mmap_canonical_loose_truthseg_event_v7_01 (200-event sample):
        # observed min=0.3908, max=44.4850, mean=35.6822 cm. Only used when
        # use_aux_features=True. sx,sy,sz need no normalization (already
        # unit-vector components).
        self.length_lim = {'min': 0.0, 'max': 45.0}
        # [NEW] pitch_cm normalization range -- from real measured data on
        # v7_01 (500-event sample): observed min=0.0156, max=0.0860,
        # mean=0.0325 cm, 0% zeros (confirmed populated for both BMT and
        # BST, unlike the bank's own `e` field which is always zero for SVT).
        self.pitch_lim = {'min': 0.0, 'max': 0.10}
        # NOTE (CLAS12): E_mean/E_std deleted — no energy channel.
        # NOTE (CLAS12): orderdict / dim_sweep_order / revert_order deleted —
        # those fed the Voxelizer/HRS box ordering, which is removed entirely.

        self.low_thr = 1   # OPEN FILTER (single-track test): was 5
        self.normalize = normalize
        
        # Tokenizer
        self.group_size = group_size
        self.normalize_by_center = normalize_by_center
        # NOTE (CLAS12): self.voxelizer deleted — ordering now via
        # clas12_band_hilbert_order in __getitem__.
        self.limit_data = limit_data
        self.limit_size = limit_size
        self.data_fraction = data_fraction
        self.len_chunk = len_chunk
        
        self.train = train
        self.chunk_training = chunk_training
        self.filter_data(low_thr = 1, high_thr = 50)   # capped at 50 to bound O(N^2) kNNN memory cost -- keeps 96.7% of events, cuts worst-case N^2 20x (was 100, see check_event_length_dist.py)
        import math
        self.data_scaler = 1 # [TOGGLE][TEMPORARY] SCALER
        
    def znormalize(self, arr, mean_, std_):
        """z-normalize"""
        return (arr - mean_) / std_
    
    def z_unnormalize(self, arr, mean_, std_):        
        return arr*std_ + mean_
    
    def minmax_normalize(self, arr, max_, min_):
        """Normalize between -1 and 1"""
        return (arr - min_) / (max_ - min_)
    
    def minmax_unnormalize(self, arr, max_, min_):
        return arr * (max_ - min_) + min_       
    
    def apply_norm(self, features):
        # CLAS12: 3 columns [eta, phi, r], no energy.
        fnorm = features.clone()
        fnorm[..., 0] = self.minmax_normalize(fnorm[..., 0], self.eta_lim['max'], self.eta_lim['min'])
        fnorm[..., 1] = self.minmax_normalize(fnorm[..., 1], self.phi_lim['max'], self.phi_lim['min'])
        fnorm[..., 2] = self.minmax_normalize(fnorm[..., 2], self.r_lim['max'], self.r_lim['min'])
        return fnorm
    
    def apply_unnorm(self, features):
        # CLAS12: 3 columns [eta, phi, r], no energy.
        fnorm = features.clone()
        fnorm[..., 0] = self.minmax_unnormalize(fnorm[..., 0], self.eta_lim['max'], self.eta_lim['min'])
        fnorm[..., 1] = self.minmax_unnormalize(fnorm[..., 1], self.phi_lim['max'], self.phi_lim['min'])
        fnorm[..., 2] = self.minmax_unnormalize(fnorm[..., 2], self.r_lim['max'], self.r_lim['min'])
        return fnorm
    
    def filter_data(self, low_thr = -1, high_thr = 10e10):
        self.idxlist = []
        self.seqlens = []
        self.tooshort = []
        self.toolong = []
        self.longest = 0
        self.shortest = 1e10

        # [FAST FILTER] If the reader exposes lengths() (RaggedNpyReader), get all
        # event lengths instantly from offsets -- avoids millions of per-event
        # data reads at startup on large datasets. Falls back to the original
        # per-event loop for readers without it (real RaggedMmap).
        if hasattr(self.memmap_feature, 'lengths'):
            import numpy as np
            lens = np.asarray(self.memmap_feature.lengths())
            keep = (lens >= low_thr) & (lens <= high_thr)
            self.tooshort = np.where(lens < low_thr)[0].tolist()
            self.toolong = np.where(lens > high_thr)[0].tolist()
            self.idxlist = np.where(keep)[0].tolist()
            self.seqlens = lens[keep].tolist()
            if self.limit_data and len(self.idxlist) > self.limit_size:
                self.idxlist = self.idxlist[:self.limit_size]
                self.seqlens = self.seqlens[:self.limit_size]
            if self.seqlens:
                self.longest = int(max(self.seqlens))
                self.shortest = int(min(self.seqlens))
        else:
            for i in range(len(self.memmap_feature)):
                len_ = self.memmap_feature[i].shape[0]
                if len_ < low_thr:
                    self.tooshort.append(i)
                elif len_ > high_thr:
                    self.toolong.append(i)
                else:
                    self.idxlist.append(i)
                    self.seqlens.append(len_)

                    if self.longest < len_:
                        self.longest = len_
                    if self.shortest > len_:
                        self.shortest = len_


                if self.limit_data and len(self.idxlist) == self.limit_size: 
                    break

        # self.idxlist = create_sampled_lists_with_seq(self.idxlist, self.seqlens)

        # [DATA FRACTION] keep only the first `data_fraction` of the filtered events.
        # Applied AFTER filtering so the fraction is of usable events; deterministic
        # (first N) for reproducibility. Default 1.0 leaves everything unchanged.
        # True count used is always len(self.idxlist).
        if getattr(self, 'data_fraction', 1.0) < 1.0:
            n_total = len(self.idxlist)
            n_keep = max(1, int(round(n_total * self.data_fraction)))
            self.idxlist = self.idxlist[:n_keep]
            self.seqlens = self.seqlens[:n_keep]
            print('[INFO] data_fraction={}: using {}/{} filtered events'.format(
                self.data_fraction, n_keep, n_total))

        print('[INFO] Filtering by N points. From {}, removed short {} long {}, remaining {}'.format(len(self.memmap_feature),
                                                                                                     len(self.tooshort),
                                                                                                     len(self.toolong),
                                                                                                     len(self.idxlist)))
        print('[INFO] Shortest: {}, Longest: {}'.format(self.shortest, self.longest))

        
        
        if not self.train and self.chunk_training:
            self.idxlist_chunking = []
            for k, idx in enumerate(self.idxlist):
                seqlen = self.seqlens[k]
                start_indices = get_chunk_start_indices(self.len_chunk, seqlen)
                for sidx in start_indices:
                    if seqlen - sidx > self.low_thr: # minimum multiplicity at 50 points.
                        self.idxlist_chunking.append((idx, sidx))
                    
            print('[INFO] Chunking the validation set. Original {} -> Chunk all {}'.format(len(self.idxlist), len(self.idxlist_chunking)))
        
    def cut_chunk(self, sequence, maxlen):
        """
        Apply chunk-based training. 
        If seq_len > maxlen, cut a sub-chunk from a random location.
        If the seq_len <= maxlen, return as it is.
        """
        N, D = sequence.shape
        start_idx = 0
        
        if maxlen > N:
            return sequence, start_idx
        
        else:
            # Select a random starting position
            start_idx = torch.randint(0, N - self.low_thr + 1, (1,)).item()
            
            # Slice out the chunk
            chunk = sequence[start_idx : start_idx + maxlen]
            return chunk, start_idx
        
        
    def __len__(self):
        if not self.train and self.chunk_training:
            return len(self.idxlist_chunking)   
        else:
            return len(self.idxlist)    
    
    def __getitem__(self, index):
        
        if not self.train and self.chunk_training:
            real_idx, start_idx = self.idxlist_chunking[index]
        else:
            real_idx = self.idxlist[index]
            
        features = torch.from_numpy(np.copy(self.memmap_feature[real_idx])).unsqueeze(0)
        target = torch.from_numpy(np.copy(self.memmap_seg_target[real_idx])).unsqueeze(0)

        # [NEW] Load auxiliary geometric features (sx,sy,sz,length), same
        # real_idx as position/target -- one row per point, same point order,
        # so this stays aligned with features/target through every reorder
        # step below. NOT run through cartesian_to_polar_batched or
        # apply_norm -- those are position-specific (eta/phi/r decomposition
        # doesn't make physical sense for a direction vector or a scalar
        # length). Handled as its own separate, parallel path instead.
        #
        # sx,sy,sz is projected onto each point's own LOCAL (radial,
        # tangential, beam-axis) frame -- NOT used as raw global Cartesian.
        # A raw global direction vector entangles "what direction" with
        # "where in the detector": the same physical direction (e.g. purely
        # radial) produces a different (sx,sy,sz) depending on phi, forcing
        # the network to relearn this position-dependence from scratch. This
        # is the same principle behind "Local Reference Frame" (LRF) methods
        # in point cloud learning (e.g. rotation-invariant point cloud
        # descriptors) -- projecting onto a local frame makes the same
        # physical direction type produce the same numbers everywhere.
        if self.use_aux_features:
            aux_raw = torch.from_numpy(np.copy(self.memmap_aux_geom[real_idx])).unsqueeze(0).float()
            aux_sxyz = aux_raw[..., 6:9]      # (1, N, 3) raw global Cartesian direction
            aux_length = aux_raw[..., 9:10]   # (1, N, 1) physical_length_cm, raw
            aux_pitch = aux_raw[..., 10:11]   # (1, N, 1) pitch_cm, raw

            x_raw, y_raw = features[..., 0], features[..., 1]  # raw (x, y), same points/order as aux_sxyz
            rho = torch.sqrt(x_raw**2 + y_raw**2).clamp(min=1e-6)
            r_hat = torch.stack([x_raw / rho, y_raw / rho, torch.zeros_like(rho)], dim=-1)     # (1, N, 3)
            phi_hat = torch.stack([-y_raw / rho, x_raw / rho, torch.zeros_like(rho)], dim=-1)  # (1, N, 3)

            s_r = (aux_sxyz * r_hat).sum(dim=-1, keepdim=True)      # radial component
            s_phi = (aux_sxyz * phi_hat).sum(dim=-1, keepdim=True)  # tangential component
            s_z = aux_sxyz[..., 2:3]                                # beam-axis component (unchanged: z-hat is already (0,0,1) globally)

            if self.aux_extra_feature == 'length':
                extra = self.minmax_normalize(aux_length, self.length_lim['max'], self.length_lim['min'])
            elif self.aux_extra_feature == 'pitch':
                extra = self.minmax_normalize(aux_pitch, self.pitch_lim['max'], self.pitch_lim['min'])
            else:  # 'both'
                length_norm = self.minmax_normalize(aux_length, self.length_lim['max'], self.length_lim['min'])
                pitch_norm = self.minmax_normalize(aux_pitch, self.pitch_lim['max'], self.pitch_lim['min'])
                extra = torch.cat([length_norm, pitch_norm], dim=-1)  # (1, N, 2)

            aux_features = torch.cat([s_r, s_phi, s_z, extra], dim=-1)  # (1, N, 4) or (1, N, 5) for 'both'

        # print(features.shape, target.shape)
        if not self.train and self.chunk_training:
            features = features[:, start_idx : start_idx+self.len_chunk]
            target = target[:, start_idx : start_idx+self.len_chunk]
            if self.use_aux_features:
                aux_features = aux_features[:, start_idx : start_idx+self.len_chunk]
            # print(features.shape, target.shape)
            
        # features, target = set_simpler(features.unsqueeze(0), target.unsqueeze(0), nleave = self.nleave, npoint_lower_thr = self.npoint_lower_thr)
        
        ## To polar representation — CLAS12: input is 3-column (x,y,z), no energy.
        ## cartesian_to_polar_batched returns [eta, phi, r].
        polar_features = cartesian_to_polar_batched(features)
        # NOTE (CLAS12): no energy column to split off / re-concat.

        ## Normalize the polar representation -> [eta, phi, r] all in ~[0,1]
        if self.normalize:
            norm_features = self.apply_norm(polar_features)
        else:
            norm_features = polar_features
        
        # Sort by R (index -1 is still r in the 3-column [eta,phi,r] layout)
        ind = norm_features[...,-1].argsort(dim=1)
        norm_features = norm_features[:, ind.squeeze()]
        if self.use_aux_features:
            aux_features = aux_features[:, ind.squeeze()]
        # CLAS12: norm_features is already 3 columns [eta,phi,r] — pass directly,
        # no [..., 1:] slice (that would drop eta and leave only 2 columns).
        knearest_points = knn_later_indices_batch(norm_features, k=self.num_pred_points)
        norm_target = target[:, ind.squeeze()]
        
        # CLAS12 ordering: exact-layer-band radius grouping + 2D Hilbert over (phi, eta)
        # within each band. Replaces both the old space_filling (rescale_serialize_Rlast)
        # and voxelize (Voxelizer HRS) branches entirely.
        nf = norm_features.squeeze(0)  # (N, 3) = [eta, phi, r]
        zsorter = clas12_band_hilbert_order(
            phi=nf[:, 1],
            eta=nf[:, 0],
            r=nf[:, 2],
        )
        serialized_points = norm_features[:, zsorter.squeeze()].squeeze(0)
        knearest_points = knearest_points[:, zsorter.squeeze()].squeeze(0)
        serialized_target = norm_target[:, zsorter.squeeze()].squeeze(0)

        # [NEW] Apply the SAME final reorder to aux features, then
        # concatenate onto the position output: (N, 3) -> (N, 7) =
        # [eta, phi, r, sx, sy, sz, length]. This must happen after
        # knearest_points/serialized_target are computed above (kNNN
        # targets are position-only, unaffected by this addition) but
        # before serialized_points is returned.
        if self.use_aux_features:
            serialized_aux = aux_features[:, zsorter.squeeze()].squeeze(0)
            serialized_points = torch.cat([serialized_points, serialized_aux], dim=-1)

        # [BAND CLASSIFICATION TOGGLE] When enabled, replace the r component of
        # each kNNN neighbor target with its integer band index (0-5), so the
        # trainer can use cross-entropy on the band while keeping (eta, phi)
        # continuous. Padding (-100) is preserved as -100 (CE ignore_index).
        # knearest_points layout: (N, k*3) = [eta, phi, r] per neighbor.
        if self.band_classification:
            kn = knearest_points.reshape(-1, self.num_pred_points, 3)  # (N, k, 3)
            r_col = kn[..., 2]
            pad_mask = (r_col == -100)
            # assign_clas12_layer asserts on out-of-band values, so only call it
            # on real (non-padding) radii.
            if (~pad_mask).any():
                bands = assign_clas12_layer(r_col[~pad_mask]).to(kn.dtype)
                r_col = r_col.clone()
                r_col[~pad_mask] = bands
                kn = torch.cat([kn[..., :2], r_col.unsqueeze(-1)], dim=-1)
            knearest_points = kn.reshape(-1, self.num_pred_points * 3)
            # Band indices and -100 padding must NOT be scaled; return target unscaled.
            # (eta/phi components also unscaled here — trainer treats this target
            # as-is; data_scaler is 1 in all current configs anyway.)
            return serialized_points * self.data_scaler, serialized_target, knearest_points

        return serialized_points * self.data_scaler, serialized_target, knearest_points * self.data_scaler


class MyCollator(object):
    def __init__(self):
        pass
        
    def __call__(self, batch):
        """
        Batchify data considering original point level input and center-level input
        pair1: features / target at original (after minor filtering)
        pair2: centers / neighs after centering and knn
        mask: masking the variable number of centers.
        """

        # Getting the longest point
        point_longest = 0
        for g, t, k in batch:
            if point_longest < g.size(0):
                point_longest = g.size(0)
        
        grouped,targets,knearest= [], [], []
        
        pad_val = -100
        glengths = []
        for g, t, k in batch:
            grouped.append(torch.nn.functional.pad(g, (0, 0, 0, point_longest - g.size(0)), value = pad_val))    
            targets.append(torch.nn.functional.pad(t, (0, point_longest - g.size(0)), value = pad_val))
            knearest.append(torch.nn.functional.pad(k, (0, 0, 0, point_longest - g.size(0)), value = pad_val))
       
        grouped = torch.stack(grouped)
        targets = torch.stack(targets)
        knearest = torch.stack(knearest)
            
        return (grouped, targets, knearest)



def get_data_loader(params, distributed):

    train_dataset = TPCBatchDataset(data_root = params.data_root, 
                                    version = params.data_version, 
                                    split = 'pretrain', 
                                    group_size = params.group_size, 
                                    normalize = True, 
                                    limit_data = params.limit_data, 
                                    limit_size = params.limit_size, 
                                    data_fraction = getattr(params, 'data_fraction', 1.0), 
                                    reader_type = getattr(params, 'reader_type', 'ragged_mmap'), 
                                    nleave = params.nleave, 
                                    order = params.order, 
                                    num_pred_points = params.klen, 
                                    len_chunk = params.len_chunk,
                                    chunk_training = params.chunk_training,
                                    voxelize = params.voxelize,
                                    bin_dir = params.stat_dir,
                                    space_filling_order = params.space_filling_order,
                                    space_filling_curve = params.space_filling_curve,
                                    band_classification = getattr(params, 'band_classification', False),
                                    use_aux_features = getattr(params, 'use_aux_features', False),
                                    aux_extra_feature = getattr(params, 'aux_extra_feature', 'length'),
                                    train = True)
    
    test_dataset = TPCBatchDataset(data_root = params.data_root, 
                                   version = params.data_version, 
                                   split = 'test', 
                                   num_pred_points = params.klen,
                                   group_size = params.group_size, 
                                   normalize = True, 
                                   nleave = params.nleave, 
                                   chunk_training = params.chunk_training,
                                   bin_dir = params.stat_dir,
                                   voxelize = params.voxelize,
                                   order = params.order,
                                   space_filling_order = params.space_filling_order,
                                   space_filling_curve = params.space_filling_curve,
                                   band_classification = getattr(params, 'band_classification', False),
                                   use_aux_features = getattr(params, 'use_aux_features', False),
                                   aux_extra_feature = getattr(params, 'aux_extra_feature', 'length'),
                                   reader_type = getattr(params, 'reader_type', 'ragged_mmap'), 
                                   train = False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    my_collate_fn = MyCollator()
    
    train_dataloader = DataLoader(train_dataset,
                            batch_size=int(params.local_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=(train_sampler is None),
                            sampler=train_sampler,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            collate_fn = my_collate_fn)
    
    test_dataloader = DataLoader(test_dataset,
                            batch_size=int(params.local_valid_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,
                            sampler=test_sampler,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            collate_fn = my_collate_fn)
    
    return train_dataloader, train_sampler, test_dataloader, test_sampler

def get_val_loader(params, distributed):

    test_dataset = TPCBatchDataset(data_root = params.data_root, 
                                   version = params.data_version, 
                                   split = 'test', 
                                   num_pred_points = params.klen,
                                   group_size = params.group_size, 
                                   normalize = True, 
                                   nleave = params.nleave, 
                                   chunk_training = params.chunk_training,
                                   train = False,
                                   order = params.order,
                                   band_classification = getattr(params, 'band_classification', False),
                                   use_aux_features = getattr(params, 'use_aux_features', False),
                                   aux_extra_feature = getattr(params, 'aux_extra_feature', 'length'),
                                   reader_type = getattr(params, 'reader_type', 'ragged_mmap'),)

   
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    my_collate_fn = MyCollator()
    
        
    test_dataloader = DataLoader(test_dataset,
                            batch_size=int(params.local_valid_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,
                            sampler=test_sampler,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            collate_fn = my_collate_fn)
    
    return test_dataloader