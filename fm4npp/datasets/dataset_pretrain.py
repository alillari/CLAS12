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
# NOTE (CLAS12): Voxelizer/HRS removed entirely — replaced by Hilbert band ordering.
# from .voxelizer import *   # (no longer used)
from fm4npp.hilbert import clas12_band_hilbert_order, assign_clas12_layer

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(42)

# NOTE (CLAS12): rescale_serialize_Rlast deleted — replaced by
# clas12_band_hilbert_order (imported from fm4npp.hilbert). It was the original
# TPC space-filling ordering, hardcoded to 4 columns and equal-axis priority.


def knn_later_indices_batch(A, k):
    """
    A: Tensor of shape (B, N, 3), where B = batch size, N = number of points per batch, D=3 coordinates.
       Assumed to be sorted by the last dimension if needed, but sorting is not mandatory for the logic here.
    k: Number of neighbors to find for each point, using only indices j > i.
    
    Returns:
        Tensor of shape (B, N, 3*k):
          - For each batch b, row i, we gather up to k neighbors from rows j>i.
          - If fewer than k neighbors exist, the remainder is padded with -100.
    """
    B, N, D = A.shape
    assert D == 3, "A must have shape (B, N, 3)"

    # 1) Compute pairwise distances for each batch
    #    - shape: (B, N, N)
    #      * A_expanded: (B, N, 1, 3)
    #      * A_tiled:    (B, 1, N, 3)
    #      => difference => norm => (B, N, N)
    A_expanded = A.unsqueeze(2)  # (B, N, 1, 3)
    A_tiled = A.unsqueeze(1)     # (B, 1, N, 3)
    pairwise_distances = torch.norm(A_expanded - A_tiled, dim=-1)  # (B, N, N)

    # 2) Only allow neighbors with strictly larger index j>i
    #    => we set i>=j to infinity so they won't be selected
    #    Build a mask for the upper triangle above the diagonal (i < j).
    #    mask_2d shape: (N, N), then broadcast to (B, N, N).
    mask_2d = torch.triu(torch.ones(N, N, device=A.device), diagonal=1).bool()  # 1 for j>i
    mask_3d = mask_2d.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
    pairwise_distances[~mask_3d] = float('inf')       # i>=j => inf

    # 3) Use top-k to find the nearest neighbors among valid (finite) ones
    #    - topk(...) along dimension=2
    #    - largest=False => we want the smallest distances
    #    * topk_vals: (B, N, k_limited)
    #    * topk_idx : (B, N, k_limited)
    #    where k_limited = min(k, N-1)
    k_limited = min(k, N-1)
    topk_vals, topk_idx = torch.topk(
        pairwise_distances, 
        k=k_limited,
        dim=2,  # neighbor dimension
        largest=False
    )  # shapes: (B, N, k_limited), (B, N, k_limited)

    # 4) If the user-specified k > k_limited, pad with inf/-1 to get final shape (B, N, k)
    if k_limited < k:
        pad_size = k - k_limited
        inf_pad = torch.full((B, N, pad_size), float('inf'), device=A.device)
        minus1_pad = torch.full((B, N, pad_size), -1, device=A.device, dtype=torch.long)

        topk_vals = torch.cat([topk_vals, inf_pad], dim=2)    # (B, N, k)
        topk_idx  = torch.cat([topk_idx,  minus1_pad], dim=2) # (B, N, k)

    # 5) Convert any inf distances to invalid => set index = -1
    inf_mask = torch.isinf(topk_vals)  # (B, N, k)
    topk_idx[inf_mask] = -1

    # 6) We now gather the actual coordinates for these neighbor indices
    #    - Create an output array full of -100 for padding
    knn_neighbors = torch.full((B, N, k, D), -100, device=A.device, dtype=A.dtype)  # (B, N, k, 3)

    # 6a) Build a "safe" version of the indices, replacing -1 with 0 to avoid index errors
    safe_idx = topk_idx.clone()
    safe_idx[safe_idx < 0] = 0

    # 6b) We'll do advanced indexing to fill valid neighbor slots
    valid_mask = (topk_idx >= 0)  # (B, N, k) => True where neighbor is valid

    # To do advanced indexing, we need the broadcasted batch/row/col indices:
    b_idx = torch.arange(B, device=A.device).view(B, 1, 1).expand(B, N, k)    # (B, N, k)
    n_idx = torch.arange(N, device=A.device).view(1, N, 1).expand(B, N, k)    # (B, N, k)
    # The "safe_idx" dimension is the neighbor index for each (b, n)
    # so we'll gather from dimension=1 in A => A[b, safe_idx, :]
    # We'll do advanced indexing on "neighbors[b, n, j, :]" = A[b, safe_idx[b, n, j], :]

    # Where valid, copy the data
    knn_neighbors[valid_mask] = A[b_idx[valid_mask], safe_idx[valid_mask], :]

    # 7) Finally, reshape to (B, N, 3*k)
    knn_neighbors = knn_neighbors.view(B, N, 3*k)
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
                 voxelize = True,
                 space_filling_order = None,
                 space_filling_curve = 'z',
                 band_classification = False,
                 bin_dir = ''):
        
        split = split
        self.band_classification = band_classification
        self.memmap_feature = RaggedMmap(os.path.join(data_root, 'features_{}'.format(split)))
        self.memmap_seg_target = RaggedMmap(os.path.join(data_root, 'seg_target_{}'.format(split)))
        self.memmap_reg_target = RaggedMmap(os.path.join(data_root, 'reg_target_{}'.format(split)))
        

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
        self.len_chunk = len_chunk
        
        self.train = train
        self.chunk_training = chunk_training
        self.filter_data(low_thr = 1, high_thr = 100)   # OPEN FILTER (single-track test): was 5-40
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

        # print(features.shape, target.shape)
        if not self.train and self.chunk_training:
            features = features[:, start_idx : start_idx+self.len_chunk]
            target = target[:, start_idx : start_idx+self.len_chunk]
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
                                   band_classification = getattr(params, 'band_classification', False),)

   
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