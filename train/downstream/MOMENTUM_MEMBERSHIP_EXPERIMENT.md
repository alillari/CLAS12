# Full-event momentum adapter with target-track membership

Local experimental mode: `adapter_sample_mode: event_segment_membership`.
Preset: `clas12_track_regression_m6_event_membership` in
`scripts/configs/mamba_clas12_track_regression_pretrained.yaml`.
Requires `--usepretrain --pretrained_ckpt <campaign-4-m6-checkpoint>` and the
same training target statistics as the matched track cohort.

## Data flow

1. Sample tracks with the established segment filters and track-level labels.
2. Encode their complete serialized parent events with the frozen backbone.
   Repeated events are deduplicated within the batch.
3. Repeat each full event feature sequence for its selected tracks. Keep all
   real tokens, including noise and other tracks, in event serialization order.
4. Mix the backbone layers using the existing learned softmax weights.
5. Append one binary channel to each mixed token feature: 1 for a hit of the
   requested track and 0 otherwise. Membership uses the original-row mapping,
   including when two distinct hits have identical coordinates.
6. Apply LayerNorm and the widened projection, then the existing adapter Mamba
   layers and attention/mean pooling over all real event tokens.
7. Predict one track target. Labels, loss masks, sample limits, and batch sizes
   still count supervised tracks; other event labels do not enter the loss.

For m6: backbone width = 1536, projection input width = 1537, projected adapter
width = 256. Neither backbone embeddings nor Mamba hidden widths are increased.
The membership bit is appended after layer mixing so it is a single explicit
channel rather than a separate channel per backbone layer.

Membership and padding masks serve different purposes. Zero membership does
not mask a token out of sequence processing or pooling. Only synthetic padding
is excluded from pooling. As in the existing adapter, sequence layers are
causal and padding is on the right.

## Comparison and limitations

This preserves more event information at the adapter input, but still uses
learned layer mixing and a 256-dimensional projection. It does not guarantee
all useful information survives or that optimization will exploit context.
The adapter does more work per track because it processes the complete event.
Full-event feature copies also increase batch memory when several tracks share
an event, although the frozen backbone is only evaluated once per unique event.

Train a fresh adapter: checkpoint guards reject exchanging checkpoints between
isolated, gathered-event, and membership-event modes. The standard trainer and
standalone momentum evaluator support the new mode. The pooled linear probe
explicitly rejects it because its track-pooling contract does not implement
membership-conditioned sequence processing.

Focused CPU tests cover full-event preservation, membership identity after
serialization, duplicate coordinates, deduplication, padding, projection
width, gradient flow, nonmember influence, checkpoint guards, and track-target
training/validation. The actual CUDA Mamba sequence kernels and large-scale
training remain to be tested. No remote deployment is part of this local change.
