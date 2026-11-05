# Noise tagging

## HDF5 AdapterOnly benchmark

This branch integrates noise tagging with the published HDF5 cluster product.
The shared data support is a separate ancestor commit; the other adapter
integrations are independent sibling branches.

```bash
python -m train.downstream.benchmark_hdf5_noise \
  --data-root /absolute/path/to/cvt-clusters-h5 \
  --output-dir /absolute/path/to/new/noise-run
```

Defaults: 5,000 accepted training samples, 3,000 updates, 1,000 validation
samples and 2,000 test samples, seed 42. Cohorts use accepted prefixes. The
runner checks disjoint source-event identities, reloads the validation-selected
checkpoint and writes configurations, cohort IDs, predictions and `summary.json`.
For a smoke test add `--train-size 32 --steps 2 --validation-size 16 --test-size 16`.
The configuration is `hdf5_noise_5k` in `scripts/configs/clas12_adapter_recipes.yaml`.

Samples are whole events. The trainer forces event sampling to retain background.
Labels use `primary_segment < 0` (including contested clusters) as noise. The
COATJAVA baseline is explicitly its unassigned clusters, not a dedicated noise
prediction. Checkpoints are selected by validation loss. Evaluation reports the
default threshold and thresholds selected on validation for 95%, 99%, and
COATJAVA-matched signal retention. Test labels never select thresholds.
The ordinary `eval_point_classification.py` CLI accepts the explicit downstream
checkpoint and optional pretrained backbone, and evaluates the HDF5 test split.

## Class-balanced training

Use `--class-balance inverse_frequency` for the matched 5k/3k control, or
`--config hdf5_noise_50k_balanced` for 50,000 training events and 30,000 updates
with 5,000 validation and 10,000 test events. The larger recipe uses batch 32,
AdamW with peak learning rate 3e-4, weight decay 1e-4, 1,000 warmup updates,
cosine decay to 1e-5, gradient norm clipping at 1.0, and validation every 1,000
updates. It retains the pilot's 759,310-parameter attention head and seed 42.
Both runs are AdapterOnly: no pretrained backbone features enter the head.

For each class c, the weight is N/(2*N_c), counted from accepted training
events. This gives equal total weight to signal and noise; it does not force
equal prediction counts. Padding is excluded. These weights are fixed during
validation and recorded in the resolved config and checkpoint. Validation loss
is pooled over clusters; per-class losses, signal retention, and noise rejection
are also logged in `checkpoints/training.log.jsonl`.

`validation_metrics.json` records calibration; `test_metrics.json` records fixed
operating points and a threshold curve. Compressed score files and per-event
prediction records allow subsequent analysis without rerunning inference. Class
weighting changes score calibration, so interpret scores through these measured
tradeoffs. For diagnosis of a previous run use `--checkpoint /path/to/checkpoint`
with the matching original recipe and cohort sizes; add `--evaluate-training`
to compare training and validation score separation.
