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
The point trainer currently fixes gradient clipping at 1.0 and uses head dropout
0.0; inherited generic clipping/dropout options do not control this head.

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

## 2026-11-05 benchmark

The matched 5k-event/3k-update control (same cohort, model, seed, and schedule)
increased test signal retention at threshold 0.5 from 33.70% to 87.82% after
balancing. At validation-selected 95% retention thresholds, test noise rejection
rose from 51.27% to 68.19%. Signal ROC AUC rose from 0.87796 to 0.91063.
Both checkpoints were selected at update 3,000.

The 50k-event/30k-update balanced run selected update 30,000. On 10,000 test
events (1,347,583 clusters), the results were:

| Operating point | Signal retained | Noise rejected |
| --- | ---: | ---: |
| Default threshold 0.5 | 95.77% | 94.27% |
| Validation target 95% retention | 95.22% | 94.76% |
| Validation target 99% retention | 98.92% | 83.20% |
| Validation target COATJAVA retention | 91.65% | 96.50% |
| COATJAVA unassigned baseline | 91.51% | 97.34% |

At approximately matched signal retention, the adapter's noise rejection is
0.83 percentage points lower than COATJAVA (paired test-event bootstrap 95%
interval: 0.75 to 0.91 points lower). This is a single-seed result; the interval
does not include variation from retraining. The larger run changes both data
volume and update budget, so their individual effects are not isolated. The
additional 8,000 test events beyond the exploratory pilot gave 95.82% signal
retention and 94.30% noise rejection at threshold 0.5.

a02c job IDs: `hdf5-noise-reweight-control-20261105-v1`,
`hdf5-noise-balanced-50k-30k-20261105-v1`, and
`hdf5-noise-benchmark-audit-20261105-v1`. Outputs are under
`/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/<job-id>/output`.
The immutable training source SHA256 is
`58f16d9d7cfb411c71f391fc9eee4ef4e5b556371971d7792c483748536e46fb`.
The audit checks exact control cohorts, common evaluation prefixes, checkpoint
selection, restored class weights, and completed update budgets.
