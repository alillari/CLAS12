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
prediction. Checkpoints are selected by validation loss; evaluation uses argmax.
The ordinary `eval_point_classification.py` CLI accepts the explicit downstream
checkpoint and optional pretrained backbone, and evaluates the HDF5 test split.
