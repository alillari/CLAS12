# Downstream Probe Diagnostics

This directory contains diagnostic tools for understanding CLAS12 downstream
track-regression behavior, especially the case where a pretrained backbone plus
adapter underperforms an adapter-only model.

These tools are not training entrypoints for production adapters. They are
controlled probes meant to isolate whether information is present in a frozen
backbone representation and whether trained adapters learned to use particular
backbone layers.

## `train_linear_probe.py`

Fits one affine linear map per frozen backbone layer:

```text
pooled layer features -> normalized (px, py, pz)
```

For each event, the script:

1. loads the downstream track-regression data through the existing trainer path,
2. runs the pretrained backbone with `return_z=True`,
3. masked-mean-pools each layer over valid hits,
4. builds the same normalized event-level regression target used by the adapter,
5. solves closed-form ridge regression independently for every layer,
6. evaluates each layer against held-out data.

The learned weights are determined by least squares with a configurable ridge
penalty. No nonlinear head is trained.

### Inputs

Required:

- `--yaml_config`: downstream track-regression YAML file.
- `--config`: config key inside that YAML.
- `--pretrained_ckpt`: pretrained backbone checkpoint containing `model_state`.
- `--output_dir`: directory for probe artifacts.

Useful controls:

- `--eventnumber`: train split event cap passed through the existing config path.
- `--batch_size`: train/eval batch size.
- `--num_workers`: override data loader workers.
- `--max_train_batches`: quick smoke-test cap for fitting.
- `--max_eval_batches`: quick smoke-test cap for evaluation.
- `--ridge_alpha`: ridge penalty, default `1e-4`.
- `--shuffled_control_events`: number of train events cached for the shuffled-label
  control; set to `0` to disable.
- `--random_backbone_control`: also fit/evaluate a randomly initialized backbone.

### Example

Small smoke test:

```bash
python train/downstream/probes/train_linear_probe.py \
  --yaml_config scripts/configs/mamba_clas12_track_regression_pretrained.yaml \
  --config clas12_track_regression_pretrained \
  --pretrained_ckpt /path/to/pretrained_backbone/ckpt_best.tar \
  --output_dir /path/to/probe_outputs/linear_probe_smoke \
  --eventnumber 10000 \
  --batch_size 32 \
  --num_workers 4 \
  --max_train_batches 10 \
  --max_eval_batches 10 \
  --random_backbone_control
```

Larger run:

```bash
python train/downstream/probes/train_linear_probe.py \
  --yaml_config scripts/configs/mamba_clas12_track_regression_pretrained.yaml \
  --config clas12_track_regression_pretrained \
  --pretrained_ckpt /path/to/pretrained_backbone/ckpt_best.tar \
  --output_dir /path/to/probe_outputs/linear_probe \
  --eventnumber 100000 \
  --batch_size 64 \
  --num_workers 8 \
  --random_backbone_control
```

### Outputs

- `per_layer_metrics.csv`: one row per probe/layer with normalized MSE, MAE, and
  R2 against the train-mean baseline.
- `probe_summary.json`: run configuration, baseline statistics, and best layer
  summaries.
- `linear_probe_weights.pt`: fitted coefficient tensors.
- `layer_metric_curves.png`: R2 and MSE versus backbone layer.
- `runtime/`: trainer-created logs and temporary runtime directories.

### Baselines and Interpretation

The primary null model is the train-mean predictor. R2 is computed relative to
that baseline:

```text
R2 = 1 - probe_SSE / train_mean_baseline_SSE
```

Interpretation:

- `R2 <= 0`: no better than predicting the train-set mean target.
- `R2 > 0`: the layer contains linearly accessible momentum information.
- Higher R2 in intermediate layers than final layers suggests useful information
  exists but is not necessarily concentrated at the final backbone output.

Additional controls:

- `gaussian_random_expected_mse` is an expected random-guess baseline using the
  train target mean/variance. It is mainly a sanity check.
- `pretrained_backbone_shuffled_labels` fits the same linear solver with shuffled
  train labels. It should be near the train-mean baseline; if it is not, suspect
  leakage or a flawed split/control setup.
- `random_backbone` is enabled by `--random_backbone_control`. A useful pretrained
  backbone should beat this control clearly.

The most informative pattern is:

```text
shuffled-label probe ~= train mean < random backbone < pretrained backbone
```

If the pretrained probe is strong but the adapter is weak, the backbone likely
contains usable information and the issue moves toward adapter extraction,
pooling, layer mixing, or optimization. If the pretrained probe is no better
than shuffled/random controls, the frozen backbone representation is not exposing
momentum in a simple linearly recoverable form.

## `audit_layer_weights.py`

Audits trained adapter checkpoints by extracting `weighted_avg_weights` from
`model_state_dict`, applying softmax, and reporting the learned layer mixture.

This tool does not need fresh data. It only answers what already-trained adapters
learned to do with their available backbone layers.

### Inputs

Required:

- one or more checkpoint files or directories,
- `--output_dir`: directory for audit artifacts.

Optional:

- `--pattern`: checkpoint glob used when searching directories, default `*.pth`.

### Example

Audit one local checkpoint directory:

```bash
python train/downstream/probes/audit_layer_weights.py \
  /path/to/adapter_checkpoints \
  --output_dir /path/to/probe_outputs/layer_weight_audit
```

Audit multiple checkpoint roots:

```bash
python train/downstream/probes/audit_layer_weights.py \
  /path/to/first_adapter_campaign \
  /path/to/second_adapter_campaign \
  --output_dir /path/to/probe_outputs/layer_weight_audit
```

### Outputs

- `layer_weight_audit.csv`: one row per checkpoint/layer.
- `layer_weight_audit.json`: checkpoint-level summaries and skipped files.
- `layer_weight_heatmap.png`: softmax layer weights by checkpoint.

Important columns:

- `softmax_weight`: normalized contribution of that layer.
- `top_layer`: layer with the largest learned weight.
- `top_weight`: softmax weight of the top layer.
- `entropy`: layer-mixture entropy.
- `effective_layers`: `exp(entropy)`, the approximate number of layers being
  used.

### Interpretation

- Uniform weights mean the adapter did not learn a layer preference, or the
  checkpoint was saved before meaningful movement.
- A concentrated top layer means the adapter selected a particular depth.
- High effective-layer count means broad layer mixing.
- Low effective-layer count means the adapter mostly relies on a small number of
  layers.

This audit alone does not prove that the selected layer is physically good. Use
it alongside the linear probe: if the best linear-probe layer differs strongly
from the adapter's learned top layer, that is evidence to investigate the adapter
layer mixer or downstream optimization.

## Practical Workflow

1. Run `audit_layer_weights.py` over existing trained adapters to see whether
   the learned layer mixer moved at all.
2. Run `train_linear_probe.py` on one representative pretrained backbone with
   small batch caps.
3. If the smoke test is healthy, repeat without caps and include
   `--random_backbone_control`.
4. Compare the best linear-probe layer with the adapter audit.
5. Only then decide whether to test richer heads, raw-input baselines, or
   partial backbone unfreezing.
