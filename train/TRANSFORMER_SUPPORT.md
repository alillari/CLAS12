# Causal transformer backbones and bidirectional adapters

`backbone_version: transformer` selects full self-attention with a causal mask
for pretraining. `adapter_version: transformer` independently selects
bidirectional attention + feed-forward refinement for classification, track
finding, and track regression. Either selection can be mixed with an existing
Mamba selection. Existing configurations, Mamba defaults, parameter names, and
checkpoint layouts are unchanged.

## Sequence encoding and geometry

The new transformer defaults to RoPE. The supported choices are:

| Setting | Values | Purpose |
| --- | --- | --- |
| `backbone_sequence_encoding` | `rope` (default), `none` (NoPE) | Backbone sequence encoding |
| `adapter_sequence_encoding` | `rope` (default), `none` (NoPE) | Adapter sequence encoding |
| `num_heads_backbone` | positive divisor of backbone width; default 4 | Backbone attention heads |
| `adapter_num_heads` | positive divisor of both adapter/input widths; default 4 | Adapter attention heads |
| `adapter_embed_dim` | default 256 | Adapter refinement width, independent of the backbone |
| `num_adapter_layers` | at least 1; classification/regression default 1 | Adapter refinement depth |
| `backbone_mlp_ratio`, `adapter_mlp_ratio` | positive; default 4 | FFN hidden width / model width |
| `backbone_rope_theta`, `adapter_rope_theta` | positive finite; default 10000 | RoPE frequency base |

RoPE rotates query/key pairs using serialized **valid-token indices**, separately
inside every attention layer. Head width must be even for RoPE. NoPE adds no
sequence-position embeddings or rotations; the causal backbone still has its
directional attention mask. Independently changing one encoding never changes
the other's selection.

`embed_method` and `pe_method` retain their detector-coordinate meaning. NoPE
does **not** remove geometry. Backbones support `pos_only`, `pos_plus_aux`, `add`,
and `concat`. Coordinate embedding and token serialization remain the existing
data contract. Pretraining predicts the existing outward-band neighbor targets;
it does not introduce a text-style token shift or a different target definition.

Padding masks use `True` for real hits. The backbone infers padding from the
existing `-100` sentinel in the first input column; an explicit mask can further
exclude tokens. Attention excludes padded keys and zeros padded hidden states.
Left, internal, right, and all-padding attention inputs are covered by tests.
RoPE counts real tokens, so inserting padding does not shift real-token phases.

## Pretraining

The maintained entrypoint also accepts the transformer family:

```bash
python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
  --yaml_config scripts/configs/transformer_pretrain.yaml \
  --config transformer_rope_v602 --run_num trial0 \
  --root_dir /absolute/path/to/new/transformer-experiment
```

Use `transformer_nope_v602` for the NoPE control. Update the configuration's data,
statistics, and output paths for your environment. These examples reuse the
existing v6_02 loader and width-256/depth-12 layout with nominal optimization
settings, not transformer-tuned hyperparameters. Parameter counts and compute
are **not** matched to Mamba by equal width/depth. No campaign is launched by
adding these configurations.

The normal forward returns `(batch, sequence, klen * 3)` predictions.
`return_z=True` returns `(None, per_layer_features, coordinate_embedding)`, with
the accumulated hidden state after each residual block, as downstream layer
mixing expects. Training, validation, optional muP width scaling, offline W&B,
and optimizer/scheduler resume use the existing trainer. Its epoch-level resume
cursor limitations remain unchanged.

## Downstream tasks

The following YAML files contain `_transformer_rope` and `_transformer_nope`
examples, with separate output directories:

- `mamba_clas12_track_regression_pretrained.yaml`: `clas12_track_regression_transformer_rope`
- `mamba_clas12_track_regression_adapteronly.yaml`: `clas12_track_regression_adapteronly_transformer_rope`
- `mamba_clas12_track_finding_adapteronly.yaml`: `clas12_track_finding_transformer_rope`
- `mamba_clas12_point_classification_adapteronly.yaml`: `clas12_point_pid_transformer_rope`

Use the existing task launchers. Add `--usepretrain --pretrained_ckpt ...` for
frozen feature transfer; omit `--usepretrain` for AdapterOnly. Backbone width,
depth, encoding, heads, FFN ratio, dropout, embedding, and prediction head must
match the pretraining checkpoint. The supplied pretrained examples match the
new width-256/depth-12 pretraining recipe. For another trained backbone, set
those fields to its saved architecture. The adapter may have a different width
and encoding from the backbone. The existing learned mixture of backbone layers
and input projection are preserved.

Transformer refinement is selected by `adapter_version`, including when the
legacy classification `use_attention_head` flag is set. Tracking retains its
existing prototype/cross-attention decoder and replaces its Mamba token
refinement with transformer blocks. Set `num_adapter_layers` to at least one for
tracking: its historical default of zero would otherwise omit token refinement.
Raw-token embedder layers, when enabled by `num_embedder_layers`, also use the
selected adapter family. Regression retains its existing pooling, normalization,
and targets. The existing regression auxiliary-input path is preserved; this
change does not add auxiliary sidecar ingestion to the classification/tracking
loaders.

## Checkpoints and validation

Transformer architecture metadata records causal/bidirectional behavior, head
count, NoPE/RoPE, theta, FFN ratio, width, and dropout for every block. Loading
rejects semantic mismatches even when tensor shapes match. New transformer
checkpoints require this metadata; historical Mamba checkpoints remain loadable.
Existing Linformer/Longformer code is retained as a legacy downstream option;
neither is used as the new causal backbone or promoted to causal pretraining.

Run the portable suite with:

```bash
OMP_NUM_THREADS=1 python -m unittest train.downstream.tests.test_transformer_support
python -m unittest discover -s train/downstream/tests
python -m unittest discover -s train/pretrain/tests
```

Tests check future-token perturbations, zero future-input gradients, prefix
equivalence, bidirectional context, NoPE permutation behavior, RoPE rotations,
padding, task-head optimizer steps, frozen/AdapterOnly transfer, evaluator
construction, semantic checkpoint rejection, and trainer validation/resume.
The optional CUDA case exercises BF16 autocast and backbone-to-adapter transfer.
CPU testing is functional validation, not a completed detector-data campaign.

This is dense attention with quadratic attention memory, using PyTorch SDPA.
Scope is full-sequence training and feature extraction; cached decoding is not
implemented. See [PyTorch SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)
for mask semantics and [RoFormer](https://arxiv.org/abs/2104.09864) for RoPE.
