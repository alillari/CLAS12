# Downstream training options

Use [Training workflow](../TRAINING_RECIPES.md) for a complete run. This reference
describes the controls supported by the ordinary track-finding and regression
CLIs. [Backbone recipes](../BACKBONE_RECIPES.md) covers pretraining; [Mamba
support](../MAMBA_SUPPORT.md) covers family-specific block/runtime options.

## Configuration and CLI precedence

`--yaml_config` selects a YAML file and `--config` selects its top-level profile.
YAML merge keys (`<<: *default`) inherit shared defaults. Environment expansion
supports `$NAME` and `${NAME:-fallback}`; string references such as
`{artifact_root}` resolve against the selected profile. Run from the repository
root and use absolute paths for data, checkpoints and external configs.

| CLI option | What it controls |
| --- | --- |
| `--eventnumber` | Overrides `limit_size` and enables a training sample limit. Default 50k; events for finding, accepted tracks for event-segment regression. |
| `--train_batch_size` | Overrides YAML training batch. Default **32** for both CLIs; pass 128 for the momentum starting recipe. Finding also replaces validation/local batch sizes; regression retains its independently configured validation batch. |
| `--backbone`, `--backbone-catalog`, `--backbone-root` | Select a catalog ID and resolve checkpoint architecture/input metadata; see [Pretrained backbones](../PRETRAINED_BACKBONES.md). Catalog and root flags are optional. |
| `--usepretrain`, `--pretrained_ckpt`, `--no-pretrain` | An explicit path or catalog selection enables pretrained features. `--usepretrain` requires a selection/path in CLI or YAML. `--no-pretrain` clears the selection and uses AdapterOnly. Historical checkpoints without full metadata need a matching recipe or catalog config. |
| `--seed` | Overrides the YAML seed for initialization and supported loader randomization. A seed does not select a random training subset; ordinary loaders use accepted-sample prefixes. |
| `--run_num` | Run identifier used in default paths/log names; does not change optimization. |
| `--checkpoint_dir` | Overrides the YAML checkpoint/log directory. Give every fit a separate directory. |
| `--log_file_name`, `--checkpoint_file_name` | Explicit training log and adapter checkpoint filenames. |
| `--root_dir`, `--global_log_dir` | Experiment-root and global-log locations. `--root_dir` does not replace every path already set by the YAML. |
| `--resolved_config_path` | JSON snapshot of resolved parameters before trainer launch. |
| `--artifact_summary` | JSON training receipt with checkpoint/log locations, selection metrics and actual final step; useful for downstream automation. |

The regression wrapper forces `num_embedder_layers: 0` and validates the
regression target and validation-loss checkpoint-selection contract.
Track finding forces event-level sampling and validates supported target modes.
`train_momentum_regression.py` delegates to the regression CLI.

## Data, representation and sample selection

| YAML option | Effect / supported choice |
| --- | --- |
| `reader_type` | `ragged_mmap` (default), or `hdf5` for the published `cvt-clusters-h5-v1` manifest. HDF5 uses named fields directly without exporting sidecars. |
| `train_split`, `validation_split`, `test_split` | HDF5 defaults to `train`, `validation`, `test`. Legacy defaults remain `pretrain`, `test`, `test`. |
| `data_root` | Training RaggedMmap root (`*_pretrain` trees). `data_root_train` is compatibility metadata in this loader. |
| `data_root_test` | Validation/evaluation root (`*_test` trees). Onboarding training profiles use `CLAS12_VALIDATION_DATA_ROOT` when set; otherwise the training product's test trees. `*_evaluation` profiles use `CLAS12_EVAL_DATA_ROOT`, defaulting to the training product. Variables select existing populations; they do not split events. |
| `stat_dir` | Loader/pretraining compatibility-stat directory; distinct from `regression_target_stats`. |
| `input_layout` | `xyz` for three-column Cartesian centers; `etaphr` for eta/phi/r input; `exyz`/`eetaphr` for energy-plus-coordinate layouts. Select the stored column/coordinate convention, not the desired output layout. |
| `serialization`, `clas12_band_config` | Use `clas12_band_hilbert` and `calibrated12` for the standard cluster ordering. `radius` selects radial ordering; `none` disables the final serialization permutation. Changes affect the sequence seen by the model and must match the checkpoint contract. |
| `embed_method`, `pe_method` | `pos_only` and `nerf` for center-only inputs. Must match a frozen checkpoint's embedding contract. |
| `voxelize`, `space_filling_order` | Keep false in these downstream CLAS12 profiles. Select the CLAS12 ordering with `serialization`; inherited TPC options assume different input columns. |
| `input_representation` | `center_only`, or regression's `clas12_pos_plus_aux_v1` for typed center/direction/length-or-pitch sidecars. Auxiliary mode requires matching data and checkpoint contracts; see the corresponding regression YAML variants. |
| `pos_dim`, `aux_dim`, `aux_extra_feature` | Auxiliary representation: three center coordinates plus three local strip-direction components and `length`, `pitch` or `both`. Use `aux_dim: 4` for one scalar, 5 for both; `pos_dim: 3`. |
| `adapter_sample_mode` | Finding uses `track_legacy` as its **event-level** loader name; `event` is an equivalent alias. Regression uses `event_segment`; legacy track-level regression is disabled. |
| `low_thr`, `high_thr` | Inclusive cluster-count bounds. Apply to the whole event for finding and each segment for regression. Starting values: 1–512 and 1–100 respectively. |
| `max_tracks` | Event-level filter on the number of unique segmentation labels, including background when present. Distinct from the number of decoder queries. |
| `segment_target_source` | Regression: `mctrue`/`seg_target` for truth segments; `coatjava` for reconstruction candidates matched to a dominant truth segment. |
| `segment_min_clusters`, `segment_exact_clusters` | Minimum or exactly-equal segment size. Standard momentum accepts at least 12 clusters, subject to `high_thr: 100`. |
| `segment_min_truth_purity` | COATJAVA candidate's required dominant-truth fraction; standard value 1.0. |
| `return_dict`, `return_reg`, `return_reg_test`, `require_reg_target` | Batch/sidecar plumbing. Keep the task preset's values; regression requires real aligned targets. |
| `return_coatjava_seg_pred`, `require_coatjava_seg_pred` | Load/require reconstruction segmentation sidecars when a comparison needs them. Finding evaluation has a separate `evaluate_coatjava` control. |
| `return_knn_target` | Additional next-point targets; false for ordinary adapter fits. |
| `compute_neighbors` | Defaults true for compatibility. HDF5 adapter recipes disable unused quadratic nearest-neighbor preprocessing; `return_knn_target: true` still computes it. |
| `num_data_workers` | CPU loader workers; affects preprocessing throughput, memory and debugging, not the training sample budget. |

The loader radius-sorts before its final serialization step; `serialization:
none` retains that radius ordering rather than the raw storage order. Coordinate
conversion and normalization are loader operations, so the layout must describe
the stored values accurately.

For HDF5, `clusters.xyz` is in cm and requires `input_layout: xyz`. Finding
uses `clusters.primary_segment` unchanged: **all `-1` clusters, including
contested ownership, remain background**. Binary noise labels are explicitly
`primary_segment < 0` (1=noise, 0=signal), not the nullable `primary_noise`
column. This policy is recorded as `primary_unassigned_is_noise_v1` in checkpoint
data metadata. No contested-cluster loss mask is introduced.

Momentum labels join `clusters.primary_tid` to `particles.tid`; compact segment
labels and particle rows are different identities. The existing per-cluster
target interface repeats the particle's observed momentum, vertex and energy,
and keeps unobserved momentum components as NaN. Momentum is already **GeV/c**.
COATJAVA cluster grouping is available through `clusters.reco_segment`.
Stable source/event identities are retained in batches. Adapter branches add
their own statistics, evaluation and comparison rules.

The auxiliary representation requires row-aligned 11-column `geometry_context`
sidecars. It projects strip direction into the local radial/tangential/z frame
and appends length divided by 45.0, pitch divided by 0.10, or both. These scales
and feature ordering are part of the checkpoint's input contract.

The standard path performs full-event/full-segment training (`chunk_training:
false`). Keep inherited grouping/chunk fields at the preset values unless
working on that loader explicitly. Changing `n_data_idx`, `len_chunk`, `portion`
or `nleave` is not the supported way to set the adapter label budget; use the CLI
limit and task filters.

## Backbone and trainable head

| Option | Meaning |
| --- | --- |
| `backbone_version` | `mamba1`, `mamba2`, `mamba3`, `transformer` (plus Longformer/Linformer compatibility paths); takes precedence over legacy `mambaversion`. Match the checkpoint. |
| `pretrained_backbone`, `backbone_catalog`, `backbone_root` | YAML equivalents of the catalog selection flags. Resolved backbone metadata replaces recipe backbone defaults before constructing datasets or models; adapter settings remain independent. |
| `backbone_output_dim`, `use_sincos_target` | Preserve the pretraining head shape; by default `klen * 3`, or `klen * 4` for sine/cosine targets. Campaign 5 resolves to 20 outputs. These do not select the downstream task/head output dimension. |
| `embed_dim`, `num_layers_backbone`, `d_state`, `d_conv`, `expand` | Backbone width/depth/block dimensions. Must match pretrained weights; input width also sizes the AdapterOnly coordinate branch. |
| `klen` | Number of coordinate triples in the backbone's next-point output (`3*klen`). Required for checkpoint compatibility even when only layer features are used. |
| `backbone_*` | Overrides shared block options for the backbone. See [Mamba support](../MAMBA_SUPPORT.md). |
| `adapter_version`, `adapter_*` | Independently select the head's Mamba family/block settings. They do not change an attention decoder or the frozen backbone. |

### Track-finding head and objective

| Option | Effect / starting value |
| --- | --- |
| `adapter_embed_dim` | Shared decoder latent width, 256. Projects different backbone widths into a fixed-capacity decoder. |
| `num_adapter_layers`, `num_embedder_layers` | Post-projection/input Mamba block counts; both **0** in the starting recipe. |
| `num_prototypes` | Number of track-query slots, 16. Bounds how many instances can be represented. `max_gt_classes` is the fallback when omitted. |
| `num_self_attn_layers`, `num_heads_decoder`, `ffn_dim` | Prototype attention refinement depth, attention heads and FFN width; starting values 2, 2 and 512. Width must permit the chosen attention head count. |
| `do_masked_attn` | Use predicted mask information in prototype refinement attention; true in the preset. |
| `softmax_mask` | Normalize mask scores across queries instead of independent sigmoids; false in the preset. Changes assignment probabilities. |
| `downstream_dropout` | Tracking head dropout, 0.1. Distinct from backbone `dropout`. |
| `track_target_mode` | `signal_only`: one target per signal track; background is negative in every signal mask, not a query target. |
| `loss_matched_ce_weight`, `loss_unmatched_ce_weight`, `loss_dice_weight`, `loss_focal_weight` | Weights for matched class, no-object class, Dice and focal losses; starting values 0.5 / 0.1 / 1 / 30. |
| `validation_selection_metric` | `dice` (default) maximizes mean track Dice at the configured option-2 threshold. Lower validation loss breaks ties. Use `ari` for historical selection. |
| `validation_ari_mode` | `signal` excludes background from ARI. `inclusive` includes valid background. Controls ARI reporting and selection when `validation_selection_metric: ari`. |
| `assignment_option` | 1 uses mask probabilities; 2 combines mask and query-class confidence and excludes no-object queries. Starting value 2. Validation records both; checkpoint selection uses option 2. |
| `assignment_threshold` | Minimum point-assignment confidence; starting value 0.2. Changes background rejection, contamination and efficiency. Keep it fixed when comparing checkpoints. |
| `match_iou_threshold`, `match_min_purity`, `match_min_efficiency` | Truth/prediction matching cuts for reported track metrics, each 0.5 in the preset. These are metric definitions, not LR or query-count settings. |

### Regression target and head

| Option | Effect / choice |
| --- | --- |
| `task: mom` / `momentum` | Three entrance Cartesian momentum components; standard starting target. |
| `task: pt_phi_eta` | Four outputs: pT, cos(phi), sin(phi), eta. |
| `task: p_phi_theta` | Four outputs: p, cos(phi), sin(phi), theta. |
| `task: 3vtx` / `3vertex`, `zvtx` / `Zvertex` | Three vertex coordinates or z only. Use the appropriate target columns/stats. |
| `regression_target_stats` | Target mean/std JSON; see [target statistics](TARGET_STATISTICS.md) for generation, weighting and reuse. |
| `regression_loss` | `mae`, `mse` or `huber` on standardized targets. The momentum preset explicitly uses MAE; omission falls back to MSE for joint tasks. |
| `pooling` | `attention` learns per-cluster scores; `mean` averages valid cluster features. Attention is the momentum starting recipe. |
| `dropout` | Regression head dropout, 0.1. Also used when constructing the backbone; frozen feature extraction runs in eval mode. |
| `checkpoint_selection` | `validation_loss` for new regression training. Physical width, bias and tails are evaluation diagnostics. |

For Mamba-family regression heads, the ordinary trainer fixes **width 256 and
one refinement layer**, with zero input-embedder layers through its CLI. The
default is Mamba2/state 64/conv 4/expand 2; `adapter_*` block overrides are
honored. `num_adapter_layers` and `adapter_embed_dim` do not change those Mamba
dimensions. For `adapter_version: transformer`, both fields are honored and
refinement depth must be at least one. `use_attention_head` does not select the
regression head type. See [Transformer support](../TRANSFORMER_SUPPORT.md) for
independent backbone/adapter encoding, heads, width/depth and FFN controls.
Scalar p/theta/phi heads are inference-only compatibility paths; new training
rejects them.

## Update budget, learning rate and stopping

| Option | Effect / interaction |
| --- | --- |
| `max_optimizer_steps` | Enables update-controlled training; the starting profiles set 30,000. Null chooses epoch-controlled training. |
| `max_epochs` | Outer cap in either mode. Set high enough to reach the step budget, especially for small label sets. |
| `total_steps` | Does not by itself set the downstream training-loop budget. Pretraining uses it differently. |
| `max_lr`, `min_lr` | Adapter LR peak and floor; 3e-4 and 1e-5. AdamW uses default betas (0.9, 0.999) for the ordinary heads. |
| `scheduler_first_cycle_steps` | Cosine cycle length; takes precedence over `first_cycle_steps`. Starting value 30,000. Shorter cycles restart during a longer run. |
| `warmup_steps` | Linear warmup updates at the start of each restart cycle; starting value 1,000. Keep below the cycle length. |
| `warmup_fraction` | Regression alternative that takes precedence over `warmup_steps`: derives warmup as a fraction of the cycle length, or of `scheduler_anneal_steps` in hold mode. |
| `scheduler_mode` | Regression: `cosine_restarts` or `cosine_hold`. Finding uses the restart scheduler; this mode field does not change its scheduler. |
| `scheduler_anneal_steps` | Regression `cosine_hold`: endpoint of the single warmup/cosine phase; LR then stays at `min_lr`. Warmup must fit inside it and it cannot exceed the update budget. |
| `adapter_weight_decay` | Head AdamW decay; 1e-4. Native SSM decay exclusions are honored. |
| `grad_clip_value` | Maximum **gradient norm** for downstream updates; 0.1. Pretraining's similarly named setting uses per-element clipping. |
| `early_stopping_patience`, `early_stopping_min_delta` | Allowed non-improving validation checks and improvement tolerance. Regression saves every lower finite loss; min_delta controls patience. Finding applies the tolerance in legacy ARI selection. Dice selection uses the exact score and a loss tie-break. |
| `early_stopping_min_steps` | Step-mode lower bound before stopping/pruning. Set above the budget (30,001 in these profiles) for a complete fixed-budget fit. |
| `early_stopping_warmup_steps` | Epoch-mode stopping warmup/fallback; do not confuse with LR warmup. |
| `max_train_batches` | Per-epoch batch cap; useful for bounded runs but changes exposure/epoch behavior. |
| `use_amp` | Finding's outer mixed-precision switch; false in the starting profile. It does not disable Mamba3's internal BF16 execution. It is not a general regression precision switch. |

Track finding also has an opt-in `use_lora` path with `lora_rank`, `lora_alpha`
and comma-separated `lora_target_modules`. It changes the trainable backbone
parameters and has a separate optimizer. Keep it false for a frozen-feature
comparison; family/module compatibility needs its own validation.

## Validation, checkpoints and evidence

| Option | Effect |
| --- | --- |
| `val_interval_steps` | Validation cadence in step mode; 1,000. Epoch mode validates per epoch. |
| `limit_test_data`, `limit_test_size` | Fixed validation sample cap, independently of training limit. Starting profiles use 5k finding events / 10k momentum tracks. |
| `valid_batch_size` | Regression validation batch; independent of training batch. Finding's CLI replaces it with `--train_batch_size`. |
| `max_validation_events` | Additional exact event cap for finding validation. Align it with `limit_test_size`. |
| `max_val_batches` | Optional validation batch cap. Regression rejects a cap that truncates its requested fixed sample. Finding can be limited by this cap as well as its event cap. |
| `drop_last_test` | Retain final partial validation batch with false; regression resolution forces false. |
| `save_validation_checkpoints` | Finding: save every validation's adapter snapshot as well as metric histories. Snapshots omit optimizer/scheduler and are for evaluation; the normal best checkpoint is full. |
| `continue_from_best` | Shared internal setting; the public CLIs start fresh fits and do not expose resume-from-checkpoint through this flag. |

Momentum selects the lowest finite validation loss. Finding selects configured
option-2 ARI with a validation-loss tie-break. Save resolved configuration,
artifact receipt, normalizer, adapter weights and matching pretrained weights.
Validation evaluates an accepted prefix of the configured test root; final
evaluation requires a separately reserved population, with whole source events
kept together for segment-level tasks.

See [evaluation and plot interpretation](eval/README.md) for the behavior checks
to make after training. Optuna and multi-run orchestration have their own
[tuning](tuning/README.md) and [campaign](campaign/GETTING_STARTED.md) interfaces.
