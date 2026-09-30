# Momentum adapter: full-event backbone context

Experimental branch: `momentum-adapter-experiments`.

`adapter_sample_mode: event_segment_context` changes the frozen backbone input
from an isolated track to its entire event. The adapter receives only that
track's token states, in its original isolated-track serialization order.
The backbone checkpoint and `MambaTrackRegressionHead` architecture are unchanged.

## Units and sampling

The supervised sample remains `(source_event_index, segment_label)`. Thus
`limit_size`, `limit_test_size`, `--eventnumber`, batch sizes, sampler ordering,
optimizer steps and loss weighting retain their existing **track** meaning.
The accepted cohort, target masks and target statistics match `event_segment`.
Tracks excluded by the label budget and noise still appear as event context;
they do not produce additional supervised losses.

Each dataset item adds the full event and integer token indices. The collator
deduplicates events within its batch. The backbone processes `(unique_events,
event_tokens, channels)`, then each layer is gathered to `(tracks, track_tokens,
width)` before depth mixing, projection, adapter Mamba and pooling. Different
batches may encode the same event again; no feature cache is used.

Two explicit permutations connect raw point-row IDs to event and track order.
This handles ties/duplicate coordinates without coordinate matching. No on-disk
mapping or new skim is required. All event tokens are retained, and right
padding preserves causal Mamba1 outputs for valid tokens. The existing
`high_thr` still limits the selected track, not its surrounding event.

The initial mode requires a pretrained Mamba1 backbone, `center_only` /
`pos_only` inputs, dictionary batches and `chunk_training: false`. Unsupported
combinations fail. Checkpoints record `adapter_sample_mode` and reject loading
between isolated-track and full-event context. Train a fresh adapter for each arm.

## Matched m6 comparison

The pretrained YAML contains two campaign-4 m6 presets (width 1536, depth 12,
`d_state=16`, `klen=1`, `p_phi_theta`). They differ only in sample mode:

- `clas12_track_regression_m6_track_context`: existing isolated-track baseline.
- `clas12_track_regression_m6_event_context`: full-event backbone context.

Use the same seed, accepted-track budget, batch size, optimization recipe,
target statistics and validation cohort. These presets inherit the base training
recipe; they do not claim to reproduce a particular tuned campaign recipe.
For a campaign comparison, apply the same existing training overrides to both.

On a CUDA host, set these paths to that host's mounted or local data:

```bash
context_python=/home/alessio/miniconda3/envs/fm4npp/bin/python
context_repo=/home/alessio/ML-work/CLAS12
context_runs=/path/to/new/m6_context_comparison
context_checkpoint=/path/to/pretrained-FMs/campaign_4/scale_w1536_d12_n39553933/ckpt_best.tar
export CLAS12_CONTEXT_DATA_ROOT=/path/to/mmap_canonical_loose_truthseg_event_v6_02
export CLAS12_CONTEXT_STATS="$context_runs/target_stats.json"

"$context_python" "$context_repo/train/downstream/scripts/compute_regression_target_stats.py" \
  --data-root "$CLAS12_CONTEXT_DATA_ROOT" --split pretrain --task p_phi_theta \
  --adapter-sample-mode event_segment --segment-min-clusters 12 \
  --limit-size 50000 --output "$CLAS12_CONTEXT_STATS"

for context_arm in track event; do
  "$context_python" "$context_repo/train/downstream/train_track_regression.py" \
    --yaml_config "$context_repo/scripts/configs/mamba_clas12_track_regression_pretrained.yaml" \
    --config "clas12_track_regression_m6_${context_arm}_context" \
    --usepretrain --pretrained_ckpt "$context_checkpoint" \
    --eventnumber 50000 --train_batch_size 32 --seed 42 \
    --root_dir "$context_runs/$context_arm" \
    --checkpoint_dir "$context_runs/$context_arm/checkpoints" \
    --resolved_config_path "$context_runs/$context_arm/resolved_config.json" \
    --artifact_summary "$context_runs/$context_arm/artifacts.json" || exit
done
```

The normal trainer and final evaluator use the same feature-gathering path.
Evaluate both fresh adapters on the same track identities and compare fixed
momentum-bin widths in delta-p/p together with angular, bias and tail metrics.
The dataset's `test` split used for checkpoint selection is validation; a final
performance claim requires a separate untouched cohort.

## Verification

Focused tests cover identical cohorts/targets, duplicate-coordinate identity,
preserved track order, full-event serialization, within-batch event deduplication,
padding, COATJAVA target masks, a CPU optimizer step, validation, and checkpoint
context mismatches:

```bash
PYTHONPATH="$context_repo" "$context_python" -m unittest \
  train.downstream.tests.test_momentum_event_context \
  train.downstream.tests.test_event_segment_dataset \
  train.downstream.tests.test_evaluate_track_regression_segment_identity \
  train.downstream.tests.test_physics_checkpoint_training
```

CPU reference-scan checks establish backbone feature alignment, not production
CUDA execution or improved momentum resolution. Full-event context increases
backbone token count and can increase peak memory; reduce batch size equally in
both arms if needed for a matched comparison.
