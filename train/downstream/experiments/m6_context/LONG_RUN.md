# Longer m6 context comparison

Run `bash train/downstream/experiments/m6_context/run_long.sh` in the isolated checkout/container, setting `CLAS12_CONTEXT_RUNS` to a fresh output directory.

The cohort comprises the first 100,000 distinct pretrain events with eligible truth tracks, retaining all eligible tracks (12 to 100 hits) in every selected event. Preparation saves exact event/segment identities and verifies both dataset implementations produce that cohort. The derived track count is passed to the existing track-oriented trainer and statistics tool. It is not a 100,000-track limit.

Both arms start fresh from the same frozen campaign 4 m6 backbone, seed 42, batch 32, and train 30,000 steps. Early stopping cannot truncate that budget. The cosine cycle is extended to 30,000 steps with 300 warmup steps; validation uses 50,000 tracks every 1,000 steps. Physics selection remains provisional with disabled bias/tail guardrails, matching the pilot.

After both training runs, the standard evaluator processes 500,000 test-split tracks per selected checkpoint. These include the 50,000 checkpoint-validation tracks: this is a larger evaluation, not an independent untouched test. The existing v6 data product has no separate validation split. Preserve per-bin widths, bias, tails and angular metrics when comparing, and do not infer seed robustness from this single matched pair.
