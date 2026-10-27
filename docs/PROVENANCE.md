# Experimental provenance (living systems, reproducible results)

This project changes rapidly: CLAS12 skims, CVT geometry, reconstruction code,
CCDB assignments, model representations, pretraining recipes, and downstream
adapters may all change during development.

We **do not pin one COATJAVA or CCDB revision for all future experiments**.
Instead, record the actual inputs and revision identifiers for each dataset
and evaluation. This preserves past results without holding back development.

## Per-dataset record (template)

Keep a small YAML or JSON sidecar with each processed dataset **outside** the Git
checkout when datasets are large. Version reusable schema definitions and data
production scripts in Git.

```yaml
dataset_id: example_replace_me
date_created: null
schema_version: null           # e.g., cluster/strip representation revision
data_generation:
  generator: null              # GEMC / generator campaign and config
  signal_sample: null
  background_sample: null      # or null for signal-only
  background_exclusions: []    # e.g., known bad background-file IDs
  background_merge_policy: null
reconstruction:
  coatjava_commit: null        # exact commit used, even if main changes later
  ccdb_variation: null
  ccdb_run: null
  ccdb_assignments: null       # identifiers/snapshot when material
skim:
  repository: null
  commit: null
  configuration: null
  cut_summary: null
features:
  coordinate_units: null
  geometry_reference: null
  token_features: null
  ordering: null
labels:
  segmentation_definition: null
  momentum_truth_surface: null
  background_unmatched_policy: null
splits:
  train: null
  validation: null
  test: null
validation:
  checks: []                  # schema, MC::True, row alignment, occupancy
```

This is a *record of what happened*, not a prescription that every field must be
decided before exploratory runs. Unknown values should remain unknown rather
than silently guessed.

## Per-training/evaluation run

Record the Git commit and dirty-worktree status, resolved YAML/CLI settings, data
ID/schema, seed, checkpoint, objective, training and validation selection, and
evaluation cohort. Keep final test events distinct from tuning/validation.

Existing downstream runners already record useful Git/config metadata. Prefer
building on that capability instead of introducing a second incompatible format.

## Why this matters

Two experiments with the same model architecture may not be comparable if
COATJAVA truth association, geometry assignments, strip features, data selection,
or momentum target definition changed. The experiment record should enable
a later reader to identify *which* changes explain any observed difference.

Update this template as real metadata needs emerge; do not freeze one detector
or training implementation as the permanent project standard.
