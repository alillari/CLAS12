# CLAS12 Foundation Model Research

This repository contains early-stage research code for applying foundation-model
methods to CLAS12-related tasks. The implementation is under active development
and should not be treated as a complete, stable, or publication-ready release.

The current codebase builds on the FM4NPP project, including model components,
training structure, and examples from that work. We are adapting and extending
those ideas for CLAS12 use cases as the project evolves.

## Status

This repository is a working research fork. Interfaces, scripts, configuration
files, model choices, and data assumptions may change without notice.

At this stage, users should expect to review paths, configuration files, and
training scripts before running experiments. Some inherited FM4NPP examples or
settings may still need to be updated for CLAS12-specific workflows.

## Start here

Task entry points: [track finding](train/downstream/TRACK_FINDING.md) and
[momentum regression](train/downstream/MOMENTUM.md). The
[downstream README](train/downstream/README.md) maps the shared tools.

1. [Environment and data setup](SETUP.md).
2. [Complete worked example](train/WORKED_EXAMPLE.md): pretrain one backbone,
   train both frozen-backbone adapters, generate stats, evaluate and plot.
3. [Training workflow](train/TRAINING_RECIPES.md): standalone adapter instructions.
4. [Training options](train/downstream/TRAINING_OPTIONS.md): supported controls,
   defaults, interactions and implementation-fixed choices.
5. [Target statistics](train/downstream/TARGET_STATISTICS.md): normalization,
   cohort selection, file generation and checkpoint consistency.
6. [Backbone recipes](train/BACKBONE_RECIPES.md): Mike's Chinchilla sweep and
   Mamba1/2/3 pretraining settings. [Mamba support](train/MAMBA_SUPPORT.md) covers
   block options, runtime requirements and checkpoint compatibility.
   [Transformer support](train/TRANSFORMER_SUPPORT.md) covers causal backbones
   and independently configured bidirectional adapters.
7. [Remote edgexpert jobs](train/REMOTE_RUNS.md): source snapshots, container
   paths, detached launch, monitoring, retries and artifact retrieval.
8. [Evaluation and plots](train/downstream/eval/README.md),
   [single runs → campaigns](train/downstream/campaign/GETTING_STARTED.md), and
   [tuning](train/downstream/tuning/GETTING_STARTED.md).

The [adapter presets](scripts/configs/clas12_adapter_recipes.yaml) provide explicit
30k-update profiles. Track finding uses a width-256 attention decoder with
16 queries and zero Mamba layers. Momentum uses a width-256 head with one Mamba2
refinement layer and attention pooling. AdapterOnly trains from coordinate
embeddings; `--usepretrain --pretrained_ckpt ...` selects pretrained features.

## Upstream Work and Attribution

This work builds on FM4NPP: Foundation Models for Nuclear and Particle Physics.
We are developing this project with guidance from BNL AI experts, including
Dr. David Park, Dr. Shinjae Yoo and Dr. Huan-Hsin Tseng.

Relevant upstream resources:

- FM4NPP OpenReview: https://openreview.net/forum?id=qaI3cLFsiX
- TPCpp-10M dataset: https://doi.org/10.5281/zenodo.16970029
- TPCpp-10M dataset paper: https://www.sciencedirect.com/science/article/pii/S2352340925011060

If you use inherited FM4NPP methods, code structure, or datasets, cite the
corresponding FM4NPP and TPCpp-10M works as appropriate.

```bibtex
@article{park2025fm4npp,
  title={FM4NPP: A Scaling Foundation Model for Nuclear and Particle Physics},
  author={Park, David and Li, Shuhang and Huang, Yi and Luo, Xihaier and Yu, Haiwang and Go, Yeonju and Pinkenburg, Christopher and Lin, Yuewei and Yoo, Shinjae and Osborn, Joseph and others},
  journal={arXiv preprint arXiv:2508.14087},
  year={2025}
}

@article{tpcpp10m2025,
  title={TPCpp-10M: Simulated proton-proton collisions in a Time Projection Chamber for AI Foundation Models},
  author={Li, Shuhang and Huang, Yi and Park, David and Luo, Xihaier and Yu, Haiwang and Go, Yeonju and Pinkenburg, Christopher and Lin, Yuewei and Yoo, Shinjae and Osborn, Joseph and Roland, Christof and Huang, Jin and Ren, Yihui},
  journal={arXiv preprint arXiv:2509.05792},
  year={2025}
}
```

## Repository Layout

```text
.
|-- fm4npp/             # FM4NPP-derived model, dataset, and utility code
|-- train/              # Training and downstream task scripts
|-- scripts/            # Configuration and run scripts
|-- dev_scripts/        # Calibration, recovered recipes and isolated experiment runners
|-- SETUP.md            # Environment and CLAS12 data setup
|-- requirements.txt    # Python dependencies
`-- README.md           # Repository and storage conventions
```

## Training workflows

The shared `main` branch brings together [CLAS12 pretraining](train/pretrain/README.md)
and [downstream adapter campaigns](train/downstream/campaign/README.md).
It is the common starting point for ongoing CLAS12 research; new implementation
experiments may live in separate branches. The pretraining integration preserves
Mike's established recipes and 50-point selection; known trainer limitations
remain documented rather than silently changing the experimental baseline.

## Setup

Follow [SETUP.md](SETUP.md), then select a task in the
[training guide](train/TRAINING_RECIPES.md). Training requires a compatible
PyTorch/CUDA/Mamba environment and access to the RaggedMmap data. The unpinned
`requirements.txt` is an inherited dependency inventory, not a tested lockfile.

See [Mamba support](train/MAMBA_SUPPORT.md) for independent backbone/adapter
selection, Mamba2 pretraining, Mamba3 SISO, and checkpoint/runtime requirements.
See [transformer support](train/TRANSFORMER_SUPPORT.md) for causal pretraining,
bidirectional adapters, and independent NoPE/RoPE sequence encoding.

## Research-product storage

Keep source code, tests, reusable configurations, required calibration inputs,
and software documentation in this repository. Store experiment checkpoints,
logs, predictions, plots, generated statistics, campaign outputs, and dated
research reports outside the checkout.

The usual sibling storage directory is `/home/alessio/ML-work/result_deep_storage`:

```bash
export CLAS12_ARTIFACT_ROOT=/home/alessio/ML-work/result_deep_storage
```

Use `campaigns/` for complete campaign artifacts, `research_reports/` for research
writeups, and `downstream_log/` for standalone training outputs under that root.
The downstream training CLIs and experiment helpers default their runtime logs
to the sibling `result_deep_storage/downstream_log` independently of the current
working directory; `CLAS12_ARTIFACT_ROOT` overrides that default. Explicit CLI
paths still take precedence. Check the selected YAML's `artifact_root`,
`checkpoint_dir`, and analysis `output_dir` too: existing configurations may
deliberately point to another external location, including a mounted host.

For inherited entrypoints or custom scripts, pass external output paths explicitly.
Ignore rules prevent accidental addition of common output directories; they do
not move files or stop a program from writing into the checkout. Python caches
may be recreated by the interpreter and remain ignored. To put them outside too,
set `PYTHONPYCACHEPREFIX="$CLAS12_ARTIFACT_ROOT/python_cache"` before running Python.

`dev_scripts/clas12_band_calibration.json` is a versioned runtime dependency of
`fm4npp/hilbert.py`, so it remains with the code. Generic how-to documentation also
belongs here; experiment-specific findings belong in external research reports.

## Notes for Contributors

Because this project is moving quickly, keep top-level documentation focused on
the current repository state and avoid presenting experimental code as a stable
release. When adapting inherited FM4NPP components, preserve attribution and make
CLAS12-specific changes explicit in code, configuration, or documentation.
