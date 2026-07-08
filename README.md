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

## Upstream Work and Attribution

This work builds on FM4NPP: Foundation Models for Nuclear and Particle Physics.
We are developing this project with guidance from BNL AI experts, including
Dr. David Park and Dr. Shinjae Yoo.

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
|-- fig/                # Figures and visual assets
|-- SETUP.md            # Setup notes inherited from the upstream structure
|-- requirements.txt    # Python dependencies
`-- example_usage.py    # Experimental usage example
```

## Setup

The inherited setup notes are in `SETUP.md`, but they may not fully reflect the
current CLAS12 research workflow. Review configuration paths and scripts before
launching training jobs.

Typical dependencies include Python, PyTorch, CUDA-capable hardware for training,
and Mamba-related packages. See `requirements.txt` and the training scripts for
the current working assumptions.

## Notes for Contributors

Because this project is moving quickly, keep top-level documentation focused on
the current repository state and avoid presenting experimental code as a stable
release. When adapting inherited FM4NPP components, preserve attribution and make
CLAS12-specific changes explicit in code, configuration, or documentation.
