# Contributing to CLAS12 FM research

This is an actively changing research repository, not a stable software release.
The aim is to preserve experimental freedom while making shared results understandable.

## Branches

- `main`: the current **shared, reasonably tested research baseline**. It is not a claim of publication-ready scientific validation.
- `mike`: an independent collaborator branch. Its owner can develop autonomously; changes are brought into shared code by review, not by automatically merging the whole branch.
- `experiment/<topic>`: changes to model behavior, training algorithms, datasets, or adapters that may not work out. Start these from current `main` where practical.
- `archive/<topic>`: historical references, including unsuccessful investigations. Do not modify or delete archive references casually.
- `chore/<topic>`: small documentation, test, or maintenance changes.

A configuration sweep **does not** need its own branch unless it also changes implementation behavior.

## Practical workflow

1. Start a task branch from `main` (or work on a collaborator's independent branch).
2. Make focused commits. Exploratory commits are welcome on experiment branches.
3. Run relevant local tests, if the changes have testable behavior. GPU/HPC and data-dependent tests remain local.
4. Open a pull request when a change is ready to join `main`. Describe the dataset contract, affected experiments, and what was actually tested.
5. Integrate *selected* useful changes from collaborator or experiment branches, preserving credit and commit references.
6. Retire old experiments by recording their purpose and evidence in `docs/BRANCHES.md`; never equate an unmerged branch with a failed scientific hypothesis without checking results.

A pull request is a proposed code change on GitHub; opening one does **not** automatically change `main`.

## Scientific provenance rather than frozen dependencies

COATJAVA, CCDB, simulation, reconstruction and detector representation are all still developing.
Do **not** require the entire project to stay on one frozen revision.
Instead, each dataset or published analysis should record which versions, configurations,
and code commits were actually used; see [PROVENANCE.md](docs/PROVENANCE.md).

New dataset representations and training recipes may coexist. Make input and target
contracts explicit, including physical units, ordering, masking, label meaning, and
geometry/CCDB variation. A version label alone is not a complete specification.

Store generated data, checkpoints, logs, plots, temporary conversions, and private
machine-specific settings outside this repository. Commit reproducible scripts and
configs, but prefer configurable paths and avoid committing large binary results.
`gitignore` rules do not remove files already tracked in Git history.

## Tests and expectations

The lightweight GitHub workflow only checks Python syntax; it **does not** run
Mamba/CUDA training, validate physics, or assert that the code works on real detector
data. Treat a green check as one small signal, not a scientific endorsement.

A bigger automated test suite can be added later when CPU-only dependencies and
fixtures are practical. Keep changes proportional to the research timeline.
