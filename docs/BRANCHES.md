# Branch inventory and historical experiments

Recorded on 2026-10-27. References below represent the observed GitHub state
at the time of the audit, **not** a claim that the corresponding science succeeded
or failed. Some branches will evolve; their listed commit SHAs are the historical
audit references.

## Shared work

| Branch | Audited head | Purpose / intended disposition |
| --- | --- | --- |
| `main` | `ffe8d6a33a364a465c203766c4bae0cc7310f787` | Original FM4NPP baseline at audit; intended to become shared CLAS12 research baseline after review |
| `adapter-dev` | `3b9170780c16f9fbe9e999fd0ea9037d34bc8267` | Main downstream, adapters, evaluation, campaigns and tests; planned foundation for new `main` |
| `integration-adapter-pretrain` | `47722abdcf41bb68f7dd706195a45d68c31dd47c` | Mike's CLAS12 pretraining integrated with common adapters; changes require reconciliation with later `adapter-dev` commits |
| `mike` | `38c82ec06265a530a4efccc853bba085d0742875` | Collaborator's independent work; do not rewrite or automatically merge |
| `archive/upstream-fm4npp-2026-02-28` | `ffe8d6a33a364a465c203766c4bae0cc7310f787` | Historical preservation of upstream baseline |

## Historical or experimental work

| Branch | Audited head | Investigation | Result/status |
| --- | --- | --- | --- |
| `alex` | `9e8e38f07fa4` | Earlier downstream/evaluation implementation; fully ancestral to `adapter-dev` | Superseded as development branch; specific scientific result not audited |
| `momentum-adapter-experiments` | `1e35394407f6` | Full-event frozen backbone context for target-track momentum regression | Experimental; scientific outcome to be documented |
| `momentum-membership-mask` | `96d926be9b3b` | Full-event context with target-track membership indicator | Experimental; scientific outcome to be documented |
| `momentum-cartesian-factorial` | `30a302de60d5` | Cartesian momentum comparison/factorial runner; includes membership branch commits | Experimental; scientific outcome to be documented |

The factorial branch contains the membership branch plus two additional commits.
The `alex` branch has no commits unique from `adapter-dev`. **Do not delete
or rename these historical branches merely because an experiment was inconclusive.**

Before archiving an experiment, record:

- Question / hypothesis and the exact reference commit
- Input dataset and schema, model, training and evaluation configuration
- Result or evidence (including null results), and whether it was actually executed
- Why the change was or was not incorporated into `main`
- Links to external figures, reports, and checkpoint manifests, if available

## Integration cautions

The current `adapter-dev` includes simplifications from October 2026 that are
not present in the September experiment forks. Merging an entire older experiment
branch can restore retired code or undo fixes. Prefer small, reviewed patches,
cherry-picks where appropriate, and targeted tests.

The pretraining integration itself explicitly leaves some trainer and distributed
behavior unresolved. Bringing its pretraining-specific changes into a current
baseline does not imply that its older downstream files should replace newer ones.

The `mike` branch contains tracked generated files under `data/`.
Its existing history is preserved; do not bring its entire data tree onto `main`.
