# Downstream adapters

Choose the task first:

| Task | Start here | Public training CLI |
| --- | --- | --- |
| Event-level track finding | [Track-finding guide](TRACK_FINDING.md) | `train_track_finding.py` |
| Track-level momentum regression | [Momentum guide](MOMENTUM.md) | `train_track_regression.py` |

Both support AdapterOnly coordinate inputs or frozen pretrained features.
The [complete worked example](../WORKED_EXAMPLE.md) uses one backbone for both
tasks. These are separate supervised fits; the momentum recipe receives supplied
track membership rather than automatically consuming the finding head's output.

Use [Training workflow](../TRAINING_RECIPES.md) for commands,
[Training options](TRAINING_OPTIONS.md) for controls, and
[Evaluation and plots](eval/README.md) for interpreting outputs. After a successful
single run, follow [From single runs to campaigns](campaign/GETTING_STARTED.md).

[Target statistics](TARGET_STATISTICS.md) and
[checkpoint policy](PHYSICS_CHECKPOINTS.md) describe momentum-specific contracts.
[Tuning](tuning/GETTING_STARTED.md) and [probes](probes/README.md) are optional follow-up
workflows, not prerequisites for fitting an adapter.
