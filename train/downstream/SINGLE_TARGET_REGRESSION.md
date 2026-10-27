# Historical single-target momentum checkpoints

New `p`, `theta` and `phi` training ablations and their presets are retired.
Use the Cartesian `mom` baseline described in [the checkpoint policy](PHYSICS_CHECKPOINTS.md).
The original recipes are preserved in Git at `b566b54`.

Historical one-output checkpoints can still be evaluated using their saved
model YAML and matching target statistics. Their output contracts remain:

| Task | Output | Decoding / diagnostic residual |
| --- | --- | --- |
| `p` | log momentum in the stored target unit | `exp(output)`; Huber on log difference |
| `theta` | polar angle in radians | unwrapped angular difference |
| `phi` | azimuth in radians | wrapped difference via `atan2(sin(delta), cos(delta))` |

Normalization remains identity. Zero momentum is invalid for `p` and `theta`;
zero transverse momentum is invalid for `phi`. Repeated phi labels use a circular
mean. Loading checks task metadata so scalar axes cannot be exchanged.

The evaluator reports only the quantity learned by each historical head, at the
MC truth entrance frame. It does not invent a full momentum vector or transport
the prediction to DOCA. Cached paired `p`/`theta` residual plots still work and
require matching track identities.
