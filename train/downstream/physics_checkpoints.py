"""Truth-binned checkpoint diagnostics and constrained, non-scalar selection.

This module has no model/trainer dependency. Residuals are signed physical
quantities; quantiles use NumPy's linear interpolation and never fitted widths.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "clas12_physics_checkpoints_v1"
METRICS = ("W_macro", "B_macro", "B_worst", "T_macro")
LIMITS = ("B_macro", "B_worst", "T_macro")
MOMENTUM_TASKS = {"mom", "momentum", "p", "theta", "phi", "p_phi_theta", "pt_phi_eta"}


def selection_mode(params):
    mode = params.get("checkpoint_selection", "physics" if params.get("task", "mom") in MOMENTUM_TASKS else "validation_loss")
    if mode not in {"physics", "validation_loss"}:
        raise ValueError("checkpoint_selection must be physics or validation_loss")
    return mode


def resolve_config(task, options=None):
    options = dict(options or {})
    residual = options.get("residual", {"theta": "theta", "phi": "phi_wrapped"}.get(task, "p_relative"))
    quantity = options.get("bin_quantity", {"theta": "theta", "phi": "phi"}.get(task, "p"))
    if residual not in {"p_relative", "p_absolute", "theta", "phi_wrapped"}:
        raise ValueError(f"Unknown physics residual {residual!r}")
    if quantity not in {"p", "theta", "phi"}:
        raise ValueError(f"Unknown truth bin quantity {quantity!r}")
    if task in {"p", "theta", "phi"}:
        learned = "phi" if residual == "phi_wrapped" else "p" if residual.startswith("p_") else residual
        if learned != task:
            raise ValueError(f"task={task} cannot predict residual={residual}")
    defaults = {"p": np.arange(.25, 3.01, .25), "theta": np.linspace(0, np.pi, 7), "phi": np.linspace(-np.pi, np.pi, 13)}
    edges = np.asarray(options.get("bin_edges", defaults[quantity]), dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("physics bin_edges must be finite and strictly increasing")
    config = {
        "residual": residual, "bin_quantity": quantity, "bin_edges": edges.tolist(),
        "bin_unit": "GeV" if quantity == "p" else "rad",
        "residual_unit": "fraction" if residual == "p_relative" else "GeV" if residual == "p_absolute" else "rad",
        "momentum_scale_to_gev": float(options.get("momentum_scale_to_gev", .001)),
        "tail_threshold": float(options.get("tail_threshold", .1 if residual.startswith("p_") else .05)),
        "min_bin_entries": int(options.get("min_bin_entries", 200)),
        "min_valid_bins": int((len(edges)-1) if options.get("min_valid_bins") is None else options["min_valid_bins"]),
        "guardrails": {name: (None if (options.get("guardrails") or {}).get(name) is None else float(options["guardrails"][name])) for name in LIMITS},
        "require_configured_guardrails": bool(options.get("require_configured_guardrails", False)),
        "width_tie_atol": float(options.get("width_tie_atol", 1e-6)),
        "width_tie_rtol": float(options.get("width_tie_rtol", 0.)),
    }
    unknown = set(options) - set(config)
    if unknown:
        raise ValueError(f"Unknown physics_checkpoint options: {sorted(unknown)}")
    if config["min_bin_entries"] < 2 or not 1 <= config["min_valid_bins"] <= len(edges)-1:
        raise ValueError("Require min_bin_entries >= 2 and 1 <= min_valid_bins <= number of bins")
    for key in ("tail_threshold", "momentum_scale_to_gev", "width_tie_atol", "width_tie_rtol"):
        value = config[key]
        if not np.isfinite(value) or value < 0 or (key in {"tail_threshold", "momentum_scale_to_gev"} and value == 0):
            raise ValueError(f"Invalid physics_checkpoint {key}")
    for key, value in config["guardrails"].items():
        if value is not None and (not np.isfinite(value) or value < 0 or (key == "T_macro" and value > 1)):
            raise ValueError(f"Invalid guardrail {key}")
    if set(options.get("guardrails") or {}) - set(LIMITS):
        raise ValueError("Guardrails must use B_macro, B_worst, T_macro")
    return config


def kinematics(native, task, scale):
    """Native means denormalized output; the p-only native output is log(p)."""
    native = np.asarray(native, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if task == "p":
            return {"p": np.exp(native[:, 0]) * scale}
        if task in {"theta", "phi"}:
            value = native[:, 0]
            return {task: np.arctan2(np.sin(value), np.cos(value)) if task == "phi" else value}
        if task == "p_phi_theta":
            p, cosphi, sinphi, theta = native.T
            phi = np.arctan2(sinphi, cosphi)
            phi[(cosphi == 0) & (sinphi == 0)] = np.nan
            return {"p": p * scale, "theta": theta, "phi": phi}
        if task == "pt_phi_eta":
            pt, cosphi, sinphi, eta = native.T
            phi = np.arctan2(sinphi, cosphi)
            phi[(cosphi == 0) & (sinphi == 0)] = np.nan
            return {"p": pt * np.cosh(eta) * scale, "theta": np.arctan2(np.ones_like(eta), np.sinh(eta)), "phi": phi}
        if task in {"mom", "momentum"}:
            px, py, pz = (native[:, :3] * scale).T
            pt = np.hypot(px, py)
            p = np.hypot(pt, pz)
            return {"p": p, "theta": np.where(p > 0, np.arctan2(pt, pz), np.nan),
                    "phi": np.where(pt > 0, np.arctan2(py, px), np.nan)}
    raise ValueError(f"No physics checkpoint decoder for task {task!r}")


def residual_arrays(prediction, truth, task, config, truth_xyz=None):
    """Binning is exclusively derived from truth, including angular-only heads."""
    pred = kinematics(prediction, task, config["momentum_scale_to_gev"])
    true = kinematics(truth, task, config["momentum_scale_to_gev"])
    if truth_xyz is not None:
        true_bins = kinematics(truth_xyz, "mom", config["momentum_scale_to_gev"])
    else:
        true_bins = true
    quantity = config["bin_quantity"]
    if quantity not in true_bins:
        raise ValueError(f"Binning {task} by true {quantity} requires raw truth_xyz")
    kind = config["residual"]
    key = "p" if kind.startswith("p_") else "phi" if kind == "phi_wrapped" else "theta"
    a, b = pred[key].copy(), true[key].copy()
    if key == "p":
        # Negative or zero physical p is an invalid prediction, not a track to
        # drop silently or rescue by taking its absolute value.
        a[a <= 0] = np.nan
        b[b <= 0] = np.nan
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = (a-b)/b if kind == "p_relative" else a-b
        if kind == "phi_wrapped":
            residual = np.arctan2(np.sin(residual), np.cos(residual))
    bins = true_bins[quantity]
    return bins, residual, np.isfinite(b) & np.isfinite(bins)


def residual_statistics(values, threshold):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("q16", "q50", "q84", "w68", "abs_median_bias", "tail_fraction")}
    q16, q50, q84 = np.quantile(values.astype(np.longdouble), [.16, .5, .84], method="linear")
    return dict(q16=float(q16), q50=float(q50), q84=float(q84), w68=float(q84/2-q16/2),
                abs_median_bias=float(abs(q50)), tail_fraction=float(np.mean(np.abs(values) > threshold)))


def summarize(bins, residual, config, truth_valid=None):
    bins, residual = np.asarray(bins, dtype=float), np.asarray(residual, dtype=float)
    if bins.ndim != 1 or bins.shape != residual.shape:
        raise ValueError("Truth bin values and residuals must be equal-length vectors")
    valid_truth = np.isfinite(bins)
    if truth_valid is not None:
        valid_truth &= np.asarray(truth_valid, dtype=bool)
    finite = valid_truth & np.isfinite(residual)
    edges = np.asarray(config["bin_edges"])
    in_range = valid_truth & (bins >= edges[0]) & (bins <= edges[-1])
    per_bin = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = valid_truth & (bins >= low) & ((bins <= high) if index == len(edges)-2 else (bins < high))
        n_truth, n_finite = int(mask.sum()), int((mask & finite).sum())
        valid = n_truth >= config["min_bin_entries"] and n_finite == n_truth
        per_bin.append(dict(index=index, low=float(low), high=float(high), n_truth=n_truth,
            n_finite=n_finite, valid=valid,
            invalid_reason=None if valid else "insufficient_occupancy" if n_truth < config["min_bin_entries"] else "nonfinite_predictions",
            **residual_statistics(residual[mask & finite] if valid else [], config["tail_threshold"])))
    usable = [row for row in per_bin if row["valid"]]
    support = hashlib.sha256()
    support.update(np.asarray(valid_truth, dtype=np.uint8).tobytes())
    support.update(np.where(valid_truth, bins, 0.).astype("<f8").tobytes())
    result = dict(
        n_samples=len(bins), n_valid_truth=int(valid_truth.sum()), n_invalid_truth=int((~valid_truth).sum()),
        n_invalid_predictions=int((valid_truth & ~np.isfinite(residual)).sum()),
        n_outside_bins=int((valid_truth & ~in_range).sum()), n_valid_bins=len(usable),
        valid_bin_indices=[r["index"] for r in usable], truth_support_hash=support.hexdigest(),
        per_bin=per_bin, inclusive={"n": int(finite.sum()), **residual_statistics(residual[finite], config["tail_threshold"])},
        W_macro=None, B_macro=None, B_worst=None, T_macro=None,
    )
    if usable:
        result.update(W_macro=float(np.mean([r["w68"] for r in usable], dtype=np.longdouble)),
            B_macro=float(np.mean([r["abs_median_bias"] for r in usable], dtype=np.longdouble)),
            B_worst=float(max(r["abs_median_bias"] for r in usable)),
            T_macro=float(np.mean([r["tail_fraction"] for r in usable])))
    return result


def summarize_native(prediction, truth, task, config, truth_xyz=None):
    bins, residual, valid = residual_arrays(prediction, truth, task, config, truth_xyz)
    result = summarize(bins, residual, config, valid)
    # Freeze the complete ordered truth sample, not just its bin occupancies.
    digest = hashlib.sha256(result["truth_support_hash"].encode())
    for array in (truth, truth_xyz):
        if array is not None:
            array = np.asarray(array, dtype="<f8")
            digest.update(str(array.shape).encode())
            digest.update(np.isfinite(array).astype(np.uint8).tobytes())
            digest.update(np.where(np.isfinite(array), array, 0.).tobytes())
    result["truth_support_hash"] = digest.hexdigest()
    return result


def assess(record, config):
    finite = all(record.get(key) is not None and np.isfinite(record[key]) for key in METRICS)
    loss = record.get("validation_loss")
    finite &= record.get("validation_loss_finite", loss is None or np.isfinite(loss))
    checks = {
        "finite_summary": bool(finite),
        "sufficient_bins": record["n_valid_bins"] >= config["min_valid_bins"],
        "finite_predictions": record["n_invalid_predictions"] == 0,
    }
    guardrails = {key: {"limit": limit, "enabled": limit is not None,
        "passed": None if limit is None else bool(finite and record[key] <= limit)}
        for key, limit in config["guardrails"].items()}
    configured = all(g["enabled"] for g in guardrails.values())
    record.update(summary_checks=checks, guardrails=guardrails,
        guardrails_configured=configured, summary_valid=all(checks.values()),
        passes_guardrails=all(g["passed"] is not False for g in guardrails.values()) and (configured or not config["require_configured_guardrails"]))
    record["eligible"] = record["summary_valid"] and record["passes_guardrails"]
    record["rejection_reasons"] = ([key for key, passed in checks.items() if not passed]
        + [key for key, guard in guardrails.items() if guard["passed"] is False]
        + (["guardrails_unconfigured"] if config["require_configured_guardrails"] and not configured else []))
    return record


def rank_checkpoints(records, config):
    """Pareto ignores guardrails; selection never falls back to rejected rows."""
    if len({r["truth_support_hash"] for r in records}) > 1:
        raise ValueError("Validation truth sample/bin values changed across checkpoints")
    for record in records:
        assess(record, config)
        record.update(on_pareto_frontier=False, selected=False)
    valid = [r for r in records if r["summary_valid"]]
    if len({tuple(r["valid_bin_indices"]) for r in valid}) > 1:
        raise ValueError("Cannot rank checkpoints with different valid-bin support")
    for a in valid:
        a["on_pareto_frontier"] = not any(
            all(b[k] <= a[k] for k in METRICS) and any(b[k] < a[k] for k in METRICS)
            for b in valid if b is not a)
    acceptable = [r for r in valid if r["eligible"]]
    selected = None
    if acceptable:
        minimum = min(r["W_macro"] for r in acceptable)
        tolerance = max(config["width_tie_atol"], config["width_tie_rtol"] * abs(minimum))
        tied = [r for r in acceptable if r["W_macro"] <= minimum + tolerance]
        selected = min(tied, key=lambda r: (r["B_macro"], r["B_worst"], r["T_macro"], r["W_macro"], r["step"], r["epoch"]))
        selected["selected"] = True
    return selected


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class CheckpointSummary:
    def __init__(self, config, output_dir, records=None):
        self.config, self.output_dir = config, Path(output_dir)
        self.records = list(records or [])
        self.selected = rank_checkpoints(self.records, config)

    def add(self, summary, *, step, epoch, validation_loss, checkpoint):
        if self.records and summary["truth_support_hash"] != self.records[0]["truth_support_hash"]:
            raise ValueError("Validation truth sample/bin values changed across checkpoints")
        if any(r["step"] == step and r["epoch"] == epoch for r in self.records):
            raise ValueError(f"Duplicate checkpoint step/epoch: {step}/{epoch}")
        row = dict(summary, step=int(step), epoch=int(epoch),
            validation_loss=float(validation_loss), validation_loss_finite=bool(np.isfinite(validation_loss)), checkpoint=str(checkpoint))
        self.records.append(row)
        self.selected = rank_checkpoints(self.records, self.config)
        return row

    def write(self, plots=True):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        selected = self.selected
        status = ("selected" if selected and selected["guardrails_configured"] else ("selected_provisional_partial_guardrails" if any(v is not None for v in self.config["guardrails"].values()) else "selected_provisional_guardrails_disabled")
                  if selected else "guardrails_unconfigured" if self.config["require_configured_guardrails"] and any(v is None for v in self.config["guardrails"].values())
                  else "no_checkpoint_passed")
        candidates = sorted([r for r in self.records if r["summary_valid"]], key=lambda r: tuple(r[k] for k in METRICS))[:5]
        # Nonfinite validation loss remains explicitly invalid but JSON is strict.
        serializable = [{**r, "validation_loss": r["validation_loss"] if r["validation_loss"] is not None and np.isfinite(r["validation_loss"]) else None} for r in self.records]
        payload = dict(schema=SCHEMA, config=self.config, selection_status=status,
            selected_checkpoint=selected["checkpoint"] if selected else None,
            selected_step=selected["step"] if selected else None,
            best_candidates=[{key:r[key] for key in ("step", "epoch", "checkpoint", *METRICS, "rejection_reasons")} for r in candidates],
            pareto_frontier=[r["checkpoint"] for r in self.records if r["on_pareto_frontier"]], checkpoints=serializable)
        atomic_json(self.output_dir / "checkpoint_summary.json", payload)
        atomic_json(self.output_dir / "pareto_frontier.json", {"schema": SCHEMA, "config": self.config,
            "checkpoints": [r for r in serializable if r["on_pareto_frontier"]]})
        fields = ["step", "epoch", "validation_loss", *METRICS, "n_valid_bins", "summary_valid", "passes_guardrails", "on_pareto_frontier", "selected", "checkpoint"]
        with (self.output_dir / "checkpoint_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(serializable)
        table = [f"Selection: **{status}**", "",
                 f"Residual: `{self.config['residual']}` ({self.config['residual_unit']}); "
                 f"truth bins: `{self.config['bin_quantity']}` ({self.config['bin_unit']}); "
                 f"tail: abs(residual) > {self.config['tail_threshold']}; "
                 f"minimum occupancy: {self.config['min_bin_entries']}.", "", "| Step | Epoch | Val loss | W macro | B macro | B worst | T macro | Valid bins | Passed | Pareto | Selected |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|"]
        def number(value):
            return "—" if value is None or not np.isfinite(value) else f"{value:.6g}"
        for r in self.records:
            values = [str(r["step"]), str(r["epoch"]), number(r["validation_loss"]),
                      *[number(r[k]) for k in METRICS], str(r["n_valid_bins"]),
                      *["yes" if r[k] else "no" for k in ("eligible", "on_pareto_frontier", "selected")]]
            table.append("| " + " | ".join(values) + " |")
        table += ["", "Guardrails: `" + json.dumps(self.config["guardrails"]) + "`",
                  "", "Best width candidates (not fallback selections):"]
        table += [f"- Step {r['step']}: W={r['W_macro']:.6g}; rejection reasons: {r['rejection_reasons']}" for r in candidates]
        (self.output_dir / "checkpoint_summary.md").write_text("\n".join(table) + "\n")
        if plots and self.records:
            self.plot()
        return payload

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        rows = sorted(self.records, key=lambda r: r["step"])
        labels = {key: f"{key} [{self.config['residual_unit']}]" for key in METRICS[:3]}
        labels['T_macro'] = 'T_macro [fraction]'

        fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
        for ax, key in zip(axes.flat, ("validation_loss", *METRICS, "n_valid_bins")):
            values = [np.nan if r[key] is None else r[key] for r in rows]
            ax.plot([r["step"] for r in rows], values, ".-", color="0.5")
            for r, value in zip(rows, values):
                ax.scatter(r["step"], value, color="tab:green" if r["eligible"] else "tab:red", s=16)
                if r["selected"]:
                    ax.scatter(r["step"], value, marker="*", s=140, color="tab:blue", zorder=5)
            limit = self.config["guardrails"].get(key)
            if limit is not None:
                ax.axhline(limit, color="tab:red", ls="--", lw=1)
            if key == "n_valid_bins":
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.axhline(self.config["min_valid_bins"], color="tab:red", ls="--", lw=1)
            ax.set(xlabel="Optimizer step", ylabel=labels.get(key, key))
            ax.grid(alpha=.2)
        fig.suptitle("Validation diagnostics: green=eligible, red=rejected, star=selected")
        fig.savefig(self.output_dir / "checkpoint_metrics_by_step.png", dpi=150)
        fig.savefig(self.output_dir / "checkpoint_metrics_by_step.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, key in zip(axes, LIMITS):
            for r in rows:
                if r["summary_valid"]:
                    ax.scatter(r["W_macro"], r[key], marker="*" if r["selected"] else "o",
                        color="tab:blue" if r["on_pareto_frontier"] else "0.7", s=90 if r["selected"] else 25)
                    if r["on_pareto_frontier"]:
                        ax.annotate(str(r["step"]), (r["W_macro"], r[key]), fontsize=7)
            ax.margins(x=.15, y=.15)
            ax.set(xlabel=labels["W_macro"], ylabel=labels[key])
            ax.grid(alpha=.2)
        fig.suptitle("Four-objective Pareto frontier (blue), shown in pairwise projections")
        fig.savefig(self.output_dir / "checkpoint_pareto.png", dpi=150)
        fig.savefig(self.output_dir / "checkpoint_pareto.pdf")
        plt.close(fig)


def write_evaluation_summary(output_dir, prediction, truth, task, config, *,
                             truth_xyz=None, checkpoint=None, metadata=None):
    """Evaluate a checkpoint without selecting on held-out data.

    Training selection flags are read from its validation ledger if available;
    they are never inferred from this evaluation sample.
    """
    metadata = metadata or {}
    config = metadata.get("physics_checkpoint_config") or config
    row = summarize_native(prediction, truth, task, config, truth_xyz)
    loss = metadata.get("current_loss")
    row.update(step=metadata.get("global_step"), epoch=metadata.get("epoch"),
               checkpoint=str(checkpoint) if checkpoint is not None else None,
               validation_loss=float(loss) if loss is not None and np.isfinite(loss) else None,
               validation_loss_finite=loss is None or bool(np.isfinite(loss)))
    assess(row, config)
    row.update(selected=None, on_pareto_frontier=None)
    ledger = metadata.get("physics_checkpoint_summary_path")
    if ledger and Path(ledger).is_file():
        history = json.loads(Path(ledger).read_text())
        matches = [r for r in history["checkpoints"] if r["step"] == row["step"] and r["epoch"] == row["epoch"]]
        if len(matches) == 1:
            row.update({key: matches[0][key] for key in ("selected", "on_pareto_frontier")})
    payload = dict(schema=SCHEMA, purpose="evaluation_diagnostics_only", config=config,
                   selection_flags_source="training_validation_ledger" if row["selected"] is not None else "unavailable",
                   checkpoint=row)
    atomic_json(Path(output_dir) / "physics_checkpoint_summary.json", payload)
    return payload
