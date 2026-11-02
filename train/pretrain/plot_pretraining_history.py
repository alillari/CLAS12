#!/usr/bin/env python3
"""Plot the maintained pretraining CSV (split,step,loss,lr) without retraining."""

import argparse
import csv
import math
from pathlib import Path


def read_history(path):
    rows = []
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        if not {"split", "step", "loss", "lr"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected pretraining CSV columns: split,step,loss,lr")
        for index, row in enumerate(reader, 2):
            if row["split"] not in {"train", "val"}:
                raise ValueError(f"Unexpected split at line {index}: {row['split']!r}")
            step, loss, lr = int(row["step"]), float(row["loss"]), float(row["lr"])
            if step < 0 or not math.isfinite(loss) or not math.isfinite(lr):
                raise ValueError(f"Invalid step/loss/lr at line {index}")
            rows.append((row["split"], step, loss, lr))
    if not rows or not any(row[0] == "val" for row in rows):
        raise ValueError("Pretraining history has no validation observations")
    return rows


def plot_history(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (loss_axis, lr_axis) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                           constrained_layout=True)
    for split, label in (("train", "Logged training batch"), ("val", "Validation")):
        selected = sorted((row for row in rows if row[0] == split), key=lambda row: row[1])
        if not selected:
            continue
        steps, losses, rates = zip(*[(row[1], row[2], row[3]) for row in selected])
        loss_axis.plot(steps, losses, marker="." if split == "val" else None, label=label)
        lr_axis.plot(steps, rates, label=label)
        if split == "val":
            best = min(selected, key=lambda row: row[2])
            loss_axis.scatter(best[1], best[2], marker="*", s=130, color="black",
                              label=f"Best validation: {best[2]:.5g} at {best[1]}", zorder=3)
    if all(row[2] > 0 for row in rows):
        loss_axis.set_yscale("log")
    loss_axis.set_ylabel("Coordinate MSE")
    loss_axis.set_title("Pretraining: raw logged observations")
    lr_axis.set(xlabel="Optimizer update", ylabel="Logged learning rate")
    for axis in (loss_axis, lr_axis):
        axis.grid(alpha=.25)
        axis.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"pretraining_history.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Pretraining CSV written beside train.log")
    parser.add_argument("--output-dir", type=Path, help="Default: plots beside the CSV")
    args = parser.parse_args()
    rows = read_history(args.log)
    output = args.output_dir or args.log.resolve().parent / "plots"
    plot_history(rows, output)
    print(f"Wrote pretraining history plots to {output}")


if __name__ == "__main__":
    main()
