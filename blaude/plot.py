"""Create a loss chart from saved CSV logs without changing the run."""

import argparse
import csv
from pathlib import Path


def plot_history(history_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(history_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("History contains no measurements")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    training = [row for row in rows if row["train_loss"]]
    if training:
        ax.plot([int(r["step"]) for r in training], [float(r["train_loss"]) for r in training],
                marker="o", label="Training (interval average)")
    full = all(r["validation_full_split"].lower() == "true" for r in rows)
    ax.plot([int(r["step"]) for r in rows], [float(r["validation_loss"]) for r in rows],
            marker="o", label="Validation (full split)" if full else "Validation (fixed sample)")
    ax.set(title="Blaude training progress", xlabel="Optimizer step", ylabel="Next-token cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new chart filename")
    plot_history(args.history, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
