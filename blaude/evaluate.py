"""Evaluate a training checkpoint on all validation targets, never the test split."""

import argparse
from pathlib import Path

import torch

from .data import write_json
from .generate import load_model
from .train import evaluate, load_data


def evaluate_checkpoint(checkpoint, data_dir, output, batch_size=8, threads=2):
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new metrics filename")
    if min(batch_size, threads) < 1:
        raise ValueError("Batch size and threads must be positive")
    torch.set_num_threads(threads)
    model, _, state = load_model(checkpoint, data_dir)
    _, validation, _, _, hashes = load_data(data_dir, model.config.context_length)
    if hashes != state["settings"]["source_data_sha256"]:
        raise ValueError("Evaluation data differs from the checkpoint's prepared dataset")
    metrics = evaluate(model, validation, batch_size, max_batches=0)
    report = {"checkpoint_step": state["step"], "checkpoint_sha256": state["loaded_checkpoint_sha256"],
              "parameters": model.parameter_count(), "validation": metrics,
              "source_data_sha256": hashes, "test_split_used": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f"Full validation loss {metrics['loss']:.4f}; perplexity {metrics['perplexity']:.2f}; {metrics['tokens']:,} targets")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/week1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        evaluate_checkpoint(args.checkpoint, args.data, args.output, args.batch_size, args.threads)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
