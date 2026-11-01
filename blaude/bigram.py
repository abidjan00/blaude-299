"""Count-based next-token baseline, fitted only on training stories."""

import argparse
import math
from pathlib import Path
import numpy as np

from .data import write_json
from .train import load_data


class BigramModel:
    def __init__(self, vocab_size, alpha=0.1):
        if vocab_size < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Vocabulary and smoothing alpha must be positive")
        self.alpha = alpha
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.uint64)

    def fit(self, dataset):
        self.counts.fill(0)
        for begin, end in zip(dataset.offsets[:-1], dataset.offsets[1:]):
            story = dataset.tokens[int(begin):int(end)]
            if np.any(story >= len(self.counts)):
                raise ValueError("Training token outside vocabulary")
            # Do not count the last token of one story with the next story's first.
            np.add.at(self.counts, (story[:-1], story[1:]), 1)
        return self

    def evaluate(self, dataset, batch_size=4, max_batches=0):
        if batch_size < 1 or max_batches < 0:
            raise ValueError("Invalid evaluation batch settings")
        totals = self.counts.sum(axis=1)
        total_nll, token_count, batches = 0.0, 0, 0
        for batch in dataset.batches(batch_size, shuffle=False):
            valid = batch["attention_mask"] & (batch["labels"] != -100)
            previous, following = batch["input_ids"][valid], batch["labels"][valid]
            probabilities = ((self.counts[previous, following] + self.alpha) /
                             (totals[previous] + self.alpha * len(self.counts)))
            total_nll -= float(np.log(probabilities).sum())
            token_count += len(previous)
            batches += 1
            if max_batches and batches >= max_batches:
                break
        if not token_count:
            raise ValueError("No evaluation targets")
        loss = total_nll / token_count
        return {"loss": loss, "perplexity": math.exp(loss), "tokens": token_count,
                "full_split": batches == math.ceil(len(dataset) / batch_size)}


def run_baseline(data_dir, output_dir, alpha=0.1, context_length=64, batch_size=4, validation_batches=0):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output exists; choose a new folder")
    train, validation, vocab, _, hashes = load_data(data_dir, context_length)
    model = BigramModel(vocab, alpha).fit(train)
    metrics = model.evaluate(validation, batch_size, validation_batches)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "bigram.npz", counts=model.counts, alpha=alpha)
    report = {"model": "add-alpha count bigram", "fit_split": "train", "evaluation_split": "validation",
              "alpha": alpha, "vocab_size": vocab, "context_length": context_length,
              "batch_size": batch_size, "validation_batches": validation_batches,
              "training_pairs": int(model.counts.sum()), "validation": metrics,
              "source_data_sha256": hashes, "test_split_used": False}
    write_json(output_dir / "metrics.json", report)
    print(f"Bigram validation loss: {metrics['loss']:.4f}; {metrics['tokens']:,} targets; full split: {metrics['full_split']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/week1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bigram"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=0)
    args = vars(parser.parse_args())
    try:
        run_baseline(args.pop("data"), args.pop("output"), **args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
