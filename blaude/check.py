"""Audit prepared artifacts without fitting anything on held-out data."""

import argparse
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

from .data import DELIMITER, file_hash, read_jsonl
from .loader import TokenDataset


def check(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    seen = set()
    result = {}
    for split, expected in manifest["splits"].items():
        rows = list(read_jsonl(directory / f"{split}.jsonl"))
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate story IDs within or between splits")
        seen.update(ids)
        dataset = TokenDataset(directory, split)
        if len(rows) != expected["stories"] or len(dataset.offsets) != len(rows) + 1:
            raise ValueError("Story count mismatch")
        if len(dataset.tokens) != expected["tokens_including_eos"]:
            raise ValueError("Token count mismatch")
        if len(dataset) != expected["training_windows_at_context_256"]:
            raise ValueError("Window count mismatch")
        if np.any(dataset.tokens >= tokenizer.get_vocab_size()):
            raise ValueError("Out-of-vocabulary token ID")
        if np.any(dataset.tokens == tokenizer.token_to_id("<|unk|>")):
            raise ValueError("Unexpected unknown token in byte-level encoding")
        if np.any(dataset.tokens[dataset.offsets[1:] - 1] != tokenizer.token_to_id(DELIMITER)):
            raise ValueError("Missing end-of-story token")
        # Check encoding alignment on every story without displaying holdout text.
        for i, row in enumerate(rows):
            sequence = dataset.tokens[dataset.offsets[i]:dataset.offsets[i + 1] - 1].tolist()
            if tokenizer.decode(sequence) != row["text"]:
                raise ValueError(f"Encoding round-trip mismatch in {split}")
        # Inspect one real batch from each split, including exact padding behavior.
        batch = next(dataset.batches(batch_size=8))
        if not np.array_equal(batch["labels"] == -100, ~batch["attention_mask"]):
            raise ValueError("Padding mask mismatch")
        result[split] = {"stories": len(rows), "tokens": len(dataset.tokens), "windows": len(dataset)}
    return {"status": "passed", "checked_files": len(manifest["files"]), "splits": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="data/week1")
    args = parser.parse_args()
    print(json.dumps(check(args.directory), indent=2))
