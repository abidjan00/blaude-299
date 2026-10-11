"""Run Week 1: python -m blaude.prepare --stories 20000."""

import argparse
import json
from pathlib import Path
import platform
import sys
from importlib.metadata import version

from .data import (DELIMITER, download_sample, file_hash, prepare_records,
                   split_records, write_json, write_jsonl)
from .loader import TokenDataset
from .tokenizer import encode_split, train_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=int, default=20_000)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=299)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/week1"))
    args = parser.parse_args()
    if args.stories < 20 or args.vocab_size < 259:
        parser.error("Use at least 20 stories and a vocabulary of at least 259")
    if args.output.exists():
        parser.error("Output already exists. Choose another --output directory to preserve the previous run.")
    raw_path, source = download_sample(args.raw_dir, args.stories)
    stories = raw_path.read_text(encoding="utf-8").split(DELIMITER)
    # Only the downloader's trailing whitespace after the final delimiter is dropped.
    records, cleaning, examples = prepare_records(stories[:-1])
    splits = split_records(records, args.seed)
    args.output.mkdir(parents=True)
    for split, rows in splits.items():
        write_jsonl(args.output / f"{split}.jsonl", rows)
    print(f"Retained {len(records):,} unique stories; training tokenizer on train only", flush=True)
    tokenizer = train_tokenizer(args.output / "train.jsonl", args.output / "tokenizer.json", args.vocab_size)
    split_stats = {}
    for split in splits:
        split_stats[split] = encode_split(tokenizer, args.output / f"{split}.jsonl", args.output / split)
        dataset = TokenDataset(args.output, split)
        split_stats[split]["training_windows_at_context_256"] = len(dataset)
        print(f"{split}: {split_stats[split]}", flush=True)

    # Demonstration uses training examples only. Holdouts are not fitted or previewed.
    demo = []
    for row in splits["train"][:3]:
        encoding = tokenizer.encode(row["text"])
        assert tokenizer.decode(encoding.ids) == row["text"]
        demo.append({"story_id": row["id"], "cleaned_text": row["text"],
                     "first_24_tokens": encoding.tokens[:24], "first_24_ids": encoding.ids[:24],
                     "token_count": len(encoding.ids), "round_trip_matches": True})
    training_indices = {row["source_index"] for row in splits["train"]}
    training_examples = [example for i, example in enumerate(examples)
                         if records[i]["source_index"] in training_indices]
    write_json(args.output / "examples.json", {"cleaning": training_examples, "tokenization": demo})
    (args.output / "DATA_NOTICE.md").write_text(
        "# TinyStories data notice\n\nProvider: Ronen Eldan and Yuanzhi Li.\n"
        f"Source: {source['source_url']}\n\nLicense: CDLA-Sharing-1.0 ({source['license_url']}).\n\n"
        "These files were changed: a source prefix was selected, whitespace and Unicode were "
        "normalized, quality heuristics and duplicate removal were applied, stories were split "
        "into train/validation/test sets, and derived token arrays were created. Preserve this "
        "notice and the upstream attribution when sharing data.\n", encoding="utf-8")
    manifest = {
        "source": source, "seed": args.seed, "cleaning": cleaning,
        "split_method": "seeded hash order of normalized story IDs; 90/5/5",
        "filter_settings": {"min_words": 20, "max_words": 1000, "min_ascii_letter_fraction": 0.95},
        "splits": split_stats, "tokenizer_training_split": "train",
        "vocab_size": tokenizer.get_vocab_size(), "context_length": 256,
        "special_token_ids": {name: tokenizer.token_to_id(name) for name in ["<|pad|>", "<|unk|>", DELIMITER]},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": version("numpy"), "tokenizers": version("tokenizers")},
        "files": {p.name: file_hash(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    }
    write_json(args.output / "manifest.json", manifest)
    batch = next(TokenDataset(args.output).batches())
    print(json.dumps({"output": str(args.output), "vocab_size": tokenizer.get_vocab_size(),
                      "batch_shapes": {key: list(value.shape) for key, value in batch.items()}}, indent=2))


if __name__ == "__main__":
    main()
