"""Show a text-to-token example and the shape of one training batch."""

import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer

from .data import clean_text
from .loader import TokenDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/week1"))
    parser.add_argument("--text", default="  A little fox\n  found a red ball.  ")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    cleaned = clean_text(args.text)
    encoded = tokenizer.encode(cleaned)
    batch = next(TokenDataset(args.data).batches())
    print(json.dumps({
        "original_text": args.text,
        "cleaned_text": cleaned,
        "tokens": encoded.tokens,
        "token_ids": encoded.ids,
        "decoded_text": tokenizer.decode(encoded.ids),
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "note": "This demonstrates the tokenizer and loader. The language model has not been built yet.",
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
