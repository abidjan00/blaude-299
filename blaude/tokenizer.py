"""Learn a byte-level BPE vocabulary from the training split only."""

from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import DELIMITER, SPECIAL_TOKENS, read_jsonl


def train_tokenizer(train_path, output_path, vocab_size=4096):
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("Vocabulary must fit all 256 byte values and special tokens")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, show_progress=False,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator((row["text"] for row in read_jsonl(train_path)), trainer=trainer)
    tokenizer.save(str(output_path))
    return tokenizer


def encode_split(tokenizer, jsonl_path, output_prefix):
    """Save uint32 IDs and story offsets; append one explicit EOS per story."""
    all_ids, offsets = [], [0]
    eos_id = tokenizer.token_to_id(DELIMITER)
    if eos_id is None:
        raise ValueError("Tokenizer is missing its end-of-story token")
    for row in read_jsonl(jsonl_path):
        all_ids.extend(tokenizer.encode(row["text"], add_special_tokens=False).ids)
        all_ids.append(eos_id)
        offsets.append(len(all_ids))
    np.save(f"{output_prefix}.tokens.npy", np.asarray(all_ids, dtype=np.uint32))
    np.save(f"{output_prefix}.offsets.npy", np.asarray(offsets, dtype=np.int64))
    return {"stories": len(offsets) - 1, "tokens_including_eos": len(all_ids)}
