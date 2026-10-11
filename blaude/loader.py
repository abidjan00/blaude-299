"""NumPy next-token batches, ready to convert to PyTorch tensors in Week 2."""

from pathlib import Path
import numpy as np


class TokenDataset:
    """Every story is chunked separately; targets never cross story boundaries.

    Short chunks are padded. Targets at padding positions are -100, the usual
    ignored target value for cross-entropy. A causal attention mask belongs in
    the future model; attention_mask here marks real tokens versus padding.
    """

    def __init__(self, directory, split="train", context_length=256, pad_id=0):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if context_length < 1:
            raise ValueError("Context length must be positive")
        self.context_length = context_length
        self.pad_id = pad_id
        self.tokens = np.load(Path(directory) / f"{split}.tokens.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(Path(directory) / f"{split}.offsets.npy", mmap_mode="r", allow_pickle=False)
        if (self.tokens.ndim != 1 or self.offsets.ndim != 1 or len(self.offsets) < 2
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.tokens)
                or np.any(np.diff(self.offsets) < 2)):
            raise ValueError("Invalid token data or story offsets")
        self.windows = [
            (start, min(context_length, int(end) - start - 1))
            for begin, end in zip(self.offsets[:-1], self.offsets[1:])
            for start in range(int(begin), int(end) - 1, context_length)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        start, length = self.windows[index]
        inputs = np.full(self.context_length, self.pad_id, dtype=np.int64)
        targets = np.full(self.context_length, -100, dtype=np.int64)
        mask = np.zeros(self.context_length, dtype=np.bool_)
        inputs[:length] = self.tokens[start:start + length]
        targets[:length] = self.tokens[start + 1:start + length + 1]
        mask[:length] = True
        return {"input_ids": inputs, "labels": targets, "attention_mask": mask}

    def batches(self, batch_size=8, shuffle=False, seed=299):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        indices = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            rows = [self[int(i)] for i in indices[start:start + batch_size]]
            yield {key: np.stack([row[key] for row in rows]) for key in rows[0]}
