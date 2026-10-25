"""Convert the Week 1 NumPy batches without changing their target alignment."""

import torch


def to_torch_batch(batch, device="cpu"):
    return {
        "input_ids": torch.as_tensor(batch["input_ids"], dtype=torch.long, device=device),
        "labels": torch.as_tensor(batch["labels"], dtype=torch.long, device=device),
        "attention_mask": torch.as_tensor(batch["attention_mask"], dtype=torch.bool, device=device),
    }
