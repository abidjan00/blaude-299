"""Versioned CPU checkpoints. Write a replacement before replacing a good file."""

from pathlib import Path
import os
import torch


def save_checkpoint(path, model, optimizer, settings, step, total_tokens):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    state = {"format_version": 1, "settings": settings, "step": step,
             "total_training_targets": total_tokens,
             "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "torch_rng_state": torch.get_rng_state()}
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"settings", "step", "total_training_targets", "model_state_dict",
                "optimizer_state_dict", "torch_rng_state"}
    if state.get("format_version") != 1 or not required <= state.keys():
        raise ValueError("Use a Week 3 training checkpoint, not the Week 2 diagnostic snapshot")
    return state
