"""Week 3, part 1: CPU training, held-out validation, and progress logs.

This first version does not save model weights or resume interrupted training.
"""

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from .data import file_hash, write_json
from .loader import TokenDataset
from .model import BlaudeModel, ModelConfig
from .torch_data import to_torch_batch


def evaluate(model, dataset, batch_size, max_batches=10):
    """Token-weighted loss on a fixed validation prefix; zero means all batches."""
    if batch_size < 1 or max_batches < 0:
        raise ValueError("batch_size must be positive and max_batches nonnegative")
    was_training = model.training
    model.eval()
    total_loss, total_tokens, batch_count = 0.0, 0, 0
    try:
        with torch.no_grad():
            for numpy_batch in dataset.batches(batch_size, shuffle=False):
                batch = to_torch_batch(numpy_batch)
                output = model(**batch)
                count = int((batch["attention_mask"] & (batch["labels"] != -100)).sum())
                loss = output["loss"].item()
                if not math.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss")
                total_loss += loss * count
                total_tokens += count
                batch_count += 1
                if max_batches and batch_count >= max_batches:
                    break
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("Validation data contains no targets")
    loss = total_loss / total_tokens
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 700 else None,
            "tokens": total_tokens, "batches": batch_count,
            "full_split": batch_count == math.ceil(len(dataset) / batch_size)}


def load_data(data_dir, context_length):
    """Verify training/validation artifacts; never open the reserved test split."""
    data_dir = Path(data_dir)
    if not (data_dir / "manifest.json").exists():
        raise ValueError("Prepare the data first: python -m blaude.prepare --stories 20000")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    names = ["tokenizer.json"] + [f"{split}.{kind}.npy"
            for split in ("train", "validation") for kind in ("tokens", "offsets")]
    hashes = {}
    for name in names:
        hashes[name] = file_hash(data_dir / name)
        if hashes[name] != manifest["files"].get(name):
            raise ValueError(f"Data checksum mismatch: {name}")
    tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size() != manifest["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match the manifest")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("Tokenizer is missing the padding token")
    datasets = [TokenDataset(data_dir, split=split, context_length=context_length, pad_id=pad_id)
                for split in ("train", "validation")]
    return *datasets, tokenizer.get_vocab_size(), pad_id, hashes


def run_training(data_dir, output_dir, *, steps=50, batch_size=4, eval_every=25,
                 validation_batches=10, learning_rate=0.0003, seed=299,
                 threads=2, preset="small"):
    if min(steps, batch_size, eval_every, threads) < 1 or validation_batches < 0:
        raise ValueError("steps, batch size, evaluation interval, and threads must be positive; validation batches must be nonnegative")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if preset not in {"small", "full"}:
        raise ValueError("preset must be small or full")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output already exists; choose a new --output folder")
    context_length = 64 if preset == "small" else 256
    train_data, validation_data, vocab_size, pad_id, hashes = load_data(data_dir, context_length)
    config = ModelConfig(vocab_size=vocab_size, pad_id=pad_id)
    if preset == "small":
        config = replace(config, context_length=64, d_model=64, n_layers=2, d_ff=256)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = BlaudeModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "stage": "Week 3 part 1", "model": asdict(config), "parameters": model.parameter_count(),
        "steps": steps, "batch_size": batch_size, "eval_every": eval_every,
        "validation_batches": validation_batches, "validation_selection": "fixed prefix, no shuffle",
        "learning_rate": learning_rate, "weight_decay": 0.01, "gradient_clip": 1.0,
        "seed": seed, "threads": threads, "preset": preset, "device": "cpu",
        "torch_version": str(torch.__version__), "source_data_sha256": hashes,
        "source_code_sha256": {name: file_hash(Path(__file__).parent / name)
                               for name in ("train.py", "model.py", "loader.py", "torch_data.py")},
        "weights_saved": False, "resumable": False,
    }
    write_json(output_dir / "config.json", settings)
    started = time.perf_counter()
    print(f"Week 3 part 1: {model.parameter_count():,} parameters; {steps} CPU steps", flush=True)
    print("Logs only in this version; model weights are not saved.", flush=True)
    if validation_batches:
        print(f"Validation uses at most {validation_batches} fixed batches, not necessarily the full split.", flush=True)
    history = []
    total_train_tokens = 0
    fields = ["step", "epoch", "train_loss", "train_tokens", "validation_loss",
              "validation_perplexity", "validation_tokens", "validation_full_split", "elapsed_seconds"]
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def record(step, epoch, train_loss, train_tokens):
            validation = evaluate(model, validation_data, batch_size, validation_batches)
            row = {"step": step, "epoch": epoch, "train_loss": train_loss,
                   "train_tokens": train_tokens, "validation_loss": validation["loss"],
                   "validation_perplexity": validation["perplexity"],
                   "validation_tokens": validation["tokens"],
                   "validation_full_split": validation["full_split"],
                   "elapsed_seconds": round(time.perf_counter() - started, 3)}
            writer.writerow(row)
            handle.flush()
            history.append(row)
            train_text = "before training" if train_loss is None else f"train loss {train_loss:.4f}"
            print(f"Step {step}: {train_text}; validation loss {validation['loss']:.4f}", flush=True)

        record(0, 0, None, 0)
        epoch = 1
        batches = iter(train_data.batches(batch_size, shuffle=True, seed=seed))
        window_loss, window_tokens = 0.0, 0
        for step in range(1, steps + 1):
            try:
                numpy_batch = next(batches)
            except StopIteration:
                epoch += 1
                batches = iter(train_data.batches(batch_size, shuffle=True, seed=seed + epoch - 1))
                numpy_batch = next(batches)
            model.train()
            batch = to_torch_batch(numpy_batch)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            count = int((batch["attention_mask"] & (batch["labels"] != -100)).sum())
            total_train_tokens += count
            window_loss += loss.item() * count
            window_tokens += count
            if step % eval_every == 0 or step == steps:
                record(step, epoch, window_loss / window_tokens, window_tokens)
                window_loss, window_tokens = 0.0, 0
    summary = {"status": "completed", "steps_completed": steps, "epochs_started": epoch,
               "total_training_targets": total_train_tokens, "initial_validation_loss": history[0]["validation_loss"],
               "final_validation_loss": history[-1]["validation_loss"],
               "final_validation_full_split": history[-1]["validation_full_split"],
               "weights_saved": False, "resumable": False,
               "limitation": "Short training check; no claim of text-generation quality. Test split unused."}
    write_json(output_dir / "summary.json", summary)
    print(f"Completed. Progress logs saved in {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/week1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/week3-part1"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--validation-batches", type=int, default=10, help="0 evaluates the entire validation split")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=299)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preset", choices=["small", "full"], default="small")
    args = vars(parser.parse_args())
    try:
        run_training(args.pop("data"), args.pop("output"), **args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
