"""Validate the architecture and overfit one tiny training batch on a CPU.

This is a diagnostic experiment, not a full trainer or a general-purpose model.
Run: python -m blaude.week2 --data data/week1 --output artifacts/week2
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import torch
from tokenizers import Tokenizer

from .data import file_hash, write_json
from .loader import TokenDataset
from .model import BlaudeModel, ModelConfig
from .torch_data import to_torch_batch


def evaluate_batch(model, batch):
    model.eval()
    with torch.no_grad():
        output = model(**batch)
        valid = batch["attention_mask"] & (batch["labels"] != -100)
        accuracy = (output["logits"].argmax(-1)[valid] == batch["labels"][valid]).float().mean().item()
        return {"loss": output["loss"].item(), "next_token_accuracy": accuracy}


def run_experiment(data_dir, output_dir, steps=200, seed=299, threads=2):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if steps < 1 or threads < 1:
        raise ValueError("steps and threads must be positive")
    if output_dir.exists():
        raise ValueError("Output already exists; choose a new --output directory to preserve that run")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Prepared data is missing. Run python -m blaude.prepare --stories 20000 first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Only the training split is used here; validation and test stay out of fitting.
    data_hashes = {}
    for name in ("tokenizer.json", "train.tokens.npy", "train.offsets.npy"):
        data_hashes[name] = file_hash(data_dir / name)
        if data_hashes[name] != manifest["files"][name]:
            raise ValueError(f"Training artifact checksum mismatch: {name}")
    tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size() != manifest["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match the data manifest")

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), pad_id=tokenizer.token_to_id("<|pad|>"))
    full_batch = to_torch_batch(next(TokenDataset(data_dir, context_length=config.context_length).batches(2)))
    model = BlaudeModel(config).eval()
    result = model(**full_batch)
    result["loss"].backward()
    finite_gradients = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    if not finite_gradients:
        raise RuntimeError("Architecture check found missing or non-finite gradients")
    with torch.no_grad():
        # Alter only the suffix. Every earlier logit should remain identical.
        cut = min(8, full_batch["input_ids"].shape[1] - 1)
        changed = full_batch["input_ids"].clone()
        changed[:, cut:] = (changed[:, cut:] + 1) % config.vocab_size
        changed_logits = model(changed, full_batch["attention_mask"])["logits"]
        difference = (changed_logits[:, :cut] - result["logits"][:, :cut]).abs().max().item()
    if difference > 1e-6:
        raise RuntimeError(f"Future tokens changed earlier predictions: {difference}")
    architecture = {
        "config": asdict(config), "parameters": model.parameter_count(),
        "input_shape": list(full_batch["input_ids"].shape),
        "logits_shape": list(result["logits"].shape),
        "initial_batch_loss": result["loss"].item(),
        "all_gradients_finite": finite_gradients,
        "future_token_max_logit_difference": difference,
    }
    print(f"Full model: {architecture['parameters']:,} parameters; logits {architecture['logits_shape']}", flush=True)
    print(f"Causal check: maximum change in earlier logits = {difference:.9f}", flush=True)
    del model, result, changed_logits

    # Use a smaller architecture with the same components to keep the diagnostic fast.
    tiny_config = replace(config, context_length=64, d_model=64, n_layers=2, d_ff=256, dropout=0.0)
    tiny_dataset = TokenDataset(data_dir, context_length=tiny_config.context_length)
    batch = to_torch_batch(next(tiny_dataset.batches(2, shuffle=True, seed=seed)))
    torch.manual_seed(seed)
    tiny = BlaudeModel(tiny_config)
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=0.003, weight_decay=0.0)
    initial = evaluate_batch(tiny, batch)
    history = [{"step": 0, **initial}]
    print(f"Tiny model: {tiny.parameter_count():,} parameters; initial loss {initial['loss']:.4f}", flush=True)
    for step in range(1, steps + 1):
        tiny.train()
        optimizer.zero_grad(set_to_none=True)
        loss = tiny(**batch)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tiny.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == steps:
            metrics = evaluate_batch(tiny, batch)
            history.append({"step": step, **metrics})
            print(f"Step {step:3}: loss {metrics['loss']:.4f}; fixed-batch accuracy {metrics['next_token_accuracy']:.1%}", flush=True)

    final = history[-1]
    passed = final["loss"] < initial["loss"] * 0.1 and final["next_token_accuracy"] >= 0.95
    with torch.no_grad():
        predictions = tiny(**batch)["logits"].argmax(-1)
    valid = batch["labels"] != -100
    output_dir.mkdir(parents=True)
    # This diagnostic snapshot has no optimizer state; resumable training is Week 3.
    snapshot_path = output_dir / "tiny_model.pt"
    torch.save({"config": asdict(tiny_config), "model_state_dict": tiny.state_dict(),
                "purpose": "Week 2 tiny-batch memorization diagnostic", "seed": seed}, snapshot_path)
    restored_data = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    restored = BlaudeModel(ModelConfig(**restored_data["config"]))
    restored.load_state_dict(restored_data["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        restore_difference = (restored(**batch)["logits"] - tiny(**batch)["logits"]).abs().max().item()
    if restore_difference != 0:
        raise RuntimeError("Reloaded snapshot did not reproduce the original logits")
    report = {
        "status": "passed" if passed else "failed",
        "experiment": "Week 2 architecture checks and tiny training-batch memorization",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "threads": threads, "platform": platform.platform()},
        "seed": seed, "architecture": architecture,
        "tiny_batch": {
            "config": asdict(tiny_config), "parameters": tiny.parameter_count(),
            "split": "train", "input_shape": list(batch["input_ids"].shape),
            "real_target_count": int(valid.sum()), "steps": steps,
            "learning_rate": 0.003, "initial": initial, "final": final,
            "history": history,
            "pass_criteria": {"final_loss_below_initial_fraction": 0.1, "minimum_next_token_accuracy": 0.95},
            "first_16_target_ids": batch["labels"][valid][:16].tolist(),
            "first_16_predicted_ids": predictions[valid][:16].tolist(),
        },
        "snapshot_reload_max_logit_difference": restore_difference,
        "source_data_sha256": data_hashes,
        "source_code_sha256": {name: file_hash(Path(__file__).parent / name)
                               for name in ("model.py", "torch_data.py", "week2.py", "loader.py")},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitation": "Accuracy is teacher-forced on the same tiny batch used for fitting. It does not measure generalization or free text generation.",
    }
    write_json(output_dir / "metrics.json", report)
    (output_dir / "loss_history.csv").write_text(
        "step,loss,next_token_accuracy\n" + "".join(
            f"{row['step']},{row['loss']:.9f},{row['next_token_accuracy']:.9f}\n" for row in history), encoding="utf-8")
    print(f"Week 2 diagnostic: {report['status']}. Results saved in {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/week1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/week2"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=299)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        report = run_experiment(args.data, args.output, args.steps, args.seed, args.threads)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if report["status"] != "passed":
        raise SystemExit("The tiny batch did not meet the learning criteria; inspect metrics.json.")


if __name__ == "__main__":
    main()
