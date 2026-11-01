"""Generate early text samples from a saved training checkpoint on CPU."""

import argparse
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

from .checkpoint import load_checkpoint
from .data import file_hash, write_json
from .model import BlaudeModel, ModelConfig


def load_model(checkpoint_path, data_dir):
    checkpoint_hash = file_hash(checkpoint_path)
    state = load_checkpoint(checkpoint_path)
    if checkpoint_hash != file_hash(checkpoint_path):
        raise ValueError("Checkpoint changed while loading; retry or use a separate saved copy")
    state["loaded_checkpoint_sha256"] = checkpoint_hash
    tokenizer_path = Path(data_dir) / "tokenizer.json"
    if file_hash(tokenizer_path) != state["settings"]["source_data_sha256"]["tokenizer.json"]:
        raise ValueError("Tokenizer does not match the checkpoint")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = ModelConfig(**state["settings"]["model"])
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the model")
    model = BlaudeModel(config)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, tokenizer, state


def generate(model, tokenizer, prompt, max_new_tokens=80, seed=299):
    """Fixed early-sample protocol: temperature 0.8, top 40, no repetition penalty."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a nonempty text prompt")
    if not 1 <= max_new_tokens <= 512 or not 0 <= seed < 2**63:
        raise ValueError("Use 1-512 new tokens and a nonnegative seed below 2**63")
    ids = tokenizer.encode(prompt).ids
    eos = tokenizer.token_to_id("<|endoftext|>")
    blocked = [tokenizer.token_to_id(name) for name in ("<|pad|>", "<|unk|>")]
    if eos is None or any(value is None for value in blocked):
        raise ValueError("Tokenizer is missing required special tokens")
    if not ids or any(value in ids for value in [eos, *blocked]):
        raise ValueError("Prompt must contain ordinary text, not reserved special tokens")
    initial_length = len(ids)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    stopped_on_eos = False
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Keep the newest context and restart learned positions at zero.
                inputs = torch.tensor([ids[-model.config.context_length:]], dtype=torch.long)
                scores = model(inputs)["logits"][0, -1].clone() / 0.8
                scores[blocked] = -float("inf")
                if not torch.isfinite(scores).any():
                    raise RuntimeError("Model has no finite candidate score")
                values, indices = torch.topk(scores, min(40, int(torch.isfinite(scores).sum())))
                if not torch.isfinite(values).all():
                    raise RuntimeError("Invalid model scores during sampling")
                selected = torch.multinomial(values.softmax(-1), 1, generator=generator).item()
                token = int(indices[selected])
                ids.append(token)
                if token == eos:
                    stopped_on_eos = True
                    break
    finally:
        model.train(was_training)
    return {"prompt": prompt, "text": tokenizer.decode(ids, skip_special_tokens=True),
            "new_token_ids": ids[initial_length:], "new_tokens": len(ids) - initial_length,
            "stopped_on_eos": stopped_on_eos, "seed": seed,
            "temperature": 0.8, "top_k": 40, "max_new_tokens": max_new_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/week1"))
    parser.add_argument("--prompt", default="Once upon a time, there was a little")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=299)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.output and args.output.exists():
        parser.error("Output exists; choose a new sample filename")
    try:
        torch.set_num_threads(args.threads)
        model, tokenizer, state = load_model(args.checkpoint, args.data)
        result = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.seed)
        result.update(checkpoint_step=state["step"], checkpoint_sha256=state["loaded_checkpoint_sha256"])
        # Byte-level tokens can decode to Unicode, including replacement characters
        # in untrained samples. Windows' legacy console encoding may reject them.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(result["text"])
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
