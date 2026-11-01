from types import SimpleNamespace
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

torch = pytest.importorskip("torch", reason="Install requirements-week2.txt")
from blaude.checkpoint import save_checkpoint
from blaude.data import file_hash, write_json
from blaude.evaluate import evaluate_checkpoint
from blaude.generate import generate, load_model
from blaude.model import BlaudeModel, ModelConfig


@pytest.fixture
def tokenizer():
    result = Tokenizer(models.WordLevel({"<|pad|>": 0, "<|unk|>": 1,
        "<|endoftext|>": 2, "cat": 3, "dog": 4, "ran": 5, "home": 6}, unk_token="<|unk|>"))
    result.add_special_tokens(["<|pad|>", "<|unk|>", "<|endoftext|>"])
    result.pre_tokenizer = pre_tokenizers.Whitespace()
    return result


class FixedModel(torch.nn.Module):
    def __init__(self, token=3):
        super().__init__()
        self.config = SimpleNamespace(context_length=3)
        self.token = token
        self.inputs = []

    def forward(self, input_ids):
        assert not torch.is_grad_enabled()
        self.inputs.append(input_ids.clone())
        scores = torch.zeros((*input_ids.shape, 7))
        scores[..., :2] = 999  # Reserved tokens must still be excluded.
        scores[..., self.token] = 100
        return {"logits": scores}


def test_sampling_truncates_context_blocks_specials_and_restores_mode(tokenizer):
    model = FixedModel().train()
    before = torch.get_rng_state().clone()
    a = generate(model, tokenizer, "cat dog ran home", max_new_tokens=5)
    b = generate(model, tokenizer, "cat dog ran home", max_new_tokens=5)
    assert a == b and a["new_token_ids"] == [3] * 5
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert all(batch.shape == (1, 3) for batch in model.inputs)
    assert model.inputs[0].tolist() == [[4, 5, 6]]
    assert not a["stopped_on_eos"]


def test_sampling_stops_at_eos(tokenizer):
    result = generate(FixedModel(token=2).eval(), tokenizer, "cat", max_new_tokens=20)
    assert result["new_token_ids"] == [2] and result["stopped_on_eos"]
    assert "<|endoftext|>" not in result["text"]


@pytest.mark.parametrize("prompt,limit", [("", 5), ("   ", 5), ("<|pad|>", 5), ("cat", 0), ("cat", 513)])
def test_invalid_prompts_and_limits(tokenizer, prompt, limit):
    with pytest.raises(ValueError):
        generate(FixedModel(), tokenizer, prompt, limit)


def test_checkpoint_loading_and_full_validation_without_test_data(tmp_path, tokenizer):
    torch.set_num_threads(1)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    for split in ("train", "validation"):
        np.save(tmp_path / f"{split}.tokens.npy", np.array([3, 4, 2, 5, 2], dtype=np.uint32))
        np.save(tmp_path / f"{split}.offsets.npy", np.array([0, 3, 5], dtype=np.int64))
    hashes = {p.name: file_hash(p) for p in tmp_path.iterdir()}
    write_json(tmp_path / "manifest.json", {"files": hashes, "vocab_size": 7})
    from dataclasses import asdict
    model = BlaudeModel(ModelConfig(vocab_size=7, context_length=4, d_model=8,
                                   n_heads=2, n_layers=1, d_ff=16, dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, {"model": asdict(model.config), "source_data_sha256": hashes}, 1, 3)
    restored, _, _ = load_model(path, tmp_path)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    report = evaluate_checkpoint(path, tmp_path, tmp_path / "metrics.json", batch_size=1, threads=1)
    assert report["validation"]["tokens"] == 3 and report["validation"]["full_split"]
    assert not report["test_split_used"]
    with (tmp_path / "tokenizer.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="Tokenizer does not match"):
        load_model(path, tmp_path)


def test_windows_console_handles_unicode_samples(tmp_path, monkeypatch):
    import io
    import sys
    import blaude.generate as module
    path = tmp_path / "model.pt"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(sys, "argv", ["generate", "--checkpoint", str(path)])
    monkeypatch.setattr(module, "load_model", lambda *args: (None, None, {"step": 0, "loaded_checkpoint_sha256": "fixture"}))
    monkeypatch.setattr(module, "generate", lambda *args: {"text": "A cat \ufffd café বাংলা"})
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    module.main()
    stream.flush()
    assert "বাংলা" in buffer.getvalue().decode("utf-8")


def test_checkpoint_replaced_during_load_is_rejected(monkeypatch):
    import blaude.generate as module
    hashes = iter(["before", "after"])
    monkeypatch.setattr(module, "file_hash", lambda path: next(hashes))
    monkeypatch.setattr(module, "load_checkpoint", lambda path: {})
    with pytest.raises(ValueError, match="changed while loading"):
        module.load_model("live.pt", "data")
