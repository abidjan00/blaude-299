"""Checks for validation isolation, progress accounting, and reproducible training."""

import csv
import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models

torch = pytest.importorskip("torch", reason="Install requirements-week2.txt to run training tests")
from blaude.data import file_hash, write_json
from blaude.loader import TokenDataset
from blaude.model import BlaudeModel, ModelConfig
from blaude.torch_data import to_torch_batch
from blaude.train import evaluate, run_training


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    tokenizer = Tokenizer(models.WordLevel(
        {"<|pad|>": 0, "<|unk|>": 1, "<|endoftext|>": 2,
         "train_a": 3, "train_b": 4, "validation_a": 5, "validation_b": 6},
        unk_token="<|unk|>"))
    tokenizer.save(str(directory / "tokenizer.json"))
    for split, tokens, offsets in [
        ("train", [3, 4, 3, 2, 4, 3, 2], [0, 4, 7]),
        ("validation", [5, 6, 5, 6, 2, 6, 2], [0, 5, 7]),
    ]:
        np.save(directory / f"{split}.tokens.npy", np.array(tokens, dtype=np.uint32))
        np.save(directory / f"{split}.offsets.npy", np.array(offsets, dtype=np.int64))
    write_json(directory / "manifest.json", {
        "vocab_size": 7, "files": {path.name: file_hash(path) for path in directory.iterdir()}})
    # No test split exists: training must never need it.
    return directory


def test_validation_weights_tokens_and_preserves_weights_gradients_and_mode(prepared):
    torch.set_num_threads(1)
    torch.manual_seed(10)
    model = BlaudeModel(ModelConfig(vocab_size=7, context_length=8, d_model=16,
                                   n_heads=2, n_layers=1, d_ff=32, dropout=0.2))
    dataset = TokenDataset(prepared, split="validation", context_length=8)
    # Preserve existing training gradients as well as parameters during evaluation.
    model(**to_torch_batch(next(dataset.batches(2))))["loss"].backward()
    weights = [p.detach().clone() for p in model.parameters()]
    gradients = [p.grad.clone() for p in model.parameters()]
    metrics = evaluate(model, dataset, batch_size=1, max_batches=0)
    assert model.training
    assert metrics["tokens"] == 5 and metrics["batches"] == 2 and metrics["full_split"]
    model.eval()
    with torch.no_grad():
        manual_loss = model(**to_torch_batch(next(dataset.batches(2))))["loss"].item()
    assert metrics["loss"] == pytest.approx(manual_loss, abs=1e-6)
    capped = evaluate(model, dataset, batch_size=1, max_batches=1)
    assert not model.training
    assert capped["tokens"] == 4 and not capped["full_split"]
    for parameter, weight, gradient in zip(model.parameters(), weights, gradients):
        assert torch.equal(parameter, weight)
        assert torch.equal(parameter.grad, gradient)


def test_training_visits_batches_restarts_epochs_and_keeps_validation_separate(prepared, tmp_path, monkeypatch):
    calls, seeds = [], []
    original_forward = BlaudeModel.forward
    original_batches = TokenDataset.batches

    def observed_forward(self, input_ids, attention_mask=None, labels=None):
        calls.append((self.training, torch.is_grad_enabled(), input_ids[attention_mask].tolist()))
        return original_forward(self, input_ids, attention_mask, labels)

    def observed_batches(self, batch_size=8, shuffle=False, seed=299):
        if shuffle:
            seeds.append(seed)
        return original_batches(self, batch_size, shuffle, seed)

    monkeypatch.setattr(BlaudeModel, "forward", observed_forward)
    monkeypatch.setattr(TokenDataset, "batches", observed_batches)
    output = tmp_path / "run"
    summary = run_training(prepared, output, steps=5, batch_size=1, eval_every=2,
                           validation_batches=0, threads=1)
    assert seeds == [299, 300, 301]
    train_calls = [ids for training, gradients, ids in calls if training and gradients]
    assert len(train_calls) == 5
    assert sorted(map(len, train_calls[:2])) == [2, 3]
    assert all(set(ids) <= {3, 4} for ids in train_calls)
    validation_calls = [(gradients, ids) for training, gradients, ids in calls if not training]
    assert len(validation_calls) == 8
    assert all(not gradients and set(ids) <= {5, 6} for gradients, ids in validation_calls)
    rows = list(csv.DictReader((output / "history.csv").open()))
    assert [int(row["step"]) for row in rows] == [0, 2, 4, 5]
    assert rows[0]["train_loss"] == ""
    assert sum(int(row["train_tokens"]) for row in rows) == summary["total_training_targets"]
    assert summary["epochs_started"] == 3 and summary["steps_completed"] == 5
    assert summary["final_validation_full_split"]
    assert not summary["weights_saved"] and not summary["resumable"]
    assert json.loads((output / "summary.json").read_text()) == summary
    assert {p.name for p in output.iterdir()} == {"config.json", "history.csv", "summary.json"}


def test_fixed_seed_reproduces_losses(prepared, tmp_path):
    results = []
    for name in ("first", "second"):
        result = run_training(prepared, tmp_path / name, steps=3, eval_every=2, threads=1)
        results.append(result)
    assert results[0] == results[1]


def test_checksum_failure_and_existing_output_preserve_files(prepared, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep this")
    with pytest.raises(ValueError, match="already exists"):
        run_training(prepared, output)
    assert marker.read_text() == "keep this"
    with (prepared / "validation.tokens.npy").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_training(prepared, tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"batch_size": 0}, {"eval_every": 0}, {"validation_batches": -1},
    {"learning_rate": float("nan")}, {"learning_rate": -0.1}, {"preset": "unknown"},
])
def test_invalid_settings_fail_before_creating_output(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_training(tmp_path / "missing", tmp_path / "run", **kwargs)
    assert not (tmp_path / "run").exists()
