import math
import numpy as np
import pytest

pytest.importorskip("torch", reason="Install requirements-week2.txt")
from blaude.bigram import BigramModel
from blaude.loader import TokenDataset


def test_story_boundaries_unseen_tokens_and_padding(tmp_path):
    np.save(tmp_path / "train.tokens.npy", np.array([3, 4, 2, 4, 2], dtype=np.uint32))
    np.save(tmp_path / "train.offsets.npy", np.array([0, 3, 5], dtype=np.int64))
    train = TokenDataset(tmp_path, context_length=4)
    model = BigramModel(6, alpha=0.5).fit(train)
    assert model.counts.sum() == 3
    assert model.counts[4, 2] == 2 and model.counts[3, 4] == 1
    assert model.counts[2, 4] == 0  # Never cross story endings.
    before = model.counts.copy()
    np.save(tmp_path / "validation.tokens.npy", np.array([5, 4, 2], dtype=np.uint32))
    np.save(tmp_path / "validation.offsets.npy", np.array([0, 3], dtype=np.int64))
    validation = TokenDataset(tmp_path, split="validation", context_length=4)
    metrics = model.evaluate(validation)
    expected = -(math.log(1 / 6) + math.log(2.5 / 5)) / 2
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["tokens"] == 2
    assert np.array_equal(before, model.counts)
