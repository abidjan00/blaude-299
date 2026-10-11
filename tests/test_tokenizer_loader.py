import numpy as np
import pytest
from tokenizers import Tokenizer

from blaude.data import DELIMITER, write_jsonl
from blaude.loader import TokenDataset
from blaude.tokenizer import encode_split, train_tokenizer


def test_tokenizer_learns_only_supplied_train_file_and_round_trips(tmp_path):
    train = tmp_path / "train.jsonl"
    write_jsonl(train, [{"text": "A small fox ran home. " * 10}])
    write_jsonl(tmp_path / "test.jsonl", [{"text": "HOLDOUTONLY " * 100}])
    tokenizer = train_tokenizer(train, tmp_path / "tokenizer.json", 300)
    # Uppercase H never appears in training; it is present only as a base byte.
    assert "HOLDOUTONLY" not in tokenizer.get_vocab()
    assert not any(token.startswith("HO") for token in tokenizer.get_vocab())
    restored = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    for text in ["A small fox ran home.", "New words: café, বাংলা, 😀!", "two  spaces\nnext line"]:
        ids = restored.encode(text).ids
        assert restored.decode(ids) == text
        assert restored.token_to_id("<|unk|>") not in ids
    assert restored.token_to_id("<|pad|>") == 0
    assert restored.token_to_id(DELIMITER) == 2


def test_encoded_offsets_and_eos(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = [{"text": "The cat sat."}, {"text": "The bird flew away."}]
    write_jsonl(source, rows)
    tokenizer = train_tokenizer(source, tmp_path / "tokenizer.json", 300)
    stats = encode_split(tokenizer, source, tmp_path / "train")
    tokens = np.load(tmp_path / "train.tokens.npy")
    offsets = np.load(tmp_path / "train.offsets.npy")
    assert stats["stories"] == 2
    assert offsets[0] == 0 and offsets[-1] == len(tokens)
    for i, row in enumerate(rows):
        sequence = tokens[offsets[i]:offsets[i + 1]].tolist()
        assert sequence[-1] == 2
        assert tokenizer.decode(sequence[:-1]) == row["text"]


def test_targets_are_shifted_without_crossing_stories_and_padding_is_ignored(tmp_path):
    np.save(tmp_path / "train.tokens.npy", np.array([10, 11, 12, 13, 2, 20, 21, 2], dtype=np.uint32))
    np.save(tmp_path / "train.offsets.npy", np.array([0, 5, 8], dtype=np.int64))
    dataset = TokenDataset(tmp_path, context_length=3)
    assert len(dataset) == 3
    assert dataset[0]["input_ids"].tolist() == [10, 11, 12]
    assert dataset[0]["labels"].tolist() == [11, 12, 13]
    assert dataset[1]["input_ids"].tolist() == [13, 0, 0]
    assert dataset[1]["labels"].tolist() == [2, -100, -100]
    assert dataset[2]["input_ids"].tolist() == [20, 21, 0]
    assert dataset[2]["labels"].tolist() == [21, 2, -100]
    assert dataset[2]["attention_mask"].tolist() == [True, True, False]
    batches = list(dataset.batches(batch_size=2))
    assert [batch["input_ids"].shape for batch in batches] == [(2, 3), (1, 3)]
    a = list(dataset.batches(2, shuffle=True, seed=42))
    b = list(dataset.batches(2, shuffle=True, seed=42))
    assert all(np.array_equal(x["labels"], y["labels"]) for x, y in zip(a, b))
    with pytest.raises(ValueError, match="positive"):
        list(dataset.batches(0))


def test_bad_offsets_fail_early(tmp_path):
    np.save(tmp_path / "train.tokens.npy", np.array([10, 11, 2], dtype=np.uint32))
    np.save(tmp_path / "train.offsets.npy", np.array([0, 6], dtype=np.int64))
    with pytest.raises(ValueError, match="offsets"):
        TokenDataset(tmp_path)
