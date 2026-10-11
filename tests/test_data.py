import hashlib
import io
import json

import pytest

from blaude import data


def story(number):
    return f"Story {number}. A little fox went to the park with a friend. They played with a red ball and came home happy."


def test_cleaning_duplicates_and_quality_filters():
    records, stats, _ = data.prepare_records([
        story(1), "  " + story(1).upper().replace(" ", "\n ") + "  ",
        "", "too short", story(2) + "\x00", story(3) + " https://example.com",
        story(4) + " <|pad|>", "猫 " * 30,
    ])
    assert len(records) == 1
    assert stats["removed"] == {
        "duplicate": 1, "empty": 1, "word_count": 1, "control_character": 1,
        "link_or_email": 1, "reserved_token": 1, "non_english_script": 1,
    }
    assert data.clean_text("\ufeffＡ cat\r\n  sat.") == "A cat sat."


def test_split_is_disjoint_complete_reproducible_and_order_independent():
    records, _, _ = data.prepare_records(story(i) for i in range(100))
    split = data.split_records(records)
    reversed_split = data.split_records(list(reversed(records)))
    assert split == reversed_split
    ids = {name: {row["id"] for row in rows} for name, rows in split.items()}
    assert [len(ids[name]) for name in ("train", "validation", "test")] == [90, 5, 5]
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert set.union(*ids.values()) == {row["id"] for row in records}
    assert data.split_records(records, seed=300) != split


def test_split_rejects_duplicate_ids_and_too_little_data():
    records, _, _ = data.prepare_records(story(i) for i in range(30))
    with pytest.raises(ValueError, match="duplicate"):
        data.split_records(records + [records[0]])
    with pytest.raises(ValueError, match="20"):
        data.split_records(records[:5])


def test_download_stops_at_complete_stories_and_checks_cached_bytes(tmp_path, monkeypatch):
    content = (story(1) + data.DELIMITER + story(2) + data.DELIMITER + "unfinished").encode()

    def fake_open(request, timeout):
        url = getattr(request, "full_url", request)
        return io.BytesIO(b"license: cdla-sharing-1.0" if url.endswith("README.md") else content)

    monkeypatch.setattr(data, "urlopen", fake_open)
    path, metadata = data.download_sample(tmp_path, count=2)
    assert path.read_text(encoding="utf-8").count(data.DELIMITER) == 2
    assert "unfinished" not in path.read_text(encoding="utf-8")
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: pytest.fail("Cache should avoid the network"))
    assert data.download_sample(tmp_path, 2)[0] == path
    path.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        data.download_sample(tmp_path, 2)


def test_incomplete_download_does_not_leave_a_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: io.BytesIO(b"unfinished"))
    with pytest.raises(ValueError, match="Source ended"):
        data.download_sample(tmp_path, 2)
    assert not list(tmp_path.glob("*.txt"))
    assert not list(tmp_path.glob("*.part"))
