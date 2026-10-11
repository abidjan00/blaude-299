"""Download a bounded, pinned sample and clean it before making splits."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import Request, urlopen

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
BASE_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}"
SOURCE_URL = f"{BASE_URL}/TinyStories-train.txt"
LICENSE_URL = "https://cdla.dev/sharing-1-0/"
DELIMITER = "<|endoftext|>"
SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", DELIMITER]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_sample(raw_dir, count=20_000):
    """Read the first count complete stories, then close the remote stream.

    This is a reproducible prefix sample, not a random sample of the full corpus.
    The raw cache is checksummed; partial or modified caches are never reused.
    """
    if count < 1:
        raise ValueError("Story count must be positive")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"tinystories-{REVISION[:12]}-{count}.txt"
    manifest_path = target.with_suffix(".json")
    if target.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sha256"] != file_hash(target):
            raise ValueError(f"Raw cache checksum mismatch: {target}")
        if manifest["source_url"] != SOURCE_URL or manifest["stories"] != count:
            raise ValueError("Raw cache metadata does not match the requested source")
        return target, manifest

    partial = target.with_suffix(".part")
    buffer = b""
    read_bytes = 0
    found = 0
    marker = DELIMITER.encode("ascii")
    # Bound accidental downloads as well as an unterminated malformed story.
    max_bytes = max(4_000_000, count * 10_000)
    request = Request(SOURCE_URL, headers={"User-Agent": "Blaude-course299/0.1"})
    try:
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while found < count:
                chunk = response.read(64 * 1024)
                if not chunk:
                    raise ValueError(f"Source ended after {found} of {count} complete stories")
                read_bytes += len(chunk)
                if read_bytes > max_bytes:
                    raise ValueError("Download exceeded the sample byte budget")
                buffer += chunk
                while marker in buffer and found < count:
                    story, buffer = buffer.split(marker, 1)
                    story.decode("utf-8", errors="strict")
                    output.write(story.strip() + b"\n" + marker + b"\n")
                    found += 1
                if len(buffer) > 1_000_000:
                    raise ValueError("Source contains an unexpectedly long unterminated story")
                if found and found % 5000 == 0:
                    print(f"Downloaded {found:,} stories", flush=True)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)

    with urlopen(f"{BASE_URL}/README.md", timeout=60) as response:
        (raw_dir / "TinyStories_README.md").write_bytes(response.read())
    manifest = {
        "dataset": "roneneldan/TinyStories", "revision": REVISION,
        "source_url": SOURCE_URL, "license": "CDLA-Sharing-1.0",
        "license_url": LICENSE_URL, "sampling": "first N complete stories in source order",
        "stories": found, "network_bytes_read": read_bytes,
        "saved_bytes": target.stat().st_size, "sha256": file_hash(target),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    return target, manifest


def clean_text(text):
    text = unicodedata.normalize("NFKC", text).replace("\ufeff", "")
    return " ".join(text.split())


def rejection_reason(text, min_words=20, max_words=1000):
    if not text:
        return "empty"
    if any(token in text for token in SPECIAL_TOKENS):
        return "reserved_token"
    if any(unicodedata.category(char).startswith("C") for char in text):
        return "control_character"
    if not min_words <= len(text.split()) <= max_words:
        return "word_count"
    letters = [char for char in text if char.isalpha()]
    if not letters or sum(char.isascii() for char in letters) / len(letters) < 0.95:
        return "non_english_script"
    if re.search(r"https?://|www\.|\b[^\s@]+@[^\s@]+\.[^\s@]+", text, re.I):
        return "link_or_email"
    if "\ufffd" in text:
        return "replacement_character"
    return None


def prepare_records(stories, min_words=20, max_words=1000):
    """Remove basic quality issues and case/whitespace-equivalent duplicates.

    Script and length checks are heuristics, not a semantic safety classifier.
    IDs group text variants before splitting, avoiding exact duplicate leakage.
    """
    if min_words < 1 or max_words < min_words:
        raise ValueError("Invalid word-count limits")
    records, seen, reasons, examples = [], set(), Counter(), []
    total = 0
    for source_index, raw in enumerate(stories):
        total += 1
        text = clean_text(raw)
        reason = rejection_reason(text, min_words, max_words)
        story_id = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
        if reason:
            reasons[reason] += 1
        elif story_id in seen:
            reasons["duplicate"] += 1
        else:
            seen.add(story_id)
            records.append({"id": story_id, "source_index": source_index, "text": text})
            if len(examples) < 3:
                examples.append({"raw": raw, "cleaned": text})
    return records, {"input_stories": total, "kept": len(records), "removed": dict(reasons)}, examples


def split_records(records, seed=299):
    """Assign deterministic 90/5/5 splits without depending on source order."""
    if len(records) < 20:
        raise ValueError("At least 20 unique cleaned stories are needed for three splits")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Remove duplicate story IDs before splitting")
    ordered = sorted(records, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())
    held_out = len(ordered) // 20
    return {
        "train": ordered[: -2 * held_out],
        "validation": ordered[-2 * held_out: -held_out],
        "test": ordered[-held_out:],
    }


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
