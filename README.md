# Blaude

Blaude is a Course 299 project to build a small English language model from scratch.
Weeks 1 and 2 are implemented: prepare stories, train a tokenizer, load batches,
and build a decoder-only Transformer. A tiny-batch experiment checks that it can
learn. Full dataset training and a chat interface are later work.

## Run Week 2 in VS Code

Open **Terminal > New Terminal** in this project folder. Install CPU-only PyTorch
once (internet required), then run the model checks and tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-week2.txt
.\.venv\Scripts\python.exe -m blaude.week2
.\.venv\Scripts\python.exe -m pytest -q
```

If the environment is already active (`(.venv)` appears at the prompt), you can
use `python -m blaude.week2` and `python -m pytest -q` instead. On a new computer,
complete the Week 1 setup and data preparation below first. Week 2 uses only the
training split and runs on a CPU without an API key or pretrained weights.

The terminal shows the full model's parameter count and output shape, a causal
attention check, and the small model's decreasing loss. Results are saved under
`artifacts/week2/`. To repeat without overwriting an earlier run:

```powershell
.\.venv\Scripts\python.exe -m blaude.week2 --output artifacts/week2-repeat
```

The full model has four Transformer blocks, four attention heads, 256-wide
embeddings, and context length 256. The diagnostic uses a smaller version with
two blocks, width 64, context 64, and no dropout. It must reduce its initial loss
by more than 90% and reach at least 95% next-token accuracy on the same fixed batch.
This intentionally tests memorization, not general language ability. There is no
chat interface or free-text generation command yet.

See [the model guide](docs/week_2_model.md) for explanations of the code and masks.

## Run the completed Week 1 work

Open PowerShell in this folder. The local Python environment and prepared data
already exist on this computer. These commands do not download or retrain anything:

```powershell
.\.venv\Scripts\python.exe -m blaude.demo
.\.venv\Scripts\python.exe -m blaude.check data/week1
.\.venv\Scripts\python.exe -m pytest -q
```

The demo shows a short sentence, its cleaned version, its token IDs, and a
training batch. It does not generate new text yet. Read
[the weekly update](reports/week_1_update.md) for the results and a short class update.

## Set up on another computer

Use Python 3.11. From the project folder, run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m blaude.prepare --stories 20000
.\.venv\Scripts\python.exe -m blaude.check data/week1
.\.venv\Scripts\python.exe -m blaude.demo
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux, use `python3.11 -m venv .venv` and `.venv/bin/python` instead.
The lock file records the exact packages used for this run. `pyproject.toml`
also supports installing the project with `pip install -e ".[dev]"`.
No API key, GPU, PyTorch installation, or pretrained model is needed for Week 1.

## Prepare a new run

The downloader stops after the requested number of complete stories. The default
20,000-story sample uses about 18 MB of raw text. It comes from the start of a
fixed TinyStories source version, rather than a random sample of the full dataset.

```powershell
.\.venv\Scripts\python.exe -m blaude.prepare --stories 20000 --output data/my-run
.\.venv\Scripts\python.exe -m blaude.check data/my-run
.\.venv\Scripts\python.exe -m blaude.demo --data data/my-run
```

Use a new output folder for each run. Existing output folders are protected from
overwriting. An interrupted run may leave an incomplete output folder; use another
folder to retry. Raw downloads are cached and verified by SHA-256 before reuse.
The preparation command supports `--stories`, `--vocab-size`, `--seed`,
`--raw-dir`, and `--output`; see `python -m blaude.prepare --help`.

## How the pipeline works

1. Download complete stories from a pinned TinyStories revision.
2. Normalize Unicode and whitespace. Reject empty text, unexpected control
   characters, reserved tokens, links/emails, unsuitable lengths, and text with
   less than 95% ASCII letters among its alphabetic characters.
3. Remove duplicates after whitespace normalization and case folding.
4. Sort story IDs by a seeded hash and make disjoint 90%/5%/5% splits.
5. Train a byte-level BPE tokenizer on the **training split only**. Its vocabulary
   and merge rules are newly learned using the Hugging Face Tokenizers library;
   the BPE algorithm itself is supplied by that library.
6. Encode each split using the same tokenizer and append an end-of-story token.
7. Save arrays, examples, source information, counts, settings, and checksums.

The quality filters are simple checks, not a full language detector or a content
safety review. They do not remove paraphrases or near duplicates. The scripts
keep validation and test data out of tokenizer fitting. The original source's
training file is split locally; these are not the official TinyStories splits.

## Training batches

```python
from blaude.loader import TokenDataset

dataset = TokenDataset("data/week1", split="train", context_length=256)
for batch in dataset.batches(batch_size=8, shuffle=True, seed=299):
    inputs = batch["input_ids"]       # int64: [batch, context]
    targets = batch["labels"]         # the next token at each position
    padding_mask = batch["attention_mask"]
    break
```

Each story is divided into chunks of at most 256 input tokens. Targets are shifted
one token forward. Chunks never cross story boundaries, and short final chunks are
padded. Padding targets use `-100`, which the model's loss function ignores.
The boolean attention mask marks padding; Week 2's model adds a causal mask.
The first token of a story is context, with no prediction target before it.
`blaude.torch_data.to_torch_batch` converts these arrays to PyTorch tensors.

## Files

| Location | Purpose |
|---|---|
| `blaude/data.py` | Downloading, cleaning, duplicate checks, and splitting |
| `blaude/tokenizer.py` | Training the tokenizer and saving token arrays |
| `blaude/loader.py` | Input/target batches for the future model |
| `blaude/prepare.py` | Run the complete preparation pipeline |
| `blaude/check.py` | Verify checksums, splits, encodings, and batch padding |
| `blaude/demo.py` | A small demonstration for the weekly update |
| `blaude/model.py` | Embeddings, causal attention, and Transformer blocks |
| `blaude/torch_data.py` | Convert Week 1 batches to PyTorch tensors |
| `blaude/week2.py` | Architecture checks and the tiny-batch learning experiment |
| `tests/` | Offline tests for data handling and training batches |
| `data/raw/` | Download cache and source metadata |
| `data/week1/` | Prepared splits, tokenizer, arrays, examples, and manifest |
| `reports/` | Weekly updates, experiment results, and recorded checks |
| `artifacts/week2/` | Local diagnostic weights, metrics, and loss history |

Data, the local environment, and temporary document files are excluded from Git.
The source code and scripts recreate the data artifacts.

## Data source

TinyStories is provided by Ronen Eldan and Yuanzhi Li and contains synthetic short
English stories. Its dataset card lists **CDLA-Sharing-1.0**. Each prepared folder
contains a notice describing our changes, the provider, source, and license link.
Keep this attribution and notice with any shared dataset files.

- [Dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)
- [TinyStories paper](https://arxiv.org/abs/2305.07759)
- [Dataset license](https://cdla.dev/sharing-1-0/)
- [Tokenizers documentation](https://huggingface.co/docs/tokenizers/quicktour)
