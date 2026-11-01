# Week 3: training, recovery, a baseline, and charts

After the Week 1 data preparation and Week 2 installation, install the chart
dependency in the VS Code terminal. These examples assume the environment is active:

```powershell
python -m pip install -r requirements-week3.txt
python -m blaude.train --steps 300 --batch-size 8 --eval-every 100 --validation-batches 0 --output artifacts/week3-pilot
python -m blaude.bigram --output artifacts/week3-bigram
```

The pilot uses the smaller 628,480-parameter model and evaluates the entire
validation split. This is an early experiment, not finished language training.
The default `python -m blaude.train` remains a quick 50-step demonstration.

## What is saved

- `last.pt`: the last completed evaluation checkpoint, including model weights,
  optimizer state, random state, settings, completed step, and target count.
- `best.pt`: the checkpoint with the lowest validation loss in this output folder.
- `history.csv`: measured losses and the number of real targets behind them.
- `loss.png`: training and validation curves, with sample/full validation labelled.
- `config.json` and `summary.json`: settings, data/code hashes, and completion counts.

Checkpoints are written at step 0, every evaluation interval, and the final step.
The next checkpoint is written to a temporary file before replacing the previous
one. If interrupted, only progress after the last successful checkpoint is lost.
Completed evaluation losses and finite-gradient checks help spot instability.

## Continue a stopped or finished run

```powershell
python -m blaude.train --resume artifacts/week3-pilot/last.pt --steps 400 --batch-size 8 --eval-every 100 --validation-batches 0 --output artifacts/week3-continued
```

`--steps 400` means reach step 400 in total, not perform 400 additional steps.
Always use a new output folder. Repeat the original model, batch, seed, learning
rate, evaluation, and thread settings. The trainer rejects mismatched data, code,
PyTorch version, or training settings instead of silently changing the experiment.
Exact continuation is tested with dropout and at epoch boundaries in the same
CPU environment. Cross-platform bitwise agreement is not promised.

The resumed folder logs only the continuation, with an initial validation row
at the saved step. Its best.pt is the best in that continuation; retain the old
folder's best.pt too. The Week 2 tiny_model.pt lacks the required optimizer state
and cannot be used as a training-resume checkpoint.

## Understand the comparison model

`blaude.bigram` counts how often each token follows each previous token in the
training stories. It never joins two stories and never fits validation/test data.
Add-alpha smoothing (alpha 0.1, fixed before evaluation) gives unseen pairs a
nonzero probability. It saves counts in bigram.npz and validation loss in metrics.json.

Compare its full-validation loss with the Transformer's full-validation loss.
Both use the same tokenizer and all 222,861 next-token targets in validation.
The baseline sees all training pairs; a short Transformer pilot has a smaller
training budget. This is a reference comparison, not equal-compute benchmarking.
For a sample comparison, match context length, batch size, and validation-batch cap
exactly; different chunk sizes otherwise select different sample targets.

## Read or recreate a chart

```powershell
python -m blaude.plot artifacts/week3-pilot/history.csv --output artifacts/week3-pilot/loss-copy.png
python -m pytest -q --basetemp artifacts/pytest-week3
```

Training loss is an average of recent batches with dropout enabled. Validation
uses current weights with dropout disabled. Padding is ignored and tokens are
weighted equally. Tests check exact continuation, optimizer/random-state recovery,
safe checkpoint replacement, baseline story boundaries, and evaluation isolation.

Downloaded data, checkpoints, charts, and experiment logs stay local in ignored
folders. The repository contains the code needed to recreate them on her laptop.
