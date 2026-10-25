# Week 3, first half: training and progress tracking

Week 2 checked whether the model could memorize one small batch. This part adds
a reusable training loop that moves through different batches of stories.

## What is finished

- Train on shuffled batches from the training split using AdamW.
- Start a newly shuffled pass if all training batches have been used.
- Clip gradients and stop clearly if the loss or gradients become non-finite.
- Check prediction error on separate validation data without updating weights.
- Show progress in the terminal and save settings and a CSV loss log.
- Protect existing runs by requiring a new output folder.

Saving and resuming weights, the bigram comparison model, the longer pilot
experiment, and training charts remain for the second half of Week 3.
The proposal is unchanged.

## Run it in VS Code

Open the project folder and select **Terminal > New Terminal**. On another
computer, follow the README's Week 1 setup first, including preparing the data.
Install the same PyTorch package used in Week 2 if it is not already installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-week2.txt
.\.venv\Scripts\python.exe -m blaude.train
```

If the environment is active, `python -m blaude.train` is enough.
No extra package is needed beyond Weeks 1 and 2. No chat window opens.

The default uses a 628,480-parameter model with two Transformer blocks, width 64,
context 64, and dropout 0.1. It runs 50 steps, with four chunks per batch, on CPU.
This keeps the demonstration short. The `--preset full` option selects the
5,322,240-parameter Week 2 architecture and takes more computing time.

An example command with explicit settings is:

```powershell
.\.venv\Scripts\python.exe -m blaude.train --steps 50 --batch-size 4 --eval-every 25 --validation-batches 10 --output artifacts/week3-demo
```

Use a fresh `--output` folder each time. An interrupted run may leave partial
logs. It cannot be resumed in this version; a new run starts with random weights.

## What the terminal shows

First it shows the model size and validation loss before training (step 0).
At steps 25 and 50 it shows training loss and validation loss. Loss measures how
wrong the next-token predictions are; lower is better.

Training loss is the average across the batches since the last printed update.
It is measured while training, with dropout enabled and weights changing.
Validation loss uses the current weights with dropout disabled. The two numbers
are useful for progress tracking but are not directly identical measurements.
Both averages weight each real target token equally and ignore padding.

By default validation uses the same first 10 batches each time. This is a quick,
fixed sample, not the full validation result. Use `--validation-batches 0` to
evaluate the entire validation split. The reserved test split is never opened.
A falling loss during this short check does not prove readable text generation.

## What each code section does

In `blaude/train.py`:

1. `load_data()` checks file hashes, loads the tokenizer, and opens the training
   and validation splits created in Week 1.
2. `run_training()` creates randomly initialized weights and an AdamW optimizer.
   It sets a seed so the same settings can be repeated in the same environment.
3. The loop gets the next shuffled training batch, predicts next tokens, computes
   loss, calculates gradients, clips them, and updates the weights. Targets are
   already shifted by the Week 1 loader; they are not shifted again.
4. `evaluate()` temporarily disables dropout and gradient recording. It measures
   validation loss, then restores the previous training mode.
5. `record()` prints progress and flushes a row to the CSV after each validation
   check. The final step is logged even when it is not a multiple of the interval.

## Files saved locally

The default output folder is `artifacts/week3-part1/`, which Git ignores.

| File | Contents |
|---|---|
| `config.json` | Model and training settings, seed, software version, and file hashes |
| `history.csv` | Step, epoch, training loss, validation loss, token counts, and time |
| `summary.json` | Completion status, steps, and initial/final validation loss |

`train_tokens` in the CSV counts targets since the previous log entry, not all
targets since training began. `validation_full_split` identifies whether every
validation batch was evaluated. Perplexity is the exponential of validation
loss; it is another prediction-error measure, not percentage accuracy.

These are logs, not saved model weights. Do not start a long training session
expecting to keep its learned weights until the second half adds checkpoints.

## Check the code

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/pytest-week3-demo
```

The new tests check that validation leaves weights and gradients unchanged,
averages tokens correctly, and restores training mode. They also check batch
and epoch progress, separation of training and validation data, repeatability,
checksum failures, invalid settings, and protection of existing output folders.
