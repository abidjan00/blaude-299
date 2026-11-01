# Week 4: train the full model and inspect early text

Week 4 uses the complete 5,322,240-parameter architecture: four blocks, width 256,
four heads, and context 256. It starts from random weights, not the smaller Week 3
pilot. The tokenizer and prepared stories are the same as Week 1.

## Train one complete pass

In the activated VS Code Python environment, after Weeks 1-3 setup:

```powershell
python -m blaude.train --preset full --steps 1367 --batch-size 16 --eval-every 100 --validation-batches 20 --threads 4 --output artifacts/week4-full
```

For the original preparation there are 21,857 training chunks at context 256.
Batch size 16 gives 1,367 updates for one epoch, including a smaller last batch.
That visits all 18,000 training stories and 3,920,251 next-token training targets.
Different data or context lengths change the number of updates needed.

Every 100 steps the program evaluates the same first 20 validation batches,
checks for non-finite loss/gradients, saves last.pt, and saves best.pt when the
validation-sample loss improves. It also saves at the final step. Gradient
clipping at 1.0 helps limit extreme updates. Best means lowest validation-sample
loss in this run, not best final test performance.

Keep the laptop plugged in and avoid sleep while training. CPU time depends on
the device. The default small 50-step run is still available for a quick software
demonstration, but does not replace this full-model training pass.

## Continue after stopping

```powershell
python -m blaude.train --resume artifacts/week4-full/last.pt --preset full --steps 1367 --batch-size 16 --eval-every 100 --validation-batches 20 --threads 4 --output artifacts/week4-recovered
```

This works if the saved step is below 1,367. The trainer starts after the last
saved update and recreates the shuffled data position and optimizer state.
Keep the original settings and code. Each continuation needs a new output folder.
Progress since the last saved checkpoint is lost. If it already finished, choose
a higher total step count to train longer; that would be a new experiment.

## Generate an early text sample

```powershell
python -m blaude.generate --checkpoint artifacts/week4-full/best.pt --prompt "Once upon a time, there was a little" --output artifacts/week4-full/sample.json
```

This loads weights, verifies the tokenizer fingerprint, and predicts one token
at a time using its own previous predictions. Unlike the Week 2 memorization
check, future answers are not supplied. It keeps only the latest context window
and stops at the story-ending token or the length limit.

The early-sample protocol uses seed 299, temperature 0.8, top 40 candidates, and
up to 80 new tokens. Padding and unknown tokens cannot be sampled. A seed makes
the random choices repeatable within the same environment. It is not evidence
that the generated story is correct. Prompt and length can be changed with
`--prompt` and `--max-new-tokens`; broader setting comparisons remain later work.

Use the same prompts and seeds when comparing checkpoints. Include awkward
samples, not only successful ones. Check whether sentences are readable, stay
on topic, and repeat phrases. These informal samples are not the final 20-prompt
evaluation planned for Week 7.

## Evaluate all validation stories

```powershell
python -m blaude.evaluate --checkpoint artifacts/week4-full/best.pt --output artifacts/week4-full/full-validation.json
python -m blaude.bigram --output artifacts/week4-bigram
```

The evaluator checks all 222,861 validation targets without changing weights.
Compare this full-validation loss with the bigram model's full-validation loss.
The bigram uses only the previous token; the Transformer can use earlier context.
The training chart uses a fixed sample during progress checks and must not be
mislabelled as full validation. The reserved test split is untouched.

## Show the work on her laptop

The repository includes source code, not the data or checkpoints. She can run
the training command herself after data preparation. Alternatively, copy best.pt
and the matching tokenizer.json into a local folder; pass that folder as `--data`
to generate. Generation only needs the tokenizer and checkpoint. Full validation
and resuming also need the original prepared training/validation files.

For a demonstration, open loss.png, run a sample, and open full-validation.json.
Explain that one epoch is an early training run, not a finished general chatbot.
The logs distinguish partial validation checks from full evaluation, and the
checkpoint identifies exactly which weights produced the saved sample.

```powershell
python -m pytest -q --basetemp artifacts/pytest-week4
```

The tests include deterministic sampling, context truncation, end-token stopping,
reserved-token blocking, tokenizer mismatch rejection, and full-validation counts.
