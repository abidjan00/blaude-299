# How Blaude's Week 2 model works

Week 1 turned stories into token IDs. Week 2 builds the network that reads those
IDs and predicts the next token. The implementation is in `blaude/model.py`.

## The path through the model

```text
Token IDs [batch, length]
  -> token embeddings + learned position embeddings
  -> embedding dropout
  -> four Transformer blocks
       normalize -> causal attention -> residual addition
       normalize -> feed-forward network -> residual addition
  -> final normalization -> output layer
Logits [batch, length, vocabulary]
```

An embedding is a learned vector representing a token. Position embeddings tell
the model where each token appears. All weights start randomly; no pretrained
model is loaded. PyTorch supplies the basic layers, automatic differentiation,
and optimizer, while this project implements the attention and model structure.
It does not use the `nn.Transformer` wrapper.

Each block has attention and a feed-forward network. Attention calculates query,
key, and value vectors, splits them into four heads, scores allowed connections,
and combines values. The feed-forward network uses linear layers and GELU at each
position. Layer normalization controls the scale of values. Residual additions
preserve a direct path through the block and help gradients reach earlier layers.

The default model has width 256, four blocks, four heads, feed-forward width 1,024,
and maximum context 256. The vocabulary comes from the saved tokenizer. The output
projection and token embedding have separate weights.

## The two masks

For a three-token input, the causal mask allows these connections:

```text
                     Key position
                     0  1  2
Query position 0     Y  N  N
Query position 1     Y  Y  N
Query position 2     Y  Y  Y
```

A position can see itself because its target is the next token. Later tokens are
hidden. Disallowed scores become negative infinity before softmax, giving those
connections zero probability. A separate padding mask hides unused key positions.

The model accepts nonempty, right-padded sequences, matching the Week 1 loader.
Left padding, gaps, and entirely empty rows raise an error. This avoids attention
rows with no allowed keys, which could produce NaNs. Predictions at padded query
positions are zeroed and their losses are ignored.

## Targets are already shifted

```text
Input IDs:   [20, 21,    0]
Target IDs:  [21,  2, -100]
Valid mask:  [ T,  T,    F]
```

At input 20, the correct next token is 21. At input 21, it is the ending token 2.
The last position is padding. `TokenDataset` already makes these pairs:
**do not shift targets again inside the model or training loop**.

The output gives vocabulary scores, called logits, at each position. Cross-entropy
compares the scores to the correct ID and ignores targets marked `-100`. When no
valid targets exist, the model raises an error instead of returning an undefined loss.

## Connect the model to Week 1 data

```python
from blaude.loader import TokenDataset
from blaude.model import BlaudeModel, ModelConfig
from blaude.torch_data import to_torch_batch

# The default vocabulary matches the prepared 4,096-token tokenizer.
model = BlaudeModel(ModelConfig())
numpy_batch = next(TokenDataset("data/week1").batches(batch_size=2))
batch = to_torch_batch(numpy_batch)
result = model(**batch)
print(result["logits"].shape)  # [2, 256, 4096]
print(result["loss"].item())   # prediction error before training
result["loss"].backward()      # calculate gradients
```

The adapter keeps IDs and targets as int64 tensors and masks as booleans.
`model.eval()` disables dropout for comparisons, but does not disable gradients.
Use `torch.no_grad()` when gradient calculation is unnecessary.

## The small learning experiment

`python -m blaude.week2` checks the full architecture, then trains a smaller
version on two fixed training chunks. The small version has two blocks, width 64,
four heads, context 64, and no dropout. It uses AdamW at learning rate 0.003, no
weight decay, and gradient clipping at 1.0 for 200 steps.

Passing means the final loss is below 10% of the initial loss and the fixed-batch
next-token accuracy reaches 95%. This is intentional overfitting: it verifies
that the model and data alignment allow learning at all.

Accuracy is measured on the same batch used for fitting, with correct earlier
tokens provided (teacher forcing). It is not test-set accuracy or proof of coherent
free-text generation. Validation and test data are not used here. The full-sized
model receives shape, masking, and gradient checks, not a full training run.

## Saved evidence

- `metrics.json`: configurations, checks, loss history, runtime, and source hashes.
- `loss_history.csv`: loss and fixed-batch accuracy every 25 steps.
- `tiny_model.pt`: a diagnostic snapshot of the small model and its configuration.

The snapshot has no optimizer state and cannot resume training. The experiment
reloads it with `weights_only=True` and verifies identical predictions. It stays
in the Git-ignored `artifacts/` folder. Week 3 adds a reusable training loop,
validation, and recovery from interrupted training.

## Automated tests

The model tests check future-token isolation, zero attention to forbidden keys,
padding invariance, finite gradients, target alignment, the Week 1 loader, bad
inputs, and tiny-batch learning. They are explicitly skipped if PyTorch is absent;
install `requirements-week2.txt` before claiming the model tests have run. A
broken PyTorch installation is an error, not a successful skip.

References:

- [Cross-entropy and ignored targets](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.cross_entropy.html)
- [Layer normalization](https://docs.pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html)
- [CPU installation](https://pytorch.org/get-started/previous-versions/)
