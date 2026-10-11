# Blaude Week 1 Update

Date: 12 October 2026

## Work completed

I prepared the data that Blaude will use for learning. I downloaded a small
TinyStories sample and wrote code to clean the text, check for repeated stories,
and separate the stories into training, validation, and test sets. I then trained
a new tokenizer and built a loader that prepares text for next-token prediction.

The language model itself has not been built or trained yet. This week's result
is the working data pipeline that the model will use.

## Results

| Item | Result |
|---|---:|
| Stories downloaded | 20,000 |
| Stories retained | 20,000 |
| Stories rejected or deduplicated | 0 |
| Training stories | 18,000 |
| Validation stories | 1,000 |
| Test stories | 1,000 |
| Tokenizer vocabulary, including special tokens | 4,096 |
| Training tokens, including story-ending tokens | 3,938,251 |
| Validation tokens, including story-ending tokens | 223,861 |
| Test tokens, including story-ending tokens | 218,490 |
| Input length per training chunk | Up to 256 tokens |
| Example batch size | 8 chunks |

The cleaning code normalizes spacing and Unicode, checks basic text quality,
and removes case/whitespace-equivalent duplicates before making the splits.
No stories in this sample triggered the removal rules. This means the checks
found no matching problems; it does not establish that every story is suitable
in every respect.

The tokenizer was trained only on the 18,000 training stories. It learns common
word pieces and converts them into numbers. The vocabulary was trained from
scratch using the Tokenizers library; no pretrained vocabulary was downloaded.

## Demonstration

Run this from the project folder:

```powershell
.\.venv\Scripts\python.exe -m blaude.demo
```

The demo starts with this text, including extra spaces and a line break:

```text
  A little fox
  found a red ball.
```

The cleaned version is:

```text
A little fox found a red ball.
```

The demo displays the actual tokens and IDs from the saved tokenizer, decodes
them back to text, and shows a batch with shape `[8, 256]`. The numeric results
are also saved in `reports/week_1_demo.json`. Additional examples from the
training data are available in `data/week1/examples.json`.

## Checks completed

- All nine automated tests passed. They cover duplicate removal, split isolation,
  cached-download verification, incomplete downloads, tokenizer round trips,
  next-token targets, padding, and invalid story offsets.
- The artifact audit checked all 12 generated artifact files against their
  checksums and confirmed there are no repeated story IDs between splits.
- All 20,000 encoded stories decode back to their cleaned text. The audit also
  checked token IDs, end-of-story markers, counts, and padding masks.
- A second preparation run was used to check whether the same source and
  settings reproduce the saved files. See `reports/week_1_verification.json`.

## Limitations and next steps

This is a manageable starting sample taken from the beginning of the source
file. It is not a random sample of the full dataset. The filters catch simple
quality problems and exact normalized duplicates; they do not detect every
paraphrase or judge the meaning of a story. A later data review may lead to
better filters or a larger dataset.

Next week, I will build the Transformer model, check its attention mask, and test
whether it can learn from a very small batch before starting longer training.

## Short update for class

“This week, I prepared 20,000 stories for Blaude and divided them into training,
validation, and test sets. I trained a new tokenizer with 4,096 tokens and built
the code that creates training batches. The data checks and all nine tests passed.
I can now demonstrate how a sentence becomes token IDs and how those IDs are
prepared for the model. Next, I will start building the Transformer.”

Source: [TinyStories by Ronen Eldan and Yuanzhi Li](https://huggingface.co/datasets/roneneldan/TinyStories).
The dataset card lists [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
