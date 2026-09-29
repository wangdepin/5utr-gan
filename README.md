# 5utr-gan

A generative adversarial network that writes synthetic human 5' untranslated
region (5' UTR) sequences, base by base.

The 5' UTR is the stretch of mRNA between the transcription start site and the
start codon. It is short, unusually GC-rich, and it largely decides how
efficiently a transcript is translated — which makes it the part of a synthetic
mRNA you tune when you want more protein per transcript. This repository asks a
narrow question: **can a sequence model trained only on real 5' UTRs produce new
sequences whose statistics are indistinguishable from real ones?**

The answer this repository arrives at is *not yet*, and the interesting part is
why. The evaluation below measures the gap and localises it.

---

## Method

Generating DNA is discrete-sequence generation: the model emits one of `{A,C,G,T}`
per position, and `argmax` has no gradient, so a discriminator's error signal
cannot reach the generator by the usual route. The two design choices that
follow from that constraint are the substance of this project.

**Generator — causal dilated convolutions (TCN).** Six residual blocks of
weight-normalised 1-D convolutions, each block doubling its dilation
(`1,2,4,…,32`), giving a receptive field that covers the whole sequence while
staying causal: position *i* only ever sees positions `< i`. Because the
convolutions are causal, teacher-forced training needs a *single* forward pass to
score every position at once. 0.96 M parameters.

**Discriminator — 32 parallel spectral-normalised CNNs.** Each branch gets its
own learned embedding of the one-hot input and scores the sequence
independently; the losses are averaged. Spectral normalisation bounds the
Lipschitz constant of each branch, and the ensemble makes the generator fit a
committee rather than exploit one critic. 1.90 M parameters.

**Passing gradients through the sampling step — Gumbel-Softmax.** Sampling uses
`F.gumbel_softmax(..., hard=True)`, a straight-through estimator: the forward
pass emits a genuine one-hot base so the discriminator sees valid DNA, while the
backward pass substitutes the soft relaxation so gradients flow. The temperature
`tau` is annealed (×0.955 on a schedule) so the relaxation tightens toward a true
categorical draw as training proceeds.

**Training runs in three stages:** maximum-likelihood pretraining of the
generator, discriminator pretraining against real-vs-generated and
real-vs-background pairs, then adversarial training.

---

## Data

> **⚠️ Provenance is being documented — do not cite these numbers yet.**
> The exact source database, assembly/annotation version, and the filters that
> define the positive and negative sets are not yet recorded in this repository.
> This section will be completed before the repository is considered finished.

`data/train.pos` — 12 000 real human 5' UTR sequences (positive set).
`data/train.neg` — 12 000 background sequences (negative set).

What is verifiable from the files themselves:

| | positives | negatives |
|---|---|---|
| sequences in source pool | 14 102 | 54 764 |
| GC content (median) | 0.667 | 0.592 |
| length (median) | 158 nt | 184 nt |
| overlap between sets | \multicolumn{2}{c}{472 sequences} |

The two sets differ in composition and length distribution and overlap by only
472 sequences, so the negative set is a distinct collection of real sequences —
**not** a shuffled version of the positives. `train.pos` / `train.neg` are the
first 12 000 rows of a fixed shuffle of each pool.

Sequences are one-hot encoded (`encoding.py`), truncated or left-padded to a
fixed length, and **reversed**, so that generation proceeds from the start codon
outwards. Padding is the all-zero vector and is excluded from the training loss.

Only the two training splits are tracked in git; the larger source pools are
listed in `.gitignore`.

---

## Usage

```bash
pip install -r requirements.txt

# Train (CUDA, Apple MPS and CPU are selected automatically)
python train.py --max_len 50 --pre_epochs 100 --total_batch 150

# Score generated sequences against the real set
python evaluate.py --real data/train.pos --generated results/<run>/gen_5utr_batch150
```

Each run writes to `results/<timestamp>/`: generated sequences, TensorBoard logs,
and a zip snapshot of the exact scripts used, so any result can be traced back to
the code that produced it.

| flag | default | meaning |
|---|---|---|
| `--max_len` | 50 | sequence length the model is trained on |
| `--pre_epochs` | 100 | generator MLE pretraining epochs |
| `--total_batch` | 500 | adversarial steps |
| `--d_steps` | 5 | discriminator updates per generator update |
| `--tau` | 1.0 | initial Gumbel-Softmax temperature |

---

## Evaluation

`evaluate.py` compares a generated set against a real reference on statistics
that a model can only match by learning sequence structure rather than base
frequencies:

- **k-mer Jensen-Shannon divergence** (k = 1…6) — the headline number. Matching
  marginal base frequencies is easy; matching 6-mer frequencies requires
  context.
- **GC content mean and spread** — real 5' UTRs are GC-rich *and* heterogeneous,
  so the standard deviation matters as much as the mean.
- **uATG count** — real 5' UTRs are depleted of upstream start codons, which
  would trigger out-of-frame translation. A weak functional sanity check the
  model is never trained on.
- **distinct fraction** — guards against mode collapse.

The negative background set is scored the same way, as a reference point for
what an *unrelated set of real sequences* scores. A generator that does not beat
that line has not learned anything specific to 5' UTRs.

### How much structure is there to learn?

Before reading any training loss, it is worth knowing the ceiling.
`sequence_entropy.py` fits Markov models of increasing order to the real set and
measures held-out cross-entropy per base:

```
$ python sequence_entropy.py --sequences data/train.pos --seq_len 50

uniform bound = 1.3863 nats/base

 order   in-sample   held-out
   0      1.3452      1.3474     <- base composition alone
   3      1.3074      1.3105
   5      1.2889      1.2984
   6      1.2584      1.2904     <- best held-out

explainable by local context 0.0959 nats/base (6.9% of the uniform bound)
```

**Local 5' UTR sequence is nearly incompressible.** Everything a next-base model
can learn from context lives in a 0.096 nat window between a uniform draw and
the best Markov estimate. A generator scoring 1.26 nats therefore looks
impressive and means very little — it is what a 6th-order Markov chain gets.

This is the reason the evaluation leans on k-mer divergence rather than
likelihood: in this regime likelihood barely separates a good model from a
trivial one, while k-mer statistics still do.

<!-- RESULTS TABLE: filled in from the completed rerun -->

---

## Repository layout

```
train.py           three-stage training loop, hyperparameters, CLI
generator.py       causal dilated TCN + Gumbel-Softmax sampling
discriminator.py   32-branch spectral-normalised CNN ensemble
dataset.py         teacher-forcing pairs, padding mask
encoding.py        one-hot encoding, padding, reversal
evaluate.py        k-mer / GC / uATG comparison against a real reference
sequence_entropy.py  Markov-order entropy baseline for the reference set
run_docker.sh      training in the official PyTorch image
```
