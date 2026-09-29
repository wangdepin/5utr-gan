# 5utr-gan

A generative adversarial network that writes synthetic human 5' untranslated
region (5' UTR) sequences, base by base.

The 5' UTR is the stretch of mRNA between the transcription start site and the
start codon. It is short, unusually GC-rich, and it largely decides how
efficiently a transcript is translated — which makes it the part of a synthetic
mRNA you tune when you want more protein per transcript. This repository asks a
narrow question: **can a sequence model trained only on real 5' UTRs produce new
sequences whose statistics are indistinguishable from real ones?**

The answer turns out to be yes — but from the maximum-likelihood stage, not the
adversarial one. Fixing the pretraining objective closes the gap almost
entirely, and the GAN training that follows reopens it. Both results are
measured below.

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

**Discriminator — one spectral-normalised CNN read through 32 learned views.**
The one-hot input is projected by 32 separate embedding matrices, and each
projection is scored by the *same* convolutional critic; the 32 losses are
averaged. Note what this is and is not: the embeddings hold 1% of the
parameters and the scoring stack 99%, so this is not an ensemble of 32
independent critics but a single critic forced to separate real from generated
under 32 different linear views of the input. Spectral normalisation bounds the
critic's Lipschitz constant. 1.08 M parameters at `--max_len 50` (the flattened
head scales with sequence length).

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

Human 5' UTR sequences from **Ensembl release 105, GRCh38**. The two classes are
defined by translational output: the positive set holds UTRs of highly
translated / highly expressed transcripts, the negative set holds the rest.

> **Incomplete:** the expression or translation-efficiency measure used to split
> the classes, and its cutoff, are not recorded here. They are needed to
> reproduce the split from scratch.

`data/train.pos` — 12 000 5' UTRs of high-translation transcripts.
`data/train.neg` — 12 000 5' UTRs of the remaining transcripts.

The split is visible in the sequences themselves:

| | positives | negatives |
|---|---|---|
| sequences in source pool | 14 102 | 54 764 |
| GC content (median) | 0.667 | 0.594 |
| length (median) | 158 nt | 184 nt |
| uATG per 100 nt | 0.633 | 0.942 |
| sequences with ≥1 uATG | 44.7 % | 62.8 % |

The uATG gap holds within every length bin (0.193 against 0.409 at 50–99 nt,
0.787 against 1.393 at 150–199 nt), so it is not an artefact of the negatives
being longer. Upstream start codons initiate out-of-frame translation and
depress translational output, so a depletion of them in the positive set is what
the class definition predicts.

Both sets are real Ensembl sequences — the negatives are *not* shuffled
positives; they overlap by only 472 sequences. `train.pos` / `train.neg` are the
first 12 000 rows of a fixed shuffle of each pool.

Sequences are one-hot encoded (`encoding.py`), truncated or left-padded to a
fixed length, and **reversed**, so that generation proceeds from the start codon
outwards. Padding is the all-zero vector and is excluded from the training loss.

Only the two training splits are tracked in git; the larger source pools are
listed in `.gitignore`.

---

## Usage

```bash
pixi install          # or: pip install -r requirements.txt

# Train (CUDA, Apple MPS and CPU are selected automatically)
python train.py --max_len 50 --pre_epochs 100 --total_batch 500

# Score generated sequences against the real set
python evaluate.py --real data/train.pos --generated results/<run>/gen_5utr_batch500

# Ablate the pretraining objective against the original one
python train.py --objective bce_gumbel ...
```

On a Slurm cluster, `run_turso.sbatch` submits one configuration per job:

```bash
sbatch run_turso.sbatch nll         fixed
sbatch run_turso.sbatch bce_gumbel  original
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
| `--objective` | `nll` | `nll` (likelihood) or `bce_gumbel` (the original objective, for ablation) |

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
- **uATG count** — upstream start codons trigger out-of-frame translation and
  depress translational output, and the positive set is depleted of them
  (0.63 per 100 nt against 0.94 in the negatives). Read this one carefully:
  because the classes were split on translational output, uATG density is
  correlated with the *selection criterion*, so matching it is not independent
  biological validation. It does show the generator captured a sequence property
  that defines the positive class, which no explicit term in the loss asked for.
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

There is a second trap. The original pretraining objective reports a *lower*
number than the corrected one — around 0.73 against 1.27 — which looks like an
improvement and is not one. The 0.73 is binary cross-entropy against a random
Gumbel draw; it is not a log-likelihood and the two numbers are not on the same
scale. A training curve alone would have endorsed the broken objective, which is
precisely what `evaluate.py` exists to prevent.

### Result: the pretraining objective was the whole story

Two runs, identical in every respect except the generator's pretraining loss —
same seed, same architecture, same 100 epochs, same data. `--objective
bce_gumbel` reproduces the original loss; `--objective nll` is the correction.
Both are scored after pretraining, before any adversarial step.

| | real | original (`bce_gumbel`) | corrected (`nll`) | background control |
|---|---|---|---|---|
| freq(A) | 0.181 | 0.233 | **0.180** | 0.228 |
| freq(C) | 0.320 | 0.273 | **0.323** | 0.283 |
| GC mean ± sd | 0.642 ± 0.148 | 0.542 ± 0.072 | **0.637 ± 0.145** | 0.566 ± 0.143 |
| uATG per sequence | 0.208 | 0.644 | **0.193** | 0.374 |
| distinct fraction | 0.725 | 1.000 | 1.000 | 0.833 |
| **6-mer JS divergence** | — | **0.1006** | **0.0065** | 0.0350 |

JS divergence is biased upward by small samples, so the divergences were
recomputed on equal-sized subsamples (n = 4992) across three random seeds. The
generated sets are all exactly 4992 sequences, so only the control varies:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| original (`bce_gumbel`) | 0.1006 | 0.1006 | 0.1006 |
| corrected (`nll`) | 0.0065 | 0.0065 | 0.0065 |
| background control | 0.0370 | 0.0365 | 0.0376 |

The correction moves 6-mer divergence by a factor of 15, and the generated
sequences end up **closer to the positive set than a different collection of
real sequences is** (0.007 against 0.035). GC content recovers both its mean and
its spread. The uATG rate lands at 0.193 against a real 0.208, where the broken
objective overshot threefold — the generator is trained only on the positive set
and never sees a uATG term in its loss, though as noted above this statistic
tracks the criterion the classes were split on.

Read the failure this way: binary cross-entropy against a random Gumbel draw can
only align the marginal base frequencies, so the original generator learned
composition and nothing else. Every metric above is consistent with that, and so
is the near-uniform output it produced.

### Result: the adversarial phase makes a well-fit generator worse

With the objective corrected, pretraining alone lands within 0.0065 bits of the
real 6-mer distribution. The question is then whether the GAN stage — the part
this architecture exists for — adds anything. It does not.

Both objectives were run to 500 adversarial steps on a V100, 41 minutes each,
sampled every 50 steps. Reference points: real GC is 0.642 ± 0.148 with 0.208 uATG per sequence, and the
background negative set scores 0.0350.

| | corrected (`nll`) | | | original (`bce_gumbel`) | | |
|---|---|---|---|---|---|---|
| stage | 6-mer JS | GC ± sd | uATG | 6-mer JS | GC ± sd | uATG |
| **pretrain** | **0.0065** | **0.637 ± 0.145** | 0.193 | 0.1006 | 0.542 ± 0.072 | 0.644 |
| adv 50 | 0.3182 | 0.616 ± 0.043 | 0.141 | 0.1003 | 0.541 ± 0.072 | 0.644 |
| adv 150 | 0.0520 | 0.686 ± 0.067 | 0.159 | 0.0970 | 0.546 ± 0.073 | 0.641 |
| adv 250 | 0.0268 | 0.656 ± 0.080 | 0.222 | 0.0631 | 0.610 ± 0.077 | 0.518 |
| adv 300 | 0.0264 | 0.658 ± 0.092 | 0.184 | 0.0689 | 0.703 ± 0.076 | 0.310 |
| adv 400 | 0.0547 | 0.555 ± 0.104 | 0.317 | 0.0952 | 0.760 ± 0.077 | 0.175 |
| adv 500 | 0.0705 | 0.530 ± 0.111 | 0.370 | 0.0434 | 0.665 ± 0.078 | 0.196 |

**Adversarial training moves the two starting points in opposite directions.**
From the broken pretrained model it helps, roughly halving divergence over 500
steps (0.1006 → 0.0434). From the corrected one it does nothing but damage: the
first 50 steps blow the fit up 50-fold, and by step 500 the model is eleven
times worse than where it started (0.0065 → 0.0705). Its uATG rate drifts from
0.193 to 0.370 — almost exactly the negative set's 0.374, which is the wrong
direction biologically.

That opposition is probably why the adversarial stage looked justified
originally. Judged from a generator that had only learned base composition, the
discriminator does supply useful signal. Judged from one that already fits the
6-mer distribution, it only has worse places to go.

The GC standard deviation separates the two runs most sharply. Real UTRs vary
widely (± 0.148); the corrected pretrained generator reproduces that (± 0.145).
Every checkpoint of the original run sits at ± 0.072–0.078 and never recovers,
even where its 6-mer divergence improves — it is matching the distribution's
centre while missing its width. The corrected run starts at the right width and
narrows under adversarial pressure. Either way the discriminator buys agreement
on aggregate statistics by spending diversity.

The single best model in this table is the corrected generator **before any
adversarial step**, by a factor of six over the best adversarial checkpoint
anywhere in the grid.

This is consistent with what has been reported for discrete-sequence GANs more
broadly — that adversarially trained text generators tend not to beat a
well-tuned maximum-likelihood baseline once quality and diversity are measured
together (Caccia et al., *Language GANs Falling Short*, ICLR 2020). The
mechanism here is visible in the code: the generator's only route to a gradient
is a straight-through Gumbel-Softmax estimator, which is biased and noisy, while
MLE supplies the exact gradient of the exact objective.

**Conclusion.** For this task the TCN generator is worth keeping and the
adversarial machinery is not. The honest description of this repository is an
autoregressive 5' UTR model, plus a GAN stage that was measured and found
harmful once the generator underneath it was trained correctly.

### What I would do next

- **Drop the discriminator, keep the generator.** Everything the GAN stage was
  meant to deliver is already there after pretraining.
- **Condition on the property instead of filtering by it.** Both classes are
  available; training one conditional model on high- and low-translation UTRs
  would let it generate *toward* a target rather than imitating one class.
- **Validate on something not implied by the split.** Every metric here is
  either a sequence statistic or, in uATG's case, correlated with the selection
  criterion. A held-out predictor of translation efficiency, trained
  independently, would test whether generated UTRs are functionally plausible
  rather than statistically plausible.
- **Report held-out likelihood.** The losses quoted here are training losses,
  which is why the evaluation rests on generated-sequence statistics instead.

---

## Repository layout

```
train.py           three-stage training loop, hyperparameters, CLI
generator.py       causal dilated TCN + Gumbel-Softmax sampling
discriminator.py   spectral-normalised CNN critic over 32 learned input views
dataset.py         teacher-forcing pairs, padding mask
encoding.py        one-hot encoding, padding, reversal
evaluate.py        k-mer / GC / uATG comparison against a real reference
sequence_entropy.py  Markov-order entropy baseline for the reference set
run_docker.sh      training in the official PyTorch image
run_turso.sbatch   Slurm job script (one objective per job)
pixi.toml          pinned environment (torch 2.5.1 + CUDA 12.1)
```
