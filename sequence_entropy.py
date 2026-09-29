"""Estimate how much of a sequence set's entropy is explainable by local context.

Fits Markov models of increasing order and reports cross-entropy per base, held
out on unseen sequences. This sets the scale for a generator's likelihood: the
gap between the uniform bound (log 4) and the best held-out Markov estimate is
all the structure a next-base model can possibly capture.

Usage:
    python sequence_entropy.py --sequences data/train.pos --seq_len 50
"""

import argparse
import math
import random
from collections import Counter

BASES = "ACGT"


def load(path, seq_len):
    return [line.strip().upper()[-seq_len:] for line in open(path)
            if len(line.strip()) >= seq_len]


def fit(seqs, order):
    """Count context -> next-base transitions."""
    contexts, transitions = Counter(), Counter()
    for seq in seqs:
        for i in range(order, len(seq)):
            ctx = seq[i - order : i]
            contexts[ctx] += 1
            transitions[(ctx, seq[i])] += 1
    return contexts, transitions


def cross_entropy(seqs, order, contexts, transitions, alpha=1.0):
    """Mean negative log-likelihood per base, with add-alpha smoothing.

    Smoothing keeps the estimate finite on contexts absent from the fit, which
    is what makes the held-out number meaningful at high order.
    """
    total, n = 0.0, 0
    for seq in seqs:
        for i in range(order, len(seq)):
            ctx = seq[i - order : i]
            p = (transitions[(ctx, seq[i])] + alpha) / (contexts[ctx] + alpha * len(BASES))
            total -= math.log(p)
            n += 1
    return total / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--max_order", type=int, default=6)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    opt = parser.parse_args()

    seqs = load(opt.sequences, opt.seq_len)
    random.Random(opt.seed).shuffle(seqs)
    cut = int((1 - opt.holdout) * len(seqs))
    train, test = seqs[:cut], seqs[cut:]

    print(f"train={len(train)}  held-out={len(test)}  length={opt.seq_len}")
    print(f"uniform bound = {math.log(len(BASES)):.4f} nats/base\n")
    print(" order   in-sample   held-out")

    best = None
    for order in range(opt.max_order + 1):
        contexts, transitions = fit(train, order)
        in_sample = cross_entropy(train, order, contexts, transitions)
        held_out = cross_entropy(test, order, contexts, transitions)
        print(f"   {order}      {in_sample:.4f}      {held_out:.4f}")
        best = held_out if best is None else min(best, held_out)

    explainable = math.log(len(BASES)) - best
    print(f"\nbest held-out estimate      {best:.4f} nats/base")
    print(f"explainable by local context {explainable:.4f} nats/base "
          f"({100 * explainable / math.log(len(BASES)):.1f}% of the uniform bound)")


if __name__ == "__main__":
    main()
