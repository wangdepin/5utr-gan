"""Compare generated 5' UTR sequences against a real reference set.

Reports nucleotide composition, k-mer divergence and uAUG frequency -- the
statistics that say whether a generator learned UTR sequence structure or only
learned the marginal base frequencies.

Usage:
    python evaluate.py --real data/train.pos --generated results/<run>/gen_5utr_batch175
"""

import argparse
from collections import Counter

import numpy as np

BASES = "ACGT"


def load(path, seq_len):
    """Read one sequence per line, truncated to the trailing `seq_len` bases.

    The trailing end is kept because it is the region adjacent to the start
    codon, and it is what `encoding.one_hot_encoding` feeds the model.
    """
    seqs = []
    with open(path) as fin:
        for line in fin:
            seq = line.strip().upper()
            if len(seq) >= seq_len:
                seqs.append(seq[-seq_len:])
    if not seqs:
        raise ValueError(f"no sequence of length >= {seq_len} in {path}")
    return seqs


def base_frequencies(seqs):
    counts = Counter("".join(seqs))
    total = sum(counts[b] for b in BASES)
    return {b: counts[b] / total for b in BASES}


def gc_content(seqs):
    return np.array([sum(c in "GC" for c in s) / len(s) for s in seqs])


def kmer_distribution(seqs, k):
    counts = Counter()
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            counts[seq[i : i + k]] += 1
    total = sum(counts.values())
    return {kmer: c / total for kmer, c in counts.items()}


def jensen_shannon(p, q):
    """Jensen-Shannon divergence in bits between two k-mer distributions."""
    keys = set(p) | set(q)
    pv = np.array([p.get(key, 0.0) for key in keys])
    qv = np.array([q.get(key, 0.0) for key in keys])
    m = 0.5 * (pv + qv)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


def uaug_rate(seqs):
    """Mean number of upstream ATGs per sequence.

    Real 5' UTRs are depleted of upstream start codons because they trigger
    out-of-frame translation, so this is a weak functional sanity check.
    """
    return float(np.mean([s.count("ATG") for s in seqs]))


def distinct_fraction(seqs):
    return len(set(seqs)) / len(seqs)


def report(real, generated, max_k):
    print(f"sequences        real={len(real):>6}   generated={len(generated):>6}")
    print(f"length                {len(real[0]):>6}              {len(generated[0]):>6}")

    real_freq, gen_freq = base_frequencies(real), base_frequencies(generated)
    for base in BASES:
        print(f"freq({base})            {real_freq[base]:>6.3f}              {gen_freq[base]:>6.3f}")

    real_gc, gen_gc = gc_content(real), gc_content(generated)
    print(f"GC mean +- sd     {real_gc.mean():.3f} +- {real_gc.std():.3f}"
          f"       {gen_gc.mean():.3f} +- {gen_gc.std():.3f}")
    print(f"uATG per seq          {uaug_rate(real):>6.3f}              {uaug_rate(generated):>6.3f}")
    print(f"distinct fraction     {distinct_fraction(real):>6.3f}              "
          f"{distinct_fraction(generated):>6.3f}")

    print("\nk-mer Jensen-Shannon divergence vs real (bits, lower is better)")
    for k in range(1, max_k + 1):
        div = jensen_shannon(kmer_distribution(real, k), kmer_distribution(generated, k))
        print(f"  k={k}   {div:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="Reference sequences, one per line")
    parser.add_argument("--generated", required=True, help="Generated sequences, one per line")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Compare this many trailing bases (default: generated length)")
    parser.add_argument("--max_k", type=int, default=6, help="Largest k-mer size to score")
    opt = parser.parse_args()

    seq_len = opt.seq_len
    if seq_len is None:
        with open(opt.generated) as fin:
            seq_len = len(fin.readline().strip())

    report(load(opt.real, seq_len), load(opt.generated, seq_len), opt.max_k)


if __name__ == "__main__":
    main()
