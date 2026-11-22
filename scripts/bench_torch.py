#!/usr/bin/env python3
"""Run an exported sliced plan of the cubic Ising model in PyTorch, on the same tree and slicing.

Usage: bench_torch.py EXPORT THREADS REPEATS LABEL

EXPORT is a file written by `ising_sliced ... EXPORT`. Every slice takes its views of the bond
tensors at the slice's values (the first sliced label fastest, as in the Rust executor), then
contracts the exported steps pairwise with torch.einsum, keeping the labels each step keeps, and
adds the slice's result to the total. Prints one JSON line per run.
"""

import json
import math
import resource
import sys
import time

import torch

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main():
    path, threads, repeats, label = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    torch.set_num_threads(threads)
    with open(path) as f:
        plan = json.load(f)
    beta = plan["beta"]
    inputs = plan["inputs"]
    sliced = plan["sliced"]
    steps = plan["steps"]
    same, flip = math.exp(beta), math.exp(-beta)
    bond = torch.tensor([[same, flip], [flip, same]], dtype=torch.float64)
    slices = 2 ** len(sliced)  # every spin label has extent 2

    def one_slice(slice_id):
        values = {l: (slice_id >> k) & 1 for k, l in enumerate(sliced)}
        tensors, labels = [], []
        for ls in inputs:
            tensors.append(bond[tuple(values[l] if l in values else slice(None) for l in ls)])
            labels.append([l for l in ls if l not in values])
        for a, b, kept in steps:
            names = {}
            for l in labels[a] + labels[b]:
                names.setdefault(l, LETTERS[len(names)])
            spec = "{},{}->{}".format(
                "".join(names[l] for l in labels[a]),
                "".join(names[l] for l in labels[b]),
                "".join(names[l] for l in kept),
            )
            tensors.append(torch.einsum(spec, tensors[a], tensors[b]))
            labels.append(kept)
            tensors[a] = tensors[b] = None
        return tensors[-1]

    for _ in range(repeats):
        started = time.perf_counter()
        z = torch.zeros((), dtype=torch.float64)
        for slice_id in range(slices):
            z += one_slice(slice_id)
        seconds = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print(json.dumps({
            "plan": label, "engine": "torch " + torch.__version__, "threads": threads,
            "slices": slices, "seconds": seconds, "ln_z": math.log(z.item()), "peak_rss_gib": peak,
        }), flush=True)


if __name__ == "__main__":
    main()
