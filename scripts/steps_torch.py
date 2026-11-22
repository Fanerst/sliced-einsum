#!/usr/bin/env python3
"""Slice 0 of an exported cubic Ising plan in PyTorch, one timed step at a time.

Usage: steps_torch.py EXPORT THREADS [TOP]

The PyTorch counterpart of `examples/ising_steps.rs`: prints one JSON line with the total and the
TOP (default 12) slowest steps, with the same fields.
"""

import json
import math
import sys
import time

import torch

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main():
    path, threads = sys.argv[1], int(sys.argv[2])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    torch.set_num_threads(threads)
    with open(path) as f:
        plan = json.load(f)
    beta, inputs, sliced, steps = plan["beta"], plan["inputs"], set(plan["sliced"]), plan["steps"]
    same, flip = math.exp(beta), math.exp(-beta)
    bond = torch.tensor([[same, flip], [flip, same]], dtype=torch.float64)
    tensors, labels = [], []
    for ls in inputs:
        tensors.append(bond[tuple(0 if l in sliced else slice(None) for l in ls)])
        labels.append([l for l in ls if l not in sliced])
    records = []
    started = time.perf_counter()
    for k, (a, b, kept) in enumerate(steps):
        names = {}
        for l in labels[a] + labels[b]:
            names.setdefault(l, LETTERS[len(names)])
        spec = "{},{}->{}".format(
            "".join(names[l] for l in labels[a]),
            "".join(names[l] for l in labels[b]),
            "".join(names[l] for l in kept),
        )
        t0 = time.perf_counter()
        out = torch.einsum(spec, tensors[a], tensors[b])
        seconds = time.perf_counter() - t0
        union = set(labels[a]) | set(labels[b])
        records.append({
            "step": k, "left_rank": len(labels[a]), "right_rank": len(labels[b]),
            "out_rank": len(kept), "shared": len(set(labels[a]) & set(labels[b])),
            "batch": len([l for l in kept if l in labels[a] and l in labels[b]]),
            "log2_flops": len(union), "seconds": seconds,
        })
        tensors[a] = tensors[b] = None
        tensors.append(out)
        labels.append(kept)
    total = time.perf_counter() - started
    records.sort(key=lambda r: -r["seconds"])
    print(json.dumps({"engine": "torch " + torch.__version__, "threads": threads,
                      "steps": len(steps), "seconds": total, "slowest": records[:top]}))


if __name__ == "__main__":
    main()
