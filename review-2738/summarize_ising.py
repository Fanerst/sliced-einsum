"""Summarize logs/ising.jsonl of the #2738 review: time ranges per plan and implementation.

Usage: python3 summarize_ising.py logs/ising.jsonl
"""

import collections
import json
import math
import statistics
import sys

NAMES = {
    "d-6x6x6-none": "6x6x6, unsliced",
    "d-6x6x6-t26": "6x6x6, 2^26",
    "d-6x6x6-t24": "6x6x6, 2^24",
    "d-5x5x8-none": "5x5x8, unsliced",
    "d-5x5x8-t17": "5x5x8, 2^17",
    "f-5x5x8-t15": "5x5x8, 2^15",
}

times = collections.defaultdict(list)
rss = collections.defaultdict(list)
lnz = collections.defaultdict(set)
port = {}
for line in open(sys.argv[1]):
    d = json.loads(line)
    key = (d["plan"], d.get("impl"))
    for run in d["runs"]:
        times[key].append(run["seconds"])
        lnz[d["plan"]].add(round(run["ln_z"], 10))
    rss[key].append(d["peak_rss_gib"])
    if d.get("impl") == "port":
        r = d["report"]
        port[d["plan"]] = {
            "slices": r["slice_count"],
            "flops": r["total_flops"],
            "log2_peak": r["log2_peak"],
            "log2_arena": math.log2(d["arena_elements"]),
            "misses": [x["plan_misses"] for x in d["runs"]],
            "hits": [x["plan_hits"] for x in d["runs"]],
        }


def rng(xs):
    if not xs:
        return "-"
    return f"{min(xs):.2f}-{max(xs):.2f} s (median {statistics.median(xs):.2f}, n={len(xs)})"


for plan, name in NAMES.items():
    p = port[plan]
    print(f"{name:16} slices {p['slices']:6d}")
    for impl in ("proto-c4db040", "proto", "port"):
        print(f"    {impl:14} {rng(times.get((plan, impl), []))}"
              f"  rss {max(rss.get((plan, impl), [float('nan')])):.2f} GiB")
    print(f"    ln Z {sorted(lnz[plan])}  arena/peak 2^{p['log2_arena'] - p['log2_peak']:.2f}"
          f" ({2 ** (p['log2_arena'] - p['log2_peak']):.2f}x)  misses {p['misses']} hits {p['hits'][:2]}")
