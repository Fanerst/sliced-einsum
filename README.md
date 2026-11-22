# sliced-einsum

A prototype of a sliced einsum on [tenferro](https://github.com/tensor4all/tenferro-rs), for
[tensor4all/tenferro-rs#2676](https://github.com/tensor4all/tenferro-rs/issues/2676). It is v0 in
that issue's sense: CPU only, contracted labels only, built on tenferro's public API at `main`
c4db040 and on [omeco](https://github.com/GiggleLiu/omeco) 0.2.6, the version tenferro already pins.
It shows the shape of the three steps in the maintainers' plan; it does not propose an API
spelling, which the issue's design note decides.

## What it does

1. **The memory target is a constraint.** `SlicedPlan::new` takes a contraction tree (omeco's
   TreeSA, or SSA pairs from the caller) and runs omeco's `slice_code` with the target. The slicer
   picks labels to slice until the largest intermediate of one slice fits. A target it cannot meet
   is an error at planning time, before anything runs. The `PlanReport` gives the sliced labels,
   the slice count, the flops, largest intermediate and peak elements of one slice, and the total
   flops against the unsliced tree.
2. **A slice loop.** `SlicedPlan::prepare` builds one tenferro `ConcreteEinsumPlan` for the sliced
   network, with the plan's tree passed as `EinsumOptimize::Tree(ContractionTree::from_pairs(..))`.
   For each assignment of the sliced labels, the executor takes strided views of the operands
   that carry them (`TypedTensorView::from_slice`, no copies) and runs that plan into the
   destination with `Dest::accumulate`.
3. **Slice ranges.** `Prepared::execute_range` runs any range of slice ids into a caller-owned
   accumulator. Disjoint ranges summed give the full result, which is the hook a multi-device or
   MPI driver needs.

Only contracted labels are sliced, as in the issue's v0: an output label among the sliced labels
is a planning error.

## Quick start

```rust
use std::collections::HashMap;
use sliced_einsum::{Network, SliceOptions, SlicedPlan, ViewMode};
use tenferro_cpu::CpuEngine;
use tenferro_tensor::Tensor;

// ij,jk,ki-> (the trace of a product of three 3x3 matrices), with j sliced.
let network = Network {
    inputs: vec![vec![0, 1], vec![1, 2], vec![2, 0]],
    output: vec![],
    sizes: HashMap::from([(0, 3), (1, 3), (2, 3)]),
};
let options = SliceOptions { slicing: Some(vec![1]), ..SliceOptions::default() };
let plan = SlicedPlan::new(&network, &options)?;
assert_eq!(plan.report().slices, 3);

let a = Tensor::from_vec_col_major(vec![3, 3], (0..9).map(f64::from).collect())?;
let prepared = plan.prepare(&[&a, &a, &a])?;
let (trace, _) = prepared.execute(&[&a, &a, &a], &mut CpuEngine::serial(), ViewMode::Offset)?;
assert_eq!(trace.as_slice::<f64>()?, &[2376.0]);
```

With `SliceOptions { sc_target: 24.0, ..SliceOptions::default() }` instead, TreeSA finds the tree
and the slicer chooses the labels.

## Layout

| Path | Contents |
|---|---|
| `src/plan.rs` | `Network`, `SliceOptions`, `SlicedPlan`, `PlanReport`: tree, slicer, cost, SSA pairs |
| `src/exec.rs` | `Prepared`: the views of a slice, the slice loop, ranges, `ViewMode` |
| `src/network.rs` | The 30-qubit circuit of [sampled-einsum](https://github.com/Fanerst/sampled-einsum) (its `data/n30m14`), with output qubits fixed or open |
| `tests/dense_reference.rs` | Sliced against unsliced on random networks (see below) |
| `examples/ising_sliced.rs` | The open cubic Ising model in spin-hypergraph form: plan, slice, run, export |
| `examples/circuit_sliced.rs` | The circuit with some qubits open, checked against the reference amplitudes |
| `scripts/bench_ising.sh` | One tree per lattice, one slicing per target, then tenferro and PyTorch in turns |
| `scripts/bench_torch.py` | An exported plan run in PyTorch, step by step on the same tree and slicing |

## Tests

`cargo test` checks the slice loop against tenferro's own unsliced einsum, on random networks
with hyperedges (labels on three or more operands) and extents 1 to 3:

- explicit slicings of one to three contracted labels on the same tree, with both view modes,
  open and closed networks (over 100 plans);
- every contracted label sliced, so each slice is an outer product;
- a sliced label of extent 1;
- an output label in the slicing, which planning rejects;
- the slicer under a target two bits below the unsliced tree, on graph-shaped networks of 20
  operands, where every plan meets the target;
- disjoint ranges summed against the whole, and a range past the slice count rejected;
- complex128 operands.

## Running the examples

```sh
# 6x6x6 at beta = 0.2 with 16 threads: a TreeSA tree (default preset), sliced to 2^24 elements
cargo run --release --example ising_sliced -- 6 6 6 0.2 16 treesa-default 24 offset

# The benchmark of one lattice: one tree, slicings at 2^26 and 2^24, then tenferro and PyTorch
# in turns, 2 runs each (PY is a Python with PyTorch)
cargo build --release --examples
PY=python3 scripts/bench_ising.sh 6 6 6 "26 24" 2 bench/mine default

# The circuit reads its data from the sampled-einsum repository
git clone https://github.com/Fanerst/sampled-einsum ../sampled-einsum
cargo run --release --example circuit_sliced -- ../sampled-einsum/data/n30m14 16 treesa none offset
```

TreeSA and the slicer are randomized, so a rerun can find a different tree and slicing; the
export files that `scripts/bench_ising.sh` writes keep each plan so it can be replayed.

## Results

The benchmark is the partition function of the open cubic Ising model at β = 0.2 in
spin-hypergraph form: one extent-2 label per spin, and a 2x2 tensor per bond. Each spin's label
sits on all of its bonds, so every spin is a hyperedge of up to six operands. `scripts/bench_ising.sh`
makes one TreeSA tree per lattice (omeco's default preset) and one slicing per target on that
tree. It then runs each saved plan in turns:
- with tenferro, two or three runs with each view mode;
- with PyTorch 2.5.1 on the same tree and slicing, each step a `torch.einsum` of two tensors
  (`scripts/bench_torch.py`).

Both engines use 16 threads (a 16-thread `CpuEngine` pool; `torch.set_num_threads(16)` with MKL)
on a shared 2 x 24-core Intel Xeon Platinum 8268 host, in f64. The records are under `bench/`, one
folder per slicer preset:
- the exported plans: `*-none.json` for the unsliced tree and `*-tNN.json` for each target. Pass one
  as `TREE` with `TARGET` = `saved` to replay its tree and slicing exactly;
- the planning logs (`*-plan.log`);
- the runs: `*-runs.jsonl` for tenferro and `*-torch.jsonl` for PyTorch.

**Correctness:** every run of a lattice gives the same ln Z to ten decimals, in both engines, at
every slicing and with both view modes: 161.3559583740 for 6x6x6 and 149.2822137108 for 5x5x8.

### 6x6x6

The unsliced tree's largest intermediate holds 2^30 elements (8 GiB), and the tree takes 6.3e11
flops. Flops are relative to that tree.

| Target | Slicer | Slices | Flops | tenferro | PyTorch | Peak memory, tenferro / PyTorch |
|---|---|---|---|---|---|---|
| none | | 1 | 1.00x | 6.6–8.0 s | 10.8–10.9 s | 17.2 / 24.5 GiB |
| 2^28 | fast | 4 | 1.01x | 9.7–11.8 s | 13.3–13.6 s | 4.4 / 6.4 GiB |
| 2^26 | fast | 16 | 1.78x | 37.0–38.5 s | 22.9–23.7 s | 2.2 / 2.7 GiB |
| 2^26 | default | 16 | 1.13x | 14.3–14.8 s | 16.6–18.3 s | 1.2 / 1.9 GiB |
| 2^24 | fast | 64 | 1.38x | 29.9–30.4 s | 40.0–40.2 s | 0.6 / 1.0 GiB |
| 2^24 | default | 64 | 1.13x | 10.8–11.0 s | 33.0–33.4 s | 0.4 / 0.8 GiB |
| 2^22 | fast | 256 | 1.36x | 59.9–60.4 s | 38.1–39.2 s | 0.2 / 0.6 GiB |

With the default slicer, a 2^24 target cuts the peak from 17.2 GiB to 0.4 GiB for 1.13x the flops
and 1.45x the time.

### 5x5x8

The unsliced tree's largest intermediate holds 2^25 elements, and the tree takes 5.5e9 flops. The
slicer can rebuild the tree while it slices, so a sliced plan can need fewer flops than the
unsliced tree.

| Per-slice largest | Slicer | Slices | Flops | tenferro | PyTorch |
|---|---|---|---|---|---|
| 2^25 (unsliced) | | 1 | 1.00x | 0.48–0.77 s | 1.44–1.53 s |
| 2^20 | fast | 32 | 0.81x | 0.64–0.73 s | 2.23–2.38 s |
| 2^19 | fast | 64 | 1.41x | 0.92–0.99 s | 2.97–3.08 s |
| 2^19 | default | 64 | 0.72x | 0.66–0.70 s | 2.52–2.71 s |
| 2^17 | fast | 1,024 | 1.94x | 4.23–4.33 s | 29.9–30.3 s |
| 2^17 | default | 512 | 1.12x | 2.08–2.14 s | 15.2–15.8 s |
| 2^15 | fast | 16,384 | 10.42x | 36.4–36.9 s | 364–372 s |

### What the numbers show

- **The memory target holds.** Peak resident memory follows the per-slice bound, from 17.2 GiB
  unsliced to 0.2 GiB at 2^22 on 6x6x6.
- **The slicer's result varies.** omeco's slicer is randomized, and on the same tree the fast
  preset gave 1.38x and 1.80x the flops in two runs at 2^24, against 1.13x and 1.17x for the
  default preset. At 2^22 the default preset once gave 1,024 slices and 2.05x. The plan's total
  flops, which the report gives before anything runs, are the number to check.
- **The slice loop itself is cheap.** At 16,384 slices of 2^15 elements (8.1 million binary steps),
  tenferro takes 73x the unsliced time for 10.4x the flops. Assuming the unsliced flop rate, that
  is about 4 µs per step on top of the flops, where PyTorch spends about 45 µs per `torch.einsum`.
- **Every slice re-plans every step.** tenferro's prepared einsum dispatches each binary step
  through the uncached `dot_general_read`, so `CpuEngine::plan_cache()` records no hits or misses,
  and offset and rebased views take the same time. The offset in the CPU binary-plan key, which
  [#2676](https://github.com/tensor4all/tenferro-rs/issues/2676) worried about, is not on this
  path. What a slice repeats is the planning of each step, which a prepared binary contraction
  replayed on new buffers ([#2705](https://github.com/tensor4all/tenferro-rs/issues/2705)) would
  remove.
- **Which engine wins depends on the steps.** `examples/ising_steps.rs` and `scripts/steps_torch.py`
  time slice 0 one step at a time. This runs each step as its own einsum, 2–4x slower in total
  than the prepared plan, so it ranks step kinds rather than reproducing the plan's times.
  - Steps dominated by matrix multiplication are faster in tenferro: one 2^39-flop step of the
    unsliced 6x6x6 plan takes 2.25 s against 4.82 s.
  - Steps whose result is large for their flop count are much slower. These keep most shared
    labels as batch labels and sum one or two: a rank-30 by rank-6 step with four batch labels,
    one summed label and a 2^30-element (8 GiB) result takes 7.7 s against 1.6 s.
  - The contraction is not what costs on those steps. That step was timed in parts on another
    host (two Xeon Gold 6530, 16 cores pinned):
    - `dot_general_read` does the contraction in about 0.5 s;
    - `einsum` then takes two more passes to put the result in the requested label order: a copy
      of the whole result on one thread (about 4.8 s), then a transpose that uses few threads
      (about 0.4 s);
    - contracting straight into a transposed view of a destination already in the requested
      order (`dot_general_read_into` on a `transpose_view`) gives the same result in about 0.5 s.
  - Sliced plans keep the same kind of step, at the size of a slice. In slice 0 of each
    fast-preset plan, the slowest steps again sum one to three labels into a result at the
    slice's bound; at 2^26, two of them take 0.7 s and 1.0 s against 0.07 s and 0.18 s.
  - This cost comes from how tenferro's einsum orders a result, not from slicing. It is the
    likely reason PyTorch wins on the two fast-preset plans, at 2^26 and 2^22.

### The 30-qubit circuit

The circuit of the sampled-einsum prototype is too shallow at 14 cycles for slicing to matter.
Closed at one bitstring, it takes 2.0e8 flops with a 2^20 largest intermediate on a TreeSA tree,
0.14 s. With the artensor tree that comes with the data, which was found for the open network, it
takes 3.7e11 flops with a 2^30 largest intermediate and 42 s. Leaving 24 qubits open makes the
output itself the largest tensor, which v0 does not slice. `examples/circuit_sliced.rs` runs it and
checks bitstring 0's amplitude against the reference amplitudes in the data (agreement to 1.8e-5,
the reference's precision).

## Observations for the design note

These are the points where the numbers above bear on the open questions of
[#2676](https://github.com/tensor4all/tenferro-rs/issues/2676).

- **A separate plan rather than an `EinsumOptions` field.** The loop and the partitioning hook
  both need the slice count and per-slice cost before execution: to split ranges across workers,
  and to judge a slicing whose flops vary from 0.7x to 10x with the target and the slicer's seed.
  A plan object holds that; an options field alone would not expose it.
- **Opt-in rather than automatic slicing.** Slicing changed the time of a 6x6x6 run by 1.4x to 8x
  depending on the target and the slicing found. Part of the high end is the result-ordering cost
  above, which can be fixed apart from slicing. Slicing silently when a target is missed would
  make that cost appear without warning; a typed planning error with the report is what the
  prototype does.
- **A seed.** TreeSA and the slicer are randomized, so a deterministic plan (cf.
  [#1963](https://github.com/tensor4all/tenferro-rs/issues/1963)) needs a seed, or best-of-N with
  fixed seeds.
- **Plan reuse across slices** is worth about 4 µs per step here on CPU. It matters for many small
  slices, and is the same replay of a prepared binary contraction as
  [#2705](https://github.com/tensor4all/tenferro-rs/issues/2705).

## What v0 leaves out

- **AD.** The VJP of a sliced contraction is a sliced contraction of per-slice VJPs, recomputing
  each slice's forward pass; nothing here differentiates.
- **CUDA.** The views and the loop would work on a device engine, but nothing here was run on one.
- **Output-label slicing,** where each slice writes a disjoint block of the output.
- **Traced mode.** A slice loop in a traced graph would be a sub-program run once per slice
  ([#2691](https://github.com/tensor4all/tenferro-rs/issues/2691)), not an unrolled graph.

## Provenance and license

Written with Claude Code (model `claude-opus-5-5`) at the repository owner's request. Licensed
under MIT or Apache-2.0, at your option.
