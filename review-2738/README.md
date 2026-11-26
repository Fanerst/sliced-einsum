# Review of tensor4all/tenferro-rs#2738

The measurements behind
[the review of #2738](https://github.com/tensor4all/tenferro-rs/pull/2738#issuecomment-7183467617),
tenferro-rs's port of this prototype (`SlicedEinsumPlan`, merged as `b0f5fdf2`).

Everything here ran on tenferro-rs `main` `1b67fb8f`, which also has the sampled einsum (#2739).
The runs used the CPU only, in f64, on the shared 2 x 24-core Intel Xeon Platinum 8268 host of
the prototype's own benchmarks. The review gives the results. The JSON lines are in `logs/`.

## Build

The harness builds on a tenferro-rs snapshot by path, since the host could not reach github.com.
Run this from this folder:

```bash
curl -L https://codeload.github.com/tensor4all/tenferro-rs/tar.gz/1b67fb8f8606f820cf33a0d5ceab0ef21d63612a | tar xz
```

```bash
mv tenferro-rs-1b67fb8f8606f820cf33a0d5ceab0ef21d63612a tenferro-rs-1b67fb8f
```

```bash
(cd harness && cargo build --release --locked --bins --examples)
```

`harness/` is a workspace of two crates:

- **`proto`:** the repository root's `src/` and `examples/ising_sliced.rs`, unchanged, built on
  `1b67fb8f` instead of `c4db040`.
- **`port`:** the prototype's benchmark through `SlicedEinsumPlan` (`ising_port`), and the planning
  probes below.

The control build of the prototype as published, on tenferro-rs `c4db040`, is the repository
root itself (`cargo build --release --example ising_sliced` there).

## What each finding ran

The numbering follows the review. The planning probes ran with `RAYON_NUM_THREADS` between 2
and 8, on cores that the timed runs did not use. `network.json` is `data/n30m14/network.json` of
[Fanerst/sampled-einsum `c482b79a`](https://github.com/Fanerst/sampled-einsum/tree/c482b79a0f1e407390ef7822a89e7706220021d1/data/n30m14):
the 30-qubit, 14-cycle circuit, with artensor's tree as its `path`.

1. **The output-label fallback** (`fullstate.jsonl`, `fullstate-auto-sweep.jsonl`,
   `strip-prune.jsonl`):
   - `fullstate_probe network.json all 30,31,32,33,34 auto`. It plans with the port's slicer on
     the explicit tree. It also plans omeco's add rule with output labels skipped
     (`exclusion_greedy`, then passed as explicit labels), and the default `Auto`. The same was
     run with `24 24,25,26,27,28,29,30 auto` and `12 24,26,28,30`.
   - The sweep: `fullstate_probe network.json OPEN T..T+6 auto` with T = OPEN, for OPEN = 16, 20,
     24, 28 and 30, three times.
   - `strip_prune_probe network.json all 30,31,32,33 3`, and the same with `24 24,25,26,27 3` and
     `20 20,21,22,23 3`. It replays the fallback and prunes its labels at the true extents.
2. **Determinism** (`det-*.jsonl`, `partition*.jsonl`):
   - `det_probe ising:6:6:6:../bench/default/6x6x6-none.json tree 24 20`, and the same at 22.
     Likewise `ising:5:5:8:../bench/fast/5x5x8-none.json` at 15 and 17.
   - These ran three ways: with omeco 0.2.6 as is (`det-stock`), with `omeco-det1.diff`
     (omeco#45's tie-break only, `det-det1`), and with `omeco-det2.diff` (also `Slicer::legs` as a
     `BTreeMap`, `det-det2`). For a variant, apply the diff in a copy of omeco 0.2.6's source (`patch -p1` in its root) and add
     `[patch.crates-io] omeco = { path = "..." }` to `harness/Cargo.toml`.
   - `partition_probe 5 5 8 ../bench/fast/5x5x8-none.json 19 8 20`, twice per variant.
     `partition.jsonl` is a first run at zero field, where every split matched by spin-flip
     symmetry. `partition-field.jsonl` has the field on the bonds.
3. **Operands:** `operand_probe 256 12 4 2` (`operand.jsonl`).
4. **Plan quality:** `det_probe` with `auto` and `autosa` (`det-stock-auto.jsonl`,
   `det-stock-autosa.jsonl`).

**Benchmark** (`ising.jsonl`):

- **The runs:** `CPUS=$(./pick_cpus.sh) ROUNDS=2 REP=2 ./run_ising.sh`, then
  `./run_control.sh` with the same `CPUS`. `python3 summarize_ising.py logs/ising.jsonl`
  summarizes them.
- **Cores:** `pick_cpus.sh` takes the 16 least-busy cores of the host's second NUMA node; adjust
  its ranges for another host.
- **Plan names:** `d-NAME` is `../bench/default/NAME.json`, and `f-NAME` is
  `../bench/fast/NAME.json`. The logs give them as `bench/d-NAME.json`, from the run's own copies of
  those files.
