//! Usage: ising_port LX LY LZ BETA THREADS PLANFILE MODE REPEATS
//!
//! The prototype's open cubic Ising benchmark (`examples/ising_sliced.rs` of
//! Fanerst/sliced-einsum) through tenferro-einsum's `SlicedEinsumPlan`.
//!
//! - `PLANFILE`: a prototype export (`pairs`: the unsliced tree; `sliced` and `sliced_pairs`:
//!   the sliced labels and the tree of one slice).
//! - `MODE`: `saved` replays the export's labels on its slice tree (an explicit `Tree`);
//!   `none` runs the unsliced tree as one slice; `tT` lets the port's slicer choose labels for
//!   target 2^T on the export's unsliced tree (explicit order: labels only); `autoT` plans
//!   from scratch (`EinsumOptimize::default()`, then the slicer) for target 2^T.
//!
//! Prints one JSON line: the report, planning time, each run's time, `ln Z`, the per-slice
//! plan's cache statistics and arena, and peak resident memory.

use std::time::Instant;

use port_bench::{peak_rss_gib, plan_pairs, report_json, Net, Result};
use serde_json::json;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::{EinsumOptimize, EinsumOptions, SliceOptions, SlicedEinsumPlan};
use tenferro_tensor::Tensor;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 9 {
        return Err("usage: ising_port LX LY LZ BETA THREADS PLANFILE MODE REPEATS".into());
    }
    let (lx, ly, lz): (u32, u32, u32) = (args[1].parse()?, args[2].parse()?, args[3].parse()?);
    let beta: f64 = args[4].parse()?;
    let threads: usize = args[5].parse()?;
    let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[6])?)?;
    let mode = args[7].as_str();
    let repeats: usize = args[8].parse()?;

    let net = Net::ising(lx, ly, lz);
    let (same, flip) = (beta.exp(), (-beta).exp());
    let tensors: Vec<Tensor> = net
        .inputs
        .iter()
        .map(|_| Tensor::from_vec_col_major(vec![2, 2], vec![same, flip, flip, same]))
        .collect::<std::result::Result<_, _>>()?;
    let inputs: Vec<&Tensor> = tensors.iter().collect();

    let (optimize, slicing) = if mode == "saved" {
        let labels: Vec<u32> = serde_json::from_value(saved["sliced"].clone())?;
        (
            EinsumOptimize::Tree(net.tree(&plan_pairs(&saved, "sliced_pairs")?)?),
            SliceOptions::default().labels(labels),
        )
    } else if mode == "none" {
        (
            EinsumOptimize::Tree(net.tree(&plan_pairs(&saved, "pairs")?)?),
            SliceOptions::default(),
        )
    } else if let Some(t) = mode.strip_prefix("auto") {
        (
            EinsumOptimize::default(),
            SliceOptions::default().target_log2_elements(t.parse()?),
        )
    } else if let Some(t) = mode.strip_prefix('t') {
        (
            EinsumOptimize::Tree(net.tree(&plan_pairs(&saved, "pairs")?)?),
            SliceOptions::default().target_log2_elements(t.parse()?),
        )
    } else {
        return Err(format!("unknown mode {mode}").into());
    };

    let started = Instant::now();
    let plan = SlicedEinsumPlan::prepare(
        &inputs,
        &net.spec(),
        &EinsumOptions::default().optimize(optimize),
        &slicing,
    )?;
    let plan_seconds = started.elapsed().as_secs_f64();

    let mut engine = CpuEngine::new(Pool::builder().threads(threads).build()?);
    let mut runs = Vec::new();
    for _ in 0..repeats {
        let before = plan.cpu_cache_stats()?;
        let engine_before = engine.plan_cache().stats();
        let t0 = Instant::now();
        let out: Tensor = plan.execute(&inputs, &mut engine)?;
        let seconds = t0.elapsed().as_secs_f64();
        let after = plan.cpu_cache_stats()?;
        let engine_after = engine.plan_cache().stats();
        let z = out.as_slice::<f64>()?[0];
        runs.push(json!({
            "seconds": seconds,
            "ln_z": z.ln(),
            "plan_misses": after.misses - before.misses,
            "plan_hits": after.hits - before.hits,
            "engine_misses": engine_after.misses - engine_before.misses,
            "engine_hits": engine_after.hits - engine_before.hits,
        }));
    }
    let line = json!({
        "impl": "port",
        "lattice": [lx, ly, lz],
        "beta": beta,
        "threads": threads,
        "plan_file": args[6],
        "mode": mode,
        "plan_seconds": plan_seconds,
        "report": report_json(plan.report()),
        "arena_elements": plan.cpu_arena_elements(),
        "runs": runs,
        "peak_rss_gib": peak_rss_gib(),
    });
    println!("{line}");
    Ok(())
}
