//! Usage: circuit_sliced DATA_DIR THREADS TREE TARGET MODE [REPEATS] [OPEN]
//!
//! - `DATA_DIR`: the sampled-einsum prototype's `data/n30m14`.
//! - `TREE`: `treesa` (omeco's TreeSA, fast preset), `treesa-default` (its default preset, with
//!   the slicer's default too), or `artensor` (the bundled tree of the open network, which stays
//!   valid once output labels are fixed).
//! - `TARGET`: log2 of the most elements one slice's largest intermediate may hold, or `none`.
//! - `MODE`: `offset` or `rebased` (see `ViewMode`).
//! - `REPEATS` (default 2) runs.
//! - `OPEN` (default 0): the last `OPEN` qubits stay open; the others are fixed at bitstring 0.
//!
//! With `OPEN = 0` the network is a scalar contraction whose value is bitstring 0's amplitude;
//! with `OPEN = k` it gives the `2^k` amplitudes that agree with bitstring 0 on the fixed qubits,
//! one of which is bitstring 0's. The example plans the network, slices it to the target with
//! omeco's slicer, runs it, and prints one JSON line: the plan, each run's time and plan-cache
//! counts, bitstring 0's amplitude against the reference, and the process's peak resident memory.

use std::path::Path;
use std::time::Instant;

use num_complex::Complex64;
use sliced_einsum::{network, SliceOptions, SlicedPlan, ViewMode};
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_tensor::Tensor;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage: circuit_sliced DATA_DIR THREADS TREE TARGET MODE [REPEATS] [OPEN]";

/// The process's peak resident set, in GiB, from `/proc/self/status` (Linux only).
fn peak_rss_gib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / (1024.0 * 1024.0))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        return Err(USAGE.into());
    }
    let dir = Path::new(&args[1]);
    let threads: usize = args[2].parse()?;
    let tree_source = args[3].as_str();
    let target: Option<f64> = match args[4].as_str() {
        "none" => None,
        t => Some(t.parse()?),
    };
    let mode = match args[5].as_str() {
        "offset" => ViewMode::Offset,
        "rebased" => ViewMode::Rebased,
        other => return Err(format!("mode must be offset or rebased, not {other}").into()),
    };
    let repeats: usize = args.get(6).map_or(Ok(2), |r| r.parse())?;
    let open: usize = args.get(7).map_or(Ok(0), |o| o.parse())?;

    let circuit = network::load(dir)?;
    let bitstring = circuit.bitstrings[0].clone();
    let reference = circuit.reference[0];
    let qubits = bitstring.len();
    if open > qubits {
        return Err(format!("{open} open qubits of {qubits}").into());
    }
    let pattern = format!("{}{}", &bitstring[..qubits - open], "*".repeat(open));
    // Bitstring 0's entry of the result: its open qubits, the first one fastest.
    let index = bitstring[qubits - open..]
        .chars()
        .enumerate()
        .map(|(j, c)| usize::from(c == '1') << j)
        .sum::<usize>();
    let (network, tensors) = circuit.fixed(&pattern)?;
    let operands: Vec<&Tensor> = tensors.iter().collect();

    let fast = tree_source != "treesa-default";
    let started = Instant::now();
    let tree = match tree_source {
        "treesa" | "treesa-default" => SlicedPlan::new(
            &network,
            &SliceOptions {
                slicing: Some(Vec::new()),
                fast,
                ..SliceOptions::default()
            },
        )?
        .pairs()
        .to_vec(),
        "artensor" => circuit.path.clone(),
        other => {
            return Err(
                format!("tree must be treesa, treesa-default or artensor, not {other}").into(),
            )
        }
    };
    let tree_seconds = started.elapsed().as_secs_f64();
    let options = SliceOptions {
        sc_target: target.unwrap_or(f64::INFINITY),
        tree: Some(tree),
        slicing: if target.is_some() {
            None
        } else {
            Some(Vec::new())
        },
        fast,
    };
    let plan = SlicedPlan::new(&network, &options)?;
    let plan_seconds = started.elapsed().as_secs_f64() - tree_seconds;
    let report = plan.report();
    let prepared = plan.prepare(&operands)?;

    let mut engine = if threads > 1 {
        CpuEngine::new(Pool::builder().threads(threads).build()?)
    } else {
        CpuEngine::serial()
    };
    let mut runs = Vec::new();
    for _ in 0..repeats {
        let (out, stats) = prepared.execute(&operands, &mut engine, mode)?;
        let amplitude = out.as_slice::<Complex64>()?[index];
        runs.push(serde_json::json!({
            "seconds": stats.seconds,
            "plan_misses": stats.plan_misses,
            "plan_hits": stats.plan_hits,
            "amplitude": [amplitude.re, amplitude.im],
            "relative_error": (amplitude - reference).norm() / reference.norm(),
        }));
    }
    let cost = |c: &sliced_einsum::Cost| {
        serde_json::json!({
            "flops": c.flops,
            "log2_largest": c.largest.log2(),
            "log2_peak": c.peak.log2(),
        })
    };
    let line = serde_json::json!({
        "threads": threads,
        "tree": tree_source,
        "open": open,
        "target": target,
        "mode": format!("{mode:?}"),
        "tree_seconds": tree_seconds,
        "slice_plan_seconds": plan_seconds,
        "sliced_labels": report.sliced_labels,
        "slices": report.slices,
        "per_slice": cost(&report.per_slice),
        "total_flops": report.total_flops,
        "unsliced": cost(&report.unsliced),
        "flop_overhead": report.total_flops / report.unsliced.flops,
        "runs": runs,
        "peak_rss_gib": peak_rss_gib(),
    });
    println!("{line}");
    Ok(())
}
