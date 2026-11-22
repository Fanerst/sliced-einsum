//! Usage: ising_sliced LX LY LZ BETA THREADS TREE TARGET MODE [REPEATS] [EXPORT] [SLICER]
//!
//! The partition function of the open `LX x LY x LZ` cubic Ising model with unit couplings, in
//! spin-hypergraph form: one label of extent 2 per spin, one 2x2 tensor `exp(beta s_i s_j)` per
//! bond, and each spin's label on all of its bonds, so every spin is a hyperedge of up to six
//! operands.
//!
//! - `TREE`: `treesa` (omeco's TreeSA, fast preset), `treesa-default` (its default preset, with
//!   the slicer's default too), or the path of an earlier `EXPORT` file, whose `pairs` are used
//!   as the tree (TreeSA is randomized, so runs that compare slicings should share one tree).
//! - `TARGET`: log2 of the most elements one slice's largest intermediate may hold, `none`, or
//!   `saved` to replay the sliced labels and slice tree of the `TREE` export file exactly (omeco's
//!   slicer is randomized too).
//! - `MODE`: `offset` or `rebased` (see `ViewMode`).
//! - `REPEATS` (default 2) runs.
//! - `EXPORT`: write the lattice, beta, the unsliced tree's `pairs`, the sliced labels and the
//!   steps of one slice to this JSON file, for `scripts/bench_torch.py`; `-` writes nothing.
//! - `SLICER`: `fast` or `default`, omeco's preset for the slicer when `TREE` is a file. With
//!   `treesa` and `treesa-default` the slicer takes the same preset as TreeSA.
//!
//! Prints one JSON line: the plan, each run's time and `ln Z`, and the process's peak resident
//! memory.

use std::collections::HashMap;
use std::time::Instant;

use sliced_einsum::{Network, SliceOptions, SlicedPlan, ViewMode};
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_tensor::Tensor;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const USAGE: &str =
    "usage: ising_sliced LX LY LZ BETA THREADS TREE TARGET MODE [REPEATS] [EXPORT] [SLICER]";

fn peak_rss_gib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / (1024.0 * 1024.0))
}

/// The bonds of an open `lx x ly x lz` cubic lattice, as pairs of site labels.
fn bonds(lx: u32, ly: u32, lz: u32) -> Vec<[u32; 2]> {
    let site = |x: u32, y: u32, z: u32| x + lx * (y + ly * z);
    let mut bonds = Vec::new();
    for z in 0..lz {
        for y in 0..ly {
            for x in 0..lx {
                if x + 1 < lx {
                    bonds.push([site(x, y, z), site(x + 1, y, z)]);
                }
                if y + 1 < ly {
                    bonds.push([site(x, y, z), site(x, y + 1, z)]);
                }
                if z + 1 < lz {
                    bonds.push([site(x, y, z), site(x, y, z + 1)]);
                }
            }
        }
    }
    bonds
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 9 {
        return Err(USAGE.into());
    }
    let (lx, ly, lz): (u32, u32, u32) = (args[1].parse()?, args[2].parse()?, args[3].parse()?);
    let beta: f64 = args[4].parse()?;
    let threads: usize = args[5].parse()?;
    let tree_source = args[6].as_str();
    let replay = args[7] == "saved";
    let target: Option<f64> = match args[7].as_str() {
        "none" | "saved" => None,
        t => Some(t.parse()?),
    };
    let mode = match args[8].as_str() {
        "offset" => ViewMode::Offset,
        "rebased" => ViewMode::Rebased,
        other => return Err(format!("mode must be offset or rebased, not {other}").into()),
    };
    let repeats: usize = args.get(9).map_or(Ok(2), |r| r.parse())?;
    let export = args.get(10).filter(|path| path.as_str() != "-");
    let fast = match (tree_source, args.get(11).map(String::as_str)) {
        ("treesa-default", _) | (_, Some("default")) => false,
        (_, None | Some("fast")) => true,
        (_, Some(other)) => {
            return Err(format!("slicer must be fast or default, not {other}").into())
        }
    };

    let bonds = bonds(lx, ly, lz);
    let network = Network {
        inputs: bonds.iter().map(|b| b.to_vec()).collect(),
        output: Vec::new(),
        sizes: (0..lx * ly * lz).map(|s| (s, 2)).collect::<HashMap<_, _>>(),
    };
    // exp(beta s s') for s, s' in {+1, -1}, column-major.
    let (same, flip) = (beta.exp(), (-beta).exp());
    let tensors: Vec<Tensor> = bonds
        .iter()
        .map(|_| Tensor::from_vec_col_major(vec![2, 2], vec![same, flip, flip, same]))
        .collect::<std::result::Result<_, _>>()?;
    let operands: Vec<&Tensor> = tensors.iter().collect();

    let started = Instant::now();
    let saved: Option<serde_json::Value> = match tree_source {
        "treesa" | "treesa-default" => None,
        path => Some(serde_json::from_str(&std::fs::read_to_string(path)?)?),
    };
    let tree: Vec<(usize, usize)> = match &saved {
        None => SlicedPlan::new(
            &network,
            &SliceOptions {
                slicing: Some(Vec::new()),
                fast,
                ..SliceOptions::default()
            },
        )?
        .pairs()
        .to_vec(),
        Some(saved) => serde_json::from_value(saved["pairs"].clone())?,
    };
    let tree_seconds = started.elapsed().as_secs_f64();
    let options = match (&saved, replay) {
        // The exported slicing on the exported tree of one slice: the same plan again.
        (Some(saved), true) => SliceOptions {
            tree: Some(serde_json::from_value(saved["sliced_pairs"].clone())?),
            slicing: Some(serde_json::from_value(saved["sliced"].clone())?),
            fast,
            ..SliceOptions::default()
        },
        (None, true) => return Err("TARGET saved needs TREE to be an export file".into()),
        (_, false) => SliceOptions {
            sc_target: target.unwrap_or(f64::INFINITY),
            tree: Some(tree.clone()),
            slicing: if target.is_some() {
                None
            } else {
                Some(Vec::new())
            },
            fast,
        },
    };
    let plan = SlicedPlan::new(&network, &options)?;
    let plan_seconds = started.elapsed().as_secs_f64() - tree_seconds;
    let report = plan.report();
    // The cost of the unsliced tree, also when a saved slicing replays its own slice tree.
    let unsliced = SlicedPlan::new(
        &network,
        &SliceOptions {
            tree: Some(tree.clone()),
            slicing: Some(Vec::new()),
            fast,
            ..SliceOptions::default()
        },
    )?
    .report()
    .unsliced;
    let prepared = plan.prepare(&operands)?;

    let mut engine = if threads > 1 {
        CpuEngine::new(Pool::builder().threads(threads).build()?)
    } else {
        CpuEngine::serial()
    };
    let mut runs = Vec::new();
    for _ in 0..repeats {
        let (out, stats) = prepared.execute(&operands, &mut engine, mode)?;
        let z = out.as_slice::<f64>()?[0];
        runs.push(serde_json::json!({
            "seconds": stats.seconds,
            "plan_misses": stats.plan_misses,
            "plan_hits": stats.plan_hits,
            "ln_z": z.ln(),
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
        "lattice": [lx, ly, lz],
        "beta": beta,
        "threads": threads,
        "tree": tree_source,
        "target": target,
        "replayed": replay,
        "mode": format!("{mode:?}"),
        "tree_seconds": tree_seconds,
        "slice_plan_seconds": plan_seconds,
        "sliced_labels": report.sliced_labels,
        "slices": report.slices,
        "per_slice": cost(&report.per_slice),
        "total_flops": report.total_flops,
        "unsliced": cost(&unsliced),
        "flop_overhead": report.total_flops / unsliced.flops,
        "runs": runs,
        "peak_rss_gib": peak_rss_gib(),
    });
    println!("{line}");
    if let Some(path) = export {
        let steps: Vec<serde_json::Value> = plan
            .steps()
            .into_iter()
            .map(|(a, b, kept)| serde_json::json!([a, b, kept]))
            .collect();
        let saved = serde_json::json!({
            "lattice": [lx, ly, lz],
            "beta": beta,
            "inputs": network.inputs,
            "pairs": tree,
            "sliced": report.sliced_labels,
            "sliced_pairs": plan.pairs(),
            "steps": steps,
        });
        std::fs::write(path, saved.to_string())?;
    }
    Ok(())
}
