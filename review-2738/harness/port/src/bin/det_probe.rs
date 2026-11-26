//! Usage: det_probe NET ORDER TARGET K
//!
//! Planning only. Plans the same network, order and target `K` times in this process and
//! prints what the port's slicer chose each time, then how many distinct plans came out.
//!
//! - `NET`: `ising:LX:LY:LZ:PLANFILE` (the export's unsliced tree) or `file:NETWORK_JSON`.
//! - `ORDER`: `tree` (the given tree, explicit order: the slicer only picks labels), `auto`
//!   (`EinsumOptimize::default()`, then the slicer reshapes the tree) or `autosa` (`Auto` with
//!   omeco's TreeSA defaults, so the slicer starts from an annealed tree).

use std::collections::BTreeSet;

use port_bench::{plan_pairs, plan_report, Net, Result};
use serde_json::json;
use tenferro_einsum::{ContractionOptimizerOptions, EinsumOptimize, SliceOptions};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: det_probe NET ORDER TARGET K".into());
    }
    let (net, pairs) = if let Some(rest) = args[1].strip_prefix("ising:") {
        let parts: Vec<&str> = rest.splitn(4, ':').collect();
        let net = Net::ising(parts[0].parse()?, parts[1].parse()?, parts[2].parse()?);
        let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(parts[3])?)?;
        (net, plan_pairs(&saved, "pairs")?)
    } else if let Some(file) = args[1].strip_prefix("file:") {
        Net::from_file(file)?
    } else {
        return Err("NET must be ising:LX:LY:LZ:PLANFILE or file:NETWORK_JSON".into());
    };
    let target: f64 = args[3].parse()?;
    let k: usize = args[4].parse()?;
    let tensors = net.zeros()?;
    let tree = net.tree(&pairs)?;

    let mut ordered = BTreeSet::new();
    let mut sets = BTreeSet::new();
    let mut flops = Vec::new();
    for i in 0..k {
        let optimize = match args[2].as_str() {
            "tree" => EinsumOptimize::Tree(tree.clone()),
            "auto" => EinsumOptimize::default(),
            // `Auto` with omeco's TreeSA defaults (the prototype's `treesa-default` tree), so
            // the slicer starts from an annealed tree and may re-anneal it.
            "autosa" => {
                let mut sa = ContractionOptimizerOptions::default();
                sa.betas = (0..300).map(|i| 0.01 + 0.05 * f64::from(i)).collect();
                sa.ntrials = 10;
                sa.niters = 50;
                sa.score = omeco::ScoreFunction::default();
                EinsumOptimize::Auto(sa)
            }
            other => return Err(format!("ORDER must be tree, auto or autosa, not {other}").into()),
        };
        let slicing = SliceOptions::default().target_log2_elements(target);
        match plan_report(&net, &tensors, optimize, &slicing) {
            Ok((plan, seconds)) => {
                let r = plan.report();
                let mut set = r.sliced_labels.clone();
                set.sort_unstable();
                ordered.insert(r.sliced_labels.clone());
                sets.insert(set);
                flops.push(r.total_flops / r.unsliced_flops);
                println!(
                    "{}",
                    json!({"i": i, "labels": r.sliced_labels, "slices": r.slice_count,
                           "flop_overhead": r.total_flops / r.unsliced_flops,
                           "total_flops": r.total_flops, "plan_seconds": seconds})
                );
            }
            Err(e) => println!("{}", json!({"i": i, "error": e})),
        }
    }
    let min = flops.iter().copied().fold(f64::INFINITY, f64::min);
    let max = flops.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "{}",
        json!({"summary": true, "net": args[1], "order": args[2], "target": target, "plans": k,
               "distinct_label_sets": sets.len(), "distinct_label_orders": ordered.len(),
               "flop_overhead_min": min, "flop_overhead_max": max})
    );
    Ok(())
}
