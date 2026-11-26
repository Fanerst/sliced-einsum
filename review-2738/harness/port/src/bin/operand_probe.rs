//! Usage: operand_probe EXTENT TARGET THREADS REPEATS
//!
//! omeco's slicer counts operands in its space complexity; tenferro's target does not. On the
//! trace of four `EXTENT x EXTENT` matrices, `ab,bc,cd,da->` with the order
//! `((ab bc) cd) da`, every intermediate shrinks once `a` is sliced, but the operands
//! themselves exceed the target. Plans the port's slicer (explicit order), the same order with
//! the labels the intermediates need (omeco's add rule on intermediates only), and the port's
//! labels pruned at the true extents; runs each and prints one JSON line per plan.

use std::time::Instant;

use port_bench::{exclusion_greedy, plan_report, prune, report_json, Net, Result};
use serde_json::json;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::{EinsumOptimize, SliceOptions};
use tenferro_tensor::Tensor;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: operand_probe EXTENT TARGET THREADS REPEATS".into());
    }
    let extent: usize = args[1].parse()?;
    let target: f64 = args[2].parse()?;
    let threads: usize = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    let (a, b, c, d) = (0_u32, 1, 2, 3);
    let net = Net {
        inputs: vec![vec![a, b], vec![b, c], vec![c, d], vec![d, a]],
        output: Vec::new(),
        sizes: [(a, extent), (b, extent), (c, extent), (d, extent)].into(),
    };
    let pairs = [(0, 1), (4, 2), (5, 3)];
    let tree = net.tree(&pairs)?;
    // Deterministic data with a trace of moderate size.
    let tensors: Vec<Tensor> = (0..4)
        .map(|k| {
            let data: Vec<f64> = (0..extent * extent)
                .map(|i| (((i * 7 + k * 13) % 17) as f64 - 8.0) / (extent as f64))
                .collect();
            Tensor::from_vec_col_major(vec![extent, extent], data)
        })
        .collect::<std::result::Result<_, _>>()?;
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let mut engine = CpuEngine::new(Pool::builder().threads(threads).build()?);

    let port = plan_report(
        &net,
        &tensors,
        EinsumOptimize::Tree(tree.clone()),
        &SliceOptions::default().target_log2_elements(target),
    )?;
    let port_labels = port.0.report().sliced_labels.clone();
    let needed = exclusion_greedy(&tree, &net.sizes, &[], target).ok_or("no label left")?;
    let pruned = prune(&tree, &net.sizes, &port_labels, target);
    let mut plans = vec![("port_slicer", port.0)];
    for (name, labels) in [("intermediates_only", needed), ("port_pruned", pruned)] {
        let (plan, _) = plan_report(
            &net,
            &tensors,
            EinsumOptimize::Tree(tree.clone()),
            &SliceOptions::default()
                .target_log2_elements(target)
                .labels(labels),
        )?;
        plans.push((name, plan));
    }
    for (name, plan) in &plans {
        let mut runs = Vec::new();
        for _ in 0..repeats {
            let t0 = Instant::now();
            let out: Tensor = plan.execute(&inputs, &mut engine)?;
            runs.push(json!({"seconds": t0.elapsed().as_secs_f64(), "value": out.as_slice::<f64>()?[0]}));
        }
        println!(
            "{}",
            json!({"plan": name, "extent": extent, "target": target, "threads": threads,
                   "report": report_json(plan.report()), "runs": runs})
        );
    }
    Ok(())
}
