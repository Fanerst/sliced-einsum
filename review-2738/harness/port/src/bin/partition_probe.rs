//! Usage: partition_probe LX LY LZ PLANFILE TARGET THREADS TRIES
//!
//! Two callers that each plan the same network for the same target and split the slice ids
//! between them, as `execute_range_into`'s partitioning is meant to be used across processes.
//! Plans the Ising network (the export's unsliced tree, explicit order, slicer-chosen labels)
//! until two plannings differ in their sliced labels or their order (at most TRIES), then runs
//! slices `[0, n/2)` with plan A and `[n/2, n)` with plan B into separate accumulators, and
//! compares `ln` of their sum with plan A's full result.

use port_bench::{plan_pairs, plan_report, Net, Result};
use serde_json::json;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::{EinsumOptimize, SliceOptions};
use tenferro_tensor::{DType, Dest, Tensor};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 {
        return Err("usage: partition_probe LX LY LZ PLANFILE TARGET THREADS TRIES".into());
    }
    let (lx, ly, lz): (u32, u32, u32) = (args[1].parse()?, args[2].parse()?, args[3].parse()?);
    let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[4])?)?;
    let target: f64 = args[5].parse()?;
    let threads: usize = args[6].parse()?;
    let tries: usize = args[7].parse()?;
    let net = Net::ising(lx, ly, lz);
    // exp(beta s s' + h (s + s') / 2) with a field h, so that no slice half is Z / 2 by the
    // spin-flip symmetry of the zero-field model.
    let (beta, h) = (0.2_f64, 0.3_f64);
    let (same, flip) = (beta.exp(), (-beta).exp());
    let tensors: Vec<Tensor> = net
        .inputs
        .iter()
        .map(|_| {
            Tensor::from_vec_col_major(vec![2, 2], vec![same * h.exp(), flip, flip, same * (-h).exp()])
        })
        .collect::<std::result::Result<_, _>>()?;
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let tree = net.tree(&plan_pairs(&saved, "pairs")?)?;
    let plan = || {
        plan_report(
            &net,
            &tensors,
            EinsumOptimize::Tree(tree.clone()),
            &SliceOptions::default().target_log2_elements(target),
        )
        .map(|(p, _)| p)
    };
    let a = plan()?;
    let mut b = plan()?;
    let mut attempts = 1;
    while b.report().sliced_labels == a.report().sliced_labels && attempts < tries {
        b = plan()?;
        attempts += 1;
    }
    let mut engine = CpuEngine::new(Pool::builder().threads(threads).build()?);
    let full: Tensor = a.execute(&inputs, &mut engine)?;
    let (na, nb) = (a.slice_count(), b.slice_count());
    let mut low = Tensor::zeros(DType::F64, Vec::<usize>::new())?;
    let mut high = Tensor::zeros(DType::F64, Vec::<usize>::new())?;
    a.execute_range_into(&inputs, 0..na / 2, &mut engine, Dest::accumulate(&mut low))?;
    b.execute_range_into(&inputs, nb / 2..nb, &mut engine, Dest::accumulate(&mut high))?;
    let full = full.as_slice::<f64>()?[0];
    let split = low.as_slice::<f64>()?[0] + high.as_slice::<f64>()?[0];
    let mut set_a = a.report().sliced_labels.clone();
    let mut set_b = b.report().sliced_labels.clone();
    set_a.sort_unstable();
    set_b.sort_unstable();
    println!(
        "{}",
        json!({"lattice": [lx, ly, lz], "target": target, "plannings": attempts + 1,
               "labels_a": a.report().sliced_labels, "labels_b": b.report().sliced_labels,
               "same_set": set_a == set_b, "slices_a": na, "slices_b": nb,
               "ln_z_full": full.ln(), "ln_z_split": split.ln(),
               "relative_error": (split - full).abs() / full.abs()})
    );
    Ok(())
}
