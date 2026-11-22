//! Usage: ising_steps EXPORT THREADS [TOP]
//!
//! Runs slice 0 of an exported cubic Ising plan (`ising_sliced ... EXPORT`) one step at a time,
//! each step a binary tenferro einsum on the step's two tensors, and prints one JSON line: the
//! total, and the `TOP` (default 12) slowest steps with their labels, sizes and flops. The same
//! steps in PyTorch: `scripts/steps_torch.py`.

use std::collections::HashSet;
use std::time::Instant;

use serde::Deserialize;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::{EinsumOps, EinsumSubscripts};
use tenferro_tensor::Tensor;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Export {
    beta: f64,
    inputs: Vec<Vec<u32>>,
    sliced: Vec<u32>,
    steps: Vec<(usize, usize, Vec<u32>)>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: ising_steps EXPORT THREADS [TOP]".into());
    }
    let export: Export = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let threads: usize = args[2].parse()?;
    let top: usize = args.get(3).map_or(Ok(12), |t| t.parse())?;
    let mut engine = if threads > 1 {
        CpuEngine::new(Pool::builder().threads(threads).build()?)
    } else {
        CpuEngine::serial()
    };

    // Slice 0: every sliced spin at its first value.
    let sliced: HashSet<u32> = export.sliced.iter().copied().collect();
    let (same, flip) = (export.beta.exp(), (-export.beta).exp());
    let bond = [same, flip, flip, same];
    let mut tensors: Vec<Option<Tensor>> = Vec::new();
    let mut labels: Vec<Vec<u32>> = Vec::new();
    for leaf in &export.inputs {
        let kept: Vec<u32> = leaf
            .iter()
            .copied()
            .filter(|l| !sliced.contains(l))
            .collect();
        let data: Vec<f64> = match (sliced.contains(&leaf[0]), sliced.contains(&leaf[1])) {
            (false, false) => bond.to_vec(),
            (true, false) => vec![bond[0], bond[2]],
            (false, true) => vec![bond[0], bond[1]],
            (true, true) => vec![bond[0]],
        };
        tensors.push(Some(Tensor::from_vec_col_major(vec![2; kept.len()], data)?));
        labels.push(kept);
    }

    let mut records = Vec::new();
    let started = Instant::now();
    for (k, (a, b, kept)) in export.steps.iter().enumerate() {
        let left = tensors[*a].take().ok_or("a step reuses a tensor")?;
        let right = tensors[*b].take().ok_or("a step reuses a tensor")?;
        let spec = EinsumSubscripts::new(&[&labels[*a], &labels[*b]], kept);
        let t0 = Instant::now();
        let out = [&left, &right].einsum(&spec, &mut engine)?;
        let seconds = t0.elapsed().as_secs_f64();
        let union: HashSet<u32> = labels[*a].iter().chain(&labels[*b]).copied().collect();
        let shared = labels[*a].iter().filter(|l| labels[*b].contains(l)).count();
        let batch = kept
            .iter()
            .filter(|l| labels[*a].contains(l) && labels[*b].contains(l))
            .count();
        records.push(serde_json::json!({
            "step": k,
            "left_rank": labels[*a].len(),
            "right_rank": labels[*b].len(),
            "out_rank": kept.len(),
            "shared": shared,
            "batch": batch,
            "log2_flops": union.len(),
            "seconds": seconds,
        }));
        tensors.push(Some(out));
        labels.push(kept.clone());
    }
    let total = started.elapsed().as_secs_f64();
    records.sort_by(|x, y| {
        y["seconds"]
            .as_f64()
            .partial_cmp(&x["seconds"].as_f64())
            .expect("finite")
    });
    let steps = records.len();
    records.truncate(top);
    println!(
        "{}",
        serde_json::json!({ "engine": "tenferro", "threads": threads, "steps": steps, "seconds": total, "slowest": records })
    );
    Ok(())
}
