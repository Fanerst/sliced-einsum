//! Shared pieces of the #2738 review benchmarks: networks, plan files, a label-excluding
//! greedy slicer, and report formatting.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};
use tenferro_einsum::{
    ContractionTree, EinsumOptimize, EinsumOptions, EinsumSubscripts, SliceOptions,
    SlicedEinsumPlan, SlicedEinsumReport, Subscripts,
};
use tenferro_tensor::{DType, Tensor};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Peak resident memory of this process (VmHWM), GiB.
pub fn peak_rss_gib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / (1024.0 * 1024.0))
}

/// The bonds of an open `lx x ly x lz` cubic lattice (same order as the prototype).
pub fn ising_bonds(lx: u32, ly: u32, lz: u32) -> Vec<[u32; 2]> {
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

/// Labeled operands, output labels and extents.
#[derive(Clone, Debug)]
pub struct Net {
    pub inputs: Vec<Vec<u32>>,
    pub output: Vec<u32>,
    pub sizes: HashMap<u32, usize>,
}

impl Net {
    /// The spin-hypergraph Ising network (closed: scalar output).
    pub fn ising(lx: u32, ly: u32, lz: u32) -> Self {
        let bonds = ising_bonds(lx, ly, lz);
        Self {
            inputs: bonds.iter().map(|b| b.to_vec()).collect(),
            output: Vec::new(),
            sizes: (0..lx * ly * lz).map(|s| (s, 2)).collect(),
        }
    }

    /// A network file with `inputs`, `output` (all extents 2) and an SSA `path`.
    pub fn from_file(path: &str) -> Result<(Self, Vec<(usize, usize)>)> {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let inputs: Vec<Vec<u32>> = serde_json::from_value(v["inputs"].clone())?;
        let output: Vec<u32> = serde_json::from_value(v["output"].clone())?;
        let pairs: Vec<(usize, usize)> = serde_json::from_value(v["path"].clone())?;
        let sizes = inputs.iter().flatten().map(|&l| (l, 2)).collect();
        Ok((
            Self {
                inputs,
                output,
                sizes,
            },
            pairs,
        ))
    }

    fn refs(&self) -> Vec<&[u32]> {
        self.inputs.iter().map(Vec::as_slice).collect()
    }

    pub fn spec(&self) -> EinsumSubscripts {
        EinsumSubscripts::new(&self.refs(), &self.output)
    }

    pub fn shapes(&self) -> Vec<Vec<usize>> {
        self.inputs
            .iter()
            .map(|labels| labels.iter().map(|l| self.sizes[l]).collect())
            .collect()
    }

    /// tenferro's tree for SSA `pairs` (operands `0..n`, the k-th pair makes `n + k`).
    pub fn tree(&self, pairs: &[(usize, usize)]) -> Result<ContractionTree> {
        let shapes = self.shapes();
        let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        Ok(ContractionTree::from_pairs(
            &Subscripts::new(&self.refs(), &self.output),
            &shape_refs,
            pairs,
        )?)
    }

    /// Zero f64 operands of the right shapes, for planning only.
    pub fn zeros(&self) -> Result<Vec<Tensor>> {
        self.shapes()
            .into_iter()
            .map(|shape| Ok(Tensor::zeros(DType::F64, shape)?))
            .collect()
    }
}

/// Read the SSA pairs `key` of a prototype plan export.
pub fn plan_pairs(plan: &Value, key: &str) -> Result<Vec<(usize, usize)>> {
    Ok(serde_json::from_value(plan[key].clone())?)
}

/// omeco's rule for adding a slice (the most frequent label on the tensors within 0.99 of
/// the largest, in log2 elements), applied to `tree`'s intermediates only (tenferro does not
/// count operands), skipping `forbidden` labels and breaking ties by the smallest label. Adds
/// labels until the largest intermediate fits `target`; `None` when no label is left.
pub fn exclusion_greedy(
    tree: &ContractionTree,
    sizes: &HashMap<u32, usize>,
    forbidden: &[u32],
    target: f64,
) -> Option<Vec<u32>> {
    let steps: Vec<Vec<u32>> = (0..tree.step_count())
        .map(|s| tree.step_subscripts(s).expect("step").2.to_vec())
        .collect();
    let mut sliced: Vec<u32> = Vec::new();
    loop {
        let log2 = |l: &u32| {
            if sliced.contains(l) {
                0.0
            } else {
                (sizes[l] as f64).log2()
            }
        };
        let scs: Vec<f64> = steps
            .iter()
            .map(|labels| labels.iter().map(log2).sum())
            .collect();
        let max = scs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max <= target + 1e-9 {
            return Some(sliced);
        }
        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        for (sc, labels) in scs.iter().zip(&steps) {
            if *sc > max - 0.99 {
                for l in labels {
                    if !sliced.contains(l) && !forbidden.contains(l) && sizes[l] > 1 {
                        *counts.entry(*l).or_default() += 1;
                    }
                }
            }
        }
        let mut best: Option<(u32, usize)> = None;
        for (&l, &c) in &counts {
            if best.map_or(true, |(_, bc)| c > bc) {
                best = Some((l, c));
            }
        }
        sliced.push(best?.0);
    }
}

/// log2 elements of the largest intermediate of `tree` with `sliced` at extent 1.
pub fn largest_log2(tree: &ContractionTree, sizes: &HashMap<u32, usize>, sliced: &[u32]) -> f64 {
    (0..tree.step_count())
        .map(|s| {
            tree.step_subscripts(s)
                .expect("step")
                .2
                .iter()
                .filter(|l| !sliced.contains(l))
                .map(|l| (sizes[l] as f64).log2())
                .sum::<f64>()
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Drop sliced labels the target does not need, checked at the true extents: try each label,
/// last chosen first, and keep it un-sliced when the largest intermediate still fits.
pub fn prune(
    tree: &ContractionTree,
    sizes: &HashMap<u32, usize>,
    labels: &[u32],
    target: f64,
) -> Vec<u32> {
    let mut kept = labels.to_vec();
    for label in labels.iter().rev() {
        let without: Vec<u32> = kept.iter().copied().filter(|l| l != label).collect();
        if largest_log2(tree, sizes, &without) <= target + 1e-9 {
            kept = without;
        }
    }
    kept
}

/// Plan with `optimize` and `slicing`, returning the report or the error's text.
pub fn plan_report(
    net: &Net,
    tensors: &[Tensor],
    optimize: EinsumOptimize,
    slicing: &SliceOptions,
) -> std::result::Result<(SlicedEinsumPlan, f64), String> {
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let started = std::time::Instant::now();
    let plan = SlicedEinsumPlan::prepare(
        &inputs,
        &net.spec(),
        &EinsumOptions::default().optimize(optimize),
        slicing,
    )
    .map_err(|e| e.to_string())?;
    Ok((plan, started.elapsed().as_secs_f64()))
}

/// A report as JSON, with flops relative to the tree before slicing.
pub fn report_json(r: &SlicedEinsumReport) -> Value {
    json!({
        "sliced_labels": r.sliced_labels,
        "slice_count": r.slice_count,
        "log2_slices": (r.slice_count as f64).log2(),
        "log2_largest": r.slice_largest_intermediate_elements.log2(),
        "log2_peak": r.slice_peak_intermediate_elements.log2(),
        "slice_flops": r.slice_flops,
        "total_flops": r.total_flops,
        "unsliced_flops": r.unsliced_flops,
        "unsliced_log2_largest": r.unsliced_largest_intermediate_elements.log2(),
        "flop_overhead": r.total_flops / r.unsliced_flops,
    })
}
