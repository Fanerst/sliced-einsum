//! Planning: a contraction tree, the labels to slice, and the cost of one slice.

use std::collections::{HashMap, HashSet};

use omeco::{
    nested_complexity, nested_flop, optimize_code, peak_memory, slice_code, EinCode, NestedEinsum,
    SlicedEinsum, TreeSA, TreeSASlicer,
};

use crate::{err, Result};

/// An einsum over labeled operands: each operand's labels (one per axis, none repeated), the
/// output's labels, and every label's extent.
#[derive(Clone, Debug)]
pub struct Network {
    /// Labels of each operand, in axis order.
    pub inputs: Vec<Vec<u32>>,
    /// Labels of the output, in axis order.
    pub output: Vec<u32>,
    /// Extent of every label.
    pub sizes: HashMap<u32, usize>,
}

/// How [`SlicedPlan::new`] chooses the tree and the sliced labels.
#[derive(Clone, Debug)]
pub struct SliceOptions {
    /// The memory target: log2 of the most elements that the largest intermediate of one slice
    /// may hold. Ignored when [`Self::slicing`] is given.
    pub sc_target: f64,
    /// The contraction tree as SSA pairs: operands are `0..n`, and the k-th pair makes `n + k`.
    /// `None` runs omeco's TreeSA.
    pub tree: Option<Vec<(usize, usize)>>,
    /// Slice exactly these labels, on the given or found tree, instead of running the slicer.
    pub slicing: Option<Vec<u32>>,
    /// Use omeco's `fast` presets for TreeSA and the slicer.
    pub fast: bool,
}

impl Default for SliceOptions {
    fn default() -> Self {
        Self {
            sc_target: 30.0,
            tree: None,
            slicing: None,
            fast: true,
        }
    }
}

/// The cost of one contraction tree, from omeco's counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    /// Multiply-adds.
    pub flops: f64,
    /// Elements of the largest intermediate.
    pub largest: f64,
    /// Elements of the most tensors live at once: operands, intermediates and the result.
    pub peak: f64,
}

/// What a plan does.
#[derive(Clone, Debug)]
pub struct PlanReport {
    /// The sliced labels; slice ids count through them with the first one fastest.
    pub sliced_labels: Vec<u32>,
    /// Product of the sliced labels' extents.
    pub slices: usize,
    /// The cost of one slice.
    pub per_slice: Cost,
    /// `per_slice.flops * slices`.
    pub total_flops: f64,
    /// The cost of the tree before slicing.
    pub unsliced: Cost,
}

/// A sliced contraction plan: the labels to slice, a contraction tree for one slice, and its
/// cost. [`SlicedPlan::prepare`] binds it to operands for execution.
#[derive(Clone, Debug)]
pub struct SlicedPlan {
    pub(crate) network: Network,
    pub(crate) sliced: Vec<u32>,
    pub(crate) extents: Vec<usize>,
    pub(crate) pairs: Vec<(usize, usize)>,
    pub(crate) sliced_inputs: Vec<Vec<u32>>,
    report: PlanReport,
}

impl SlicedPlan {
    /// Plan `network` under `options`.
    ///
    /// # Errors
    ///
    /// A label without an extent or repeated within an operand, a tree that is not one tree over
    /// the operands, a sliced label that is an output label or not in the network, a target the
    /// slicer cannot meet, or omeco returning no tree.
    pub fn new(network: &Network, options: &SliceOptions) -> Result<Self> {
        validate(network)?;
        let tree = match &options.tree {
            Some(pairs) => nested_from_pairs(network, pairs)?,
            None => {
                let code = EinCode::new(network.inputs.clone(), network.output.clone());
                let treesa = if options.fast {
                    TreeSA::fast()
                } else {
                    TreeSA::default()
                };
                optimize_code(&code, &network.sizes, &treesa)
                    .ok_or_else(|| err("omeco found no contraction tree"))?
            }
        };
        let unsliced = cost(&tree, &network.sizes, &network.inputs);
        let sliced = match &options.slicing {
            Some(labels) => SlicedEinsum::new(labels.clone(), tree),
            None if unsliced.largest.log2() <= options.sc_target => SlicedEinsum::unsliced(tree),
            None => {
                let slicer = if options.fast {
                    TreeSASlicer::fast()
                } else {
                    TreeSASlicer::default()
                };
                let slicer = slicer.with_sc_target(options.sc_target);
                slice_code(&tree, &network.sizes, &slicer, &network.inputs)
                    .ok_or_else(|| err("omeco's slicer returned no plan"))?
            }
        };
        let plan = Self::from_sliced(network, sliced, unsliced)?;
        let reached = plan.report.per_slice.largest.log2();
        if options.slicing.is_none() && reached > options.sc_target + 1e-9 {
            return Err(err(format!(
                "the slicer reached 2^{reached:.2} elements per slice, above the target 2^{}",
                options.sc_target
            )));
        }
        Ok(plan)
    }

    fn from_sliced(network: &Network, sliced: SlicedEinsum<u32>, unsliced: Cost) -> Result<Self> {
        let mut seen = HashSet::new();
        for label in &sliced.slicing {
            if network.output.contains(label) {
                return Err(err(format!(
                    "label {label} is an output label; v0 slices only contracted labels"
                )));
            }
            if !network.sizes.contains_key(label) {
                return Err(err(format!("sliced label {label} is not in the network")));
            }
            if !seen.insert(*label) {
                return Err(err(format!("label {label} is sliced twice")));
            }
        }
        let extents: Vec<usize> = sliced.slicing.iter().map(|l| network.sizes[l]).collect();
        let slices = extents.iter().product::<usize>();
        let mut one_slice = network.sizes.clone();
        for label in &sliced.slicing {
            one_slice.insert(*label, 1);
        }
        let per_slice = cost(&sliced.eins, &one_slice, &network.inputs);
        let pairs = ssa_pairs(&sliced.eins, network.inputs.len())?;
        let sliced_inputs = network
            .inputs
            .iter()
            .map(|labels| {
                labels
                    .iter()
                    .copied()
                    .filter(|l| !sliced.slicing.contains(l))
                    .collect()
            })
            .collect();
        let report = PlanReport {
            sliced_labels: sliced.slicing.clone(),
            slices,
            per_slice,
            total_flops: per_slice.flops * slices as f64,
            unsliced,
        };
        Ok(Self {
            network: network.clone(),
            sliced: sliced.slicing,
            extents,
            pairs,
            sliced_inputs,
            report,
        })
    }

    /// What the plan does.
    pub fn report(&self) -> &PlanReport {
        &self.report
    }

    /// The tree of one slice as SSA pairs: operands are `0..n`, the k-th pair makes `n + k`.
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// The network this plan contracts.
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// The steps of one slice, in the order of [`Self::pairs`]: the two tensors each step joins
    /// and the labels its result keeps, which are those that a later operand or the output still
    /// needs. Sliced labels are on no tensor of a slice.
    pub fn steps(&self) -> Vec<(usize, usize, Vec<u32>)> {
        let mut labels = self.sliced_inputs.clone();
        let mut holders: HashMap<u32, usize> = HashMap::new();
        for label in labels.iter().flatten().chain(&self.network.output) {
            *holders.entry(*label).or_default() += 1;
        }
        let mut steps = Vec::with_capacity(self.pairs.len());
        for (k, &(a, b)) in self.pairs.iter().enumerate() {
            let joined: Vec<u32> = labels[a].iter().chain(&labels[b]).copied().collect();
            for label in &joined {
                if let Some(count) = holders.get_mut(label) {
                    *count -= 1;
                }
            }
            let kept: Vec<u32> = if k + 1 == self.pairs.len() {
                self.network.output.clone()
            } else {
                let mut kept = Vec::new();
                for label in joined {
                    if holders.get(&label).is_some_and(|&c| c > 0) && !kept.contains(&label) {
                        kept.push(label);
                    }
                }
                kept
            };
            for label in &kept {
                *holders.entry(*label).or_default() += 1;
            }
            labels.push(kept.clone());
            steps.push((a, b, kept));
        }
        steps
    }
}

fn validate(network: &Network) -> Result<()> {
    if network.inputs.is_empty() {
        return Err(err("a network needs at least one operand"));
    }
    for (k, labels) in network.inputs.iter().enumerate() {
        let mut seen = HashSet::new();
        for label in labels {
            if !network.sizes.contains_key(label) {
                return Err(err(format!("label {label} of operand {k} has no extent")));
            }
            if !seen.insert(label) {
                return Err(err(format!("label {label} repeats within operand {k}")));
            }
        }
    }
    let present: HashSet<&u32> = network.inputs.iter().flatten().collect();
    for label in &network.output {
        if !present.contains(label) {
            return Err(err(format!("output label {label} is on no operand")));
        }
    }
    Ok(())
}

fn cost(tree: &NestedEinsum<u32>, sizes: &HashMap<u32, usize>, inputs: &[Vec<u32>]) -> Cost {
    Cost {
        flops: nested_flop(tree, sizes) as f64,
        largest: nested_complexity(tree, sizes, inputs).sc.exp2(),
        peak: peak_memory(tree, sizes, inputs) as f64,
    }
}

/// SSA pairs, operands `0..n` and the k-th pair making `n + k`, as omeco's nested form. An
/// intermediate keeps the labels that a later operand or the output still needs.
fn nested_from_pairs(network: &Network, pairs: &[(usize, usize)]) -> Result<NestedEinsum<u32>> {
    let n = network.inputs.len();
    if pairs.len() + 1 != n {
        return Err(err(format!(
            "{n} operands take {} pairs, not {}",
            n - 1,
            pairs.len()
        )));
    }
    let mut nodes: Vec<Option<(NestedEinsum<u32>, Vec<u32>)>> = network
        .inputs
        .iter()
        .enumerate()
        .map(|(k, labels)| Some((NestedEinsum::leaf(k), labels.clone())))
        .collect();
    // How many live tensors (and the output) still carry each label.
    let mut holders: HashMap<u32, usize> = HashMap::new();
    for label in network.inputs.iter().flatten().chain(&network.output) {
        *holders.entry(*label).or_default() += 1;
    }
    for (step, &(a, b)) in pairs.iter().enumerate() {
        let mut take = |id: usize| {
            nodes
                .get_mut(id)
                .and_then(Option::take)
                .ok_or_else(|| err(format!("pair {step} uses {id}, which is not live")))
        };
        let (left, left_labels) = take(a)?;
        let (right, right_labels) = take(b)?;
        for label in left_labels.iter().chain(&right_labels) {
            *holders.get_mut(label).expect("every label is counted") -= 1;
        }
        let kept: Vec<u32> = if step + 1 == pairs.len() {
            network.output.clone()
        } else {
            let mut kept = Vec::new();
            for label in left_labels.iter().chain(&right_labels) {
                if holders[label] > 0 && !kept.contains(label) {
                    kept.push(*label);
                }
            }
            kept
        };
        for label in &kept {
            *holders.get_mut(label).expect("every label is counted") += 1;
        }
        let eins = EinCode::new(vec![left_labels, right_labels], kept.clone());
        nodes.push(Some((NestedEinsum::node(vec![left, right], eins), kept)));
    }
    let root = nodes.pop().flatten();
    if nodes.iter().any(Option::is_some) {
        return Err(err("the pairs leave more than one tree"));
    }
    root.map(|(tree, _)| tree)
        .ok_or_else(|| err("the pairs do not end in a tree"))
}

/// omeco's nested form as SSA pairs. A node with more than two arguments is folded left to
/// right.
fn ssa_pairs(tree: &NestedEinsum<u32>, n: usize) -> Result<Vec<(usize, usize)>> {
    fn walk(tree: &NestedEinsum<u32>, n: usize, pairs: &mut Vec<(usize, usize)>) -> Result<usize> {
        match tree {
            NestedEinsum::Leaf { tensor_index } => {
                if *tensor_index >= n {
                    return Err(err(format!("leaf {tensor_index} is not an operand")));
                }
                Ok(*tensor_index)
            }
            NestedEinsum::Node { args, .. } => {
                let mut acc = None;
                for arg in args {
                    let id = walk(arg, n, pairs)?;
                    acc = Some(match acc {
                        None => id,
                        Some(previous) => {
                            pairs.push((previous, id));
                            n + pairs.len() - 1
                        }
                    });
                }
                acc.ok_or_else(|| err("a node without arguments"))
            }
        }
    }
    let mut pairs = Vec::with_capacity(n.saturating_sub(1));
    walk(tree, n, &mut pairs)?;
    if pairs.len() + 1 != n {
        return Err(err(format!(
            "the tree contracts {} times for {n} operands",
            pairs.len()
        )));
    }
    Ok(pairs)
}
