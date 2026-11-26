//! Usage: strip_prune_probe NETWORK_JSON OPEN TARGETS REPS
//!
//! Planning only. Replays the port's output-label fallback on an explicit order (omeco's
//! slicer, fast preset with `niters = 0`, on the network without output labels and with the
//! target lowered by log2 of the output's size), counts the labels it slices, then drops the
//! labels the target does not need at the true extents and plans the rest through the port.

use omeco::{EinCode, NestedEinsum, TreeSASlicer};
use port_bench::{plan_report, prune, report_json, Net, Result};
use serde_json::json;
use tenferro_einsum::{ContractionTree, EinsumOptimize, SliceOptions};

/// The port's `to_omeco`: `tree` as omeco's nested form, keeping the labels `keep` accepts.
fn to_omeco(tree: &ContractionTree, n: usize, keep: impl Fn(&u32) -> bool) -> NestedEinsum<u32> {
    let mut nodes: Vec<Option<NestedEinsum<u32>>> =
        (0..n).map(|i| Some(NestedEinsum::leaf(i))).collect();
    for step in 0..tree.step_count() {
        let (left, right) = tree.step_pair(step).expect("step");
        let (lhs, rhs, out) = tree.step_subscripts(step).expect("step");
        let filtered = |labels: &[u32]| labels.iter().copied().filter(&keep).collect::<Vec<_>>();
        let eins = EinCode::new(vec![filtered(lhs), filtered(rhs)], filtered(out));
        let args = vec![
            nodes[left].take().expect("operand used once"),
            nodes[right].take().expect("operand used once"),
        ];
        nodes.push(Some(NestedEinsum::node(args, eins)));
    }
    nodes.pop().flatten().expect("root")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: strip_prune_probe NETWORK_JSON OPEN TARGETS REPS".into());
    }
    let (mut net, path) = Net::from_file(&args[1])?;
    if args[2] != "all" {
        net.output.truncate(args[2].parse()?);
    }
    let targets: Vec<f64> = args[3]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let reps: usize = args[4].parse()?;
    let tensors = net.zeros()?;
    let tree = net.tree(&path)?;
    let output_log2: f64 = net
        .output
        .iter()
        .map(|l| (net.sizes[l] as f64).log2())
        .sum();
    let output = net.output.clone();
    let keep = |l: &u32| !output.contains(l);
    let ixs: Vec<Vec<u32>> = net
        .inputs
        .iter()
        .map(|axes| axes.iter().copied().filter(keep).collect())
        .collect();
    for &target in &targets {
        for rep in 0..reps {
            let slicer = TreeSASlicer::fast()
                .with_sc_target(target - output_log2)
                .with_niters(0);
            let Some(sliced) = omeco::slice_code(&to_omeco(&tree, net.inputs.len(), keep), &net.sizes, &slicer, &ixs)
            else {
                println!("{}", json!({"target": target, "rep": rep, "error": "slicer returned None"}));
                continue;
            };
            let labels: Vec<u32> = sliced
                .slicing
                .iter()
                .copied()
                .filter(|l| net.sizes[l] > 1)
                .collect();
            let pruned = prune(&tree, &net.sizes, &labels, target);
            let planned = match plan_report(
                &net,
                &tensors,
                EinsumOptimize::Tree(tree.clone()),
                &SliceOptions::default()
                    .target_log2_elements(target)
                    .labels(pruned.clone()),
            ) {
                Ok((plan, _)) => json!({"ok": report_json(plan.report())}),
                Err(e) => json!({"error": e}),
            };
            println!(
                "{}",
                json!({"open": net.output.len(), "target": target, "rep": rep,
                       "stripped_target": target - output_log2,
                       "fallback_labels": labels.len(), "pruned_labels": pruned.len(),
                       "pruned_plan": planned})
            );
        }
    }
    Ok(())
}
