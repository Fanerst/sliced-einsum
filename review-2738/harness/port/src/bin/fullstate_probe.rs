//! Usage: fullstate_probe NETWORK_JSON OPEN TARGETS [auto]
//!
//! Planning only. How the port's output-label workaround (rerun the slicer without output
//! labels, target lowered by log2 of the output) chooses slices on a network with a large
//! output, against omeco's add rule with the output labels excluded, on the same tree.
//!
//! - `NETWORK_JSON`: `inputs`, `output` and an SSA `path` (the 30-qubit circuit of the
//!   sampled-einsum prototype, with artensor's tree for the open network).
//! - `OPEN`: keep the first `OPEN` output labels open and sum the rest (`all` keeps every one).
//! - `TARGETS`: comma-separated log2 targets.
//! - `auto`: also plan from scratch (`EinsumOptimize::default()`) at each target.
//!
//! Prints one JSON line per target.

use port_bench::{exclusion_greedy, plan_report, prune, report_json, Net, Result};
use serde_json::{json, Value};
use tenferro_einsum::{EinsumOptimize, SliceOptions};

fn outcome(result: std::result::Result<(tenferro_einsum::SlicedEinsumPlan, f64), String>) -> Value {
    match result {
        Ok((plan, seconds)) => json!({"ok": report_json(plan.report()), "plan_seconds": seconds}),
        Err(e) => json!({"error": e}),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: fullstate_probe NETWORK_JSON OPEN TARGETS [auto]".into());
    }
    let (mut net, path) = Net::from_file(&args[1])?;
    if args[2] != "all" {
        let open: usize = args[2].parse()?;
        net.output.truncate(open);
    }
    let targets: Vec<f64> = args[3]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let auto = args.get(4).is_some_and(|a| a == "auto");
    let tensors = net.zeros()?;
    let tree = net.tree(&path)?;

    let unsliced = outcome(plan_report(
        &net,
        &tensors,
        EinsumOptimize::Tree(tree.clone()),
        &SliceOptions::default(),
    ));
    println!(
        "{}",
        json!({"open": net.output.len(), "operands": net.inputs.len(), "unsliced": unsliced})
    );

    for &target in &targets {
        let port_result = plan_report(
            &net,
            &tensors,
            EinsumOptimize::Tree(tree.clone()),
            &SliceOptions::default().target_log2_elements(target),
        );
        // The port's labels with the unneeded ones dropped at the true extents.
        let pruned = match &port_result {
            Ok((plan, _)) => outcome(plan_report(
                &net,
                &tensors,
                EinsumOptimize::Tree(tree.clone()),
                &SliceOptions::default()
                    .target_log2_elements(target)
                    .labels(prune(&tree, &net.sizes, &plan.report().sliced_labels, target)),
            )),
            Err(_) => Value::Null,
        };
        let port = outcome(port_result);
        let greedy = match exclusion_greedy(&tree, &net.sizes, &net.output, target) {
            Some(labels) => outcome(plan_report(
                &net,
                &tensors,
                EinsumOptimize::Tree(tree.clone()),
                &SliceOptions::default()
                    .target_log2_elements(target)
                    .labels(labels),
            )),
            None => json!({"error": "no label left"}),
        };
        let auto_plan = if auto {
            outcome(plan_report(
                &net,
                &tensors,
                EinsumOptimize::default(),
                &SliceOptions::default().target_log2_elements(target),
            ))
        } else {
            Value::Null
        };
        println!(
            "{}",
            json!({
                "open": net.output.len(),
                "target": target,
                "port_tree": port,
                "port_tree_pruned": pruned,
                "excluding_greedy_tree": greedy,
                "port_auto": auto_plan,
            })
        );
    }
    Ok(())
}
