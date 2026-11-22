//! Sliced contractions of random networks against tenferro's unsliced einsum.

use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sliced_einsum::{Network, SliceOptions, SlicedPlan, ViewMode};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumOps, EinsumSubscripts};
use tenferro_tensor::Tensor;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A random network: some labels on three or more operands (hyperedges), extents 1 to `max_extent`,
/// and an output of one or two labels unless `closed`.
fn random_network(
    rng: &mut StdRng,
    labels: u32,
    operands: usize,
    max_extent: usize,
    closed: bool,
) -> Network {
    let sizes: HashMap<u32, usize> = (0..labels)
        .map(|l| (l, rng.gen_range(1..=max_extent)))
        .collect();
    let all: Vec<u32> = (0..labels).collect();
    let mut inputs: Vec<Vec<u32>> = (0..operands)
        .map(|_| {
            let rank = rng.gen_range(1..=3.min(labels as usize));
            all.choose_multiple(rng, rank).copied().collect()
        })
        .collect();
    for l in 0..labels {
        if !inputs.iter().any(|ls| ls.contains(&l)) {
            let k = rng.gen_range(0..operands);
            inputs[k].push(l);
        }
    }
    let output = if closed {
        Vec::new()
    } else {
        let count = rng.gen_range(1..=2);
        all.choose_multiple(rng, count).copied().collect()
    };
    Network {
        inputs,
        output,
        sizes,
    }
}

fn shape(network: &Network, labels: &[u32]) -> Vec<usize> {
    labels.iter().map(|l| network.sizes[l]).collect()
}

fn random_f64(rng: &mut StdRng, network: &Network) -> Result<Vec<Tensor>> {
    network
        .inputs
        .iter()
        .map(|labels| {
            let shape = shape(network, labels);
            let len = shape.iter().product();
            let data: Vec<f64> = (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Ok(Tensor::from_vec_col_major(shape, data)?)
        })
        .collect()
}

fn random_c64(rng: &mut StdRng, network: &Network) -> Result<Vec<Tensor>> {
    network
        .inputs
        .iter()
        .map(|labels| {
            let shape = shape(network, labels);
            let len = shape.iter().product();
            let data: Vec<Complex64> = (0..len)
                .map(|_| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
                .collect();
            Ok(Tensor::from_vec_col_major(shape, data)?)
        })
        .collect()
}

/// tenferro's own einsum of the whole network, with its default planner.
fn einsum(network: &Network, operands: &[&Tensor]) -> Result<Tensor> {
    let inputs: Vec<&[u32]> = network.inputs.iter().map(Vec::as_slice).collect();
    let spec = EinsumSubscripts::new(&inputs, &network.output);
    Ok(operands.einsum(&spec, &mut CpuEngine::serial())?)
}

fn assert_close_f64(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (got, want) = (got.as_slice::<f64>()?, want.as_slice::<f64>()?);
    let scale = want.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12 * scale,
            "{what}: element {k} is {g}, not {w}"
        );
    }
    Ok(())
}

fn sliced(
    network: &Network,
    operands: &[&Tensor],
    options: &SliceOptions,
    mode: ViewMode,
) -> Result<(Tensor, SlicedPlan)> {
    let plan = SlicedPlan::new(network, options)?;
    let prepared = plan.prepare(operands)?;
    let (out, stats) = prepared.execute(operands, &mut CpuEngine::serial(), mode)?;
    assert_eq!(stats.slices, plan.report().slices);
    drop(prepared);
    Ok((out, plan))
}

/// The labels that are contracted rather than kept in the output.
fn contracted(network: &Network) -> Vec<u32> {
    let mut labels: Vec<u32> = network
        .inputs
        .iter()
        .flatten()
        .copied()
        .filter(|l| !network.output.contains(l))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    labels.sort_unstable();
    labels
}

#[test]
fn explicit_slicing_matches_the_unsliced_einsum() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(2676);
    let mut checked = 0;
    for case in 0..60 {
        let network = random_network(&mut rng, 7, 4, 3, case % 2 == 0);
        let tensors = random_f64(&mut rng, &network)?;
        let operands: Vec<&Tensor> = tensors.iter().collect();
        let want = einsum(&network, &operands)?;

        // The plan's own tree without slicing agrees with tenferro's planner.
        let unsliced = SliceOptions {
            slicing: Some(Vec::new()),
            ..SliceOptions::default()
        };
        let (got, plan) = sliced(&network, &operands, &unsliced, ViewMode::Offset)?;
        assert_close_f64(&got, &want, &format!("case {case}, unsliced"))?;
        let tree = plan.pairs().to_vec();

        let candidates = contracted(&network);
        for count in 1..=candidates.len().min(3) {
            let labels: Vec<u32> = candidates
                .choose_multiple(&mut rng, count)
                .copied()
                .collect();
            for mode in [ViewMode::Offset, ViewMode::Rebased] {
                let options = SliceOptions {
                    tree: Some(tree.clone()),
                    slicing: Some(labels.clone()),
                    ..SliceOptions::default()
                };
                let (got, plan) = sliced(&network, &operands, &options, mode)?;
                let extents: usize = labels.iter().map(|l| network.sizes[l]).product();
                assert_eq!(plan.report().slices, extents);
                assert_close_f64(&got, &want, &format!("case {case}, {labels:?}, {mode:?}"))?;
                checked += 1;
            }
        }
    }
    assert!(checked > 100, "only {checked} sliced plans were checked");
    Ok(())
}

#[test]
fn every_contracted_label_sliced() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(7);
    for case in 0..20 {
        let network = random_network(&mut rng, 6, 4, 2, case % 2 == 1);
        let tensors = random_f64(&mut rng, &network)?;
        let operands: Vec<&Tensor> = tensors.iter().collect();
        let want = einsum(&network, &operands)?;
        let options = SliceOptions {
            slicing: Some(contracted(&network)),
            ..SliceOptions::default()
        };
        let (got, _) = sliced(&network, &operands, &options, ViewMode::Offset)?;
        assert_close_f64(&got, &want, &format!("case {case}"))?;
    }
    Ok(())
}

#[test]
fn a_sliced_label_of_extent_one() -> Result<()> {
    // ij,jk,kl->il with j of extent 1.
    let network = Network {
        inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        output: vec![0, 3],
        sizes: HashMap::from([(0, 3), (1, 1), (2, 4), (3, 2)]),
    };
    let mut rng = StdRng::seed_from_u64(1);
    let tensors = random_f64(&mut rng, &network)?;
    let operands: Vec<&Tensor> = tensors.iter().collect();
    let options = SliceOptions {
        slicing: Some(vec![1]),
        ..SliceOptions::default()
    };
    let (got, plan) = sliced(&network, &operands, &options, ViewMode::Rebased)?;
    assert_eq!(plan.report().slices, 1);
    assert_close_f64(&got, &einsum(&network, &operands)?, "extent-1 label")
}

#[test]
fn an_output_label_cannot_be_sliced() {
    let network = Network {
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        sizes: HashMap::from([(0, 2), (1, 2), (2, 2)]),
    };
    let options = SliceOptions {
        slicing: Some(vec![2]),
        ..SliceOptions::default()
    };
    let error = SlicedPlan::new(&network, &options).expect_err("output label sliced");
    assert!(error.to_string().contains("output label"), "{error}");
}

/// A closed network shaped like a random graph: `edges` labels of extent 2, each on two distinct
/// operands, so that each of the `vertices` operands carries a few bonds, as in a circuit or
/// lattice network.
fn random_graph_network(rng: &mut StdRng, vertices: usize, edges: u32) -> Network {
    let mut inputs = vec![Vec::new(); vertices];
    for label in 0..edges {
        let a = rng.gen_range(0..vertices);
        let mut b = rng.gen_range(0..vertices - 1);
        if b >= a {
            b += 1;
        }
        inputs[a].push(label);
        inputs[b].push(label);
    }
    for (k, labels) in inputs.iter_mut().enumerate() {
        if labels.is_empty() {
            // A bond to the next operand keeps every operand in the network.
            let label = edges + k as u32;
            labels.push(label);
        }
    }
    let mut sizes: HashMap<u32, usize> = (0..edges).map(|l| (l, 2)).collect();
    let mut extra: Vec<u32> = inputs
        .iter()
        .flatten()
        .copied()
        .filter(|l| *l >= edges)
        .collect();
    extra.sort_unstable();
    for (k, label) in extra.iter().enumerate() {
        sizes.insert(*label, 2);
        // Put each extra label on a second operand too, so it is contracted.
        let other = (k + 1) % vertices;
        if !inputs[other].contains(label) {
            inputs[other].push(*label);
        }
    }
    Network {
        inputs,
        output: Vec::new(),
        sizes,
    }
}

#[test]
fn the_slicer_meets_a_memory_target() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(30);
    let mut met = 0;
    for case in 0..12 {
        let network = random_graph_network(&mut rng, 20, 32);
        let tensors = random_f64(&mut rng, &network)?;
        let operands: Vec<&Tensor> = tensors.iter().collect();
        let free = SlicedPlan::new(
            &network,
            &SliceOptions {
                slicing: Some(Vec::new()),
                ..SliceOptions::default()
            },
        )?;
        let largest = free.report().unsliced.largest.log2();
        assert!(
            largest >= 4.0,
            "case {case}: unsliced 2^{largest}, too small to slice"
        );
        let target = (largest - 2.0).floor();
        let options = SliceOptions {
            sc_target: target,
            tree: Some(free.pairs().to_vec()),
            ..SliceOptions::default()
        };
        match SlicedPlan::new(&network, &options) {
            Err(e) if e.to_string().contains("the slicer reached") => {
                eprintln!("case {case}: unsliced 2^{largest:.2}, target 2^{target}: {e}");
                continue;
            }
            other => drop(other?),
        }
        let (got, plan) = sliced(&network, &operands, &options, ViewMode::Offset)?;
        let report = plan.report();
        assert!(
            report.per_slice.largest.log2() <= target + 1e-9,
            "case {case}: {report:?}"
        );
        assert!(report.slices > 1, "case {case}: nothing was sliced");
        assert_close_f64(&got, &einsum(&network, &operands)?, &format!("case {case}"))?;
        met += 1;
    }
    assert!(
        met >= 10,
        "the slicer met a target in only {met} of 12 cases"
    );
    Ok(())
}

#[test]
fn disjoint_ranges_sum_to_the_whole() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(99);
    let network = Network {
        inputs: vec![vec![0, 1, 2], vec![1, 2, 3], vec![3, 4], vec![4, 0, 5]],
        output: vec![5],
        sizes: HashMap::from([(0, 3), (1, 2), (2, 3), (3, 2), (4, 3), (5, 2)]),
    };
    let tensors = random_f64(&mut rng, &network)?;
    let operands: Vec<&Tensor> = tensors.iter().collect();
    let options = SliceOptions {
        slicing: Some(vec![0, 1, 4]),
        ..SliceOptions::default()
    };
    let plan = SlicedPlan::new(&network, &options)?;
    assert_eq!(plan.report().slices, 18);
    let prepared = plan.prepare(&operands)?;
    let mut engine = CpuEngine::serial();
    let mut total = vec![0.0; 2];
    for range in [0..5, 5..6, 6..13, 13..18] {
        let mut part = prepared.zeros()?;
        let stats = prepared.execute_range(
            &operands,
            range.clone(),
            &mut engine,
            &mut part,
            ViewMode::Rebased,
        )?;
        assert_eq!(stats.slices, range.len());
        for (t, p) in total.iter_mut().zip(part.as_slice::<f64>()?) {
            *t += p;
        }
    }
    let summed = Tensor::from_vec_col_major(vec![2], total)?;
    assert_close_f64(&summed, &einsum(&network, &operands)?, "ranges")?;
    let error = prepared
        .execute_range(
            &operands,
            17..19,
            &mut engine,
            &mut prepared.zeros()?,
            ViewMode::Offset,
        )
        .expect_err("range past the slices");
    assert!(error.to_string().contains("outside"), "{error}");
    Ok(())
}

#[test]
fn complex128_operands() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(128);
    for case in 0..10 {
        let network = random_network(&mut rng, 7, 4, 3, case % 2 == 0);
        let tensors = random_c64(&mut rng, &network)?;
        let operands: Vec<&Tensor> = tensors.iter().collect();
        let want = einsum(&network, &operands)?;
        let candidates = contracted(&network);
        let labels: Vec<u32> = candidates.choose_multiple(&mut rng, 2).copied().collect();
        let options = SliceOptions {
            slicing: Some(labels),
            ..SliceOptions::default()
        };
        let (got, _) = sliced(&network, &operands, &options, ViewMode::Offset)?;
        let (got, want) = (got.as_slice::<Complex64>()?, want.as_slice::<Complex64>()?);
        let scale = want.iter().fold(1.0_f64, |m, z| m.max(z.norm()));
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).norm() <= 1e-12 * scale,
                "case {case}: {g} against {w}"
            );
        }
    }
    Ok(())
}
