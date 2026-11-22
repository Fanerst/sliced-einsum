//! The 30-qubit circuit network bundled with the sampled-einsum prototype
//! (`sampled-einsum/data/n30m14`), and the same network closed at one bitstring.

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex64;
use serde::Deserialize;
use tenferro_tensor::Tensor;

use crate::plan::Network;
use crate::{err, Result};

#[derive(Deserialize)]
struct Export {
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    path: Vec<[usize; 2]>,
    bitstrings: Vec<String>,
    reference: Vec<[f64; 2]>,
}

/// The circuit network: open at the 30 output qubits, every label of extent 2.
pub struct Circuit {
    /// Labels and extents.
    pub network: Network,
    /// The operands, complex128 in compact column-major layout.
    pub tensors: Vec<Tensor>,
    /// artensor's contraction tree for the open network, as SSA pairs.
    pub path: Vec<(usize, usize)>,
    /// Bitstrings, character `j` for output label `j`.
    pub bitstrings: Vec<String>,
    /// Reference amplitudes of the bitstrings.
    pub reference: Vec<Complex64>,
}

/// Read `network.json` and `tensors.bin` from `dir`.
///
/// # Errors
///
/// Missing or malformed files.
pub fn load(dir: &Path) -> Result<Circuit> {
    let export: Export = serde_json::from_str(&std::fs::read_to_string(dir.join("network.json"))?)?;
    let bytes = std::fs::read(dir.join("tensors.bin"))?;
    let mut sizes = HashMap::new();
    for &label in export.inputs.iter().flatten() {
        sizes.insert(label, 2);
    }
    let mut at = 0;
    let mut tensors = Vec::with_capacity(export.inputs.len());
    for labels in &export.inputs {
        let len = 1usize << labels.len();
        let end = at + 16 * len;
        let chunk = bytes
            .get(at..end)
            .ok_or_else(|| err("tensors.bin is shorter than network.json says"))?;
        let (values, _) = chunk.as_chunks::<16>();
        let data: Vec<Complex64> = values
            .iter()
            .map(|z| {
                let (re, im) = z.split_at(8);
                let re = f64::from_le_bytes(re.try_into().expect("8 bytes"));
                let im = f64::from_le_bytes(im.try_into().expect("8 bytes"));
                Complex64::new(re, im)
            })
            .collect();
        tensors.push(Tensor::from_vec_col_major(vec![2; labels.len()], data)?);
        at = end;
    }
    if at != bytes.len() {
        return Err(err("tensors.bin is longer than network.json says"));
    }
    Ok(Circuit {
        network: Network {
            inputs: export.inputs,
            output: export.output,
            sizes,
        },
        tensors,
        path: export.path.iter().map(|p| (p[0], p[1])).collect(),
        bitstrings: export.bitstrings,
        reference: export
            .reference
            .iter()
            .map(|z| Complex64::new(z[0], z[1]))
            .collect(),
    })
}

impl Circuit {
    /// The network with the output labels fixed where `pattern` has `0` or `1`; a `*` leaves that
    /// qubit open. With no `*` it is a scalar contraction whose value is the amplitude, and with
    /// `k` of them it gives the `2^k` amplitudes that agree with the pattern elsewhere. Each
    /// operand that carries a fixed label is replaced by its sub-tensor at the fixed values.
    ///
    /// # Errors
    ///
    /// A pattern whose length differs from the output's, or a character other than 0, 1 or `*`.
    pub fn fixed(&self, pattern: &str) -> Result<(Network, Vec<Tensor>)> {
        let output = &self.network.output;
        if pattern.len() != output.len() {
            return Err(err(format!(
                "a pattern of {} characters for {} qubits",
                pattern.len(),
                output.len()
            )));
        }
        let mut fixed = HashMap::new();
        let mut open = Vec::new();
        for (label, c) in output.iter().zip(pattern.chars()) {
            match c {
                '0' => drop(fixed.insert(*label, 0)),
                '1' => drop(fixed.insert(*label, 1)),
                '*' => open.push(*label),
                other => return Err(err(format!("pattern character {other:?}"))),
            }
        }
        let mut inputs = Vec::with_capacity(self.tensors.len());
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for (labels, tensor) in self.network.inputs.iter().zip(&self.tensors) {
            if labels.iter().all(|l| !fixed.contains_key(l)) {
                inputs.push(labels.clone());
                tensors.push(tensor.duplicate()?);
                continue;
            }
            let data = tensor.as_slice::<Complex64>()?;
            let kept: Vec<usize> = (0..labels.len())
                .filter(|&a| !fixed.contains_key(&labels[a]))
                .collect();
            let base: usize = (0..labels.len())
                .filter_map(|a| fixed.get(&labels[a]).map(|&v| v << a))
                .sum();
            let sub: Vec<Complex64> = (0..1usize << kept.len())
                .map(|j| {
                    let index = kept
                        .iter()
                        .enumerate()
                        .fold(base, |acc, (bit, &a)| acc | (((j >> bit) & 1) << a));
                    data[index]
                })
                .collect();
            inputs.push(kept.iter().map(|&a| labels[a]).collect());
            tensors.push(Tensor::from_vec_col_major(vec![2; kept.len()], sub)?);
        }
        Ok((
            Network {
                inputs,
                output: open,
                sizes: self.network.sizes.clone(),
            },
            tensors,
        ))
    }
}
