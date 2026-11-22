//! Execution: one prepared tenferro plan, run once per slice into an accumulator.

use std::ops::Range;
use std::time::Instant;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{
    ConcreteEinsumPlan, ContractionTree, EinsumOptimize, EinsumOptions, EinsumSubscripts,
    Subscripts,
};
use tenferro_tensor::{DType, Dest, Tensor, TensorRead, TensorView, TypedTensorView};

use crate::plan::SlicedPlan;
use crate::{err, Result};

/// How a slice's view of an operand addresses the operand's buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    /// The view spans the operand's whole buffer and starts at an element offset.
    Offset,
    /// The view's buffer starts at the slice's first element, so its offset is 0 for every
    /// slice.
    Rebased,
}

/// What one call to [`Prepared::execute_range`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RangeStats {
    /// Slices contracted.
    pub slices: usize,
    /// Wall-clock seconds.
    pub seconds: f64,
    /// Lookups in the CPU engine's plan cache during the range that missed.
    pub plan_misses: u64,
    /// Lookups in the CPU engine's plan cache during the range that hit.
    pub plan_hits: u64,
}

/// A [`SlicedPlan`] bound to operand dtypes and shapes, with tenferro's plan for one slice.
#[derive(Debug)]
pub struct Prepared<'p> {
    plan: &'p SlicedPlan,
    inner: ConcreteEinsumPlan,
    /// Compact column-major strides of each operand.
    strides: Vec<Vec<usize>>,
    /// For each operand and each sliced label, the operand's axis that carries the label.
    axes: Vec<Vec<Option<usize>>>,
    output_shape: Vec<usize>,
    dtype: DType,
}

impl SlicedPlan {
    /// Bind the plan to `operands`, which are host tensors in compact column-major layout, and
    /// prepare tenferro's plan for one slice with this plan's tree.
    ///
    /// # Errors
    ///
    /// Operands whose count, dtypes or shapes differ from the network, a dtype other than f32,
    /// f64, complex64 or complex128, or a tenferro planning error.
    pub fn prepare<'p>(&'p self, operands: &[&Tensor]) -> Result<Prepared<'p>> {
        let network = &self.network;
        if operands.len() != network.inputs.len() {
            return Err(err(format!(
                "{} operands for a network of {}",
                operands.len(),
                network.inputs.len()
            )));
        }
        let dtype = operands[0].dtype();
        for (k, (tensor, labels)) in operands.iter().zip(&network.inputs).enumerate() {
            if tensor.dtype() != dtype {
                return Err(err(format!("operand {k} has another dtype")));
            }
            let expected: Vec<usize> = labels.iter().map(|l| network.sizes[l]).collect();
            if tensor.shape() != expected.as_slice() {
                return Err(err(format!(
                    "operand {k} has shape {:?}, not {expected:?}",
                    tensor.shape()
                )));
            }
        }
        let strides: Vec<Vec<usize>> = operands
            .iter()
            .map(|t| column_major_strides(t.shape()))
            .collect();
        let axes: Vec<Vec<Option<usize>>> = network
            .inputs
            .iter()
            .map(|labels| {
                self.sliced
                    .iter()
                    .map(|s| labels.iter().position(|l| l == s))
                    .collect()
            })
            .collect();
        let output_shape: Vec<usize> = network.output.iter().map(|l| network.sizes[l]).collect();

        // The views of slice 0 have the shapes of every slice.
        let first = vec![0; self.sliced.len()];
        let reads = views(operands, &strides, &axes, &first, ViewMode::Offset)?;
        let shapes: Vec<Vec<usize>> = reads.iter().map(|r| r.shape().to_vec()).collect();
        let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        let input_refs: Vec<&[u32]> = self.sliced_inputs.iter().map(Vec::as_slice).collect();
        let tree = ContractionTree::from_pairs(
            &Subscripts::new(&input_refs, &network.output),
            &shape_refs,
            &self.pairs,
        )?;
        let spec = EinsumSubscripts::new(&input_refs, &network.output);
        let options = EinsumOptions::default().optimize(EinsumOptimize::Tree(tree));
        let inner = ConcreteEinsumPlan::prepare(reads.as_slice(), &spec, &options)?;
        Ok(Prepared {
            plan: self,
            inner,
            strides,
            axes,
            output_shape,
            dtype,
        })
    }
}

impl Prepared<'_> {
    /// The number of slices.
    pub fn slices(&self) -> usize {
        self.plan.report().slices
    }

    /// The shape of the result.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// A zero tensor of the result's dtype and shape, to accumulate into.
    ///
    /// # Errors
    ///
    /// A tenferro allocation error.
    pub fn zeros(&self) -> Result<Tensor> {
        Ok(Tensor::zeros(self.dtype, self.output_shape.clone())?)
    }

    /// Contract the slices with ids in `range` and add them into `out`.
    ///
    /// # Errors
    ///
    /// A range past the slice count, operands other than the prepared ones, an `out` of another
    /// dtype or shape, or a tenferro execution error.
    pub fn execute_range(
        &self,
        operands: &[&Tensor],
        range: Range<usize>,
        engine: &mut CpuEngine,
        out: &mut Tensor,
        mode: ViewMode,
    ) -> Result<RangeStats> {
        if range.start > range.end || range.end > self.slices() {
            return Err(err(format!(
                "slices {range:?} are outside 0..{}",
                self.slices()
            )));
        }
        if out.dtype() != self.dtype || out.shape() != self.output_shape.as_slice() {
            return Err(err("the accumulator has another dtype or shape"));
        }
        let before = engine.plan_cache().stats();
        let started = Instant::now();
        let mut assignment = vec![0; self.plan.extents.len()];
        for id in range.clone() {
            decode(id, &self.plan.extents, &mut assignment);
            let reads = views(operands, &self.strides, &self.axes, &assignment, mode)?;
            self.inner
                .execute_into(reads.as_slice(), engine, Dest::accumulate(&mut *out))?;
        }
        let seconds = started.elapsed().as_secs_f64();
        let after = engine.plan_cache().stats();
        Ok(RangeStats {
            slices: range.len(),
            seconds,
            plan_misses: after.misses - before.misses,
            plan_hits: after.hits - before.hits,
        })
    }

    /// Contract every slice into a new tensor.
    ///
    /// # Errors
    ///
    /// As [`Self::execute_range`].
    pub fn execute(
        &self,
        operands: &[&Tensor],
        engine: &mut CpuEngine,
        mode: ViewMode,
    ) -> Result<(Tensor, RangeStats)> {
        let mut out = self.zeros()?;
        let stats = self.execute_range(operands, 0..self.slices(), engine, &mut out, mode)?;
        Ok((out, stats))
    }
}

/// Slice id `id` as one value per sliced label, the first label fastest.
fn decode(mut id: usize, extents: &[usize], assignment: &mut [usize]) {
    for (value, &extent) in assignment.iter_mut().zip(extents) {
        *value = id % extent;
        id /= extent;
    }
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    shape
        .iter()
        .map(|&extent| {
            let this = stride;
            stride *= extent;
            this
        })
        .collect()
}

/// Each operand as tenferro reads it for one slice: whole when it carries no sliced label, and
/// otherwise a strided view without the sliced axes, at the slice's values.
fn views<'a>(
    operands: &[&'a Tensor],
    strides: &[Vec<usize>],
    axes: &[Vec<Option<usize>>],
    assignment: &[usize],
    mode: ViewMode,
) -> Result<Vec<TensorRead<'a>>> {
    let mut reads = Vec::with_capacity(operands.len());
    for (k, tensor) in operands.iter().enumerate() {
        if axes[k].iter().all(Option::is_none) {
            reads.push(TensorRead::from_tensor(tensor));
            continue;
        }
        let mut offset = 0;
        let mut sliced_axis = vec![false; tensor.shape().len()];
        for (value, axis) in assignment.iter().zip(&axes[k]) {
            if let Some(axis) = *axis {
                offset += value * strides[k][axis];
                sliced_axis[axis] = true;
            }
        }
        let shape: Vec<usize> = (0..tensor.shape().len())
            .filter(|&a| !sliced_axis[a])
            .map(|a| tensor.shape()[a])
            .collect();
        let view_strides: Vec<isize> = (0..tensor.shape().len())
            .filter(|&a| !sliced_axis[a])
            .map(|a| strides[k][a] as isize)
            .collect();
        reads.push(view(tensor, &shape, &view_strides, offset, mode)?);
    }
    Ok(reads)
}

fn view<'a>(
    tensor: &'a Tensor,
    shape: &[usize],
    strides: &[isize],
    offset: usize,
    mode: ViewMode,
) -> Result<TensorRead<'a>> {
    macro_rules! typed {
        ($scalar:ty, $variant:ident) => {{
            let data = tensor.as_slice::<$scalar>()?;
            let view = match mode {
                ViewMode::Offset => {
                    TypedTensorView::from_slice(shape, strides, offset as isize, data)?
                }
                ViewMode::Rebased => {
                    TypedTensorView::from_slice(shape, strides, 0, &data[offset..])?
                }
            };
            TensorRead::from_view(TensorView::$variant(view))
        }};
    }
    Ok(match tensor.dtype() {
        DType::F32 => typed!(f32, F32),
        DType::F64 => typed!(f64, F64),
        DType::C32 => typed!(Complex32, C32),
        DType::C64 => typed!(Complex64, C64),
        other => return Err(err(format!("dtype {other:?} is not supported"))),
    })
}
