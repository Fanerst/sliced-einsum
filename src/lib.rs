//! Prototype of a sliced einsum on tenferro, for
//! [tensor4all/tenferro-rs#2676](https://github.com/tensor4all/tenferro-rs/issues/2676).
//!
//! It follows the three steps of the maintainers' plan on that issue, on tenferro's public API:
//!
//! 1. **The memory target is a constraint.** [`SlicedPlan::new`] takes a contraction tree (omeco's
//!    TreeSA, or one the caller gives) and runs omeco's `slice_code` with the target, which picks
//!    labels to slice until the largest intermediate of one slice fits. The [`PlanReport`] gives
//!    the sliced labels, the slice count, the flops and peak elements of one slice, and the total
//!    against the unsliced tree. A target the slicer cannot meet is an error before any execution.
//! 2. **A slice loop.** [`Prepared`] holds one tenferro `ConcreteEinsumPlan` for the sliced
//!    network, built with the plan's tree as `EinsumOptimize::Tree`. For each assignment of the
//!    sliced labels it takes strided views of the operands that carry them, with no copies, and
//!    runs that plan into the destination with `Dest::accumulate`.
//! 3. **Slice ranges.** [`Prepared::execute_range`] runs any range of slice ids into a
//!    caller-owned accumulator, so disjoint ranges can go to different workers and be summed.
//!
//! As in the issue's v0, only contracted labels are sliced: an output label among the sliced
//! labels is an error.
//!
//! # Examples
//!
//! ```
//! use std::collections::HashMap;
//! use sliced_einsum::{Network, SliceOptions, SlicedPlan, ViewMode};
//! use tenferro_cpu::CpuEngine;
//! use tenferro_tensor::Tensor;
//!
//! // ij,jk,ki-> (the trace of a product of three 3x3 matrices), with j sliced.
//! let network = Network {
//!     inputs: vec![vec![0, 1], vec![1, 2], vec![2, 0]],
//!     output: vec![],
//!     sizes: HashMap::from([(0, 3), (1, 3), (2, 3)]),
//! };
//! let options = SliceOptions { slicing: Some(vec![1]), ..SliceOptions::default() };
//! let plan = SlicedPlan::new(&network, &options)?;
//! assert_eq!(plan.report().slices, 3);
//!
//! let a = Tensor::from_vec_col_major(vec![3, 3], (0..9).map(f64::from).collect())?;
//! let prepared = plan.prepare(&[&a, &a, &a])?;
//! let (trace, _) = prepared.execute(&[&a, &a, &a], &mut CpuEngine::serial(), ViewMode::Offset)?;
//! // trace(A^3) for A = [[0, 3, 6], [1, 4, 7], [2, 5, 8]] (column-major 0..9).
//! assert_eq!(trace.as_slice::<f64>()?, &[2376.0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod exec;
pub mod network;
mod plan;

pub use exec::{Prepared, RangeStats, ViewMode};
pub use plan::{Cost, Network, PlanReport, SliceOptions, SlicedPlan};

/// The prototype's error type: a boxed error with a message.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) fn err(message: impl Into<String>) -> Box<dyn std::error::Error> {
    message.into().into()
}
