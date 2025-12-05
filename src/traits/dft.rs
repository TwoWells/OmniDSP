use crate::core::error::Result;
use num_complex::Complex;

/// Specification for a DFT operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DftSpec {
    pub length: usize,
    pub direction: DftDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DftDirection {
    Forward,
    Inverse,
}

/// The Worker: Executes the DFT.
/// T is the element type (e.g., Complex<f32>).
pub trait DftPlan<T>: Send + Sync {
    /// Process the input and write to output.
    /// Input and Output must be the same length as specified in the Spec.
    fn process(&self, input: &[T], output: &mut [T]) -> Result<()>;
}

/// The Factory: Creates Plans.
/// T is the element type this factory supports.
pub trait Dft<T>: Send + Sync {
    fn create_plan(&self, spec: DftSpec) -> Result<Box<dyn DftPlan<T>>>;
}
