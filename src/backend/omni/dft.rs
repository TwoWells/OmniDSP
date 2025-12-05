use crate::core::error::Result;
use crate::traits::dft::{Dft, DftDirection, DftPlan, DftSpec};
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct OmniDft;

impl OmniDft {
    pub fn new() -> Self {
        Self
    }
}

// --- F32 Plan ---
struct OmniDftPlanF32 {
    spec: DftSpec,
    fft: Arc<dyn Fft<f32>>,
    scratch_len: usize,
}

impl DftPlan<Complex<f32>> for OmniDftPlanF32 {
    fn process(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) -> Result<()> {
        if input.len() != self.spec.length || output.len() != self.spec.length {
             return Err(crate::core::error::OmniError::BufferSizeMismatch {
                 expected: self.spec.length,
                 actual: input.len(), 
             });
        }

        // Copy input to output (RustFFT is in-place or out-of-place via copy)
        output.copy_from_slice(input);

        // Allocate scratch (Synchronous execution safety)
        let mut scratch = vec![Complex::new(0.0, 0.0); self.scratch_len];

        // Execute
        self.fft.process_with_scratch(output, &mut scratch);

        Ok(())
    }

    fn get_spec(&self) -> DftSpec {
        self.spec
    }
}

// --- F64 Plan ---
struct OmniDftPlanF64 {
    spec: DftSpec,
    fft: Arc<dyn Fft<f64>>,
    scratch_len: usize,
}

impl DftPlan<Complex<f64>> for OmniDftPlanF64 {
    fn process(&self, input: &[Complex<f64>], output: &mut [Complex<f64>]) -> Result<()> {
        if input.len() != self.spec.length || output.len() != self.spec.length {
             return Err(crate::core::error::OmniError::BufferSizeMismatch {
                 expected: self.spec.length,
                 actual: input.len(),
             });
        }
        output.copy_from_slice(input);
        let mut scratch = vec![Complex::new(0.0, 0.0); self.scratch_len];
        self.fft.process_with_scratch(output, &mut scratch);
        Ok(())
    }

    fn get_spec(&self) -> DftSpec {
        self.spec
    }
}

// --- Factory Implementation ---

impl Dft<Complex<f32>> for OmniDft {
    fn create_plan(&self, spec: DftSpec) -> Result<Box<dyn DftPlan<Complex<f32>>>> {
        let mut planner = FftPlanner::new();
        let direction = match spec.direction {
            DftDirection::Forward => rustfft::FftDirection::Forward,
            DftDirection::Inverse => rustfft::FftDirection::Inverse,
        };
        
        let fft = planner.plan_fft(spec.length, direction);
        let scratch_len = fft.get_inplace_scratch_len();

        Ok(Box::new(OmniDftPlanF32 {
            spec,
            fft,
            scratch_len,
        }))
    }
}

impl Dft<Complex<f64>> for OmniDft {
    fn create_plan(&self, spec: DftSpec) -> Result<Box<dyn DftPlan<Complex<f64>>>> {
        let mut planner = FftPlanner::new();
        let direction = match spec.direction {
            DftDirection::Forward => rustfft::FftDirection::Forward,
            DftDirection::Inverse => rustfft::FftDirection::Inverse,
        };
        
        let fft = planner.plan_fft(spec.length, direction);
        let scratch_len = fft.get_inplace_scratch_len();

        Ok(Box::new(OmniDftPlanF64 {
            spec,
            fft,
            scratch_len,
        }))
    }
}