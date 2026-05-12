// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! [`RustDft`] — pure Rust DFT factory wrapping `RustFFT`.

use std::sync::{Arc, Mutex};

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::dft::{Dft, DftPlan};
use omnidsp_core::types::Direction;

/// Pure Rust DFT factory.
///
/// Creates [`RustDftPlan`] instances backed by `RustFFT`.  Supports
/// arbitrary lengths (not just power-of-2).
pub struct RustDft;

/// Execution plan for a DFT operation backed by `RustFFT`.
///
/// Holds the precomputed FFT plan from `RustFFT` and a scratch buffer behind
/// a `Mutex` so that `Send + Sync` is satisfied while keeping execution
/// allocation-free.
pub struct RustDftPlan<T: rustfft::FftNum> {
    fft: Arc<dyn Fft<T>>,
    length: usize,
    direction: Direction,
    scratch: Mutex<Vec<Complex<T>>>,
}

impl<T: rustfft::FftNum> DftPlan<T> for RustDftPlan<T> {
    fn process(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> Result<()> {
        if input.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: input.len(),
            });
        }
        if output.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: output.len(),
            });
        }

        output.copy_from_slice(input);

        let mut scratch = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;
        self.fft.process_with_scratch(output, &mut scratch);

        // RustFFT does not scale — apply 1/N for inverse transforms.
        if self.direction == Direction::Inverse {
            #[allow(
                clippy::cast_precision_loss,
                reason = "DFT lengths are well within f64 mantissa range"
            )]
            let scale = T::from_f64(1.0 / self.length as f64)
                .ok_or_else(|| Error::Internal("failed to convert scale factor".to_owned()))?;
            for sample in output.iter_mut() {
                *sample = Complex::new(sample.re * scale, sample.im * scale);
            }
        }

        Ok(())
    }
}

macro_rules! impl_dft {
    ($t:ty) => {
        impl Dft<$t> for RustDft {
            type Plan = RustDftPlan<$t>;

            fn create_plan(&self, length: usize, direction: Direction) -> Result<Self::Plan> {
                if length == 0 {
                    return Err(Error::InvalidSpec("DFT length must be non-zero".to_owned()));
                }

                let mut planner = FftPlanner::<$t>::new();
                let fft = match direction {
                    Direction::Forward => planner.plan_fft_forward(length),
                    Direction::Inverse => planner.plan_fft_inverse(length),
                };
                let scratch_len = fft.get_inplace_scratch_len();

                Ok(RustDftPlan {
                    fft,
                    length,
                    direction,
                    scratch: Mutex::new(vec![Complex::default(); scratch_len]),
                })
            }
        }
    };
}

impl_dft!(f32);
impl_dft!(f64);

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;

    const EPSILON_F32: f32 = 1e-5;
    const EPSILON_F64: f64 = 1e-12;

    fn assert_complex_slice_eq_f32(a: &[Complex<f32>], b: &[Complex<f32>], eps: f32) {
        assert_eq!(a.len(), b.len(), "slice lengths differ");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x.re - y.re).abs() < eps && (x.im - y.im).abs() < eps,
                "mismatch at index {i}: got ({}, {}), expected ({}, {})",
                x.re,
                x.im,
                y.re,
                y.im,
            );
        }
    }

    fn assert_complex_slice_eq_f64(a: &[Complex<f64>], b: &[Complex<f64>], eps: f64) {
        assert_eq!(a.len(), b.len(), "slice lengths differ");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x.re - y.re).abs() < eps && (x.im - y.im).abs() < eps,
                "mismatch at index {i}: got ({}, {}), expected ({}, {})",
                x.re,
                x.im,
                y.re,
                y.im,
            );
        }
    }

    #[test]
    fn zero_length_returns_error() {
        let dft = RustDft;
        let result = Dft::<f32>::create_plan(&dft, 0, Direction::Forward);
        assert!(result.is_err(), "zero length should return error");
    }

    #[test]
    fn dc_signal_f32() {
        let dft = RustDft;
        let plan = Dft::<f32>::create_plan(&dft, 4, Direction::Forward)
            .expect("plan creation should succeed");
        let input = [Complex::new(1.0_f32, 0.0); 4];
        let mut output = [Complex::default(); 4];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        let expected = [
            Complex::new(4.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        assert_complex_slice_eq_f32(&output, &expected, EPSILON_F32);
    }

    #[test]
    fn dc_signal_f64() {
        let dft = RustDft;
        let plan = Dft::<f64>::create_plan(&dft, 4, Direction::Forward)
            .expect("plan creation should succeed");
        let input = [Complex::new(1.0_f64, 0.0); 4];
        let mut output = [Complex::default(); 4];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        let expected = [
            Complex::new(4.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        assert_complex_slice_eq_f64(&output, &expected, EPSILON_F64);
    }

    #[test]
    fn round_trip_f32() {
        let dft = RustDft;
        let fwd = Dft::<f32>::create_plan(&dft, 4, Direction::Forward)
            .expect("forward plan should succeed");
        let inv = Dft::<f32>::create_plan(&dft, 4, Direction::Inverse)
            .expect("inverse plan should succeed");

        let input = [
            Complex::new(1.0_f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd.process(&input, &mut freq)
            .expect("forward should succeed");
        inv.process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f32(&recovered, &input, EPSILON_F32);
    }

    #[test]
    fn round_trip_f64() {
        let dft = RustDft;
        let fwd = Dft::<f64>::create_plan(&dft, 8, Direction::Forward)
            .expect("forward plan should succeed");
        let inv = Dft::<f64>::create_plan(&dft, 8, Direction::Inverse)
            .expect("inverse plan should succeed");

        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), -f64::from(i)))
            .collect();
        let mut freq = vec![Complex::default(); 8];
        let mut recovered = vec![Complex::default(); 8];

        fwd.process(&input, &mut freq)
            .expect("forward should succeed");
        inv.process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f64(&recovered, &input, EPSILON_F64);
    }

    #[test]
    fn non_power_of_two() {
        let dft = RustDft;
        let fwd = Dft::<f64>::create_plan(&dft, 7, Direction::Forward)
            .expect("non-power-of-2 plan should succeed");
        let inv = Dft::<f64>::create_plan(&dft, 7, Direction::Inverse)
            .expect("non-power-of-2 plan should succeed");

        let input: Vec<Complex<f64>> = (0..7).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut freq = vec![Complex::default(); 7];
        let mut recovered = vec![Complex::default(); 7];

        fwd.process(&input, &mut freq)
            .expect("forward should succeed");
        inv.process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f64(&recovered, &input, EPSILON_F64);
    }

    #[test]
    fn buffer_length_mismatch() {
        let dft = RustDft;
        let plan = Dft::<f32>::create_plan(&dft, 4, Direction::Forward)
            .expect("plan creation should succeed");

        let input = [Complex::new(1.0_f32, 0.0); 3];
        let mut output = [Complex::default(); 4];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "mismatched input length should return error"
        );

        let input = [Complex::new(1.0_f32, 0.0); 4];
        let mut output = [Complex::default(); 3];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "mismatched output length should return error"
        );
    }

    /// 4-point FFT of [1, 2, 3, 4] — verified against numpy.fft.fft.
    #[test]
    fn known_4point_f64() {
        let dft = RustDft;
        let plan = Dft::<f64>::create_plan(&dft, 4, Direction::Forward)
            .expect("plan creation should succeed");

        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output = [Complex::default(); 4];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        // numpy.fft.fft([1,2,3,4]) = [10+0j, -2+2j, -2+0j, -2-2j]
        let expected = [
            Complex::new(10.0, 0.0),
            Complex::new(-2.0, 2.0),
            Complex::new(-2.0, 0.0),
            Complex::new(-2.0, -2.0),
        ];
        assert_complex_slice_eq_f64(&output, &expected, EPSILON_F64);
    }
}
