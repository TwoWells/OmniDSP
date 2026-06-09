// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! [`RustDftC2c`] — pure Rust DFT factory wrapping `RustFFT`.

use std::sync::{Arc, Mutex};

use num_complex::Complex;
use rustfft::{Fft, FftNum, FftPlanner};

use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec, DftNorm};
use omnidsp_core::types::{Direction, DspFloat};

// ─── Type erasure ────────────────────────────────────────────────────

/// Private trait that hides `rustfft::FftNum` from the public API.
///
/// `RustDftC2cPlan<T>` stores `Box<dyn DftC2cEngine<T>>` instead of
/// `Arc<dyn Fft<T>>`, so the `FftNum` bound stays on the impl block
/// (inside `create_plan`) and doesn't leak into the struct definition
/// or the dispatch layer.
trait DftC2cEngine<T>: Send + Sync {
    fn process(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);
}

impl<T: FftNum> DftC2cEngine<T> for Arc<dyn Fft<T>> {
    fn process(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        Fft::process_with_scratch(&**self, buffer, scratch);
    }
}

// ─── Public types ────────────────────────────────────────────────────

/// Pure Rust DFT factory.
///
/// Creates [`RustDftC2cPlan`] instances backed by `RustFFT`.  Supports
/// arbitrary lengths (not just power-of-2).
#[derive(Debug, Clone, Copy)]
pub struct RustDftC2c;

/// Execution plan for a DFT operation backed by `RustFFT`.
///
/// Holds the precomputed FFT plan and a scratch buffer behind a `Mutex`
/// so that `Send + Sync` is satisfied while keeping execution
/// allocation-free.
pub struct RustDftC2cPlan<T> {
    engine: Box<dyn DftC2cEngine<T>>,
    length: usize,
    direction: Direction,
    norm: DftNorm,
    scratch: Mutex<Vec<Complex<T>>>,
}

// ─── Trait implementations ───────────────────────────────────────────

impl<T: DspFloat> DftC2cPlan<T> for RustDftC2cPlan<T> {
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
        self.engine.process(output, &mut scratch);

        // RustFFT is unnormalized — apply scaling based on the norm convention.
        let scale = self.compute_scale()?;
        if let Some(s) = scale {
            for sample in output.iter_mut() {
                *sample = Complex::new(sample.re * s, sample.im * s);
            }
        }

        Ok(())
    }
}

impl<T: DspFloat> RustDftC2cPlan<T> {
    /// Compute the scaling factor for the configured normalization, or `None`
    /// if no scaling is needed.
    fn compute_scale(&self) -> Result<Option<T>> {
        norm_scale::<T>(self.norm, self.direction, self.length)
    }
}

/// Scaling factor a [`DftNorm`] convention applies to a transform of real
/// length `n` running in `direction`, or `None` when no scaling is needed.
///
/// Shared by all three primitives so that the forward-only [`RustDftR2c`] and
/// inverse-only [`RustDftC2r`] apply exactly the c2c per-direction factors (see
/// [`DftNorm`]): r2c scales like a c2c **forward** plan, c2r like a c2c
/// **inverse** plan.
///
/// [`RustDftR2c`]: crate::RustDftR2c
/// [`RustDftC2r`]: crate::RustDftC2r
pub fn norm_scale<T: DspFloat>(norm: DftNorm, direction: Direction, n: usize) -> Result<Option<T>> {
    #[allow(
        clippy::cast_precision_loss,
        reason = "DFT lengths are well within f64 mantissa range"
    )]
    let nf = n as f64;

    let factor = match norm {
        DftNorm::None => return Ok(None),
        DftNorm::Inverse => match direction {
            Direction::Inverse => 1.0 / nf,
            Direction::Forward => return Ok(None),
        },
        DftNorm::Ortho => 1.0 / nf.sqrt(),
    };

    let scale = T::from_f64(factor)
        .ok_or_else(|| Error::Internal("failed to convert scale factor".to_owned()))?;
    Ok(Some(scale))
}

macro_rules! impl_dft {
    ($t:ty) => {
        impl DftC2c<$t> for RustDftC2c {
            type Plan = RustDftC2cPlan<$t>;

            fn create_plan(&self, spec: &DftC2cSpec<$t>) -> Result<Self::Plan> {
                let length = spec.length();

                let mut planner = FftPlanner::<$t>::new();
                let fft = match spec.direction() {
                    Direction::Forward => planner.plan_fft_forward(length),
                    Direction::Inverse => planner.plan_fft_inverse(length),
                };
                let scratch_len = fft.get_inplace_scratch_len();

                Ok(RustDftC2cPlan {
                    engine: Box::new(fft),
                    length,
                    direction: spec.direction(),
                    norm: spec.norm(),
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

    fn fwd<T>(length: usize) -> DftC2cSpec<T> {
        DftC2cSpec::new(length, Direction::Forward, DftNorm::Inverse).expect("valid forward spec")
    }

    fn inv<T>(length: usize) -> DftC2cSpec<T> {
        DftC2cSpec::new(length, Direction::Inverse, DftNorm::Inverse).expect("valid inverse spec")
    }

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
    fn zero_length_spec_rejected() {
        let result = DftC2cSpec::<f32>::new(0, Direction::Forward, DftNorm::None);
        assert!(
            result.is_err(),
            "zero length should be rejected by the spec constructor"
        );
    }

    #[test]
    fn dc_signal_f32() {
        let dft = RustDftC2c;
        let plan = DftC2c::<f32>::create_plan(&dft, &fwd(4)).expect("plan creation should succeed");
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
        let dft = RustDftC2c;
        let plan = DftC2c::<f64>::create_plan(&dft, &fwd(4)).expect("plan creation should succeed");
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
        let dft = RustDftC2c;
        let fwd_plan =
            DftC2c::<f32>::create_plan(&dft, &fwd(4)).expect("forward plan should succeed");
        let inv_plan =
            DftC2c::<f32>::create_plan(&dft, &inv(4)).expect("inverse plan should succeed");

        let input = [
            Complex::new(1.0_f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");
        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f32(&recovered, &input, EPSILON_F32);
    }

    #[test]
    fn round_trip_f64() {
        let dft = RustDftC2c;
        let fwd_plan =
            DftC2c::<f64>::create_plan(&dft, &fwd(8)).expect("forward plan should succeed");
        let inv_plan =
            DftC2c::<f64>::create_plan(&dft, &inv(8)).expect("inverse plan should succeed");

        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), -f64::from(i)))
            .collect();
        let mut freq = vec![Complex::default(); 8];
        let mut recovered = vec![Complex::default(); 8];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");
        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f64(&recovered, &input, EPSILON_F64);
    }

    #[test]
    fn non_power_of_two() {
        let dft = RustDftC2c;
        let fwd_plan =
            DftC2c::<f64>::create_plan(&dft, &fwd(7)).expect("non-power-of-2 plan should succeed");
        let inv_plan =
            DftC2c::<f64>::create_plan(&dft, &inv(7)).expect("non-power-of-2 plan should succeed");

        let input: Vec<Complex<f64>> = (0..7).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut freq = vec![Complex::default(); 7];
        let mut recovered = vec![Complex::default(); 7];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");
        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f64(&recovered, &input, EPSILON_F64);
    }

    #[test]
    fn buffer_length_mismatch() {
        let dft = RustDftC2c;
        let plan = DftC2c::<f32>::create_plan(&dft, &fwd(4)).expect("plan creation should succeed");

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
        let dft = RustDftC2c;
        // Use unnormalized forward to match numpy.fft.fft (which is unnormalized forward).
        let spec = DftC2cSpec::new(4, Direction::Forward, DftNorm::None).expect("valid spec");
        let plan = DftC2c::<f64>::create_plan(&dft, &spec).expect("plan creation should succeed");

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

    /// Ortho normalization: round-trip should be identity, and forward
    /// output should be scaled by 1/√N compared to unnormalized.
    #[test]
    fn ortho_round_trip() {
        let dft = RustDftC2c;
        let fwd_spec = DftC2cSpec::new(4, Direction::Forward, DftNorm::Ortho).expect("valid spec");
        let inv_spec = DftC2cSpec::new(4, Direction::Inverse, DftNorm::Ortho).expect("valid spec");
        let fwd_plan = DftC2c::<f64>::create_plan(&dft, &fwd_spec).expect("forward ortho plan");
        let inv_plan = DftC2c::<f64>::create_plan(&dft, &inv_spec).expect("inverse ortho plan");

        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");

        // DC bin should be sum/√N = 10/2 = 5
        assert!(
            (freq[0].re - 5.0).abs() < EPSILON_F64,
            "ortho DC should be 10/√4 = 5.0, got {}",
            freq[0].re
        );

        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f64(&recovered, &input, EPSILON_F64);
    }

    /// Unnormalized: round-trip scales by N.
    #[test]
    fn unnormalized_round_trip_scales_by_n() {
        let dft = RustDftC2c;
        let n = 4;
        let fwd_spec = DftC2cSpec::new(n, Direction::Forward, DftNorm::None).expect("valid spec");
        let inv_spec = DftC2cSpec::new(n, Direction::Inverse, DftNorm::None).expect("valid spec");
        let fwd_plan = DftC2c::<f64>::create_plan(&dft, &fwd_spec).expect("forward plan");
        let inv_plan = DftC2c::<f64>::create_plan(&dft, &inv_spec).expect("inverse plan");

        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");
        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        // Round-trip should be input * N
        #[allow(clippy::cast_precision_loss, reason = "test constant")]
        let nf = n as f64;
        let expected: Vec<Complex<f64>> = input
            .iter()
            .map(|c| Complex::new(c.re * nf, c.im * nf))
            .collect();
        assert_complex_slice_eq_f64(&recovered, &expected, EPSILON_F64);
    }
}
