// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! DCT module — Discrete Cosine Transform via FFT.
//!
//! [`OmniDct`] implements the [`Dct`] trait generically over any [`DftC2c`]
//! and [`VecOps`] implementation.  Supports DCT type II and type III with
//! optional orthonormal normalization.
//!
//! The algorithm uses symmetric extension to convert the real DCT into a
//! complex FFT of twice the length, then extracts the result via twiddle
//! factor multiplication.
//!
//! Internal scratch buffers are behind a [`Mutex`] so that the plan
//! satisfies `Send + Sync` while taking `&self`.

use std::f64::consts::PI;
use std::fmt;
use std::ops::{AddAssign, MulAssign};
use std::sync::Mutex;

use num_complex::Complex;
use num_traits::Float;

use crate::error::{Error, Result};
use crate::traits::dct::{Dct, DctNorm, DctPlan, DctSpec, DctType};
use crate::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec, DftNorm};
use crate::traits::vecops::VecOps;
use crate::types::Direction;

// ─── Public types ──────────────────────────────────────────────────────

/// Generic DCT factory backed by [`DftC2c`] and [`VecOps`].
///
/// Creates [`OmniDctPlan`]s for specific lengths and DCT types.  The factory
/// owns the DFT factory and `VecOps` instance; plans own their sub-plans.
#[derive(Debug, Clone)]
pub struct OmniDct<D, V> {
    dft: D,
    vecops: V,
}

impl<D, V> OmniDct<D, V> {
    /// Create a new DCT factory.
    #[must_use]
    pub const fn new(dft: D, vecops: V) -> Self {
        Self { dft, vecops }
    }
}

/// Execution plan for a DCT operation.
///
/// Created by [`OmniDct::create_plan`](Dct::create_plan).  Immutable and
/// thread-safe (`Send + Sync`).
///
/// **Memory:** allocates `2 × 2N` complex values for scratch buffers
/// (behind a [`Mutex`]) plus `N` complex values for precomputed twiddle
/// factors.
pub struct OmniDctPlan<T, P, V> {
    dft_plan: P,
    twiddles: Vec<Complex<T>>,
    vecops: V,
    scratch: Mutex<DctScratch<T>>,
    length: usize,
    dct_type: DctType,
    norm: DctNorm,
}

impl<T, P, V> fmt::Debug for OmniDctPlan<T, P, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniDctPlan")
            .field("length", &self.length)
            .field("dct_type", &self.dct_type)
            .field("norm", &self.norm)
            .finish_non_exhaustive()
    }
}

// ─── Plan internals ───────────────────────────────────────────────────

/// Scratch buffers for the DCT pipeline.
#[allow(
    clippy::struct_field_names,
    reason = "buf_ prefix clarifies these are reusable buffers"
)]
struct DctScratch<T> {
    /// FFT/IFFT input buffer (length 2N).
    buf_in: Vec<Complex<T>>,
    /// FFT/IFFT output buffer (length 2N).
    buf_out: Vec<Complex<T>>,
    /// Twiddle multiply result (length N, DCT-II only).
    buf_tw: Vec<Complex<T>>,
}

// ─── Trait implementations ─────────────────────────────────────────────

#[allow(
    clippy::cast_precision_loss,
    reason = "DCT lengths are small enough that usize→f64 is exact"
)]
impl<T, P, V> DctPlan<T> for OmniDctPlan<T, P, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    P: DftC2cPlan<T>,
    V: VecOps<T>,
{
    fn process(&self, input: &[T], output: &mut [T]) -> Result<()> {
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

        match self.dct_type {
            DctType::II => self.process_dct2(input, output),
            DctType::III => self.process_dct3(input, output),
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "DCT lengths are small enough that usize→f64 is exact"
)]
impl<T, P, V> OmniDctPlan<T, P, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    P: DftC2cPlan<T>,
    V: VecOps<T>,
{
    /// DCT-II via symmetric extension → forward FFT → twiddle multiply → extract real.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire DCT pipeline"
    )]
    fn process_dct2(&self, input: &[T], output: &mut [T]) -> Result<()> {
        let n = self.length;
        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;

        let DctScratch {
            buf_in,
            buf_out,
            buf_tw,
        } = &mut *guard;

        // 1. Symmetric extension: y[i] = x[i], y[2N-1-i] = x[i]
        for (i, &x) in input.iter().enumerate() {
            buf_in[i] = Complex::new(x, T::zero());
            buf_in[2 * n - 1 - i] = Complex::new(x, T::zero());
        }

        // 2. Forward FFT (2N-point, no normalization)
        self.dft_plan.process(buf_in, buf_out)?;

        // 3. Twiddle multiply: buf_tw[k] = Y[k] * twiddle[k], then extract real parts.
        self.vecops.cmul(&buf_out[..n], &self.twiddles, buf_tw)?;
        self.vecops.extract_real(buf_tw, output)?;

        // 4. Orthonormal scaling
        if self.norm == DctNorm::Ortho {
            let n_f = T::from(n).unwrap_or_else(T::zero);
            // output[0] *= sqrt(1/(4N))
            output[0] *= T::one() / (T::from(2.0).unwrap_or_else(T::zero) * n_f.sqrt());
            // output[k>0] *= sqrt(1/(2N))
            let scale = T::one() / (T::from(2.0).unwrap_or_else(T::zero) * n_f).sqrt();
            self.vecops.scale(&mut output[1..], scale);
        }

        Ok(())
    }

    /// DCT-III via spectrum construction → inverse FFT → extract real.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire DCT pipeline"
    )]
    fn process_dct3(&self, input: &[T], output: &mut [T]) -> Result<()> {
        let n = self.length;
        let two_n = 2 * n;
        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;

        let DctScratch {
            buf_in, buf_out, ..
        } = &mut *guard;

        // For ortho normalization, pre-scale the input
        // input'[0] = input[0] * 2*sqrt(N), input'[k>0] = input[k] * sqrt(2N)
        let n_f = T::from(n).unwrap_or_else(T::zero);
        let (scale_0, scale_k) = if self.norm == DctNorm::Ortho {
            (
                T::from(2.0).unwrap_or_else(T::zero) * n_f.sqrt(),
                (T::from(2.0).unwrap_or_else(T::zero) * n_f).sqrt(),
            )
        } else {
            (T::one(), T::one())
        };

        // 1. Build spectrum: Z[k] = input[k] * conj(twiddle[k])
        //    twiddle[k] = exp(-jπk/(2N)), so conj(twiddle[k]) = exp(+jπk/(2N))
        for (k, &tw) in self.twiddles.iter().enumerate() {
            let scale = if k == 0 { scale_0 } else { scale_k };
            let x = input[k] * scale;
            buf_in[k] = Complex::new(x * tw.re, x * (-tw.im));
        }

        // Z[N] = 0
        buf_in[n] = Complex::new(T::zero(), T::zero());

        // Hermitian symmetry: Z[2N-k] = conj(Z[k]) for k=1..N-1
        for k in 1..n {
            buf_in[two_n - k] = buf_in[k].conj();
        }

        // 2. Inverse FFT (2N-point, no normalization)
        self.dft_plan.process(buf_in, buf_out)?;

        // 3. Extract real parts
        self.vecops.extract_real(&buf_out[..n], output)?;

        // 4. Ortho: divide by 2N
        if self.norm == DctNorm::Ortho {
            let inv_two_n = T::one() / (T::from(2.0).unwrap_or_else(T::zero) * n_f);
            self.vecops.scale(output, inv_two_n);
        }

        Ok(())
    }
}

// ─── Factory implementation ───────────────────────────────────────────

#[allow(
    clippy::cast_precision_loss,
    reason = "DCT lengths are small enough that usize→f64 is exact"
)]
impl<T, D, V> Dct<T> for OmniDct<D, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    D: DftC2c<T>,
    V: VecOps<T>,
{
    type Plan = OmniDctPlan<T, D::Plan, V>;

    fn create_plan(&self, spec: &DctSpec<T>) -> Result<Self::Plan> {
        if spec.length == 0 {
            return Err(Error::InvalidSpec("DCT length must be non-zero".to_owned()));
        }

        let n = spec.length;
        let two_n = 2 * n;

        // Create DFT sub-plan: forward for DCT-II, inverse for DCT-III.
        let direction = match spec.dct_type {
            DctType::II => Direction::Forward,
            DctType::III => Direction::Inverse,
        };
        let dft_spec = DftC2cSpec::new(two_n, direction, DftNorm::None)?;
        let dft_plan = self.dft.create_plan(&dft_spec)?;

        // Precompute twiddle factors: exp(-jπk/(2N)) for k=0..N-1
        let twiddles: Vec<Complex<T>> = (0..n)
            .map(|k| {
                let angle = -PI * k as f64 / two_n as f64;
                Complex::new(
                    T::from(angle.cos()).unwrap_or_else(T::zero),
                    T::from(angle.sin()).unwrap_or_else(T::zero),
                )
            })
            .collect();

        // Allocate scratch buffers
        let zero = Complex::new(T::zero(), T::zero());
        let scratch = DctScratch {
            buf_in: vec![zero; two_n],
            buf_out: vec![zero; two_n],
            buf_tw: vec![zero; n],
        };

        Ok(OmniDctPlan {
            dft_plan,
            twiddles,
            vecops: self.vecops.clone(),
            scratch: Mutex::new(scratch),
            length: n,
            dct_type: spec.dct_type,
            norm: spec.norm,
        })
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
#[allow(
    clippy::cast_precision_loss,
    reason = "test sizes are small enough that usize→f64 is exact"
)]
mod tests {
    use super::*;
    use crate::test_utils::{TestDftC2c, TestVecOps};

    include!(testdata!("dct_scipy.rs"));

    const EPSILON: f64 = 1e-10;

    fn assert_approx_eq(actual: &[f64], expected: &[f64], eps: f64) {
        assert_eq!(actual.len(), expected.len(), "slice lengths differ");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < eps,
                "mismatch at index {i}: got {a}, expected {e} (diff = {})",
                (a - e).abs()
            );
        }
    }

    fn make_factory() -> OmniDct<TestDftC2c, TestVecOps> {
        OmniDct::new(TestDftC2c, TestVecOps)
    }

    // ── Validation tests ─────────────────────────────────────────────

    #[test]
    fn rejects_zero_length() {
        let factory = make_factory();
        let spec = DctSpec::<f64>::new(0, DctType::II, DctNorm::None);
        assert!(
            factory.create_plan(&spec).is_err(),
            "zero length should be rejected"
        );
    }

    #[test]
    fn rejects_input_length_mismatch() {
        let factory = make_factory();
        let spec = DctSpec::<f64>::new(4, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [1.0, 2.0, 3.0]; // length 3, expects 4
        let mut output = [0.0; 4];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "mismatched input length should error"
        );
    }

    #[test]
    fn rejects_output_length_mismatch() {
        let factory = make_factory();
        let spec = DctSpec::<f64>::new(4, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 3]; // length 3, expects 4
        assert!(
            plan.process(&input, &mut output).is_err(),
            "mismatched output length should error"
        );
    }

    // ── DCT-II analytical tests ──────────────────────────────────────

    #[test]
    fn dct2_length_1() {
        let factory = make_factory();
        let spec = DctSpec::<f64>::new(1, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [3.0];
        let mut output = [0.0];
        plan.process(&input, &mut output)
            .expect("process should succeed");
        // DCT-II of [c] = 2*c*cos(0) = 2*c
        assert_approx_eq(&output, &[6.0], EPSILON);
    }

    #[test]
    fn dct2_dc_input() {
        // DCT-II of constant signal concentrates energy at bin 0
        let factory = make_factory();
        let n = 8;
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [5.0; 8];
        let mut output = [0.0; 8];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        // Bin 0 = 2 * N * c = 2 * 8 * 5 = 80
        assert!(
            (output[0] - 80.0).abs() < EPSILON,
            "DC bin should be 2*N*c, got {}",
            output[0]
        );
        // All other bins should be zero
        for (k, &v) in output.iter().enumerate().skip(1) {
            assert!(
                v.abs() < EPSILON,
                "bin {k} should be zero for DC input, got {v}"
            );
        }
    }

    #[test]
    fn dct2_impulse() {
        // DCT-II of [1, 0, 0, 0]: X[k] = 2 * cos(πk/(2N))
        let factory = make_factory();
        let spec = DctSpec::<f64>::new(4, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [1.0, 0.0, 0.0, 0.0];
        let mut output = [0.0; 4];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        let expected: Vec<f64> = (0..4)
            .map(|k| 2.0 * (PI * f64::from(k) / 8.0).cos())
            .collect();
        assert_approx_eq(&output, &expected, EPSILON);
    }

    // ── DCT-III analytical tests ─────────────────────────────────────

    #[test]
    fn dct3_length_1() {
        let factory = make_factory();
        let spec = DctSpec::<f64>::new(1, DctType::III, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [3.0];
        let mut output = [0.0];
        plan.process(&input, &mut output)
            .expect("process should succeed");
        // DCT-III of [c] = c
        assert_approx_eq(&output, &[3.0], EPSILON);
    }

    // ── Round-trip tests ─────────────────────────────────────────────

    #[test]
    fn round_trip_unnormalized() {
        // DCT-III(DCT-II(x)) = 2N * x
        let factory = make_factory();
        let n = 8;
        let input = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];

        let spec_ii = DctSpec::<f64>::new(n, DctType::II, DctNorm::None);
        let spec_iii = DctSpec::<f64>::new(n, DctType::III, DctNorm::None);
        let plan_ii = factory
            .create_plan(&spec_ii)
            .expect("DCT-II plan creation should succeed");
        let plan_iii = factory
            .create_plan(&spec_iii)
            .expect("DCT-III plan creation should succeed");

        let mut freq = [0.0; 8];
        plan_ii
            .process(&input, &mut freq)
            .expect("DCT-II should succeed");

        let mut recovered = [0.0; 8];
        plan_iii
            .process(&freq, &mut recovered)
            .expect("DCT-III should succeed");

        // recovered = 2N * input
        let expected: Vec<f64> = input.iter().map(|&x| x * 2.0 * n as f64).collect();
        assert_approx_eq(&recovered, &expected, 1e-8);
    }

    #[test]
    fn round_trip_ortho() {
        // DCT-III_ortho(DCT-II_ortho(x)) = x
        let factory = make_factory();
        let n = 16;
        let input = DCT_INPUT_SINE16;

        let spec_ii = DctSpec::<f64>::new(n, DctType::II, DctNorm::Ortho);
        let spec_iii = DctSpec::<f64>::new(n, DctType::III, DctNorm::Ortho);
        let plan_ii = factory
            .create_plan(&spec_ii)
            .expect("DCT-II ortho plan creation should succeed");
        let plan_iii = factory
            .create_plan(&spec_iii)
            .expect("DCT-III ortho plan creation should succeed");

        let mut freq = vec![0.0; n];
        plan_ii
            .process(input, &mut freq)
            .expect("DCT-II ortho should succeed");

        let mut recovered = vec![0.0; n];
        plan_iii
            .process(&freq, &mut recovered)
            .expect("DCT-III ortho should succeed");

        assert_approx_eq(&recovered, input, 1e-10);
    }

    #[test]
    fn round_trip_ortho_non_power_of_two() {
        // N=17 verifies the 2N-point FFT handles odd lengths
        let factory = make_factory();
        let n = 17;
        let input = DCT_INPUT_COS17;

        let spec_ii = DctSpec::<f64>::new(n, DctType::II, DctNorm::Ortho);
        let spec_iii = DctSpec::<f64>::new(n, DctType::III, DctNorm::Ortho);
        let plan_ii = factory
            .create_plan(&spec_ii)
            .expect("plan creation should succeed");
        let plan_iii = factory
            .create_plan(&spec_iii)
            .expect("plan creation should succeed");

        let mut freq = vec![0.0; n];
        plan_ii
            .process(input, &mut freq)
            .expect("DCT-II should succeed");

        let mut recovered = vec![0.0; n];
        plan_iii
            .process(&freq, &mut recovered)
            .expect("DCT-III should succeed");

        assert_approx_eq(&recovered, input, 1e-9);
    }

    // ── Scipy reference tests ────────────────────────────────────────

    #[test]
    fn dct2_scipy_ramp8() {
        let factory = make_factory();
        let n = DCT_INPUT_RAMP8.len();
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_RAMP8, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_RAMP8, 1e-10);
    }

    #[test]
    fn dct2_ortho_scipy_ramp8() {
        let factory = make_factory();
        let n = DCT_INPUT_RAMP8.len();
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::Ortho);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_RAMP8, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_ORTHO_RAMP8, 1e-10);
    }

    #[test]
    fn dct3_scipy_ramp8() {
        let factory = make_factory();
        let n = DCT_INPUT_RAMP8.len();
        let spec = DctSpec::<f64>::new(n, DctType::III, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_RAMP8, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_RAMP8, 1e-10);
    }

    #[test]
    fn dct3_ortho_scipy_ramp8() {
        let factory = make_factory();
        let n = DCT_INPUT_RAMP8.len();
        let spec = DctSpec::<f64>::new(n, DctType::III, DctNorm::Ortho);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_RAMP8, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_ORTHO_RAMP8, 1e-10);
    }

    #[test]
    fn dct2_scipy_sine16() {
        let factory = make_factory();
        let n = DCT_INPUT_SINE16.len();
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_SINE16, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_SINE16, 1e-10);
    }

    #[test]
    fn dct2_ortho_scipy_sine16() {
        let factory = make_factory();
        let n = DCT_INPUT_SINE16.len();
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::Ortho);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_SINE16, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_ORTHO_SINE16, 1e-10);
    }

    #[test]
    fn dct3_scipy_sine16() {
        let factory = make_factory();
        let n = DCT_INPUT_SINE16.len();
        let spec = DctSpec::<f64>::new(n, DctType::III, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_SINE16, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_SINE16, 1e-10);
    }

    #[test]
    fn dct3_ortho_scipy_sine16() {
        let factory = make_factory();
        let n = DCT_INPUT_SINE16.len();
        let spec = DctSpec::<f64>::new(n, DctType::III, DctNorm::Ortho);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_SINE16, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_ORTHO_SINE16, 1e-10);
    }

    #[test]
    fn dct2_scipy_cos17_non_power_of_two() {
        let factory = make_factory();
        let n = DCT_INPUT_COS17.len();
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_COS17, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_COS17, 1e-9);
    }

    #[test]
    fn dct2_ortho_scipy_cos17() {
        let factory = make_factory();
        let n = DCT_INPUT_COS17.len();
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::Ortho);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_COS17, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_ORTHO_COS17, 1e-9);
    }

    #[test]
    fn dct3_scipy_cos17_non_power_of_two() {
        let factory = make_factory();
        let n = DCT_INPUT_COS17.len();
        let spec = DctSpec::<f64>::new(n, DctType::III, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_COS17, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_COS17, 1e-9);
    }

    #[test]
    fn dct3_ortho_scipy_cos17() {
        let factory = make_factory();
        let n = DCT_INPUT_COS17.len();
        let spec = DctSpec::<f64>::new(n, DctType::III, DctNorm::Ortho);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_COS17, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_ORTHO_COS17, 1e-9);
    }

    #[test]
    fn dct2_scipy_random64() {
        let factory = make_factory();
        let n = DCT_INPUT_RANDOM64.len();
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_RANDOM64, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_RANDOM64, 1e-8);
    }

    #[test]
    fn dct2_ortho_scipy_random64() {
        let factory = make_factory();
        let n = DCT_INPUT_RANDOM64.len();
        let spec = DctSpec::<f64>::new(n, DctType::II, DctNorm::Ortho);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_RANDOM64, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_ORTHO_RANDOM64, 1e-8);
    }

    #[test]
    fn dct3_scipy_random64() {
        let factory = make_factory();
        let n = DCT_INPUT_RANDOM64.len();
        let spec = DctSpec::<f64>::new(n, DctType::III, DctNorm::None);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_RANDOM64, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_RANDOM64, 1e-8);
    }

    #[test]
    fn dct3_ortho_scipy_random64() {
        let factory = make_factory();
        let n = DCT_INPUT_RANDOM64.len();
        let spec = DctSpec::<f64>::new(n, DctType::III, DctNorm::Ortho);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.process(DCT_INPUT_RANDOM64, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_ORTHO_RANDOM64, 1e-8);
    }
}
