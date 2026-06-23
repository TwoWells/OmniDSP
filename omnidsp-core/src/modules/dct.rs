// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! DCT module — Discrete Cosine Transform via the real-DFT primitives.
//!
//! [`OmniDct`] builds DCT plans via an inherent `create_plan`, generic over any
//! [`DftR2c`] / [`DftC2r`] and [`VecOps`] implementation.  Supports DCT type II
//! and type III with optional orthonormal normalization.
//!
//! The algorithm uses symmetric extension to convert the real DCT into a
//! transform of twice the length.  DCT-II runs the real `2N`-point
//! symmetric extension through a forward [`DftR2c`] (taking the first `N` of the
//! `N + 1` half-spectrum bins) and a twiddle multiply; DCT-III builds the
//! `N + 1`-bin Hermitian half-spectrum and runs it through a [`HermitianC2r`]-
//! shaped inverse [`DftC2r`].
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
use crate::hermitian::{HermitianC2r, HermitianC2rPlan};
use crate::traits::dct::{DctNorm, DctPlan, DctSpec, DctType};
use crate::traits::dft::{DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan, DftR2cSpec};
use crate::traits::vecops::VecOps;
use crate::types::DspFloat;

// ─── Public types ──────────────────────────────────────────────────────

/// Generic DCT factory backed by [`DftR2c`] / [`DftC2r`] and [`VecOps`].
///
/// Creates [`OmniDctPlan`]s for specific lengths and DCT types.  The factory
/// owns the real-DFT factories (`r2c` for DCT-II forward, `c2r` for DCT-III
/// inverse) and the `VecOps` instance; plans own their sub-plans.  The c2r
/// factory is Hermitian-shaped internally.
#[derive(Debug, Clone)]
pub struct OmniDct<R, C, V> {
    r2c: R,
    c2r: C,
    vecops: V,
}

impl<R, C, V> OmniDct<R, C, V> {
    /// Create a new DCT factory.
    #[must_use]
    pub const fn new(r2c: R, c2r: C, vecops: V) -> Self {
        Self { r2c, c2r, vecops }
    }
}

/// Execution plan for a DCT operation.
///
/// Created by [`OmniDct::create_plan`].  Immutable and thread-safe
/// (`Send + Sync`).
///
/// **Memory:** allocates a `2N` real buffer plus an `N + 1` complex
/// half-spectrum buffer (behind a [`Mutex`]), an `N` complex twiddle-product
/// buffer (DCT-II), and `N` complex precomputed twiddle factors.
///
/// `RP` is the DCT-II forward [`DftR2cPlan`]; `CP` is the DCT-III inverse
/// [`DftC2rPlan`] (in practice a [`HermitianC2rPlan`]).  Only one is populated
/// per plan; neither is ever named by users.
pub struct OmniDctPlan<T, RP, CP, V> {
    transform: DctTransform<RP, CP>,
    twiddles: Vec<Complex<T>>,
    vecops: V,
    scratch: Mutex<DctScratch<T>>,
    length: usize,
    dct_type: DctType,
    norm: DctNorm,
}

impl<T, RP, CP, V> fmt::Debug for OmniDctPlan<T, RP, CP, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniDctPlan")
            .field("length", &self.length)
            .field("dct_type", &self.dct_type)
            .field("norm", &self.norm)
            .finish_non_exhaustive()
    }
}

// ─── Plan internals ───────────────────────────────────────────────────

/// The transform sub-plan a DCT plan holds: a forward [`DftR2cPlan`] for
/// DCT-II, or a Hermitian-shaped inverse [`DftC2rPlan`] for DCT-III.  Exactly
/// one is constructed per [`OmniDctPlan`], chosen by [`DctType`].
enum DctTransform<RP, CP> {
    /// DCT-II: forward real-to-complex transform of the symmetric extension.
    Forward(RP),
    /// DCT-III: inverse complex-to-real transform of the built half-spectrum.
    Inverse(CP),
}

/// Scratch buffers for the DCT pipeline.
struct DctScratch<T> {
    /// `2N` real buffer: symmetric extension (DCT-II) / c2r output (DCT-III).
    real: Vec<T>,
    /// `N + 1` complex half-spectrum: r2c output (DCT-II) / c2r input (DCT-III).
    half: Vec<Complex<T>>,
    /// `N` complex twiddle-multiply result (DCT-II only).
    twiddle_prod: Vec<Complex<T>>,
}

// ─── Trait implementations ─────────────────────────────────────────────

#[allow(
    clippy::cast_precision_loss,
    reason = "DCT lengths are small enough that usize→f64 is exact"
)]
impl<T, RP, CP, V> DctPlan<T> for OmniDctPlan<T, RP, CP, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    RP: DftR2cPlan<T>,
    CP: DftC2rPlan<T>,
    V: VecOps<T>,
{
    fn execute(&self, input: &[T], output: &mut [T]) -> Result<()> {
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
impl<T, RP, CP, V> OmniDctPlan<T, RP, CP, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    RP: DftR2cPlan<T>,
    CP: DftC2rPlan<T>,
    V: VecOps<T>,
{
    /// DCT-II via symmetric extension → forward r2c → twiddle multiply → extract real.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire DCT pipeline"
    )]
    fn process_dct2(&self, input: &[T], output: &mut [T]) -> Result<()> {
        let n = self.length;
        let two_n = 2 * n;
        let DctTransform::Forward(fwd) = &self.transform else {
            return Err(Error::Internal(
                "DCT-II plan is missing its forward transform".to_owned(),
            ));
        };
        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;

        let DctScratch {
            real,
            half,
            twiddle_prod,
        } = &mut *guard;

        // 1. Symmetric extension: y[i] = x[i], y[2N-1-i] = x[i]
        for (i, &x) in input.iter().enumerate() {
            real[i] = x;
            real[two_n - 1 - i] = x;
        }

        // 2. Forward r2c (2N real → N+1 half-spectrum; real is consumed)
        fwd.execute(real, half)?;

        // 3. Twiddle multiply the first N bins, then extract real parts.
        self.vecops.cmul(&half[..n], &self.twiddles, twiddle_prod)?;
        self.vecops.extract_real(twiddle_prod, output)?;

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

    /// DCT-III via half-spectrum construction → inverse c2r → extract real.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire DCT pipeline"
    )]
    fn process_dct3(&self, input: &[T], output: &mut [T]) -> Result<()> {
        let n = self.length;
        let DctTransform::Inverse(inv) = &self.transform else {
            return Err(Error::Internal(
                "DCT-III plan is missing its inverse transform".to_owned(),
            ));
        };

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

        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;

        let DctScratch { real, half, .. } = &mut *guard;

        // 1. Build the half-spectrum: Z[k] = input[k] * conj(twiddle[k])
        //    twiddle[k] = exp(-jπk/(2N)), so conj(twiddle[k]) = exp(+jπk/(2N)).
        //    The upper half is the c2r's job (Z is Hermitian by construction).
        for (k, &tw) in self.twiddles.iter().enumerate() {
            let scale = if k == 0 { scale_0 } else { scale_k };
            let x = input[k] * scale;
            half[k] = Complex::new(x * tw.re, x * (-tw.im));
        }

        // Z[N] = 0 (Nyquist of the 2N-point real signal).
        half[n] = Complex::new(T::zero(), T::zero());

        // 2. Inverse c2r (N+1 half-spectrum → 2N real; half is consumed).
        inv.execute(half, real)?;

        // 3. Extract the first N real samples.
        output.copy_from_slice(&real[..n]);

        // 4. Ortho: divide by 2N
        if self.norm == DctNorm::Ortho {
            let inv_two_n = T::one() / (T::from(2.0).unwrap_or_else(T::zero) * n_f);
            self.vecops.scale(output, inv_two_n);
        }

        Ok(())
    }
}

// ─── Plan construction ────────────────────────────────────────────────

#[allow(
    clippy::cast_precision_loss,
    reason = "DCT lengths are small enough that usize→f64 is exact"
)]
impl<R, C, V> OmniDct<R, C, V> {
    /// Create a plan for a DCT described by `spec`.
    ///
    /// DCT-II runs a forward [`DftR2c`]; DCT-III runs a [`HermitianC2r`]-shaped
    /// inverse [`DftC2r`].
    ///
    /// # Errors
    ///
    /// Returns an error if DFT plan creation fails.  The length invariant is
    /// enforced by [`DctSpec::new`], so it is not re-checked here.
    #[allow(
        clippy::type_complexity,
        reason = "composite real-DFT plan type (r2c forward + Hermitian-shaped c2r \
                  inverse); the dispatch layer names it via `type Plan` aliases"
    )]
    pub fn create_plan<T>(
        &self,
        spec: &DctSpec,
    ) -> Result<OmniDctPlan<T, <R as DftR2c<T>>::Plan, HermitianC2rPlan<<C as DftC2r<T>>::Plan>, V>>
    where
        T: DspFloat + AddAssign + MulAssign,
        R: DftR2c<T>,
        C: DftC2r<T> + Clone,
        V: VecOps<T>,
    {
        let n = spec.length();
        let two_n = 2 * n;

        // Create the real-DFT sub-plan over the 2N-point real signal: a forward
        // r2c for DCT-II, a Hermitian-shaped inverse c2r for DCT-III.  Both use
        // DftNorm::None — the ortho factors are applied explicitly below.
        let transform = match spec.dct_type() {
            DctType::II => {
                let fwd_spec = DftR2cSpec::new(two_n, DftNorm::None)?;
                DctTransform::Forward(self.r2c.create_plan(&fwd_spec)?)
            }
            DctType::III => {
                let inv_spec = DftC2rSpec::new(two_n, DftNorm::None)?;
                let inv = HermitianC2r::new(self.c2r.clone()).create_plan(&inv_spec)?;
                DctTransform::Inverse(inv)
            }
        };

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

        // Allocate scratch buffers: a 2N real buffer, an N+1 half-spectrum, and
        // an N twiddle-product buffer (DCT-II).
        let zero = Complex::new(T::zero(), T::zero());
        let scratch = DctScratch {
            real: vec![T::zero(); two_n],
            half: vec![zero; n + 1],
            twiddle_prod: vec![zero; n],
        };

        Ok(OmniDctPlan {
            transform,
            twiddles,
            vecops: self.vecops.clone(),
            scratch: Mutex::new(scratch),
            length: n,
            dct_type: spec.dct_type(),
            norm: spec.norm(),
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
    use crate::test_utils::{TestDftC2r, TestDftR2c, TestVecOps};

    #[allow(
        clippy::wildcard_imports,
        reason = "bulk golden-vector import in tests"
    )]
    use omnidsp_testdata::dct_scipy::*;

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

    fn make_factory() -> OmniDct<TestDftR2c, TestDftC2r, TestVecOps> {
        OmniDct::new(TestDftR2c, TestDftC2r, TestVecOps)
    }

    // ── Validation tests ─────────────────────────────────────────────

    #[test]
    fn rejects_zero_length() {
        assert!(
            DctSpec::new(0, DctType::II, DctNorm::None).is_err(),
            "zero length should be rejected by the spec constructor"
        );
    }

    #[test]
    fn rejects_input_length_mismatch() {
        let factory = make_factory();
        let spec = DctSpec::new(4, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [1.0, 2.0, 3.0]; // length 3, expects 4
        let mut output = [0.0; 4];
        assert!(
            plan.execute(&input, &mut output).is_err(),
            "mismatched input length should error"
        );
    }

    #[test]
    fn rejects_output_length_mismatch() {
        let factory = make_factory();
        let spec = DctSpec::new(4, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 3]; // length 3, expects 4
        assert!(
            plan.execute(&input, &mut output).is_err(),
            "mismatched output length should error"
        );
    }

    // ── DCT-II analytical tests ──────────────────────────────────────

    #[test]
    fn dct2_length_1() {
        let factory = make_factory();
        let spec = DctSpec::new(1, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [3.0];
        let mut output = [0.0];
        plan.execute(&input, &mut output)
            .expect("process should succeed");
        // DCT-II of [c] = 2*c*cos(0) = 2*c
        assert_approx_eq(&output, &[6.0], EPSILON);
    }

    #[test]
    fn dct2_dc_input() {
        // DCT-II of constant signal concentrates energy at bin 0
        let factory = make_factory();
        let n = 8;
        let spec = DctSpec::new(n, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [5.0; 8];
        let mut output = [0.0; 8];
        plan.execute(&input, &mut output)
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
        let spec = DctSpec::new(4, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [1.0, 0.0, 0.0, 0.0];
        let mut output = [0.0; 4];
        plan.execute(&input, &mut output)
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
        let spec = DctSpec::new(1, DctType::III, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let input = [3.0];
        let mut output = [0.0];
        plan.execute(&input, &mut output)
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

        let spec_ii = DctSpec::new(n, DctType::II, DctNorm::None).expect("valid dct spec");
        let spec_iii = DctSpec::new(n, DctType::III, DctNorm::None).expect("valid dct spec");
        let plan_ii = factory
            .create_plan(&spec_ii)
            .expect("DCT-II plan creation should succeed");
        let plan_iii = factory
            .create_plan(&spec_iii)
            .expect("DCT-III plan creation should succeed");

        let mut freq = [0.0; 8];
        plan_ii
            .execute(&input, &mut freq)
            .expect("DCT-II should succeed");

        let mut recovered = [0.0; 8];
        plan_iii
            .execute(&freq, &mut recovered)
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

        let spec_ii = DctSpec::new(n, DctType::II, DctNorm::Ortho).expect("valid dct spec");
        let spec_iii = DctSpec::new(n, DctType::III, DctNorm::Ortho).expect("valid dct spec");
        let plan_ii = factory
            .create_plan(&spec_ii)
            .expect("DCT-II ortho plan creation should succeed");
        let plan_iii = factory
            .create_plan(&spec_iii)
            .expect("DCT-III ortho plan creation should succeed");

        let mut freq = vec![0.0; n];
        plan_ii
            .execute(input, &mut freq)
            .expect("DCT-II ortho should succeed");

        let mut recovered = vec![0.0; n];
        plan_iii
            .execute(&freq, &mut recovered)
            .expect("DCT-III ortho should succeed");

        assert_approx_eq(&recovered, input, 1e-10);
    }

    #[test]
    fn round_trip_ortho_non_power_of_two() {
        // N=17 verifies the 2N-point FFT handles odd lengths
        let factory = make_factory();
        let n = 17;
        let input = DCT_INPUT_COS17;

        let spec_ii = DctSpec::new(n, DctType::II, DctNorm::Ortho).expect("valid dct spec");
        let spec_iii = DctSpec::new(n, DctType::III, DctNorm::Ortho).expect("valid dct spec");
        let plan_ii = factory
            .create_plan(&spec_ii)
            .expect("plan creation should succeed");
        let plan_iii = factory
            .create_plan(&spec_iii)
            .expect("plan creation should succeed");

        let mut freq = vec![0.0; n];
        plan_ii
            .execute(input, &mut freq)
            .expect("DCT-II should succeed");

        let mut recovered = vec![0.0; n];
        plan_iii
            .execute(&freq, &mut recovered)
            .expect("DCT-III should succeed");

        assert_approx_eq(&recovered, input, 1e-9);
    }

    // ── Scipy reference tests ────────────────────────────────────────

    #[test]
    fn dct2_scipy_ramp8() {
        let factory = make_factory();
        let n = DCT_INPUT_RAMP8.len();
        let spec = DctSpec::new(n, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_RAMP8, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_RAMP8, 1e-10);
    }

    #[test]
    fn dct2_ortho_scipy_ramp8() {
        let factory = make_factory();
        let n = DCT_INPUT_RAMP8.len();
        let spec = DctSpec::new(n, DctType::II, DctNorm::Ortho).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_RAMP8, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_ORTHO_RAMP8, 1e-10);
    }

    #[test]
    fn dct3_scipy_ramp8() {
        let factory = make_factory();
        let n = DCT_INPUT_RAMP8.len();
        let spec = DctSpec::new(n, DctType::III, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_RAMP8, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_RAMP8, 1e-10);
    }

    #[test]
    fn dct3_ortho_scipy_ramp8() {
        let factory = make_factory();
        let n = DCT_INPUT_RAMP8.len();
        let spec = DctSpec::new(n, DctType::III, DctNorm::Ortho).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_RAMP8, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_ORTHO_RAMP8, 1e-10);
    }

    #[test]
    fn dct2_scipy_sine16() {
        let factory = make_factory();
        let n = DCT_INPUT_SINE16.len();
        let spec = DctSpec::new(n, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_SINE16, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_SINE16, 1e-10);
    }

    #[test]
    fn dct2_ortho_scipy_sine16() {
        let factory = make_factory();
        let n = DCT_INPUT_SINE16.len();
        let spec = DctSpec::new(n, DctType::II, DctNorm::Ortho).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_SINE16, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_ORTHO_SINE16, 1e-10);
    }

    #[test]
    fn dct3_scipy_sine16() {
        let factory = make_factory();
        let n = DCT_INPUT_SINE16.len();
        let spec = DctSpec::new(n, DctType::III, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_SINE16, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_SINE16, 1e-10);
    }

    #[test]
    fn dct3_ortho_scipy_sine16() {
        let factory = make_factory();
        let n = DCT_INPUT_SINE16.len();
        let spec = DctSpec::new(n, DctType::III, DctNorm::Ortho).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_SINE16, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_ORTHO_SINE16, 1e-10);
    }

    #[test]
    fn dct2_scipy_cos17_non_power_of_two() {
        let factory = make_factory();
        let n = DCT_INPUT_COS17.len();
        let spec = DctSpec::new(n, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_COS17, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_COS17, 1e-9);
    }

    #[test]
    fn dct2_ortho_scipy_cos17() {
        let factory = make_factory();
        let n = DCT_INPUT_COS17.len();
        let spec = DctSpec::new(n, DctType::II, DctNorm::Ortho).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_COS17, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_ORTHO_COS17, 1e-9);
    }

    #[test]
    fn dct3_scipy_cos17_non_power_of_two() {
        let factory = make_factory();
        let n = DCT_INPUT_COS17.len();
        let spec = DctSpec::new(n, DctType::III, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_COS17, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_COS17, 1e-9);
    }

    #[test]
    fn dct3_ortho_scipy_cos17() {
        let factory = make_factory();
        let n = DCT_INPUT_COS17.len();
        let spec = DctSpec::new(n, DctType::III, DctNorm::Ortho).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_COS17, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_ORTHO_COS17, 1e-9);
    }

    #[test]
    fn dct2_scipy_random64() {
        let factory = make_factory();
        let n = DCT_INPUT_RANDOM64.len();
        let spec = DctSpec::new(n, DctType::II, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_RANDOM64, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_RANDOM64, 1e-8);
    }

    #[test]
    fn dct2_ortho_scipy_random64() {
        let factory = make_factory();
        let n = DCT_INPUT_RANDOM64.len();
        let spec = DctSpec::new(n, DctType::II, DctNorm::Ortho).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_RANDOM64, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT2_ORTHO_RANDOM64, 1e-8);
    }

    #[test]
    fn dct3_scipy_random64() {
        let factory = make_factory();
        let n = DCT_INPUT_RANDOM64.len();
        let spec = DctSpec::new(n, DctType::III, DctNorm::None).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_RANDOM64, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_RANDOM64, 1e-8);
    }

    #[test]
    fn dct3_ortho_scipy_random64() {
        let factory = make_factory();
        let n = DCT_INPUT_RANDOM64.len();
        let spec = DctSpec::new(n, DctType::III, DctNorm::Ortho).expect("valid dct spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let mut output = vec![0.0; n];
        plan.execute(DCT_INPUT_RANDOM64, &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, DCT3_ORTHO_RANDOM64, 1e-8);
    }
}
