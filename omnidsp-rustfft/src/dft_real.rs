// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! [`RustDftR2c`] / [`RustDftC2r`] — the pure Rust real-DFT floor.
//!
//! These wrap [`realfft`](https://crates.io/crates/realfft), the real-FFT
//! companion to the `rustfft` already backing [`RustDftC2c`](crate::RustDftC2c).
//! Together they are the **universal real-FFT floor**: a backend with no native
//! real kernel inherits them.  `realfft` is pure safe Rust, so
//! `#![forbid(unsafe_code)]` still holds.
//!
//! Both plans hold the `realfft` scratch behind a [`Mutex`], matching the
//! `RustFFT` scratch pattern in [`RustDftC2cPlan`](crate::RustDftC2cPlan).  The
//! `realfft` `process_with_scratch` API consumes its input as working memory, so
//! the plans take their input by `&mut` and hand the caller's buffer straight to
//! the kernel — no internal copy.
//!
//! Normalization follows the family-level [`DftNorm`] convention via the shared
//! [`norm_scale`] helper, so module round-trips compose identically across c2c
//! and r2c/c2r.

use std::sync::{Arc, Mutex};

use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::FftNum;

use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::dft::{
    DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan, DftR2cSpec,
};
use omnidsp_core::types::{Direction, DspFloat};

use crate::dft::norm_scale;

// ─── Type erasure ────────────────────────────────────────────────────

/// Private trait that hides `realfft`'s `RealToComplex` (and its `FftNum`
/// bound) from the public plan struct, mirroring `DftC2cEngine`.
trait DftR2cEngine<T>: Send + Sync {
    fn process(
        &self,
        input: &mut [T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<()>;
}

impl<T: FftNum> DftR2cEngine<T> for Arc<dyn RealToComplex<T>> {
    fn process(
        &self,
        input: &mut [T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<()> {
        RealToComplex::process_with_scratch(&**self, input, output, scratch)
            .map_err(|e| Error::Internal(format!("realfft r2c failed: {e}")))
    }
}

/// Private trait that hides `realfft`'s `ComplexToReal` (and its `FftNum`
/// bound) from the public plan struct.
trait DftC2rEngine<T>: Send + Sync {
    fn process(
        &self,
        input: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<()>;
}

impl<T: FftNum> DftC2rEngine<T> for Arc<dyn ComplexToReal<T>> {
    fn process(
        &self,
        input: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<()> {
        ComplexToReal::process_with_scratch(&**self, input, output, scratch)
            .map_err(|e| Error::Internal(format!("realfft c2r failed: {e}")))
    }
}

// ─── r2c: real → complex half-spectrum ───────────────────────────────

/// Pure Rust real-to-complex DFT factory.
///
/// Creates [`RustDftR2cPlan`] instances backed by `realfft`.  Forward-only;
/// maps `length` real samples to a `length / 2 + 1` complex half-spectrum.
#[derive(Debug, Clone, Copy)]
pub struct RustDftR2c;

/// Execution plan for a real-to-complex DFT backed by `realfft`.
///
/// Holds the precomputed transform plus the `realfft` scratch behind a single
/// [`Mutex`] (keeping `Send + Sync` with allocation-free execution).  The
/// caller's `&mut` input is handed straight to `realfft`, which consumes it as
/// working memory — there is no internal input copy.
pub struct RustDftR2cPlan<T> {
    engine: Box<dyn DftR2cEngine<T>>,
    length: usize,
    norm: DftNorm,
    scratch: Mutex<Vec<Complex<T>>>,
}

impl<T: DspFloat> DftR2cPlan<T> for RustDftR2cPlan<T> {
    fn process(&self, input: &mut [T], output: &mut [Complex<T>]) -> Result<()> {
        let bins = self.length / 2 + 1;
        if input.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: input.len(),
            });
        }
        if output.len() != bins {
            return Err(Error::BufferMismatch {
                expected: bins,
                actual: output.len(),
            });
        }

        let mut work = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;
        self.engine.process(input, output, work.as_mut_slice())?;
        // Release the scratch lock before scaling `output` (caller-owned).
        drop(work);

        // r2c is forward-only — it scales like a c2c forward plan.
        if let Some(s) = norm_scale::<T>(self.norm, Direction::Forward, self.length)? {
            for bin in output.iter_mut() {
                *bin = Complex::new(bin.re * s, bin.im * s);
            }
        }

        Ok(())
    }
}

// ─── c2r: complex half-spectrum → real ───────────────────────────────

/// Pure Rust complex-to-real DFT factory.
///
/// Creates [`RustDftC2rPlan`] instances backed by `realfft`.  Inverse-only;
/// maps a `length / 2 + 1` complex half-spectrum back to `length` real samples.
#[derive(Debug, Clone, Copy)]
pub struct RustDftC2r;

/// Execution plan for a complex-to-real DFT backed by `realfft`.
///
/// Holds the precomputed transform plus the `realfft` scratch behind a single
/// [`Mutex`].  The caller's `&mut` half-spectrum is handed straight to
/// `realfft`, which consumes it as working memory — there is no
/// internal input copy.
///
/// # DC / Nyquist convention
///
/// This is the **raw** primitive: it performs **no** DC/Nyquist
/// projection.  A half-spectrum that is the transform of a real signal is
/// Hermitian — its DC bin (index 0), and, when `length` is even, its Nyquist
/// bin (index `length / 2`), are purely real — and `realfft` flags non-zero
/// imaginary parts there as an error.  Drift-tolerant projection onto the
/// nearest valid Hermitian spectrum is the job of the
/// [`HermitianC2r`](omnidsp_core::hermitian::HermitianC2r) shaping decorator,
/// not of this primitive: a direct call with a dirty DC/Nyquist surfaces
/// `realfft`'s native behavior (the documented §1 escape hatch).
pub struct RustDftC2rPlan<T> {
    engine: Box<dyn DftC2rEngine<T>>,
    length: usize,
    norm: DftNorm,
    scratch: Mutex<Vec<Complex<T>>>,
}

impl<T: DspFloat> DftC2rPlan<T> for RustDftC2rPlan<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [T]) -> Result<()> {
        let bins = self.length / 2 + 1;
        if input.len() != bins {
            return Err(Error::BufferMismatch {
                expected: bins,
                actual: input.len(),
            });
        }
        if output.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: output.len(),
            });
        }

        let mut work = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;
        self.engine.process(input, output, work.as_mut_slice())?;
        // Release the scratch lock before scaling `output` (caller-owned).
        drop(work);

        // c2r is inverse-only — it scales like a c2c inverse plan.
        if let Some(s) = norm_scale::<T>(self.norm, Direction::Inverse, self.length)? {
            for sample in output.iter_mut() {
                let scaled = *sample * s;
                *sample = scaled;
            }
        }

        Ok(())
    }
}

// ─── Factory impls (concrete float widths) ───────────────────────────

/// Implement both real-DFT factory traits for a concrete float width.
///
/// Mirrors `RustDftC2c`'s `impl_dft!`: the `realfft` planner and the
/// `Complex::default()` buffer fills need concrete types, so the impls are
/// generated per width rather than via a blanket `impl<T>`.
macro_rules! impl_real_dft {
    ($t:ty) => {
        impl DftR2c<$t> for RustDftR2c {
            type Plan = RustDftR2cPlan<$t>;

            fn create_plan(&self, spec: &DftR2cSpec<$t>) -> Result<Self::Plan> {
                let length = spec.length();

                let mut planner = RealFftPlanner::<$t>::new();
                let fft = planner.plan_fft_forward(length);
                let scratch_len = fft.get_scratch_len();

                Ok(RustDftR2cPlan {
                    engine: Box::new(fft),
                    length,
                    norm: spec.norm(),
                    scratch: Mutex::new(vec![Complex::default(); scratch_len]),
                })
            }
        }

        impl DftC2r<$t> for RustDftC2r {
            type Plan = RustDftC2rPlan<$t>;

            fn create_plan(&self, spec: &DftC2rSpec<$t>) -> Result<Self::Plan> {
                let length = spec.length();

                let mut planner = RealFftPlanner::<$t>::new();
                let fft = planner.plan_fft_inverse(length);
                let scratch_len = fft.get_scratch_len();

                Ok(RustDftC2rPlan {
                    engine: Box::new(fft),
                    length,
                    norm: spec.norm(),
                    scratch: Mutex::new(vec![Complex::default(); scratch_len]),
                })
            }
        }
    };
}

impl_real_dft!(f32);
impl_real_dft!(f64);

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use omnidsp_core::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec};

    use super::{
        Complex, DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan, DftR2cSpec,
        Direction, DspFloat, RustDftC2r, RustDftR2c,
    };
    use crate::RustDftC2c;

    const EPS_F32: f32 = 1e-3;
    const EPS_F64: f64 = 1e-9;

    /// A length-`n` complex zero buffer without requiring `T: Default`.
    fn czeros<T: DspFloat>(n: usize) -> Vec<Complex<T>> {
        vec![Complex::new(T::zero(), T::zero()); n]
    }

    /// The ramp `[1, 2, …, n]` as a real signal.
    fn ramp<T: DspFloat>(n: usize) -> Vec<T> {
        (0..n)
            .map(|i| T::from_usize(i + 1).expect("usize fits in float"))
            .collect()
    }

    fn assert_complex_close<T: DspFloat + std::fmt::Display>(
        got: &[Complex<T>],
        want: &[Complex<T>],
        eps: T,
    ) {
        assert_eq!(got.len(), want.len(), "complex slice lengths differ");
        for (i, (x, y)) in got.iter().zip(want).enumerate() {
            assert!(
                (x.re - y.re).abs() < eps && (x.im - y.im).abs() < eps,
                "complex mismatch at {i}: got ({}, {}), want ({}, {})",
                x.re,
                x.im,
                y.re,
                y.im,
            );
        }
    }

    fn assert_real_close<T: DspFloat + std::fmt::Display>(got: &[T], want: &[T], eps: T) {
        assert_eq!(got.len(), want.len(), "real slice lengths differ");
        for (i, (x, y)) in got.iter().zip(want).enumerate() {
            assert!(
                (*x - *y).abs() < eps,
                "real mismatch at {i}: got {x}, want {y}",
            );
        }
    }

    fn run_r2c<T: DspFloat>(n: usize, norm: DftNorm, input: &[T]) -> Vec<Complex<T>>
    where
        RustDftR2c: DftR2c<T>,
    {
        let spec = DftR2cSpec::<T>::new(n, norm).expect("valid r2c spec");
        let plan = DftR2c::<T>::create_plan(&RustDftR2c, &spec).expect("r2c plan");
        // r2c consumes its input — hand it a throwaway copy.
        let mut scratch = input.to_vec();
        let mut out = czeros::<T>(n / 2 + 1);
        plan.process(&mut scratch, &mut out).expect("r2c process");
        out
    }

    fn run_c2r<T: DspFloat>(n: usize, norm: DftNorm, input: &[Complex<T>]) -> Vec<T>
    where
        RustDftC2r: DftC2r<T>,
    {
        let spec = DftC2rSpec::<T>::new(n, norm).expect("valid c2r spec");
        let plan = DftC2r::<T>::create_plan(&RustDftC2r, &spec).expect("c2r plan");
        // c2r consumes its input — hand it a throwaway copy.
        let mut scratch = input.to_vec();
        let mut out = vec![T::zero(); n];
        plan.process(&mut scratch, &mut out).expect("c2r process");
        out
    }

    /// `c2r(r2c(x)) ≈ x` under `DftNorm::Inverse` for every width and parity.
    fn check_round_trip<T: DspFloat + std::fmt::Display>(n: usize, eps: T)
    where
        RustDftR2c: DftR2c<T>,
        RustDftC2r: DftC2r<T>,
    {
        let input = ramp::<T>(n);
        let spectrum = run_r2c::<T>(n, DftNorm::Inverse, &input);
        let recovered = run_c2r::<T>(n, DftNorm::Inverse, &spectrum);
        assert_real_close(&recovered, &input, eps);
    }

    #[test]
    fn round_trip_f32() {
        for n in [1_usize, 2, 7, 8] {
            check_round_trip::<f32>(n, EPS_F32);
        }
    }

    #[test]
    fn round_trip_f64() {
        for n in [1_usize, 2, 7, 8] {
            check_round_trip::<f64>(n, EPS_F64);
        }
    }

    /// `r2c(x)` equals the first `N/2+1` bins of the full c2c spectrum.
    fn check_oracle<T: DspFloat + std::fmt::Display>(n: usize, eps: T)
    where
        RustDftR2c: DftR2c<T>,
        RustDftC2c: DftC2c<T>,
    {
        let real = ramp::<T>(n);
        let embedded: Vec<Complex<T>> = real.iter().map(|&v| Complex::new(v, T::zero())).collect();

        let half = run_r2c::<T>(n, DftNorm::None, &real);

        let c_spec = DftC2cSpec::<T>::new(n, Direction::Forward, DftNorm::None).expect("c2c spec");
        let c_plan = DftC2c::<T>::create_plan(&RustDftC2c, &c_spec).expect("c2c plan");
        let mut full = czeros::<T>(n);
        c_plan.process(&embedded, &mut full).expect("c2c process");

        let bins = n / 2 + 1;
        assert_complex_close(&half, &full[..bins], eps);
    }

    #[test]
    fn oracle_r2c_matches_c2c_f32() {
        for n in [1_usize, 2, 7, 8] {
            check_oracle::<f32>(n, EPS_F32);
        }
    }

    #[test]
    fn oracle_r2c_matches_c2c_f64() {
        for n in [1_usize, 2, 7, 8] {
            check_oracle::<f64>(n, EPS_F64);
        }
    }

    /// Ortho r2c output is the unnormalized output scaled by `1/√N`.
    fn check_r2c_norm<T: DspFloat + std::fmt::Display>(n: usize, eps: T)
    where
        RustDftR2c: DftR2c<T>,
    {
        let real = ramp::<T>(n);
        let none = run_r2c::<T>(n, DftNorm::None, &real);
        let ortho = run_r2c::<T>(n, DftNorm::Ortho, &real);

        let inv_sqrt_n = T::one() / T::from_usize(n).expect("usize fits").sqrt();
        let expected: Vec<Complex<T>> = none
            .iter()
            .map(|c| Complex::new(c.re * inv_sqrt_n, c.im * inv_sqrt_n))
            .collect();
        assert_complex_close(&ortho, &expected, eps);
    }

    #[test]
    fn r2c_norm_scaling() {
        check_r2c_norm::<f32>(8, EPS_F32);
        check_r2c_norm::<f64>(7, EPS_F64);
    }

    /// `Inverse` and `Ortho` c2r outputs are the unnormalized output scaled by
    /// `1/N` and `1/√N` respectively.
    fn check_c2r_norm<T: DspFloat + std::fmt::Display>(n: usize, eps: T)
    where
        RustDftR2c: DftR2c<T>,
        RustDftC2r: DftC2r<T>,
    {
        let real = ramp::<T>(n);
        let half = run_r2c::<T>(n, DftNorm::None, &real);

        let none = run_c2r::<T>(n, DftNorm::None, &half);
        let inverse = run_c2r::<T>(n, DftNorm::Inverse, &half);
        let ortho = run_c2r::<T>(n, DftNorm::Ortho, &half);

        let inv_n = T::one() / T::from_usize(n).expect("usize fits");
        let inv_sqrt_n = T::one() / T::from_usize(n).expect("usize fits").sqrt();
        let exp_inverse: Vec<T> = none.iter().map(|&v| v * inv_n).collect();
        let exp_ortho: Vec<T> = none.iter().map(|&v| v * inv_sqrt_n).collect();
        assert_real_close(&inverse, &exp_inverse, eps);
        assert_real_close(&ortho, &exp_ortho, eps);
    }

    #[test]
    fn c2r_norm_scaling() {
        check_c2r_norm::<f32>(8, EPS_F32);
        check_c2r_norm::<f64>(7, EPS_F64);
    }

    #[test]
    fn r2c_buffer_mismatch() {
        let spec = DftR2cSpec::<f64>::new(8, DftNorm::None).expect("valid spec");
        let plan = DftR2c::<f64>::create_plan(&RustDftR2c, &spec).expect("plan");

        let mut short_input = [0.0_f64; 7];
        let mut out = czeros::<f64>(5);
        assert!(
            plan.process(&mut short_input, &mut out).is_err(),
            "wrong input length should error",
        );

        let mut input = [0.0_f64; 8];
        let mut short_out = czeros::<f64>(4);
        assert!(
            plan.process(&mut input, &mut short_out).is_err(),
            "wrong output length should error",
        );
    }

    #[test]
    fn c2r_buffer_mismatch() {
        let spec = DftC2rSpec::<f64>::new(8, DftNorm::None).expect("valid spec");
        let plan = DftC2r::<f64>::create_plan(&RustDftC2r, &spec).expect("plan");

        let mut short_input = czeros::<f64>(4);
        let mut out = [0.0_f64; 8];
        assert!(
            plan.process(&mut short_input, &mut out).is_err(),
            "wrong input length should error",
        );

        let mut input = czeros::<f64>(5);
        let mut short_out = [0.0_f64; 7];
        assert!(
            plan.process(&mut input, &mut short_out).is_err(),
            "wrong output length should error",
        );
    }
}
