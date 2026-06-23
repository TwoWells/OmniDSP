// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Test utilities: naive c2c/r2c/c2r DFT doubles and scalar `VecOps` for unit
//! tests.
//!
//! Only compiled when `cfg(test)` is active.

use std::f64::consts::TAU;

use num_complex::Complex;

use crate::create::CreateProc;
use crate::design::resample::ResampleSpec;
use crate::dispatch::RawDft;
use crate::error::{Error, Result};
use crate::modules::resample::{OmniResample, OmniResampleProcessor};
use crate::traits::dft::{
    DftC2c, DftC2cPlan, DftC2cSpec, DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan,
    DftR2cSpec,
};
use crate::traits::vecops::VecOps;
use crate::types::{Direction, DspFloat};

/// DFT factory for tests.
///
/// Uses Cooley-Tukey radix-2 FFT for power-of-2 lengths and falls
/// back to an O(N²) direct DFT for other lengths.  No external
/// dependencies — just textbook algorithms.
#[derive(Debug, Clone)]
pub struct TestDftC2c;

/// DFT plan for tests.
///
/// Respects [`DftNorm`] so that module tests exercising different
/// normalization conventions get correct (or correctly wrong) results.
#[derive(Debug)]
pub struct TestDftC2cPlan {
    length: usize,
    direction: Direction,
    norm: DftNorm,
}

#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
impl DftC2cPlan<f64> for TestDftC2cPlan {
    fn execute(&self, input: &[Complex<f64>], output: &mut [Complex<f64>]) -> Result<()> {
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

        let sign = match self.direction {
            Direction::Forward => -1.0,
            Direction::Inverse => 1.0,
        };

        output.copy_from_slice(input);

        if self.length.is_power_of_two() {
            fft_radix2(output, sign);
        } else {
            dft_naive(input, output, sign);
        }

        if let Some(scale) = self.compute_scale() {
            for c in output.iter_mut() {
                *c = Complex::new(c.re * scale, c.im * scale);
            }
        }

        Ok(())
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
impl TestDftC2cPlan {
    /// Compute the scaling factor for the configured normalization.
    fn compute_scale(&self) -> Option<f64> {
        let n = self.length as f64;
        match self.norm {
            DftNorm::None => None,
            DftNorm::Inverse => match self.direction {
                Direction::Inverse => Some(1.0 / n),
                Direction::Forward => None,
            },
            DftNorm::Ortho => Some(1.0 / n.sqrt()),
        }
    }
}

impl DftC2c<f64> for TestDftC2c {
    type Plan = TestDftC2cPlan;

    fn create_plan(&self, spec: &DftC2cSpec) -> Result<Self::Plan> {
        Ok(TestDftC2cPlan {
            length: spec.length(),
            direction: spec.direction(),
            norm: spec.norm(),
        })
    }
}

// ─── r2c / c2r doubles ───────────────────────────────────────────────

/// Scaling factor a [`DftNorm`] convention applies to a transform of real
/// length `n` running in `direction`, or `None` when no scaling is needed.
///
/// Mirrors `omnidsp-rustfft`'s `norm_scale` so the doubles and the realfft floor
/// apply identical per-direction factors: the forward-only [`TestDftR2c`] scales
/// like a c2c forward plan, the inverse-only [`TestDftC2r`] like a c2c inverse
/// plan (see [`DftNorm`]).
///
/// The duplication is intentional: `omnidsp-core` cannot depend on
/// `omnidsp-rustfft` (the dependency runs the other way), so the two `norm_scale`
/// bodies are kept deliberately identical rather than shared.
#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
fn norm_scale(norm: DftNorm, direction: Direction, n: usize) -> Option<f64> {
    let n = n as f64;
    match norm {
        DftNorm::None => None,
        DftNorm::Inverse => match direction {
            Direction::Inverse => Some(1.0 / n),
            Direction::Forward => None,
        },
        DftNorm::Ortho => Some(1.0 / n.sqrt()),
    }
}

/// Real-to-complex DFT factory for tests.
///
/// Runs the same radix-2 / naive transform as [`TestDftC2c`] on the real input
/// (embedded as complex) and keeps the first `length / 2 + 1` bins — the
/// half-spectrum.  Forward-only.
#[derive(Debug, Clone)]
pub struct TestDftR2c;

/// Real-to-complex DFT plan for tests.  Respects [`DftNorm`] (forward slice).
#[derive(Debug)]
pub struct TestDftR2cPlan {
    length: usize,
    norm: DftNorm,
}

impl DftR2cPlan<f64> for TestDftR2cPlan {
    fn execute(&self, input: &mut [f64], output: &mut [Complex<f64>]) -> Result<()> {
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

        // Embed the real input as complex and run the full forward FFT, then
        // keep the lower half-spectrum.
        let embedded: Vec<Complex<f64>> = input.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let mut full = vec![Complex::new(0.0, 0.0); self.length];
        if self.length.is_power_of_two() {
            full.copy_from_slice(&embedded);
            fft_radix2(&mut full, -1.0);
        } else {
            dft_naive(&embedded, &mut full, -1.0);
        }

        output.copy_from_slice(&full[..bins]);

        // r2c is forward-only — it scales like a c2c forward plan.
        if let Some(scale) = norm_scale(self.norm, Direction::Forward, self.length) {
            for c in output.iter_mut() {
                *c = Complex::new(c.re * scale, c.im * scale);
            }
        }

        Ok(())
    }
}

impl DftR2c<f64> for TestDftR2c {
    type Plan = TestDftR2cPlan;

    fn create_plan(&self, spec: &DftR2cSpec) -> Result<Self::Plan> {
        Ok(TestDftR2cPlan {
            length: spec.length(),
            norm: spec.norm(),
        })
    }
}

/// Complex-to-real DFT factory for tests.
///
/// Hermitian-extends the `length / 2 + 1` half-spectrum back to the full
/// `length` bins, runs the inverse radix-2 / naive transform, and takes the real
/// part.  Inverse-only.
#[derive(Debug, Clone)]
pub struct TestDftC2r;

/// Complex-to-real DFT plan for tests.  Respects [`DftNorm`] (inverse slice).
///
/// A **raw, bare primitive**: it assumes a clean Hermitian
/// half-spectrum and does **no** DC/Nyquist projection — that shaping lives in
/// [`HermitianC2r`](crate::hermitian::HermitianC2r), matching `RustDftC2r`'s
/// floor.  Taking the real part of the inverse transform already
/// discards any DC/Nyquist imaginary drift, so it is harmless here but is not
/// the double's responsibility to clean.
#[derive(Debug)]
pub struct TestDftC2rPlan {
    length: usize,
    norm: DftNorm,
}

impl DftC2rPlan<f64> for TestDftC2rPlan {
    fn execute(&self, input: &mut [Complex<f64>], output: &mut [f64]) -> Result<()> {
        let n = self.length;
        let bins = n / 2 + 1;
        if input.len() != bins {
            return Err(Error::BufferMismatch {
                expected: bins,
                actual: input.len(),
            });
        }
        if output.len() != n {
            return Err(Error::BufferMismatch {
                expected: n,
                actual: output.len(),
            });
        }

        // Raw bare primitive: no DC/Nyquist projection here — the
        // input is assumed clean-Hermitian, and any drift is shaped upstream by
        // `HermitianC2r`.  Mirror the lower half into the conjugate-symmetric
        // upper half and take the real part of the inverse transform.
        let mut full = vec![Complex::new(0.0, 0.0); n];
        full[..bins].copy_from_slice(input);
        for k in bins..n {
            full[k] = full[n - k].conj();
        }

        let mut time = vec![Complex::new(0.0, 0.0); n];
        if n.is_power_of_two() {
            time.copy_from_slice(&full);
            fft_radix2(&mut time, 1.0);
        } else {
            dft_naive(&full, &mut time, 1.0);
        }

        for (o, c) in output.iter_mut().zip(time.iter()) {
            *o = c.re;
        }

        // c2r is inverse-only — it scales like a c2c inverse plan.
        if let Some(scale) = norm_scale(self.norm, Direction::Inverse, n) {
            for o in output.iter_mut() {
                *o *= scale;
            }
        }

        Ok(())
    }
}

impl DftC2r<f64> for TestDftC2r {
    type Plan = TestDftC2rPlan;

    fn create_plan(&self, spec: &DftC2rSpec) -> Result<Self::Plan> {
        Ok(TestDftC2rPlan {
            length: spec.length(),
            norm: spec.norm(),
        })
    }
}

// ─── FFT internals ───────────────────────────────────────────────────

/// O(N²) direct DFT — fallback for non-power-of-2 lengths.
#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
fn dft_naive(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: f64) {
    let n = input.len();
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = Complex::new(0.0, 0.0);
        for (j, &x) in input.iter().enumerate() {
            let angle = sign * TAU * (k * j) as f64 / n as f64;
            sum += x * Complex::new(angle.cos(), angle.sin());
        }
        *out = sum;
    }
}

/// In-place Cooley-Tukey radix-2 FFT.  `data.len()` must be a power of 2.
#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
fn fft_radix2(data: &mut [Complex<f64>], sign: f64) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly passes.
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle_step = sign * TAU / size as f64;
        for start in (0..n).step_by(size) {
            for k in 0..half {
                let angle = angle_step * k as f64;
                let twiddle = Complex::new(angle.cos(), angle.sin());
                let u = data[start + k];
                let t = twiddle * data[start + k + half];
                data[start + k] = u + t;
                data[start + k + half] = u - t;
            }
        }
        size *= 2;
    }
}

/// Scalar loop `VecOps` for tests.
///
/// Uses all default implementations from the [`VecOps`] trait.
#[derive(Debug, Clone)]
pub struct TestVecOps;

impl VecOps<f64> for TestVecOps {}

/// Minimal foundational backend for tests.
///
/// Bundles the test DFT doubles ([`TestDftC2c`] / [`TestDftR2c`] /
/// [`TestDftC2r`]) and the scalar [`VecOps`] so it satisfies the foundational
/// [`Backend<f64>`](crate::dispatch::Backend) contract, and routes resampling
/// through [`OmniResample`].  This is what the CQT octave decimator dispatches
/// against: under the dispatch model only a `Backend` routes sub-processors, and
/// `OmniResample` alone is not a backend (it has no DFT family).
#[derive(Debug, Clone)]
pub struct TestBackend;

impl RawDft<f64> for TestBackend {
    type C2c = TestDftC2c;
    type R2c = TestDftR2c;
    type C2r = TestDftC2r;

    fn raw_dftc2c(&self) -> Self::C2c {
        TestDftC2c
    }
    fn raw_dftr2c(&self) -> Self::R2c {
        TestDftR2c
    }
    fn raw_dftc2r(&self) -> Self::C2r {
        TestDftC2r
    }
}

// The test backend is its own (scalar-default) `VecOps` provider: the empty
// impl inherits the scalar defaults (the same ones `TestVecOps` uses), which is
// what the composed modules call for their bulk vector ops.
impl VecOps<f64> for TestBackend {}

impl CreateProc<ResampleSpec> for TestBackend {
    type Proc<T>
        = OmniResampleProcessor<T>
    where
        Self: crate::dispatch::Backend<T>;

    fn create_proc<T>(&self, spec: &ResampleSpec) -> Result<Self::Proc<T>>
    where
        Self: crate::dispatch::Backend<T>,
        T: DspFloat,
    {
        OmniResample::new().create_proc::<T>(spec)
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    reason = "tests use expect for clarity and small exact usize→f64 casts"
)]
mod tests {
    use num_complex::Complex;

    use super::{TestDftC2c, TestDftC2r, TestDftR2c};
    use crate::traits::dft::{
        DftC2c, DftC2cPlan, DftC2cSpec, DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c,
        DftR2cPlan, DftR2cSpec,
    };
    use crate::types::Direction;

    const EPS: f64 = 1e-9;

    /// The ramp `[1, 2, …, n]` as a real signal.
    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i + 1) as f64).collect()
    }

    fn r2c(n: usize, norm: DftNorm, input: &[f64]) -> Vec<Complex<f64>> {
        let spec = DftR2cSpec::new(n, norm).expect("valid r2c spec");
        let plan = DftR2c::<f64>::create_plan(&TestDftR2c, &spec).expect("r2c plan");
        // r2c consumes its input — hand it a throwaway copy.
        let mut scratch = input.to_vec();
        let mut out = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        plan.execute(&mut scratch, &mut out).expect("r2c process");
        out
    }

    fn c2r(n: usize, norm: DftNorm, input: &[Complex<f64>]) -> Vec<f64> {
        let spec = DftC2rSpec::new(n, norm).expect("valid c2r spec");
        let plan = DftC2r::<f64>::create_plan(&TestDftC2r, &spec).expect("c2r plan");
        // c2r consumes its input — hand it a throwaway copy.
        let mut scratch = input.to_vec();
        let mut out = vec![0.0_f64; n];
        plan.execute(&mut scratch, &mut out).expect("c2r process");
        out
    }

    /// Full forward c2c spectrum of a real signal, via the [`TestDftC2c`] double
    /// — the in-crate oracle the r2c double is validated against.
    fn c2c_forward(n: usize, input: &[f64]) -> Vec<Complex<f64>> {
        let embedded: Vec<Complex<f64>> = input.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let spec = DftC2cSpec::new(n, Direction::Forward, DftNorm::None).expect("c2c spec");
        let plan = DftC2c::<f64>::create_plan(&TestDftC2c, &spec).expect("c2c plan");
        let mut out = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&embedded, &mut out).expect("c2c process");
        out
    }

    fn assert_complex_close(got: &[Complex<f64>], want: &[Complex<f64>]) {
        assert_eq!(got.len(), want.len(), "complex slice lengths differ");
        for (i, (x, y)) in got.iter().zip(want).enumerate() {
            assert!(
                (x.re - y.re).abs() < EPS && (x.im - y.im).abs() < EPS,
                "complex mismatch at {i}: got ({}, {}), want ({}, {})",
                x.re,
                x.im,
                y.re,
                y.im,
            );
        }
    }

    fn assert_real_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "real slice lengths differ");
        for (i, (x, y)) in got.iter().zip(want).enumerate() {
            assert!(
                (x - y).abs() < EPS,
                "real mismatch at {i}: got {x}, want {y}",
            );
        }
    }

    /// `r2c(x)` equals the first `N/2+1` bins of the full c2c spectrum — the
    /// double matches its in-crate oracle across parities (the floor's
    /// `r2c == c2c[..N/2+1]` check, mirrored here without crossing the crate
    /// boundary to the realfft floor).
    #[test]
    fn r2c_matches_c2c_half_spectrum() {
        for n in [1_usize, 2, 7, 8, 16] {
            let x = ramp(n);
            let half = r2c(n, DftNorm::None, &x);
            let full = c2c_forward(n, &x);
            assert_complex_close(&half, &full[..=n / 2]);
        }
    }

    /// `c2r(r2c(x)) ≈ x` under `DftNorm::Inverse` for every parity.
    #[test]
    fn round_trip_identity() {
        for n in [1_usize, 2, 7, 8, 16] {
            let x = ramp(n);
            let spectrum = r2c(n, DftNorm::Inverse, &x);
            let recovered = c2r(n, DftNorm::Inverse, &spectrum);
            assert_real_close(&recovered, &x);
        }
    }

    /// Ortho r2c output is the unnormalized output scaled by `1/√N`.
    #[test]
    fn r2c_ortho_scaling() {
        for n in [7_usize, 8] {
            let x = ramp(n);
            let none = r2c(n, DftNorm::None, &x);
            let ortho = r2c(n, DftNorm::Ortho, &x);
            let s = 1.0 / (n as f64).sqrt();
            let expected: Vec<Complex<f64>> = none
                .iter()
                .map(|c| Complex::new(c.re * s, c.im * s))
                .collect();
            assert_complex_close(&ortho, &expected);
        }
    }

    /// `Inverse` and `Ortho` c2r outputs are the unnormalized output scaled by
    /// `1/N` and `1/√N` respectively.
    #[test]
    fn c2r_norm_scaling() {
        for n in [7_usize, 8] {
            let x = ramp(n);
            let half = r2c(n, DftNorm::None, &x);
            let none = c2r(n, DftNorm::None, &half);
            let inverse = c2r(n, DftNorm::Inverse, &half);
            let ortho = c2r(n, DftNorm::Ortho, &half);
            let inv_n = 1.0 / n as f64;
            let inv_sqrt = 1.0 / (n as f64).sqrt();
            let exp_inverse: Vec<f64> = none.iter().map(|&v| v * inv_n).collect();
            let exp_ortho: Vec<f64> = none.iter().map(|&v| v * inv_sqrt).collect();
            assert_real_close(&inverse, &exp_inverse);
            assert_real_close(&ortho, &exp_ortho);
        }
    }

    // The c2r DC/(even-`N`)Nyquist drift-tolerance test moved to
    // `crate::hermitian`: drift projection is now a property of the
    // `HermitianC2r` shaping decorator, not of the bare `TestDftC2r` primitive.

    #[test]
    fn buffer_mismatch_errors() {
        let r_spec = DftR2cSpec::new(8, DftNorm::None).expect("valid r2c spec");
        let r_plan = DftR2c::<f64>::create_plan(&TestDftR2c, &r_spec).expect("r2c plan");
        let mut half = vec![Complex::new(0.0, 0.0); 5];
        assert!(
            r_plan.execute(&mut [0.0_f64; 7], &mut half).is_err(),
            "wrong r2c input length must error",
        );
        let mut short_half = vec![Complex::new(0.0, 0.0); 4];
        assert!(
            r_plan.execute(&mut [0.0_f64; 8], &mut short_half).is_err(),
            "wrong r2c output length must error",
        );

        let c_spec = DftC2rSpec::new(8, DftNorm::None).expect("valid c2r spec");
        let c_plan = DftC2r::<f64>::create_plan(&TestDftC2r, &c_spec).expect("c2r plan");
        let mut short_in = vec![Complex::new(0.0, 0.0); 4];
        let mut real_out = [0.0_f64; 8];
        assert!(
            c_plan.execute(&mut short_in, &mut real_out).is_err(),
            "wrong c2r input length must error",
        );
        let mut good_in = vec![Complex::new(0.0, 0.0); 5];
        let mut short_out = [0.0_f64; 7];
        assert!(
            c_plan.execute(&mut good_in, &mut short_out).is_err(),
            "wrong c2r output length must error",
        );
    }
}
