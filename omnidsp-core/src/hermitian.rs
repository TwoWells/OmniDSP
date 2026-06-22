// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Hermitian boundary-shaping decorators for the real-DFT primitives.
//!
//! The DFT of a real signal is Hermitian: its DC bin — and, for even `length`,
//! its Nyquist bin — are purely real.  Float rounding leaves ~epsilon imaginary
//! there, and every `r2c → multiply → c2r` chain (Conv/FIR/CrossCorr/DCT)
//! accumulates a little more.  Rather than leave each vendor kernel to tolerate
//! (or reject) that drift, `OmniDSP` enforces the shape **once, at the
//! composition layer**:
//!
//! - [`HermitianC2r`] projects the half-spectrum onto the nearest valid
//!   Hermitian spectrum (`in[0].im = 0`, even-`N` `in[N/2].im = 0`) **before**
//!   the inverse transform.  This is **load-bearing**: it stops a
//!   `realfft`-style "non-real DC/Nyquist" error — or a kernel that honors the
//!   imaginary part — from diverging on legitimate round-trip drift.
//! - [`HermitianR2c`] cleans the same bins on the **output** of the forward
//!   transform.  This is **hygiene, not safety**: the c2r-side
//!   projection already absorbs round-trip drift, so its sole value is a
//!   bit-exactly-real DC/Nyquist for consumers that inspect the raw
//!   half-spectrum directly.
//!
//! Both project **in place** — no scratch, no copy — around an inner plan.  The
//! inner real-DFT primitives consume their `&mut` input, so the
//! c2r projection writes the caller's buffer it was going to lose anyway.
//!
//! The bare [`DftC2c`](crate::traits::dft::DftC2c) primitive has no Hermitian
//! convention to enforce and so gets no wrapper; only the two real
//! transforms are shaped.

use num_complex::Complex;

use crate::error::Result;
use crate::traits::dft::{DftC2r, DftC2rPlan, DftC2rSpec, DftR2c, DftR2cPlan, DftR2cSpec};
use crate::types::DspFloat;

// ─── c2r: input-projecting decorator (load-bearing) ──────────────────

/// Factory decorator that Hermitian-projects a [`DftC2r`] plan's input.
///
/// Wraps any `C: DftC2r<T>` and returns a plan that zeroes the DC bin — and,
/// for even `length`, the Nyquist bin — imaginary parts **in place** on the
/// caller's `&mut` half-spectrum before delegating to the inner c2r plan.
///
/// This is the shaped, drift-tolerant form modules compose; the
/// inner primitive stays raw.
#[derive(Debug, Clone, Copy)]
pub struct HermitianC2r<C> {
    inner: C,
}

impl<C> HermitianC2r<C> {
    /// Wrap an inner [`DftC2r`] factory in the Hermitian input projection.
    #[must_use]
    pub const fn new(inner: C) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped inner factory.
    pub const fn inner(&self) -> &C {
        &self.inner
    }

    /// Unwrap, returning the inner factory.
    #[must_use]
    pub fn into_inner(self) -> C {
        self.inner
    }
}

/// Execution plan produced by [`HermitianC2r`].
///
/// Projects the half-spectrum onto the nearest valid Hermitian spectrum in
/// place, then runs the inner [`DftC2rPlan`].
#[derive(Debug)]
pub struct HermitianC2rPlan<P> {
    inner: P,
    length: usize,
}

impl<T, P> DftC2rPlan<T> for HermitianC2rPlan<P>
where
    T: DspFloat,
    P: DftC2rPlan<T>,
{
    fn process(&self, input: &mut [Complex<T>], output: &mut [T]) -> Result<()> {
        let bins = self.length / 2 + 1;
        // Project in place onto the nearest valid Hermitian spectrum.
        // Skip on a length mismatch — the inner plan owns that error so it
        // is reported (rather than panicking here on an out-of-range index).
        if input.len() == bins {
            input[0].im = T::zero();
            if self.length.is_multiple_of(2) {
                input[bins - 1].im = T::zero();
            }
        }
        self.inner.process(input, output)
    }
}

impl<T, C> DftC2r<T> for HermitianC2r<C>
where
    T: DspFloat,
    C: DftC2r<T>,
{
    type Plan = HermitianC2rPlan<C::Plan>;

    fn create_plan(&self, spec: &DftC2rSpec<T>) -> Result<Self::Plan> {
        Ok(HermitianC2rPlan {
            inner: self.inner.create_plan(spec)?,
            length: spec.length(),
        })
    }
}

// ─── r2c: output-cleaning decorator (hygiene) ────────────────────────

/// Factory decorator that cleans a [`DftR2c`] plan's output to be Hermitian.
///
/// Wraps any `R: DftR2c<T>` and returns a plan that delegates to the inner r2c
/// plan, then zeroes the DC bin — and, for even `length`, the Nyquist bin —
/// imaginary parts **in place** on the output half-spectrum.
///
/// This is hygiene, not safety: it guarantees a bit-exactly-real
/// DC/Nyquist for consumers inspecting the raw half-spectrum, identically across
/// backends.  Modules that only feed the spectrum back through c2r need not wrap
/// r2c (the c2r-side projection absorbs the drift).
#[derive(Debug, Clone, Copy)]
pub struct HermitianR2c<R> {
    inner: R,
}

impl<R> HermitianR2c<R> {
    /// Wrap an inner [`DftR2c`] factory in the Hermitian output cleaning.
    #[must_use]
    pub const fn new(inner: R) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped inner factory.
    pub const fn inner(&self) -> &R {
        &self.inner
    }

    /// Unwrap, returning the inner factory.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

/// Execution plan produced by [`HermitianR2c`].
///
/// Runs the inner [`DftR2cPlan`], then cleans the DC/Nyquist imaginary parts on
/// the output half-spectrum in place.
#[derive(Debug)]
pub struct HermitianR2cPlan<P> {
    inner: P,
    length: usize,
}

impl<T, P> DftR2cPlan<T> for HermitianR2cPlan<P>
where
    T: DspFloat,
    P: DftR2cPlan<T>,
{
    fn process(&self, input: &mut [T], output: &mut [Complex<T>]) -> Result<()> {
        self.inner.process(input, output)?;
        let bins = self.length / 2 + 1;
        // After a successful transform `output.len() == bins`; the guard keeps
        // the indexing panic-free regardless of the inner plan's behavior.
        if output.len() == bins {
            output[0].im = T::zero();
            if self.length.is_multiple_of(2) {
                output[bins - 1].im = T::zero();
            }
        }
        Ok(())
    }
}

impl<T, R> DftR2c<T> for HermitianR2c<R>
where
    T: DspFloat,
    R: DftR2c<T>,
{
    type Plan = HermitianR2cPlan<R::Plan>;

    fn create_plan(&self, spec: &DftR2cSpec<T>) -> Result<Self::Plan> {
        Ok(HermitianR2cPlan {
            inner: self.inner.create_plan(spec)?,
            length: spec.length(),
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "tests use expect for clarity, small exact usize→f64 casts, and assert \
              the decorator sets DC/Nyquist imaginaries to bit-exact zero"
)]
mod tests {
    use num_complex::Complex;

    use super::{HermitianC2r, HermitianR2c};
    use crate::test_utils::{TestDftC2r, TestDftR2c};
    use crate::traits::dft::{
        DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan, DftR2cSpec,
    };

    const EPS: f64 = 1e-9;

    /// The ramp `[1, 2, …, n]` as a real signal.
    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i + 1) as f64).collect()
    }

    /// Run the bare [`TestDftR2c`] (no shaping) — the clean reference spectrum.
    fn bare_r2c(n: usize, input: &[f64]) -> Vec<Complex<f64>> {
        let spec = DftR2cSpec::<f64>::new(n, DftNorm::None).expect("valid r2c spec");
        let plan = DftR2c::<f64>::create_plan(&TestDftR2c, &spec).expect("r2c plan");
        let mut scratch = input.to_vec();
        let mut out = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        plan.process(&mut scratch, &mut out).expect("r2c process");
        out
    }

    /// Run [`HermitianC2r`] over [`TestDftC2r`] on the given half-spectrum.
    fn shaped_c2r(n: usize, input: &mut [Complex<f64>]) -> Vec<f64> {
        let factory = HermitianC2r::new(TestDftC2r);
        let spec = DftC2rSpec::<f64>::new(n, DftNorm::Inverse).expect("valid c2r spec");
        let plan = DftC2r::<f64>::create_plan(&factory, &spec).expect("c2r plan");
        let mut out = vec![0.0_f64; n];
        plan.process(input, &mut out).expect("c2r process");
        out
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

    /// `HermitianC2r` projects a DC/(even-`N`)Nyquist-perturbed half-spectrum
    /// back onto the valid Hermitian spectrum **in place** and reconstructs the
    /// same real signal as the clean input — the drift case, now exercised
    /// through the wrapper.
    #[test]
    fn c2r_projects_dc_nyquist_drift_in_place() {
        for n in [7_usize, 8] {
            let x = ramp(n);
            let clean = bare_r2c(n, &x);

            // Perturb DC (and, for even N, Nyquist) imaginary parts well above
            // EPS, so honoring them would diverge from the clean reconstruction.
            let mut dirty = clean.clone();
            dirty[0].im = 2.5;
            if n.is_multiple_of(2) {
                let nyquist = dirty.len() - 1;
                dirty[nyquist].im = 2.5;
            }

            let from_clean = shaped_c2r(n, &mut clean.clone());
            let from_dirty = shaped_c2r(n, &mut dirty);

            // Projection happened in place on the caller's buffer.
            assert_eq!(
                dirty[0].im, 0.0,
                "HermitianC2r must zero the DC imaginary in place",
            );
            if n.is_multiple_of(2) {
                let nyquist = dirty.len() - 1;
                assert_eq!(
                    dirty[nyquist].im, 0.0,
                    "HermitianC2r must zero the Nyquist imaginary in place for even N",
                );
            }
            // …and the reconstruction matches the clean-input result.
            assert_real_close(&from_dirty, &from_clean);
        }
    }

    /// `HermitianR2c` forces the output DC/(even-`N`)Nyquist imaginary parts to
    /// be exactly zero while leaving every other value untouched.
    #[test]
    fn r2c_cleans_output_dc_nyquist() {
        for n in [7_usize, 8] {
            let x = ramp(n);
            let bare = bare_r2c(n, &x);

            let factory = HermitianR2c::new(TestDftR2c);
            let spec = DftR2cSpec::<f64>::new(n, DftNorm::None).expect("valid r2c spec");
            let plan = DftR2c::<f64>::create_plan(&factory, &spec).expect("r2c plan");
            let mut scratch = x.clone();
            let mut shaped = vec![Complex::new(0.0, 0.0); n / 2 + 1];
            plan.process(&mut scratch, &mut shaped)
                .expect("r2c process");

            assert_eq!(
                shaped[0].im, 0.0,
                "HermitianR2c must zero the DC imaginary on the output",
            );
            if n.is_multiple_of(2) {
                let nyquist = shaped.len() - 1;
                assert_eq!(
                    shaped[nyquist].im, 0.0,
                    "HermitianR2c must zero the Nyquist imaginary for even N",
                );
            }
            // Real parts are unchanged everywhere; interior imaginary parts are
            // preserved (only the DC/Nyquist boundary imaginary is cleaned) — so
            // a decorator that zeroed *all* imaginaries would fail here.
            for (i, (s, b)) in shaped.iter().zip(&bare).enumerate() {
                assert!(
                    (s.re - b.re).abs() < EPS,
                    "real part changed at bin {i}: shaped {}, bare {}",
                    s.re,
                    b.re,
                );
                let is_dc = i == 0;
                let is_nyquist = n.is_multiple_of(2) && i == shaped.len() - 1;
                if !is_dc && !is_nyquist {
                    assert!(
                        (s.im - b.im).abs() < EPS,
                        "interior imaginary changed at bin {i}: shaped {}, bare {}",
                        s.im,
                        b.im,
                    );
                }
            }
        }
    }

    /// `HermitianR2c → HermitianC2r` is the identity (round-trip) under
    /// `DftNorm::Inverse`, which the shaped path composes.
    #[test]
    fn shaped_round_trip_identity() {
        for n in [1_usize, 2, 7, 8, 16] {
            let x = ramp(n);

            let r2c = HermitianR2c::new(TestDftR2c);
            let r_spec = DftR2cSpec::<f64>::new(n, DftNorm::Inverse).expect("valid r2c spec");
            let r_plan = DftR2c::<f64>::create_plan(&r2c, &r_spec).expect("r2c plan");
            let mut scratch = x.clone();
            let mut spectrum = vec![Complex::new(0.0, 0.0); n / 2 + 1];
            r_plan
                .process(&mut scratch, &mut spectrum)
                .expect("r2c process");

            let recovered = shaped_c2r(n, &mut spectrum);
            assert_real_close(&recovered, &x);
        }
    }
}
