// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Hilbert transform module — analytic signal computation.
//!
//! [`OmniHilbert`] is a factory generic over [`DftR2c`] (forward), [`DftC2c`]
//! (inverse), and [`VecOps`] that creates [`OmniHilbertPlan`]s from a
//! [`HilbertSpec`].
//!
//! The Hilbert transform produces the complex analytic signal from a real input.
//! The real part of the output equals the input; the imaginary part is the
//! Hilbert transform of the input.
//!
//! # Algorithm
//!
//! 1. Forward [`DftR2c`] of the real input → the `N/2 + 1` half-spectrum.
//! 2. Apply the analytic half-mask: DC and (even-`N`) Nyquist unchanged,
//!    positive frequencies doubled.
//! 3. Zero-extend to the full `N` bins (negative frequencies dropped) and run a
//!    [`DftC2c`] inverse → the complex analytic signal.
//!
//! Hilbert is the one real-input module that mixes a real forward transform with
//! a *complex* inverse (its output is complex, not real), so it composes
//! [`DftR2c`] + bare [`DftC2c`] — no [`HermitianC2r`](crate::hermitian::HermitianC2r)
//! shaping is involved.
//!
//! Internal scratch buffers are behind a [`Mutex`] so that the plan satisfies
//! `Send + Sync` while taking `&self`.

use std::fmt;
use std::sync::Mutex;

use num_complex::Complex;

use crate::error::{Error, Result};
use crate::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec, DftNorm, DftR2c, DftR2cPlan, DftR2cSpec};
use crate::traits::vecops::VecOps;
use crate::types::{Direction, DspFloat};

// ─── Spec ────────────────────────────────────────────────────────────

/// Hilbert transform specification.
///
/// Describes the signal length for the analytic signal computation.  The spec
/// is non-generic; precision is chosen at `create_plan::<T>`.  The field is
/// private and the spec is valid-by-construction.
///
/// # Examples
///
/// ```
/// use omnidsp_core::modules::hilbert::HilbertSpec;
///
/// let spec = HilbertSpec::new(1024).unwrap();
/// assert_eq!(spec.length(), 1024);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HilbertSpec {
    length: usize,
}

impl HilbertSpec {
    /// Create a new Hilbert transform spec for the given signal length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn new(length: usize) -> Result<Self> {
        if length == 0 {
            return Err(Error::InvalidSpec(
                "Hilbert transform length must be non-zero".into(),
            ));
        }
        Ok(Self { length })
    }

    /// Signal length (and FFT length).
    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }
}

// ─── Public types ──────────────────────────────────────────────────────

/// Generic Hilbert transform factory backed by [`DftR2c`] (forward),
/// [`DftC2c`] (inverse), and [`VecOps`].
///
/// Creates [`OmniHilbertPlan`]s for specific signal lengths.  The factory owns
/// the forward `r2c` and inverse `c2c` factories and the `VecOps` instance;
/// plans own their sub-plans.  Unlike the other real-input modules, the inverse
/// is a bare [`DftC2c`] (the analytic output is complex) — no Hermitian shaping.
#[derive(Debug, Clone)]
pub struct OmniHilbert<R, C, V> {
    r2c: R,
    c2c: C,
    vecops: V,
}

impl<R, C, V> OmniHilbert<R, C, V> {
    /// Create a new Hilbert transform factory.
    #[must_use]
    pub const fn new(r2c: R, c2c: C, vecops: V) -> Self {
        Self { r2c, c2c, vecops }
    }
}

/// Execution plan for a Hilbert transform (analytic signal).
///
/// Created by [`OmniHilbert::create_plan`].  Immutable and thread-safe
/// (`Send + Sync`).  Each call to [`execute`](Self::execute) computes the
/// analytic signal of one input independently — no state between calls.
///
/// **Memory:** one real buffer and one complex buffer, each of `length`.
///
/// `RP` is the forward [`DftR2cPlan`]; `CP` is the inverse [`DftC2cPlan`].
/// Neither is ever named by users.
pub struct OmniHilbertPlan<T, RP, CP, V> {
    /// Forward real-to-complex DFT plan.
    fwd: RP,
    /// Inverse complex-to-complex DFT plan.
    inv: CP,
    /// Pre-computed analytic half-mask (real-valued, length `N/2 + 1`).
    mask: Vec<T>,
    /// `VecOps` handle for spectral mask application.
    vecops: V,
    /// Signal length.
    length: usize,
    /// Scratch buffer behind Mutex for Send + Sync.
    scratch: Mutex<HilbertScratch<T>>,
}

impl<T, RP, CP, V> fmt::Debug for OmniHilbertPlan<T, RP, CP, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniHilbertPlan")
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

// ─── Plan internals ──────────────────────────────────────────────────

struct HilbertScratch<T> {
    /// Real input copy (length `N`), consumed by the forward r2c.
    real: Vec<T>,
    /// Full-length analytic spectrum (length `N`): masked half-spectrum in the
    /// lower bins, zeroed negative frequencies above.
    spectrum: Vec<Complex<T>>,
}

// ─── Plan methods ────────────────────────────────────────────────────

impl<T, RP, CP, V> OmniHilbertPlan<T, RP, CP, V>
where
    T: DspFloat,
    RP: DftR2cPlan<T>,
    CP: DftC2cPlan<T>,
    V: VecOps<T>,
{
    /// Compute the analytic signal of a real input.
    ///
    /// The real part of `output` equals `input`.  The imaginary part is
    /// the Hilbert transform of `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` or `output` length does not match the
    /// plan length, or if FFT execution fails.
    pub fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
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

        let n = self.length;
        let bins = n / 2 + 1;

        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::Internal("hilbert scratch mutex poisoned".to_owned()))?;

        let HilbertScratch { real, spectrum } = &mut *scratch;

        // 1. Copy the real input into scratch (the forward r2c consumes it).
        real.copy_from_slice(input);

        // 2. Forward r2c → the lower `bins` of the spectrum buffer.
        self.fwd.execute(real, &mut spectrum[..bins])?;

        // 3. Apply the analytic half-mask in place (DC ×1, positive ×2,
        //    even-N Nyquist ×1).
        self.vecops
            .cscale_inplace(&mut spectrum[..bins], &self.mask)?;

        // 4. Zero the upper half (negative frequencies dropped by the mask).
        for c in &mut spectrum[bins..] {
            *c = Complex::new(T::zero(), T::zero());
        }

        // 5. Inverse c2c: analytic spectrum → complex analytic signal.
        self.inv.execute(spectrum, output)?;
        drop(scratch);

        Ok(())
    }

    /// Signal length this plan was created for.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }
}

// ─── Plan trait ──────────────────────────────────────────────────────

/// Execution object for a configured Hilbert transform.
///
/// The named, `Send + Sync` plan trait for the Hilbert module — the analytic
/// counterpart to the eager plan traits
/// [`ConvPlan`](crate::traits::conv::ConvPlan) /
/// [`DctPlan`](crate::traits::dct::DctPlan).  It lets the analytic-signal
/// `process` be called generically (e.g. by the shared conformance suite)
/// without naming the concrete plan.
/// Implemented by [`OmniHilbertPlan`]; vendor overrides may implement it
/// directly.
pub trait HilbertPlan<T>: Send + Sync {
    /// Compute the analytic signal of a real `input`, writing the complex
    /// result to `output`.
    ///
    /// Both buffers must have the length the plan was created for.  The real
    /// part of `output` equals `input`; the imaginary part is the Hilbert
    /// transform of `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if either buffer length does not match the plan length,
    /// or if FFT execution fails.
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()>;
}

impl<T, RP, CP, V> HilbertPlan<T> for OmniHilbertPlan<T, RP, CP, V>
where
    T: DspFloat,
    RP: DftR2cPlan<T>,
    CP: DftC2cPlan<T>,
    V: VecOps<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
        // Delegate to the inherent `execute`.  Inherent methods take precedence
        // over trait methods in resolution, so this is not recursive.
        self.execute(input, output)
    }
}

// ─── Factory ─────────────────────────────────────────────────────────

impl<R, C, V> OmniHilbert<R, C, V> {
    /// Create a Hilbert transform plan from a [`HilbertSpec`].
    ///
    /// # Errors
    ///
    /// Returns an error if DFT plan creation fails.  The length invariant is
    /// enforced by [`HilbertSpec::new`], so it is not re-checked here.
    pub fn create_plan<T>(
        &self,
        spec: &HilbertSpec,
    ) -> Result<OmniHilbertPlan<T, R::Plan, C::Plan, V>>
    where
        T: DspFloat,
        R: DftR2c<T>,
        C: DftC2c<T>,
        V: VecOps<T>,
    {
        let n = spec.length();
        let bins = n / 2 + 1;

        // Forward r2c, inverse c2c.  DftNorm::Inverse gives the round-trip
        // identity (1/N on the inverse only).
        let fwd_spec = DftR2cSpec::new(n, DftNorm::Inverse)?;
        let fwd = self.r2c.create_plan(&fwd_spec)?;
        let inv_spec = DftC2cSpec::new(n, Direction::Inverse, DftNorm::Inverse)?;
        let inv = self.c2c.create_plan(&inv_spec)?;

        // Build the analytic *half*-mask (length `N/2 + 1`): the negative
        // frequencies are dropped by zero-extension, so only the lower bins
        // carry a factor.
        let two =
            T::from(2.0).ok_or_else(|| Error::Internal("cannot convert 2.0 to T".to_owned()))?;
        let one = T::one();
        let zero = T::zero();

        let mut mask = vec![zero; bins];
        // DC is always passed through unchanged.
        mask[0] = one;
        if n > 1 && n.is_multiple_of(2) {
            // Even length:
            //   H[1..N/2] = 2 (positive frequencies)
            //   H[N/2] = 1 (Nyquist)
            for m in mask.iter_mut().take(n / 2).skip(1) {
                *m = two;
            }
            mask[n / 2] = one;
        } else if n > 1 {
            // Odd length: every non-DC half-spectrum bin is a positive
            //   frequency → H[1..bins] = 2 (no Nyquist bin).
            for m in mask.iter_mut().skip(1) {
                *m = two;
            }
        }

        let scratch = HilbertScratch {
            real: vec![zero; n],
            spectrum: vec![Complex::new(zero, zero); n],
        };

        Ok(OmniHilbertPlan {
            fwd,
            inv,
            mask,
            vecops: self.vecops.clone(),
            length: n,
            scratch: Mutex::new(scratch),
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
#[allow(
    clippy::cast_precision_loss,
    reason = "tests use small N where usize→f64 is exact"
)]
#[allow(
    clippy::many_single_char_names,
    reason = "mathematical variable names are clearer in signal processing tests"
)]
#[allow(clippy::suboptimal_flops, reason = "clarity over performance in tests")]
mod tests {
    use std::f64::consts::TAU;

    use num_complex::Complex;

    use super::{HilbertPlan, HilbertSpec, OmniHilbert};
    use crate::test_utils::{TestDftC2c, TestDftR2c, TestVecOps};

    fn make_factory() -> OmniHilbert<TestDftR2c, TestDftC2c, TestVecOps> {
        OmniHilbert::new(TestDftR2c, TestDftC2c, TestVecOps)
    }

    #[test]
    fn spec_equality() {
        let a = HilbertSpec::new(256).expect("valid hilbert spec");
        let b = HilbertSpec::new(256).expect("valid hilbert spec");
        let c = HilbertSpec::new(512).expect("valid hilbert spec");
        assert_eq!(a, b, "same-length specs should be equal");
        assert_ne!(a, c, "different-length specs should differ");
    }

    #[test]
    fn zero_length_rejected() {
        assert!(
            HilbertSpec::new(0).is_err(),
            "zero-length spec should be rejected by the spec constructor"
        );
    }

    #[test]
    fn length_one() {
        let factory = make_factory();
        let spec = HilbertSpec::new(1).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input = [1.0];
        let mut output = [Complex::new(0.0, 0.0)];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        // Length-1 signal: analytic signal is just the input with zero imaginary.
        assert!(
            (output[0].re - 1.0).abs() < 1e-12,
            "real part should equal input"
        );
        assert!(
            output[0].im.abs() < 1e-12,
            "imaginary part should be zero for length-1"
        );
    }

    #[test]
    fn dc_signal_has_zero_imaginary() {
        let factory = make_factory();
        let n = 64;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input: Vec<f64> = vec![3.0; n];
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        for (i, z) in output.iter().enumerate() {
            assert!(
                (z.re - 3.0).abs() < 1e-10,
                "real part should equal input at index {i}"
            );
            assert!(
                z.im.abs() < 1e-10,
                "imaginary part should be zero for DC at index {i}"
            );
        }
    }

    #[test]
    fn pure_cosine_even_length() {
        // Analytic signal of cos(2πf·n/N) = cos(...) + j·sin(...)
        // for a single positive frequency.
        let factory = make_factory();
        let n = 128;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freq = 5.0; // 5 cycles in N samples
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).cos())
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        // Verify: real part ≈ cos, imaginary part ≈ sin
        let tol = 1e-10;
        for (i, z) in output.iter().enumerate() {
            let expected_re = (TAU * freq * i as f64 / n as f64).cos();
            let expected_im = (TAU * freq * i as f64 / n as f64).sin();
            assert!(
                (z.re - expected_re).abs() < tol,
                "real mismatch at {i}: got {}, expected {expected_re}",
                z.re
            );
            assert!(
                (z.im - expected_im).abs() < tol,
                "imag mismatch at {i}: got {}, expected {expected_im}",
                z.im
            );
        }
    }

    #[test]
    fn pure_cosine_odd_length() {
        let factory = make_factory();
        let n = 127;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freq = 3.0;
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).cos())
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        let tol = 1e-10;
        for (i, z) in output.iter().enumerate() {
            let expected_re = (TAU * freq * i as f64 / n as f64).cos();
            let expected_im = (TAU * freq * i as f64 / n as f64).sin();
            assert!(
                (z.re - expected_re).abs() < tol,
                "real mismatch at {i}: got {}, expected {expected_re}",
                z.re
            );
            assert!(
                (z.im - expected_im).abs() < tol,
                "imag mismatch at {i}: got {}, expected {expected_im}",
                z.im
            );
        }
    }

    #[test]
    fn envelope_constant_for_pure_tone() {
        // |z[n]|² should be constant for a pure cosine.
        let factory = make_factory();
        let n = 256;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freq = 10.0;
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).cos())
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        // Envelope should be 1.0 everywhere for unit-amplitude cosine.
        let tol = 1e-10;
        for (i, z) in output.iter().enumerate() {
            let envelope = z.re.hypot(z.im);
            assert!(
                (envelope - 1.0).abs() < tol,
                "envelope should be 1.0 at index {i}, got {envelope}"
            );
        }
    }

    #[test]
    fn linearity() {
        let factory = make_factory();
        let n = 64;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let a = 2.5;
        let b = -1.3;
        let x: Vec<f64> = (0..n)
            .map(|i| (TAU * 3.0 * i as f64 / n as f64).cos())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (TAU * 7.0 * i as f64 / n as f64).sin())
            .collect();

        // H{a*x + b*y}
        let combined: Vec<f64> = x.iter().zip(&y).map(|(&xi, &yi)| a * xi + b * yi).collect();
        let mut out_combined = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&combined, &mut out_combined)
            .expect("process should succeed");

        // a*H{x} + b*H{y}
        let mut out_x = vec![Complex::new(0.0, 0.0); n];
        let mut out_y = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&x, &mut out_x)
            .expect("process should succeed");
        plan.execute(&y, &mut out_y)
            .expect("process should succeed");

        let tol = 1e-10;
        for i in 0..n {
            let expected = Complex::new(
                a * out_x[i].re + b * out_y[i].re,
                a * out_x[i].im + b * out_y[i].im,
            );
            assert!(
                (out_combined[i].re - expected.re).abs() < tol,
                "linearity: real mismatch at {i}"
            );
            assert!(
                (out_combined[i].im - expected.im).abs() < tol,
                "linearity: imag mismatch at {i}"
            );
        }
    }

    #[test]
    fn plan_reuse() {
        let factory = make_factory();
        let n = 32;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * 2.0 * i as f64 / n as f64).cos())
            .collect();

        let mut out1 = vec![Complex::new(0.0, 0.0); n];
        let mut out2 = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut out1)
            .expect("first call should succeed");
        plan.execute(&input, &mut out2)
            .expect("second call should succeed");

        for i in 0..n {
            assert!(
                (out1[i].re - out2[i].re).abs() < 1e-15,
                "plan reuse: real mismatch at {i}"
            );
            assert!(
                (out1[i].im - out2[i].im).abs() < 1e-15,
                "plan reuse: imag mismatch at {i}"
            );
        }
    }

    #[test]
    fn buffer_mismatch_input() {
        let factory = make_factory();
        let spec = HilbertSpec::new(16).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input = vec![0.0; 8]; // wrong length
        let mut output = vec![Complex::new(0.0, 0.0); 16];
        let result = plan.execute(&input, &mut output);
        assert!(result.is_err(), "wrong input length should fail");
    }

    #[test]
    fn buffer_mismatch_output() {
        let factory = make_factory();
        let spec = HilbertSpec::new(16).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input = vec![0.0; 16];
        let mut output = vec![Complex::new(0.0, 0.0); 8]; // wrong length
        let result = plan.execute(&input, &mut output);
        assert!(result.is_err(), "wrong output length should fail");
    }

    #[test]
    fn sine_becomes_negative_cosine() {
        // H{sin(ωt)} = -cos(ωt)
        // So analytic signal of sin should have imag = -cos.
        let factory = make_factory();
        let n = 128;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freq = 4.0;
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).sin())
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        let tol = 1e-10;
        for (i, z) in output.iter().enumerate() {
            let expected_re = (TAU * freq * i as f64 / n as f64).sin();
            let expected_im = -(TAU * freq * i as f64 / n as f64).cos();
            assert!(
                (z.re - expected_re).abs() < tol,
                "sin real mismatch at {i}: got {}, expected {expected_re}",
                z.re
            );
            assert!(
                (z.im - expected_im).abs() < tol,
                "sin imag mismatch at {i}: got {}, expected {expected_im}",
                z.im
            );
        }
    }

    #[test]
    fn nyquist_component_preserved_even() {
        // For even N, a signal at Nyquist (f = N/2) should pass through
        // unchanged (mask = 1 at Nyquist).
        let factory = make_factory();
        let n = 64;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        // cos(π·n) = (-1)^n — the Nyquist frequency signal.
        let input: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        // The Nyquist signal is real-only in spectrum, mask=1, so analytic
        // signal real part = input, imag part = 0.
        let tol = 1e-10;
        for (i, z) in output.iter().enumerate() {
            assert!(
                (z.re - input[i]).abs() < tol,
                "nyquist real mismatch at {i}"
            );
            assert!(
                z.im.abs() < tol,
                "nyquist imag should be zero at {i}, got {}",
                z.im
            );
        }
    }

    #[test]
    fn multi_frequency_sum() {
        // Sum of cosines at different frequencies — each becomes its
        // own complex exponential in the analytic signal.
        let factory = make_factory();
        let n = 256;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freqs = [3.0, 7.0, 15.0];
        let amps = [1.0, 0.5, 0.25];

        let input: Vec<f64> = (0..n)
            .map(|i| {
                freqs
                    .iter()
                    .zip(&amps)
                    .map(|(&f, &a)| a * (TAU * f * i as f64 / n as f64).cos())
                    .sum()
            })
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        let tol = 1e-10;
        for (i, z) in output.iter().enumerate() {
            let expected_re: f64 = freqs
                .iter()
                .zip(&amps)
                .map(|(&f, &a)| a * (TAU * f * i as f64 / n as f64).cos())
                .sum();
            let expected_im: f64 = freqs
                .iter()
                .zip(&amps)
                .map(|(&f, &a)| a * (TAU * f * i as f64 / n as f64).sin())
                .sum();
            assert!(
                (z.re - expected_re).abs() < tol,
                "multi-freq real mismatch at {i}"
            );
            assert!(
                (z.im - expected_im).abs() < tol,
                "multi-freq imag mismatch at {i}"
            );
        }
    }

    #[test]
    fn real_part_equals_input() {
        // For any input, real part of analytic signal must equal the input.
        let factory = make_factory();
        let n = 64;
        let spec = HilbertSpec::new(n).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        // Arbitrary non-trivial signal.
        let input: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (TAU * 5.0 * t).sin() + 0.3 * (TAU * 13.0 * t).cos() + 0.7
            })
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.execute(&input, &mut output)
            .expect("process should succeed");

        let tol = 1e-10;
        for (i, z) in output.iter().enumerate() {
            assert!(
                (z.re - input[i]).abs() < tol,
                "real part must equal input at index {i}: got {}, expected {}",
                z.re,
                input[i]
            );
        }
    }

    #[test]
    fn accessor_length() {
        let factory = make_factory();
        let spec = HilbertSpec::new(512).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        assert_eq!(plan.length(), 512, "length accessor should match spec");
    }

    #[test]
    fn implements_plan_trait() {
        // A generic consumer bound on `HilbertPlan` compiles and runs.
        fn check<P: HilbertPlan<f64>>(plan: &P, input: &[f64], output: &mut [Complex<f64>]) {
            plan.execute(input, output)
                .expect("trait process should succeed");
        }

        let factory = make_factory();
        let spec = HilbertSpec::new(16).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input: Vec<f64> = (0_i32..16).map(f64::from).collect();
        let mut via_trait = vec![Complex::new(0.0, 0.0); 16];
        check(&plan, &input, &mut via_trait);

        // The trait method must agree with the inherent `process`.
        let mut direct = vec![Complex::new(0.0, 0.0); 16];
        plan.execute(&input, &mut direct)
            .expect("inherent process should succeed");
        for (i, (t, d)) in via_trait.iter().zip(direct.iter()).enumerate() {
            assert!(
                (t - d).norm() < 1e-15,
                "trait vs inherent mismatch at index {i}"
            );
        }
    }

    // ─── Scipy reference tests ───────────────────────────────────────

    use omnidsp_testdata::hilbert_scipy::{
        HAND_N4_EXPECTED, HAND_N4_INPUT, HAND_N8_COS2_EXPECTED, HAND_N8_COS2_INPUT,
    };

    #[test]
    fn hand_computed_n4() {
        let factory = make_factory();
        let spec = HilbertSpec::new(HAND_N4_INPUT.len()).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut output = vec![Complex::new(0.0, 0.0); HAND_N4_INPUT.len()];
        plan.execute(HAND_N4_INPUT, &mut output)
            .expect("process should succeed");

        let tol = 1e-12;
        for (i, (z, &(re, im))) in output.iter().zip(HAND_N4_EXPECTED).enumerate() {
            assert!(
                (z.re - re).abs() < tol,
                "n4 real mismatch at {i}: got {}, expected {re}",
                z.re
            );
            assert!(
                (z.im - im).abs() < tol,
                "n4 imag mismatch at {i}: got {}, expected {im}",
                z.im
            );
        }
    }

    #[test]
    fn hand_computed_n8_cos2() {
        let factory = make_factory();
        let spec = HilbertSpec::new(HAND_N8_COS2_INPUT.len()).expect("valid hilbert spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut output = vec![Complex::new(0.0, 0.0); HAND_N8_COS2_INPUT.len()];
        plan.execute(HAND_N8_COS2_INPUT, &mut output)
            .expect("process should succeed");

        let tol = 1e-12;
        for (i, (z, &(re, im))) in output.iter().zip(HAND_N8_COS2_EXPECTED).enumerate() {
            assert!(
                (z.re - re).abs() < tol,
                "n8_cos2 real mismatch at {i}: got {}, expected {re}",
                z.re
            );
            assert!(
                (z.im - im).abs() < tol,
                "n8_cos2 imag mismatch at {i}: got {}, expected {im}",
                z.im
            );
        }
    }
}
