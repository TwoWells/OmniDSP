// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Hilbert transform module — analytic signal computation.
//!
//! [`OmniHilbert`] is a factory generic over [`DftC2c`] and [`VecOps`] that creates
//! [`OmniHilbertPlan`]s from a [`HilbertSpec`].
//!
//! The Hilbert transform produces the complex analytic signal from a real input.
//! The real part of the output equals the input; the imaginary part is the
//! Hilbert transform of the input.
//!
//! # Algorithm
//!
//! 1. Forward FFT of the real input.
//! 2. Apply a spectral mask: DC and Nyquist unchanged, positive frequencies
//!    doubled, negative frequencies zeroed.
//! 3. Inverse FFT → complex analytic signal.
//!
//! Internal scratch buffers are behind a [`Mutex`] so that the plan satisfies
//! `Send + Sync` while taking `&self`.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign};
use std::sync::Mutex;

use num_complex::Complex;
use num_traits::{Float, FromPrimitive};

use crate::error::{Error, Result};
use crate::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec, DftNorm};
use crate::traits::vecops::VecOps;
use crate::types::Direction;

// ─── Spec ────────────────────────────────────────────────────────────

/// Hilbert transform specification.
///
/// Describes the signal length for the analytic signal computation.
/// The type parameter `T` ties the spec to a specific float type for
/// dispatch-layer integration.
///
/// # Examples
///
/// ```
/// use omnidsp_core::modules::hilbert::HilbertSpec;
///
/// let spec = HilbertSpec::<f64>::new(1024);
/// assert_eq!(spec.length, 1024);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HilbertSpec<T> {
    /// Signal length (and FFT length).
    pub length: usize,
    _marker: PhantomData<T>,
}

impl<T> PartialEq for HilbertSpec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length
    }
}

impl<T> Eq for HilbertSpec<T> {}

impl<T> HilbertSpec<T> {
    /// Create a new Hilbert transform spec for the given signal length.
    #[must_use]
    pub const fn new(length: usize) -> Self {
        Self {
            length,
            _marker: PhantomData,
        }
    }
}

// ─── Public types ──────────────────────────────────────────────────────

/// Generic Hilbert transform factory backed by [`DftC2c`] and [`VecOps`].
///
/// Creates [`OmniHilbertPlan`]s for specific signal lengths.  The factory
/// owns the DFT factory and `VecOps` instance; plans own their sub-plans.
#[derive(Debug, Clone)]
pub struct OmniHilbert<D, V> {
    dft: D,
    vecops: V,
}

impl<D, V> OmniHilbert<D, V> {
    /// Create a new Hilbert transform factory.
    #[must_use]
    pub const fn new(dft: D, vecops: V) -> Self {
        Self { dft, vecops }
    }
}

/// Execution plan for a Hilbert transform (analytic signal).
///
/// Created by [`OmniHilbert::create_plan`].  Immutable and thread-safe
/// (`Send + Sync`).  Each call to [`process`](Self::process) computes the
/// analytic signal of one input independently — no state between calls.
///
/// **Memory:** one buffer of `length` complex values for scratch.
pub struct OmniHilbertPlan<T, P, V> {
    /// Forward DFT plan.
    fwd: P,
    /// Inverse DFT plan.
    inv: P,
    /// Pre-computed spectral mask (real-valued, length N).
    mask: Vec<T>,
    /// `VecOps` handle for spectral mask application.
    vecops: V,
    /// Signal length.
    length: usize,
    /// Scratch buffer behind Mutex for Send + Sync.
    scratch: Mutex<HilbertScratch<T>>,
}

impl<T, P, V> fmt::Debug for OmniHilbertPlan<T, P, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniHilbertPlan")
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

// ─── Plan internals ──────────────────────────────────────────────────

struct HilbertScratch<T> {
    /// Working buffer for FFT output / masked spectrum.
    buf_spectrum: Vec<Complex<T>>,
}

// ─── Plan methods ────────────────────────────────────────────────────

impl<T, P, V> OmniHilbertPlan<T, P, V>
where
    T: Float + AddAssign + MulAssign + FromPrimitive + Send + Sync + 'static,
    P: DftC2cPlan<T>,
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
    pub fn process(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
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

        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::Internal("hilbert scratch mutex poisoned".to_owned()))?;

        // Pack real input into complex buffer (imaginary = 0).
        self.vecops
            .real_to_complex(input, &mut scratch.buf_spectrum)?;

        // Forward FFT: packed input → spectrum in output.
        self.fwd.process(&scratch.buf_spectrum, output)?;

        // Apply spectral mask in-place on output (which holds the spectrum).
        self.vecops.cscale_inplace(output, &self.mask)?;

        // Inverse FFT: masked spectrum → analytic signal.
        // Copy masked spectrum to scratch, then IFFT into output.
        scratch.buf_spectrum.copy_from_slice(output);
        self.inv.process(&scratch.buf_spectrum, output)?;
        drop(scratch);

        Ok(())
    }

    /// Signal length this plan was created for.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }
}

// ─── Factory ─────────────────────────────────────────────────────────

impl<D, V> OmniHilbert<D, V> {
    /// Create a Hilbert transform plan from a [`HilbertSpec`].
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero or DFT plan creation fails.
    pub fn create_plan<T>(&self, spec: &HilbertSpec<T>) -> Result<OmniHilbertPlan<T, D::Plan, V>>
    where
        T: Float + AddAssign + MulAssign + FromPrimitive + Send + Sync + 'static,
        D: DftC2c<T>,
        V: VecOps<T>,
    {
        if spec.length == 0 {
            return Err(Error::InvalidSpec(
                "Hilbert transform length must be non-zero".to_owned(),
            ));
        }

        let n = spec.length;

        // Create forward and inverse DFT plans.
        let fwd_spec = DftC2cSpec::new(n, Direction::Forward, DftNorm::Inverse);
        let inv_spec = DftC2cSpec::new(n, Direction::Inverse, DftNorm::Inverse);
        let fwd = self.dft.create_plan(&fwd_spec)?;
        let inv = self.dft.create_plan(&inv_spec)?;

        // Build the spectral mask.
        let two =
            T::from(2.0).ok_or_else(|| Error::Internal("cannot convert 2.0 to T".to_owned()))?;
        let one = T::one();
        let zero = T::zero();

        let mut mask = vec![zero; n];
        // DC is always passed through unchanged.
        mask[0] = one;
        if n > 1 && n.is_multiple_of(2) {
            // Even length:
            //   H[1..N/2] = 2 (positive frequencies)
            //   H[N/2] = 1 (Nyquist)
            //   H[N/2+1..N] = 0 (negative frequencies)
            for m in mask.iter_mut().take(n / 2).skip(1) {
                *m = two;
            }
            mask[n / 2] = one;
        } else if n > 1 {
            // Odd length:
            //   H[1..ceil(N/2)] = 2 (positive frequencies)
            //   H[ceil(N/2)..N] = 0 (negative frequencies)
            for m in mask.iter_mut().take(n.div_ceil(2)).skip(1) {
                *m = two;
            }
        }

        let scratch = HilbertScratch {
            buf_spectrum: vec![Complex::new(zero, zero); n],
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

    use super::{HilbertSpec, OmniHilbert};
    use crate::test_utils::{TestDftC2c, TestVecOps};

    fn make_factory() -> OmniHilbert<TestDftC2c, TestVecOps> {
        OmniHilbert::new(TestDftC2c, TestVecOps)
    }

    #[test]
    fn spec_equality() {
        let a = HilbertSpec::<f64>::new(256);
        let b = HilbertSpec::<f64>::new(256);
        let c = HilbertSpec::<f64>::new(512);
        assert_eq!(a, b, "same-length specs should be equal");
        assert_ne!(a, c, "different-length specs should differ");
    }

    #[test]
    fn zero_length_rejected() {
        let factory = make_factory();
        let spec = HilbertSpec::<f64>::new(0);
        let result = factory.create_plan(&spec);
        assert!(result.is_err(), "zero-length spec should be rejected");
    }

    #[test]
    fn length_one() {
        let factory = make_factory();
        let spec = HilbertSpec::<f64>::new(1);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input = [1.0];
        let mut output = [Complex::new(0.0, 0.0)];
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(n);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input: Vec<f64> = vec![3.0; n];
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(n);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freq = 5.0; // 5 cycles in N samples
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).cos())
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(n);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freq = 3.0;
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).cos())
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(n);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freq = 10.0;
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).cos())
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(n);
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
        plan.process(&combined, &mut out_combined)
            .expect("process should succeed");

        // a*H{x} + b*H{y}
        let mut out_x = vec![Complex::new(0.0, 0.0); n];
        let mut out_y = vec![Complex::new(0.0, 0.0); n];
        plan.process(&x, &mut out_x)
            .expect("process should succeed");
        plan.process(&y, &mut out_y)
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
        let spec = HilbertSpec::<f64>::new(n);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * 2.0 * i as f64 / n as f64).cos())
            .collect();

        let mut out1 = vec![Complex::new(0.0, 0.0); n];
        let mut out2 = vec![Complex::new(0.0, 0.0); n];
        plan.process(&input, &mut out1)
            .expect("first call should succeed");
        plan.process(&input, &mut out2)
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
        let spec = HilbertSpec::<f64>::new(16);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input = vec![0.0; 8]; // wrong length
        let mut output = vec![Complex::new(0.0, 0.0); 16];
        let result = plan.process(&input, &mut output);
        assert!(result.is_err(), "wrong input length should fail");
    }

    #[test]
    fn buffer_mismatch_output() {
        let factory = make_factory();
        let spec = HilbertSpec::<f64>::new(16);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let input = vec![0.0; 16];
        let mut output = vec![Complex::new(0.0, 0.0); 8]; // wrong length
        let result = plan.process(&input, &mut output);
        assert!(result.is_err(), "wrong output length should fail");
    }

    #[test]
    fn sine_becomes_negative_cosine() {
        // H{sin(ωt)} = -cos(ωt)
        // So analytic signal of sin should have imag = -cos.
        let factory = make_factory();
        let n = 128;
        let spec = HilbertSpec::<f64>::new(n);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let freq = 4.0;
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).sin())
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(n);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        // cos(π·n) = (-1)^n — the Nyquist frequency signal.
        let input: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(n);
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
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(n);
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
        plan.process(&input, &mut output)
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
        let spec = HilbertSpec::<f64>::new(512);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        assert_eq!(plan.length(), 512, "length accessor should match spec");
    }

    // ─── Scipy reference tests ───────────────────────────────────────

    include!(testdata!("hilbert_scipy.rs"));
}
