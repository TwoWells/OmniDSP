// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Constant-Q Transform module — log-frequency spectral analysis.
//!
//! [`OmniCqt`] is a factory generic over [`Dft`] and [`VecOps`] that creates
//! [`OmniCqtPlan`]s from a [`CqtSpec`].
//!
//! The CQT computes log-frequency spectral coefficients using pre-FFT'd
//! conjugate kernels (one per bin).  Each call to [`OmniCqtPlan::process`]
//! takes a single real-valued frame and produces one complex coefficient per
//! bin.  The caller is responsible for framing (hop + overlap), just like an
//! FFT-based spectrogram.
//!
//! # Kernel time-axis convention
//!
//! Kernels are placed starting at index 0 (causal):
//!
//! ```text
//! kernel[n] = (1/N_k) · window[n] · exp(j·2π·f_k·n / sr),  n = 0 … N_k−1
//! ```
//!
//! This differs from librosa, which centers the kernel at `(N_k−1)/2`.
//! For hop-based multi-frame processing both conventions produce the same
//! spectrogram content — the only difference is a half-kernel time offset
//! in which sample each frame's coefficient is attributed to (analogous
//! to `center=True` vs `center=False` in an STFT).  Individual-frame
//! magnitudes differ by ≈2e-4 for single tones due to the phase rotation
//! landing inside the frequency-domain summation.
//!
//! Internal scratch buffers are behind a [`Mutex`] so that the plan satisfies
//! `Send + Sync` on `&self`.  The lock is uncontended in the common
//! single-threaded case.

use std::f64::consts::TAU;
use std::fmt;
use std::ops::{AddAssign, MulAssign};
use std::sync::Mutex;

use num_complex::Complex;
use num_traits::Float;

use crate::design::cqt::CqtSpec;
use crate::error::{Error, Result};
use crate::traits::dft::{Dft, DftNorm, DftPlan, DftSpec};
use crate::traits::vecops::VecOps;
use crate::types::Direction;

// ─── Public types ──────────────────────────────────────────────────────

/// Generic CQT factory backed by [`Dft`] and [`VecOps`].
///
/// Creates [`OmniCqtPlan`]s for specific CQT configurations.  The factory
/// owns the DFT factory and `VecOps` instance; plans own their sub-plans.
#[derive(Debug, Clone)]
pub struct OmniCqt<D, V> {
    dft: D,
    vecops: V,
}

impl<D, V> OmniCqt<D, V> {
    /// Create a new CQT factory.
    #[must_use]
    pub const fn new(dft: D, vecops: V) -> Self {
        Self { dft, vecops }
    }
}

/// Execution plan for a Constant-Q Transform.
///
/// Created by [`OmniCqt::create_plan`].  Immutable and thread-safe
/// (`Send + Sync`).  Each call to [`process`](Self::process) computes the
/// CQT of one frame independently — no state is carried between calls.
///
/// Memory usage: `num_bins × fft_length` complex values for the stored
/// kernels.  For a piano-range CQT (168 bins, FFT length ~16k), this is
/// approximately 20 MB at `f64`.
///
/// Uses [`VecOps`] for the per-bin complex multiply.
pub struct OmniCqtPlan<T, P, V> {
    /// Pre-computed frequency-domain conjugate kernels, scaled by 1/N.
    kernels: Vec<Vec<Complex<T>>>,
    /// Forward DFT plan.
    fwd: P,
    /// `VecOps` handle.
    vecops: V,
    /// FFT length.
    fft_length: usize,
    /// Hop length (recommended advance between frames).
    hop_length: usize,
    /// Per-bin center frequencies in Hz.
    bin_frequencies: Vec<f64>,
    /// Scratch buffers behind Mutex for Send + Sync.
    scratch: Mutex<CqtScratch<T>>,
}

impl<T, P, V> fmt::Debug for OmniCqtPlan<T, P, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniCqtPlan")
            .field("num_bins", &self.kernels.len())
            .field("fft_length", &self.fft_length)
            .field("hop_length", &self.hop_length)
            .finish_non_exhaustive()
    }
}

// ─── Plan internals ───────────────────────────────────────────────────

struct CqtScratch<T> {
    fft: FftScratch<T>,
    /// Scratch for [`OmniCqtPlan::process_magnitude`] complex coefficients.
    buf_coeffs: Vec<Complex<T>>,
}

#[allow(
    clippy::struct_field_names,
    reason = "buf_ prefix clarifies these are reusable buffers"
)]
struct FftScratch<T> {
    /// Complex input buffer for FFT.
    buf_input: Vec<Complex<T>>,
    /// FFT output (frequency-domain representation of the frame).
    buf_spectrum: Vec<Complex<T>>,
    /// Per-bin complex multiply result.
    buf_product: Vec<Complex<T>>,
}

// ─── Plan methods ─────────────────────────────────────────────────────

impl<T, P, V> OmniCqtPlan<T, P, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    P: DftPlan<T>,
    V: VecOps<T>,
{
    /// Compute the CQT of one frame.
    ///
    /// `input` must have length [`fft_length`](Self::fft_length).
    /// `output` must have length [`num_bins`](Self::num_bins).
    /// Each output element is the complex CQT coefficient for that bin.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if buffer lengths are wrong.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire processing pipeline"
    )]
    pub fn process(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
        if input.len() != self.fft_length {
            return Err(Error::BufferMismatch {
                expected: self.fft_length,
                actual: input.len(),
            });
        }
        let num_bins = self.kernels.len();
        if output.len() != num_bins {
            return Err(Error::BufferMismatch {
                expected: num_bins,
                actual: output.len(),
            });
        }

        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;

        Self::process_inner(
            &self.fwd,
            &self.vecops,
            &self.kernels,
            input,
            output,
            &mut guard.fft,
        )
    }

    /// Compute the CQT magnitude spectrum of one frame.
    ///
    /// Convenience wrapper around [`process`](Self::process) that writes
    /// `|CQT[k]|` to `output` instead of complex coefficients.
    ///
    /// `input` must have length [`fft_length`](Self::fft_length).
    /// `output` must have length [`num_bins`](Self::num_bins).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if buffer lengths are wrong.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire processing pipeline"
    )]
    pub fn process_magnitude(&self, input: &[T], output: &mut [T]) -> Result<()> {
        if input.len() != self.fft_length {
            return Err(Error::BufferMismatch {
                expected: self.fft_length,
                actual: input.len(),
            });
        }
        let num_bins = self.kernels.len();
        if output.len() != num_bins {
            return Err(Error::BufferMismatch {
                expected: num_bins,
                actual: output.len(),
            });
        }

        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;

        let CqtScratch { fft, buf_coeffs } = &mut *guard;

        Self::process_inner(
            &self.fwd,
            &self.vecops,
            &self.kernels,
            input,
            buf_coeffs,
            fft,
        )?;
        for (mag, c) in output.iter_mut().zip(buf_coeffs.iter()) {
            *mag = (c.re * c.re + c.im * c.im).sqrt();
        }
        Ok(())
    }

    /// Core processing: FFT input, then per-bin cmul + sum.
    fn process_inner(
        fwd: &P,
        vecops: &V,
        kernels: &[Vec<Complex<T>>],
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut FftScratch<T>,
    ) -> Result<()> {
        let FftScratch {
            buf_input,
            buf_spectrum,
            buf_product,
        } = scratch;

        // 1. Convert real input to complex and forward-FFT.
        for (c, &r) in buf_input.iter_mut().zip(input) {
            *c = Complex::new(r, T::zero());
        }
        fwd.process(buf_input, buf_spectrum)?;

        // 2. For each bin: complex multiply spectrum × conjugate kernel,
        //    then sum the products.  The stored kernels already include
        //    the 1/N scaling so the sum directly equals the time-domain
        //    correlation.
        for (k, kernel) in kernels.iter().enumerate() {
            vecops.cmul(buf_spectrum, kernel, buf_product)?;
            let mut sum = Complex::new(T::zero(), T::zero());
            for c in buf_product.iter() {
                sum.re += c.re;
                sum.im += c.im;
            }
            output[k] = sum;
        }

        Ok(())
    }

    /// Number of frequency bins.
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        self.kernels.len()
    }

    /// FFT length used internally.
    #[must_use]
    pub const fn fft_length(&self) -> usize {
        self.fft_length
    }

    /// Hop length (recommended advance between frames).
    #[must_use]
    pub const fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Per-bin center frequencies in Hz.
    #[must_use]
    pub fn bin_frequencies(&self) -> &[f64] {
        &self.bin_frequencies
    }
}

// ─── Factory ──────────────────────────────────────────────────────────

impl<D, V> OmniCqt<D, V> {
    /// Create a CQT plan from a [`CqtSpec`].
    ///
    /// The spec provides per-bin center frequencies and window coefficients.
    /// Construct one via [`design`](crate::design::cqt::design) or
    /// [`CqtSpec::new`] for full control.
    ///
    /// # Errors
    ///
    /// Returns an error if DFT plan creation fails.
    #[allow(
        clippy::cast_precision_loss,
        reason = "kernel lengths and sample indices are small enough for f64"
    )]
    pub fn create_plan<T>(&self, spec: &CqtSpec<T>) -> Result<OmniCqtPlan<T, D::Plan, V>>
    where
        T: Float + AddAssign + MulAssign + Send + Sync,
        D: Dft<T>,
        V: VecOps<T>,
    {
        let fft_length = spec.fft_length();
        let fwd_spec = DftSpec::new(fft_length, Direction::Forward, DftNorm::None);
        let fwd = self.dft.create_plan(&fwd_spec)?;

        let inv_n = T::from(fft_length)
            .ok_or_else(|| Error::Internal("cannot convert FFT length to T".into()))?
            .recip();

        let zero = Complex::new(T::zero(), T::zero());

        // Scratch for kernel FFT during construction.
        let mut kern_time = vec![zero; fft_length];
        let mut kern_freq = vec![zero; fft_length];

        let bins = spec.bins();
        let mut kernels = Vec::with_capacity(bins.len());
        let mut bin_frequencies = Vec::with_capacity(bins.len());

        for bin in bins {
            let coeffs = &bin.window;
            let n_k = coeffs.len();
            bin_frequencies.push(bin.frequency);

            let inv_nk = T::from(n_k)
                .ok_or_else(|| Error::Internal("cannot convert kernel length to T".into()))?
                .recip();

            // Build time-domain kernel at index 0 (causal convention):
            //   kernel[n] = (1/N_k) * window[n] * exp(j·2π·f_k·n / sr)
            // and zero-pad to fft_length.
            // See module-level docs for why this differs from librosa's
            // centered placement and why the difference is benign.
            kern_time.fill(zero);
            for (n, &w) in coeffs.iter().enumerate() {
                let angle = TAU * bin.frequency * n as f64 / spec.sample_rate();
                let cos_val = T::from(angle.cos())
                    .ok_or_else(|| Error::Internal("cannot convert cos to T".into()))?;
                let sin_val = T::from(angle.sin())
                    .ok_or_else(|| Error::Internal("cannot convert sin to T".into()))?;
                let scaled = w * inv_nk;
                kern_time[n] = Complex::new(scaled * cos_val, scaled * sin_val);
            }

            // FFT the kernel → frequency domain.
            fwd.process(&kern_time, &mut kern_freq)?;

            // Conjugate and scale by 1/N so that the frequency-domain
            // dot product equals the time-domain correlation directly.
            let kernel: Vec<Complex<T>> = kern_freq
                .iter()
                .map(|c| Complex::new(c.re * inv_n, -(c.im * inv_n)))
                .collect();

            kernels.push(kernel);
        }

        let num_bins = kernels.len();
        let scratch = CqtScratch {
            fft: FftScratch {
                buf_input: vec![zero; fft_length],
                buf_spectrum: vec![zero; fft_length],
                buf_product: vec![zero; fft_length],
            },
            buf_coeffs: vec![zero; num_bins],
        };

        Ok(OmniCqtPlan {
            kernels,
            fwd,
            vecops: self.vecops.clone(),
            fft_length,
            hop_length: spec.hop_length(),
            bin_frequencies,
            scratch: Mutex::new(scratch),
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use std::f64::consts::TAU;

    use num_complex::Complex;

    use super::*;
    use crate::design::cqt;
    use crate::test_utils::{TestDft, TestVecOps};
    use crate::types::Window;

    /// Small CQT spec for fast tests: one octave, 12 bins/octave.
    fn small_spec() -> CqtSpec<f64> {
        cqt::design(44100.0, 440.0, 880.0, 12, &Window::Hann).expect("valid design")
    }

    /// Helper: create a CQT plan for the given spec.
    fn make_plan(spec: &CqtSpec<f64>) -> OmniCqtPlan<f64, <TestDft as Dft<f64>>::Plan, TestVecOps> {
        let factory = OmniCqt::new(TestDft, TestVecOps);
        factory
            .create_plan(spec)
            .expect("plan creation should succeed")
    }

    // ── Plan accessors ────────────────────────────────────────────

    #[test]
    fn num_bins_matches_spec() {
        let spec = small_spec();
        let plan = make_plan(&spec);

        assert_eq!(
            plan.num_bins(),
            spec.num_bins(),
            "num_bins should match spec"
        );
    }

    #[test]
    fn fft_length_matches_spec() {
        let spec = small_spec();
        let plan = make_plan(&spec);

        assert_eq!(
            plan.fft_length(),
            spec.fft_length(),
            "fft_length should match spec"
        );
        assert!(
            plan.fft_length().is_power_of_two(),
            "fft_length should be a power of two"
        );
    }

    #[test]
    fn fft_length_ge_longest_kernel() {
        let spec = small_spec();
        let plan = make_plan(&spec);

        let max_kernel = spec
            .bins()
            .iter()
            .map(|b| b.window.len())
            .max()
            .expect("has bins");
        assert!(
            plan.fft_length() >= max_kernel,
            "fft_length {} should be >= longest kernel {}",
            plan.fft_length(),
            max_kernel
        );
    }

    #[test]
    fn hop_length_matches_spec() {
        let spec = small_spec();
        let plan = make_plan(&spec);

        assert_eq!(
            plan.hop_length(),
            spec.hop_length(),
            "hop_length should match spec"
        );
    }

    #[test]
    fn bin_frequencies_match_spec() {
        let spec = small_spec();
        let plan = make_plan(&spec);

        assert_eq!(
            plan.bin_frequencies().len(),
            spec.num_bins(),
            "bin_frequencies length should match spec"
        );
        for (i, (actual, expected)) in plan
            .bin_frequencies()
            .iter()
            .zip(spec.bins().iter().map(|b| b.frequency))
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-10,
                "bin {i} frequency mismatch: expected {expected}, got {actual}"
            );
        }
    }

    // ── Buffer validation ─────────────────────────────────────────

    #[test]
    fn process_rejects_wrong_input_length() {
        let plan = make_plan(&small_spec());
        let input = vec![0.0_f64; plan.fft_length() + 1];
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];

        assert!(
            plan.process(&input, &mut output).is_err(),
            "should reject wrong input length"
        );
    }

    #[test]
    fn process_rejects_wrong_output_length() {
        let plan = make_plan(&small_spec());
        let input = vec![0.0_f64; plan.fft_length()];
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins() + 1];

        assert!(
            plan.process(&input, &mut output).is_err(),
            "should reject wrong output length"
        );
    }

    // ── Zero input ────────────────────────────────────────────────

    #[test]
    fn zero_input_produces_zero_output() {
        let plan = make_plan(&small_spec());
        let input = vec![0.0_f64; plan.fft_length()];
        let mut output = vec![Complex::new(999.0, 999.0); plan.num_bins()];

        plan.process(&input, &mut output)
            .expect("process should succeed");

        for (k, c) in output.iter().enumerate() {
            assert!(
                c.norm() < 1e-10,
                "bin {k} should be zero for zero input, got magnitude {}",
                c.norm()
            );
        }
    }

    // ── Pure tone detection ───────────────────────────────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "FFT length is small enough for f64"
    )]
    fn pure_tone_peaks_at_correct_bin() {
        let spec = small_spec();
        let plan = make_plan(&spec);
        let n = plan.fft_length();

        // Pick a bin in the middle of the range.
        let target_bin = plan.num_bins() / 2;
        let target_freq = plan.bin_frequencies()[target_bin];

        // Generate a pure sine at the bin's center frequency.
        let sr = spec.sample_rate();
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * target_freq * i as f64 / sr).sin())
            .collect();

        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        let magnitudes: Vec<f64> = output.iter().map(|c| c.norm()).collect();
        let peak_mag = magnitudes[target_bin];

        // The target bin should have the largest magnitude.
        for (k, &mag) in magnitudes.iter().enumerate() {
            if k != target_bin {
                assert!(
                    peak_mag > mag,
                    "target bin {target_bin} (mag {peak_mag:.6}) should exceed \
                     bin {k} (mag {mag:.6}) for a pure tone at {target_freq:.1} Hz"
                );
            }
        }
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "FFT length is small enough for f64"
    )]
    fn pure_tone_at_first_bin() {
        let spec = small_spec();
        let plan = make_plan(&spec);
        let n = plan.fft_length();

        let target_freq = plan.bin_frequencies()[0];
        let sr = spec.sample_rate();
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * target_freq * i as f64 / sr).sin())
            .collect();

        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        let magnitudes: Vec<f64> = output.iter().map(|c| c.norm()).collect();
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("no NaN"))
            .expect("at least one bin")
            .0;

        assert_eq!(
            peak_bin, 0,
            "peak should be at bin 0 for a tone at {target_freq:.1} Hz, \
             but was at bin {peak_bin}"
        );
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "FFT length is small enough for f64"
    )]
    fn pure_tone_at_last_bin() {
        let spec = small_spec();
        let plan = make_plan(&spec);
        let n = plan.fft_length();
        let last = plan.num_bins() - 1;

        let target_freq = plan.bin_frequencies()[last];
        let sr = spec.sample_rate();
        let input: Vec<f64> = (0..n)
            .map(|i| (TAU * target_freq * i as f64 / sr).sin())
            .collect();

        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        let magnitudes: Vec<f64> = output.iter().map(|c| c.norm()).collect();
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("no NaN"))
            .expect("at least one bin")
            .0;

        assert_eq!(
            peak_bin, last,
            "peak should be at last bin for a tone at {target_freq:.1} Hz, \
             but was at bin {peak_bin}"
        );
    }

    // ── Plan reuse ────────────────────────────────────────────────

    #[test]
    fn plan_reuse_produces_consistent_results() {
        let plan = make_plan(&small_spec());
        let n = plan.fft_length();

        // Non-trivial input (impulse at sample 0).
        let mut input = vec![0.0_f64; n];
        input[0] = 1.0;

        let mut out1 = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        let mut out2 = vec![Complex::new(0.0, 0.0); plan.num_bins()];

        plan.process(&input, &mut out1)
            .expect("first process should succeed");
        plan.process(&input, &mut out2)
            .expect("second process should succeed");

        for (k, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).norm() < 1e-10,
                "bin {k} differs between calls: {a} vs {b}"
            );
        }
    }

    // ── process_magnitude ─────────────────────────────────────────

    #[test]
    fn process_magnitude_matches_manual() {
        let plan = make_plan(&small_spec());
        let n = plan.fft_length();

        let mut input = vec![0.0_f64; n];
        input[0] = 1.0;

        let mut complex_out = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut complex_out)
            .expect("process should succeed");

        let mut mag_out = vec![0.0_f64; plan.num_bins()];
        plan.process_magnitude(&input, &mut mag_out)
            .expect("process_magnitude should succeed");

        for (k, (&mag, c)) in mag_out.iter().zip(complex_out.iter()).enumerate() {
            let expected = c.norm();
            assert!(
                (mag - expected).abs() < 1e-10,
                "bin {k} magnitude mismatch: expected {expected}, got {mag}"
            );
        }
    }

    // ── Debug formatting ──────────────────────────────────────────

    #[test]
    fn debug_format_is_readable() {
        let plan = make_plan(&small_spec());
        let debug = format!("{plan:?}");
        assert!(
            debug.contains("OmniCqtPlan"),
            "debug format should contain type name"
        );
        assert!(
            debug.contains("num_bins"),
            "debug format should contain num_bins"
        );
    }

    // ── Numpy reference tests ─────────────────────────────────────
    //
    // Generated by scripts/gen_cqt_process_reference.py.
    // Regenerate with: make gen-cqt-process-reference

    include!(testdata!("cqt_process_numpy.rs"));

    fn numpy_spec() -> CqtSpec<f64> {
        cqt::design(
            CQT_PROC_SAMPLE_RATE,
            CQT_PROC_MIN_FREQ,
            CQT_PROC_MAX_FREQ,
            CQT_PROC_BINS_PER_OCTAVE,
            &Window::Hann,
        )
        .expect("valid design")
    }

    fn assert_complex_approx_eq(
        actual: &[Complex<f64>],
        expected: &[(f64, f64)],
        tol: f64,
        label: &str,
    ) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (a, &(re, im))) in actual.iter().zip(expected).enumerate() {
            let diff = (Complex::new(a.re - re, a.im - im)).norm();
            assert!(
                diff < tol,
                "{label}: bin {i} mismatch: got ({}, {}), expected ({re}, {im}), diff={diff}",
                a.re,
                a.im
            );
        }
    }

    #[test]
    fn numpy_all_tones() {
        let spec = numpy_spec();
        assert_eq!(
            spec.fft_length(),
            CQT_PROC_FFT_LENGTH,
            "FFT length should match numpy"
        );
        assert_eq!(
            spec.num_bins(),
            CQT_PROC_NUM_BINS,
            "bin count should match numpy"
        );

        let plan = make_plan(&spec);
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(CQT_PROC_ALL_TONES_INPUT, &mut output)
            .expect("process should succeed");

        assert_complex_approx_eq(&output, CQT_PROC_ALL_TONES_OUTPUT, 1e-10, "all-tones CQT");
    }

    #[test]
    fn numpy_pure_tone() {
        let spec = numpy_spec();
        let plan = make_plan(&spec);
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(CQT_PROC_TONE_INPUT, &mut output)
            .expect("process should succeed");

        assert_complex_approx_eq(&output, CQT_PROC_TONE_OUTPUT, 1e-10, "pure-tone CQT");

        // Verify the target bin has the largest magnitude.
        let magnitudes: Vec<f64> = output.iter().map(|c| c.norm()).collect();
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("no NaN"))
            .expect("at least one bin")
            .0;
        assert_eq!(
            peak_bin, CQT_PROC_TONE_BIN,
            "peak should be at bin {CQT_PROC_TONE_BIN}, was at {peak_bin}"
        );
    }

    #[test]
    fn numpy_two_tone() {
        let spec = numpy_spec();
        let plan = make_plan(&spec);
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(CQT_PROC_TWO_TONE_INPUT, &mut output)
            .expect("process should succeed");

        assert_complex_approx_eq(&output, CQT_PROC_TWO_TONE_OUTPUT, 1e-10, "two-tone CQT");
    }
}
