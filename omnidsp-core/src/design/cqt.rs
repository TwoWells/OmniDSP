// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Constant-Q Transform design — bin frequencies, kernel lengths, and FFT size.
//!
//! [`design`] takes user-friendly parameters (sample rate, frequency range,
//! bins per octave, window function) and produces a [`CqtSpec`] ready for
//! [`OmniCqt::create_plan`](crate::modules::cqt::OmniCqt).
//!
//! Users with pre-computed bin layouts and window coefficients can construct
//! a [`CqtSpec`] directly via [`CqtSpec::new`].

#![allow(
    clippy::cast_precision_loss,
    reason = "bin counts and lengths are small enough for f64"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "kernel lengths and FFT sizes are small positive values"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "2^(k/B) formulas are clearer with powf than exp2"
)]

use num_traits::Float;

use crate::error::{Error, Result};
use crate::types::Window;

// ─── Spec ───────────────────────────────────────────────────────────

/// Per-bin specification for the CQT.
///
/// Each bin carries its center frequency and pre-computed window
/// coefficients.  The kernel length is `window.len()`.
///
/// When produced by [`design`], the kernel length for a bin at frequency
/// `f` is `ceil(Q · sample_rate / f)` where `Q` is the
/// [`quality_factor`] for the chosen `bins_per_octave`.
#[derive(Debug, Clone)]
pub struct CqtBinSpec<T> {
    /// Center frequency in Hz.
    pub frequency: f64,
    /// Window coefficients for this bin's kernel.
    pub window: Vec<T>,
}

/// CQT specification — plan-ready description of a Constant-Q Transform.
///
/// Holds everything [`OmniCqt::create_plan`](crate::modules::cqt::OmniCqt)
/// needs to build frequency-domain kernels.  Construct via [`design`] or
/// directly via [`CqtSpec::new`].
///
/// - **`sample_rate`** — sampling frequency in Hz, used to build the
///   complex exponential kernel for each bin.
/// - **`fft_length`** — transform size; must be ≥ the longest bin's
///   `window.len()`.  When produced by [`design`], this is the next
///   power of two ≥ the longest kernel.
/// - **`hop_length`** — recommended advance between frames (advisory;
///   not used by the plan itself).  When produced by [`design`], this
///   is `max(shortest_kernel / 4, 1)`.
/// - **`bins`** — per-bin center frequencies and window coefficients,
///   ordered low to high frequency.  Each bin's `window.len()` is
///   its kernel length.
///
/// # Examples
///
/// Via the design convenience:
///
/// ```
/// use omnidsp_core::design::cqt;
/// use omnidsp_core::types::Window;
///
/// let spec = cqt::design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).unwrap();
/// assert!(!spec.bins().is_empty());
/// assert!(spec.fft_length().is_power_of_two());
/// ```
///
/// Direct construction:
///
/// ```
/// use omnidsp_core::design::cqt::{CqtSpec, CqtBinSpec};
///
/// let bins = vec![
///     CqtBinSpec { frequency: 440.0, window: vec![0.0_f64, 0.5, 1.0, 0.5, 0.0] },
/// ];
/// let spec = CqtSpec::new(44100.0, 8, 2, bins).unwrap();
/// assert_eq!(spec.num_bins(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct CqtSpec<T> {
    sample_rate: f64,
    fft_length: usize,
    hop_length: usize,
    bins: Vec<CqtBinSpec<T>>,
}

impl<T> CqtSpec<T> {
    /// Construct a CQT spec directly from pre-computed data.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if:
    /// - `bins` is empty
    /// - `sample_rate` is not positive
    /// - `fft_length` is zero
    /// - any bin has an empty window
    /// - any bin's window is longer than `fft_length`
    pub fn new(
        sample_rate: f64,
        fft_length: usize,
        hop_length: usize,
        bins: Vec<CqtBinSpec<T>>,
    ) -> Result<Self> {
        if sample_rate <= 0.0 {
            return Err(Error::InvalidSpec("sample rate must be positive".into()));
        }
        if fft_length == 0 {
            return Err(Error::InvalidSpec("FFT length must be non-zero".into()));
        }
        if bins.is_empty() {
            return Err(Error::InvalidSpec("at least one bin is required".into()));
        }
        for (i, bin) in bins.iter().enumerate() {
            if bin.window.is_empty() {
                return Err(Error::InvalidSpec(format!("bin {i} has an empty window")));
            }
            if bin.window.len() > fft_length {
                return Err(Error::InvalidSpec(format!(
                    "bin {i} window length ({}) exceeds FFT length ({fft_length})",
                    bin.window.len()
                )));
            }
        }
        Ok(Self {
            sample_rate,
            fft_length,
            hop_length,
            bins,
        })
    }

    /// Sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// FFT length (>= longest kernel).
    #[must_use]
    pub const fn fft_length(&self) -> usize {
        self.fft_length
    }

    /// Hop length in samples (recommended advance between frames).
    #[must_use]
    pub const fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Per-bin specifications, ordered low to high frequency.
    #[must_use]
    pub fn bins(&self) -> &[CqtBinSpec<T>] {
        &self.bins
    }

    /// Number of frequency bins.
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        self.bins.len()
    }
}

// ─── Quality factor ────────────────────────────────────────────────

/// Compute the quality factor for a given number of bins per octave.
///
/// `Q = 1 / (2^(1/B) − 1)` where `B` is `bins_per_octave`.
///
/// # Panics
///
/// Panics if `bins_per_octave` is zero.
#[must_use]
pub fn quality_factor(bins_per_octave: u32) -> f64 {
    assert!(bins_per_octave > 0, "bins_per_octave must be at least 1");
    1.0 / (2.0_f64.powf(1.0 / f64::from(bins_per_octave)) - 1.0)
}

// ─── Design function ────────────────────────────────────────────────

/// Design a Constant-Q Transform from user-friendly parameters.
///
/// Computes per-bin center frequencies, kernel lengths, window
/// coefficients, FFT size, and hop length, returning a [`CqtSpec`]
/// ready for [`OmniCqt::create_plan`](crate::modules::cqt::OmniCqt).
///
/// **Bin placement:** `f_k = min_freq · 2^(k/B)` for `k = 0, 1, ...`
/// while `f_k < max_freq` and `f_k < sample_rate / 2`.  The upper
/// bound is exclusive — one octave at 12 bins/octave produces exactly
/// 12 bins.
///
/// **Kernel length:** `ceil(Q · sample_rate / f_k)` where
/// `Q = 1 / (2^(1/B) − 1)` is the [`quality_factor`].  Lower
/// frequencies get longer kernels (better frequency resolution);
/// higher frequencies get shorter kernels (better time resolution).
///
/// **Window:** each bin's kernel is windowed independently at its own
/// length using `window_fn`.
///
/// # Errors
///
/// Returns [`Error::InvalidSpec`] if:
/// - `sample_rate` is not positive
/// - `min_freq` is not positive
/// - `min_freq >= max_freq`
/// - `max_freq >= sample_rate / 2` (at or above Nyquist)
/// - `bins_per_octave` is zero
/// - window generation fails for any bin
pub fn design<T: Float>(
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
    bins_per_octave: u32,
    window: &Window<T>,
) -> Result<CqtSpec<T>> {
    if sample_rate <= 0.0 {
        return Err(Error::InvalidSpec("sample rate must be positive".into()));
    }
    if min_freq <= 0.0 {
        return Err(Error::InvalidSpec(
            "minimum frequency must be positive".into(),
        ));
    }
    if min_freq >= max_freq {
        return Err(Error::InvalidSpec(format!(
            "min_freq ({min_freq}) must be less than max_freq ({max_freq})"
        )));
    }
    let nyquist = sample_rate / 2.0;
    if max_freq >= nyquist {
        return Err(Error::InvalidSpec(format!(
            "max_freq ({max_freq}) must be below Nyquist ({nyquist})"
        )));
    }
    if bins_per_octave == 0 {
        return Err(Error::InvalidSpec(
            "bins_per_octave must be at least 1".into(),
        ));
    }

    let b = f64::from(bins_per_octave);
    let q = quality_factor(bins_per_octave);

    // Build per-bin specs with window coefficients.
    let mut bins = Vec::new();

    // First bin is always min_freq.
    let first_len = (q * sample_rate / min_freq).ceil() as usize;
    let first_len = first_len.max(1);
    bins.push(CqtBinSpec {
        frequency: min_freq,
        window: window.coefficients(first_len)?,
    });

    // Remaining bins: f_k = min_freq · 2^(k/B) while f_k < max_freq and < Nyquist.
    let mut k = 1u32;
    loop {
        let freq = min_freq * 2.0_f64.powf(f64::from(k) / b);
        if freq >= max_freq || freq >= nyquist {
            break;
        }
        let kernel_len = (q * sample_rate / freq).ceil() as usize;
        let kernel_len = kernel_len.max(1);
        bins.push(CqtBinSpec {
            frequency: freq,
            window: window.coefficients(kernel_len)?,
        });
        k += 1;
    }

    // FFT length: next power of two ≥ longest kernel (first bin = lowest freq).
    let fft_length = bins[0].window.len().next_power_of_two();

    // Hop length: based on shortest kernel (last bin = highest freq).
    let min_kernel = bins[bins.len() - 1].window.len();
    let hop_length = (min_kernel / 4).max(1);

    Ok(CqtSpec {
        sample_rate,
        fft_length,
        hop_length,
        bins,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;

    #[allow(
        clippy::wildcard_imports,
        reason = "bulk golden-vector import in tests"
    )]
    use omnidsp_testdata::cqt_numpy::*;

    // ── Validation ─────────────────────────────────────────────

    #[test]
    fn rejects_zero_sample_rate() {
        let err = design::<f64>(0.0, 27.5, 4186.0, 24, &Window::Hann);
        assert!(err.is_err(), "sample rate 0 should be rejected");
    }

    #[test]
    fn rejects_negative_sample_rate() {
        let err = design::<f64>(-44100.0, 27.5, 4186.0, 24, &Window::Hann);
        assert!(err.is_err(), "negative sample rate should be rejected");
    }

    #[test]
    fn rejects_zero_min_freq() {
        let err = design::<f64>(44100.0, 0.0, 4186.0, 24, &Window::Hann);
        assert!(err.is_err(), "min_freq 0 should be rejected");
    }

    #[test]
    fn rejects_negative_min_freq() {
        let err = design::<f64>(44100.0, -10.0, 4186.0, 24, &Window::Hann);
        assert!(err.is_err(), "negative min_freq should be rejected");
    }

    #[test]
    fn rejects_min_ge_max() {
        let err = design::<f64>(44100.0, 5000.0, 1000.0, 24, &Window::Hann);
        assert!(err.is_err(), "min_freq >= max_freq should be rejected");
    }

    #[test]
    fn rejects_min_eq_max() {
        let err = design::<f64>(44100.0, 1000.0, 1000.0, 24, &Window::Hann);
        assert!(err.is_err(), "min_freq == max_freq should be rejected");
    }

    #[test]
    fn rejects_max_at_nyquist() {
        let err = design::<f64>(44100.0, 27.5, 22050.0, 24, &Window::Hann);
        assert!(err.is_err(), "max_freq at Nyquist should be rejected");
    }

    #[test]
    fn rejects_max_above_nyquist() {
        let err = design::<f64>(44100.0, 27.5, 30000.0, 24, &Window::Hann);
        assert!(err.is_err(), "max_freq above Nyquist should be rejected");
    }

    #[test]
    fn rejects_zero_bins_per_octave() {
        let err = design::<f64>(44100.0, 27.5, 4186.0, 0, &Window::Hann);
        assert!(err.is_err(), "bins_per_octave 0 should be rejected");
    }

    // ── Quality factor ─────────────────────────────────────────

    #[test]
    fn quality_factor_12_bins() {
        let q = quality_factor(12);
        let expected_q = 1.0 / (2.0_f64.powf(1.0 / 12.0) - 1.0);
        assert!(
            (q - expected_q).abs() < 1e-10,
            "Q for 12 bins/octave: expected {expected_q}, got {q}"
        );
    }

    #[test]
    fn quality_factor_24_bins() {
        let q = quality_factor(24);
        let expected_q = 1.0 / (2.0_f64.powf(1.0 / 24.0) - 1.0);
        assert!(
            (q - expected_q).abs() < 1e-10,
            "Q for 24 bins/octave: expected {expected_q}, got {q}"
        );
    }

    // ── Bin generation ─────────────────────────────────────────

    #[test]
    fn piano_range_bin_count() {
        // 27.5 Hz to 4186 Hz ≈ 7.25 octaves, 24 bins/octave → ~174 bins
        let spec = design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).expect("valid design");
        assert!(
            spec.bins.len() > 150 && spec.bins.len() < 200,
            "piano range 24 bins/oct should produce ~174 bins, got {}",
            spec.bins.len()
        );
    }

    #[test]
    fn one_octave_12_bins() {
        let spec = design(44100.0, 440.0, 880.0, 12, &Window::<f64>::Hann).expect("valid design");
        // 1 octave, 12 bins/octave, exclusive upper bound → exactly 12 bins
        assert_eq!(spec.bins.len(), 12, "one octave, 12 bins/oct → 12 bins");
    }

    #[test]
    fn bins_are_ascending() {
        let spec = design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).expect("valid design");
        for w in spec.bins.windows(2) {
            assert!(
                w[1].frequency > w[0].frequency,
                "bins must be ascending: {} should be > {}",
                w[1].frequency,
                w[0].frequency
            );
        }
    }

    #[test]
    fn first_bin_is_min_freq() {
        let spec = design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).expect("valid design");
        assert!(
            (spec.bins[0].frequency - 27.5).abs() < 1e-10,
            "first bin should be min_freq"
        );
    }

    #[test]
    fn last_bin_below_max_freq() {
        let spec = design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).expect("valid design");
        let last_freq = spec.bins[spec.bins.len() - 1].frequency;
        assert!(
            last_freq < 4186.0,
            "last bin ({last_freq}) should be below max_freq (exclusive)"
        );
    }

    #[test]
    fn bins_excluded_above_nyquist() {
        // max_freq near Nyquist — bins should stop below sr/2
        let spec =
            design(44100.0, 10000.0, 22000.0, 12, &Window::<f64>::Hann).expect("valid design");
        let nyquist = 44100.0 / 2.0;
        for bin in &spec.bins {
            assert!(
                bin.frequency < nyquist,
                "bin at {} Hz should be below Nyquist ({nyquist})",
                bin.frequency
            );
        }
    }

    // ── Kernel lengths ─────────────────────────────────────────

    #[test]
    fn kernel_lengths_decrease_with_frequency() {
        let spec = design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).expect("valid design");
        for w in spec.bins.windows(2) {
            assert!(
                w[0].window.len() >= w[1].window.len(),
                "lower freq ({} Hz, len {}) should have longer kernel than higher freq ({} Hz, len {})",
                w[0].frequency,
                w[0].window.len(),
                w[1].frequency,
                w[1].window.len()
            );
        }
    }

    #[test]
    fn kernel_length_formula() {
        let spec = design(44100.0, 440.0, 880.0, 12, &Window::<f64>::Hann).expect("valid design");
        let q = quality_factor(12);
        for bin in &spec.bins {
            let expected = (q * 44100.0 / bin.frequency).ceil() as usize;
            assert_eq!(
                bin.window.len(),
                expected,
                "kernel length for {} Hz should be ceil(Q * sr / f)",
                bin.frequency
            );
        }
    }

    // ── FFT length ─────────────────────────────────────────────

    #[test]
    fn fft_length_is_power_of_two() {
        let spec = design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).expect("valid design");
        assert!(
            spec.fft_length.is_power_of_two(),
            "FFT length {} must be a power of two",
            spec.fft_length
        );
    }

    #[test]
    fn fft_length_ge_longest_kernel() {
        let spec = design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).expect("valid design");
        let max_kernel = spec.bins[0].window.len();
        assert!(
            spec.fft_length >= max_kernel,
            "FFT length {} must be >= longest kernel {}",
            spec.fft_length,
            max_kernel
        );
    }

    // ── Hop length ─────────────────────────────────────────────

    #[test]
    fn hop_length_positive() {
        let spec = design(44100.0, 27.5, 4186.0, 24, &Window::<f64>::Hann).expect("valid design");
        assert!(spec.hop_length >= 1, "hop length must be at least 1");
    }

    // ── Edge cases ─────────────────────────────────────────────

    #[test]
    fn single_bin_per_octave() {
        let spec = design(44100.0, 100.0, 1000.0, 1, &Window::<f64>::Hann).expect("valid design");
        // ~3.32 octaves, 1 bin/octave, exclusive → bins at 100, 200, 400, 800
        assert_eq!(
            spec.bins.len(),
            4,
            "100–1000 Hz exclusive, 1 bin/oct → 4 bins"
        );
    }

    // ── Direct construction ───────────────────────────────────

    #[test]
    fn new_rejects_empty_bins() {
        let result = CqtSpec::<f64>::new(44100.0, 256, 64, vec![]);
        assert!(result.is_err(), "empty bins should be rejected");
    }

    #[test]
    fn new_rejects_empty_window() {
        let bins = vec![CqtBinSpec::<f64> {
            frequency: 440.0,
            window: vec![],
        }];
        let result = CqtSpec::new(44100.0, 256, 64, bins);
        assert!(result.is_err(), "empty window should be rejected");
    }

    #[test]
    fn new_rejects_window_exceeding_fft() {
        let bins = vec![CqtBinSpec {
            frequency: 440.0,
            window: vec![1.0_f64; 512],
        }];
        let result = CqtSpec::new(44100.0, 256, 64, bins);
        assert!(
            result.is_err(),
            "window exceeding FFT length should be rejected"
        );
    }

    #[test]
    fn new_accepts_valid_spec() {
        let bins = vec![CqtBinSpec {
            frequency: 440.0,
            window: vec![0.0_f64, 0.5, 1.0, 0.5, 0.0],
        }];
        let spec = CqtSpec::new(44100.0, 8, 2, bins).expect("valid spec");
        assert_eq!(spec.num_bins(), 1, "should have one bin");
        assert_eq!(spec.fft_length(), 8, "fft_length should be 8");
        assert_eq!(spec.hop_length(), 2, "hop_length should be 2");
    }

    // ── Numpy reference tests ──────────────────────────────────

    struct CqtRef {
        sample_rate: f64,
        min_freq: f64,
        max_freq: f64,
        bins_per_octave: u32,
        q: f64,
        freqs: &'static [f64],
        kernel_lengths: &'static [usize],
        fft_length: usize,
        hop_length: usize,
    }

    fn assert_design_matches(r: &CqtRef) {
        let spec = design(
            r.sample_rate,
            r.min_freq,
            r.max_freq,
            r.bins_per_octave,
            &Window::<f64>::Hann,
        )
        .expect("valid design");

        let q = quality_factor(r.bins_per_octave);
        assert!(
            (q - r.q).abs() < 1e-8,
            "Q mismatch: expected {}, got {}",
            r.q,
            q
        );
        assert_eq!(
            spec.bins.len(),
            r.freqs.len(),
            "bin count mismatch: expected {}, got {}",
            r.freqs.len(),
            spec.bins.len()
        );
        for (i, bin) in spec.bins.iter().enumerate() {
            assert!(
                (bin.frequency - r.freqs[i]).abs() < 1e-6,
                "bin {i} frequency mismatch: expected {}, got {}",
                r.freqs[i],
                bin.frequency
            );
            assert_eq!(
                bin.window.len(),
                r.kernel_lengths[i],
                "bin {i} kernel length mismatch: expected {}, got {}",
                r.kernel_lengths[i],
                bin.window.len()
            );
        }
        assert_eq!(
            spec.fft_length, r.fft_length,
            "FFT length mismatch: expected {}, got {}",
            r.fft_length, spec.fft_length
        );
        assert_eq!(
            spec.hop_length, r.hop_length,
            "hop length mismatch: expected {}, got {}",
            r.hop_length, spec.hop_length
        );
    }

    #[test]
    fn numpy_piano_24_bins() {
        assert_design_matches(&CqtRef {
            sample_rate: CQT_PIANO_24_SAMPLE_RATE,
            min_freq: CQT_PIANO_24_MIN_FREQ,
            max_freq: CQT_PIANO_24_MAX_FREQ,
            bins_per_octave: CQT_PIANO_24_BINS_PER_OCTAVE,
            q: CQT_PIANO_24_Q,
            freqs: CQT_PIANO_24_FREQS,
            kernel_lengths: CQT_PIANO_24_KERNEL_LENGTHS,
            fft_length: CQT_PIANO_24_FFT_LENGTH,
            hop_length: CQT_PIANO_24_HOP_LENGTH,
        });
    }

    #[test]
    fn numpy_one_octave_12_bins() {
        assert_design_matches(&CqtRef {
            sample_rate: CQT_OCTAVE_12_SAMPLE_RATE,
            min_freq: CQT_OCTAVE_12_MIN_FREQ,
            max_freq: CQT_OCTAVE_12_MAX_FREQ,
            bins_per_octave: CQT_OCTAVE_12_BINS_PER_OCTAVE,
            q: CQT_OCTAVE_12_Q,
            freqs: CQT_OCTAVE_12_FREQS,
            kernel_lengths: CQT_OCTAVE_12_KERNEL_LENGTHS,
            fft_length: CQT_OCTAVE_12_FFT_LENGTH,
            hop_length: CQT_OCTAVE_12_HOP_LENGTH,
        });
    }

    #[test]
    fn numpy_near_nyquist() {
        assert_design_matches(&CqtRef {
            sample_rate: CQT_NYQUIST_12_SAMPLE_RATE,
            min_freq: CQT_NYQUIST_12_MIN_FREQ,
            max_freq: CQT_NYQUIST_12_MAX_FREQ,
            bins_per_octave: CQT_NYQUIST_12_BINS_PER_OCTAVE,
            q: CQT_NYQUIST_12_Q,
            freqs: CQT_NYQUIST_12_FREQS,
            kernel_lengths: CQT_NYQUIST_12_KERNEL_LENGTHS,
            fft_length: CQT_NYQUIST_12_FFT_LENGTH,
            hop_length: CQT_NYQUIST_12_HOP_LENGTH,
        });
    }
}
