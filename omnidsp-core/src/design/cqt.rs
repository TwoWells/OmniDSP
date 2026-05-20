// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Constant-Q Transform design — bin frequencies, kernel lengths, and FFT size.
//!
//! [`CqtSpec`] captures the user's parameters (sample rate, frequency range,
//! bins per octave).  [`design`] computes the quality factor, per-bin
//! specifications, FFT length, and hop length that the CQT module needs
//! to build its kernels.
//!
//! This module is pure arithmetic on `f64` — no dependency on primitives or
//! implementation crates.  Window selection and actual kernel generation are
//! the CQT module's responsibility.

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

use crate::error::{Error, Result};

// ─── Spec (backend-agnostic) ─────────────────────────────────────────

/// CQT specification — backend-agnostic description of the desired
/// Constant-Q Transform.
///
/// # Examples
///
/// ```
/// use omnidsp_core::design::cqt::CqtSpec;
///
/// let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).unwrap();
/// assert!((spec.sample_rate() - 44100.0).abs() < f64::EPSILON);
/// assert_eq!(spec.bins_per_octave(), 24);
/// ```
#[derive(Debug, Clone)]
pub struct CqtSpec {
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
    bins_per_octave: u32,
}

impl CqtSpec {
    /// Create a new CQT specification.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if:
    /// - `sample_rate` is not positive
    /// - `min_freq` is not positive
    /// - `min_freq >= max_freq`
    /// - `max_freq >= sample_rate / 2` (at or above Nyquist)
    /// - `bins_per_octave` is zero
    pub fn new(
        sample_rate: f64,
        min_freq: f64,
        max_freq: f64,
        bins_per_octave: u32,
    ) -> Result<Self> {
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
        Ok(Self {
            sample_rate,
            min_freq,
            max_freq,
            bins_per_octave,
        })
    }

    /// Sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Minimum frequency in Hz.
    #[must_use]
    pub const fn min_freq(&self) -> f64 {
        self.min_freq
    }

    /// Maximum frequency in Hz.
    #[must_use]
    pub const fn max_freq(&self) -> f64 {
        self.max_freq
    }

    /// Bins per octave.
    #[must_use]
    pub const fn bins_per_octave(&self) -> u32 {
        self.bins_per_octave
    }
}

// ─── Design output ──────────────────────────────────────────────────

/// Per-bin specification for the CQT.
#[derive(Debug, Clone, PartialEq)]
pub struct CqtBinSpec {
    /// Center frequency in Hz.
    pub frequency: f64,
    /// Kernel length in samples (at the original sample rate).
    pub kernel_length: usize,
}

/// Result of CQT design.
///
/// Produced by [`design`] from a [`CqtSpec`].  Contains all parameters
/// the CQT module needs to build frequency-domain kernels.
#[derive(Debug, Clone)]
pub struct CqtDesign {
    /// Original sample rate in Hz.
    pub sample_rate: f64,
    /// Quality factor Q = 1 / (2^(1/B) − 1).
    pub quality_factor: f64,
    /// Hop length in samples.
    pub hop_length: usize,
    /// FFT length (power of two, ≥ longest kernel).
    pub fft_length: usize,
    /// Per-bin specifications, ordered low to high frequency.
    pub bins: Vec<CqtBinSpec>,
}

// ─── Design function ────────────────────────────────────────────────

/// Design a Constant-Q Transform from a [`CqtSpec`].
///
/// Computes the quality factor, per-bin center frequencies, kernel lengths,
/// FFT size (next power of two ≥ longest kernel), and a default hop length
/// based on the shortest kernel.
///
/// Bins are placed at `f_k = min_freq · 2^(k/B)` for `k = 0, 1, ...`
/// while `f_k < max_freq` and `f_k < sample_rate / 2`.  The upper bound
/// is exclusive — one octave at 12 bins/octave produces exactly 12 bins.
///
/// # Examples
///
/// ```
/// use omnidsp_core::design::cqt::{CqtSpec, design};
///
/// let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).unwrap();
/// let d = design(&spec);
/// assert!(d.quality_factor > 0.0);
/// assert!(!d.bins.is_empty());
/// assert!(d.fft_length.is_power_of_two());
/// ```
#[must_use]
pub fn design(spec: &CqtSpec) -> CqtDesign {
    let b = f64::from(spec.bins_per_octave);
    let nyquist = spec.sample_rate / 2.0;

    // Q = 1 / (2^(1/B) − 1)
    let quality_factor = 1.0 / (2.0_f64.powf(1.0 / b) - 1.0);

    // First bin is always min_freq (CqtSpec guarantees 0 < min_freq < max_freq < nyquist).
    let first_len = (quality_factor * spec.sample_rate / spec.min_freq).ceil() as usize;
    let mut bins = vec![CqtBinSpec {
        frequency: spec.min_freq,
        kernel_length: first_len.max(1),
    }];

    // Remaining bins: f_k = min_freq · 2^(k/B) while f_k < max_freq and < Nyquist
    let mut k = 1u32;
    loop {
        let freq = spec.min_freq * 2.0_f64.powf(f64::from(k) / b);
        if freq >= spec.max_freq || freq >= nyquist {
            break;
        }
        let kernel_len = (quality_factor * spec.sample_rate / freq).ceil() as usize;
        bins.push(CqtBinSpec {
            frequency: freq,
            kernel_length: kernel_len.max(1),
        });
        k += 1;
    }

    // FFT length: next power of two ≥ longest kernel (first bin = lowest freq)
    let fft_length = bins[0].kernel_length.next_power_of_two();

    // Hop length: based on shortest kernel (last bin = highest freq)
    let min_kernel = bins[bins.len() - 1].kernel_length;
    let hop_length = (min_kernel / 4).max(1);

    CqtDesign {
        sample_rate: spec.sample_rate,
        quality_factor,
        hop_length,
        fft_length,
        bins,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;

    include!(testdata!("cqt_numpy.rs"));

    // ── Validation ─────────────────────────────────────────────

    #[test]
    fn rejects_zero_sample_rate() {
        let err = CqtSpec::new(0.0, 27.5, 4186.0, 24);
        assert!(err.is_err(), "sample rate 0 should be rejected");
    }

    #[test]
    fn rejects_negative_sample_rate() {
        let err = CqtSpec::new(-44100.0, 27.5, 4186.0, 24);
        assert!(err.is_err(), "negative sample rate should be rejected");
    }

    #[test]
    fn rejects_zero_min_freq() {
        let err = CqtSpec::new(44100.0, 0.0, 4186.0, 24);
        assert!(err.is_err(), "min_freq 0 should be rejected");
    }

    #[test]
    fn rejects_negative_min_freq() {
        let err = CqtSpec::new(44100.0, -10.0, 4186.0, 24);
        assert!(err.is_err(), "negative min_freq should be rejected");
    }

    #[test]
    fn rejects_min_ge_max() {
        let err = CqtSpec::new(44100.0, 5000.0, 1000.0, 24);
        assert!(err.is_err(), "min_freq >= max_freq should be rejected");
    }

    #[test]
    fn rejects_min_eq_max() {
        let err = CqtSpec::new(44100.0, 1000.0, 1000.0, 24);
        assert!(err.is_err(), "min_freq == max_freq should be rejected");
    }

    #[test]
    fn rejects_max_at_nyquist() {
        let err = CqtSpec::new(44100.0, 27.5, 22050.0, 24);
        assert!(err.is_err(), "max_freq at Nyquist should be rejected");
    }

    #[test]
    fn rejects_max_above_nyquist() {
        let err = CqtSpec::new(44100.0, 27.5, 30000.0, 24);
        assert!(err.is_err(), "max_freq above Nyquist should be rejected");
    }

    #[test]
    fn rejects_zero_bins_per_octave() {
        let err = CqtSpec::new(44100.0, 27.5, 4186.0, 0);
        assert!(err.is_err(), "bins_per_octave 0 should be rejected");
    }

    // ── Quality factor ─────────────────────────────────────────

    #[test]
    fn quality_factor_12_bins() {
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 12).expect("valid spec");
        let d = design(&spec);
        let expected_q = 1.0 / (2.0_f64.powf(1.0 / 12.0) - 1.0);
        assert!(
            (d.quality_factor - expected_q).abs() < 1e-10,
            "Q for 12 bins/octave: expected {expected_q}, got {}",
            d.quality_factor
        );
    }

    #[test]
    fn quality_factor_24_bins() {
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        let expected_q = 1.0 / (2.0_f64.powf(1.0 / 24.0) - 1.0);
        assert!(
            (d.quality_factor - expected_q).abs() < 1e-10,
            "Q for 24 bins/octave: expected {expected_q}, got {}",
            d.quality_factor
        );
    }

    // ── Bin generation ─────────────────────────────────────────

    #[test]
    fn piano_range_bin_count() {
        // 27.5 Hz to 4186 Hz ≈ 7.25 octaves, 24 bins/octave → ~174 bins
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        assert!(
            d.bins.len() > 150 && d.bins.len() < 200,
            "piano range 24 bins/oct should produce ~174 bins, got {}",
            d.bins.len()
        );
    }

    #[test]
    fn one_octave_12_bins() {
        let spec = CqtSpec::new(44100.0, 440.0, 880.0, 12).expect("valid spec");
        let d = design(&spec);
        // 1 octave, 12 bins/octave, exclusive upper bound → exactly 12 bins
        assert_eq!(d.bins.len(), 12, "one octave, 12 bins/oct → 12 bins");
    }

    #[test]
    fn bins_are_ascending() {
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        for w in d.bins.windows(2) {
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
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        assert!(
            (d.bins[0].frequency - 27.5).abs() < 1e-10,
            "first bin should be min_freq"
        );
    }

    #[test]
    fn last_bin_below_max_freq() {
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        let last_freq = d.bins[d.bins.len() - 1].frequency;
        assert!(
            last_freq < 4186.0,
            "last bin ({last_freq}) should be below max_freq (exclusive)"
        );
    }

    #[test]
    fn bins_excluded_above_nyquist() {
        // max_freq near Nyquist — bins should stop below sr/2
        let spec = CqtSpec::new(44100.0, 10000.0, 22000.0, 12).expect("valid spec");
        let d = design(&spec);
        let nyquist = 44100.0 / 2.0;
        for bin in &d.bins {
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
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        for w in d.bins.windows(2) {
            assert!(
                w[0].kernel_length >= w[1].kernel_length,
                "lower freq ({} Hz, len {}) should have longer kernel than higher freq ({} Hz, len {})",
                w[0].frequency,
                w[0].kernel_length,
                w[1].frequency,
                w[1].kernel_length
            );
        }
    }

    #[test]
    fn kernel_length_formula() {
        let spec = CqtSpec::new(44100.0, 440.0, 880.0, 12).expect("valid spec");
        let d = design(&spec);
        for bin in &d.bins {
            let expected = (d.quality_factor * 44100.0 / bin.frequency).ceil() as usize;
            assert_eq!(
                bin.kernel_length, expected,
                "kernel length for {} Hz should be ceil(Q * sr / f)",
                bin.frequency
            );
        }
    }

    // ── FFT length ─────────────────────────────────────────────

    #[test]
    fn fft_length_is_power_of_two() {
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        assert!(
            d.fft_length.is_power_of_two(),
            "FFT length {} must be a power of two",
            d.fft_length
        );
    }

    #[test]
    fn fft_length_ge_longest_kernel() {
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        let max_kernel = d.bins[0].kernel_length;
        assert!(
            d.fft_length >= max_kernel,
            "FFT length {} must be >= longest kernel {}",
            d.fft_length,
            max_kernel
        );
    }

    // ── Hop length ─────────────────────────────────────────────

    #[test]
    fn hop_length_positive() {
        let spec = CqtSpec::new(44100.0, 27.5, 4186.0, 24).expect("valid spec");
        let d = design(&spec);
        assert!(d.hop_length >= 1, "hop length must be at least 1");
    }

    // ── Edge cases ─────────────────────────────────────────────

    #[test]
    fn single_bin_per_octave() {
        let spec = CqtSpec::new(44100.0, 100.0, 1000.0, 1).expect("valid spec");
        let d = design(&spec);
        // ~3.32 octaves, 1 bin/octave, exclusive → bins at 100, 200, 400, 800
        assert_eq!(d.bins.len(), 4, "100–1000 Hz exclusive, 1 bin/oct → 4 bins");
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
        let spec = CqtSpec::new(r.sample_rate, r.min_freq, r.max_freq, r.bins_per_octave)
            .expect("valid spec");
        let d = design(&spec);

        assert!(
            (d.quality_factor - r.q).abs() < 1e-8,
            "Q mismatch: expected {}, got {}",
            r.q,
            d.quality_factor
        );
        assert_eq!(
            d.bins.len(),
            r.freqs.len(),
            "bin count mismatch: expected {}, got {}",
            r.freqs.len(),
            d.bins.len()
        );
        for (i, bin) in d.bins.iter().enumerate() {
            assert!(
                (bin.frequency - r.freqs[i]).abs() < 1e-6,
                "bin {i} frequency mismatch: expected {}, got {}",
                r.freqs[i],
                bin.frequency
            );
            assert_eq!(
                bin.kernel_length, r.kernel_lengths[i],
                "bin {i} kernel length mismatch: expected {}, got {}",
                r.kernel_lengths[i], bin.kernel_length
            );
        }
        assert_eq!(
            d.fft_length, r.fft_length,
            "FFT length mismatch: expected {}, got {}",
            r.fft_length, d.fft_length
        );
        assert_eq!(
            d.hop_length, r.hop_length,
            "hop length mismatch: expected {}, got {}",
            r.hop_length, d.hop_length
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
