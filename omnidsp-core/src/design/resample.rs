// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Resampling design — polyphase prototype filter and plan-ready spec.
//!
//! # Algorithm
//!
//! 1. **Rational approximation:** the rate ratio `output_rate / input_rate`
//!    is approximated as `L / M` via continued fractions, bounded by
//!    `max_phases` to limit the number of polyphase sub-filters.
//! 2. **Prototype filter:** a lowpass FIR with cutoff `1 / (2 · max(L, M))`
//!    (normalized to the upsampled rate) prevents aliasing.  Filter length
//!    is estimated from the quality parameter via Kaiser's formula.
//! 3. **Output:** a [`ResampleSpec`] holding `L`, `M`, the prototype
//!    coefficients, and the processing mode.
//!
//! [`design`] takes plain parameters (rates, quality, window) and returns
//! the spec.  Users with pre-computed coefficients can construct a
//! [`ResampleSpec`] directly via [`ResampleSpec::new`].

#![allow(
    clippy::cast_precision_loss,
    reason = "sample-rate integers are small enough for f64"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "filter order estimate is a small positive value"
)]

use num_traits::Float;

use crate::error::{Error, Result};
use crate::types::Window;

// ─── Spec (backend-agnostic) ─────────────────────────────────────────

/// Processing mode for resampling — controls output length semantics.
///
/// The two modes produce identical sample values for the overlapping
/// range; they differ only in how many output samples are emitted.
///
/// Which mode produces more output depends on the filter length `H`
/// relative to the upsampling factor `L`:
/// - `H < L` (typical for high L): **Streaming** produces more
///   (the phase accumulator emits outputs beyond the filter's support).
/// - `H > L` (long filter, small L): **Batch** produces more
///   (includes the filter's ring-down tail past the last input).
/// - `H = L`: both produce the same count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMode {
    /// Streaming: `⌈N × L / M⌉` outputs per call.
    ///
    /// The phase accumulator drives output count, independent of the
    /// filter length.  Internal state carries across calls for continuous
    /// block-by-block processing.  This is the right choice for real-time
    /// audio and any pipeline where data arrives in chunks.
    Streaming,

    /// Batch: `⌈((N − 1) × L + filter_len) / M⌉` outputs.
    ///
    /// Matches scipy's `upfirdn` / finite convolution model.  The output
    /// length accounts for the prototype filter's support, so a single
    /// call produces the complete resampled signal including the filter's
    /// group delay transient.  Use this for one-shot processing of
    /// complete signals.
    Batch,
}

/// Validated resampling quality parameter.
///
/// Wraps a `u8` in the range `0..=10`.  Higher values produce longer
/// filters with narrower transition bands and deeper stopband rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleQuality(u8);

impl ResampleQuality {
    /// Maximum quality value.
    pub const MAX: u8 = 10;

    /// Create a new quality value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `value` exceeds [`Self::MAX`].
    pub fn new(value: u8) -> Result<Self> {
        if value > Self::MAX {
            return Err(Error::InvalidSpec(format!(
                "resample quality ({value}) must be in 0..={max}",
                max = Self::MAX,
            )));
        }
        Ok(Self(value))
    }

    /// Return the raw quality value.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }
}

/// Resampling specification — plan-ready data for a polyphase resampler.
///
/// Holds the rational conversion factors, prototype anti-aliasing FIR
/// filter, and processing mode.  Construct via [`design`] for automatic
/// filter generation, or via [`ResampleSpec::new`] with pre-computed
/// coefficients.
///
/// # Examples
///
/// ```
/// use omnidsp_core::design::resample::{
///     self, ResampleSpec, ResampleMode, ResampleQuality, DEFAULT_MAX_PHASES,
/// };
/// use omnidsp_core::types::Window;
///
/// // Via design():
/// let spec = resample::design(
///     44100.0_f64, 48000.0,
///     ResampleQuality::new(5).unwrap(),
///     &Window::Hamming,
///     DEFAULT_MAX_PHASES,
///     ResampleMode::Streaming,
/// ).unwrap();
/// assert_eq!(spec.up_factor(), 160);
/// assert_eq!(spec.down_factor(), 147);
///
/// // Or directly from pre-computed data:
/// let spec = ResampleSpec::new(2, 1, vec![0.5_f64, 1.0, 0.5], ResampleMode::Streaming).unwrap();
/// assert_eq!(spec.up_factor(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ResampleSpec<T> {
    up_factor: usize,
    down_factor: usize,
    prototype_filter: Vec<T>,
    mode: ResampleMode,
}

/// Default maximum number of polyphase phases.
pub const DEFAULT_MAX_PHASES: usize = 256;

/// Hard upper limit on polyphase phases.
pub const MAX_PHASES_LIMIT: usize = 1024;

impl<T> ResampleSpec<T> {
    /// Create a spec from pre-computed polyphase data.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `up_factor` or `down_factor` is zero,
    /// or if `prototype_filter` is empty.
    pub fn new(
        up_factor: usize,
        down_factor: usize,
        prototype_filter: Vec<T>,
        mode: ResampleMode,
    ) -> Result<Self> {
        if up_factor == 0 || down_factor == 0 {
            return Err(Error::InvalidSpec(
                "resample up_factor and down_factor must be positive".into(),
            ));
        }
        if prototype_filter.is_empty() {
            return Err(Error::InvalidSpec(
                "resample prototype filter must not be empty".into(),
            ));
        }
        Ok(Self {
            up_factor,
            down_factor,
            prototype_filter,
            mode,
        })
    }

    /// Upsampling factor L (number of polyphase phases).
    #[must_use]
    pub const fn up_factor(&self) -> usize {
        self.up_factor
    }

    /// Downsampling factor M.
    #[must_use]
    pub const fn down_factor(&self) -> usize {
        self.down_factor
    }

    /// Prototype FIR filter coefficients.
    #[must_use]
    pub fn prototype_filter(&self) -> &[T] {
        &self.prototype_filter
    }

    /// Processing mode.
    #[must_use]
    pub const fn mode(&self) -> ResampleMode {
        self.mode
    }

    /// Override the processing mode.
    #[must_use]
    pub const fn with_mode(mut self, mode: ResampleMode) -> Self {
        self.mode = mode;
        self
    }
}

// ─── Polyphase design ────────────────────────────────────────────────

/// Design a polyphase resampling prototype filter.
///
/// Computes rational conversion factors L/M via best rational approximation
/// (continued fractions) bounded by `max_phases`, then produces a lowpass
/// FIR prototype filter whose cutoff prevents aliasing.
///
/// `max_phases` is clamped to `1..=1024`.  Use [`DEFAULT_MAX_PHASES`] (256)
/// when unsure.
///
/// # Errors
///
/// Returns [`Error::InvalidSpec`] if either rate is not positive.
pub fn design<T: Float>(
    input_rate: T,
    output_rate: T,
    quality: ResampleQuality,
    window: &Window<T>,
    max_phases: usize,
    mode: ResampleMode,
) -> Result<ResampleSpec<T>> {
    let sr_in = to_f64(input_rate)?;
    let sr_out = to_f64(output_rate)?;

    if sr_in <= 0.0 {
        return Err(Error::InvalidSpec(
            "input sample rate must be positive".into(),
        ));
    }
    if sr_out <= 0.0 {
        return Err(Error::InvalidSpec(
            "output sample rate must be positive".into(),
        ));
    }

    let max_phases = max_phases.clamp(1, MAX_PHASES_LIMIT);
    let max = max_phases as u64;

    // ── Rational factor L/M ──────────────────────────────────────

    let (up, down) = best_rational(sr_out / sr_in, max);

    // ── Prototype filter design ──────────────────────────────────

    // Cutoff normalized to upsampled rate: fc = 1 / (2 · max(L, M)).
    // Clamp strictly below 0.5 (Nyquist) for the passthrough case (L=M=1).
    let max_factor = up.max(down) as f64;
    let cutoff_normalized = (1.0 / (2.0 * max_factor)).min(0.5 - 1e-8);

    // Quality → attenuation and transition width
    let (attenuation_db, transition_fraction) = quality_params(quality);
    let delta_f = cutoff_normalized * transition_fraction;

    // Kaiser's formula for filter order
    let order_est = (attenuation_db - 7.95) / (2.285 * 2.0 * std::f64::consts::PI * delta_f);
    let order = (order_est.ceil() as usize).clamp(16, 4096);
    // Ensure even (Type I FIR)
    let order = if order.is_multiple_of(2) {
        order
    } else {
        (order + 1).min(4096)
    };

    // The upsampled sample rate for the prototype filter is L * input_rate.
    let upsampled_rate = (up as f64) * sr_in;
    let cutoff_hz = cutoff_normalized * upsampled_rate;

    let fir_result = super::fir::design(
        crate::types::FilterType::Lowpass,
        order,
        from_f64(upsampled_rate)?,
        from_f64(cutoff_hz)?,
        None,
        window,
    )?;

    ResampleSpec::new(
        up as usize,
        down as usize,
        fir_result.coefficients().to_vec(),
        mode,
    )
}

// ─── Internals ───────────────────────────────────────────────────────

/// Find the best rational approximation p/q to `x` with p,q ≤ `max_val`,
/// using the continued-fraction algorithm.
///
/// For exact rational values like 48000/44100, this produces the exact
/// reduced fraction (160/147).  For irrational or high-precision ratios,
/// it finds the closest L/M within the bound.
fn best_rational(x: f64, max_val: u64) -> (u64, u64) {
    if x <= 0.0 {
        return (0, 1);
    }

    // Convergents of the continued fraction expansion.
    // h[n] / k[n] are the successive best rational approximations.
    //
    // Recurrence: h[n] = a[n] * h[n-1] + h[n-2]
    //             k[n] = a[n] * k[n-1] + k[n-2]
    //
    // Seed values so the recurrence produces h[0]=a[0], k[0]=1.
    let (mut h2, mut k2): (u64, u64) = (0, 1); // h[-2], k[-2]
    let (mut h1, mut k1): (u64, u64) = (1, 0); // h[-1], k[-1]

    let mut remainder = x;
    let mut best_h = 1_u64;
    let mut best_k = 1_u64;

    for _ in 0..64 {
        let a = remainder.floor() as u64;

        let h = a.saturating_mul(h1).saturating_add(h2);
        let k = a.saturating_mul(k1).saturating_add(k2);

        if h > max_val || k > max_val {
            // This convergent exceeds the bound.  Try the best
            // semiconvergent: largest a' ≤ a that still fits.
            if h1 > 0 && k1 > 0 {
                let a_h = (max_val - h2) / h1;
                let a_k = (max_val - k2) / k1;
                let a_clamp = a_h.min(a_k);
                if a_clamp > 0 {
                    let h_semi = a_clamp * h1 + h2;
                    let k_semi = a_clamp * k1 + k2;
                    if h_semi > 0 && k_semi > 0 {
                        let err_best = (best_h as f64 / best_k as f64 - x).abs();
                        let err_semi = (h_semi as f64 / k_semi as f64 - x).abs();
                        if err_semi < err_best {
                            best_h = h_semi;
                            best_k = k_semi;
                        }
                    }
                }
            }
            break;
        }

        if h > 0 && k > 0 {
            best_h = h;
            best_k = k;
        }

        h2 = h1;
        k2 = k1;
        h1 = h;
        k1 = k;

        let frac = remainder - remainder.floor();
        if frac < 1e-12 {
            break; // exact rational — CF terminates
        }
        remainder = 1.0 / frac;
    }

    (best_h, best_k)
}

/// Map quality to (attenuation in dB, transition width fraction).
fn quality_params(quality: ResampleQuality) -> (f64, f64) {
    let t = f64::from(quality.value()) / f64::from(ResampleQuality::MAX);
    // Attenuation: 40 dB (quality 0) → 120 dB (quality 10)
    let attenuation_db = t.mul_add(80.0, 40.0);
    // Transition fraction of cutoff: 0.45 (quality 0) → 0.10 (quality 10)
    let transition_fraction = t.mul_add(-0.35, 0.45);
    (attenuation_db, transition_fraction)
}

fn to_f64<T: Float>(val: T) -> Result<f64> {
    val.to_f64()
        .ok_or_else(|| Error::Internal("failed to convert to f64".into()))
}

fn from_f64<T: Float>(val: f64) -> Result<T> {
    T::from(val).ok_or_else(|| Error::Internal("failed to convert from f64".into()))
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;

    fn q(val: u8) -> ResampleQuality {
        ResampleQuality::new(val).expect("valid quality")
    }

    fn designed(sr_in: f64, sr_out: f64, quality: u8) -> ResampleSpec<f64> {
        design(
            sr_in,
            sr_out,
            q(quality),
            &Window::Hamming,
            DEFAULT_MAX_PHASES,
            ResampleMode::Streaming,
        )
        .expect("valid design")
    }

    fn designed_with_phases(
        sr_in: f64,
        sr_out: f64,
        quality: u8,
        max_phases: usize,
    ) -> ResampleSpec<f64> {
        design(
            sr_in,
            sr_out,
            q(quality),
            &Window::Hamming,
            max_phases,
            ResampleMode::Streaming,
        )
        .expect("valid design")
    }

    // ── ResampleQuality ──────────────────────────────────────────

    #[test]
    fn quality_valid_range() {
        for v in 0..=10 {
            assert!(
                ResampleQuality::new(v).is_ok(),
                "quality {v} should be valid"
            );
        }
    }

    #[test]
    fn quality_out_of_range() {
        assert!(
            ResampleQuality::new(11).is_err(),
            "quality 11 should be rejected"
        );
        assert!(
            ResampleQuality::new(255).is_err(),
            "quality 255 should be rejected"
        );
    }

    // ── Rate validation (now in design()) ──────────────────────────

    #[test]
    fn design_zero_input_rate_is_error() {
        assert!(
            design(
                0.0,
                48000.0,
                q(5),
                &Window::<f64>::Hamming,
                DEFAULT_MAX_PHASES,
                ResampleMode::Streaming
            )
            .is_err(),
            "zero input rate should be rejected"
        );
    }

    #[test]
    fn design_negative_input_rate_is_error() {
        assert!(
            design(
                -44100.0,
                48000.0,
                q(5),
                &Window::<f64>::Hamming,
                DEFAULT_MAX_PHASES,
                ResampleMode::Streaming
            )
            .is_err(),
            "negative input rate should be rejected"
        );
    }

    #[test]
    fn design_zero_output_rate_is_error() {
        assert!(
            design(
                44100.0,
                0.0,
                q(5),
                &Window::<f64>::Hamming,
                DEFAULT_MAX_PHASES,
                ResampleMode::Streaming
            )
            .is_err(),
            "zero output rate should be rejected"
        );
    }

    #[test]
    fn design_negative_output_rate_is_error() {
        assert!(
            design(
                44100.0,
                -48000.0,
                q(5),
                &Window::<f64>::Hamming,
                DEFAULT_MAX_PHASES,
                ResampleMode::Streaming
            )
            .is_err(),
            "negative output rate should be rejected"
        );
    }

    // ── ResampleSpec direct construction ────────────────────────────

    #[test]
    fn spec_accessors() {
        let spec = ResampleSpec::new(
            3,
            2,
            vec![0.25, 0.5, 1.0, 0.5, 0.25],
            ResampleMode::Streaming,
        )
        .expect("valid resample spec");
        assert_eq!(spec.up_factor(), 3, "up_factor");
        assert_eq!(spec.down_factor(), 2, "down_factor");
        assert_eq!(spec.prototype_filter().len(), 5, "prototype length");
        assert_eq!(spec.mode(), ResampleMode::Streaming, "mode");
    }

    #[test]
    fn spec_with_mode() {
        let spec = ResampleSpec::new(2, 1, vec![1.0_f64], ResampleMode::Streaming)
            .expect("valid resample spec")
            .with_mode(ResampleMode::Batch);
        assert_eq!(
            spec.mode(),
            ResampleMode::Batch,
            "mode should be overridden"
        );
    }

    #[test]
    fn spec_rejects_zero_up_factor() {
        assert!(
            ResampleSpec::new(0, 1, vec![1.0_f64], ResampleMode::Streaming).is_err(),
            "zero up_factor should be rejected"
        );
    }

    #[test]
    fn spec_rejects_zero_down_factor() {
        assert!(
            ResampleSpec::new(1, 0, vec![1.0_f64], ResampleMode::Streaming).is_err(),
            "zero down_factor should be rejected"
        );
    }

    #[test]
    fn spec_rejects_empty_prototype() {
        assert!(
            ResampleSpec::<f64>::new(1, 1, vec![], ResampleMode::Streaming).is_err(),
            "empty prototype filter should be rejected"
        );
    }

    // ── Rational approximation ───────────────────────────────────

    #[test]
    fn rational_44100_to_48000() {
        let (up, down) = best_rational(48000.0 / 44100.0, 256);
        assert_eq!(up, 160, "L should be 160");
        assert_eq!(down, 147, "M should be 147");
    }

    #[test]
    fn rational_48000_to_44100() {
        let (up, down) = best_rational(44100.0 / 48000.0, 256);
        assert_eq!(up, 147, "L should be 147");
        assert_eq!(down, 160, "M should be 160");
    }

    #[test]
    fn rational_2x_decimation() {
        let (up, down) = best_rational(22050.0 / 44100.0, 256);
        assert_eq!(up, 1, "L should be 1 (pure decimation)");
        assert_eq!(down, 2, "M should be 2");
    }

    #[test]
    fn rational_2x_interpolation() {
        let (up, down) = best_rational(44100.0 / 22050.0, 256);
        assert_eq!(up, 2, "L should be 2 (pure interpolation)");
        assert_eq!(down, 1, "M should be 1");
    }

    #[test]
    fn rational_same_rate() {
        let (up, down) = best_rational(1.0, 256);
        assert_eq!(up, 1, "L should be 1 (passthrough)");
        assert_eq!(down, 1, "M should be 1 (passthrough)");
    }

    #[test]
    fn rational_fractional_rate() {
        // 44100.5 → 48000: no exact integer ratio, should approximate.
        let ratio = 48000.0 / 44100.5;
        let (up, down) = best_rational(ratio, 256);
        assert!(up <= 256, "L should be within bound");
        assert!(down <= 256, "M should be within bound");
        // The approximation should be close to the true ratio.
        let approx = up as f64 / down as f64;
        let err = (approx - ratio).abs();
        assert!(
            err < 1e-4,
            "approximation {up}/{down} = {approx} should be close to {ratio}, err={err}"
        );
    }

    #[test]
    fn rational_respects_max_bound() {
        // With max=10, 160/147 won't fit — should find best within bound.
        let (up, down) = best_rational(48000.0 / 44100.0, 10);
        assert!(up <= 10, "L={up} should be ≤ 10");
        assert!(down <= 10, "M={down} should be ≤ 10");
        // Should still be a reasonable approximation.
        let ratio = 48000.0 / 44100.0;
        let approx = up as f64 / down as f64;
        let err = (approx - ratio).abs();
        assert!(
            err < 0.05,
            "approximation {up}/{down} = {approx} should be close to {ratio}, err={err}"
        );
    }

    // ── Full design tests ────────────────────────────────────────

    #[test]
    fn design_44100_to_48000() {
        let result = designed(44100.0, 48000.0, 5);
        assert_eq!(result.up_factor(), 160, "L=160");
        assert_eq!(result.down_factor(), 147, "M=147");
        assert!(
            !result.prototype_filter().is_empty(),
            "prototype filter should not be empty"
        );
        // Even order → odd number of taps
        assert!(
            !result.prototype_filter().len().is_multiple_of(2),
            "filter length should be odd (even order + 1) — got {}",
            result.prototype_filter().len()
        );
    }

    #[test]
    fn design_passthrough() {
        let result = designed(44100.0, 44100.0, 3);
        assert_eq!(result.up_factor(), 1, "L=1");
        assert_eq!(result.down_factor(), 1, "M=1");
    }

    #[test]
    fn design_decimation() {
        let result = designed(44100.0, 22050.0, 5);
        assert_eq!(result.up_factor(), 1, "L=1");
        assert_eq!(result.down_factor(), 2, "M=2");
    }

    #[test]
    fn design_interpolation() {
        let result = designed(22050.0, 44100.0, 5);
        assert_eq!(result.up_factor(), 2, "L=2");
        assert_eq!(result.down_factor(), 1, "M=1");
    }

    #[test]
    fn prototype_is_lowpass() {
        let result = designed(44100.0, 48000.0, 5);
        let dc_gain: f64 = result.prototype_filter().iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 1e-8,
            "prototype DC gain should be 1.0, got {dc_gain}"
        );
    }

    #[test]
    fn prototype_is_symmetric() {
        let result = designed(44100.0, 48000.0, 5);
        let taps = result.prototype_filter();
        let n = taps.len();
        for i in 0..n / 2 {
            assert!(
                (taps[i] - taps[n - 1 - i]).abs() < 1e-10,
                "prototype taps should be symmetric at index {i}"
            );
        }
    }

    #[test]
    fn quality_increases_order() {
        // Use a small ratio so orders don't all clamp to 4096.
        let low = designed(22050.0, 44100.0, 0);
        let mid = designed(22050.0, 44100.0, 5);
        let high = designed(22050.0, 44100.0, 10);
        assert!(
            low.prototype_filter().len() < mid.prototype_filter().len(),
            "higher quality should produce longer filter: q0={} vs q5={}",
            low.prototype_filter().len(),
            mid.prototype_filter().len()
        );
        assert!(
            mid.prototype_filter().len() < high.prototype_filter().len(),
            "higher quality should produce longer filter: q5={} vs q10={}",
            mid.prototype_filter().len(),
            high.prototype_filter().len()
        );
    }

    #[test]
    fn design_with_custom_max_phases() {
        // With max_phases=10, 160/147 won't fit — uses approximation.
        let result = designed_with_phases(44100.0, 48000.0, 5, 10);
        assert!(
            result.up_factor() <= 10,
            "L should be ≤ 10, got {}",
            result.up_factor()
        );
        assert!(
            result.down_factor() <= 10,
            "M should be ≤ 10, got {}",
            result.down_factor()
        );
    }

    // ── f32 path ─────────────────────────────────────────────────

    #[test]
    fn f32_design_works() {
        let result = design(
            44100.0_f32,
            48000.0_f32,
            q(5),
            &Window::Hamming,
            DEFAULT_MAX_PHASES,
            ResampleMode::Streaming,
        )
        .expect("f32 design");
        assert_eq!(result.up_factor(), 160, "L=160");
        assert_eq!(result.down_factor(), 147, "M=147");
        let dc_gain: f32 = result.prototype_filter().iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 1e-4,
            "f32 prototype DC gain should be ~1.0, got {dc_gain}"
        );
    }

    // ── Scipy reference comparison ───────────────────────────────
    //
    // Generated by scripts/gen_resample_reference.py.
    // Regenerate with: make gen-resample-reference

    #[allow(
        clippy::wildcard_imports,
        reason = "bulk golden-vector import in tests"
    )]
    use omnidsp_testdata::resample_scipy::*;

    /// Evaluate FIR magnitude response at a given digital frequency.
    fn fir_magnitude(taps: &[f64], omega: f64) -> f64 {
        let (re, im) = taps
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(re, im), (n, &h)| {
                let phase = omega * n as f64;
                (h.mul_add(phase.cos(), re), h.mul_add(-phase.sin(), im))
            });
        re.hypot(im)
    }

    /// Assert two FIR filters produce the same frequency response.
    fn assert_fir_response_match(ours: &[f64], scipy: &[f64], fs: f64, tol: f64, label: &str) {
        for i in 1..1000 {
            let freq = fs / 2.0 * f64::from(i) / 1000.0;
            let omega = 2.0 * std::f64::consts::PI * freq / fs;
            let g_ours = fir_magnitude(ours, omega);
            let g_scipy = fir_magnitude(scipy, omega);
            let err = (g_ours - g_scipy).abs();
            assert!(
                err < tol,
                "{label}: mismatch at {freq:.1} Hz — ours={g_ours:.12e}, scipy={g_scipy:.12e}, err={err:.2e}"
            );
        }
    }

    /// Helper: design with `max_phases` high enough for exact integer-rate results.
    fn design_exact(sr_in: f64, sr_out: f64, quality: u8) -> ResampleSpec<f64> {
        designed_with_phases(sr_in, sr_out, quality, MAX_PHASES_LIMIT)
    }

    #[test]
    fn scipy_44100_to_22050() {
        let result = design_exact(44100.0, 22050.0, 5);
        assert_eq!(result.up_factor(), RESAMPLE_44100_22050_Q5_UP, "L mismatch");
        assert_eq!(
            result.down_factor(),
            RESAMPLE_44100_22050_Q5_DOWN,
            "M mismatch"
        );
        assert_eq!(
            result.prototype_filter().len(),
            RESAMPLE_44100_22050_Q5_ORDER + 1,
            "order mismatch"
        );
        assert_fir_response_match(
            result.prototype_filter(),
            RESAMPLE_44100_22050_Q5_TAPS,
            44100.0,
            1e-10,
            "44100→22050",
        );
    }

    #[test]
    fn scipy_22050_to_44100() {
        let result = design_exact(22050.0, 44100.0, 5);
        assert_eq!(result.up_factor(), RESAMPLE_22050_44100_Q5_UP, "L mismatch");
        assert_eq!(
            result.down_factor(),
            RESAMPLE_22050_44100_Q5_DOWN,
            "M mismatch"
        );
        assert_eq!(
            result.prototype_filter().len(),
            RESAMPLE_22050_44100_Q5_ORDER + 1,
            "order mismatch"
        );
        let upsampled_rate = f64::from(result.up_factor() as u32) * 22050.0;
        assert_fir_response_match(
            result.prototype_filter(),
            RESAMPLE_22050_44100_Q5_TAPS,
            upsampled_rate,
            1e-10,
            "22050→44100",
        );
    }

    #[test]
    fn scipy_44100_to_44100() {
        let result = design_exact(44100.0, 44100.0, 5);
        assert_eq!(result.up_factor(), RESAMPLE_44100_44100_Q5_UP, "L mismatch");
        assert_eq!(
            result.down_factor(),
            RESAMPLE_44100_44100_Q5_DOWN,
            "M mismatch"
        );
        assert_eq!(
            result.prototype_filter().len(),
            RESAMPLE_44100_44100_Q5_ORDER + 1,
            "order mismatch"
        );
        assert_fir_response_match(
            result.prototype_filter(),
            RESAMPLE_44100_44100_Q5_TAPS,
            44100.0,
            1e-10,
            "44100→44100",
        );
    }

    #[test]
    fn scipy_48000_to_16000() {
        let result = design_exact(48000.0, 16000.0, 3);
        assert_eq!(result.up_factor(), RESAMPLE_48000_16000_Q3_UP, "L mismatch");
        assert_eq!(
            result.down_factor(),
            RESAMPLE_48000_16000_Q3_DOWN,
            "M mismatch"
        );
        assert_eq!(
            result.prototype_filter().len(),
            RESAMPLE_48000_16000_Q3_ORDER + 1,
            "order mismatch"
        );
        assert_fir_response_match(
            result.prototype_filter(),
            RESAMPLE_48000_16000_Q3_TAPS,
            48000.0,
            1e-10,
            "48000→16000",
        );
    }

    #[test]
    fn scipy_16000_to_48000() {
        let result = design_exact(16000.0, 48000.0, 3);
        assert_eq!(result.up_factor(), RESAMPLE_16000_48000_Q3_UP, "L mismatch");
        assert_eq!(
            result.down_factor(),
            RESAMPLE_16000_48000_Q3_DOWN,
            "M mismatch"
        );
        assert_eq!(
            result.prototype_filter().len(),
            RESAMPLE_16000_48000_Q3_ORDER + 1,
            "order mismatch"
        );
        let upsampled_rate = f64::from(result.up_factor() as u32) * 16000.0;
        assert_fir_response_match(
            result.prototype_filter(),
            RESAMPLE_16000_48000_Q3_TAPS,
            upsampled_rate,
            1e-10,
            "16000→48000",
        );
    }

    // ── Large-ratio cases (44100 ↔ 48000) ───────────────────────

    #[test]
    fn scipy_44100_to_48000_q0() {
        let result = design_exact(44100.0, 48000.0, 0);
        assert_eq!(result.up_factor(), RESAMPLE_44100_48000_Q0_UP, "L mismatch");
        assert_eq!(
            result.down_factor(),
            RESAMPLE_44100_48000_Q0_DOWN,
            "M mismatch"
        );
        assert_eq!(
            result.prototype_filter().len(),
            RESAMPLE_44100_48000_Q0_ORDER + 1,
            "order mismatch"
        );
        let upsampled_rate = f64::from(result.up_factor() as u32) * 44100.0;
        assert_fir_response_match(
            result.prototype_filter(),
            RESAMPLE_44100_48000_Q0_TAPS,
            upsampled_rate,
            1e-10,
            "44100→48000 q0",
        );
    }

    #[test]
    fn scipy_44100_to_48000_q5() {
        let result = design_exact(44100.0, 48000.0, 5);
        assert_eq!(result.up_factor(), RESAMPLE_44100_48000_Q5_UP, "L mismatch");
        assert_eq!(
            result.down_factor(),
            RESAMPLE_44100_48000_Q5_DOWN,
            "M mismatch"
        );
        assert_eq!(
            result.prototype_filter().len(),
            RESAMPLE_44100_48000_Q5_ORDER + 1,
            "order mismatch"
        );
        let upsampled_rate = f64::from(result.up_factor() as u32) * 44100.0;
        assert_fir_response_match(
            result.prototype_filter(),
            RESAMPLE_44100_48000_Q5_TAPS,
            upsampled_rate,
            1e-10,
            "44100→48000 q5",
        );
    }

    #[test]
    fn scipy_48000_to_44100_q5() {
        let result = design_exact(48000.0, 44100.0, 5);
        assert_eq!(result.up_factor(), RESAMPLE_48000_44100_Q5_UP, "L mismatch");
        assert_eq!(
            result.down_factor(),
            RESAMPLE_48000_44100_Q5_DOWN,
            "M mismatch"
        );
        assert_eq!(
            result.prototype_filter().len(),
            RESAMPLE_48000_44100_Q5_ORDER + 1,
            "order mismatch"
        );
        let upsampled_rate = f64::from(result.up_factor() as u32) * 48000.0;
        assert_fir_response_match(
            result.prototype_filter(),
            RESAMPLE_48000_44100_Q5_TAPS,
            upsampled_rate,
            1e-10,
            "48000→44100 q5",
        );
    }
}
