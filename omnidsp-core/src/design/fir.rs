// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FIR filter design via the windowed-sinc method.
//!
//! Produces a `Vec<T>` of tap coefficients ready to pass to
//! [`Fir::create_plan`](crate::traits::fir::Fir::create_plan).

#![allow(
    clippy::cast_precision_loss,
    reason = "sample indices are small enough for f64"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "sinc formulas are clearer without mul_add"
)]

use std::f64::consts::PI;

use num_traits::Float;

use crate::error::{Error, Result};
use crate::types::{FilterType, Window, WindowFn};

// ─── Public API ───────────────────────────────────────────────────────

/// Design a FIR filter using the windowed-sinc method.
///
/// Returns `order + 1` tap coefficients.  The taps are gain-normalized:
/// lowpass filters have unity DC gain, highpass filters have unity gain at
/// Nyquist, and bandpass/bandstop filters have unity gain at the band center
/// or at DC respectively.
///
/// # Arguments
///
/// * `filter_type` — frequency response shape.
/// * `order` — filter order (number of taps is `order + 1`).
/// * `sample_rate` — sampling frequency (Hz).
/// * `cutoff1` — primary cutoff frequency (Hz).
/// * `cutoff2` — secondary cutoff (Hz), required for bandpass/bandstop.
/// * `window_fn` — window function to apply to the ideal impulse response.
///
/// # Errors
///
/// Returns [`Error::InvalidSpec`] if:
/// - `order` is zero
/// - `sample_rate` is not positive
/// - cutoff frequencies are outside `(0, sample_rate / 2)`
/// - bandpass/bandstop is missing `cutoff2` or has `cutoff2 <= cutoff1`
/// - lowpass/highpass has `cutoff2` present
pub fn design<T: Float>(
    filter_type: FilterType,
    order: usize,
    sample_rate: T,
    cutoff1: T,
    cutoff2: Option<T>,
    window_fn: &WindowFn<T>,
) -> Result<Vec<T>> {
    let sr = to_f64(sample_rate)?;
    let c1 = to_f64(cutoff1)?;
    let c2 = cutoff2.map(to_f64).transpose()?;

    validate(filter_type, order, sr, c1, c2)?;

    let num_taps = order + 1;
    let fn1 = c1 / sr; // normalized cutoff1

    // Compute ideal impulse response
    let ideal = match filter_type {
        FilterType::Lowpass => ideal_lowpass(num_taps, fn1),
        FilterType::Highpass => ideal_highpass(num_taps, fn1),
        FilterType::Bandpass => {
            // validate() ensures c2 is present for bandpass
            let fn2 = c2.unwrap_or(0.0) / sr;
            ideal_bandpass(num_taps, fn1, fn2)
        }
        FilterType::Bandstop => {
            let fn2 = c2.unwrap_or(0.0) / sr;
            ideal_bandstop(num_taps, fn1, fn2)
        }
    };

    // Generate and apply window
    let window = Window::from_fn(window_fn, num_taps)?;
    let windowed: Vec<f64> = ideal
        .iter()
        .zip(window.coefficients())
        .map(|(h, w)| *h * to_f64(*w).unwrap_or(0.0))
        .collect();

    // Normalize gain
    let normalized = normalize(filter_type, &windowed, fn1, c2.map(|c| c / sr));

    // Convert to target type
    normalized.iter().map(|&v| from_f64(v)).collect()
}

/// Estimate FIR filter order using Kaiser's formula.
///
/// Given a transition band width (normalized to sample rate, i.e. in the range
/// `(0, 0.5)`) and desired stopband attenuation in positive dB, returns the
/// estimated filter order (always even, minimum 2).
///
/// # Examples
///
/// ```
/// use omnidsp_core::design::fir::estimate_order;
///
/// // 1 kHz transition width at 44.1 kHz, 60 dB attenuation
/// let order = estimate_order(1000.0 / 44100.0, 60.0);
/// assert!(order >= 50, "order should be reasonable for these specs");
/// ```
#[must_use]
pub fn estimate_order(transition_width: f64, stopband_attenuation_db: f64) -> usize {
    let tw = transition_width.abs().max(1e-10);
    let atten = stopband_attenuation_db.abs().max(21.0);

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "order estimate is a small positive value"
    )]
    let order = ((atten - 7.95) / (2.285 * 2.0 * PI * tw)).ceil() as usize;

    // Ensure even order (Type I FIR), minimum 2
    let order = order.max(2);
    if order.is_multiple_of(2) {
        order
    } else {
        order + 1
    }
}

// ─── Validation ───────────────────────────────────────────────────────

fn validate(
    filter_type: FilterType,
    order: usize,
    sr: f64,
    c1: f64,
    c2: Option<f64>,
) -> Result<()> {
    if order == 0 {
        return Err(Error::InvalidSpec("FIR order must be non-zero".into()));
    }
    if sr <= 0.0 {
        return Err(Error::InvalidSpec("sample rate must be positive".into()));
    }
    let nyquist = sr / 2.0;
    if c1 <= 0.0 || c1 >= nyquist {
        return Err(Error::InvalidSpec(format!(
            "cutoff1 ({c1}) must be in (0, {nyquist})"
        )));
    }

    match filter_type {
        FilterType::Lowpass | FilterType::Highpass => {
            if c2.is_some() {
                return Err(Error::InvalidSpec(
                    "cutoff2 must not be specified for lowpass/highpass".into(),
                ));
            }
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            let Some(c2_val) = c2 else {
                return Err(Error::InvalidSpec(
                    "cutoff2 is required for bandpass/bandstop".into(),
                ));
            };
            if c2_val <= 0.0 || c2_val >= nyquist {
                return Err(Error::InvalidSpec(format!(
                    "cutoff2 ({c2_val}) must be in (0, {nyquist})"
                )));
            }
            if c2_val <= c1 {
                return Err(Error::InvalidSpec(format!(
                    "cutoff2 ({c2_val}) must be greater than cutoff1 ({c1})"
                )));
            }
        }
    }
    Ok(())
}

// ─── Conversion helpers ───────────────────────────────────────────────

fn to_f64<T: Float>(val: T) -> Result<f64> {
    val.to_f64()
        .ok_or_else(|| Error::Internal("failed to convert to f64".into()))
}

fn from_f64<T: Float>(val: f64) -> Result<T> {
    T::from(val).ok_or_else(|| Error::Internal("failed to convert from f64".into()))
}

// ─── Ideal impulse responses ──────────────────────────────────────────

/// Lowpass sinc: `h[n] = sin(2π·fc·m) / (π·m)` where `m = n - (N-1)/2`.
fn ideal_lowpass(num_taps: usize, fc: f64) -> Vec<f64> {
    let center = (num_taps - 1) as f64 / 2.0;
    (0..num_taps)
        .map(|n| {
            let m = n as f64 - center;
            if m.abs() < 1e-10 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * m).sin() / (PI * m)
            }
        })
        .collect()
}

/// Highpass: `δ[n] - h_lp[n]` (spectral inversion of lowpass).
fn ideal_highpass(num_taps: usize, fc: f64) -> Vec<f64> {
    let center = (num_taps - 1) as f64 / 2.0;
    let lp = ideal_lowpass(num_taps, fc);
    lp.iter()
        .enumerate()
        .map(|(n, &h)| {
            let m = n as f64 - center;
            let delta = if m.abs() < 1e-10 { 1.0 } else { 0.0 };
            delta - h
        })
        .collect()
}

/// Bandpass: `h_lp(fc2) - h_lp(fc1)`.
fn ideal_bandpass(num_taps: usize, fc1: f64, fc2: f64) -> Vec<f64> {
    let lp1 = ideal_lowpass(num_taps, fc1);
    let lp2 = ideal_lowpass(num_taps, fc2);
    lp1.iter().zip(&lp2).map(|(a, b)| b - a).collect()
}

/// Bandstop: `δ[n] - (h_lp(fc2) - h_lp(fc1))`.
fn ideal_bandstop(num_taps: usize, fc1: f64, fc2: f64) -> Vec<f64> {
    let center = (num_taps - 1) as f64 / 2.0;
    let bp = ideal_bandpass(num_taps, fc1, fc2);
    bp.iter()
        .enumerate()
        .map(|(n, &h)| {
            let m = n as f64 - center;
            let delta = if m.abs() < 1e-10 { 1.0 } else { 0.0 };
            delta - h
        })
        .collect()
}

// ─── Gain normalization ───────────────────────────────────────────────

/// Normalize filter taps so that the gain at the reference frequency is unity.
fn normalize(filter_type: FilterType, taps: &[f64], fn1: f64, fn2: Option<f64>) -> Vec<f64> {
    let gain = match filter_type {
        // DC gain (ω = 0): sum of all taps.
        FilterType::Lowpass | FilterType::Bandstop => taps.iter().sum::<f64>(),
        // Nyquist gain (ω = π): alternating sum.
        FilterType::Highpass => taps
            .iter()
            .enumerate()
            .map(|(i, &h)| if i % 2 == 0 { h } else { -h })
            .sum::<f64>(),
        // Gain at band center.
        FilterType::Bandpass => {
            let fc = f64::midpoint(fn1, fn2.unwrap_or(fn1));
            eval_gain(taps, fc)
        }
    };

    if gain.abs() < 1e-15 {
        return taps.to_vec();
    }
    taps.iter().map(|&h| h / gain).collect()
}

/// Evaluate the filter gain at normalized frequency `f` (0..0.5).
fn eval_gain(taps: &[f64], f: f64) -> f64 {
    let omega = 2.0 * PI * f;
    taps.iter()
        .enumerate()
        .map(|(n, &h)| h * (omega * n as f64).cos())
        .sum()
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    // ── Lowpass ──────────────────────────────────────────────────

    #[test]
    fn lowpass_dc_gain_is_unity() {
        let taps = design::<f64>(
            FilterType::Lowpass,
            30,
            44100.0,
            1000.0,
            None,
            &WindowFn::Hamming,
        )
        .expect("lowpass design should succeed");

        let dc_gain: f64 = taps.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < TOL,
            "lowpass DC gain should be 1.0, got {dc_gain}"
        );
    }

    #[test]
    fn lowpass_correct_length() {
        let order = 64;
        let taps = design::<f64>(
            FilterType::Lowpass,
            order,
            44100.0,
            1000.0,
            None,
            &WindowFn::Hann,
        )
        .expect("lowpass design should succeed");

        assert_eq!(taps.len(), order + 1, "should have order+1 taps");
    }

    #[test]
    fn lowpass_symmetric() {
        let taps = design::<f64>(
            FilterType::Lowpass,
            30,
            44100.0,
            1000.0,
            None,
            &WindowFn::Hamming,
        )
        .expect("lowpass design should succeed");

        let n = taps.len();
        for i in 0..n / 2 {
            assert!(
                (taps[i] - taps[n - 1 - i]).abs() < TOL,
                "lowpass taps should be symmetric at index {i}"
            );
        }
    }

    #[test]
    fn lowpass_rectangular_is_pure_sinc() {
        // With a rectangular window, the taps are the ideal sinc (normalized).
        let taps = design::<f64>(
            FilterType::Lowpass,
            4,
            2.0, // fs=2 → Nyquist=1, fc=0.25 means quarter-band
            0.25,
            None,
            &WindowFn::Rectangular,
        )
        .expect("lowpass design should succeed");

        // 5 taps, center is index 2
        // Ideal sinc at fc=0.25 (normalized to fs):
        //   fn = 0.25/2.0 = 0.125
        //   h[n] = sin(2π·0.125·m) / (π·m) for m = n-2
        //   h[0]: m=-2, sin(-π/2)/(-2π) = -1/(-2π) = 1/(2π)
        //   h[1]: m=-1, sin(-π/4)/(-π) = (√2/2)/π
        //   h[2]: m=0,  2·0.125 = 0.25
        //   h[3]: m=1,  sin(π/4)/π = (√2/2)/π
        //   h[4]: m=2,  sin(π/2)/(2π) = 1/(2π)
        // These are then divided by their sum for normalization.
        assert_eq!(taps.len(), 5, "should have 5 taps");
        // Verify symmetry
        assert!(
            (taps[0] - taps[4]).abs() < TOL,
            "endpoints should be symmetric"
        );
        assert!(
            (taps[1] - taps[3]).abs() < TOL,
            "inner taps should be symmetric"
        );
        // Center tap should be the largest
        assert!(
            taps[2] > taps[1] && taps[2] > taps[0],
            "center tap should be largest"
        );
        // DC gain should be 1
        let dc: f64 = taps.iter().sum();
        assert!((dc - 1.0).abs() < TOL, "DC gain should be 1.0, got {dc}");
    }

    // ── Highpass ─────────────────────────────────────────────────

    #[test]
    fn highpass_nyquist_gain_is_unity() {
        let taps = design::<f64>(
            FilterType::Highpass,
            30,
            44100.0,
            10000.0,
            None,
            &WindowFn::Hamming,
        )
        .expect("highpass design should succeed");

        // Alternating sum = gain at Nyquist
        let nyquist_gain: f64 = taps
            .iter()
            .enumerate()
            .map(|(i, &h)| if i % 2 == 0 { h } else { -h })
            .sum();
        assert!(
            (nyquist_gain - 1.0).abs() < TOL,
            "highpass Nyquist gain should be 1.0, got {nyquist_gain}"
        );
    }

    #[test]
    fn highpass_dc_gain_near_zero() {
        let taps = design::<f64>(
            FilterType::Highpass,
            30,
            44100.0,
            10000.0,
            None,
            &WindowFn::Hamming,
        )
        .expect("highpass design should succeed");

        let dc_gain: f64 = taps.iter().sum();
        assert!(
            dc_gain.abs() < 0.01,
            "highpass DC gain should be near 0, got {dc_gain}"
        );
    }

    // ── Bandpass ─────────────────────────────────────────────────

    #[test]
    fn bandpass_gain_at_center() {
        let taps = design::<f64>(
            FilterType::Bandpass,
            60,
            44100.0,
            1000.0,
            Some(5000.0),
            &WindowFn::Hamming,
        )
        .expect("bandpass design should succeed");

        // Evaluate gain at band center (3000 Hz)
        let fc = 3000.0 / 44100.0;
        let gain = eval_gain(&taps, fc);
        assert!(
            (gain - 1.0).abs() < 0.05,
            "bandpass gain at center should be ≈1.0, got {gain}"
        );
    }

    #[test]
    fn bandpass_dc_gain_near_zero() {
        let taps = design::<f64>(
            FilterType::Bandpass,
            60,
            44100.0,
            1000.0,
            Some(5000.0),
            &WindowFn::Hamming,
        )
        .expect("bandpass design should succeed");

        // DC gain is small but not zero at moderate order with windowing.
        let dc_gain: f64 = taps.iter().sum();
        assert!(
            dc_gain.abs() < 0.1,
            "bandpass DC gain should be near 0, got {dc_gain}"
        );
    }

    // ── Bandstop ─────────────────────────────────────────────────

    #[test]
    fn bandstop_dc_gain_is_unity() {
        let taps = design::<f64>(
            FilterType::Bandstop,
            60,
            44100.0,
            1000.0,
            Some(5000.0),
            &WindowFn::Hamming,
        )
        .expect("bandstop design should succeed");

        let dc_gain: f64 = taps.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < TOL,
            "bandstop DC gain should be 1.0, got {dc_gain}"
        );
    }

    #[test]
    fn bandstop_rejection_at_center() {
        let taps = design::<f64>(
            FilterType::Bandstop,
            60,
            44100.0,
            1000.0,
            Some(5000.0),
            &WindowFn::Hamming,
        )
        .expect("bandstop design should succeed");

        // Gain at band center should be near zero
        let fc = 3000.0 / 44100.0;
        let gain = eval_gain(&taps, fc);
        assert!(
            gain.abs() < 0.1,
            "bandstop gain at center should be near 0, got {gain}"
        );
    }

    // ── Validation ───────────────────────────────────────────────

    #[test]
    fn order_zero_is_error() {
        assert!(
            design::<f64>(
                FilterType::Lowpass,
                0,
                44100.0,
                1000.0,
                None,
                &WindowFn::Hann
            )
            .is_err(),
            "order 0 should be rejected"
        );
    }

    #[test]
    fn negative_sample_rate_is_error() {
        assert!(
            design::<f64>(FilterType::Lowpass, 10, -1.0, 0.25, None, &WindowFn::Hann).is_err(),
            "negative sample rate should be rejected"
        );
    }

    #[test]
    fn cutoff_at_nyquist_is_error() {
        assert!(
            design::<f64>(
                FilterType::Lowpass,
                10,
                44100.0,
                22050.0,
                None,
                &WindowFn::Hann
            )
            .is_err(),
            "cutoff at Nyquist should be rejected"
        );
    }

    #[test]
    fn cutoff_above_nyquist_is_error() {
        assert!(
            design::<f64>(
                FilterType::Lowpass,
                10,
                44100.0,
                30000.0,
                None,
                &WindowFn::Hann
            )
            .is_err(),
            "cutoff above Nyquist should be rejected"
        );
    }

    #[test]
    fn bandpass_missing_cutoff2_is_error() {
        assert!(
            design::<f64>(
                FilterType::Bandpass,
                10,
                44100.0,
                1000.0,
                None,
                &WindowFn::Hann
            )
            .is_err(),
            "bandpass without cutoff2 should be rejected"
        );
    }

    #[test]
    fn bandpass_cutoff2_le_cutoff1_is_error() {
        assert!(
            design::<f64>(
                FilterType::Bandpass,
                10,
                44100.0,
                5000.0,
                Some(1000.0),
                &WindowFn::Hann
            )
            .is_err(),
            "cutoff2 <= cutoff1 should be rejected"
        );
    }

    #[test]
    fn lowpass_with_cutoff2_is_error() {
        assert!(
            design::<f64>(
                FilterType::Lowpass,
                10,
                44100.0,
                1000.0,
                Some(5000.0),
                &WindowFn::Hann
            )
            .is_err(),
            "lowpass with cutoff2 should be rejected"
        );
    }

    // ── Order estimation ─────────────────────────────────────────

    #[test]
    fn estimate_order_increases_with_attenuation() {
        let o1 = estimate_order(0.05, 40.0);
        let o2 = estimate_order(0.05, 80.0);
        assert!(
            o2 > o1,
            "higher attenuation should give higher order: {o1} vs {o2}"
        );
    }

    #[test]
    fn estimate_order_increases_with_narrow_transition() {
        let o1 = estimate_order(0.1, 60.0);
        let o2 = estimate_order(0.01, 60.0);
        assert!(
            o2 > o1,
            "narrower transition should give higher order: {o1} vs {o2}"
        );
    }

    #[test]
    fn estimate_order_is_even() {
        for tw in [0.01, 0.05, 0.1, 0.2] {
            for atten in [20.0, 40.0, 60.0, 80.0, 100.0] {
                let o = estimate_order(tw, atten);
                assert!(
                    o.is_multiple_of(2),
                    "order should be even for tw={tw}, atten={atten}: got {o}"
                );
            }
        }
    }

    // ── f32 path ─────────────────────────────────────────────────

    #[test]
    fn f32_lowpass_works() {
        let taps = design::<f32>(
            FilterType::Lowpass,
            30,
            44100.0_f32,
            1000.0_f32,
            None,
            &WindowFn::Hamming,
        )
        .expect("f32 lowpass should succeed");

        assert_eq!(taps.len(), 31, "should have 31 taps");
        let dc: f32 = taps.iter().sum();
        assert!(
            (dc - 1.0).abs() < 1e-5,
            "f32 lowpass DC gain should be ≈1.0, got {dc}"
        );
    }
}
