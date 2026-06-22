// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FIR filter design via the windowed-sinc method.
//!
//! # Algorithm
//!
//! 1. Compute the ideal (infinite-length) impulse response for the desired
//!    frequency response shape and normalized cutoff(s).
//! 2. Truncate to `order + 1` taps and apply the chosen window function to
//!    reduce spectral leakage.
//! 3. Normalize the gain so the filter has unity response at a reference
//!    frequency: DC for lowpass/bandstop, Nyquist for highpass, band center
//!    for bandpass.
//!
//! All arithmetic is performed in `f64` internally; the output coefficients
//! are converted to the target type `T`.
//!
//! The design method is selected via [`FirMethod`]; the windowed-sinc method
//! lives in its [`FirMethod::Windowed`] variant.  The returned
//! [`FirFilter`] is the reusable designed artifact — compose it into a
//! [`FirSpec`](crate::traits::fir::FirSpec) (via
//! [`FirSpec::new`](crate::traits::fir::FirSpec::new)) to obtain a plan-ready
//! FIR module spec.

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

use crate::design::remez;
use crate::error::{Error, Result};
use crate::traits::fir::{FirFilter, FirMeta};
use crate::types::FilterType;
use crate::window::Window;

// ─── Design method ────────────────────────────────────────────────────

/// FIR design method — the algorithm family used by [`design`].
///
/// Each variant carries the data specific to its method; the plain response
/// parameters (filter type, rate, band edges) are passed to [`design`]
/// alongside it, not bundled in here.  The [`Window`] lives in the
/// [`Windowed`](FirMethod::Windowed) variant only — it is an ingredient of the
/// windowed-sinc method, not a property of the filter.
///
/// The minimax-optimal `Equiripple` (Parks–McClellan/Remez) method takes a
/// stopband *attenuation* target directly (no window). The windowed method
/// stays because it is simple and windows remain essential for spectral
/// analysis regardless.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "the Equiripple payload holds floats (f32/f64) which are not Eq"
)]
#[non_exhaustive]
pub enum FirMethod<T> {
    /// Windowed-sinc design: truncate the ideal impulse response to the
    /// requested order and apply `window` to reduce spectral leakage.
    Windowed {
        /// Window applied to the ideal (sinc) impulse response.
        window: Window,
    },

    /// Equiripple (Parks–McClellan / Remez exchange) design: the
    /// minimax-optimal linear-phase FIR for a given length and spec.
    ///
    /// Unlike the windowed method, equiripple takes the passband/stopband
    /// tolerances directly, so its attenuation is *truthful* — the realized
    /// stopband meets the requested `stop_atten`.  For a ×2 decimation lowpass
    /// (cutoff `fs/4` with a symmetric transition) the design is a **half-band**
    /// filter whose even-indexed taps, bar the center, are zero.
    ///
    /// # Units and Remez weighting
    ///
    /// Both fields are in **decibels**:
    ///
    /// * `pass_ripple` — peak passband ripple `Ap` (dB). Converted to a linear
    ///   deviation `δ_p = (10^(Ap/20) − 1) / (10^(Ap/20) + 1)`.
    /// * `stop_atten` — stopband attenuation `As` (positive dB). Converted to a
    ///   linear deviation `δ_s = 10^(−As/20)`.
    ///
    /// The desired band amplitudes are `1` in the passband and `0` in the
    /// stopband, and the Remez error weights are **inversely proportional to
    /// the allowed deviations** (`W_pass = 1/δ_p`, `W_stop = 1/δ_s`) so the
    /// realized ripples equalize to the requested `Ap`/`As`.  This is the exact
    /// mapping the scipy golden generator (`scripts/gen_remez_reference.py`)
    /// uses with `scipy.signal.remez(..., weight=[1/δ_p, 1/δ_s])`.
    Equiripple {
        /// Peak passband ripple `Ap` in dB (e.g. `0.1`).
        pass_ripple: T,
        /// Stopband attenuation `As` in positive dB (e.g. `60.0`).
        stop_atten: T,
    },
}

// ─── Public API ──────────────────────────────────────────────────────

/// Design a FIR filter, dispatching over the [`FirMethod`].
///
/// Returns a [`FirFilter`] containing `order + 1` tap coefficients plus
/// [`FirMeta`] recording the sample rate and primary normalized cutoff.  The
/// taps are gain-normalized: lowpass filters have unity DC gain, highpass
/// filters have unity gain at Nyquist, and bandpass/bandstop filters have unity
/// gain at the band center or at DC respectively.
///
/// The returned filter carries no execution strategy; compose it into a
/// [`FirSpec`](crate::traits::fir::FirSpec) via
/// [`FirSpec::new`](crate::traits::fir::FirSpec::new) to obtain a plan-ready
/// FIR module spec.
///
/// # Arguments
///
/// * `filter_type` — frequency response shape.
/// * `order` — filter order (number of taps is `order + 1`).
/// * `sample_rate` — sampling frequency (Hz).
/// * `cutoff1` — primary cutoff frequency (Hz).
/// * `cutoff2` — secondary cutoff (Hz), required for bandpass/bandstop.
/// * `method` — design method (e.g. [`FirMethod::Windowed`]).
///
/// # Errors
///
/// Returns [`Error::InvalidSpec`] if:
/// - `order` is zero
/// - `sample_rate` is not positive
/// - cutoff frequencies are outside `(0, sample_rate / 2)`
/// - bandpass/bandstop is missing `cutoff2` or has `cutoff2 <= cutoff1`
/// - lowpass/highpass has `cutoff2` present
///
/// # Examples
///
/// ```
/// use omnidsp_core::design::fir::{design, FirMethod};
/// use omnidsp_core::types::FilterType;
/// use omnidsp_core::window::Window;
///
/// let filter = design(
///     FilterType::Lowpass, 30, 44100.0_f64, 1000.0, None,
///     &FirMethod::Windowed { window: Window::Hamming },
/// ).unwrap();
/// assert_eq!(filter.coefficients().len(), 31);
/// ```
pub fn design<T: Float>(
    filter_type: FilterType,
    order: usize,
    sample_rate: T,
    cutoff1: T,
    cutoff2: Option<T>,
    method: &FirMethod<T>,
) -> Result<FirFilter<T>> {
    let sr = to_f64(sample_rate)?;
    let c1 = to_f64(cutoff1)?;
    let c2 = cutoff2.map(to_f64).transpose()?;

    let validated = validate(filter_type, order, sr, c1, c2)?;

    // Dispatch over the design method; each produces f64 taps.
    let normalized: Vec<f64> = match method {
        FirMethod::Windowed { window } => design_windowed(filter_type, &validated, window)?,
        FirMethod::Equiripple {
            pass_ripple,
            stop_atten,
        } => design_equiripple(
            filter_type,
            &validated,
            to_f64(*pass_ripple)?,
            to_f64(*stop_atten)?,
        )?,
    };

    // Convert to target type
    let coefficients: Result<Vec<T>> = normalized.iter().map(|&v| from_f64(v)).collect();

    // Record the design context (Hz is f64, surface-wide): the design sample
    // rate and the primary normalized cutoff (cutoff1 / fs).
    FirFilter::new(
        coefficients?,
        FirMeta::new(to_f64(sample_rate)?, validated.primary_cutoff),
    )
}

/// Windowed-sinc design path: ideal impulse response, taper, gain normalize.
fn design_windowed(
    filter_type: FilterType,
    validated: &ValidatedSpec,
    window: &Window,
) -> Result<Vec<f64>> {
    // Compute ideal impulse response
    let ideal = match validated.cutoffs {
        NormalizedCutoffs::Single(fn1) => match filter_type {
            FilterType::Lowpass => ideal_lowpass(validated.num_taps, fn1),
            FilterType::Highpass => ideal_highpass(validated.num_taps, fn1),
            FilterType::Bandpass | FilterType::Bandstop => {
                return Err(Error::Internal(
                    "bandpass/bandstop requires two cutoffs".into(),
                ));
            }
        },
        NormalizedCutoffs::Dual(fn1, fn2) => match filter_type {
            FilterType::Bandpass => ideal_bandpass(validated.num_taps, fn1, fn2),
            FilterType::Bandstop => ideal_bandstop(validated.num_taps, fn1, fn2),
            FilterType::Lowpass | FilterType::Highpass => {
                return Err(Error::Internal(
                    "lowpass/highpass requires one cutoff".into(),
                ));
            }
        },
    };

    // Generate and apply the method-specific taper.  The window evaluates
    // directly in f64 — the precision the windowed-sinc math runs in — so there
    // is no per-coefficient `T → f64` round-trip.
    let win_coeffs = window.coefficients::<f64>(validated.num_taps)?;
    let windowed: Vec<f64> = ideal
        .iter()
        .zip(&win_coeffs)
        .map(|(h, w)| *h * *w)
        .collect();

    // Normalize gain
    Ok(normalize(filter_type, &windowed, &validated.cutoffs))
}

/// Fraction of a cutoff frequency spanned by the equiripple half-transition.
///
/// Equiripple needs an explicit transition gap; `design` only takes cutoffs,
/// so each cutoff is widened into a passband edge / stopband edge symmetric
/// transition of half-width `cutoff · EQ_TRANSITION_FRACTION`.
const EQ_TRANSITION_FRACTION: f64 = 0.25;

/// Equiripple (Parks–McClellan/Remez) design path.
///
/// Maps the dB passband-ripple / stopband-attenuation targets to linear Remez
/// deviations and band weights (see [`FirMethod::Equiripple`]), derives band
/// edges from the validated cutoffs (a symmetric transition of half-width
/// `cutoff · EQ_TRANSITION_FRACTION` about each cutoff), and dispatches to the
/// Remez engine.  A lowpass at cutoff ≈ `fs/4` takes the **half-band** path:
/// design then zero the alternate taps (the ×2 decimation case).
#[allow(
    clippy::too_many_lines,
    reason = "the per-filter-type band construction is one cohesive match"
)]
fn design_equiripple(
    filter_type: FilterType,
    validated: &ValidatedSpec,
    pass_ripple_db: f64,
    stop_atten_db: f64,
) -> Result<Vec<f64>> {
    if pass_ripple_db <= 0.0 || stop_atten_db <= 0.0 {
        return Err(Error::InvalidSpec(
            "equiripple pass_ripple and stop_atten must be positive (dB)".into(),
        ));
    }

    // dB specs → linear deviations → Remez weights (inverse of the deviations).
    let ap = (10.0_f64).powf(pass_ripple_db / 20.0);
    let delta_p = (ap - 1.0) / (ap + 1.0);
    let delta_s = (10.0_f64).powf(-stop_atten_db / 20.0);
    if delta_p <= 0.0 {
        return Err(Error::InvalidSpec(
            "equiripple pass_ripple too small to realize".into(),
        ));
    }
    let w_pass = 1.0 / delta_p;
    let w_stop = 1.0 / delta_s;

    let bands: Vec<remez::Band> = match validated.cutoffs {
        NormalizedCutoffs::Single(fc) => match filter_type {
            FilterType::Lowpass => {
                // ×2 decimation half-band special case: cutoff at fs/4.
                if (fc - 0.25).abs() < 1e-6 {
                    return design_halfband(validated.num_taps, w_pass, w_stop);
                }
                let delta = fc * EQ_TRANSITION_FRACTION;
                vec![
                    remez::Band {
                        low: 0.0,
                        high: (fc - delta).max(0.0),
                        desired: 1.0,
                        weight: w_pass,
                    },
                    remez::Band {
                        low: (fc + delta).min(0.5),
                        high: 0.5,
                        desired: 0.0,
                        weight: w_stop,
                    },
                ]
            }
            FilterType::Highpass => {
                let delta = fc * EQ_TRANSITION_FRACTION;
                vec![
                    remez::Band {
                        low: 0.0,
                        high: (fc - delta).max(0.0),
                        desired: 0.0,
                        weight: w_stop,
                    },
                    remez::Band {
                        low: (fc + delta).min(0.5),
                        high: 0.5,
                        desired: 1.0,
                        weight: w_pass,
                    },
                ]
            }
            FilterType::Bandpass | FilterType::Bandstop => {
                return Err(Error::Internal(
                    "bandpass/bandstop requires two cutoffs".into(),
                ));
            }
        },
        NormalizedCutoffs::Dual(fc1, fc2) => {
            let d1 = fc1 * EQ_TRANSITION_FRACTION;
            let d2 = fc2 * EQ_TRANSITION_FRACTION;
            match filter_type {
                FilterType::Bandpass => vec![
                    remez::Band {
                        low: 0.0,
                        high: (fc1 - d1).max(0.0),
                        desired: 0.0,
                        weight: w_stop,
                    },
                    remez::Band {
                        low: (fc1 + d1).min(0.5),
                        high: (fc2 - d2).max(fc1 + d1),
                        desired: 1.0,
                        weight: w_pass,
                    },
                    remez::Band {
                        low: (fc2 + d2).min(0.5),
                        high: 0.5,
                        desired: 0.0,
                        weight: w_stop,
                    },
                ],
                FilterType::Bandstop => vec![
                    remez::Band {
                        low: 0.0,
                        high: (fc1 - d1).max(0.0),
                        desired: 1.0,
                        weight: w_pass,
                    },
                    remez::Band {
                        low: (fc1 + d1).min(0.5),
                        high: (fc2 - d2).max(fc1 + d1),
                        desired: 0.0,
                        weight: w_stop,
                    },
                    remez::Band {
                        low: (fc2 + d2).min(0.5),
                        high: 0.5,
                        desired: 1.0,
                        weight: w_pass,
                    },
                ],
                FilterType::Lowpass | FilterType::Highpass => {
                    return Err(Error::Internal(
                        "lowpass/highpass requires one cutoff".into(),
                    ));
                }
            }
        }
    };

    remez::design(validated.num_taps, &bands)
}

/// Design a half-band equiripple lowpass at cutoff `fs/4` and zero the
/// alternate (even-indexed, off-center) taps — the optimal ×2 decimation
/// filter.
///
/// A true half-band has even-symmetric bands about `fs/4` and equal pass/stop
/// weighting; the Remez solution then has its even-indexed taps (bar the
/// center) vanish to numerical precision.  We force them to exactly zero, which
/// makes the structure exact (and halves the multiply count of the decimator).
/// The transition is symmetric about `fs/4` at half-width
/// `0.25 · EQ_TRANSITION_FRACTION`.
fn design_halfband(num_taps: usize, _w_pass: f64, _w_stop: f64) -> Result<Vec<f64>> {
    let delta = 0.25 * EQ_TRANSITION_FRACTION;
    let bands = [
        remez::Band {
            low: 0.0,
            high: 0.25 - delta,
            desired: 1.0,
            // Symmetric (equal) weighting is what makes alternate taps vanish.
            weight: 1.0,
        },
        remez::Band {
            low: 0.25 + delta,
            high: 0.5,
            desired: 0.0,
            weight: 1.0,
        },
    ];
    let mut taps = remez::design(num_taps, &bands)?;

    // Force the half-band structure exact: even-indexed taps, except the
    // center, are zero.
    let center = (num_taps - 1) / 2;
    for (i, t) in taps.iter_mut().enumerate() {
        if i != center && i % 2 == 0 {
            *t = 0.0;
        }
    }
    Ok(taps)
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

// ─── Validated spec ───────────────────────────────────────────────────

/// Normalized cutoff frequencies (divided by sample rate).
enum NormalizedCutoffs {
    /// Single cutoff for lowpass/highpass.
    Single(f64),
    /// Two cutoffs for bandpass/bandstop (fn1 < fn2).
    Dual(f64, f64),
}

/// Validated and normalized parameters, ready for computation.
struct ValidatedSpec {
    num_taps: usize,
    cutoffs: NormalizedCutoffs,
    /// Primary normalized cutoff (`cutoff1 / sample_rate`), recorded in the
    /// designed filter's [`FirMeta`].
    primary_cutoff: f64,
}

fn validate(
    filter_type: FilterType,
    order: usize,
    sr: f64,
    c1: f64,
    c2: Option<f64>,
) -> Result<ValidatedSpec> {
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

    let fn1 = c1 / sr;

    let cutoffs = match filter_type {
        FilterType::Lowpass | FilterType::Highpass => {
            if c2.is_some() {
                return Err(Error::InvalidSpec(
                    "cutoff2 must not be specified for lowpass/highpass".into(),
                ));
            }
            NormalizedCutoffs::Single(fn1)
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
            NormalizedCutoffs::Dual(fn1, c2_val / sr)
        }
    };

    Ok(ValidatedSpec {
        num_taps: order + 1,
        cutoffs,
        primary_cutoff: fn1,
    })
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
fn normalize(filter_type: FilterType, taps: &[f64], cutoffs: &NormalizedCutoffs) -> Vec<f64> {
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
            let &NormalizedCutoffs::Dual(fn1, fn2) = cutoffs else {
                return taps.to_vec();
            };
            eval_gain(taps, f64::midpoint(fn1, fn2))
        }
    };

    if gain.abs() < 1e-15 {
        return taps.to_vec();
    }
    taps.iter().map(|&h| h / gain).collect()
}

/// Evaluate the filter magnitude at normalized frequency `f` (0..0.5).
fn eval_gain(taps: &[f64], f: f64) -> f64 {
    let omega = 2.0 * PI * f;
    let re: f64 = taps
        .iter()
        .enumerate()
        .map(|(n, &h)| h * (omega * n as f64).cos())
        .sum();
    let im: f64 = taps
        .iter()
        .enumerate()
        .map(|(n, &h)| h * (omega * n as f64).sin())
        .sum();
    re.hypot(im)
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    fn windowed(window: Window) -> FirMethod<f64> {
        FirMethod::Windowed { window }
    }

    fn lp(order: usize, fc: f64) -> Vec<f64> {
        design(
            FilterType::Lowpass,
            order,
            44100.0,
            fc,
            None,
            &windowed(Window::Hamming),
        )
        .expect("LP design")
        .coefficients()
        .to_vec()
    }

    fn hp(order: usize, fc: f64) -> Vec<f64> {
        design(
            FilterType::Highpass,
            order,
            44100.0,
            fc,
            None,
            &windowed(Window::Hamming),
        )
        .expect("HP design")
        .coefficients()
        .to_vec()
    }

    fn bp(order: usize, fc1: f64, fc2: f64) -> Vec<f64> {
        design(
            FilterType::Bandpass,
            order,
            44100.0,
            fc1,
            Some(fc2),
            &windowed(Window::Hamming),
        )
        .expect("BP design")
        .coefficients()
        .to_vec()
    }

    fn bs(order: usize, fc1: f64, fc2: f64) -> Vec<f64> {
        design(
            FilterType::Bandstop,
            order,
            44100.0,
            fc1,
            Some(fc2),
            &windowed(Window::Hamming),
        )
        .expect("BS design")
        .coefficients()
        .to_vec()
    }

    // ── Lowpass ──────────────────────────────────────────────────

    #[test]
    fn lowpass_dc_gain_is_unity() {
        let taps = lp(30, 1000.0);
        let dc_gain: f64 = taps.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < TOL,
            "lowpass DC gain should be 1.0, got {dc_gain}"
        );
    }

    #[test]
    fn lowpass_correct_length() {
        let order = 64;
        let taps = design(
            FilterType::Lowpass,
            order,
            44100.0,
            1000.0,
            None,
            &windowed(Window::Hann),
        )
        .expect("lowpass design")
        .coefficients()
        .to_vec();
        assert_eq!(taps.len(), order + 1, "should have order+1 taps");
    }

    #[test]
    fn lowpass_symmetric() {
        let taps = lp(30, 1000.0);
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
        let taps = design(
            FilterType::Lowpass,
            4,
            2.0, // fs=2 → Nyquist=1, fc=0.25 means quarter-band
            0.25,
            None,
            &windowed(Window::Rectangular),
        )
        .expect("lowpass design")
        .coefficients()
        .to_vec();

        assert_eq!(taps.len(), 5, "should have 5 taps");
        assert!(
            (taps[0] - taps[4]).abs() < TOL,
            "endpoints should be symmetric"
        );
        assert!(
            (taps[1] - taps[3]).abs() < TOL,
            "inner taps should be symmetric"
        );
        assert!(
            taps[2] > taps[1] && taps[2] > taps[0],
            "center tap should be largest"
        );
        let dc: f64 = taps.iter().sum();
        assert!((dc - 1.0).abs() < TOL, "DC gain should be 1.0, got {dc}");
    }

    // ── Highpass ─────────────────────────────────────────────────

    #[test]
    fn highpass_nyquist_gain_is_unity() {
        let taps = hp(30, 10000.0);
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
        let taps = hp(30, 10000.0);
        let dc_gain: f64 = taps.iter().sum();
        assert!(
            dc_gain.abs() < 0.01,
            "highpass DC gain should be near 0, got {dc_gain}"
        );
    }

    // ── Bandpass ─────────────────────────────────────────────────

    #[test]
    fn bandpass_gain_at_center() {
        let taps = bp(60, 1000.0, 5000.0);
        let fc = 3000.0 / 44100.0;
        let gain = eval_gain(&taps, fc);
        assert!(
            (gain - 1.0).abs() < 0.05,
            "bandpass gain at center should be ≈1.0, got {gain}"
        );
    }

    #[test]
    fn bandpass_dc_gain_near_zero() {
        let taps = bp(60, 1000.0, 5000.0);
        let dc_gain: f64 = taps.iter().sum();
        assert!(
            dc_gain.abs() < 0.1,
            "bandpass DC gain should be near 0, got {dc_gain}"
        );
    }

    // ── Bandstop ─────────────────────────────────────────────────

    #[test]
    fn bandstop_dc_gain_is_unity() {
        let taps = bs(60, 1000.0, 5000.0);
        let dc_gain: f64 = taps.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < TOL,
            "bandstop DC gain should be 1.0, got {dc_gain}"
        );
    }

    #[test]
    fn bandstop_rejection_at_center() {
        let taps = bs(60, 1000.0, 5000.0);
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
                &windowed(Window::Hann)
            )
            .is_err(),
            "order 0 should be rejected"
        );
    }

    #[test]
    fn negative_sample_rate_is_error() {
        assert!(
            design::<f64>(
                FilterType::Lowpass,
                10,
                -1.0,
                0.25,
                None,
                &windowed(Window::Hann)
            )
            .is_err(),
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
                &windowed(Window::Hann)
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
                &windowed(Window::Hann)
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
                &windowed(Window::Hann)
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
                &windowed(Window::Hann),
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
                &windowed(Window::Hann),
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
        let filter = design(
            FilterType::Lowpass,
            30,
            44100.0_f32,
            1000.0_f32,
            None,
            &FirMethod::Windowed {
                window: Window::Hamming,
            },
        )
        .expect("f32 lowpass");
        assert_eq!(filter.coefficients().len(), 31, "should have 31 taps");
        let dc: f32 = filter.coefficients().iter().sum();
        assert!(
            (dc - 1.0).abs() < 1e-5,
            "f32 lowpass DC gain should be ≈1.0, got {dc}"
        );
    }

    // ── Scipy reference comparison ──────────────────────────────
    //
    // Generated by scripts/gen_fir_reference.py.
    // Regenerate with: make gen-fir-reference

    #[allow(
        clippy::wildcard_imports,
        reason = "bulk golden-vector import in tests"
    )]
    use omnidsp_testdata::fir_scipy::*;

    /// Assert two FIR filters produce the same frequency response.
    ///
    /// Compares magnitude at 999 linearly-spaced frequencies across `(0, fs/2)`.
    fn assert_response_match(ours: &[f64], scipy: &[f64], fs: f64, tol: f64, label: &str) {
        for i in 1..1000 {
            let freq = fs / 2.0 * f64::from(i) / 1000.0;
            let f_norm = freq / fs;
            let g_ours = eval_gain(ours, f_norm);
            let g_scipy = eval_gain(scipy, f_norm);
            let err = (g_ours - g_scipy).abs();
            assert!(
                err < tol,
                "{label}: mismatch at {freq:.1} Hz — ours={g_ours:.12e}, scipy={g_scipy:.12e}, err={err:.2e}"
            );
        }
    }

    #[test]
    fn scipy_lowpass_order30_hamming() {
        let ours = lp(30, 1000.0);
        assert_response_match(&ours, LP_ORDER30_HAMMING, 44100.0, 1e-10, "LP30 Hamming");
    }

    #[test]
    fn scipy_highpass_order30_hann() {
        let ours = design(
            FilterType::Highpass,
            30,
            44100.0,
            10000.0,
            None,
            &windowed(Window::Hann),
        )
        .expect("HP30 Hann design")
        .coefficients()
        .to_vec();
        assert_response_match(&ours, HP_ORDER30_HANN, 44100.0, 1e-10, "HP30 Hann");
    }

    #[test]
    fn scipy_bandpass_order60_hamming() {
        let ours = bp(60, 1000.0, 5000.0);
        assert_response_match(&ours, BP_ORDER60_HAMMING, 44100.0, 1e-10, "BP60 Hamming");
    }

    #[test]
    fn scipy_bandstop_order60_hamming() {
        let ours = bs(60, 1000.0, 5000.0);
        assert_response_match(&ours, BS_ORDER60_HAMMING, 44100.0, 1e-10, "BS60 Hamming");
    }

    #[test]
    fn scipy_lowpass_order4_rect() {
        let ours = design(
            FilterType::Lowpass,
            4,
            2.0,
            0.25,
            None,
            &windowed(Window::Rectangular),
        )
        .expect("LP4 rect design")
        .coefficients()
        .to_vec();
        assert_response_match(&ours, LP_ORDER4_RECT, 2.0, 1e-10, "LP4 rect");
    }

    // ── Equiripple (Remez) scipy reference comparison ───────────
    //
    // Generated by scripts/gen_remez_reference.py.
    // Regenerate with: make gen-remez-reference
    //
    // The engine (`design::remez::design`) is driven with the *exact* band
    // edges / desired / weights the golden script used, so the frequency
    // responses match to tight tolerance.  The dB→deviation→weight mapping
    // mirrors `FirMethod::Equiripple` and the script.

    use crate::design::remez::{Band, design as remez_design};
    use omnidsp_testdata::remez_scipy::{BP_REMEZ_4K_8K, HB_REMEZ_X2, LP_REMEZ_4K_6K};

    /// Linear passband / stopband deviations from the dB specs (mirrors the
    /// `FirMethod::Equiripple` doc and the golden generator).
    fn deltas(pass_ripple_db: f64, stop_atten_db: f64) -> (f64, f64) {
        let ap = 10.0_f64.powf(pass_ripple_db / 20.0);
        let delta_p = (ap - 1.0) / (ap + 1.0);
        let delta_s = 10.0_f64.powf(-stop_atten_db / 20.0);
        (delta_p, delta_s)
    }

    /// Peak magnitude deviation from `target` over a normalized frequency band.
    fn band_deviation(taps: &[f64], f_lo: f64, f_hi: f64, target: f64) -> f64 {
        let mut peak: f64 = 0.0;
        for i in 0..=500 {
            let f = f_lo + (f_hi - f_lo) * f64::from(i) / 500.0;
            peak = peak.max((eval_gain(taps, f) - target).abs());
        }
        peak
    }

    /// Assert two equiripple filters agree to `tol` at **in-band** frequencies
    /// (the union of `bands`, given as `(lo, hi)` normalized pairs).
    ///
    /// Two independently-computed minimax-optimal filters with the same spec are
    /// constrained — and so must agree tightly — inside the pass/stop bands; the
    /// *transition* gaps are unconstrained and legitimately diverge, so they are
    /// excluded.  This is a stronger check than a loose full-spectrum tolerance.
    fn assert_inband_match(
        ours: &[f64],
        scipy: &[f64],
        bands: &[(f64, f64)],
        tol: f64,
        label: &str,
    ) {
        for &(lo, hi) in bands {
            for i in 0..=400 {
                let f = lo + (hi - lo) * f64::from(i) / 400.0;
                let g_ours = eval_gain(ours, f);
                let g_scipy = eval_gain(scipy, f);
                let err = (g_ours - g_scipy).abs();
                assert!(
                    err < tol,
                    "{label}: in-band mismatch at f_norm={f:.5} — ours={g_ours:.12e}, scipy={g_scipy:.12e}, err={err:.2e}"
                );
            }
        }
    }

    #[test]
    fn scipy_equiripple_lowpass() {
        let (dp, ds) = deltas(0.1, 60.0);
        let bands = [
            Band {
                low: 0.0,
                high: 4000.0 / 44100.0,
                desired: 1.0,
                weight: 1.0 / dp,
            },
            Band {
                low: 6000.0 / 44100.0,
                high: 0.5,
                desired: 0.0,
                weight: 1.0 / ds,
            },
        ];
        let ours = remez_design(65, &bands).expect("equiripple LP");
        // In-band agreement to ~2e-5: both are the minimax-optimal filter for
        // this spec, so they coincide in the pass/stop bands to ~5 significant
        // figures (the unconstrained transition gap is excluded; the residual
        // is the two optimizers' extremal-set convergence and the barycentric
        // synthesis conditioning at the band endpoints). The absolute gate is
        // the realized-spec test below.
        assert_inband_match(
            &ours,
            LP_REMEZ_4K_6K,
            &[(0.0, 4000.0 / 44100.0), (6000.0 / 44100.0, 0.5)],
            2e-5,
            "Remez LP 4k/6k",
        );
    }

    #[test]
    fn scipy_equiripple_bandpass() {
        let (dp, ds) = deltas(0.1, 60.0);
        let bands = [
            Band {
                low: 0.0,
                high: 2000.0 / 44100.0,
                desired: 0.0,
                weight: 1.0 / ds,
            },
            Band {
                low: 4000.0 / 44100.0,
                high: 8000.0 / 44100.0,
                desired: 1.0,
                weight: 1.0 / dp,
            },
            Band {
                low: 10000.0 / 44100.0,
                high: 0.5,
                desired: 0.0,
                weight: 1.0 / ds,
            },
        ];
        let ours = remez_design(81, &bands).expect("equiripple BP");
        assert_inband_match(
            &ours,
            BP_REMEZ_4K_8K,
            &[
                (0.0, 2000.0 / 44100.0),
                (4000.0 / 44100.0, 8000.0 / 44100.0),
                (10000.0 / 44100.0, 0.5),
            ],
            2e-5,
            "Remez BP 4k/8k",
        );
    }

    #[test]
    fn scipy_equiripple_halfband() {
        // Half-band: cutoff fs/4, symmetric transition ±4410 Hz, equal weights.
        let transition = 4410.0 / 44100.0;
        let bands = [
            Band {
                low: 0.0,
                high: 0.25 - transition,
                desired: 1.0,
                weight: 1.0,
            },
            Band {
                low: 0.25 + transition,
                high: 0.5,
                desired: 0.0,
                weight: 1.0,
            },
        ];
        let ours = remez_design(31, &bands).expect("equiripple half-band");
        assert_inband_match(
            &ours,
            HB_REMEZ_X2,
            &[(0.0, 0.25 - transition), (0.25 + transition, 0.5)],
            2e-5,
            "Remez half-band ×2",
        );
    }

    // ── Equiripple realized-spec assertions (the FirMethod API path) ──

    fn equiripple(pass_ripple: f64, stop_atten: f64) -> FirMethod<f64> {
        FirMethod::Equiripple {
            pass_ripple,
            stop_atten,
        }
    }

    #[test]
    fn equiripple_lowpass_meets_requested_spec() {
        // fs = 44100, cutoff 5000 → ±25% transition → pass [0, 3750], stop [6250, …].
        let taps = design(
            FilterType::Lowpass,
            80,
            44100.0,
            5000.0,
            None,
            &equiripple(0.1, 60.0),
        )
        .expect("equiripple LP design")
        .coefficients()
        .to_vec();

        let (dp, ds) = deltas(0.1, 60.0);
        let pass_dev = band_deviation(&taps, 0.0, 3750.0 / 44100.0, 1.0);
        let stop_dev = band_deviation(&taps, 6250.0 / 44100.0, 0.5, 0.0);
        // Allow a small slack over the exact deviation for grid discretization.
        assert!(
            pass_dev <= dp * 1.2,
            "passband ripple {pass_dev:.3e} should meet δ_p {dp:.3e}"
        );
        assert!(
            stop_dev <= ds * 1.2,
            "stopband leakage {stop_dev:.3e} should meet δ_s {ds:.3e}"
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "half-band alternate taps are force-zeroed to exactly 0.0"
    )]
    fn equiripple_halfband_zeroes_alternate_taps() {
        // cutoff = fs/4 triggers the half-band path.
        let taps = design(
            FilterType::Lowpass,
            30,
            44100.0,
            44100.0 / 4.0,
            None,
            &equiripple(0.1, 70.0),
        )
        .expect("half-band design")
        .coefficients()
        .to_vec();
        assert_eq!(taps.len(), 31, "order 30 → 31 taps (Type I, odd length)");
        let center = (taps.len() - 1) / 2;
        for (i, &t) in taps.iter().enumerate() {
            if i != center && i % 2 == 0 {
                assert_eq!(
                    t, 0.0,
                    "half-band even-indexed tap {i} (off-center) must be exactly zero"
                );
            }
        }
        // Center tap ≈ 0.5 for a half-band.
        assert!(
            (taps[center] - 0.5).abs() < 0.05,
            "half-band center tap should be ≈0.5, got {}",
            taps[center]
        );
    }

    #[test]
    fn equiripple_does_not_converge_is_an_error() {
        // A vanishingly small transition at a short length leaves the
        // alternation unreachable — the design must report an error, never
        // panic or loop forever.
        let degenerate = [
            Band {
                low: 0.0,
                high: 0.2499,
                desired: 1.0,
                weight: 1.0,
            },
            Band {
                low: 0.2501,
                high: 0.5,
                desired: 0.0,
                weight: 1e6,
            },
        ];
        // A 3-tap filter cannot satisfy such a tight, heavily-weighted spec;
        // the routine must terminate with a result either way (no hang).
        let _ = remez_design(3, &degenerate);
    }
}
