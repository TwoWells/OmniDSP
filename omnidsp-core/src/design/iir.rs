// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! IIR filter design via classical analog prototype methods.
//!
//! # Pipeline
//!
//! All classical IIR designs (Butterworth, Chebyshev, elliptic) share
//! the same pipeline — only the analog prototype differs:
//!
//! 1. **Analog prototype** — poles (and zeros) in the s-plane,
//!    normalized to cutoff = 1 rad/s.  Butterworth has poles on the unit
//!    circle and no finite zeros; Chebyshev II and elliptic add zeros on
//!    the imaginary axis.
//! 2. **Frequency transform** — LP→LP, LP→HP, LP→BP, or LP→BS.
//!    Transforms both poles and zeros; adds extra zeros for HP/BP/BS.
//! 3. **Bilinear transform** — maps the s-plane to the z-plane and
//!    yields a zero-pole-gain (ZPK) representation.
//! 4. **SOS formation** — pairs z-plane poles with zeros into
//!    second-order sections ([`BiquadSection`]).
//! 5. **Gain normalization** — scales numerators so the cascade has
//!    unity gain at the reference frequency.
//!
//! To add a new filter family, provide a new analog prototype
//! constructor.  The rest of the pipeline is shared.

#![allow(
    clippy::cast_precision_loss,
    reason = "filter orders are small enough for f64"
)]

use std::f64::consts::PI;

use num_complex::Complex64;

use crate::error::{Error, Result};
use crate::traits::iir::IirSpec;
use crate::types::{BiquadSection, FilterType};

// ─── Public API ──────────────────────────────────────────────────────

/// IIR filter family (analog prototype shape).
///
/// Each family produces a different analog prototype in the s-plane.
/// The rest of the design pipeline (frequency transform, bilinear
/// transform, SOS formation) is shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterFamily {
    /// Maximally flat passband.  Poles on the unit circle, no finite zeros.
    Butterworth,
}

/// Design an IIR filter as cascaded biquad sections.
///
/// Returns a (non-generic, f64) [`IirSpec`] with normalized coefficients
/// (`a0 = 1`); the cast to the operation's `T` happens at the IIR processor's
/// create edge.  Sections use the same sign convention as scipy's `sosfilt` (see
/// [`BiquadSection`]).  Frequency arguments are `f64` (the surface-wide "Hz is
/// f64" rule).
///
/// The number of sections is `⌈order / 2⌉` for lowpass/highpass and
/// `order` for bandpass/bandstop (frequency transforms double the
/// prototype order for band filters).
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
/// use omnidsp_core::design::iir::{FilterFamily, design};
/// use omnidsp_core::types::FilterType;
///
/// let spec = design(
///     FilterFamily::Butterworth,
///     FilterType::Lowpass, 4, 44100.0, 1000.0, None,
/// ).unwrap();
/// assert_eq!(spec.sections().len(), 2);
/// ```
pub fn design(
    family: FilterFamily,
    filter_type: FilterType,
    order: usize,
    sample_rate: f64,
    cutoff1: f64,
    cutoff2: Option<f64>,
) -> Result<IirSpec> {
    let validated = validate(filter_type, order, sample_rate, cutoff1, cutoff2)?;

    let prototype = match family {
        FilterFamily::Butterworth => AnalogPrototype::butterworth(order),
    };

    // The sections stay in the f64 design precision; the cast to the operation's
    // `T` happens at the IIR processor's create edge.
    let sections = design_from_prototype(prototype, &validated, sample_rate);
    IirSpec::new(sections)
}

fn design_from_prototype(
    prototype: AnalogPrototype,
    spec: &ValidatedSpec,
    sr: f64,
) -> Vec<BiquadSection<f64>> {
    let (mut sections, ref_omega) = match *spec {
        ValidatedSpec::Lowpass { fc } => {
            let wc = prewarp(fc, sr);
            let sections = prototype.into_lowpass(wc).into_digital(sr).into_sos();
            (sections, 0.0)
        }
        ValidatedSpec::Highpass { fc } => {
            let wc = prewarp(fc, sr);
            let sections = prototype.into_highpass(wc).into_digital(sr).into_sos();
            (sections, PI)
        }
        ValidatedSpec::Bandpass { fc1, fc2 } => {
            let wc1 = prewarp(fc1, sr);
            let wc2 = prewarp(fc2, sr);
            let w0 = (wc1 * wc2).sqrt();
            let bw = wc2 - wc1;
            let w_center = 2.0 * (w0 / (2.0 * sr)).atan();
            let sections = prototype.into_bandpass(w0, bw).into_digital(sr).into_sos();
            (sections, w_center)
        }
        ValidatedSpec::Bandstop { fc1, fc2 } => {
            let wc1 = prewarp(fc1, sr);
            let wc2 = prewarp(fc2, sr);
            let w0 = (wc1 * wc2).sqrt();
            let bw = wc2 - wc1;
            let sections = prototype.into_bandstop(w0, bw).into_digital(sr).into_sos();
            (sections, 0.0)
        }
    };

    normalize_gain(&mut sections, ref_omega);

    sections
}

// ─── Validated spec ───────────────────────────────────────────────────

/// Validated filter parameters — encodes the filter type so callers can
/// match exhaustively without re-checking `FilterType`.
enum ValidatedSpec {
    Lowpass { fc: f64 },
    Highpass { fc: f64 },
    Bandpass { fc1: f64, fc2: f64 },
    Bandstop { fc1: f64, fc2: f64 },
}

fn validate(
    filter_type: FilterType,
    order: usize,
    sr: f64,
    c1: f64,
    c2: Option<f64>,
) -> Result<ValidatedSpec> {
    if order == 0 {
        return Err(Error::InvalidSpec("order must be non-zero".into()));
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
        FilterType::Lowpass => {
            reject_cutoff2(c2)?;
            Ok(ValidatedSpec::Lowpass { fc: c1 })
        }
        FilterType::Highpass => {
            reject_cutoff2(c2)?;
            Ok(ValidatedSpec::Highpass { fc: c1 })
        }
        FilterType::Bandpass => {
            let fc2 = require_cutoff2(c2, c1, nyquist)?;
            Ok(ValidatedSpec::Bandpass { fc1: c1, fc2 })
        }
        FilterType::Bandstop => {
            let fc2 = require_cutoff2(c2, c1, nyquist)?;
            Ok(ValidatedSpec::Bandstop { fc1: c1, fc2 })
        }
    }
}

fn reject_cutoff2(c2: Option<f64>) -> Result<()> {
    if c2.is_some() {
        return Err(Error::InvalidSpec(
            "cutoff2 must not be specified for lowpass/highpass".into(),
        ));
    }
    Ok(())
}

fn require_cutoff2(c2: Option<f64>, c1: f64, nyquist: f64) -> Result<f64> {
    let Some(val) = c2 else {
        return Err(Error::InvalidSpec(
            "cutoff2 is required for bandpass/bandstop".into(),
        ));
    };
    if val <= 0.0 || val >= nyquist {
        return Err(Error::InvalidSpec(format!(
            "cutoff2 ({val}) must be in (0, {nyquist})"
        )));
    }
    if val <= c1 {
        return Err(Error::InvalidSpec(format!(
            "cutoff2 ({val}) must be greater than cutoff1 ({c1})"
        )));
    }
    Ok(val)
}

// ─── Analog prototype ─────────────────────────────────────────────────

/// Analog filter prototype in the s-plane.
///
/// Holds poles and zeros for a normalized (cutoff = 1 rad/s) analog
/// lowpass prototype.  Each filter family provides its own constructor:
///
/// - [`butterworth`](Self::butterworth) — poles on the unit circle, no
///   finite zeros.  Maximally flat passband.
///
/// Future constructors (Chebyshev I/II, elliptic) add prototype zeros
/// on the imaginary axis.  The frequency transforms and bilinear
/// mapping operate on both poles and zeros identically.
struct AnalogPrototype {
    poles: Vec<Complex64>,
    zeros: Vec<Complex64>,
}

impl AnalogPrototype {
    /// Butterworth prototype: `order` poles equally spaced on the left
    /// half of the unit circle, no finite zeros.
    fn butterworth(order: usize) -> Self {
        let poles = (0..order)
            .map(|k| {
                let theta = PI * (2 * k + order + 1) as f64 / (2 * order) as f64;
                Complex64::new(theta.cos(), theta.sin())
            })
            .collect();
        Self {
            poles,
            zeros: vec![],
        }
    }

    // ── Frequency transforms ────────────────────────────────────
    //
    // Each transform maps both poles and zeros.  Transforms that
    // change the transfer function degree (HP, BP, BS) also add
    // extra zeros to maintain the pole/zero balance.

    /// LP→LP: scale to cutoff frequency `wc`.
    fn into_lowpass(self, wc: f64) -> Self {
        Self {
            poles: self.poles.iter().map(|&p| p * wc).collect(),
            zeros: self.zeros.iter().map(|&z| z * wc).collect(),
        }
    }

    /// LP→HP: `wc / s` substitution.
    ///
    /// Poles and zeros invert; `N − M` extra zeros appear at `s = 0`
    /// (where N = #poles, M = #finite zeros).
    fn into_highpass(self, wc: f64) -> Self {
        let wc_c = Complex64::new(wc, 0.0);
        let extra = self.poles.len() - self.zeros.len();
        let mut zeros: Vec<Complex64> = self.zeros.iter().map(|&z| wc_c / z).collect();
        zeros.resize(zeros.len() + extra, Complex64::new(0.0, 0.0));
        Self {
            poles: self.poles.iter().map(|&p| wc_c / p).collect(),
            zeros,
        }
    }

    /// LP→BP: bandpass with center `w0` and bandwidth `bw`.
    ///
    /// Each pole/zero maps to two via a quadratic; `N − M` extra zeros
    /// appear at `s = 0`.
    fn into_bandpass(self, w0: f64, bw: f64) -> Self {
        let extra = self.poles.len() - self.zeros.len();
        let mut zeros = quadratic_bp(&self.zeros, w0, bw);
        zeros.resize(zeros.len() + extra, Complex64::new(0.0, 0.0));
        Self {
            poles: quadratic_bp(&self.poles, w0, bw),
            zeros,
        }
    }

    /// LP→BS: bandstop with center `w0` and bandwidth `bw`.
    ///
    /// Each pole/zero maps to two via a quadratic; `2(N − M)` extra
    /// zeros appear at `±jω₀` (the notch frequencies).
    fn into_bandstop(self, w0: f64, bw: f64) -> Self {
        let extra = self.poles.len() - self.zeros.len();
        let jw0 = Complex64::new(0.0, w0);
        let mut zeros = quadratic_bs(&self.zeros, w0, bw);
        for _ in 0..extra {
            zeros.push(jw0);
            zeros.push(-jw0);
        }
        Self {
            poles: quadratic_bs(&self.poles, w0, bw),
            zeros,
        }
    }

    /// Bilinear transform: map s-plane to z-plane.
    ///
    /// Each pole and zero is mapped via `z = (2fs + s) / (2fs − s)`.
    /// Implicit zeros at `s = ∞` (the excess of poles over finite
    /// zeros) map to `z = −1`.  After this step, `#poles == #zeros`.
    fn into_digital(self, fs: f64) -> DigitalZpk {
        let c = Complex64::new(2.0 * fs, 0.0);
        let bilinear = |s: &Complex64| (c + s) / (c - s);
        let poles: Vec<Complex64> = self.poles.iter().map(bilinear).collect();
        let mut zeros: Vec<Complex64> = self.zeros.iter().map(bilinear).collect();
        let extra = poles.len() - zeros.len();
        zeros.resize(zeros.len() + extra, Complex64::new(-1.0, 0.0));
        DigitalZpk { poles, zeros }
    }
}

/// Pre-warp a frequency to compensate for bilinear transform warping.
fn prewarp(freq: f64, fs: f64) -> f64 {
    2.0 * fs * (PI * freq / fs).tan()
}

/// LP→BP quadratic: each value `a` maps to
/// `(a·BW ± √((a·BW)² − 4·ω₀²)) / 2`.
fn quadratic_bp(values: &[Complex64], w0: f64, bw: f64) -> Vec<Complex64> {
    let w0_sq = Complex64::new(w0 * w0, 0.0);
    let four = Complex64::new(4.0, 0.0);
    let two = Complex64::new(2.0, 0.0);
    let mut out = Vec::with_capacity(2 * values.len());
    for &a in values {
        let abw = a * bw;
        let abw_sq = abw * abw;
        let sqrt_disc = (abw_sq - four * w0_sq).sqrt();
        out.push((abw + sqrt_disc) / two);
        out.push((abw - sqrt_disc) / two);
    }
    out
}

/// LP→BS quadratic: each value `a` maps to
/// `(BW ± √(BW² − 4·a²·ω₀²)) / (2·a)`.
fn quadratic_bs(values: &[Complex64], w0: f64, bw: f64) -> Vec<Complex64> {
    let bw_c = Complex64::new(bw, 0.0);
    let w0_sq = Complex64::new(w0 * w0, 0.0);
    let four = Complex64::new(4.0, 0.0);
    let two = Complex64::new(2.0, 0.0);
    let mut out = Vec::with_capacity(2 * values.len());
    for &a in values {
        let a_sq = a * a;
        let sqrt_disc = (bw_c * bw_c - four * a_sq * w0_sq).sqrt();
        out.push((bw_c + sqrt_disc) / (two * a));
        out.push((bw_c - sqrt_disc) / (two * a));
    }
    out
}

// ─── Digital zpk → SOS ──────────────────────────────────────────────

/// Digital poles and zeros in the z-plane.
///
/// Invariant: `poles.len() == zeros.len()` — every pole has a
/// corresponding zero, so each second-order section has both a
/// numerator and a denominator of equal degree.
struct DigitalZpk {
    poles: Vec<Complex64>,
    zeros: Vec<Complex64>,
}

impl DigitalZpk {
    /// Form second-order sections by pairing poles with zeros.
    ///
    /// Poles and zeros are each split into conjugate pairs and real
    /// values.  Real zeros are paired from extremes inward (so `z = −1`
    /// pairs with `z = +1` for bandpass, not with another `−1`).  Zero
    /// pairs are then assigned to pole pairs by nearest frequency.
    fn into_sos(self) -> Vec<BiquadSection<f64>> {
        let (pole_pairs, real_poles) = pair_poles(&self.poles);
        let (mut zero_pairs, real_zeros) = pair_zeros(&self.zeros);

        let mut sections = Vec::with_capacity(pole_pairs.len() + real_poles.len());

        // Second-order sections: match each pole pair with nearest zero pair.
        for &(p1, p2) in &pole_pairs {
            let a1 = -(p1 + p2).re;
            let a2 = (p1 * p2).re;
            let (b1, b2) = nearest_pair_idx(&zero_pairs, p1).map_or((0.0, 0.0), |idx| {
                let (z1, z2) = zero_pairs.swap_remove(idx);
                (-(z1 + z2).re, (z1 * z2).re)
            });
            sections.push(BiquadSection {
                b0: 1.0,
                b1,
                b2,
                a1,
                a2,
            });
        }

        // First-order sections: pair each real pole with a real zero.
        for (i, &p) in real_poles.iter().enumerate() {
            let b1 = if i < real_zeros.len() {
                -real_zeros[i]
            } else {
                0.0
            };
            sections.push(BiquadSection {
                b0: 1.0,
                b1,
                b2: 0.0,
                a1: -p,
                a2: 0.0,
            });
        }

        sections
    }
}

// ─── Pole / zero pairing ────────────────────────────────────────────

/// Separate poles into conjugate pairs and unpaired real values.
///
/// Real poles are paired adjacent (via `chunks_exact`); at most one
/// remains unpaired (odd-order LP/HP).
fn pair_poles(values: &[Complex64]) -> (Vec<(Complex64, Complex64)>, Vec<f64>) {
    let (mut pairs, reals) = extract_conjugates(values);

    let paired_count = reals.len() / 2 * 2;
    for pair in reals[..paired_count].chunks_exact(2) {
        pairs.push((Complex64::new(pair[0], 0.0), Complex64::new(pair[1], 0.0)));
    }
    let unpaired = reals[paired_count..].to_vec();

    (pairs, unpaired)
}

/// Separate zeros into conjugate pairs and unpaired real values.
///
/// Real zeros are sorted and paired from extremes inward: the most
/// negative pairs with the most positive.  This ensures bandpass
/// zeros at `z = −1` and `z = +1` form `(1 − z⁻²)` numerators
/// rather than `(1 + z⁻¹)²` or `(1 − z⁻¹)²`.
fn pair_zeros(values: &[Complex64]) -> (Vec<(Complex64, Complex64)>, Vec<f64>) {
    let (mut pairs, mut reals) = extract_conjugates(values);

    reals.sort_by(f64::total_cmp);
    let n = reals.len();
    let half = n / 2;
    for i in 0..half {
        pairs.push((
            Complex64::new(reals[i], 0.0),
            Complex64::new(reals[n - 1 - i], 0.0),
        ));
    }
    let unpaired = if n % 2 == 1 {
        vec![reals[half]]
    } else {
        vec![]
    };

    (pairs, unpaired)
}

/// Find conjugate pairs in a set of complex values.
///
/// Returns conjugate pairs and leftover real values.  Shared by
/// [`pair_poles`] and [`pair_zeros`] — the difference is how the
/// real leftovers are subsequently paired.
fn extract_conjugates(values: &[Complex64]) -> (Vec<(Complex64, Complex64)>, Vec<f64>) {
    let mut remaining: Vec<Complex64> = values.to_vec();
    let mut pairs = Vec::new();
    let mut reals = Vec::new();

    while let Some(p) = remaining.pop() {
        if p.im.abs() < 1e-10 {
            reals.push(p.re);
        } else {
            let conj = p.conj();
            if let Some(idx) = remaining.iter().position(|q| (q - conj).norm() < 1e-8) {
                pairs.push((p, remaining.remove(idx)));
            } else {
                pairs.push((p, conj));
            }
        }
    }

    (pairs, reals)
}

/// Find the zero pair whose frequency is nearest to a given pole.
///
/// "Frequency" of a pair `(a, b)` is the average absolute argument:
/// `(|arg(a)| + |arg(b)|) / 2`.  This puts `z = −1` pairs at `ω = π`,
/// `z = +1` pairs at `ω = 0`, and mixed `(−1, +1)` pairs at `ω = π/2`.
fn nearest_pair_idx(zero_pairs: &[(Complex64, Complex64)], pole: Complex64) -> Option<usize> {
    let target = pole.arg().abs();
    zero_pairs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (pair_frequency(a) - target).abs();
            let db = (pair_frequency(b) - target).abs();
            f64::total_cmp(&da, &db)
        })
        .map(|(idx, _)| idx)
}

fn pair_frequency(pair: &(Complex64, Complex64)) -> f64 {
    f64::midpoint(pair.0.arg().abs(), pair.1.arg().abs())
}

// ─── Gain normalization ──────────────────────────────────────────────

/// Evaluate the magnitude of the SOS cascade at digital frequency ω.
fn eval_sos_magnitude(sections: &[BiquadSection<f64>], omega: f64) -> f64 {
    let z_inv = Complex64::new(omega.cos(), -omega.sin());
    let z_inv2 = z_inv * z_inv;
    let one = Complex64::new(1.0, 0.0);

    let mut gain = one;
    for s in sections {
        let num = Complex64::new(s.b0, 0.0) + z_inv * s.b1 + z_inv2 * s.b2;
        let den = one + z_inv * s.a1 + z_inv2 * s.a2;
        gain = gain * num / den;
    }
    gain.norm()
}

/// Normalize the SOS cascade so magnitude at the reference frequency is unity.
fn normalize_gain(sections: &mut [BiquadSection<f64>], ref_omega: f64) {
    let mag = eval_sos_magnitude(sections, ref_omega);
    if mag < 1e-15 {
        return;
    }
    let per_section = mag.powf(-1.0 / sections.len() as f64);
    for s in sections.iter_mut() {
        s.b0 *= per_section;
        s.b1 *= per_section;
        s.b2 *= per_section;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;
    const MINUS_3DB: f64 = std::f64::consts::FRAC_1_SQRT_2;

    fn butter(
        ft: FilterType,
        order: usize,
        sr: f64,
        c1: f64,
        c2: Option<f64>,
    ) -> Result<Vec<BiquadSection<f64>>> {
        Ok(design(FilterFamily::Butterworth, ft, order, sr, c1, c2)?
            .sections()
            .to_vec())
    }

    /// Evaluate the SOS cascade magnitude at a given frequency in Hz.
    fn gain_at(sections: &[BiquadSection<f64>], freq_hz: f64, fs: f64) -> f64 {
        eval_sos_magnitude(sections, 2.0 * PI * freq_hz / fs)
    }

    // ── Section counts ──────────────────────────────────────────

    #[test]
    fn lowpass_order1_one_section() {
        let s = butter(FilterType::Lowpass, 1, 44100.0, 1000.0, None).expect("LP order 1");
        assert_eq!(s.len(), 1, "ceil(1/2) = 1 section");
    }

    #[test]
    fn lowpass_order2_one_section() {
        let s = butter(FilterType::Lowpass, 2, 44100.0, 1000.0, None).expect("LP order 2");
        assert_eq!(s.len(), 1, "ceil(2/2) = 1 section");
    }

    #[test]
    fn lowpass_order4_two_sections() {
        let s = butter(FilterType::Lowpass, 4, 44100.0, 1000.0, None).expect("LP order 4");
        assert_eq!(s.len(), 2, "ceil(4/2) = 2 sections");
    }

    #[test]
    fn lowpass_order8_four_sections() {
        let s = butter(FilterType::Lowpass, 8, 44100.0, 1000.0, None).expect("LP order 8");
        assert_eq!(s.len(), 4, "ceil(8/2) = 4 sections");
    }

    #[test]
    fn bandpass_order2_two_sections() {
        let s = butter(FilterType::Bandpass, 2, 44100.0, 1000.0, Some(5000.0)).expect("BP order 2");
        assert_eq!(s.len(), 2, "BP order 2 → 2 sections");
    }

    #[test]
    fn bandstop_order2_two_sections() {
        let s = butter(FilterType::Bandstop, 2, 44100.0, 1000.0, Some(5000.0)).expect("BS order 2");
        assert_eq!(s.len(), 2, "BS order 2 → 2 sections");
    }

    // ── First-order sections ────────────────────────────────────

    #[test]
    fn order1_lowpass_is_first_order() {
        let s = butter(FilterType::Lowpass, 1, 44100.0, 1000.0, None).expect("LP order 1");
        assert!(s[0].b2.abs() < TOL, "b2 should be 0 for first-order");
        assert!(s[0].a2.abs() < TOL, "a2 should be 0 for first-order");
    }

    #[test]
    fn order1_highpass_is_first_order() {
        let s = butter(FilterType::Highpass, 1, 44100.0, 10000.0, None).expect("HP order 1");
        assert!(s[0].b2.abs() < TOL, "b2 should be 0 for first-order");
        assert!(s[0].a2.abs() < TOL, "a2 should be 0 for first-order");
    }

    // ── DC gain ─────────────────────────────────────────────────

    #[test]
    fn lowpass_dc_gain_is_unity() {
        for order in [1, 2, 3, 4, 6, 8] {
            let s = butter(FilterType::Lowpass, order, 44100.0, 1000.0, None).expect("LP design");
            let dc = gain_at(&s, 0.0, 44100.0);
            assert!(
                (dc - 1.0).abs() < TOL,
                "LP order {order}: DC gain should be 1.0, got {dc}"
            );
        }
    }

    #[test]
    fn bandstop_dc_gain_is_unity() {
        let s = butter(FilterType::Bandstop, 2, 44100.0, 1000.0, Some(5000.0)).expect("BS order 2");
        let dc = gain_at(&s, 0.0, 44100.0);
        assert!((dc - 1.0).abs() < TOL, "BS DC gain should be 1.0, got {dc}");
    }

    // ── Nyquist gain ────────────────────────────────────────────

    #[test]
    fn highpass_nyquist_gain_is_unity() {
        for order in [1, 2, 3, 4, 6] {
            let s = butter(FilterType::Highpass, order, 44100.0, 10000.0, None).expect("HP design");
            let nyq = eval_sos_magnitude(&s, PI);
            assert!(
                (nyq - 1.0).abs() < TOL,
                "HP order {order}: Nyquist gain should be 1.0, got {nyq}"
            );
        }
    }

    // ── −3 dB at cutoff ─────────────────────────────────────────

    #[test]
    fn lowpass_minus_3db_at_cutoff() {
        let fc = 1000.0;
        let fs = 44100.0;
        for order in [1, 2, 4, 6, 8] {
            let s = butter(FilterType::Lowpass, order, fs, fc, None).expect("LP design");
            let g = gain_at(&s, fc, fs);
            assert!(
                (g - MINUS_3DB).abs() < 1e-6,
                "LP order {order}: gain at cutoff should be {MINUS_3DB}, got {g}"
            );
        }
    }

    #[test]
    fn highpass_minus_3db_at_cutoff() {
        let fc = 10000.0;
        let fs = 44100.0;
        for order in [1, 2, 4, 6] {
            let s = butter(FilterType::Highpass, order, fs, fc, None).expect("HP design");
            let g = gain_at(&s, fc, fs);
            assert!(
                (g - MINUS_3DB).abs() < 1e-6,
                "HP order {order}: gain at cutoff should be {MINUS_3DB}, got {g}"
            );
        }
    }

    #[test]
    fn bandpass_minus_3db_at_edges() {
        let fc1 = 1000.0;
        let fc2 = 5000.0;
        let fs = 44100.0;
        let s = butter(FilterType::Bandpass, 2, fs, fc1, Some(fc2)).expect("BP order 2");
        let g1 = gain_at(&s, fc1, fs);
        let g2 = gain_at(&s, fc2, fs);
        assert!(
            (g1 - MINUS_3DB).abs() < 1e-4,
            "BP gain at fc1 should be −3 dB, got {g1}"
        );
        assert!(
            (g2 - MINUS_3DB).abs() < 1e-4,
            "BP gain at fc2 should be −3 dB, got {g2}"
        );
    }

    #[test]
    fn bandstop_minus_3db_at_edges() {
        let fc1 = 1000.0;
        let fc2 = 5000.0;
        let fs = 44100.0;
        let s = butter(FilterType::Bandstop, 2, fs, fc1, Some(fc2)).expect("BS order 2");
        let g1 = gain_at(&s, fc1, fs);
        let g2 = gain_at(&s, fc2, fs);
        assert!(
            (g1 - MINUS_3DB).abs() < 1e-4,
            "BS gain at fc1 should be −3 dB, got {g1}"
        );
        assert!(
            (g2 - MINUS_3DB).abs() < 1e-4,
            "BS gain at fc2 should be −3 dB, got {g2}"
        );
    }

    // ── Passband / stopband behavior ────────────────────────────

    #[test]
    fn bandpass_gain_at_center_is_unity() {
        let fc1 = 1000.0;
        let fc2 = 5000.0;
        let fs = 44100.0;
        let s = butter(FilterType::Bandpass, 2, fs, fc1, Some(fc2)).expect("BP order 2");
        let wc1 = prewarp(fc1, fs);
        let wc2 = prewarp(fc2, fs);
        let w0 = (wc1 * wc2).sqrt();
        let w_center = 2.0 * (w0 / (2.0 * fs)).atan();
        let g = eval_sos_magnitude(&s, w_center);
        assert!(
            (g - 1.0).abs() < 1e-6,
            "BP gain at center should be 1.0, got {g}"
        );
    }

    #[test]
    fn bandpass_dc_gain_near_zero() {
        let s = butter(FilterType::Bandpass, 2, 44100.0, 1000.0, Some(5000.0)).expect("BP order 2");
        let dc = gain_at(&s, 0.0, 44100.0);
        assert!(dc < 0.01, "BP DC gain should be near 0, got {dc}");
    }

    #[test]
    fn bandstop_rejection_at_center() {
        let fc1 = 1000.0;
        let fc2 = 5000.0;
        let fs = 44100.0;
        let s = butter(FilterType::Bandstop, 2, fs, fc1, Some(fc2)).expect("BS order 2");
        let wc1 = prewarp(fc1, fs);
        let wc2 = prewarp(fc2, fs);
        let w0 = (wc1 * wc2).sqrt();
        let w_center = 2.0 * (w0 / (2.0 * fs)).atan();
        let g = eval_sos_magnitude(&s, w_center);
        assert!(g < 0.01, "BS gain at center should be near 0, got {g}");
    }

    // ── Monotonic rolloff ───────────────────────────────────────

    #[test]
    fn lowpass_monotonically_decreasing() {
        let s = butter(FilterType::Lowpass, 4, 44100.0, 1000.0, None).expect("LP order 4");
        let mut prev = gain_at(&s, 0.0, 44100.0);
        for f in (500..20000).step_by(500) {
            let g = gain_at(&s, f64::from(f), 44100.0);
            assert!(
                g <= prev + 1e-10,
                "LP should decrease: gain at {f} Hz ({g}) > previous ({prev})"
            );
            prev = g;
        }
    }

    // ── Validation ──────────────────────────────────────────────

    #[test]
    fn order_zero_is_error() {
        assert!(
            butter(FilterType::Lowpass, 0, 44100.0, 1000.0, None).is_err(),
            "order 0 should be rejected"
        );
    }

    #[test]
    fn negative_sample_rate_is_error() {
        assert!(
            butter(FilterType::Lowpass, 2, -1.0, 0.25, None).is_err(),
            "negative sample rate should be rejected"
        );
    }

    #[test]
    fn cutoff_at_nyquist_is_error() {
        assert!(
            butter(FilterType::Lowpass, 2, 44100.0, 22050.0, None).is_err(),
            "cutoff at Nyquist should be rejected"
        );
    }

    #[test]
    fn cutoff_above_nyquist_is_error() {
        assert!(
            butter(FilterType::Lowpass, 2, 44100.0, 30000.0, None).is_err(),
            "cutoff above Nyquist should be rejected"
        );
    }

    #[test]
    fn bandpass_missing_cutoff2_is_error() {
        assert!(
            butter(FilterType::Bandpass, 2, 44100.0, 1000.0, None).is_err(),
            "bandpass without cutoff2 should be rejected"
        );
    }

    #[test]
    fn bandpass_cutoff2_le_cutoff1_is_error() {
        assert!(
            butter(FilterType::Bandpass, 2, 44100.0, 5000.0, Some(1000.0)).is_err(),
            "cutoff2 <= cutoff1 should be rejected"
        );
    }

    #[test]
    fn lowpass_with_cutoff2_is_error() {
        assert!(
            butter(FilterType::Lowpass, 2, 44100.0, 1000.0, Some(5000.0)).is_err(),
            "lowpass with cutoff2 should be rejected"
        );
    }

    // ── Scipy reference comparison ──────────────────────────────
    //
    // Generated by scripts/gen_iir_reference.py.
    // Regenerate with: make gen-iir-reference

    #[allow(
        clippy::wildcard_imports,
        reason = "bulk golden-vector import in tests"
    )]
    use omnidsp_testdata::iir_scipy::*;

    /// Build `Vec<BiquadSection<f64>>` from scipy's `(b0, b1, b2, a1, a2)` tuples.
    fn scipy_to_sos(data: &[(f64, f64, f64, f64, f64)]) -> Vec<BiquadSection<f64>> {
        data.iter()
            .map(|&(b0, b1, b2, a1, a2)| BiquadSection { b0, b1, b2, a1, a2 })
            .collect()
    }

    /// Assert two SOS cascades produce the same frequency response.
    fn assert_response_match(
        ours: &[BiquadSection<f64>],
        scipy: &[BiquadSection<f64>],
        fs: f64,
        tol: f64,
        label: &str,
    ) {
        for i in 1..1000 {
            let freq = fs / 2.0 * f64::from(i) / 1000.0;
            let omega = 2.0 * PI * freq / fs;
            let g_ours = eval_sos_magnitude(ours, omega);
            let g_scipy = eval_sos_magnitude(scipy, omega);
            let err = (g_ours - g_scipy).abs();
            assert!(
                err < tol,
                "{label}: mismatch at {freq:.1} Hz — ours={g_ours:.12e}, scipy={g_scipy:.12e}, err={err:.2e}"
            );
        }
    }

    #[test]
    fn scipy_lowpass_order1() {
        let ours = butter(FilterType::Lowpass, 1, 44100.0, 1000.0, None).expect("LP1");
        let scipy = scipy_to_sos(LP_ORDER1);
        assert_response_match(&ours, &scipy, 44100.0, 1e-12, "LP1");
    }

    #[test]
    fn scipy_lowpass_order2() {
        let ours = butter(FilterType::Lowpass, 2, 44100.0, 1000.0, None).expect("LP2");
        let scipy = scipy_to_sos(LP_ORDER2);
        assert_response_match(&ours, &scipy, 44100.0, 1e-12, "LP2");
    }

    #[test]
    fn scipy_lowpass_order4() {
        let ours = butter(FilterType::Lowpass, 4, 44100.0, 1000.0, None).expect("LP4");
        let scipy = scipy_to_sos(LP_ORDER4);
        assert_response_match(&ours, &scipy, 44100.0, 1e-12, "LP4");
    }

    #[test]
    fn scipy_lowpass_order8() {
        let ours = butter(FilterType::Lowpass, 8, 44100.0, 1000.0, None).expect("LP8");
        let scipy = scipy_to_sos(LP_ORDER8);
        assert_response_match(&ours, &scipy, 44100.0, 1e-10, "LP8");
    }

    #[test]
    fn scipy_highpass_order2() {
        let ours = butter(FilterType::Highpass, 2, 44100.0, 10000.0, None).expect("HP2");
        let scipy = scipy_to_sos(HP_ORDER2);
        assert_response_match(&ours, &scipy, 44100.0, 1e-12, "HP2");
    }

    #[test]
    fn scipy_highpass_order4() {
        let ours = butter(FilterType::Highpass, 4, 44100.0, 10000.0, None).expect("HP4");
        let scipy = scipy_to_sos(HP_ORDER4);
        assert_response_match(&ours, &scipy, 44100.0, 1e-12, "HP4");
    }

    #[test]
    fn scipy_bandpass_order2() {
        let ours = butter(FilterType::Bandpass, 2, 44100.0, 1000.0, Some(5000.0)).expect("BP2");
        let scipy = scipy_to_sos(BP_ORDER2);
        assert_response_match(&ours, &scipy, 44100.0, 1e-12, "BP2");
    }

    #[test]
    fn scipy_bandpass_order4() {
        let ours = butter(FilterType::Bandpass, 4, 44100.0, 1000.0, Some(5000.0)).expect("BP4");
        let scipy = scipy_to_sos(BP_ORDER4);
        assert_response_match(&ours, &scipy, 44100.0, 1e-10, "BP4");
    }

    #[test]
    fn scipy_bandstop_order2() {
        let ours = butter(FilterType::Bandstop, 2, 44100.0, 1000.0, Some(5000.0)).expect("BS2");
        let scipy = scipy_to_sos(BS_ORDER2);
        assert_response_match(&ours, &scipy, 44100.0, 1e-12, "BS2");
    }

    #[test]
    fn scipy_bandstop_order4() {
        let ours = butter(FilterType::Bandstop, 4, 44100.0, 1000.0, Some(5000.0)).expect("BS4");
        let scipy = scipy_to_sos(BS_ORDER4);
        assert_response_match(&ours, &scipy, 44100.0, 1e-10, "BS4");
    }
}
