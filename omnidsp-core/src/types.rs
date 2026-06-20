// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Core type aliases and enums.

use num_traits::{Float, FromPrimitive};

pub use num_complex::{Complex32, Complex64};

/// Floating-point type suitable for DSP operations.
///
/// Bundles the numeric, conversion, and thread-safety bounds that all
/// `OmniDSP` primitives require.  Implemented automatically for any type
/// that satisfies the constituent bounds (`f32` and `f64` in practice).
pub trait DspFloat: Float + FromPrimitive + Send + Sync + 'static {}

impl<T: Float + FromPrimitive + Send + Sync + 'static> DspFloat for T {}

/// Transform direction for DFT and related operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward transform (time domain → frequency domain).
    Forward,
    /// Inverse transform (frequency domain → time domain).
    Inverse,
}

/// Frequency response shape for filter design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Passes frequencies below the cutoff.
    Lowpass,
    /// Passes frequencies above the cutoff.
    Highpass,
    /// Passes frequencies between two cutoffs.
    Bandpass,
    /// Rejects frequencies between two cutoffs.
    Bandstop,
}

/// A single second-order section (biquad) with normalized coefficients.
///
/// Transfer function (Direct Form I):
///
/// ```text
/// H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
/// ```
///
/// `a0` is assumed to be 1 (normalized).  If your source coefficients have a
/// non-unity `a0`, divide all five coefficients by `a0` before constructing
/// this type.
///
/// Sign convention follows scipy's `sosfilt`: the denominator polynomial is
/// `1 + a1·z⁻¹ + a2·z⁻²`, so stable filters have `a1` and `a2` values
/// corresponding to poles inside the unit circle.  Note that some DSP
/// textbooks use the opposite sign convention (`1 − a1·z⁻¹ − a2·z⁻²`).
///
/// First-order sections can be represented with `b2 = 0` and `a2 = 0`.
///
/// # Examples
///
/// ```
/// use omnidsp_core::types::BiquadSection;
///
/// // Unity passthrough (y[n] = x[n]):
/// let passthrough = BiquadSection {
///     b0: 1.0_f64, b1: 0.0, b2: 0.0,
///     a1: 0.0, a2: 0.0,
/// };
///
/// // First-order lowpass: H(z) = 0.5(1 + z⁻¹)
/// let first_order = BiquadSection {
///     b0: 0.5_f64, b1: 0.5, b2: 0.0,
///     a1: 0.0, a2: 0.0,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiquadSection<T> {
    /// Numerator (feedforward) coefficient for z⁰ (current input sample).
    pub b0: T,
    /// Numerator (feedforward) coefficient for z⁻¹ (one sample delay).
    pub b1: T,
    /// Numerator (feedforward) coefficient for z⁻² (two sample delay).
    pub b2: T,
    /// Denominator (feedback) coefficient for z⁻¹ (one sample delay).
    pub a1: T,
    /// Denominator (feedback) coefficient for z⁻² (two sample delay).
    pub a2: T,
}
