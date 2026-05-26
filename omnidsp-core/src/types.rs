// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Core type aliases and enums.

use num_traits::Float;

pub use num_complex::{Complex32, Complex64};

use crate::error::Result;

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

/// Window function description.
///
/// Describes the shape and any parameters of a window function.  Window is
/// pure math — coefficient generation plus a single `mul_inplace` — not a
/// module.
///
/// Generic over `T` so that shape parameters (e.g. Kaiser `beta`, Gaussian
/// `sigma`) match the data type.
///
/// # Examples
///
/// ```
/// use omnidsp_core::types::Window;
///
/// // Via convenience function
/// let coeffs: Vec<f64> = Window::hann(512).unwrap();
/// assert_eq!(coeffs.len(), 512);
///
/// // Via enum + coefficients()
/// let w = Window::<f64>::Hann;
/// let coeffs = w.coefficients(512).unwrap();
/// assert_eq!(coeffs.len(), 512);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window<T> {
    /// Bartlett (triangular with zero endpoints) window.
    Bartlett,
    /// Blackman window.
    Blackman,
    /// Flat-top window (used for amplitude calibration).
    FlatTop,
    /// Gaussian window with the given standard deviation parameter
    /// (relative to half the window length).
    Gaussian(T),
    /// Hamming window.
    Hamming,
    /// Hann (raised cosine) window.
    Hann,
    /// Kaiser window with the given shape parameter controlling the trade-off
    /// between main-lobe width and side-lobe level.
    Kaiser(T),
    /// Rectangular (all ones) window.
    Rectangular,
    /// Triangular (non-zero endpoints) window.
    Triangular,
    /// Tukey (tapered cosine) window with the given taper fraction
    /// (`0.0` = rectangular, `1.0` = Hann).
    Tukey(T),
}

impl<T: Float> Window<T> {
    /// Compute window coefficients for this window function at the given length.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSpec`](crate::error::Error::InvalidSpec) if `length` is zero or if the window
    /// function parameters are invalid (e.g. Kaiser `beta < 0`, Gaussian
    /// `sigma <= 0`, Tukey `alpha` outside `[0, 1]`).
    pub fn coefficients(&self, length: usize) -> Result<Vec<T>> {
        crate::design::window::compute(self, length)
    }

    /// Compute a Bartlett window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn bartlett(length: usize) -> Result<Vec<T>> {
        Self::Bartlett.coefficients(length)
    }

    /// Compute a Blackman window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn blackman(length: usize) -> Result<Vec<T>> {
        Self::Blackman.coefficients(length)
    }

    /// Compute a flat-top window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn flat_top(length: usize) -> Result<Vec<T>> {
        Self::FlatTop.coefficients(length)
    }

    /// Compute a Gaussian window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero or `sigma <= 0`.
    pub fn gaussian(sigma: T, length: usize) -> Result<Vec<T>> {
        Self::Gaussian(sigma).coefficients(length)
    }

    /// Compute a Hamming window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn hamming(length: usize) -> Result<Vec<T>> {
        Self::Hamming.coefficients(length)
    }

    /// Compute a Hann window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn hann(length: usize) -> Result<Vec<T>> {
        Self::Hann.coefficients(length)
    }

    /// Compute a Kaiser window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero or `beta < 0`.
    pub fn kaiser(beta: T, length: usize) -> Result<Vec<T>> {
        Self::Kaiser(beta).coefficients(length)
    }

    /// Compute a rectangular window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn rectangular(length: usize) -> Result<Vec<T>> {
        Self::Rectangular.coefficients(length)
    }

    /// Compute a triangular window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn triangular(length: usize) -> Result<Vec<T>> {
        Self::Triangular.coefficients(length)
    }

    /// Compute a Tukey window of the given length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero or `alpha` is
    /// outside `[0, 1]`.
    pub fn tukey(alpha: T, length: usize) -> Result<Vec<T>> {
        Self::Tukey(alpha).coefficients(length)
    }
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
