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
/// Describes the shape and any shape parameters of a window function, without
/// specifying a length.  Length is provided at the point of use — when
/// constructing a [`Window<T>`] via [`Window::from_fn`].
///
/// Generic over `T` so that shape parameters (e.g. Kaiser `beta`, Gaussian
/// `sigma`) match the data type.
///
/// For user-supplied coefficients (custom windows), skip `WindowFn` entirely
/// and construct a [`Window<T>`] directly via [`Window::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFn<T> {
    /// Bartlett (triangular with zero endpoints) window.
    Bartlett,
    /// Blackman window.
    Blackman,
    /// Flat-top window (used for amplitude calibration).
    FlatTop,
    /// Gaussian window.
    Gaussian {
        /// Standard deviation parameter (relative to half the window length).
        sigma: T,
    },
    /// Hamming window.
    Hamming,
    /// Hann (raised cosine) window.
    Hann,
    /// Kaiser window.
    Kaiser {
        /// Shape parameter controlling the trade-off between main-lobe width
        /// and side-lobe level.
        beta: T,
    },
    /// Rectangular (all ones) window.
    Rectangular,
    /// Triangular (non-zero endpoints) window.
    Triangular,
    /// Tukey (tapered cosine) window.
    Tukey {
        /// Taper fraction (`0.0` = rectangular, `1.0` = Hann).
        alpha: T,
    },
}

/// Computed window coefficients.
///
/// A `Window<T>` holds a vector of precomputed coefficients ready to be applied
/// to a signal via element-wise multiply ([`VecOps::mul_inplace`]).
///
/// Construct from a [`WindowFn<T>`] and a length, or directly from raw
/// coefficients:
///
/// ```
/// use omnidsp_core::types::{Window, WindowFn};
///
/// // From a window function
/// let w = Window::from_fn(&WindowFn::<f64>::Hann, 512).unwrap();
/// assert_eq!(w.len(), 512);
///
/// // From raw coefficients (custom window)
/// let w = Window::new(&[0.5_f64, 1.0, 0.5]);
/// assert_eq!(w.len(), 3);
/// ```
///
/// [`VecOps::mul_inplace`]: crate::traits::vecops::VecOps::mul_inplace
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window<T> {
    coefficients: Vec<T>,
}

impl<T: Float> Window<T> {
    /// Create a window from raw coefficients.
    ///
    /// Use this for custom windows where the coefficients are already known.
    /// The coefficients are copied from the slice.
    #[must_use]
    pub fn new(coefficients: &[T]) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
        }
    }

    /// Create a window by evaluating a [`WindowFn`] at the given `length`.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSpec`](crate::error::Error::InvalidSpec) if `length` is zero or if the window
    /// function parameters are invalid (e.g. Kaiser `beta < 0`, Gaussian
    /// `sigma <= 0`, Tukey `alpha` outside `[0, 1]`).
    pub fn from_fn(winfn: &WindowFn<T>, length: usize) -> Result<Self> {
        let coefficients = crate::design::window::compute(winfn, length)?;
        Ok(Self { coefficients })
    }

    /// Access the precomputed window coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }

    /// Consume the window and return the coefficient vector.
    #[must_use]
    pub fn into_coefficients(self) -> Vec<T> {
        self.coefficients
    }

    /// Return the number of coefficients (window length).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns `true` if the window has zero coefficients.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
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
