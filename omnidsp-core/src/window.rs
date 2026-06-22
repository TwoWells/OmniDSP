// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Window functions as a non-generic description.
//!
//! A [`Window`] is a plain enum describing a window function — its kind plus any
//! shape parameters — and nothing more.  It is **not** generic over the signal
//! type: shape parameters (Kaiser β, Gaussian σ, Tukey taper) are `f64` design
//! scalars, and the output precision `T` is chosen at evaluation by
//! [`coefficients::<T>`](Window::coefficients).  The same description can be
//! evaluated as `Vec<f32>` in one place and `Vec<f64>` in another.
//!
//! The enum is `#[non_exhaustive]`: new window kinds can be added without a
//! breaking change.  Parameterless kinds are unit variants ([`Window::Hann`],
//! [`Window::Blackman`], …); parameterized kinds carry their `f64` scalar
//! ([`Window::Kaiser`], [`Window::Gaussian`], [`Window::Tukey`]).
//!
//! Kaiser β is usually **solved from a target side-lobe / stopband attenuation**
//! rather than written by hand; [`kaiser::attenuation`] does that derivation and
//! returns the β scalar, which the caller wraps in [`Window::Kaiser`].
//!
//! # Examples
//!
//! ```
//! use omnidsp_core::window::{self, Window};
//!
//! // Parameterless variant, evaluated at the destination precision.
//! let coeffs: Vec<f64> = Window::Hann.coefficients(512).unwrap();
//! assert_eq!(coeffs.len(), 512);
//!
//! // Kaiser β solved from an attenuation spec, then wrapped in the variant.
//! let kaiser = Window::Kaiser(window::kaiser::attenuation(70.0));
//! let taps: Vec<f32> = kaiser.coefficients(256).unwrap();
//! assert_eq!(taps.len(), 256);
//! ```

use num_traits::Float;

use crate::error::Result;

/// A window function: its kind plus any `f64` shape parameters.
///
/// A non-generic *description* — the precision is chosen at evaluation by
/// [`coefficients::<T>`](Self::coefficients), not pinned here.  `#[non_exhaustive]`
/// so new kinds can be added without a breaking change.
///
/// # Examples
///
/// ```
/// use omnidsp_core::window::Window;
///
/// let coeffs: Vec<f64> = Window::Hann.coefficients(512).unwrap();
/// assert_eq!(coeffs.len(), 512);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Window {
    /// Bartlett (triangular with zero endpoints) window.
    Bartlett,
    /// Blackman window.
    Blackman,
    /// Blackman–Harris window: a 4-term cosine-sum, low-sidelobe window
    /// (≈ −92 dB peak sidelobes).
    BlackmanHarris,
    /// Flat-top window (used for amplitude calibration).
    FlatTop,
    /// Gaussian window with the given standard deviation parameter
    /// (relative to half the window length).
    Gaussian(f64),
    /// Hamming window.
    Hamming,
    /// Hann (raised cosine) window.
    Hann,
    /// Kaiser window with the given shape parameter β controlling the trade-off
    /// between main-lobe width and side-lobe level.
    Kaiser(f64),
    /// Nuttall window: scipy's minimum 4-term cosine-sum, low-sidelobe window
    /// (≈ −93 dB peak sidelobes).
    Nuttall,
    /// Rectangular (all ones) window.
    Rectangular,
    /// Triangular (non-zero endpoints) window.
    Triangular,
    /// Tukey (tapered cosine) window with the given taper fraction
    /// (`0.0` = rectangular, `1.0` = Hann).
    Tukey(f64),
}

impl Window {
    /// Compute window coefficients for this window at the given length.
    ///
    /// The output precision `T` is chosen here, inferred from the destination —
    /// the description itself is precision-agnostic.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSpec`](crate::error::Error::InvalidSpec) if `length` is
    /// zero or if the shape parameters are invalid (e.g. Kaiser `β < 0`,
    /// Gaussian `σ <= 0`, Tukey taper outside `[0, 1]`).
    pub fn coefficients<T: Float>(&self, length: usize) -> Result<Vec<T>> {
        crate::design::window::compute(self, length)
    }
}

/// Kaiser window β derivation.
///
/// The Kaiser window has one shape parameter, β; it can be written directly into
/// [`Window::Kaiser`], or **solved from a target side-lobe / stopband
/// attenuation** with [`attenuation`](kaiser::attenuation).
pub mod kaiser {
    /// Kaiser β **solved from a target side-lobe / stopband attenuation**
    /// `atten_db` (dB) via the standard Kaiser formula (Oppenheim & Schafer
    /// §7.6).
    ///
    /// This is the principled way to choose the window: state the attenuation
    /// requirement — e.g. the dynamic range a spectrogram must suppress kernel
    /// leakage below — and β follows by equation rather than by hand-tuning.
    /// The same formula sizes the CQT octave decimator, so kernel and decimator
    /// are designed from one spec.  Higher `atten_db` lowers the side lobes at
    /// the cost of a wider main lobe.  Wrap the result in [`Window::Kaiser`].
    ///
    /// [`Window::Kaiser`]: super::Window::Kaiser
    ///
    /// # Examples
    ///
    /// ```
    /// use omnidsp_core::window::kaiser;
    /// // 70 dB sidelobe suppression → β ≈ 6.75.
    /// assert_eq!(kaiser::attenuation(70.0), 0.1102 * (70.0 - 8.7));
    /// ```
    #[must_use]
    pub fn attenuation(atten_db: f64) -> f64 {
        crate::design::window::kaiser_beta(atten_db)
    }
}
