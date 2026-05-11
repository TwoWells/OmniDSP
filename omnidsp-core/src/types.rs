// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Core type aliases and enums.

pub use num_complex::{Complex32, Complex64};

/// Transform direction for DFT and related operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward transform.
    Forward,
    /// Inverse transform.
    Inverse,
}

/// Window function specification.
///
/// Each variant is a complete specification: the window shape, its length, and
/// any shape parameters.  `length` is part of the window — a Hann window of
/// length 512 is a different thing from a Hann window of length 1024.
///
/// Generic over `T` so that shape parameters (e.g. Kaiser `beta`, Gaussian
/// `sigma`) match the data type.  `WindowType<f32>` carries `f32` parameters
/// and produces `f32` coefficients.
///
/// The [`Custom`](WindowType::Custom) variant carries user-supplied
/// coefficients.  Its length is the length of the `Vec`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowType<T> {
    /// Bartlett (triangular with zero endpoints) window.
    Bartlett {
        /// Window length in samples.
        length: usize,
    },
    /// Blackman window.
    Blackman {
        /// Window length in samples.
        length: usize,
    },
    /// User-supplied window coefficients.
    ///
    /// The factory wraps these in a plan, giving the custom path the same
    /// accelerated [`apply()`](crate::traits::window::WindowPlan::apply) as
    /// named windows.
    Custom(Vec<T>),
    /// Flat-top window (used for amplitude calibration).
    FlatTop {
        /// Window length in samples.
        length: usize,
    },
    /// Gaussian window.
    Gaussian {
        /// Window length in samples.
        length: usize,
        /// Standard deviation parameter (relative to half the window length).
        sigma: T,
    },
    /// Hamming window.
    Hamming {
        /// Window length in samples.
        length: usize,
    },
    /// Hann (raised cosine) window.
    Hann {
        /// Window length in samples.
        length: usize,
    },
    /// Kaiser window.
    Kaiser {
        /// Window length in samples.
        length: usize,
        /// Shape parameter controlling the trade-off between main-lobe width
        /// and side-lobe level.
        beta: T,
    },
    /// Rectangular (all ones) window.
    Rectangular {
        /// Window length in samples.
        length: usize,
    },
    /// Triangular (non-zero endpoints) window.
    Triangular {
        /// Window length in samples.
        length: usize,
    },
    /// Tukey (tapered cosine) window.
    Tukey {
        /// Window length in samples.
        length: usize,
        /// Taper fraction (`0.0` = rectangular, `1.0` = Hann).
        alpha: T,
    },
}

/// A single second-order section (biquad) with normalized coefficients.
///
/// Transfer function:
/// `H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)`
///
/// `a0` is assumed to be 1 (normalized).  If your source coefficients have a
/// non-unity `a0`, divide all coefficients by `a0` before constructing this
/// type.
///
/// First-order sections can be represented with `b2 = 0` and `a2 = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiquadSection<T> {
    /// Numerator coefficient b0.
    pub b0: T,
    /// Numerator coefficient b1.
    pub b1: T,
    /// Numerator coefficient b2.
    pub b2: T,
    /// Denominator coefficient a1.
    pub a1: T,
    /// Denominator coefficient a2.
    pub a2: T,
}
