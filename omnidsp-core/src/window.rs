// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Window functions as non-generic, opaque recipes (ADR-013).
//!
//! A [`Window`] describes a window function — its kind plus any shape
//! parameters — and nothing more.  It is **not** generic over the signal type:
//! shape parameters (Kaiser β, Gaussian σ, Tukey taper) are `f64` design
//! scalars, and the output precision is chosen at evaluation by
//! [`coefficients::<T>`](Window::coefficients).  The same recipe can be
//! evaluated as `Vec<f32>` in one place and `Vec<f64>` in another.
//!
//! Windows are **constructed only through the [`window`](self) namespace**, not
//! by naming variants — the recipe is opaque (ADR-013 §3):
//!
//! - parameterless kinds are leaf functions ([`hann`], [`hamming`],
//!   [`blackman`], …) — there is nothing to name;
//! - parameterized kinds are sub-namespaces with one constructor per
//!   parameterization ([`kaiser::beta`], [`kaiser::attenuation`],
//!   [`gaussian::sigma`], [`tukey::taper`]).
//!
//! # Examples
//!
//! ```
//! use omnidsp_core::window;
//!
//! // Parameterless leaf function, evaluated at the destination precision.
//! let coeffs: Vec<f64> = window::hann().coefficients(512).unwrap();
//! assert_eq!(coeffs.len(), 512);
//!
//! // Parameterized constructor — the name says what the scalar is.
//! let kaiser = window::kaiser::attenuation(70.0);
//! let taps: Vec<f32> = kaiser.coefficients(256).unwrap();
//! assert_eq!(taps.len(), 256);
//! ```

use num_traits::Float;

use crate::error::Result;

/// A window function recipe: kind plus shape parameters (ADR-013).
///
/// Opaque and non-generic.  Construct via the [`window`](self) namespace
/// ([`hann`], [`kaiser::attenuation`], …); evaluate with
/// [`coefficients::<T>`](Self::coefficients), which picks the output precision.
///
/// # Examples
///
/// ```
/// use omnidsp_core::window;
///
/// let coeffs: Vec<f64> = window::hann().coefficients(512).unwrap();
/// assert_eq!(coeffs.len(), 512);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    kind: Kind,
}

/// The window kind and its (f64) shape parameters.
///
/// Private: the recipe is opaque (ADR-013 §3), so matching on a window lives
/// inside [`compute`](crate::design::window::compute), same crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Kind {
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
    /// Compute window coefficients for this recipe at the given length.
    ///
    /// The output precision `T` is chosen here, inferred from the destination —
    /// the recipe itself is precision-agnostic.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSpec`](crate::error::Error::InvalidSpec) if `length` is
    /// zero or if the shape parameters are invalid (e.g. Kaiser `β < 0`,
    /// Gaussian `σ <= 0`, Tukey taper outside `[0, 1]`).
    pub fn coefficients<T: Float>(&self, length: usize) -> Result<Vec<T>> {
        crate::design::window::compute(self, length)
    }

    /// The window kind and its shape parameters (crate-internal).
    ///
    /// The recipe is opaque on the public surface; the coefficient path reads it
    /// here to dispatch the math.
    pub(crate) const fn kind(&self) -> Kind {
        self.kind
    }
}

// ─── Parameterless leaf constructors ──────────────────────────────────

/// Bartlett (triangular with zero endpoints) window.
#[must_use]
pub const fn bartlett() -> Window {
    Window {
        kind: Kind::Bartlett,
    }
}

/// Blackman window.
#[must_use]
pub const fn blackman() -> Window {
    Window {
        kind: Kind::Blackman,
    }
}

/// Blackman–Harris window: a 4-term cosine-sum, low-sidelobe window
/// (≈ −92 dB peak sidelobes).  Good for separating loud adjacent tones.
#[must_use]
pub const fn blackman_harris() -> Window {
    Window {
        kind: Kind::BlackmanHarris,
    }
}

/// Triangular window with **non-zero** endpoints.
///
/// (See [`bartlett`] for the zero-endpoint variant.)
#[must_use]
pub const fn triangular() -> Window {
    Window {
        kind: Kind::Triangular,
    }
}

/// Flat-top window (used for amplitude calibration).
#[must_use]
pub const fn flat_top() -> Window {
    Window {
        kind: Kind::FlatTop,
    }
}

/// Hamming window.
#[must_use]
pub const fn hamming() -> Window {
    Window {
        kind: Kind::Hamming,
    }
}

/// Hann (raised cosine) window.
#[must_use]
pub const fn hann() -> Window {
    Window { kind: Kind::Hann }
}

/// Nuttall window: scipy's minimum 4-term cosine-sum, low-sidelobe window
/// (≈ −93 dB peak sidelobes).  Good for separating loud adjacent tones.
#[must_use]
pub const fn nuttall() -> Window {
    Window {
        kind: Kind::Nuttall,
    }
}

/// Rectangular (all ones) window.
#[must_use]
pub const fn rectangular() -> Window {
    Window {
        kind: Kind::Rectangular,
    }
}

// ─── Parameterized sub-namespaces ─────────────────────────────────────

/// Kaiser window constructors (ADR-013 §2 — committed up front).
///
/// The Kaiser window has one shape parameter, β, which can be specified
/// directly ([`beta`](kaiser::beta)) or **solved from a target side-lobe /
/// stopband attenuation** ([`attenuation`](kaiser::attenuation)).
pub mod kaiser {
    use super::{Kind, Window};

    /// Kaiser window with the given shape parameter β.
    ///
    /// β controls the main-lobe-width / side-lobe-level trade-off; larger β
    /// lowers the side lobes at the cost of a wider main lobe.
    #[must_use]
    pub const fn beta(beta: f64) -> Window {
        Window {
            kind: Kind::Kaiser(beta),
        }
    }

    /// Kaiser window whose β is **solved from a target side-lobe / stopband
    /// attenuation** `atten_db` (dB) via the standard Kaiser formula
    /// (Oppenheim & Schafer §7.6).
    ///
    /// This is the principled way to choose the window: state the attenuation
    /// requirement — e.g. the dynamic range a spectrogram must suppress kernel
    /// leakage below — and β follows by equation rather than by hand-tuning.
    /// The same formula sizes the CQT octave decimator, so kernel and decimator
    /// are designed from one spec.  Higher `atten_db` lowers the side lobes at
    /// the cost of a wider main lobe.
    ///
    /// # Examples
    ///
    /// ```
    /// use omnidsp_core::window;
    /// // 70 dB sidelobe suppression → β ≈ 6.75.
    /// let w = window::kaiser::attenuation(70.0);
    /// assert_eq!(w, window::kaiser::beta(0.1102 * (70.0 - 8.7)));
    /// ```
    #[must_use]
    pub fn attenuation(atten_db: f64) -> Window {
        beta(crate::design::window::kaiser_beta(atten_db))
    }
}

/// Gaussian window constructors (ADR-013 §2 — committed up front).
pub mod gaussian {
    use super::{Kind, Window};

    /// Gaussian window with the given standard deviation `sigma`, relative to
    /// half the window length.
    #[must_use]
    pub const fn sigma(sigma: f64) -> Window {
        Window {
            kind: Kind::Gaussian(sigma),
        }
    }
}

/// Tukey (tapered cosine) window constructors (ADR-013 §2 — committed up
/// front).
pub mod tukey {
    use super::{Kind, Window};

    /// Tukey window with the given taper fraction (`0.0` = rectangular,
    /// `1.0` = Hann).
    #[must_use]
    pub const fn taper(taper: f64) -> Window {
        Window {
            kind: Kind::Tukey(taper),
        }
    }
}
