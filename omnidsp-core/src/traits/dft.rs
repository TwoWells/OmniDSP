// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! DFT (Discrete Fourier Transform) primitive traits.
//!
//! The DFT is three **peer primitives**, not one trait with modes:
//!
//! - [`DftC2c`] — complex → complex.  Direction (forward/inverse) is a
//!   runtime field on [`DftC2cSpec`] because both share the
//!   `Complex → Complex` signature.
//! - [`DftR2c`] — real → complex half-spectrum.  Inherently forward; no
//!   direction field.
//! - [`DftC2r`] — complex half-spectrum → real.  Inherently inverse; no
//!   direction field.
//!
//! Because r2c and c2r have distinct signatures, direction is encoded in the
//! type rather than a runtime flag — there is no "wrong direction" error to
//! return.  All three share the family-level [`DftNorm`] normalization enum.
//!
//! Each primitive follows the factory+plan pattern: the factory trait
//! (`Dft*`) creates a configured plan (`Dft*Plan`) from a spec (`Dft*Spec`).
//! Plans are immutable — a single plan can be reused across many calls and
//! shared between threads.
//!
//! Specs are **valid-by-construction**: each `new` is fallible,
//! enforces its invariants once, and exposes its data through accessors so a
//! constructed spec can never be mutated into an invalid state.

use num_complex::Complex;

use crate::error::{Error, Result};
use crate::types::Direction;

// ─── Normalization (family-level) ────────────────────────────────────

/// Normalization convention for the DFT, shared family-level by all three
/// primitives ([`DftC2c`], [`DftR2c`], [`DftC2r`]).
///
/// A convention names the scaling applied per *direction*.  Each primitive
/// applies the slice that matches its fixed direction:
///
/// - [`DftR2c`] is forward-only — it scales like a [`DftC2c`] **forward** plan.
/// - [`DftC2r`] is inverse-only — it scales like a [`DftC2c`] **inverse** plan.
///
/// So the per-direction factors are (where `N` is the **real** signal length —
/// `length` for r2c/c2r, the transform length for c2c):
///
/// | convention | forward (c2c fwd / r2c) | inverse (c2c inv / c2r) |
/// |------------|-------------------------|-------------------------|
/// | [`None`](Self::None)       | ×1      | ×1      |
/// | [`Inverse`](Self::Inverse) | ×1      | ×(1/N)  |
/// | [`Ortho`](Self::Ortho)     | ×(1/√N) | ×(1/√N) |
///
/// A forward+inverse round-trip must use the same convention: `Inverse` and
/// `Ortho` recover the input, `None` scales it by `N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DftNorm {
    /// No scaling in either direction — the raw transform; `IFFT(FFT(x)) = N·x`.
    ///
    /// This is what the underlying kernels (`rustfft` / `realfft`) compute
    /// natively; the `1/N` is left to the caller.
    None,
    /// Scale the inverse by `1/N`, leave the forward unscaled; `IFFT(FFT(x)) = x`.
    ///
    /// The standard "backward" convention — the default in `numpy`, `scipy`,
    /// and MATLAB — and the one the convolution, FIR, Hilbert, and
    /// cross-correlation modules use (the convolution theorem needs a
    /// round-trip identity).
    Inverse,
    /// Scale both directions by `1/√N` (per element).
    ///
    /// Unitary / energy-preserving for c2c (Parseval holds with no extra
    /// factor).  For r2c/c2r it is the same per-element `1/√N` scaling — not a
    /// claim that the real half-spectrum operator is literally unitary (it is
    /// not square, and the DC/Nyquist bins are not doubled).
    Ortho,
}

// ─── c2c: complex → complex ──────────────────────────────────────────

/// Complex-to-complex DFT specification.
///
/// Describes the length, direction, and normalization convention for a
/// [`DftC2c`] plan.  Direction is a runtime field because forward and inverse
/// share the `Complex → Complex` signature.
///
/// The spec is non-generic: precision is a realization-edge concern, chosen at
/// `dsp.create_plan::<T>(&spec)`, not pinned by the description.  Fields are
/// private and the spec is valid-by-construction.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::dft::{DftC2cSpec, DftNorm};
/// use omnidsp_core::types::Direction;
///
/// // 1024-point forward FFT with inverse normalization (round-trip = identity)
/// let spec = DftC2cSpec::new(1024, Direction::Forward, DftNorm::Inverse).unwrap();
/// assert_eq!(spec.length(), 1024);
/// assert_eq!(spec.direction(), Direction::Forward);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DftC2cSpec {
    length: usize,
    direction: Direction,
    norm: DftNorm,
}

impl DftC2cSpec {
    /// Create a complex-to-complex DFT spec.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn new(length: usize, direction: Direction, norm: DftNorm) -> Result<Self> {
        if length == 0 {
            return Err(Error::InvalidSpec("DFT length must be non-zero".into()));
        }
        Ok(Self {
            length,
            direction,
            norm,
        })
    }

    /// Number of complex samples for both input and output buffers.
    ///
    /// Determines the frequency resolution: each bin spans
    /// `sample_rate / length` Hz.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }

    /// Transform direction (forward or inverse).
    #[must_use]
    pub const fn direction(&self) -> Direction {
        self.direction
    }

    /// Normalization convention.
    #[must_use]
    pub const fn norm(&self) -> DftNorm {
        self.norm
    }
}

/// Execution object for a configured complex-to-complex DFT.
///
/// A plan is created by a [`DftC2c`] factory and holds any precomputed state
/// (twiddle factors, scratch buffers, etc.) needed to execute the transform
/// efficiently.  Plans are immutable and take `&self`.
pub trait DftC2cPlan<T>: Send + Sync {
    /// Execute the DFT on `input`, writing the result to `output`.
    ///
    /// Both buffers must have the length the plan was created for.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if either buffer length does not
    /// match the plan length.
    fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> Result<()>;
}

/// Factory for creating [`DftC2cPlan`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system: `impl DftC2c<f32>` and `impl DftC2c<f64>` are independent
/// capabilities.  The associated [`Plan`](DftC2c::Plan) type lets each implementor
/// return a concrete plan — no `Box<dyn>`.
pub trait DftC2c<T> {
    /// The concrete plan type returned by this factory.
    type Plan: DftC2cPlan<T>;

    /// Create a plan for a DFT described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is unsupported by the implementation.
    fn create_plan(&self, spec: &DftC2cSpec) -> Result<Self::Plan>;
}

// ─── r2c: real → complex half-spectrum ───────────────────────────────

/// Real-to-complex DFT specification.
///
/// Describes a forward real-input transform of `length` real samples to a
/// `length / 2 + 1` complex half-spectrum (the upper half is the conjugate
/// mirror of the lower and is not stored).  There is no direction field — r2c
/// is inherently forward.
///
/// The spec is non-generic; precision is chosen at `create_plan::<T>`.  Fields
/// are private and the spec is valid-by-construction.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::dft::{DftR2cSpec, DftNorm};
///
/// let spec = DftR2cSpec::new(1024, DftNorm::Inverse).unwrap();
/// assert_eq!(spec.length(), 1024);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DftR2cSpec {
    length: usize,
    norm: DftNorm,
}

impl DftR2cSpec {
    /// Create a real-to-complex DFT spec.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn new(length: usize, norm: DftNorm) -> Result<Self> {
        if length == 0 {
            return Err(Error::InvalidSpec("DFT length must be non-zero".into()));
        }
        Ok(Self { length, norm })
    }

    /// Number of real input samples.
    ///
    /// The complex half-spectrum output has `length / 2 + 1` bins.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }

    /// Normalization convention.
    #[must_use]
    pub const fn norm(&self) -> DftNorm {
        self.norm
    }
}

/// Execution object for a configured real-to-complex DFT.
///
/// Created by a [`DftR2c`] factory.  Plans are immutable and take `&self`.
pub trait DftR2cPlan<T>: Send + Sync {
    /// Execute the forward real-input DFT.
    ///
    /// `input` must have the spec's `length` real samples; `output` must have
    /// `length / 2 + 1` complex bins (the half-spectrum).
    ///
    /// **Input is consumed**: the transform may overwrite `input`
    /// with scratch (the `realfft` floor hands it straight to the kernel as
    /// working memory).  Copy it first if you still need the time-domain
    /// samples afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if either buffer length does not
    /// match the plan's contract.
    fn execute(&self, input: &mut [T], output: &mut [Complex<T>]) -> Result<()>;
}

/// Factory for creating [`DftR2cPlan`] execution objects.
///
/// `T` is a type parameter so that `impl DftR2c<f32>` and `impl DftR2c<f64>`
/// are independent capabilities.
pub trait DftR2c<T> {
    /// The concrete plan type returned by this factory.
    type Plan: DftR2cPlan<T>;

    /// Create a plan for a real-to-complex DFT described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is unsupported by the implementation.
    fn create_plan(&self, spec: &DftR2cSpec) -> Result<Self::Plan>;
}

// ─── c2r: complex half-spectrum → real ───────────────────────────────

/// Complex-to-real DFT specification.
///
/// Describes an inverse transform from a `length / 2 + 1` complex
/// half-spectrum back to `length` real samples.  There is no direction field
/// — c2r is inherently inverse.
///
/// `length` is the **real output length `N`**: it cannot be recovered from the
/// `N / 2 + 1` complex input alone (both `N = 2k` and `N = 2k - 1` produce a
/// `k`-bin half-spectrum), so it must be carried explicitly.
///
/// The spec is non-generic; precision is chosen at `create_plan::<T>`.  Fields
/// are private and the spec is valid-by-construction.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::dft::{DftC2rSpec, DftNorm};
///
/// let spec = DftC2rSpec::new(1024, DftNorm::Inverse).unwrap();
/// assert_eq!(spec.length(), 1024);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DftC2rSpec {
    length: usize,
    norm: DftNorm,
}

impl DftC2rSpec {
    /// Create a complex-to-real DFT spec.
    ///
    /// `length` is the real output length `N`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `length` is zero.
    pub fn new(length: usize, norm: DftNorm) -> Result<Self> {
        if length == 0 {
            return Err(Error::InvalidSpec("DFT length must be non-zero".into()));
        }
        Ok(Self { length, norm })
    }

    /// Real output length `N`.
    ///
    /// The complex half-spectrum input has `length / 2 + 1` bins.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }

    /// Normalization convention.
    #[must_use]
    pub const fn norm(&self) -> DftNorm {
        self.norm
    }
}

/// Execution object for a configured complex-to-real DFT.
///
/// Created by a [`DftC2r`] factory.  Plans are immutable and take `&self`.
pub trait DftC2rPlan<T>: Send + Sync {
    /// Execute the inverse transform to real samples.
    ///
    /// `input` must have `length / 2 + 1` complex bins (the half-spectrum);
    /// `output` must have `length` real samples.
    ///
    /// **Input is consumed**: the transform may overwrite `input`
    /// with scratch.  Copy it first if you still need the half-spectrum
    /// afterwards.
    ///
    /// The bare primitive assumes a **clean Hermitian** half-spectrum — DC, and
    /// for even `length` Nyquist, purely real.  Drift-tolerant projection onto
    /// the nearest valid spectrum is the job of the
    /// [`HermitianC2r`](crate::hermitian::HermitianC2r) shaping decorator,
    /// not of the primitive: a direct call with a dirty DC/Nyquist
    /// surfaces the kernel's native behavior.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if either buffer length does not
    /// match the plan's contract.
    fn execute(&self, input: &mut [Complex<T>], output: &mut [T]) -> Result<()>;
}

/// Factory for creating [`DftC2rPlan`] execution objects.
///
/// `T` is a type parameter so that `impl DftC2r<f32>` and `impl DftC2r<f64>`
/// are independent capabilities.
pub trait DftC2r<T> {
    /// The concrete plan type returned by this factory.
    type Plan: DftC2rPlan<T>;

    /// Create a plan for a complex-to-real DFT described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is unsupported by the implementation.
    fn create_plan(&self, spec: &DftC2rSpec) -> Result<Self::Plan>;
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::{DftC2cSpec, DftC2rSpec, DftNorm, DftR2cSpec, Direction};
    use crate::error::Error;

    #[test]
    fn c2c_spec_rejects_zero_length() {
        let result = DftC2cSpec::new(0, Direction::Forward, DftNorm::None);
        assert!(
            matches!(result, Err(Error::InvalidSpec(_))),
            "zero length must be rejected with InvalidSpec"
        );
    }

    #[test]
    fn c2c_spec_accessors_round_trip() {
        let spec = DftC2cSpec::new(1024, Direction::Inverse, DftNorm::Ortho)
            .expect("non-zero length should be accepted");
        assert_eq!(
            spec.length(),
            1024,
            "length accessor should match constructor"
        );
        assert_eq!(
            spec.direction(),
            Direction::Inverse,
            "direction accessor should match constructor"
        );
        assert_eq!(
            spec.norm(),
            DftNorm::Ortho,
            "norm accessor should match constructor"
        );
    }

    #[test]
    fn r2c_spec_rejects_zero_length() {
        let result = DftR2cSpec::new(0, DftNorm::None);
        assert!(
            matches!(result, Err(Error::InvalidSpec(_))),
            "zero length must be rejected with InvalidSpec"
        );
    }

    #[test]
    fn r2c_spec_accessors_round_trip() {
        let spec = DftR2cSpec::new(512, DftNorm::Inverse).expect("non-zero length accepted");
        assert_eq!(
            spec.length(),
            512,
            "length accessor should match constructor"
        );
        assert_eq!(
            spec.norm(),
            DftNorm::Inverse,
            "norm accessor should match constructor"
        );
    }

    #[test]
    fn c2r_spec_rejects_zero_length() {
        let result = DftC2rSpec::new(0, DftNorm::None);
        assert!(
            matches!(result, Err(Error::InvalidSpec(_))),
            "zero length must be rejected with InvalidSpec"
        );
    }

    #[test]
    fn c2r_spec_accessors_round_trip() {
        let spec = DftC2rSpec::new(768, DftNorm::Ortho).expect("non-zero length accepted");
        assert_eq!(
            spec.length(),
            768,
            "length accessor should match constructor"
        );
        assert_eq!(
            spec.norm(),
            DftNorm::Ortho,
            "norm accessor should match constructor"
        );
    }
}
