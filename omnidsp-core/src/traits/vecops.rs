// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Element-wise vector operation primitive trait.
//!
//! [`VecOps`] is the second true primitive alongside DFT — the building block
//! that composite modules (Window, Conv, FIR, IIR, CQT, resampling) use for
//! the glue between FFT calls.  A vendor that provides [`Dft`](super::dft::Dft)
//! and `VecOps` gets every composite module for free.
//!
//! Unlike DFT, vector operations are stateless — there is nothing to
//! precompute.  No factory+plan pattern; methods are called directly on the
//! implementor (typically a unit struct like `RustVecOps` or `HwyVecOps`).

use num_complex::Complex;

use crate::error::Result;

/// Element-wise vector operations.
///
/// `T` is on the trait so that capability is expressed in the type system:
/// `impl VecOps<f32>` and `impl VecOps<f64>` are independent capabilities.
///
/// Implementations are stateless and must be safely shareable across threads.
pub trait VecOps<T>: Send + Sync {
    /// Element-wise multiply: `out[i] = a[i] * b[i]`.
    ///
    /// All three slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn mul(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>;

    /// Element-wise add: `out[i] = a[i] + b[i]`.
    ///
    /// All three slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn add(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>;

    /// Scale all elements in-place: `data[i] *= scalar`.
    fn scale(&self, data: &mut [T], scalar: T);

    /// Dot product (inner product): `sum(a[i] * b[i])`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn dot(&self, a: &[T], b: &[T]) -> Result<T>;

    /// Element-wise complex multiply: `out[i] = a[i] * b[i]`.
    ///
    /// All three slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn cmul(&self, a: &[Complex<T>], b: &[Complex<T>], out: &mut [Complex<T>]) -> Result<()>;

    /// In-place element-wise multiply: `data[i] *= other[i]`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn mul_inplace(&self, data: &mut [T], other: &[T]) -> Result<()>;

    /// In-place element-wise add: `data[i] += other[i]`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn add_inplace(&self, data: &mut [T], other: &[T]) -> Result<()>;

    /// In-place element-wise complex multiply: `data[i] *= other[i]`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn cmul_inplace(&self, data: &mut [Complex<T>], other: &[Complex<T>]) -> Result<()>;
}
