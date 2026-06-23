// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Element-wise vector operation primitive trait.
//!
//! [`VecOps`] is the second true primitive alongside DFT — the building block
//! that composite modules (Window, Conv, FIR, IIR, CQT, resampling) use for
//! the glue between FFT calls.  A vendor that provides [`DftC2c`](super::dft::DftC2c)
//! and `VecOps` gets every composite module for free.
//!
//! Unlike DFT, vector operations are stateless — there is nothing to
//! precompute.  No factory+plan pattern; methods are called directly on the
//! implementor (typically a unit struct like `RustVecOps` or `HwyVecOps`).

use num_complex::Complex;

use crate::error::{Error, Result};
use crate::types::DspFloat;

/// Check that two slice lengths match.
#[allow(
    clippy::missing_const_for_fn,
    reason = "Error construction is not const"
)]
pub(crate) fn check_lengths_2(a_len: usize, b_len: usize) -> Result<()> {
    if a_len != b_len {
        return Err(Error::BufferMismatch {
            expected: a_len,
            actual: b_len,
        });
    }
    Ok(())
}

/// Check that three slice lengths match.
#[allow(
    clippy::missing_const_for_fn,
    reason = "Error construction is not const"
)]
pub(crate) fn check_lengths_3(a_len: usize, b_len: usize, out_len: usize) -> Result<()> {
    if a_len != b_len {
        return Err(Error::BufferMismatch {
            expected: a_len,
            actual: b_len,
        });
    }
    if a_len != out_len {
        return Err(Error::BufferMismatch {
            expected: a_len,
            actual: out_len,
        });
    }
    Ok(())
}

/// Element-wise vector operations.
///
/// `T` is on the trait so that capability is expressed in the type system:
/// `impl VecOps<f32>` and `impl VecOps<f64>` are independent capabilities.
///
/// Every method has a scalar default implementation so that new methods can be
/// added without breaking existing backends.  Vendor backends override only the
/// methods they can accelerate.
///
/// Implementations are stateless and must be safely shareable across threads.
/// `Clone` is required because composite modules store their own `VecOps`
/// handle in each plan — for stateless unit structs this is zero-cost `Copy`.
pub trait VecOps<T: DspFloat>: Send + Sync + Clone {
    /// Element-wise multiply: `out[i] = a[i] * b[i]`.
    ///
    /// All three slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn mul(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
        Ok(())
    }

    /// Element-wise add: `out[i] = a[i] + b[i]`.
    ///
    /// All three slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn add(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
        Ok(())
    }

    /// Scale all elements in-place: `data[i] *= scalar`.
    fn scale(&self, data: &mut [T], scalar: T) {
        for x in data.iter_mut() {
            *x *= scalar;
        }
    }

    /// Dot product (inner product): `sum(a[i] * b[i])`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn dot(&self, a: &[T], b: &[T]) -> Result<T> {
        check_lengths_2(a.len(), b.len())?;
        Ok(a.iter().zip(b).fold(T::zero(), |acc, (&x, &y)| acc + x * y))
    }

    /// Element-wise complex multiply: `out[i] = a[i] * b[i]`.
    ///
    /// All three slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn cmul(&self, a: &[Complex<T>], b: &[Complex<T>], out: &mut [Complex<T>]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
        Ok(())
    }

    /// Complex dot product (unconjugated): `sum(a[i] * b[i])`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn cdot(&self, a: &[Complex<T>], b: &[Complex<T>]) -> Result<Complex<T>> {
        check_lengths_2(a.len(), b.len())?;
        Ok(a.iter()
            .zip(b)
            .fold(Complex::new(T::zero(), T::zero()), |acc, (&x, &y)| {
                acc + x * y
            }))
    }

    /// In-place element-wise multiply: `data[i] *= other[i]`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn mul_inplace(&self, data: &mut [T], other: &[T]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        for (x, &y) in data.iter_mut().zip(other) {
            *x *= y;
        }
        Ok(())
    }

    /// In-place element-wise add: `data[i] += other[i]`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn add_inplace(&self, data: &mut [T], other: &[T]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        for (x, &y) in data.iter_mut().zip(other) {
            *x += y;
        }
        Ok(())
    }

    /// In-place element-wise complex multiply: `data[i] *= other[i]`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn cmul_inplace(&self, data: &mut [Complex<T>], other: &[Complex<T>]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        for (x, &y) in data.iter_mut().zip(other) {
            *x = *x * y;
        }
        Ok(())
    }

    /// Complex magnitude squared: `out[i] = input[i].re² + input[i].im²`.
    ///
    /// `input` and `out` must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn mag_sq(&self, input: &[Complex<T>], out: &mut [T]) -> Result<()> {
        check_lengths_2(input.len(), out.len())?;
        for (o, z) in out.iter_mut().zip(input) {
            *o = z.re.mul_add(z.re, z.im * z.im);
        }
        Ok(())
    }

    /// Scale complex elements by corresponding real scalars in-place.
    ///
    /// `data[i].re *= scalars[i]`, `data[i].im *= scalars[i]`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn cscale_inplace(&self, data: &mut [Complex<T>], scalars: &[T]) -> Result<()> {
        check_lengths_2(data.len(), scalars.len())?;
        for (c, &s) in data.iter_mut().zip(scalars) {
            c.re *= s;
            c.im *= s;
        }
        Ok(())
    }

    /// Conjugate every element of a complex slice in-place.
    ///
    /// `data[i].im = -data[i].im`.
    fn conj_inplace(&self, data: &mut [Complex<T>]) {
        for c in data.iter_mut() {
            c.im = T::zero() - c.im;
        }
    }

    /// Pack real values into a complex buffer (imaginary parts set to zero).
    ///
    /// `out[i] = real[i] + 0i`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn real_to_complex(&self, real: &[T], out: &mut [Complex<T>]) -> Result<()> {
        check_lengths_2(real.len(), out.len())?;
        for (c, &r) in out.iter_mut().zip(real) {
            *c = Complex::new(r, T::zero());
        }
        Ok(())
    }

    /// Extract real parts from a complex buffer.
    ///
    /// `out[i] = input[i].re`.
    ///
    /// Both slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn extract_real(&self, input: &[Complex<T>], out: &mut [T]) -> Result<()> {
        check_lengths_2(input.len(), out.len())?;
        for (o, c) in out.iter_mut().zip(input) {
            *o = c.re;
        }
        Ok(())
    }

    /// Complex magnitude: `out[i] = sqrt(input[i].re² + input[i].im²)`.
    ///
    /// `input` and `out` must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn mag(&self, input: &[Complex<T>], out: &mut [T]) -> Result<()> {
        check_lengths_2(input.len(), out.len())?;
        for (o, z) in out.iter_mut().zip(input) {
            *o = z.re.mul_add(z.re, z.im * z.im).sqrt();
        }
        Ok(())
    }
}
