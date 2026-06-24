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
    /// Uses four independent partial accumulators advanced in lockstep, with a
    /// scalar tail for the remainder, so the body is a set of parallel
    /// fused-multiply-add chains rather than one serial dependency chain — this
    /// lets the auto-vectorizer pack lanes and keeps the FMA pipeline full.
    /// Because the partials are summed in a different order than a single
    /// running accumulator, the result may differ from a naive left fold in the
    /// last unit(s) in the last place; this is expected and acceptable for a
    /// dot product.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn dot(&self, a: &[T], b: &[T]) -> Result<T> {
        const LANES: usize = 4;

        check_lengths_2(a.len(), b.len())?;

        let mut partials = [T::zero(); LANES];

        let mut a_chunks = a.chunks_exact(LANES);
        let mut b_chunks = b.chunks_exact(LANES);
        for (ac, bc) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            for lane in 0..LANES {
                partials[lane] = ac[lane].mul_add(bc[lane], partials[lane]);
            }
        }

        // Scalar remainder tail (fewer than LANES elements left).
        let mut tail = T::zero();
        for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
            tail = x.mul_add(y, tail);
        }

        // Combine the partials pairwise, then fold in the tail.
        let lo = partials[0] + partials[1];
        let hi = partials[2] + partials[3];
        Ok(lo + hi + tail)
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
    /// Mirrors [`dot`](Self::dot): four independent partials (carried as
    /// separate real and imaginary running sums so each is its own fused
    /// multiply-add chain) advanced in lockstep, plus a scalar tail, combined
    /// at the end.  Breaking the single serial accumulator lets the
    /// auto-vectorizer pack lanes and keep the FMA pipeline full.  The reordered
    /// summation may differ from a naive left fold in the last unit(s) in the
    /// last place, which is expected and acceptable for a dot product.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths do not match.
    fn cdot(&self, a: &[Complex<T>], b: &[Complex<T>]) -> Result<Complex<T>> {
        const LANES: usize = 4;

        check_lengths_2(a.len(), b.len())?;

        let mut re = [T::zero(); LANES];
        let mut im = [T::zero(); LANES];

        let mut a_chunks = a.chunks_exact(LANES);
        let mut b_chunks = b.chunks_exact(LANES);
        for (ac, bc) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            for lane in 0..LANES {
                let (x, y) = (ac[lane], bc[lane]);
                // (x.re + i·x.im)·(y.re + i·y.im)
                //   re = x.re·y.re − x.im·y.im
                //   im = x.re·y.im + x.im·y.re
                re[lane] = x.re.mul_add(y.re, re[lane]);
                re[lane] = (-x.im).mul_add(y.im, re[lane]);
                im[lane] = x.re.mul_add(y.im, im[lane]);
                im[lane] = x.im.mul_add(y.re, im[lane]);
            }
        }

        // Scalar remainder tail (fewer than LANES elements left).
        let mut re_tail = T::zero();
        let mut im_tail = T::zero();
        for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
            re_tail = x.re.mul_add(y.re, re_tail);
            re_tail = (-x.im).mul_add(y.im, re_tail);
            im_tail = x.re.mul_add(y.im, im_tail);
            im_tail = x.im.mul_add(y.re, im_tail);
        }

        // Combine the partials pairwise, then fold in the tail.
        let re_sum = (re[0] + re[1]) + (re[2] + re[3]) + re_tail;
        let im_sum = (im[0] + im[1]) + (im[2] + im[3]) + im_tail;
        Ok(Complex::new(re_sum, im_sum))
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
