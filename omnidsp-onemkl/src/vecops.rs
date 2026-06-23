// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Accelerated [`VecOps`] for [`OneMklBackend`] via oneMKL VM + BLAS L1.
//!
//! The backend itself is the vector-ops provider (there is no separate
//! struct).  Only the methods with a direct oneMKL backing are overridden;
//! `mag_sq`, `mag`, `cscale_inplace`, `real_to_complex`, and `extract_real`
//! inherit the trait's scalar (LLVM auto-vectorized) defaults — an FFI crossing
//! is not justified for those trivial per-element loops.
//!
//! The two width impls are structurally identical apart from the MKL function
//! names, so a private `macro_rules!` generates both.

use num_complex::Complex;

use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::vecops::VecOps;

use crate::OneMklBackend;
use crate::ffi;

/// Check that two slice lengths match, returning [`Error::BufferMismatch`].
///
/// Replicates the core trait's `check_lengths_2` (which is crate-private and
/// not reachable from this backend crate).
#[allow(
    clippy::missing_const_for_fn,
    reason = "Error construction is not usable in const context here"
)]
fn check_lengths_2(a_len: usize, b_len: usize) -> Result<()> {
    if a_len == b_len {
        Ok(())
    } else {
        Err(Error::BufferMismatch {
            expected: a_len,
            actual: b_len,
        })
    }
}

/// Check that three slice lengths match, returning [`Error::BufferMismatch`].
///
/// Replicates the core trait's `check_lengths_3`.
#[allow(
    clippy::missing_const_for_fn,
    reason = "Error construction is not usable in const context here"
)]
fn check_lengths_3(a_len: usize, b_len: usize, out_len: usize) -> Result<()> {
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

/// Implement [`VecOps`] for `OneMklBackend` at one concrete float width.
///
/// The parameters thread the per-width `ffi::` wrapper names through the
/// otherwise-identical method bodies.
macro_rules! impl_vecops {
    (
        $float:ty,
        $mul:ident,
        $add:ident,
        $dot:ident,
        $axpy:ident,
        $scale:ident,
        $cmul:ident,
        $cdot:ident,
        $mul_inplace:ident,
        $cmul_inplace:ident,
        $conj_inplace:ident $(,)?
    ) => {
        impl VecOps<$float> for OneMklBackend {
            fn mul(&self, a: &[$float], b: &[$float], out: &mut [$float]) -> Result<()> {
                check_lengths_3(a.len(), b.len(), out.len())?;
                ffi::$mul(a, b, out)
            }

            fn add(&self, a: &[$float], b: &[$float], out: &mut [$float]) -> Result<()> {
                check_lengths_3(a.len(), b.len(), out.len())?;
                ffi::$add(a, b, out)
            }

            fn scale(&self, data: &mut [$float], scalar: $float) {
                // `scale` is infallible: on the (astronomically unlikely)
                // overflow of `usize → MKL_INT`, fall back to a scalar loop
                // rather than panic.
                if let Some(n) = ffi::try_mkl_int(data.len()) {
                    ffi::$scale(data, scalar, n);
                } else {
                    for x in data.iter_mut() {
                        *x *= scalar;
                    }
                }
            }

            fn dot(&self, a: &[$float], b: &[$float]) -> Result<$float> {
                check_lengths_2(a.len(), b.len())?;
                ffi::$dot(a, b)
            }

            fn cmul(
                &self,
                a: &[Complex<$float>],
                b: &[Complex<$float>],
                out: &mut [Complex<$float>],
            ) -> Result<()> {
                check_lengths_3(a.len(), b.len(), out.len())?;
                ffi::$cmul(a, b, out)
            }

            fn cdot(
                &self,
                a: &[Complex<$float>],
                b: &[Complex<$float>],
            ) -> Result<Complex<$float>> {
                check_lengths_2(a.len(), b.len())?;
                ffi::$cdot(a, b)
            }

            fn mul_inplace(&self, data: &mut [$float], other: &[$float]) -> Result<()> {
                check_lengths_2(data.len(), other.len())?;
                ffi::$mul_inplace(data, other)
            }

            fn add_inplace(&self, data: &mut [$float], other: &[$float]) -> Result<()> {
                check_lengths_2(data.len(), other.len())?;
                ffi::$axpy(data, other)
            }

            fn cmul_inplace(
                &self,
                data: &mut [Complex<$float>],
                other: &[Complex<$float>],
            ) -> Result<()> {
                check_lengths_2(data.len(), other.len())?;
                ffi::$cmul_inplace(data, other)
            }

            fn conj_inplace(&self, data: &mut [Complex<$float>]) {
                // `conj_inplace` is infallible: on `usize → MKL_INT` overflow,
                // fall back to a scalar loop rather than panic.
                if ffi::$conj_inplace(data).is_err() {
                    for c in data.iter_mut() {
                        c.im = -c.im;
                    }
                }
            }
        }
    };
}

impl_vecops!(
    f32,
    vmul_f32,
    vadd_f32,
    dot_f32,
    axpy_f32,
    scale_f32,
    cmul_f32,
    cdot_f32,
    mul_inplace_f32,
    cmul_inplace_f32,
    conj_inplace_f32,
);

impl_vecops!(
    f64,
    vmul_f64,
    vadd_f64,
    dot_f64,
    axpy_f64,
    scale_f64,
    cmul_f64,
    cdot_f64,
    mul_inplace_f64,
    cmul_inplace_f64,
    conj_inplace_f64,
);
