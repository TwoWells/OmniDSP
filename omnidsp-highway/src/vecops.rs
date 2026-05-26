// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! [`HwyVecOps`] — Highway SIMD implementations of element-wise operations.

use num_complex::Complex;

use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::vecops::VecOps;

/// Highway SIMD element-wise vector operations.
///
/// Each method calls the corresponding Highway SIMD kernel from `highway-sys`.
/// Stateless unit struct — trivially `Send + Sync + Copy`.
#[derive(Debug, Clone, Copy)]
pub struct HwyVecOps;

#[allow(
    clippy::missing_const_for_fn,
    reason = "Error construction is not const"
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

#[allow(
    clippy::missing_const_for_fn,
    reason = "Error construction is not const"
)]
fn check_lengths_2(a_len: usize, b_len: usize) -> Result<()> {
    if a_len != b_len {
        return Err(Error::BufferMismatch {
            expected: a_len,
            actual: b_len,
        });
    }
    Ok(())
}

impl VecOps<f32> for HwyVecOps {
    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        // SAFETY: pointers are valid for `a.len()` elements. Rust's borrow rules
        // guarantee `&[f32]` and `&mut [f32]` don't overlap, so the no-alias
        // variant is sound — the compiler gets HWY_RESTRICT on all three pointers.
        unsafe {
            highway_sys::omnidsp_mul_f32_noalias(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
        }
        Ok(())
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        // SAFETY: pointers are valid for `a.len()` elements; out may alias a or b.
        unsafe { highway_sys::omnidsp_add_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len()) };
        Ok(())
    }

    fn scale(&self, data: &mut [f32], scalar: f32) {
        // SAFETY: pointer is valid for `data.len()` elements.
        unsafe { highway_sys::omnidsp_scale_f32(data.as_mut_ptr(), scalar, data.len()) };
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        check_lengths_2(a.len(), b.len())?;
        // SAFETY: pointers are valid for `a.len()` elements.
        Ok(unsafe { highway_sys::omnidsp_dot_f32(a.as_ptr(), b.as_ptr(), a.len()) })
    }

    fn cmul(&self, a: &[Complex<f32>], b: &[Complex<f32>], out: &mut [Complex<f32>]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        // SAFETY: Complex<f32> has the same layout as [f32; 2]. Pointers are valid
        // for `a.len()` complex elements (= `a.len() * 2` floats). Rust's borrow
        // rules guarantee no aliasing, so the no-alias variant is sound.
        unsafe {
            highway_sys::omnidsp_cmul_f32_noalias(
                a.as_ptr().cast::<f32>(),
                b.as_ptr().cast::<f32>(),
                out.as_mut_ptr().cast::<f32>(),
                a.len(),
            );
        }
        Ok(())
    }

    fn mul_inplace(&self, data: &mut [f32], other: &[f32]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        // SAFETY: pointers are valid for `data.len()` elements; out aliases data
        // which the shim supports.
        unsafe {
            highway_sys::omnidsp_mul_f32(
                other.as_ptr(),
                data.as_ptr(),
                data.as_mut_ptr(),
                data.len(),
            );
        }
        Ok(())
    }

    fn add_inplace(&self, data: &mut [f32], other: &[f32]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        // SAFETY: pointers are valid for `data.len()` elements; out aliases data
        // which the shim supports.
        unsafe {
            highway_sys::omnidsp_add_f32(
                other.as_ptr(),
                data.as_ptr(),
                data.as_mut_ptr(),
                data.len(),
            );
        }
        Ok(())
    }

    fn cmul_inplace(&self, data: &mut [Complex<f32>], other: &[Complex<f32>]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        // SAFETY: Complex<f32> has the same layout as [f32; 2]. Pointers are valid
        // for `data.len()` complex elements. Out aliases data which the shim supports.
        unsafe {
            highway_sys::omnidsp_cmul_f32(
                other.as_ptr().cast::<f32>(),
                data.as_ptr().cast::<f32>(),
                data.as_mut_ptr().cast::<f32>(),
                data.len(),
            );
        }
        Ok(())
    }
}

impl VecOps<f64> for HwyVecOps {
    fn mul(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        // SAFETY: pointers are valid for `a.len()` elements. Rust's borrow rules
        // guarantee no aliasing, so the no-alias variant is sound.
        unsafe {
            highway_sys::omnidsp_mul_f64_noalias(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
        }
        Ok(())
    }

    fn add(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        // SAFETY: pointers are valid for `a.len()` elements; out may alias a or b.
        unsafe { highway_sys::omnidsp_add_f64(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len()) };
        Ok(())
    }

    fn scale(&self, data: &mut [f64], scalar: f64) {
        // SAFETY: pointer is valid for `data.len()` elements.
        unsafe { highway_sys::omnidsp_scale_f64(data.as_mut_ptr(), scalar, data.len()) };
    }

    fn dot(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        check_lengths_2(a.len(), b.len())?;
        // SAFETY: pointers are valid for `a.len()` elements.
        Ok(unsafe { highway_sys::omnidsp_dot_f64(a.as_ptr(), b.as_ptr(), a.len()) })
    }

    fn cmul(&self, a: &[Complex<f64>], b: &[Complex<f64>], out: &mut [Complex<f64>]) -> Result<()> {
        check_lengths_3(a.len(), b.len(), out.len())?;
        // SAFETY: Complex<f64> has the same layout as [f64; 2]. Pointers are valid
        // for `a.len()` complex elements (= `a.len() * 2` doubles). Rust's borrow
        // rules guarantee no aliasing, so the no-alias variant is sound.
        unsafe {
            highway_sys::omnidsp_cmul_f64_noalias(
                a.as_ptr().cast::<f64>(),
                b.as_ptr().cast::<f64>(),
                out.as_mut_ptr().cast::<f64>(),
                a.len(),
            );
        }
        Ok(())
    }

    fn mul_inplace(&self, data: &mut [f64], other: &[f64]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        // SAFETY: pointers are valid for `data.len()` elements; out aliases data
        // which the shim supports.
        unsafe {
            highway_sys::omnidsp_mul_f64(
                other.as_ptr(),
                data.as_ptr(),
                data.as_mut_ptr(),
                data.len(),
            );
        }
        Ok(())
    }

    fn add_inplace(&self, data: &mut [f64], other: &[f64]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        // SAFETY: pointers are valid for `data.len()` elements; out aliases data
        // which the shim supports.
        unsafe {
            highway_sys::omnidsp_add_f64(
                other.as_ptr(),
                data.as_ptr(),
                data.as_mut_ptr(),
                data.len(),
            );
        }
        Ok(())
    }

    fn cmul_inplace(&self, data: &mut [Complex<f64>], other: &[Complex<f64>]) -> Result<()> {
        check_lengths_2(data.len(), other.len())?;
        // SAFETY: Complex<f64> has the same layout as [f64; 2]. Pointers are valid
        // for `data.len()` complex elements. Out aliases data which the shim supports.
        unsafe {
            highway_sys::omnidsp_cmul_f64(
                other.as_ptr().cast::<f64>(),
                data.as_ptr().cast::<f64>(),
                data.as_mut_ptr().cast::<f64>(),
                data.len(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
#[allow(clippy::float_cmp, reason = "test values are exact small integers")]
mod tests {
    use super::*;

    const EPSILON_F32: f32 = 1e-6;
    const EPSILON_F64: f64 = 1e-12;

    // --- mul ---

    #[test]
    fn mul_f32() {
        let ops = HwyVecOps;
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let mut out = [0.0_f32; 3];
        ops.mul(&a, &b, &mut out).expect("mul should succeed");
        assert_eq!(out, [4.0, 10.0, 18.0], "element-wise multiply failed");
    }

    #[test]
    fn mul_f64() {
        let ops = HwyVecOps;
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        let mut out = [0.0_f64; 3];
        ops.mul(&a, &b, &mut out).expect("mul should succeed");
        assert_eq!(out, [4.0, 10.0, 18.0], "element-wise multiply failed");
    }

    #[test]
    fn mul_length_mismatch() {
        let ops = HwyVecOps;
        let a = [1.0_f32; 3];
        let b = [1.0_f32; 4];
        let mut out = [0.0_f32; 3];
        assert!(
            ops.mul(&a, &b, &mut out).is_err(),
            "mismatched lengths should return error"
        );
    }

    // --- add ---

    #[test]
    fn add_f32() {
        let ops = HwyVecOps;
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let mut out = [0.0_f32; 3];
        ops.add(&a, &b, &mut out).expect("add should succeed");
        assert_eq!(out, [5.0, 7.0, 9.0], "element-wise add failed");
    }

    // --- scale ---

    #[test]
    fn scale_f64() {
        let ops = HwyVecOps;
        let mut data = [1.0_f64, 2.0, 3.0];
        ops.scale(&mut data, 2.0);
        assert_eq!(data, [2.0, 4.0, 6.0], "scale failed");
    }

    // --- dot ---

    #[test]
    fn dot_f32() {
        let ops = HwyVecOps;
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let result = ops.dot(&a, &b).expect("dot should succeed");
        assert!(
            (result - 32.0).abs() < EPSILON_F32,
            "dot product should be 32.0, got {result}"
        );
    }

    #[test]
    fn dot_f64() {
        let ops = HwyVecOps;
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        let result = ops.dot(&a, &b).expect("dot should succeed");
        assert!(
            (result - 32.0).abs() < EPSILON_F64,
            "dot product should be 32.0, got {result}"
        );
    }

    #[test]
    fn dot_length_mismatch() {
        let ops = HwyVecOps;
        let a = [1.0_f32; 3];
        let b = [1.0_f32; 2];
        assert!(
            ops.dot(&a, &b).is_err(),
            "mismatched lengths should return error"
        );
    }

    // --- cmul ---

    #[test]
    fn cmul_f32() {
        let ops = HwyVecOps;
        let a = [Complex::new(1.0_f32, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0_f32, 6.0), Complex::new(7.0, 8.0)];
        let mut out = [Complex::default(); 2];
        ops.cmul(&a, &b, &mut out).expect("cmul should succeed");

        // (1+2i)*(5+6i) = -7+16i, (3+4i)*(7+8i) = -11+52i
        assert!(
            (out[0].re - (-7.0)).abs() < EPSILON_F32 && (out[0].im - 16.0).abs() < EPSILON_F32,
            "cmul[0] failed: got {:?}",
            out[0]
        );
        assert!(
            (out[1].re - (-11.0)).abs() < EPSILON_F32 && (out[1].im - 52.0).abs() < EPSILON_F32,
            "cmul[1] failed: got {:?}",
            out[1]
        );
    }

    // --- mul_inplace ---

    #[test]
    fn mul_inplace_f32() {
        let ops = HwyVecOps;
        let mut data = [1.0_f32, 2.0, 3.0];
        let other = [4.0_f32, 5.0, 6.0];
        ops.mul_inplace(&mut data, &other)
            .expect("mul_inplace should succeed");
        assert_eq!(data, [4.0, 10.0, 18.0], "in-place multiply failed");
    }

    // --- add_inplace ---

    #[test]
    fn add_inplace_f64() {
        let ops = HwyVecOps;
        let mut data = [1.0_f64, 2.0, 3.0];
        let other = [4.0_f64, 5.0, 6.0];
        ops.add_inplace(&mut data, &other)
            .expect("add_inplace should succeed");
        assert_eq!(data, [5.0, 7.0, 9.0], "in-place add failed");
    }

    // --- cmul_inplace ---

    #[test]
    fn cmul_inplace_f64() {
        let ops = HwyVecOps;
        let mut data = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let other = [Complex::new(5.0_f64, 6.0), Complex::new(7.0, 8.0)];
        ops.cmul_inplace(&mut data, &other)
            .expect("cmul_inplace should succeed");

        assert!(
            (data[0].re - (-7.0)).abs() < EPSILON_F64 && (data[0].im - 16.0).abs() < EPSILON_F64,
            "cmul_inplace[0] failed: got {:?}",
            data[0]
        );
        assert!(
            (data[1].re - (-11.0)).abs() < EPSILON_F64 && (data[1].im - 52.0).abs() < EPSILON_F64,
            "cmul_inplace[1] failed: got {:?}",
            data[1]
        );
    }

    // --- empty slices ---

    #[test]
    fn empty_slices() {
        let ops = HwyVecOps;
        let empty: &[f32] = &[];
        let mut out: [f32; 0] = [];

        ops.mul(empty, empty, &mut out)
            .expect("empty mul should succeed");
        ops.add(empty, empty, &mut out)
            .expect("empty add should succeed");
        ops.scale(&mut out, 2.0);

        let dot = ops.dot(empty, empty).expect("empty dot should succeed");
        assert!((dot - 0.0).abs() < EPSILON_F32, "empty dot should be 0.0");

        let empty_c: &[Complex<f32>] = &[];
        let mut out_c: [Complex<f32>; 0] = [];
        ops.cmul(empty_c, empty_c, &mut out_c)
            .expect("empty cmul should succeed");
    }

    // --- single element ---

    #[test]
    fn single_element() {
        let ops = HwyVecOps;
        let a = [3.0_f64];
        let b = [7.0_f64];
        let mut out = [0.0_f64];
        ops.mul(&a, &b, &mut out)
            .expect("single mul should succeed");
        assert_eq!(out, [21.0], "single element mul failed");

        let dot = ops.dot(&a, &b).expect("single dot should succeed");
        assert!(
            (dot - 21.0).abs() < EPSILON_F64,
            "single element dot should be 21.0"
        );
    }
}
