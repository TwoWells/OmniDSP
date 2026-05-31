// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! [`ScalarVecOps`] — scalar loop implementations of element-wise operations.

use num_complex::Complex;

use crate::error::{Error, Result};
use crate::traits::vecops::VecOps;

/// Scalar element-wise vector operations.
///
/// Scalar loop implementations that LLVM auto-vectorizes.  Stateless unit
/// struct — trivially `Send + Sync + Copy`.
#[derive(Debug, Clone, Copy)]
pub struct ScalarVecOps;

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

macro_rules! impl_vecops {
    ($t:ty) => {
        impl VecOps<$t> for ScalarVecOps {
            fn mul(&self, a: &[$t], b: &[$t], out: &mut [$t]) -> Result<()> {
                check_lengths_3(a.len(), b.len(), out.len())?;
                for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
                    *o = x * y;
                }
                Ok(())
            }

            fn add(&self, a: &[$t], b: &[$t], out: &mut [$t]) -> Result<()> {
                check_lengths_3(a.len(), b.len(), out.len())?;
                for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
                    *o = x + y;
                }
                Ok(())
            }

            fn scale(&self, data: &mut [$t], scalar: $t) {
                for x in data.iter_mut() {
                    *x *= scalar;
                }
            }

            fn dot(&self, a: &[$t], b: &[$t]) -> Result<$t> {
                check_lengths_2(a.len(), b.len())?;
                Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
            }

            fn cmul(
                &self,
                a: &[Complex<$t>],
                b: &[Complex<$t>],
                out: &mut [Complex<$t>],
            ) -> Result<()> {
                check_lengths_3(a.len(), b.len(), out.len())?;
                for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
                    *o = x * y;
                }
                Ok(())
            }

            fn mul_inplace(&self, data: &mut [$t], other: &[$t]) -> Result<()> {
                check_lengths_2(data.len(), other.len())?;
                for (x, y) in data.iter_mut().zip(other) {
                    *x *= y;
                }
                Ok(())
            }

            fn add_inplace(&self, data: &mut [$t], other: &[$t]) -> Result<()> {
                check_lengths_2(data.len(), other.len())?;
                for (x, y) in data.iter_mut().zip(other) {
                    *x += y;
                }
                Ok(())
            }

            fn cmul_inplace(&self, data: &mut [Complex<$t>], other: &[Complex<$t>]) -> Result<()> {
                check_lengths_2(data.len(), other.len())?;
                for (x, y) in data.iter_mut().zip(other) {
                    *x *= y;
                }
                Ok(())
            }

            fn mag_sq(&self, input: &[Complex<$t>], out: &mut [$t]) -> Result<()> {
                check_lengths_2(input.len(), out.len())?;
                for (o, z) in out.iter_mut().zip(input) {
                    *o = z.re.mul_add(z.re, z.im * z.im);
                }
                Ok(())
            }

            fn cscale_inplace(&self, data: &mut [Complex<$t>], scalars: &[$t]) -> Result<()> {
                check_lengths_2(data.len(), scalars.len())?;
                for (c, &s) in data.iter_mut().zip(scalars) {
                    c.re *= s;
                    c.im *= s;
                }
                Ok(())
            }
        }
    };
}

impl_vecops!(f32);
impl_vecops!(f64);

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
        let ops = ScalarVecOps;
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let mut out = [0.0_f32; 3];
        ops.mul(&a, &b, &mut out).expect("mul should succeed");
        assert_eq!(out, [4.0, 10.0, 18.0], "element-wise multiply failed");
    }

    #[test]
    fn mul_f64() {
        let ops = ScalarVecOps;
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        let mut out = [0.0_f64; 3];
        ops.mul(&a, &b, &mut out).expect("mul should succeed");
        assert_eq!(out, [4.0, 10.0, 18.0], "element-wise multiply failed");
    }

    #[test]
    fn mul_length_mismatch() {
        let ops = ScalarVecOps;
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
        let ops = ScalarVecOps;
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let mut out = [0.0_f32; 3];
        ops.add(&a, &b, &mut out).expect("add should succeed");
        assert_eq!(out, [5.0, 7.0, 9.0], "element-wise add failed");
    }

    // --- scale ---

    #[test]
    fn scale_f64() {
        let ops = ScalarVecOps;
        let mut data = [1.0_f64, 2.0, 3.0];
        ops.scale(&mut data, 2.0);
        assert_eq!(data, [2.0, 4.0, 6.0], "scale failed");
    }

    // --- dot ---

    #[test]
    fn dot_f32() {
        let ops = ScalarVecOps;
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
        let ops = ScalarVecOps;
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
        let ops = ScalarVecOps;
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
        let ops = ScalarVecOps;
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
        let ops = ScalarVecOps;
        let mut data = [1.0_f32, 2.0, 3.0];
        let other = [4.0_f32, 5.0, 6.0];
        ops.mul_inplace(&mut data, &other)
            .expect("mul_inplace should succeed");
        assert_eq!(data, [4.0, 10.0, 18.0], "in-place multiply failed");
    }

    // --- add_inplace ---

    #[test]
    fn add_inplace_f64() {
        let ops = ScalarVecOps;
        let mut data = [1.0_f64, 2.0, 3.0];
        let other = [4.0_f64, 5.0, 6.0];
        ops.add_inplace(&mut data, &other)
            .expect("add_inplace should succeed");
        assert_eq!(data, [5.0, 7.0, 9.0], "in-place add failed");
    }

    // --- cmul_inplace ---

    #[test]
    fn cmul_inplace_f64() {
        let ops = ScalarVecOps;
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

    // --- mag_sq ---

    #[test]
    fn mag_sq_f32() {
        let ops = ScalarVecOps;
        let input = [Complex::new(3.0_f32, 4.0), Complex::new(1.0, 1.0)];
        let mut out = [0.0_f32; 2];
        ops.mag_sq(&input, &mut out).expect("mag_sq should succeed");
        assert!(
            (out[0] - 25.0).abs() < EPSILON_F32,
            "mag_sq[0] should be 25.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 2.0).abs() < EPSILON_F32,
            "mag_sq[1] should be 2.0, got {}",
            out[1]
        );
    }

    #[test]
    fn mag_sq_f64() {
        let ops = ScalarVecOps;
        let input = [Complex::new(3.0_f64, 4.0), Complex::new(0.0, 5.0)];
        let mut out = [0.0_f64; 2];
        ops.mag_sq(&input, &mut out).expect("mag_sq should succeed");
        assert!(
            (out[0] - 25.0).abs() < EPSILON_F64,
            "mag_sq[0] should be 25.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 25.0).abs() < EPSILON_F64,
            "mag_sq[1] should be 25.0, got {}",
            out[1]
        );
    }

    #[test]
    fn mag_sq_length_mismatch() {
        let ops = ScalarVecOps;
        let input = [Complex::new(1.0_f32, 2.0); 3];
        let mut out = [0.0_f32; 2];
        assert!(
            ops.mag_sq(&input, &mut out).is_err(),
            "mismatched lengths should return error"
        );
    }

    // --- cscale_inplace ---

    #[test]
    fn cscale_inplace_f32() {
        let ops = ScalarVecOps;
        let mut data = [Complex::new(1.0_f32, 2.0), Complex::new(3.0, 4.0)];
        let scalars = [2.0_f32, 0.5];
        ops.cscale_inplace(&mut data, &scalars)
            .expect("cscale_inplace should succeed");
        assert!(
            (data[0].re - 2.0).abs() < EPSILON_F32 && (data[0].im - 4.0).abs() < EPSILON_F32,
            "cscale_inplace[0] failed: got {:?}",
            data[0]
        );
        assert!(
            (data[1].re - 1.5).abs() < EPSILON_F32 && (data[1].im - 2.0).abs() < EPSILON_F32,
            "cscale_inplace[1] failed: got {:?}",
            data[1]
        );
    }

    #[test]
    fn cscale_inplace_f64() {
        let ops = ScalarVecOps;
        let mut data = [Complex::new(5.0_f64, -3.0), Complex::new(0.0, 1.0)];
        let scalars = [0.0_f64, 2.0];
        ops.cscale_inplace(&mut data, &scalars)
            .expect("cscale_inplace should succeed");
        assert!(
            (data[0].re).abs() < EPSILON_F64 && (data[0].im).abs() < EPSILON_F64,
            "cscale by zero should zero the element: got {:?}",
            data[0]
        );
        assert!(
            (data[1].re).abs() < EPSILON_F64 && (data[1].im - 2.0).abs() < EPSILON_F64,
            "cscale_inplace[1] failed: got {:?}",
            data[1]
        );
    }

    #[test]
    fn cscale_inplace_length_mismatch() {
        let ops = ScalarVecOps;
        let mut data = [Complex::new(1.0_f32, 2.0); 3];
        let scalars = [1.0_f32; 2];
        assert!(
            ops.cscale_inplace(&mut data, &scalars).is_err(),
            "mismatched lengths should return error"
        );
    }

    // --- empty slices ---

    #[test]
    fn empty_slices() {
        let ops = ScalarVecOps;
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

        let mut empty_real: [f32; 0] = [];
        ops.mag_sq(empty_c, &mut empty_real)
            .expect("empty mag_sq should succeed");
    }

    // --- single element ---

    #[test]
    fn single_element() {
        let ops = ScalarVecOps;
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
