// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Raw FFI bindings to Highway SIMD kernels via the `OmniDSP` shim.
//!
//! This crate is `unsafe` by nature — it's a `-sys` crate. The safety
//! boundary lives in `omnidsp-highway`.

#![allow(unsafe_code, reason = "-sys crate: FFI declarations require unsafe")]

unsafe extern "C" {
    /// Element-wise multiply: `out[i] = a[i] * b[i]` for `i` in `0..count`.
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count` floats.
    /// - `out` may alias `a` or `b` (supports in-place operation).
    pub fn omnidsp_mul_f32(a: *const f32, b: *const f32, out: *mut f32, count: usize);

    /// Element-wise multiply: `out[i] = a[i] * b[i]` for `i` in `0..count`.
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count` doubles.
    /// - `out` may alias `a` or `b` (supports in-place operation).
    pub fn omnidsp_mul_f64(a: *const f64, b: *const f64, out: *mut f64, count: usize);

    /// Element-wise add: `out[i] = a[i] + b[i]` for `i` in `0..count`.
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count` floats.
    /// - `out` may alias `a` or `b` (supports in-place operation).
    pub fn omnidsp_add_f32(a: *const f32, b: *const f32, out: *mut f32, count: usize);

    /// Element-wise add: `out[i] = a[i] + b[i]` for `i` in `0..count`.
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count` doubles.
    /// - `out` may alias `a` or `b` (supports in-place operation).
    pub fn omnidsp_add_f64(a: *const f64, b: *const f64, out: *mut f64, count: usize);

    /// Scale in-place: `data[i] *= scalar` for `i` in `0..count`.
    ///
    /// # Safety
    ///
    /// - `data` must point to at least `count` floats.
    pub fn omnidsp_scale_f32(data: *mut f32, scalar: f32, count: usize);

    /// Scale in-place: `data[i] *= scalar` for `i` in `0..count`.
    ///
    /// # Safety
    ///
    /// - `data` must point to at least `count` doubles.
    pub fn omnidsp_scale_f64(data: *mut f64, scalar: f64, count: usize);

    /// Dot product: returns `sum(a[i] * b[i])` for `i` in `0..count`.
    ///
    /// # Safety
    ///
    /// - `a` and `b` must point to at least `count` floats.
    pub fn omnidsp_dot_f32(a: *const f32, b: *const f32, count: usize) -> f32;

    /// Dot product: returns `sum(a[i] * b[i])` for `i` in `0..count`.
    ///
    /// # Safety
    ///
    /// - `a` and `b` must point to at least `count` doubles.
    pub fn omnidsp_dot_f64(a: *const f64, b: *const f64, count: usize) -> f64;

    /// Complex element-wise multiply (interleaved `[re, im]` layout).
    /// `count` is the number of complex elements (raw floats = `count * 2`).
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count * 2` floats.
    /// - `out` may alias `a` or `b` (supports in-place operation).
    pub fn omnidsp_cmul_f32(a: *const f32, b: *const f32, out: *mut f32, count: usize);

    /// Complex element-wise multiply (interleaved `[re, im]` layout).
    /// `count` is the number of complex elements (raw floats = `count * 2`).
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count * 2` doubles.
    /// - `out` may alias `a` or `b` (supports in-place operation).
    pub fn omnidsp_cmul_f64(a: *const f64, b: *const f64, out: *mut f64, count: usize);

    /// No-alias element-wise multiply (f32): `out[i] = a[i] * b[i]`.
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count` floats.
    /// - `a`, `b`, and `out` must NOT alias each other.
    pub fn omnidsp_mul_f32_noalias(a: *const f32, b: *const f32, out: *mut f32, count: usize);

    /// No-alias element-wise multiply (f64): `out[i] = a[i] * b[i]`.
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count` doubles.
    /// - `a`, `b`, and `out` must NOT alias each other.
    pub fn omnidsp_mul_f64_noalias(a: *const f64, b: *const f64, out: *mut f64, count: usize);

    /// No-alias complex element-wise multiply (f32, interleaved `[re, im]`).
    /// `count` is the number of complex elements (raw floats = `count * 2`).
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count * 2` floats.
    /// - `a`, `b`, and `out` must NOT alias each other.
    pub fn omnidsp_cmul_f32_noalias(a: *const f32, b: *const f32, out: *mut f32, count: usize);

    /// No-alias complex element-wise multiply (f64, interleaved `[re, im]`).
    /// `count` is the number of complex elements (raw doubles = `count * 2`).
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `out` must point to at least `count * 2` doubles.
    /// - `a`, `b`, and `out` must NOT alias each other.
    pub fn omnidsp_cmul_f64_noalias(a: *const f64, b: *const f64, out: *mut f64, count: usize);

    /// One stage of a radix-2 Cooley-Tukey FFT butterfly (f32).
    ///
    /// Processes all butterfly groups for a stage with the given `half_len`.
    /// `data` is interleaved `[re, im]` complex (`n * 2` floats).
    /// `twiddles` is interleaved `[re, im]` (`half_len * 2` floats).
    /// `n` is the FFT length (number of complex elements).
    ///
    /// # Safety
    ///
    /// - `data` must point to at least `n * 2` floats.
    /// - `twiddles` must point to at least `half_len * 2` floats.
    /// - `half_len` must be a power of 2 and `<= n / 2`.
    pub fn omnidsp_butterfly_stage_f32(
        data: *mut f32,
        twiddles: *const f32,
        half_len: usize,
        n: usize,
    );

    /// One radix-4 butterfly stage (f32) — fuses two consecutive radix-2 stages.
    ///
    /// Processes all radix-4 groups for a stage pair with the given `quarter_len`.
    /// `data` is interleaved `[re, im]` complex (`n * 2` floats).
    /// `twiddles` layout: `[W1[0..q], W2[0..q], W3[0..q]]` where `q = quarter_len`.
    /// `W1[k]` applied to sub-block 2, `W2[k]` to sub-block 1, `W3[k]` to sub-block 3.
    /// Total twiddle floats: `6 * quarter_len`.
    /// `forward`: 1 for forward FFT (−j rotation), 0 for inverse (+j rotation).
    ///
    /// # Safety
    ///
    /// - `data` must point to at least `n * 2` floats.
    /// - `twiddles` must point to at least `quarter_len * 6` floats.
    /// - `quarter_len` must be a power of 2 and `<= n / 4`.
    pub fn omnidsp_butterfly_stage4_f32(
        data: *mut f32,
        twiddles: *const f32,
        quarter_len: usize,
        n: usize,
        forward: i32,
    );

    /// One radix-4 butterfly stage (f64) — fuses two consecutive radix-2 stages.
    ///
    /// Processes all radix-4 groups for a stage pair with the given `quarter_len`.
    /// `data` is interleaved `[re, im]` complex (`n * 2` doubles).
    /// `twiddles` layout: `[W1[0..q], W2[0..q], W3[0..q]]` where `q = quarter_len`.
    /// `W1[k]` applied to sub-block 2, `W2[k]` to sub-block 1, `W3[k]` to sub-block 3.
    /// Total twiddle doubles: `6 * quarter_len`.
    /// `forward`: 1 for forward FFT (−j rotation), 0 for inverse (+j rotation).
    ///
    /// # Safety
    ///
    /// - `data` must point to at least `n * 2` doubles.
    /// - `twiddles` must point to at least `quarter_len * 6` doubles.
    /// - `quarter_len` must be a power of 2 and `<= n / 4`.
    pub fn omnidsp_butterfly_stage4_f64(
        data: *mut f64,
        twiddles: *const f64,
        quarter_len: usize,
        n: usize,
        forward: i32,
    );

    /// One stage of a radix-2 Cooley-Tukey FFT butterfly (f64).
    ///
    /// Processes all butterfly groups for a stage with the given `half_len`.
    /// `data` is interleaved `[re, im]` complex (`n * 2` doubles).
    /// `twiddles` is interleaved `[re, im]` (`half_len * 2` doubles).
    /// `n` is the FFT length (number of complex elements).
    ///
    /// # Safety
    ///
    /// - `data` must point to at least `n * 2` doubles.
    /// - `twiddles` must point to at least `half_len * 2` doubles.
    /// - `half_len` must be a power of 2 and `<= n / 2`.
    pub fn omnidsp_butterfly_stage_f64(
        data: *mut f64,
        twiddles: *const f64,
        half_len: usize,
        n: usize,
    );
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "test values are exact small integers")]
mod tests {
    use super::*;

    const EPSILON_F32: f32 = 1e-6;
    const EPSILON_F64: f64 = 1e-12;

    // --- mul ---

    #[test]
    fn mul_f32_basic() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = vec![0.0_f32; 8];

        unsafe {
            omnidsp_mul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
        }

        let expected: Vec<f32> = vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0];
        assert_eq!(out, expected, "f32 element-wise multiply failed");
    }

    #[test]
    fn mul_f64_basic() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0_f64; 4];

        unsafe {
            omnidsp_mul_f64(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
        }

        let expected: Vec<f64> = vec![10.0, 40.0, 90.0, 160.0];
        assert_eq!(out, expected, "f64 element-wise multiply failed");
    }

    #[test]
    fn mul_f32_inplace() {
        let a: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        unsafe {
            omnidsp_mul_f32(a.as_ptr(), data.as_ptr(), data.as_mut_ptr(), a.len());
        }

        let expected: Vec<f32> = vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0];
        assert_eq!(data, expected, "f32 in-place multiply failed");
    }

    #[test]
    fn mul_f32_remainder() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let mut out = vec![0.0_f32; 5];

        unsafe {
            omnidsp_mul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
        }

        let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(out, expected, "f32 remainder handling failed");
    }

    // --- add ---

    #[test]
    fn add_f32_basic() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mut out = vec![0.0_f32; 8];

        unsafe {
            omnidsp_add_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
        }

        let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0];
        assert_eq!(out, expected, "f32 element-wise add failed");
    }

    #[test]
    fn add_f64_basic() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0_f64; 4];

        unsafe {
            omnidsp_add_f64(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
        }

        let expected: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5];
        assert_eq!(out, expected, "f64 element-wise add failed");
    }

    #[test]
    fn add_f32_inplace() {
        let other: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        unsafe {
            omnidsp_add_f32(other.as_ptr(), data.as_ptr(), data.as_mut_ptr(), data.len());
        }

        let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0, 55.0];
        assert_eq!(data, expected, "f32 in-place add failed");
    }

    // --- scale ---

    #[test]
    fn scale_f32_basic() {
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        unsafe {
            omnidsp_scale_f32(data.as_mut_ptr(), 3.0, data.len());
        }

        let expected: Vec<f32> = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];
        assert_eq!(data, expected, "f32 scale failed");
    }

    #[test]
    fn scale_f64_basic() {
        let mut data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        unsafe {
            omnidsp_scale_f64(data.as_mut_ptr(), 0.5, data.len());
        }

        let expected: Vec<f64> = vec![0.5, 1.0, 1.5, 2.0, 2.5];
        assert_eq!(data, expected, "f64 scale failed");
    }

    // --- dot ---

    #[test]
    fn dot_f32_basic() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0, 7.0];
        // 1*4 + 2*5 + 3*6 + 4*7 = 4 + 10 + 18 + 28 = 60

        let result = unsafe { omnidsp_dot_f32(a.as_ptr(), b.as_ptr(), a.len()) };

        assert!(
            (result - 60.0).abs() < EPSILON_F32,
            "f32 dot product: expected 60.0, got {result}"
        );
    }

    #[test]
    fn dot_f64_basic() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        let result = unsafe { omnidsp_dot_f64(a.as_ptr(), b.as_ptr(), a.len()) };

        assert!(
            (result - 32.0).abs() < EPSILON_F64,
            "f64 dot product: expected 32.0, got {result}"
        );
    }

    #[test]
    fn dot_f32_remainder() {
        // 7 elements — exercises the scalar tail.
        let a: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let b: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

        let result = unsafe { omnidsp_dot_f32(a.as_ptr(), b.as_ptr(), a.len()) };

        assert!(
            (result - 14.0).abs() < EPSILON_F32,
            "f32 dot remainder: expected 14.0, got {result}"
        );
    }

    // --- cmul ---

    #[test]
    fn cmul_f32_basic() {
        // (1+2i)*(5+6i) = (1*5 - 2*6) + (1*6 + 2*5)i = -7 + 16i
        // (3+4i)*(7+8i) = (3*7 - 4*8) + (3*8 + 4*7)i = -11 + 52i
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0_f32; 4];

        unsafe {
            omnidsp_cmul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 2);
        }

        assert!(
            (out[0] - (-7.0)).abs() < EPSILON_F32 && (out[1] - 16.0).abs() < EPSILON_F32,
            "cmul_f32[0]: expected -7+16i, got {}+{}i",
            out[0],
            out[1]
        );
        assert!(
            (out[2] - (-11.0)).abs() < EPSILON_F32 && (out[3] - 52.0).abs() < EPSILON_F32,
            "cmul_f32[1]: expected -11+52i, got {}+{}i",
            out[2],
            out[3]
        );
    }

    #[test]
    fn cmul_f64_basic() {
        // (2+3i)*(4+5i) = (8-15) + (10+12)i = -7 + 22i
        let a: Vec<f64> = vec![2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0];
        let mut out = vec![0.0_f64; 2];

        unsafe {
            omnidsp_cmul_f64(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 1);
        }

        assert!(
            (out[0] - (-7.0)).abs() < EPSILON_F64 && (out[1] - 22.0).abs() < EPSILON_F64,
            "cmul_f64: expected -7+22i, got {}+{}i",
            out[0],
            out[1]
        );
    }

    #[test]
    fn cmul_f32_inplace() {
        // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
        let other: Vec<f32> = vec![3.0, 4.0];
        let mut data: Vec<f32> = vec![1.0, 2.0];

        unsafe {
            omnidsp_cmul_f32(other.as_ptr(), data.as_ptr(), data.as_mut_ptr(), 1);
        }

        assert!(
            (data[0] - (-5.0)).abs() < EPSILON_F32 && (data[1] - 10.0).abs() < EPSILON_F32,
            "cmul_f32 in-place: expected -5+10i, got {}+{}i",
            data[0],
            data[1]
        );
    }

    // --- edge cases ---

    #[test]
    fn empty_slices() {
        unsafe {
            let mut out = [0.0_f32; 0];
            omnidsp_mul_f32([].as_ptr(), [].as_ptr(), out.as_mut_ptr(), 0);
            omnidsp_add_f32([].as_ptr(), [].as_ptr(), out.as_mut_ptr(), 0);
            omnidsp_scale_f32(out.as_mut_ptr(), 2.0, 0);

            let dot = omnidsp_dot_f32([].as_ptr(), [].as_ptr(), 0);
            assert!(
                dot.abs() < EPSILON_F32,
                "empty dot should be 0.0, got {dot}"
            );

            omnidsp_cmul_f32([].as_ptr(), [].as_ptr(), out.as_mut_ptr(), 0);
        }
    }

    #[test]
    fn single_element() {
        let a = [3.0_f32];
        let b = [7.0_f32];
        let mut out = [0.0_f32];

        unsafe {
            omnidsp_mul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 1);
        }
        assert_eq!(out[0], 21.0, "single element mul failed");

        let dot = unsafe { omnidsp_dot_f32(a.as_ptr(), b.as_ptr(), 1) };
        assert!(
            (dot - 21.0).abs() < EPSILON_F32,
            "single element dot: expected 21.0, got {dot}"
        );
    }

    // --- butterfly_stage4 ---

    /// 4-point FFT via single radix-4 stage (`quarter_len=1`, forward).
    /// Bit-reversed [1,2,3,4] → [(1,0),(3,0),(2,0),(4,0)].
    /// All twiddles are 1+0j (k=0 only).
    /// Expected: FFT([1,2,3,4]) = [(10,0),(-2,2),(-2,0),(-2,-2)].
    #[test]
    fn butterfly_stage4_f64_4point_forward() {
        let mut data: Vec<f64> = vec![1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0];
        // Twiddles: [W1[0], W2[0], W3[0]] = [(1,0), (1,0), (1,0)]
        let twiddles: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        unsafe {
            omnidsp_butterfly_stage4_f64(data.as_mut_ptr(), twiddles.as_ptr(), 1, 4, 1);
        }

        let expected: Vec<f64> = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < EPSILON_F64,
                "4pt fwd r4 mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    /// 4-point inverse FFT via single radix-4 stage.
    /// Input: FFT result [(10,0),(-2,2),(-2,0),(-2,-2)] bit-reversed.
    /// Bit-reversal of [0,1,2,3] → [0,2,1,3], so input order:
    /// [(10,0),(-2,0),(-2,2),(-2,-2)].
    /// Twiddles use +sign (inverse).
    /// Expected: IFFT unnormalized = [(4,0),(8,0),(12,0),(16,0)] = 4*[1,2,3,4].
    #[test]
    fn butterfly_stage4_f64_4point_inverse() {
        // Bit-reversed FFT output.
        let mut data: Vec<f64> = vec![10.0, 0.0, -2.0, 0.0, -2.0, 2.0, -2.0, -2.0];
        // Inverse twiddles: exp(+2πi·k/4) at k=0 → all 1+0j.
        let twiddles: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        unsafe {
            omnidsp_butterfly_stage4_f64(data.as_mut_ptr(), twiddles.as_ptr(), 1, 4, 0);
        }

        let expected: Vec<f64> = vec![4.0, 0.0, 8.0, 0.0, 12.0, 0.0, 16.0, 0.0];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < EPSILON_F64,
                "4pt inv r4 mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    /// 8-point FFT: radix-2 stage 0 then radix-4 stage pair (1,2).
    /// Verifies the two kernels compose correctly.
    #[test]
    fn butterfly_stage4_f32_8point_forward() {
        // Bit-reversed [1,2,3,4,5,6,7,8] as interleaved complex.
        let mut data: Vec<f32> = vec![
            1.0, 0.0, 5.0, 0.0, 3.0, 0.0, 7.0, 0.0, 2.0, 0.0, 6.0, 0.0, 4.0, 0.0, 8.0, 0.0,
        ];

        // Stage 0 (radix-2, half_len=1): twiddle = [1+0i].
        let tw_r2: Vec<f32> = vec![1.0, 0.0];
        unsafe {
            omnidsp_butterfly_stage_f32(data.as_mut_ptr(), tw_r2.as_ptr(), 1, 8);
        }

        // Stage pair (1,2) via radix-4, quarter_len=2.
        // W1[k] = exp(-2πi·k/8), W2[k] = exp(-2πi·2k/8), W3[k] = exp(-2πi·3k/8).
        let s2 = std::f32::consts::FRAC_1_SQRT_2;
        let twiddles_r4: Vec<f32> = vec![
            // W1: k=0 → (1,0), k=1 → (cos π/4, -sin π/4)
            1.0, 0.0, s2, -s2,
            // W2: k=0 → (1,0), k=1 → (cos π/2, -sin π/2) = (0,-1)
            1.0, 0.0, 0.0, -1.0,
            // W3: k=0 → (1,0), k=1 → (cos 3π/4, -sin 3π/4) = (-s2, -s2)
            1.0, 0.0, -s2, -s2,
        ];
        unsafe {
            omnidsp_butterfly_stage4_f32(data.as_mut_ptr(), twiddles_r4.as_ptr(), 2, 8, 1);
        }

        // Expected: FFT([1..8]) = [36, -4+9.66j, -4+4j, -4+1.66j*, -4, ...]
        let s2f = std::f32::consts::SQRT_2;
        let expected: Vec<f32> = vec![
            36.0,
            0.0,
            -4.0,
            4.0f32.mul_add(s2f, 4.0),
            -4.0,
            4.0,
            -4.0,
            4.0f32.mul_add(s2f, -4.0),
            -4.0,
            0.0,
            -4.0,
            4.0f32.mul_add(-s2f, 4.0),
            -4.0,
            -4.0,
            -4.0,
            4.0f32.mul_add(-s2f, -4.0),
        ];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "8pt fwd r4 mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    // --- butterfly_stage ---

    /// 4-point FFT stage 0 (`half_len=1`): twiddle is [1+0i].
    /// After bit-reversal of [1,2,3,4] → [1,3,2,4] (as complex: [(1,0),(3,0),(2,0),(4,0)]).
    /// Stage 0 butterflies: groups of 2, twiddle=1.
    ///   group 0: even=(1,0) odd=(3,0) → even'=(4,0) odd'=(-2,0)
    ///   group 1: even=(2,0) odd=(4,0) → even'=(6,0) odd'=(-2,0)
    #[test]
    fn butterfly_stage_f32_stage0() {
        // Bit-reversed [1,2,3,4] as interleaved complex.
        let mut data: Vec<f32> = vec![1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0];
        let twiddles: Vec<f32> = vec![1.0, 0.0]; // W_2^0 = 1+0i

        unsafe {
            omnidsp_butterfly_stage_f32(data.as_mut_ptr(), twiddles.as_ptr(), 1, 4);
        }

        let expected: Vec<f32> = vec![4.0, 0.0, -2.0, 0.0, 6.0, 0.0, -2.0, 0.0];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < EPSILON_F32,
                "stage0 mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    /// 4-point FFT stage 1 (`half_len=2`): twiddles are [1+0i, 0-1i] (forward).
    /// Input from stage 0: [(4,0),(-2,0),(6,0),(-2,0)].
    /// Stage 1: one group of 4, twiddles = [1+0i, 0-1i].
    ///   k=0: t = (1+0i)*(6,0) = (6,0), even'=(10,0), odd'=(-2,0)
    ///   k=1: t = (0-1i)*(-2,0) = (0,2), even'=(-2,2), odd'=(-2,-2)
    /// Result = FFT([1,2,3,4]) = [(10,0),(-2,2),(-2,0),(-2,-2)].
    #[test]
    fn butterfly_stage_f64_stage1() {
        // After stage 0.
        let mut data: Vec<f64> = vec![4.0, 0.0, -2.0, 0.0, 6.0, 0.0, -2.0, 0.0];
        // Forward twiddles for half_len=2: W_4^0 = 1+0i, W_4^1 = 0-1i.
        let twiddles: Vec<f64> = vec![1.0, 0.0, 0.0, -1.0];

        unsafe {
            omnidsp_butterfly_stage_f64(data.as_mut_ptr(), twiddles.as_ptr(), 2, 4);
        }

        let expected: Vec<f64> = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < EPSILON_F64,
                "stage1 mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }
}
