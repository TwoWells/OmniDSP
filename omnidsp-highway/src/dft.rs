// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! [`HwyDft`] — Highway SIMD DFT factory implementing mixed radix-4/radix-2
//! Cooley-Tukey FFT.
//!
//! Algorithm logic (loop structure, stage progression, bit-reversal) lives in
//! Rust. The hot inner loops (butterfly twiddle application, scaling) use
//! Highway SIMD kernels via `highway-sys`. Radix-4 stages fuse two radix-2
//! stages into a single pass, halving memory traffic. When the exponent is
//! odd, a single radix-2 stage handles the first pass.

use std::f64::consts::PI;

use num_complex::Complex;

use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::dft::{Dft, DftNorm, DftPlan, DftSpec};
use omnidsp_core::types::Direction;

/// Highway SIMD DFT factory.
///
/// Creates [`HwyDftPlan`] instances for power-of-2 lengths using a mixed
/// radix-4/radix-2 iterative Cooley-Tukey FFT. Non-power-of-2 lengths
/// return an error.
pub struct HwyDft;

/// Execution plan for a DFT operation using Highway SIMD butterflies.
///
/// Uses radix-4 stages where possible (two radix-2 stages fused into one
/// pass), with a single radix-2 stage at the start when the exponent is odd.
/// The plan is immutable — naturally `Send + Sync` with no interior mutability.
pub struct HwyDftPlan<T> {
    length: usize,
    direction: Direction,
    norm: DftNorm,
    /// Twiddle factors stored as interleaved `[re, im, ...]`.
    ///
    /// Layout: optional radix-2 twiddles for stage 0, then for each radix-4
    /// pair, three contiguous blocks `[W1[0..q], W2[0..q], W3[0..q]]`.
    twiddles: Vec<T>,
    /// True when the exponent is odd and stage 0 is a standalone radix-2 stage.
    has_initial_radix2: bool,
}

/// Precompute twiddle factors for the mixed radix-4/radix-2 FFT.
///
/// When the exponent is odd, stage 0 is a standalone radix-2 stage (its
/// twiddles come first). Remaining stages are paired into radix-4 groups.
/// Each radix-4 pair with `quarter_len = q` stores three twiddle blocks:
///   `[W1[0..q], W2[0..q], W3[0..q]]`
/// where `W1[k] = exp(sign·2πi·k/(4q))`, `W2[k] = W1[k]²`, `W3[k] = W1[k]³`.
///
/// Returns `(twiddles, has_initial_radix2)`.
fn compute_twiddles(length: usize, direction: Direction) -> (Vec<f64>, bool) {
    let sign = match direction {
        Direction::Forward => -1.0,
        Direction::Inverse => 1.0,
    };

    let num_stages = length.trailing_zeros() as usize;
    let has_initial_radix2 = num_stages % 2 == 1;

    // Total twiddle count is still N - 1 (same storage, different layout).
    let mut twiddles = Vec::with_capacity((length - 1) * 2);

    let start_stage = if has_initial_radix2 {
        // Stage 0: radix-2 twiddles for half_len = 1.
        let half_len = 1_usize;
        let full_len = half_len << 1;
        for k in 0..half_len {
            #[allow(
                clippy::cast_precision_loss,
                reason = "FFT lengths and indices are well within f64 mantissa range"
            )]
            let angle = sign * 2.0 * PI * (k as f64) / (full_len as f64);
            twiddles.push(angle.cos());
            twiddles.push(angle.sin());
        }
        1
    } else {
        0
    };

    // Radix-4 pairs: stages (s, s+1) with quarter_len = 2^s.
    let mut s = start_stage;
    while s + 1 < num_stages {
        let quarter_len = 1_usize << s;
        #[allow(
            clippy::cast_precision_loss,
            reason = "FFT lengths and indices are well within f64 mantissa range"
        )]
        let full_len_f = (quarter_len * 4) as f64;

        // W1: exp(sign·2πi·k / full_len)
        for k in 0..quarter_len {
            #[allow(
                clippy::cast_precision_loss,
                reason = "FFT lengths and indices are well within f64 mantissa range"
            )]
            let angle = sign * 2.0 * PI * (k as f64) / full_len_f;
            twiddles.push(angle.cos());
            twiddles.push(angle.sin());
        }
        // W2: exp(sign·2πi·2k / full_len)
        for k in 0..quarter_len {
            #[allow(
                clippy::cast_precision_loss,
                reason = "FFT lengths and indices are well within f64 mantissa range"
            )]
            let angle = sign * 2.0 * PI * ((2 * k) as f64) / full_len_f;
            twiddles.push(angle.cos());
            twiddles.push(angle.sin());
        }
        // W3: exp(sign·2πi·3k / full_len)
        for k in 0..quarter_len {
            #[allow(
                clippy::cast_precision_loss,
                reason = "FFT lengths and indices are well within f64 mantissa range"
            )]
            let angle = sign * 2.0 * PI * ((3 * k) as f64) / full_len_f;
            twiddles.push(angle.cos());
            twiddles.push(angle.sin());
        }

        s += 2;
    }

    (twiddles, has_initial_radix2)
}

/// Bit-reversal permutation in-place on a buffer of complex values stored
/// as interleaved `[re, im]` pairs.
fn bit_reverse_permute<T: Copy>(buf: &mut [T], log2n: u32) {
    let n = 1_usize << log2n;
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log2n);
        if i < j {
            // Swap complex pair (2 elements each).
            let ii = i * 2;
            let jj = j * 2;
            buf.swap(ii, jj);
            buf.swap(ii + 1, jj + 1);
        }
    }
}

macro_rules! impl_hwy_dft {
    ($t:ty, $butterfly_fn:ident, $butterfly4_fn:ident, $scale_fn:ident) => {
        impl DftPlan<$t> for HwyDftPlan<$t> {
            fn process(&self, input: &[Complex<$t>], output: &mut [Complex<$t>]) -> Result<()> {
                if input.len() != self.length {
                    return Err(Error::BufferMismatch {
                        expected: self.length,
                        actual: input.len(),
                    });
                }
                if output.len() != self.length {
                    return Err(Error::BufferMismatch {
                        expected: self.length,
                        actual: output.len(),
                    });
                }

                // Copy input to output, then work in-place.
                output.copy_from_slice(input);

                if self.length <= 1 {
                    return self.apply_norm(output);
                }

                // Reinterpret output as interleaved floats for in-place work.
                let log2n = self.length.trailing_zeros();

                // SAFETY: Complex<T> is repr(C) with layout [T; 2]. Reinterpreting
                // &mut [Complex<T>] as &mut [T] with double the length is sound.
                let buf: &mut [$t] = unsafe {
                    std::slice::from_raw_parts_mut(
                        output.as_mut_ptr().cast::<$t>(),
                        self.length * 2,
                    )
                };

                // Step 1: Bit-reversal permutation.
                bit_reverse_permute(buf, log2n);

                // Step 2: Butterfly stages.
                let num_stages = log2n as usize;
                // Optional leading radix-2 stage (when exponent is odd).
                let (mut twiddle_offset, mut stage) = if self.has_initial_radix2 {
                    let half_len = 1_usize;

                    // SAFETY: `buf` has `self.length * 2` elements, twiddle
                    // slice has `half_len * 2` elements.
                    unsafe {
                        highway_sys::$butterfly_fn(
                            buf.as_mut_ptr(),
                            self.twiddles.as_ptr(),
                            half_len,
                            self.length,
                        );
                    }
                    (half_len * 2, 1)
                } else {
                    (0, 0)
                };

                // Radix-4 stages (fused pairs).
                let forward_flag: i32 = match self.direction {
                    Direction::Forward => 1,
                    Direction::Inverse => 0,
                };

                while stage + 1 < num_stages {
                    let quarter_len = 1_usize << stage;
                    let tw_len = quarter_len * 6;

                    // SAFETY: `buf` has `self.length * 2` elements, twiddle
                    // slice has `quarter_len * 6` elements, `quarter_len` is
                    // a power of 2 and `<= self.length / 4`.
                    unsafe {
                        highway_sys::$butterfly4_fn(
                            buf.as_mut_ptr(),
                            self.twiddles[twiddle_offset..].as_ptr(),
                            quarter_len,
                            self.length,
                            forward_flag,
                        );
                    }
                    twiddle_offset += tw_len;
                    stage += 2;
                }

                self.apply_norm(output)
            }
        }

        impl HwyDftPlan<$t> {
            /// Apply normalization scaling to output buffer.
            fn apply_norm(&self, output: &mut [Complex<$t>]) -> Result<()> {
                let scale = self.compute_scale()?;
                if let Some(s) = scale {
                    // SAFETY: Complex<T> is repr(C) with layout [T; 2]. Reinterpreting
                    // as &mut [T] with double length is sound. Highway scale operates
                    // on the raw float buffer.
                    let buf: &mut [$t] = unsafe {
                        std::slice::from_raw_parts_mut(
                            output.as_mut_ptr().cast::<$t>(),
                            output.len() * 2,
                        )
                    };
                    // SAFETY: pointer is valid for `buf.len()` elements.
                    unsafe {
                        highway_sys::$scale_fn(buf.as_mut_ptr(), s, buf.len());
                    }
                }
                Ok(())
            }

            /// Compute the scaling factor for the configured normalization.
            fn compute_scale(&self) -> Result<Option<$t>> {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "DFT lengths are well within f64 mantissa range"
                )]
                let n = self.length as f64;

                let factor: f64 = match self.norm {
                    DftNorm::None => return Ok(None),
                    DftNorm::Inverse => match self.direction {
                        Direction::Inverse => 1.0 / n,
                        Direction::Forward => return Ok(None),
                    },
                    DftNorm::Ortho => 1.0 / n.sqrt(),
                };

                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "normalization factor is always in representable range"
                )]
                Ok(Some(factor as $t))
            }
        }

        impl Dft<$t> for HwyDft {
            type Plan = HwyDftPlan<$t>;

            fn create_plan(&self, spec: &DftSpec) -> Result<Self::Plan> {
                if spec.length == 0 {
                    return Err(Error::InvalidSpec("DFT length must be non-zero".to_owned()));
                }
                if !spec.length.is_power_of_two() {
                    return Err(Error::InvalidSpec(format!(
                        "HwyDft requires power-of-2 length, got {}",
                        spec.length,
                    )));
                }

                let (twiddles_f64, has_initial_radix2) =
                    compute_twiddles(spec.length, spec.direction);
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "twiddle factors (sin/cos) are always in [-1, 1]"
                )]
                let twiddles: Vec<$t> = twiddles_f64.iter().map(|&v| v as $t).collect();

                Ok(HwyDftPlan {
                    length: spec.length,
                    direction: spec.direction,
                    norm: spec.norm,
                    twiddles,
                    has_initial_radix2,
                })
            }
        }
    };
}

impl_hwy_dft!(
    f32,
    omnidsp_butterfly_stage_f32,
    omnidsp_butterfly_stage4_f32,
    omnidsp_scale_f32
);
impl_hwy_dft!(
    f64,
    omnidsp_butterfly_stage_f64,
    omnidsp_butterfly_stage4_f64,
    omnidsp_scale_f64
);

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;

    const EPSILON_F32: f32 = 1e-5;
    const EPSILON_F64: f64 = 1e-12;

    fn fwd(length: usize) -> DftSpec {
        DftSpec::new(length, Direction::Forward, DftNorm::Inverse)
    }

    fn inv(length: usize) -> DftSpec {
        DftSpec::new(length, Direction::Inverse, DftNorm::Inverse)
    }

    fn assert_complex_slice_eq_f32(a: &[Complex<f32>], b: &[Complex<f32>], eps: f32) {
        assert_eq!(a.len(), b.len(), "slice lengths differ");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x.re - y.re).abs() < eps && (x.im - y.im).abs() < eps,
                "mismatch at index {i}: got ({}, {}), expected ({}, {})",
                x.re,
                x.im,
                y.re,
                y.im,
            );
        }
    }

    fn assert_complex_slice_eq_f64(a: &[Complex<f64>], b: &[Complex<f64>], eps: f64) {
        assert_eq!(a.len(), b.len(), "slice lengths differ");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x.re - y.re).abs() < eps && (x.im - y.im).abs() < eps,
                "mismatch at index {i}: got ({}, {}), expected ({}, {})",
                x.re,
                x.im,
                y.re,
                y.im,
            );
        }
    }

    // --- Validation ---

    #[test]
    fn zero_length_returns_error() {
        let dft = HwyDft;
        let result =
            Dft::<f32>::create_plan(&dft, &DftSpec::new(0, Direction::Forward, DftNorm::None));
        assert!(result.is_err(), "zero length should return error");
    }

    #[test]
    fn non_power_of_two_returns_error() {
        let dft = HwyDft;
        let result =
            Dft::<f32>::create_plan(&dft, &DftSpec::new(3, Direction::Forward, DftNorm::None));
        assert!(result.is_err(), "non-power-of-2 should return error");

        let result =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(7, Direction::Forward, DftNorm::None));
        assert!(result.is_err(), "non-power-of-2 should return error");
    }

    // --- DC signal ---

    #[test]
    fn dc_signal_f32() {
        let dft = HwyDft;
        let plan = Dft::<f32>::create_plan(&dft, &fwd(4)).expect("plan creation should succeed");
        let input = [Complex::new(1.0_f32, 0.0); 4];
        let mut output = [Complex::default(); 4];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        let expected = [
            Complex::new(4.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        assert_complex_slice_eq_f32(&output, &expected, EPSILON_F32);
    }

    #[test]
    fn dc_signal_f64() {
        let dft = HwyDft;
        let plan = Dft::<f64>::create_plan(&dft, &fwd(4)).expect("plan creation should succeed");
        let input = [Complex::new(1.0_f64, 0.0); 4];
        let mut output = [Complex::default(); 4];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        let expected = [
            Complex::new(4.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        assert_complex_slice_eq_f64(&output, &expected, EPSILON_F64);
    }

    // --- Known 4-point ---

    /// 4-point FFT of [1, 2, 3, 4] — verified against numpy.fft.fft.
    #[test]
    fn known_4point_f64() {
        let dft = HwyDft;
        let plan =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(4, Direction::Forward, DftNorm::None))
                .expect("plan creation should succeed");

        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output = [Complex::default(); 4];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        // numpy.fft.fft([1,2,3,4]) = [10+0j, -2+2j, -2+0j, -2-2j]
        let expected = [
            Complex::new(10.0, 0.0),
            Complex::new(-2.0, 2.0),
            Complex::new(-2.0, 0.0),
            Complex::new(-2.0, -2.0),
        ];
        assert_complex_slice_eq_f64(&output, &expected, EPSILON_F64);
    }

    // --- Known 8-point ---

    /// 8-point FFT of [1,2,3,4,5,6,7,8] — verified against numpy.fft.fft.
    #[test]
    fn known_8point_f64() {
        let dft = HwyDft;
        let plan =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(8, Direction::Forward, DftNorm::None))
                .expect("plan creation should succeed");

        let input: [Complex<f64>; 8] = std::array::from_fn(|i| {
            #[allow(clippy::cast_precision_loss, reason = "test constant")]
            Complex::new((i + 1) as f64, 0.0)
        });
        let mut output = [Complex::default(); 8];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        // numpy.fft.fft([1,2,3,4,5,6,7,8])
        let sqrt2 = std::f64::consts::SQRT_2;
        let expected = [
            Complex::new(36.0, 0.0),
            Complex::new(-4.0, 4.0f64.mul_add(sqrt2, 4.0)),
            Complex::new(-4.0, 4.0),
            Complex::new(-4.0, 4.0f64.mul_add(sqrt2, -4.0)),
            Complex::new(-4.0, 0.0),
            Complex::new(-4.0, 4.0f64.mul_add(-sqrt2, 4.0)),
            Complex::new(-4.0, -4.0),
            Complex::new(-4.0, 4.0f64.mul_add(-sqrt2, -4.0)),
        ];
        assert_complex_slice_eq_f64(&output, &expected, EPSILON_F64);
    }

    // --- Round-trip ---

    #[test]
    fn round_trip_f32() {
        let dft = HwyDft;
        let fwd_plan = Dft::<f32>::create_plan(&dft, &fwd(4)).expect("forward plan should succeed");
        let inv_plan = Dft::<f32>::create_plan(&dft, &inv(4)).expect("inverse plan should succeed");

        let input = [
            Complex::new(1.0_f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");
        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f32(&recovered, &input, EPSILON_F32);
    }

    #[test]
    fn round_trip_f64() {
        let dft = HwyDft;
        let fwd_plan = Dft::<f64>::create_plan(&dft, &fwd(8)).expect("forward plan should succeed");
        let inv_plan = Dft::<f64>::create_plan(&dft, &inv(8)).expect("inverse plan should succeed");

        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), -f64::from(i)))
            .collect();
        let mut freq = vec![Complex::default(); 8];
        let mut recovered = vec![Complex::default(); 8];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");
        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f64(&recovered, &input, EPSILON_F64);
    }

    // --- Buffer mismatch ---

    #[test]
    fn buffer_length_mismatch() {
        let dft = HwyDft;
        let plan = Dft::<f32>::create_plan(&dft, &fwd(4)).expect("plan creation should succeed");

        let input = [Complex::new(1.0_f32, 0.0); 3];
        let mut output = [Complex::default(); 4];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "mismatched input length should return error"
        );

        let input = [Complex::new(1.0_f32, 0.0); 4];
        let mut output = [Complex::default(); 3];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "mismatched output length should return error"
        );
    }

    // --- Length 1 ---

    #[test]
    fn length_one() {
        let dft = HwyDft;
        let plan =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(1, Direction::Forward, DftNorm::None))
                .expect("plan creation should succeed");

        let input = [Complex::new(42.0, 7.0)];
        let mut output = [Complex::default()];
        plan.process(&input, &mut output)
            .expect("process should succeed");

        assert_complex_slice_eq_f64(&output, &input, EPSILON_F64);
    }

    // --- Normalization ---

    /// Ortho normalization: round-trip should be identity, and forward
    /// output should be scaled by 1/sqrt(N) compared to unnormalized.
    #[test]
    fn ortho_round_trip() {
        let dft = HwyDft;
        let fwd_plan =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(4, Direction::Forward, DftNorm::Ortho))
                .expect("forward ortho plan");
        let inv_plan =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(4, Direction::Inverse, DftNorm::Ortho))
                .expect("inverse ortho plan");

        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");

        // DC bin should be sum/sqrt(N) = 10/2 = 5
        assert!(
            (freq[0].re - 5.0).abs() < EPSILON_F64,
            "ortho DC should be 10/sqrt(4) = 5.0, got {}",
            freq[0].re
        );

        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f64(&recovered, &input, EPSILON_F64);
    }

    /// Unnormalized: round-trip scales by N.
    #[test]
    fn unnormalized_round_trip_scales_by_n() {
        let dft = HwyDft;
        let n = 4;
        let fwd_plan =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(n, Direction::Forward, DftNorm::None))
                .expect("forward plan");
        let inv_plan =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(n, Direction::Inverse, DftNorm::None))
                .expect("inverse plan");

        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");
        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        // Round-trip should be input * N
        #[allow(clippy::cast_precision_loss, reason = "test constant")]
        let nf = n as f64;
        let expected: Vec<Complex<f64>> = input
            .iter()
            .map(|c| Complex::new(c.re * nf, c.im * nf))
            .collect();
        assert_complex_slice_eq_f64(&recovered, &expected, EPSILON_F64);
    }

    // --- Larger size ---

    #[test]
    fn round_trip_f64_1024() {
        let dft = HwyDft;
        let n = 1024;
        let fwd_plan = Dft::<f64>::create_plan(&dft, &fwd(n)).expect("forward plan should succeed");
        let inv_plan = Dft::<f64>::create_plan(&dft, &inv(n)).expect("inverse plan should succeed");

        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss, reason = "test value")]
                let v = (i as f64) * 0.01;
                Complex::new(v.sin(), v.cos())
            })
            .collect();
        let mut freq = vec![Complex::default(); n];
        let mut recovered = vec![Complex::default(); n];

        fwd_plan
            .process(&input, &mut freq)
            .expect("forward should succeed");
        inv_plan
            .process(&freq, &mut recovered)
            .expect("inverse should succeed");

        assert_complex_slice_eq_f64(&recovered, &input, 1e-9);
    }
}
