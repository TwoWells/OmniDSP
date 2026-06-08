// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FIR filter module — direct and overlap-save streaming filter.
//!
//! [`OmniFir`] implements the [`Fir`] trait generically over any [`DftC2c`]
//! and [`VecOps`] implementation.  Supports both time-domain (direct MAC
//! loop) and frequency-domain (overlap-save) execution, selected via
//! [`FirStrategy`].
//!
//! The [`recommend_strategy`] function provides an operation-count heuristic
//! for the `Auto` case.
//!
//! Plans are **mutable** — they hold a delay line that persists across calls
//! so successive `process` calls form a continuous stream.

use std::ops::{AddAssign, MulAssign};

use num_complex::Complex;
use num_traits::Float;

use crate::error::{Error, Result};
use crate::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec, DftNorm};
use crate::traits::fir::{Fir, FirPlan, FirSpec, FirStrategy};
use crate::traits::vecops::VecOps;
use crate::types::Direction;

// ─── Public heuristic ─────────────────────────────────────────────────

/// Recommend a FIR strategy based on operation counts.
///
/// Compares the work of direct filtering (`num_taps` multiply-accumulates
/// per output sample) against overlap-save (`~2 * B * log2(B) + B` ops
/// per block of `B - (num_taps - 1)` output samples, amortized per sample).
/// Returns whichever is cheaper.
///
/// This is a generic heuristic that ignores constant factors (cache
/// effects, SIMD width, FFT implementation quality).  Vendor-specific
/// `Fir` implementations may use a different decision boundary.
#[must_use]
pub fn recommend_strategy(num_taps: usize) -> FirStrategy {
    if num_taps <= 1 {
        return FirStrategy::Direct;
    }

    // Direct: num_taps MACs per output sample.
    let direct_ops = num_taps as u64;

    // Overlap-save: block size B = next_pow2(2 * (num_taps - 1)).
    // Valid output per block: B - (num_taps - 1).
    // Work per block: 2 FFTs (fwd input + inv result) + 1 complex multiply
    //   = 2*B*log2(B) + B.
    // (The coefficient FFT is precomputed at plan creation time.)
    let overlap = (num_taps - 1) as u64;
    let block_size = (2 * overlap).next_power_of_two();
    let valid_per_block = block_size - overlap;

    let log2_b = u64::BITS - block_size.leading_zeros();
    let fft_ops_per_block = 2 * block_size * u64::from(log2_b) + block_size;

    // Amortized ops per output sample.
    let ols_ops_per_sample = fft_ops_per_block / valid_per_block;

    if direct_ops <= ols_ops_per_sample {
        FirStrategy::Direct
    } else {
        FirStrategy::OverlapSave
    }
}

// ─── Public types ──────────────────────────────────────────────────────

/// Generic FIR filter factory backed by [`DftC2c`] and [`VecOps`].
///
/// Creates [`OmniFirPlan`]s for specific filter specifications.  The factory
/// owns the DFT factory and `VecOps` instance; plans own their sub-plans.
#[derive(Debug, Clone)]
pub struct OmniFir<D, V> {
    dft: D,
    vecops: V,
}

impl<D, V> OmniFir<D, V> {
    /// Create a new FIR filter factory.
    #[must_use]
    pub const fn new(dft: D, vecops: V) -> Self {
        Self { dft, vecops }
    }
}

/// Execution plan for a streaming FIR filter.
///
/// Created by [`OmniFir::create_plan`](Fir::create_plan).  Mutable — holds
/// a delay line that persists across calls so successive `process` calls
/// form a continuous stream.  Call [`reset`](FirPlan::reset) to clear the
/// delay line without recreating the plan.
///
/// **Memory:** the direct path allocates `2 × num_taps` for the doubled
/// delay buffer.  The overlap-save path allocates 3 × `block_size` complex
/// values for FFT scratch plus `num_taps − 1` for the overlap buffer.
pub struct OmniFirPlan<T, P, V> {
    inner: PlanInner<T, P, V>,
}

impl<T, P, V> std::fmt::Debug for OmniFirPlan<T, P, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let method = match &self.inner {
            PlanInner::Direct { .. } => "Direct",
            PlanInner::OverlapSave { .. } => "OverlapSave",
        };
        f.debug_struct("OmniFirPlan")
            .field("method", &method)
            .finish_non_exhaustive()
    }
}

// ─── Plan internals ───────────────────────────────────────────────────

#[allow(
    clippy::large_enum_variant,
    reason = "only one variant is constructed per plan"
)]
enum PlanInner<T, P, V> {
    Direct { state: DirectState<T, V> },
    OverlapSave { state: OverlapSaveState<T, P, V> },
}

/// State for the direct (time-domain) path.
struct DirectState<T, V> {
    /// Filter coefficients (reversed for correlation-style MAC).
    coeffs: Vec<T>,
    /// Doubled delay buffer, length = `2 * num_taps`.
    ///
    /// Each sample is written at both `write_pos` and `write_pos + num_taps`
    /// so that `delay[write_pos..write_pos + num_taps]` is always a
    /// contiguous oldest-first view — no two-segment dot product needed.
    delay: Vec<T>,
    /// Write pointer into the delay buffer.
    write_pos: usize,
    /// `VecOps` handle for dot products.
    vecops: V,
}

/// State for the overlap-save (frequency-domain) path.
struct OverlapSaveState<T, P, V> {
    /// Number of filter taps.
    num_taps: usize,
    /// Valid output samples per block: `block_size - (num_taps - 1)`.
    valid_per_block: usize,
    /// Pre-computed frequency-domain coefficients `H[k]`.
    freq_coeffs: Vec<Complex<T>>,
    /// Overlap buffer: last `num_taps - 1` input samples from previous call.
    overlap: Vec<T>,
    /// Forward DFT plan.
    fwd: P,
    /// Inverse DFT plan.
    inv: P,
    /// `VecOps` handle.
    vecops: V,
    /// Scratch: FFT input (`block_size` complex samples).
    buf_time: Vec<Complex<T>>,
    /// Scratch: FFT output / complex multiply workspace.
    buf_freq: Vec<Complex<T>>,
    /// Scratch: IFFT output.
    buf_ifft: Vec<Complex<T>>,
}

/// Zero-pad a real slice into a pre-allocated complex buffer via `VecOps`.
///
/// First `real.len()` elements are converted via [`VecOps::real_to_complex`];
/// the remainder is zeroed.
fn pad_real_to_complex<T: Float + AddAssign + MulAssign, V: VecOps<T>>(
    vecops: &V,
    real: &[T],
    buf: &mut [Complex<T>],
) -> Result<()> {
    let n = real.len();
    vecops.real_to_complex(real, &mut buf[..n])?;
    for c in &mut buf[n..] {
        *c = Complex::new(T::zero(), T::zero());
    }
    Ok(())
}

// ─── Trait implementations ─────────────────────────────────────────────

impl<T, P, V> FirPlan<T> for OmniFirPlan<T, P, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    P: DftC2cPlan<T>,
    V: VecOps<T>,
{
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::BufferMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }

        match &mut self.inner {
            PlanInner::Direct { state } => Self::process_direct(state, input, output),
            PlanInner::OverlapSave { state } => Self::process_overlap_save(state, input, output),
        }
    }

    fn reset(&mut self) {
        match &mut self.inner {
            PlanInner::Direct { state } => {
                for x in &mut state.delay {
                    *x = T::zero();
                }
                state.write_pos = 0;
            }
            PlanInner::OverlapSave { state } => {
                for x in &mut state.overlap {
                    *x = T::zero();
                }
            }
        }
    }
}

impl<T, P, V> OmniFirPlan<T, P, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    P: DftC2cPlan<T>,
    V: VecOps<T>,
{
    /// Direct (time-domain) FIR: doubled buffer + single dot product.
    fn process_direct(state: &mut DirectState<T, V>, input: &[T], output: &mut [T]) -> Result<()> {
        let num_taps = state.coeffs.len();

        for (out, &sample) in output.iter_mut().zip(input) {
            // Write to both halves of the doubled buffer.
            state.delay[state.write_pos] = sample;
            state.delay[state.write_pos + num_taps] = sample;
            state.write_pos = (state.write_pos + 1) % num_taps;

            // Coefficients are stored reversed, so coeffs[0] corresponds to
            // the oldest sample and coeffs[N-1] to the newest.  The doubled
            // buffer guarantees a contiguous oldest-first window.
            let pos = state.write_pos;
            *out = state
                .vecops
                .dot(&state.coeffs, &state.delay[pos..pos + num_taps])?;
        }

        Ok(())
    }

    /// Overlap-save (frequency-domain) FIR.
    fn process_overlap_save(
        state: &mut OverlapSaveState<T, P, V>,
        input: &[T],
        output: &mut [T],
    ) -> Result<()> {
        let overlap_len = state.num_taps - 1;
        let mut input_pos = 0;
        let mut output_pos = 0;

        while input_pos < input.len() {
            let remaining_input = input.len() - input_pos;
            let new_samples = remaining_input.min(state.valid_per_block);

            // Build the block: overlap prefix + new input + zero-pad.
            state
                .vecops
                .real_to_complex(&state.overlap, &mut state.buf_time[..overlap_len])?;
            state.vecops.real_to_complex(
                &input[input_pos..input_pos + new_samples],
                &mut state.buf_time[overlap_len..overlap_len + new_samples],
            )?;
            for c in &mut state.buf_time[overlap_len + new_samples..] {
                *c = Complex::new(T::zero(), T::zero());
            }

            // Forward FFT.
            state.fwd.process(&state.buf_time, &mut state.buf_freq)?;

            // Complex multiply with pre-computed H[k].
            state
                .vecops
                .cmul_inplace(&mut state.buf_freq, &state.freq_coeffs)?;

            // Inverse FFT.
            state.inv.process(&state.buf_freq, &mut state.buf_ifft)?;

            // Extract valid output: skip first overlap_len samples.
            let valid_start = overlap_len;
            state.vecops.extract_real(
                &state.buf_ifft[valid_start..valid_start + new_samples],
                &mut output[output_pos..output_pos + new_samples],
            )?;

            // Update overlap buffer with the last overlap_len input samples.
            if new_samples >= overlap_len {
                state.overlap.copy_from_slice(
                    &input[input_pos + new_samples - overlap_len..input_pos + new_samples],
                );
            } else {
                let keep = overlap_len - new_samples;
                state.overlap.copy_within(new_samples..overlap_len, 0);
                state.overlap[keep..].copy_from_slice(&input[input_pos..input_pos + new_samples]);
            }

            input_pos += new_samples;
            output_pos += new_samples;
        }

        Ok(())
    }
}

impl<T, D, V> Fir<T> for OmniFir<D, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    D: DftC2c<T>,
    V: VecOps<T>,
{
    type Plan = OmniFirPlan<T, D::Plan, V>;

    fn create_plan(&self, spec: &FirSpec<T>) -> Result<Self::Plan> {
        if spec.coefficients.is_empty() {
            return Err(Error::InvalidSpec(
                "FIR coefficients must not be empty".to_owned(),
            ));
        }

        let num_taps = spec.coefficients.len();

        let strategy = match spec.strategy {
            FirStrategy::Auto => recommend_strategy(num_taps),
            other => other,
        };

        let inner = match strategy {
            FirStrategy::Direct | FirStrategy::Auto => {
                let mut coeffs = spec.coefficients.clone();
                coeffs.reverse();

                PlanInner::Direct {
                    state: DirectState {
                        delay: vec![T::zero(); 2 * num_taps],
                        write_pos: 0,
                        coeffs,
                        vecops: self.vecops.clone(),
                    },
                }
            }
            FirStrategy::OverlapSave => {
                let overlap_len = num_taps - 1;
                let block_size =
                    (2 * overlap_len)
                        .checked_next_power_of_two()
                        .ok_or_else(|| {
                            Error::InvalidSpec("FIR overlap-save block size overflow".to_owned())
                        })?;
                let valid_per_block = block_size - overlap_len;

                let fwd_spec = DftC2cSpec::new(block_size, Direction::Forward, DftNorm::Inverse)?;
                let inv_spec = DftC2cSpec::new(block_size, Direction::Inverse, DftNorm::Inverse)?;
                let fwd = self.dft.create_plan(&fwd_spec)?;
                let inv = self.dft.create_plan(&inv_spec)?;

                // Pre-compute frequency-domain coefficients.
                let zero = Complex::new(T::zero(), T::zero());
                let mut buf_time = vec![zero; block_size];
                let mut freq_coeffs = vec![zero; block_size];

                pad_real_to_complex(&self.vecops, &spec.coefficients, &mut buf_time)?;
                fwd.process(&buf_time, &mut freq_coeffs)?;

                PlanInner::OverlapSave {
                    state: OverlapSaveState {
                        num_taps,
                        valid_per_block,
                        freq_coeffs,
                        overlap: vec![T::zero(); overlap_len],
                        fwd,
                        inv,
                        vecops: self.vecops.clone(),
                        buf_time,
                        buf_freq: vec![zero; block_size],
                        buf_ifft: vec![zero; block_size],
                    },
                }
            }
        };

        Ok(OmniFirPlan { inner })
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;
    use crate::test_utils::{TestDftC2c, TestVecOps};

    const EPSILON: f64 = 1e-8;

    fn make_factory() -> OmniFir<TestDftC2c, TestVecOps> {
        OmniFir::new(TestDftC2c, TestVecOps)
    }

    fn assert_approx_eq(actual: &[f64], expected: &[f64], eps: f64, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: slice lengths differ ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < eps,
                "{label}: mismatch at index {i}: got {a}, expected {e} (diff={})",
                (a - e).abs()
            );
        }
    }

    // ── Heuristic tests ───────────────────────────────────────────────

    #[test]
    fn recommend_short_prefers_direct() {
        assert_eq!(
            recommend_strategy(4),
            FirStrategy::Direct,
            "short filter should prefer direct"
        );
    }

    #[test]
    fn recommend_long_prefers_overlap_save() {
        assert_eq!(
            recommend_strategy(1024),
            FirStrategy::OverlapSave,
            "long filter should prefer overlap-save"
        );
    }

    #[test]
    fn recommend_single_tap_is_direct() {
        assert_eq!(
            recommend_strategy(1),
            FirStrategy::Direct,
            "single tap should be direct"
        );
    }

    // ── Impulse response ──────────────────────────────────────────────

    fn test_impulse_response(strategy: FirStrategy) {
        let factory = make_factory();
        let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spec = FirSpec::new(coeffs.clone()).with_strategy(strategy);
        let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

        let mut input = vec![0.0; coeffs.len() + 4];
        input[0] = 1.0;
        let mut output = vec![0.0; input.len()];

        plan.process(&input, &mut output).expect("process");

        assert_approx_eq(
            &output[..coeffs.len()],
            &coeffs,
            EPSILON,
            &format!("impulse response ({strategy:?})"),
        );
        for (i, &v) in output[coeffs.len()..].iter().enumerate() {
            assert!(
                v.abs() < EPSILON,
                "tail should be zero at index {}: got {v} ({strategy:?})",
                coeffs.len() + i
            );
        }
    }

    #[test]
    fn impulse_response_direct() {
        test_impulse_response(FirStrategy::Direct);
    }

    #[test]
    fn impulse_response_overlap_save() {
        test_impulse_response(FirStrategy::OverlapSave);
    }

    // ── DC gain ───────────────────────────────────────────────────────

    fn test_dc_gain(strategy: FirStrategy) {
        let factory = make_factory();
        let coeffs = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let expected_gain: f64 = coeffs.iter().sum();
        let spec = FirSpec::new(coeffs.clone()).with_strategy(strategy);
        let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

        let input_val = 3.0;
        let input = vec![input_val; 100];
        let mut output = vec![0.0; 100];

        plan.process(&input, &mut output).expect("process");

        let expected = expected_gain * input_val;
        for (i, &v) in output[coeffs.len()..].iter().enumerate() {
            assert!(
                (v - expected).abs() < EPSILON,
                "DC gain mismatch at index {}: got {v}, expected {expected} ({strategy:?})",
                coeffs.len() + i
            );
        }
    }

    #[test]
    fn dc_gain_direct() {
        test_dc_gain(FirStrategy::Direct);
    }

    #[test]
    fn dc_gain_overlap_save() {
        test_dc_gain(FirStrategy::OverlapSave);
    }

    // ── Streaming continuity ──────────────────────────────────────────

    fn test_streaming_continuity(strategy: FirStrategy) {
        let factory = make_factory();
        let coeffs = vec![1.0, -0.5, 0.25, -0.125, 0.0625];
        let spec = FirSpec::new(coeffs).with_strategy(strategy);

        let mut plan_ref = Fir::<f64>::create_plan(&factory, &spec).expect("ref plan");
        let input: Vec<f64> = (0..20).map(|i| (f64::from(i)) * 0.1).collect();
        let mut output_ref = vec![0.0; 20];
        plan_ref
            .process(&input, &mut output_ref)
            .expect("ref process");

        let mut plan_stream = Fir::<f64>::create_plan(&factory, &spec).expect("stream plan");
        let split = 7;
        let mut output_stream = vec![0.0; 20];
        plan_stream
            .process(&input[..split], &mut output_stream[..split])
            .expect("chunk 1");
        plan_stream
            .process(&input[split..], &mut output_stream[split..])
            .expect("chunk 2");

        assert_approx_eq(
            &output_stream,
            &output_ref,
            EPSILON,
            &format!("streaming continuity ({strategy:?})"),
        );
    }

    #[test]
    fn streaming_continuity_direct() {
        test_streaming_continuity(FirStrategy::Direct);
    }

    #[test]
    fn streaming_continuity_overlap_save() {
        test_streaming_continuity(FirStrategy::OverlapSave);
    }

    // ── Reset ─────────────────────────────────────────────────────────

    fn test_reset(strategy: FirStrategy) {
        let factory = make_factory();
        let coeffs = vec![1.0, 0.5, 0.25];
        let spec = FirSpec::new(coeffs).with_strategy(strategy);
        let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output1 = vec![0.0; 5];
        let mut output2 = vec![0.0; 5];

        plan.process(&input, &mut output1).expect("first run");
        plan.reset();
        plan.process(&input, &mut output2).expect("second run");

        assert_approx_eq(
            &output2,
            &output1,
            EPSILON,
            &format!("reset ({strategy:?})"),
        );
    }

    #[test]
    fn reset_direct() {
        test_reset(FirStrategy::Direct);
    }

    #[test]
    fn reset_overlap_save() {
        test_reset(FirStrategy::OverlapSave);
    }

    // ── Direct and overlap-save agree ─────────────────────────────────

    #[test]
    fn direct_and_overlap_save_agree() {
        let factory = make_factory();
        let coeffs = vec![0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1];

        let spec_d = FirSpec::new(coeffs.clone()).with_strategy(FirStrategy::Direct);
        let spec_o = FirSpec::new(coeffs).with_strategy(FirStrategy::OverlapSave);

        let mut plan_d = Fir::<f64>::create_plan(&factory, &spec_d).expect("direct plan");
        let mut plan_o = Fir::<f64>::create_plan(&factory, &spec_o).expect("ols plan");

        let input: Vec<f64> = (0..50).map(|i| ((f64::from(i)) * 0.3).sin()).collect();
        let mut out_d = vec![0.0; 50];
        let mut out_o = vec![0.0; 50];

        plan_d.process(&input, &mut out_d).expect("direct process");
        plan_o.process(&input, &mut out_o).expect("ols process");

        assert_approx_eq(&out_d, &out_o, EPSILON, "direct vs overlap-save");
    }

    // ── Plan reuse ────────────────────────────────────────────────────

    fn test_plan_reuse(strategy: FirStrategy) {
        let factory = make_factory();
        let coeffs = vec![1.0, 0.0, 0.0];
        let spec = FirSpec::new(coeffs).with_strategy(strategy);
        let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

        let input1 = vec![1.0, 2.0, 3.0];
        let mut output1 = vec![0.0; 3];
        plan.process(&input1, &mut output1).expect("call 1");

        let input2 = vec![4.0, 5.0, 6.0];
        let mut output2 = vec![0.0; 3];
        plan.process(&input2, &mut output2).expect("call 2");

        assert_approx_eq(
            &output1,
            &[1.0, 2.0, 3.0],
            EPSILON,
            &format!("plan reuse call 1 ({strategy:?})"),
        );
        assert_approx_eq(
            &output2,
            &[4.0, 5.0, 6.0],
            EPSILON,
            &format!("plan reuse call 2 ({strategy:?})"),
        );
    }

    #[test]
    fn plan_reuse_direct() {
        test_plan_reuse(FirStrategy::Direct);
    }

    #[test]
    fn plan_reuse_overlap_save() {
        test_plan_reuse(FirStrategy::OverlapSave);
    }

    // ── Forced strategy correctness ───────────────────────────────────

    #[test]
    fn forced_direct_long_filter() {
        let factory = make_factory();
        let coeffs: Vec<f64> = (0..32).map(|i: i32| 1.0 / (f64::from(i) + 1.0)).collect();
        let spec = FirSpec::new(coeffs.clone()).with_strategy(FirStrategy::Direct);
        let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

        let mut input = vec![0.0; 64];
        input[0] = 1.0;
        let mut output = vec![0.0; 64];

        plan.process(&input, &mut output).expect("process");

        assert_approx_eq(
            &output[..coeffs.len()],
            &coeffs,
            EPSILON,
            "forced direct long filter impulse",
        );
    }

    #[test]
    fn forced_overlap_save_short_filter() {
        let factory = make_factory();
        let coeffs = vec![1.0, 2.0, 3.0];
        let spec = FirSpec::new(coeffs.clone()).with_strategy(FirStrategy::OverlapSave);
        let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

        let mut input = vec![0.0; 16];
        input[0] = 1.0;
        let mut output = vec![0.0; 16];

        plan.process(&input, &mut output).expect("process");

        assert_approx_eq(
            &output[..coeffs.len()],
            &coeffs,
            EPSILON,
            "forced ols short filter impulse",
        );
    }

    // ── Validation ────────────────────────────────────────────────────

    #[test]
    fn empty_coefficients_returns_error() {
        let factory = make_factory();
        let spec = FirSpec::new(vec![]);
        let result = Fir::<f64>::create_plan(&factory, &spec);
        assert!(result.is_err(), "empty coefficients should return error");
    }

    #[test]
    fn buffer_length_mismatch_returns_error() {
        let factory = make_factory();
        let spec = FirSpec::new(vec![1.0, 2.0]);
        let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 2];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "mismatched buffer lengths should return error"
        );
    }

    // ── Multi-block streaming for overlap-save ────────────────────────

    #[test]
    fn overlap_save_multi_block() {
        let factory = make_factory();
        let coeffs = vec![0.2, 0.3, 0.3, 0.2];
        let spec_d = FirSpec::new(coeffs.clone()).with_strategy(FirStrategy::Direct);
        let spec_o = FirSpec::new(coeffs).with_strategy(FirStrategy::OverlapSave);

        let mut plan_d = Fir::<f64>::create_plan(&factory, &spec_d).expect("direct plan");
        let mut plan_o = Fir::<f64>::create_plan(&factory, &spec_o).expect("ols plan");

        let input: Vec<f64> = (0..200).map(|i| ((f64::from(i)) * 0.07).sin()).collect();
        let mut out_d = vec![0.0; 200];
        let mut out_o = vec![0.0; 200];

        plan_d.process(&input, &mut out_d).expect("direct");
        plan_o.process(&input, &mut out_o).expect("ols");

        assert_approx_eq(&out_d, &out_o, EPSILON, "multi-block overlap-save");
    }

    #[test]
    fn overlap_save_streaming_varied_chunks() {
        let factory = make_factory();
        let coeffs = vec![1.0, -0.5, 0.25, -0.125];

        let spec_ref = FirSpec::new(coeffs.clone()).with_strategy(FirStrategy::Direct);
        let spec_ols = FirSpec::new(coeffs).with_strategy(FirStrategy::OverlapSave);

        let mut plan_ref = Fir::<f64>::create_plan(&factory, &spec_ref).expect("ref plan");
        let mut plan_ols = Fir::<f64>::create_plan(&factory, &spec_ols).expect("ols plan");

        let input: Vec<f64> = (0..100).map(|i| ((f64::from(i)) * 0.13).sin()).collect();

        let mut out_ref = vec![0.0; 100];
        plan_ref.process(&input, &mut out_ref).expect("ref");

        let mut out_ols = vec![0.0; 100];
        let chunks = [3, 7, 1, 15, 22, 8, 44];
        let mut pos = 0;
        for &chunk in &chunks {
            let end = (pos + chunk).min(100);
            plan_ols
                .process(&input[pos..end], &mut out_ols[pos..end])
                .expect("ols chunk");
            pos = end;
            if pos >= 100 {
                break;
            }
        }

        assert_approx_eq(&out_ols, &out_ref, EPSILON, "ols varied chunks");
    }

    // ── Scipy lfilter reference tests ─────────────────────────────────
    //
    // Generated by scripts/gen_fir_lfilter_reference.py.
    // Regenerate with: make gen-fir-lfilter-reference

    include!(testdata!("fir_lfilter_scipy.rs"));

    /// Run both strategies against scipy lfilter output.
    fn test_scipy_lfilter(taps: &[f64], input: &[f64], expected: &[f64], tol: f64, label: &str) {
        for &strategy in &[FirStrategy::Direct, FirStrategy::OverlapSave] {
            let factory = make_factory();
            let spec = FirSpec::new(taps.to_vec()).with_strategy(strategy);
            let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

            let mut output = vec![0.0; input.len()];
            plan.process(input, &mut output).expect("process");

            assert_approx_eq(&output, expected, tol, &format!("{label} ({strategy:?})"));
        }
    }

    #[test]
    fn scipy_lfilter_lowpass_order30_hamming() {
        test_scipy_lfilter(
            LFILTER_LP30_HAMMING_TAPS,
            LFILTER_INPUT,
            LFILTER_LP30_HAMMING_OUTPUT,
            1e-10,
            "LP30 Hamming lfilter",
        );
    }

    #[test]
    fn scipy_lfilter_highpass_order30_hann() {
        test_scipy_lfilter(
            LFILTER_HP30_HANN_TAPS,
            LFILTER_INPUT,
            LFILTER_HP30_HANN_OUTPUT,
            1e-10,
            "HP30 Hann lfilter",
        );
    }

    #[test]
    fn scipy_lfilter_long_streaming() {
        // Process the long signal in chunks, compare against single-shot scipy.
        for &strategy in &[FirStrategy::Direct, FirStrategy::OverlapSave] {
            let factory = make_factory();
            let spec = FirSpec::new(LFILTER_LP30_HAMMING_TAPS.to_vec()).with_strategy(strategy);
            let mut plan = Fir::<f64>::create_plan(&factory, &spec).expect("plan creation");

            let mut output = vec![0.0; LFILTER_LONG_INPUT.len()];

            // Process in varied-size chunks to stress streaming.
            let chunks = [37, 100, 63, 200, 11, 256, 357];
            let mut pos = 0;
            for &chunk in &chunks {
                let end = (pos + chunk).min(output.len());
                if pos >= end {
                    break;
                }
                plan.process(&LFILTER_LONG_INPUT[pos..end], &mut output[pos..end])
                    .expect("chunk process");
                pos = end;
            }

            assert_approx_eq(
                &output,
                LFILTER_LONG_OUTPUT,
                1e-10,
                &format!("long streaming lfilter ({strategy:?})"),
            );
        }
    }
}
