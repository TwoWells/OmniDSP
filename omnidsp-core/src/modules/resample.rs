// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Polyphase FIR resampler module.
//!
//! [`OmniResample`] is a standalone factory (no trait) that creates
//! [`OmniResamplePlan`]s for rational sample rate conversion.  The
//! polyphase structure decomposes the prototype anti-aliasing filter
//! into `L` sub-filters and processes each input sample with the
//! appropriate phase, avoiding the intermediate upsampled signal.
//!
//! Plans are **mutable** — they hold a polyphase delay line and phase
//! counter that persist across calls so successive `process` calls
//! form a continuous stream.

use std::ops::{AddAssign, MulAssign};

use num_traits::Float;

use crate::create::CreatePlan;
use crate::design::resample::{ResampleMode, ResampleSpec};
use crate::error::{Error, Result};
use crate::traits::vecops::VecOps;

// ─── Public types ──────────────────────────────────────────────────────

/// Polyphase FIR resampler factory backed by a [`VecOps`] implementation.
///
/// Creates [`OmniResamplePlan`]s from [`ResampleSpec`]s.  The factory
/// owns the `VecOps` instance; plans own a clone of it.
///
/// There is no `Resample` trait — this is a standalone factory.
/// The spec/plan contract is the same as the trait-based modules:
/// construct a spec, pass it to `create_plan`, call `process` on the plan.
#[derive(Debug, Clone)]
pub struct OmniResample<V> {
    vecops: V,
}

impl<V> OmniResample<V> {
    /// Create a new resampler factory.
    #[must_use]
    pub const fn new(vecops: V) -> Self {
        Self { vecops }
    }
}

/// Execution plan for a streaming polyphase resampler.
///
/// Created by [`OmniResample::create_plan`].  Mutable — holds a delay
/// line and phase counter that persist across calls so successive
/// `process` calls form a continuous stream.  Call [`reset`](Self::reset)
/// to clear the delay line without recreating the plan.
///
/// **Memory:** `L × taps_per_phase` for the polyphase coefficients plus
/// `2 × taps_per_phase` for the doubled delay buffer.
pub struct OmniResamplePlan<T, V> {
    /// Polyphase coefficients: `up_factor` phases × `taps_per_phase`,
    /// flat layout with stride `taps_per_phase`.  Each phase is stored
    /// reversed for direct dot product with the delay line.  Scaled by
    /// `up_factor` to compensate for the implicit zero-insertion that
    /// the polyphase structure skips.
    phases: Vec<T>,
    /// Number of taps per polyphase sub-filter (`ceil(N / L)`).
    taps_per_phase: usize,
    /// Upsampling factor L (number of polyphase phases).
    up_factor: usize,
    /// Downsampling factor M.
    down_factor: usize,
    /// Doubled delay buffer, length = `2 * taps_per_phase`.
    ///
    /// Stores every sample twice (at `write_pos` and `write_pos + tpp`)
    /// so that a contiguous slice of `tpp` elements starting at
    /// `write_pos` always contains the delay line in oldest-first order.
    /// This avoids the two-segment dot product that a plain circular
    /// buffer requires.
    delay: Vec<T>,
    /// Write pointer into the delay buffer.  After a write, it points
    /// to the oldest sample.
    write_pos: usize,
    /// Current polyphase phase index.  Range: `[0, down_factor)`.
    phase: usize,
    /// Length of the full prototype filter (needed for batch output length).
    proto_len: usize,
    /// Processing mode (streaming or batch).
    mode: ResampleMode,
    /// `VecOps` handle for dot products.
    vecops: V,
}

impl<T: std::fmt::Debug, V> std::fmt::Debug for OmniResamplePlan<T, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OmniResamplePlan")
            .field("up_factor", &self.up_factor)
            .field("down_factor", &self.down_factor)
            .field("taps_per_phase", &self.taps_per_phase)
            .finish_non_exhaustive()
    }
}

// ─── Plan methods ─────────────────────────────────────────────────────

impl<T, V> OmniResamplePlan<T, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    V: VecOps<T>,
{
    /// Resample `input`, writing to `output`.
    ///
    /// Returns the number of output samples actually written.  The caller
    /// must provide an output buffer at least [`max_output_len`](Self::max_output_len)
    /// elements long.
    ///
    /// The plan retains internal state (delay line and phase counter)
    /// across calls, so successive calls form a continuous stream.
    /// The prototype FIR filter introduces a group delay of
    /// `(N − 1) / 2` samples at the input rate, where `N` is the
    /// prototype filter length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if `output` is shorter than
    /// [`max_output_len(input.len())`](Self::max_output_len).
    pub fn process(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let max_out = self.max_output_len(input.len());
        if output.len() < max_out {
            return Err(Error::BufferMismatch {
                expected: max_out,
                actual: output.len(),
            });
        }

        let mut out_idx = 0;
        let tpp = self.taps_per_phase;

        for &sample in input {
            self.push_sample(sample);

            while self.phase < self.up_factor {
                if out_idx >= max_out {
                    // Batch mode: stop at the finite convolution length.
                    // Drain remaining phase advancement without producing output.
                    while self.phase < self.up_factor {
                        self.phase += self.down_factor;
                    }
                    break;
                }

                let coeffs_start = self.phase * tpp;
                let coeffs = &self.phases[coeffs_start..coeffs_start + tpp];
                let pos = self.write_pos;

                // Single contiguous dot product — the doubled buffer
                // guarantees `delay[pos..pos + tpp]` is always valid.
                let y = self.vecops.dot(coeffs, &self.delay[pos..pos + tpp])?;

                output[out_idx] = y;
                out_idx += 1;
                self.phase += self.down_factor;
            }

            self.phase -= self.up_factor;
        }

        Ok(out_idx)
    }

    /// Reset internal state (delay buffer, write pointer, and phase counter).
    pub fn reset(&mut self) {
        for x in &mut self.delay {
            *x = T::zero();
        }
        self.write_pos = 0;
        self.phase = 0;
    }

    /// Maximum number of output samples for a given input length.
    ///
    /// The result depends on the [`ResampleMode`]:
    /// - **Streaming**: `⌈input_len × L / M⌉` — driven by the phase
    ///   accumulator, independent of filter length.
    /// - **Batch**: `⌈((input_len − 1) × L + filter_len) / M⌉` —
    ///   the finite convolution length (matches scipy's `upfirdn`).
    ///
    /// The actual count returned by [`process`](Self::process) may be
    /// smaller depending on the current phase state.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "div_ceil is not const on stable Rust"
    )]
    pub fn max_output_len(&self, input_len: usize) -> usize {
        match self.mode {
            ResampleMode::Streaming => input_len
                .saturating_mul(self.up_factor)
                .div_ceil(self.down_factor),
            ResampleMode::Batch => {
                if input_len == 0 {
                    return 0;
                }
                // ceil(((input_len - 1) * up + proto_len) / down)
                (input_len - 1)
                    .saturating_mul(self.up_factor)
                    .saturating_add(self.proto_len)
                    .div_ceil(self.down_factor)
            }
        }
    }

    /// The upsampling factor L.
    #[must_use]
    pub const fn up_factor(&self) -> usize {
        self.up_factor
    }

    /// The downsampling factor M.
    #[must_use]
    pub const fn down_factor(&self) -> usize {
        self.down_factor
    }

    /// Push a sample into the doubled delay buffer.
    ///
    /// Writes at both `write_pos` and `write_pos + taps_per_phase` so
    /// the contiguous-read invariant is maintained.
    fn push_sample(&mut self, sample: T) {
        let tpp = self.taps_per_phase;
        self.delay[self.write_pos] = sample;
        self.delay[self.write_pos + tpp] = sample;
        self.write_pos = (self.write_pos + 1) % tpp;
    }
}

// ─── Plan trait ───────────────────────────────────────────────────────

/// Execution object for a configured streaming resampler.
///
/// The named plan trait for the resampler module, the stateful analogue of
/// [`FirPlan`](crate::traits::fir::FirPlan) (ADR-006 §2 eager plan traits,
/// ADR-007 §6).  Like `FirPlan` it is mutable (`&mut self`) and carries no
/// `Send + Sync` supertrait — the plan holds a delay line and phase counter
/// that persist across calls.  Unlike `FirPlan`, the output length varies with
/// the conversion ratio, so [`max_output_len`](Self::max_output_len) is part of
/// the contract: a generic caller needs it to size `output` before
/// [`process`](Self::process).  Implemented by [`OmniResamplePlan`]; vendor
/// overrides may implement it directly.
pub trait ResamplePlan<T> {
    /// Resample `input`, writing to `output`; returns the number of output
    /// samples written.
    ///
    /// `output` must be at least [`max_output_len(input.len())`](Self::max_output_len)
    /// elements long.  Internal state persists across calls so successive calls
    /// form a continuous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is shorter than the maximum output length
    /// for `input`.
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<usize>;

    /// Maximum number of output samples for a given input length (the buffer
    /// size [`process`](Self::process) requires).
    fn max_output_len(&self, input_len: usize) -> usize;

    /// Reset internal state (delay line and phase counter) without recreating
    /// the plan.
    fn reset(&mut self);
}

impl<T, V> ResamplePlan<T> for OmniResamplePlan<T, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    V: VecOps<T>,
{
    // Each method delegates to the inherent one.  Inherent methods take
    // precedence over trait methods in resolution, so these are not recursive.
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        self.process(input, output)
    }

    fn max_output_len(&self, input_len: usize) -> usize {
        self.max_output_len(input_len)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ─── Factory ──────────────────────────────────────────────────────────

impl<V> OmniResample<V> {
    /// Create a resampling plan from a [`ResampleSpec`].
    ///
    /// The spec provides rational conversion factors and a prototype FIR
    /// filter.  Construct one via [`design`](crate::design::resample::design)
    /// or [`ResampleSpec::new`] for pre-computed data.
    ///
    /// # Errors
    ///
    /// Returns an error if the up-factor cannot be represented in `T`.  The
    /// spec invariants (positive factors, non-empty prototype) are enforced by
    /// [`ResampleSpec::new`](crate::design::resample::ResampleSpec::new)
    /// (ADR-006 §4), so they are not re-checked here.
    pub fn create_plan<T>(&self, spec: &ResampleSpec<T>) -> Result<OmniResamplePlan<T, V>>
    where
        T: Float + AddAssign + MulAssign + Send + Sync,
        V: VecOps<T>,
    {
        let up = spec.up_factor();
        let down = spec.down_factor();
        let proto = spec.prototype_filter();

        let taps_per_phase = proto.len().div_ceil(up);

        // Decompose into L polyphase sub-filters stored in flat layout.
        //
        // Forward decomposition: phase[p][k] = h[p + k·L]
        //
        // We store each phase reversed so that a direct dot product
        // with the delay line [oldest, ..., newest] computes the
        // correct FIR convolution sum:
        //   stored[p][k] = phase[p][tpp − 1 − k] = h[p + (tpp − 1 − k)·L]
        let mut phases = vec![T::zero(); up * taps_per_phase];
        for p in 0..up {
            for k in 0..taps_per_phase {
                let src = p + k * up;
                if src < proto.len() {
                    phases[p * taps_per_phase + (taps_per_phase - 1 - k)] = proto[src];
                }
            }
        }

        // Scale by L to compensate for the zero-insertion that the
        // polyphase structure skips (matches scipy's `h *= up` in
        // `resample_poly`).
        let scale = T::from(up)
            .ok_or_else(|| Error::Internal("failed to convert up_factor to T".into()))?;
        for coeff in &mut phases {
            *coeff *= scale;
        }

        Ok(OmniResamplePlan {
            phases,
            taps_per_phase,
            up_factor: up,
            down_factor: down,
            delay: vec![T::zero(); 2 * taps_per_phase],
            write_pos: 0,
            phase: 0,
            proto_len: proto.len(),
            mode: spec.mode(),
            vecops: self.vecops.clone(),
        })
    }
}

// ─── Spec-parameterized factory ────────────────────────────────────────

/// `OmniResample` is its own [`CreatePlan<ResampleSpec>`] factory.
///
/// This lets the resampler act as a sub-plan factory directly (without a full
/// backend) — e.g. the multirate CQT's octave decimator routes through any
/// `CreatePlan<ResampleSpec>`, and the floor backend's own resampler impl is
/// generated by `impl_generic_backend!`.  Driving it straight from
/// `OmniResample` is convenient for tests and standalone composition.
impl<T, V> CreatePlan<ResampleSpec<T>> for OmniResample<V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    V: VecOps<T>,
{
    type Plan = OmniResamplePlan<T, V>;

    fn create_plan(&self, spec: &ResampleSpec<T>) -> Result<Self::Plan> {
        // Inherent `create_plan` takes precedence over the trait method, so this
        // delegates rather than recurses.
        self.create_plan(spec)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;
    use crate::design::resample::{self, DEFAULT_MAX_PHASES, ResampleMode, ResampleQuality};
    use crate::test_utils::TestVecOps;
    use crate::traits::fir::{FirFilter, FirMeta};
    use crate::window::Window;

    fn factory() -> OmniResample<TestVecOps> {
        OmniResample::new(TestVecOps)
    }

    fn q(val: u8) -> ResampleQuality {
        ResampleQuality::new(val).expect("valid quality")
    }

    fn spec(sr_in: f64, sr_out: f64, quality: u8) -> ResampleSpec<f64> {
        resample::design(
            sr_in,
            sr_out,
            q(quality),
            &Window::Hamming,
            DEFAULT_MAX_PHASES,
            ResampleMode::Streaming,
        )
        .expect("valid design")
    }

    // ── Factory / Plan creation ─────────────────────────────────────

    #[test]
    fn create_plan_44100_to_48000() {
        let f = factory();
        let plan = f
            .create_plan(&spec(44100.0, 48000.0, 5))
            .expect("create plan");
        assert_eq!(plan.up_factor(), 160, "L should be 160");
        assert_eq!(plan.down_factor(), 147, "M should be 147");
    }

    #[test]
    fn create_plan_passthrough() {
        let f = factory();
        let plan = f
            .create_plan(&spec(44100.0, 44100.0, 3))
            .expect("create plan");
        assert_eq!(plan.up_factor(), 1, "L should be 1");
        assert_eq!(plan.down_factor(), 1, "M should be 1");
    }

    #[test]
    fn create_plan_2x_interpolation() {
        let f = factory();
        let plan = f
            .create_plan(&spec(22050.0, 44100.0, 5))
            .expect("create plan");
        assert_eq!(plan.up_factor(), 2, "L should be 2");
        assert_eq!(plan.down_factor(), 1, "M should be 1");
    }

    #[test]
    fn create_plan_2x_decimation() {
        let f = factory();
        let plan = f
            .create_plan(&spec(44100.0, 22050.0, 5))
            .expect("create plan");
        assert_eq!(plan.up_factor(), 1, "L should be 1");
        assert_eq!(plan.down_factor(), 2, "M should be 2");
    }

    // ── max_output_len ──────────────────────────────────────────────

    #[test]
    fn max_output_len_passthrough() {
        let f = factory();
        let plan = f.create_plan(&spec(44100.0, 44100.0, 3)).expect("plan");
        assert_eq!(plan.max_output_len(100), 100, "1:1 ratio");
    }

    #[test]
    fn max_output_len_2x_interpolation() {
        let f = factory();
        let plan = f.create_plan(&spec(22050.0, 44100.0, 5)).expect("plan");
        assert_eq!(plan.max_output_len(100), 200, "2x interpolation");
    }

    #[test]
    fn max_output_len_2x_decimation() {
        let f = factory();
        let plan = f.create_plan(&spec(44100.0, 22050.0, 5)).expect("plan");
        assert_eq!(plan.max_output_len(100), 50, "2x decimation");
    }

    #[test]
    fn max_output_len_44100_to_48000() {
        let f = factory();
        let plan = f.create_plan(&spec(44100.0, 48000.0, 5)).expect("plan");
        // ceil(1000 * 160 / 147) = ceil(1088.435...) = 1089
        let max = plan.max_output_len(1000);
        assert_eq!(max, 1089, "44100→48000 max output for 1000 samples");
    }

    #[test]
    fn max_output_len_zero_input() {
        let f = factory();
        let plan = f.create_plan(&spec(44100.0, 48000.0, 5)).expect("plan");
        assert_eq!(plan.max_output_len(0), 0, "zero input → zero output");
    }

    #[test]
    fn max_output_len_is_upper_bound() {
        let f = factory();
        let mut plan = f.create_plan(&spec(44100.0, 48000.0, 0)).expect("plan");
        let input: Vec<f64> = (0..1000).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let max = plan.max_output_len(input.len());
        let mut output = vec![0.0_f64; max];
        let actual = plan.process(&input, &mut output).expect("process");
        assert!(
            actual <= max,
            "actual output ({actual}) should be ≤ max ({max})"
        );
    }

    // ── Output length ───────────────────────────────────────────────

    #[test]
    fn passthrough_output_length() {
        let f = factory();
        let mut plan = f.create_plan(&spec(44100.0, 44100.0, 3)).expect("plan");
        let input = vec![1.0; 100];
        let mut output = vec![0.0; plan.max_output_len(input.len())];
        let n = plan.process(&input, &mut output).expect("process");
        assert_eq!(n, 100, "passthrough should produce same length");
    }

    #[test]
    fn interpolation_2x_output_length() {
        let f = factory();
        let mut plan = f.create_plan(&spec(22050.0, 44100.0, 5)).expect("plan");
        let input = vec![1.0; 100];
        let mut output = vec![0.0; plan.max_output_len(input.len())];
        let n = plan.process(&input, &mut output).expect("process");
        assert_eq!(n, 200, "2x interpolation should double length");
    }

    #[test]
    fn decimation_2x_output_length() {
        let f = factory();
        let mut plan = f.create_plan(&spec(44100.0, 22050.0, 5)).expect("plan");
        let input = vec![1.0; 100];
        let mut output = vec![0.0; plan.max_output_len(input.len())];
        let n = plan.process(&input, &mut output).expect("process");
        assert_eq!(n, 50, "2x decimation should halve length");
    }

    // ── Streaming continuity ────────────────────────────────────────

    #[test]
    fn streaming_continuity() {
        let f = factory();

        // Process in one shot.
        let mut plan_single = f.create_plan(&spec(44100.0, 48000.0, 0)).expect("plan");
        let input: Vec<f64> = (0..200).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let max_out = plan_single.max_output_len(input.len());
        let mut single_output = vec![0.0; max_out];
        let n_single = plan_single
            .process(&input, &mut single_output)
            .expect("single");

        // Process in chunks.
        let mut plan_chunks = f.create_plan(&spec(44100.0, 48000.0, 0)).expect("plan");
        let mut chunk_output = Vec::new();
        for chunk in input.chunks(50) {
            let max = plan_chunks.max_output_len(chunk.len());
            let mut buf = vec![0.0; max];
            let n = plan_chunks.process(chunk, &mut buf).expect("chunk");
            chunk_output.extend_from_slice(&buf[..n]);
        }

        assert_eq!(
            chunk_output.len(),
            n_single,
            "chunked should produce same total as single-shot"
        );
        for (i, (&a, &b)) in single_output[..n_single]
            .iter()
            .zip(chunk_output.iter())
            .enumerate()
        {
            assert!((a - b).abs() < 1e-10, "sample {i}: single={a}, chunked={b}");
        }
    }

    // ── Reset ───────────────────────────────────────────────────────

    #[test]
    fn reset_reproduces_output() {
        let f = factory();
        let mut plan = f.create_plan(&spec(44100.0, 48000.0, 0)).expect("plan");
        let input: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let max = plan.max_output_len(input.len());

        let mut out1 = vec![0.0; max];
        let n1 = plan.process(&input, &mut out1).expect("first");

        plan.reset();

        let mut out2 = vec![0.0; max];
        let n2 = plan.process(&input, &mut out2).expect("second");

        assert_eq!(n1, n2, "output count should match after reset");
        for (i, (&a, &b)) in out1[..n1].iter().zip(out2[..n2].iter()).enumerate() {
            assert!((a - b).abs() < 1e-10, "sample {i}: first={a}, second={b}");
        }
    }

    // ── Error conditions ────────────────────────────────────────────

    #[test]
    fn process_rejects_small_output() {
        let f = factory();
        let mut plan = f.create_plan(&spec(22050.0, 44100.0, 3)).expect("plan");
        let input = vec![1.0; 100];
        let mut output = vec![0.0; 10]; // way too small
        assert!(
            plan.process(&input, &mut output).is_err(),
            "should reject too-small output buffer"
        );
    }

    #[test]
    fn empty_input_produces_no_output() {
        let f = factory();
        let mut plan = f.create_plan(&spec(44100.0, 48000.0, 5)).expect("plan");
        let input: &[f64] = &[];
        let mut output = vec![0.0; 10];
        let n = plan.process(input, &mut output).expect("process");
        assert_eq!(n, 0, "empty input should produce no output");
    }

    // ── DC convergence ──────────────────────────────────────────────

    #[test]
    fn dc_input_converges_to_unity() {
        let f = factory();
        let mut plan = f.create_plan(&spec(44100.0, 48000.0, 0)).expect("plan");
        let input = vec![1.0_f64; 500];
        let max = plan.max_output_len(input.len());
        let mut output = vec![0.0; max];
        let n = plan.process(&input, &mut output).expect("process");

        // Skip the group delay and check steady-state output ≈ 1.0.
        let steady_start = n / 2;
        for (i, &y) in output[steady_start..n].iter().enumerate() {
            assert!(
                (y - 1.0).abs() < 0.01,
                "DC output should converge to 1.0, sample {}: {y}",
                steady_start + i
            );
        }
    }

    // ── Sine passthrough ────────────────────────────────────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "output length is small enough for exact f64"
    )]
    fn sine_below_nyquist_survives_interpolation() {
        let f = factory();
        let mut plan = f.create_plan(&spec(22050.0, 44100.0, 5)).expect("plan");

        let freq = 1000.0;
        let sr_in = 22050.0;
        let n_in = 1000;
        let input: Vec<f64> = (0..n_in)
            .map(|i| (2.0 * std::f64::consts::PI * freq * f64::from(i) / sr_in).sin())
            .collect();

        let max = plan.max_output_len(input.len());
        let mut output = vec![0.0; max];
        let n_out = plan.process(&input, &mut output).expect("process");

        // Check RMS energy is roughly preserved (skip group delay edges).
        let skip = n_out / 4;
        let chunk = &output[skip..n_out - skip];
        let energy: f64 = chunk.iter().map(|x| x * x).sum::<f64>() / chunk.len() as f64;

        // For a sine wave, mean x² = 0.5.
        assert!(
            (energy - 0.5).abs() < 0.05,
            "sine energy should be ~0.5, got {energy}"
        );
    }

    // ── Debug impl ──────────────────────────────────────────────────

    #[test]
    fn debug_format() {
        let f = factory();
        let plan = f.create_plan(&spec(44100.0, 48000.0, 3)).expect("plan");
        let debug = format!("{plan:?}");
        assert!(
            debug.contains("OmniResamplePlan"),
            "debug should contain type name"
        );
        assert!(
            debug.contains("up_factor"),
            "debug should contain up_factor"
        );
    }

    // ── Plan trait ──────────────────────────────────────────────────

    /// A generic consumer bound on `ResamplePlan` compiles and runs, exercising
    /// `max_output_len`, `process`, and `reset` through the trait.
    #[test]
    fn implements_plan_trait() {
        fn check<P: ResamplePlan<f64>>(plan: &mut P, input: &[f64]) -> usize {
            let max = plan.max_output_len(input.len());
            let mut output = vec![0.0; max];
            let n = plan
                .process(input, &mut output)
                .expect("trait process should succeed");
            plan.reset();
            n
        }

        let f = factory();
        let mut plan = f
            .create_plan(&spec(22050.0, 44100.0, 3))
            .expect("create plan");
        let input = vec![1.0_f64; 100];
        let n = check(&mut plan, &input);
        assert_eq!(
            n, 200,
            "2x interpolation through the trait should double the length"
        );
    }

    // ── Scipy upfirdn reference tests ───────────────────────────────
    //
    // Generated by scripts/gen_resample_poly_reference.py.
    // Regenerate with: make gen-resample-poly-reference
    //
    // These tests inject the same prototype filter that scipy used,
    // bypassing our design layer, so that the polyphase algorithm
    // itself is compared bit-for-bit against scipy's upfirdn.

    #[allow(
        clippy::wildcard_imports,
        reason = "bulk golden-vector import in tests"
    )]
    use omnidsp_testdata::resample_poly_scipy::*;

    /// Build a plan from an explicit prototype filter (test-only).
    ///
    /// Wraps the raw prototype in a [`FirFilter`] with no recorded design
    /// context ([`FirMeta::unknown`], so the cross-spec cutoff check is
    /// skipped), composes a [`ResampleSpec`], and routes through the real
    /// `create_plan` — exercising the spec path rather than duplicating the
    /// polyphase decomposition.
    fn plan_from_prototype(
        prototype: &[f64],
        up_factor: usize,
        down_factor: usize,
    ) -> OmniResamplePlan<f64, TestVecOps> {
        let filter = FirFilter::new(prototype.to_vec(), FirMeta::unknown())
            .expect("non-empty prototype filter");
        let spec = ResampleSpec::new(filter, up_factor, down_factor, ResampleMode::Streaming)
            .expect("valid resample spec");
        factory()
            .create_plan(&spec)
            .expect("create plan from prototype")
    }

    fn assert_approx_eq(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{label}: mismatch at index {i}: got {a}, expected {e} (diff={})",
                (a - e).abs()
            );
        }
    }

    fn run_scipy_case(
        proto: &[f64],
        up: usize,
        down: usize,
        input: &[f64],
        expected: &[f64],
        label: &str,
    ) {
        let mut plan = plan_from_prototype(proto, up, down);
        let max = plan.max_output_len(input.len());
        let mut output = vec![0.0; max];
        let n = plan.process(input, &mut output).expect(label);

        // Our streaming resampler may produce up to 1 sample more than
        // upfirdn (which caps at the convolution length).  Compare the
        // samples that both produce.
        let compare_len = n.min(expected.len());
        assert!(
            compare_len == expected.len(),
            "{label}: reference has {ref_len} samples but we produced only {n}",
            ref_len = expected.len(),
        );
        assert_approx_eq(
            &output[..compare_len],
            &expected[..compare_len],
            1e-10,
            label,
        );
    }

    #[test]
    fn scipy_44100_to_48000() {
        run_scipy_case(
            RPOLY_44_48_PROTO,
            RPOLY_44_48_UP,
            RPOLY_44_48_DOWN,
            RPOLY_44_48_INPUT,
            RPOLY_44_48_OUTPUT,
            "44100→48000",
        );
    }

    #[test]
    fn scipy_48000_to_44100() {
        run_scipy_case(
            RPOLY_48_44_PROTO,
            RPOLY_48_44_UP,
            RPOLY_48_44_DOWN,
            RPOLY_48_44_INPUT,
            RPOLY_48_44_OUTPUT,
            "48000→44100",
        );
    }

    #[test]
    fn scipy_44100_to_22050() {
        run_scipy_case(
            RPOLY_44_22_PROTO,
            RPOLY_44_22_UP,
            RPOLY_44_22_DOWN,
            RPOLY_44_22_INPUT,
            RPOLY_44_22_OUTPUT,
            "44100→22050",
        );
    }

    #[test]
    fn scipy_22050_to_44100() {
        run_scipy_case(
            RPOLY_22_44_PROTO,
            RPOLY_22_44_UP,
            RPOLY_22_44_DOWN,
            RPOLY_22_44_INPUT,
            RPOLY_22_44_OUTPUT,
            "22050→44100",
        );
    }

    #[test]
    fn scipy_long_streaming() {
        // Process in varied-size chunks, compare against single-shot scipy.
        let mut plan = plan_from_prototype(RPOLY_LONG_PROTO, RPOLY_LONG_UP, RPOLY_LONG_DOWN);

        let mut all_output = Vec::new();
        let chunks = [37, 100, 63, 200, 11, 256, 357];
        let mut pos = 0;
        for &chunk in &chunks {
            let end = (pos + chunk).min(RPOLY_LONG_INPUT.len());
            if pos >= end {
                break;
            }
            let max = plan.max_output_len(end - pos);
            let mut buf = vec![0.0; max];
            let n = plan
                .process(&RPOLY_LONG_INPUT[pos..end], &mut buf)
                .expect("chunk");
            all_output.extend_from_slice(&buf[..n]);
            pos = end;
        }

        // Our streaming output may have 1 extra sample vs upfirdn.
        let compare_len = all_output.len().min(RPOLY_LONG_OUTPUT.len());
        assert!(
            compare_len == RPOLY_LONG_OUTPUT.len(),
            "long streaming: reference has {} samples but we produced only {}",
            RPOLY_LONG_OUTPUT.len(),
            all_output.len(),
        );
        assert_approx_eq(
            &all_output[..compare_len],
            &RPOLY_LONG_OUTPUT[..compare_len],
            1e-10,
            "long streaming 44100→48000",
        );
    }

    // ── Batch mode ──────────────────────────────────────────────────

    fn plan_from_prototype_batch(
        prototype: &[f64],
        up_factor: usize,
        down_factor: usize,
    ) -> OmniResamplePlan<f64, TestVecOps> {
        let mut plan = plan_from_prototype(prototype, up_factor, down_factor);
        plan.mode = ResampleMode::Batch;
        plan
    }

    #[test]
    fn batch_output_len_matches_scipy() {
        // scipy's upfirdn: ceil(((N-1)*L + H) / M)
        let plan = plan_from_prototype_batch(RPOLY_44_48_PROTO, RPOLY_44_48_UP, RPOLY_44_48_DOWN);
        let max = plan.max_output_len(RPOLY_44_48_INPUT.len());
        assert_eq!(
            max,
            RPOLY_44_48_OUTPUT.len(),
            "batch max_output_len should match scipy output length"
        );
    }

    #[test]
    fn batch_produces_exact_scipy_count() {
        let mut plan =
            plan_from_prototype_batch(RPOLY_44_48_PROTO, RPOLY_44_48_UP, RPOLY_44_48_DOWN);
        let max = plan.max_output_len(RPOLY_44_48_INPUT.len());
        let mut output = vec![0.0; max];
        let n = plan
            .process(RPOLY_44_48_INPUT, &mut output)
            .expect("batch process");

        assert_eq!(
            n,
            RPOLY_44_48_OUTPUT.len(),
            "batch should produce exactly scipy's output count"
        );
        assert_approx_eq(&output[..n], RPOLY_44_48_OUTPUT, 1e-10, "batch 44100→48000");
    }

    #[test]
    fn batch_mode_via_spec() {
        let f = factory();
        let s = spec(44100.0, 48000.0, 0).with_mode(ResampleMode::Batch);
        let mut plan = f.create_plan(&s).expect("batch plan");

        // Verify batch mode is active: process some data and check
        // that the output count matches max_output_len exactly
        // (batch caps output at the finite convolution length).
        let input = vec![1.0_f64; 100];
        let max = plan.max_output_len(input.len());
        let mut output = vec![0.0; max];
        let n = plan.process(&input, &mut output).expect("batch process");
        assert!(
            n <= max,
            "batch output ({n}) should be ≤ max_output_len ({max})"
        );
    }

    #[test]
    fn batch_empty_input() {
        let mut plan =
            plan_from_prototype_batch(RPOLY_44_48_PROTO, RPOLY_44_48_UP, RPOLY_44_48_DOWN);
        assert_eq!(
            plan.max_output_len(0),
            0,
            "batch with 0 input should produce 0"
        );
        let mut output = vec![0.0; 10];
        let n = plan.process(&[], &mut output).expect("batch empty");
        assert_eq!(n, 0, "batch empty input should produce 0 output");
    }
}
