// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Polyphase FIR resampler module.
//!
//! [`OmniResample`] is a standalone factory (no trait) that creates
//! [`OmniResampleProcessor`]s for rational sample rate conversion.  The
//! polyphase structure decomposes the prototype anti-aliasing filter
//! into `L` sub-filters and processes each input sample with the
//! appropriate phase, avoiding the intermediate upsampled signal.
//!
//! Processors are **stateful** — they hold a polyphase delay line and phase
//! counter that persist across calls so successive `process` calls form a
//! continuous stream.  Batch is how you *drive* the processor:
//! `process(everything) + finish` flushes the convolution tail and reproduces
//! scipy's `upfirdn`; `process(chunk)` … streams.

use crate::design::resample::ResampleSpec;
use crate::error::{Error, Result};
use crate::types::DspFloat;

// ─── Public types ──────────────────────────────────────────────────────

/// Polyphase FIR resampler factory.
///
/// Creates [`OmniResampleProcessor`]s from [`ResampleSpec`]s.  The resampler is
/// a **concrete scalar module**: its hot path is a per-output-sample dot
/// product — a tight inner loop — so it stays scalar rather than composing
/// [`VecOps`](crate::traits::vecops::VecOps).  A `VecOps` `dot` per output
/// sample would cross a vendor's FFI boundary on *every* sample, where the
/// crossing cost dwarfs a short dot (bulk `VecOps` amortizes that cost over a
/// buffer; a per-sample call cannot).  A vendor that wants accelerated
/// resampling supplies a native resampler through the dispatch override
/// instead.
///
/// There is no `Resample` trait — this is a standalone factory.
/// The spec/processor contract is the same as the trait-based modules:
/// construct a spec, pass it to `create_proc`, call `process`/`finish` on the
/// processor.
#[derive(Debug, Clone, Default)]
pub struct OmniResample;

impl OmniResample {
    /// Create a new resampler factory.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Stateful execution object for a streaming polyphase resampler — a
/// **Processor**.
///
/// Created by [`OmniResample::create_proc`].  Holds a delay line and phase
/// counter that persist across calls so successive `process` calls form a
/// continuous stream; [`finish`](Self::finish) flushes the convolution tail at
/// end-of-stream.  Call [`reset`](Self::reset) to clear the delay line without
/// recreating the processor.
///
/// **Memory:** `L × taps_per_phase` for the polyphase coefficients plus
/// `2 × taps_per_phase` for the doubled delay buffer.
pub struct OmniResampleProcessor<T> {
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
    /// Length of the full prototype filter (needed for the finish-tail flush).
    proto_len: usize,
    /// Total input samples consumed since the last `reset`.  Lets `finish` size
    /// the convolution tail exactly: the finite-convolution (`upfirdn`) length
    /// minus the streaming count the `process` calls already emitted.
    samples_in: usize,
}

impl<T: std::fmt::Debug> std::fmt::Debug for OmniResampleProcessor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OmniResampleProcessor")
            .field("up_factor", &self.up_factor)
            .field("down_factor", &self.down_factor)
            .field("taps_per_phase", &self.taps_per_phase)
            .finish_non_exhaustive()
    }
}

// ─── Plan methods ─────────────────────────────────────────────────────

impl<T> OmniResampleProcessor<T>
where
    T: DspFloat,
{
    /// Resample the streaming `input`, writing to `output`; returns the number
    /// of output samples written.
    ///
    /// The caller must provide an output buffer at least
    /// [`max_output_len`](Self::max_output_len) elements long.  The processor
    /// retains internal state (delay line and phase counter) across calls, so
    /// successive calls form a continuous stream; the convolution ring-down tail
    /// is emitted by [`finish`](Self::finish), not here.  The prototype FIR
    /// filter introduces a group delay of `(N − 1) / 2` samples at the input
    /// rate, where `N` is the prototype filter length.
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

        self.samples_in += input.len();
        let mut out_idx = 0;
        let tpp = self.taps_per_phase;

        for &sample in input {
            self.push_sample(sample);

            while self.phase < self.up_factor {
                let coeffs_start = self.phase * tpp;
                let coeffs = &self.phases[coeffs_start..coeffs_start + tpp];
                let pos = self.write_pos;

                // In-core scalar dot — the doubled buffer guarantees
                // `delay[pos..pos + tpp]` is always valid.  This per-output-sample
                // inner product is a tight loop, so it stays scalar rather than
                // routing through `VecOps`: a `VecOps::dot` here would cross a
                // vendor's FFI boundary once per output sample, where the crossing
                // cost dominates a short dot.  The `fold` matches the scalar
                // `VecOps::dot` arithmetic exactly, so resampling values are
                // unchanged.
                let y = coeffs
                    .iter()
                    .zip(&self.delay[pos..pos + tpp])
                    .fold(T::zero(), |acc, (&c, &d)| acc + c * d);

                // The streaming count `⌈N·L/M⌉` is exactly `max_output_len`, and
                // `output` is checked to hold it, so `out_idx` stays in bounds.
                output[out_idx] = y;
                out_idx += 1;
                self.phase += self.down_factor;
            }

            self.phase -= self.up_factor;
        }

        Ok(out_idx)
    }

    /// Signal end-of-stream: flush the convolution ring-down tail.
    ///
    /// Feeds the prototype's remaining support past the last real input sample
    /// so the finite convolution runs out, then emits **exactly** the tail scipy
    /// `upfirdn` would: the finite-convolution length for everything seen since
    /// the last `reset`, minus the streaming count [`process`](Self::process)
    /// already produced.  So `process(everything) + finish` reproduces `upfirdn`
    /// exactly when the prototype spans more than one polyphase phase
    /// (`proto_len > up_factor`, the normal case); a shorter prototype is
    /// degenerate, and the streaming count may already exceed the finite
    /// convolution.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if `output` is shorter than
    /// [`finish_output_len`](Self::finish_output_len).
    pub fn finish(&mut self, output: &mut [T]) -> Result<usize> {
        // The exact tail is the finite-convolution (`upfirdn`) length minus what
        // the streaming `process` calls already emitted; cap at it so the total
        // lands on `upfirdn` rather than over-running by the phase remainder.
        let tail_count = self
            .batch_output_len(self.samples_in)
            .saturating_sub(self.max_output_len(self.samples_in));
        let zeros = vec![T::zero(); self.tail_input_len()];
        let produced = self.process(&zeros, output)?;
        self.reset();
        Ok(produced.min(tail_count))
    }

    /// One-shot convenience: resample a complete `input` to a complete `output`
    /// on a fresh stream — `reset`, then `process(input)`, then `finish`.
    /// Returns the total number of samples written and leaves the processor
    /// clean.  `output` must hold at least
    /// [`max_output_len`](Self::max_output_len) for `input` plus the finish tail
    /// — equivalently the scipy `upfirdn` length
    /// `⌈((input_len − 1)·L + filter_len) / M⌉`.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is too short, or if execution fails.
    pub fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        self.reset();
        let n = self.process(input, output)?;
        let t = self.finish(&mut output[n..])?;
        Ok(n + t)
    }

    /// Reset internal state (delay buffer, write pointer, and phase counter).
    pub fn reset(&mut self) {
        for x in &mut self.delay {
            *x = T::zero();
        }
        self.write_pos = 0;
        self.phase = 0;
        self.samples_in = 0;
    }

    /// Maximum number of output samples [`process`](Self::process) writes for a
    /// given input length: `⌈input_len × L / M⌉`.
    ///
    /// Driven by the phase accumulator, independent of filter length.  The
    /// ring-down tail past the input is emitted separately by
    /// [`finish`](Self::finish); size its buffer with
    /// [`finish_output_len`](Self::finish_output_len).  The actual count
    /// returned by `process` may be smaller depending on the current phase state.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "div_ceil is not const on stable Rust"
    )]
    pub fn max_output_len(&self, input_len: usize) -> usize {
        input_len
            .saturating_mul(self.up_factor)
            .div_ceil(self.down_factor)
    }

    /// Number of trailing zero input samples [`finish`](Self::finish) feeds to
    /// flush the convolution tail: the prototype's support past the last real
    /// input sample, `ceil((proto_len − 1) / L)` input samples.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "div_ceil is not const on stable Rust"
    )]
    fn tail_input_len(&self) -> usize {
        self.proto_len.saturating_sub(1).div_ceil(self.up_factor)
    }

    /// Full finite-convolution (scipy `upfirdn`) output length for `input_len`
    /// input samples: `⌈((input_len − 1)·L + proto_len) / M⌉` — the streaming
    /// count plus the ring-down tail, the total `process(all) + finish` yields.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "div_ceil is not const on stable Rust"
    )]
    fn batch_output_len(&self, input_len: usize) -> usize {
        if input_len == 0 {
            return 0;
        }
        (input_len - 1)
            .saturating_mul(self.up_factor)
            .saturating_add(self.proto_len)
            .div_ceil(self.down_factor)
    }

    /// Upper bound on output samples [`finish`](Self::finish) writes — the size
    /// to allocate for its `output` buffer.
    #[must_use]
    pub fn finish_output_len(&self) -> usize {
        self.max_output_len(self.tail_input_len())
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

// ─── Processor trait ──────────────────────────────────────────────────

/// Stateful execution object for a configured streaming resampler — a
/// **Processor**.
///
/// The named processor trait for the resampler module, the analogue of the FIR
/// [`FirProcessor`](crate::traits::fir::FirProcessor): it is mutable
/// (`&mut self`) and carries no `Send + Sync` supertrait — the processor holds a
/// delay line and phase counter that persist across calls.  Unlike a same-rate
/// FIR, the output length varies with the conversion ratio, so
/// [`max_output_len`](Self::max_output_len) is part of the contract: a generic
/// caller needs it to size `output` before [`process`](Self::process).
/// [`finish`](Self::finish) flushes the convolution tail at end-of-stream.
/// Implemented by [`OmniResampleProcessor`]; vendor overrides may implement it
/// directly.
pub trait ResampleProcessor<T> {
    /// Resample the streaming `input`, writing to `output`; returns the number
    /// of output samples written.
    ///
    /// `output` must be at least [`max_output_len(input.len())`](Self::max_output_len)
    /// elements long.  Internal state persists across calls so successive calls
    /// form a continuous stream; the ring-down tail is emitted by
    /// [`finish`](Self::finish).
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is shorter than the maximum output length
    /// for `input`.
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<usize>;

    /// Signal end-of-stream: flush the convolution ring-down tail into `output`,
    /// returning the count, and leave the processor clean.  `output` must be at
    /// least [`finish_output_len`](Self::finish_output_len) elements long.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is too short.
    fn finish(&mut self, output: &mut [T]) -> Result<usize>;

    /// One-shot convenience: `(reset →) process(all) → finish` on a fresh
    /// stream, returning the total count and leaving the processor clean.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is too short.
    fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<usize>;

    /// Maximum number of output samples [`process`](Self::process) writes for a
    /// given input length.
    fn max_output_len(&self, input_len: usize) -> usize;

    /// Upper bound on output samples [`finish`](Self::finish) writes.
    fn finish_output_len(&self) -> usize;

    /// Reset internal state (delay line and phase counter) without recreating
    /// the processor.
    fn reset(&mut self);
}

impl<T> ResampleProcessor<T> for OmniResampleProcessor<T>
where
    T: DspFloat,
{
    // Each method delegates to the inherent one.  Inherent methods take
    // precedence over trait methods in resolution, so these are not recursive.
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        self.process(input, output)
    }

    fn finish(&mut self, output: &mut [T]) -> Result<usize> {
        self.finish(output)
    }

    fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        self.execute(input, output)
    }

    fn max_output_len(&self, input_len: usize) -> usize {
        self.max_output_len(input_len)
    }

    fn finish_output_len(&self) -> usize {
        self.finish_output_len()
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ─── Factory ──────────────────────────────────────────────────────────

impl OmniResample {
    /// Create a resampling processor from a [`ResampleSpec`].
    ///
    /// The spec provides rational conversion factors and a prototype FIR
    /// filter (carried in f64); its coefficients are cast to the operation's
    /// precision `T` at this create edge.  Construct one via
    /// [`design`](crate::design::resample::design) or [`ResampleSpec::new`] for
    /// pre-computed data.
    ///
    /// # Errors
    ///
    /// Returns an error if the up-factor or a coefficient cannot be represented
    /// in `T`.  The spec invariants (positive factors, non-empty prototype) are
    /// enforced by [`ResampleSpec::new`](crate::design::resample::ResampleSpec::new),
    /// so they are not re-checked here.
    pub fn create_proc<T>(&self, spec: &ResampleSpec) -> Result<OmniResampleProcessor<T>>
    where
        T: DspFloat,
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
        //
        // The prototype is carried in f64; cast each tap to `T` at this edge.
        let mut phases = vec![T::zero(); up * taps_per_phase];
        for p in 0..up {
            for k in 0..taps_per_phase {
                let src = p + k * up;
                if src < proto.len() {
                    phases[p * taps_per_phase + (taps_per_phase - 1 - k)] = T::from(proto[src])
                        .ok_or_else(|| {
                            Error::Internal("cannot represent resample coefficient in T".into())
                        })?;
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

        Ok(OmniResampleProcessor {
            phases,
            taps_per_phase,
            up_factor: up,
            down_factor: down,
            delay: vec![T::zero(); 2 * taps_per_phase],
            write_pos: 0,
            phase: 0,
            proto_len: proto.len(),
            samples_in: 0,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;
    use crate::design::resample::{self, DEFAULT_MAX_PHASES, ResampleQuality};
    use crate::traits::fir::{FirFilter, FirMeta};
    use crate::window::Window;

    fn factory() -> OmniResample {
        OmniResample::new()
    }

    fn q(val: u8) -> ResampleQuality {
        ResampleQuality::new(val).expect("valid quality")
    }

    fn spec(sr_in: f64, sr_out: f64, quality: u8) -> ResampleSpec {
        resample::design(
            sr_in,
            sr_out,
            q(quality),
            &Window::Hamming,
            DEFAULT_MAX_PHASES,
        )
        .expect("valid design")
    }

    // ── Factory / Plan creation ─────────────────────────────────────

    #[test]
    fn create_plan_44100_to_48000() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 5))
            .expect("create plan");
        assert_eq!(plan.up_factor(), 160, "L should be 160");
        assert_eq!(plan.down_factor(), 147, "M should be 147");
    }

    #[test]
    fn create_plan_passthrough() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(44100.0, 44100.0, 3))
            .expect("create plan");
        assert_eq!(plan.up_factor(), 1, "L should be 1");
        assert_eq!(plan.down_factor(), 1, "M should be 1");
    }

    #[test]
    fn create_plan_2x_interpolation() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(22050.0, 44100.0, 5))
            .expect("create plan");
        assert_eq!(plan.up_factor(), 2, "L should be 2");
        assert_eq!(plan.down_factor(), 1, "M should be 1");
    }

    #[test]
    fn create_plan_2x_decimation() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(44100.0, 22050.0, 5))
            .expect("create plan");
        assert_eq!(plan.up_factor(), 1, "L should be 1");
        assert_eq!(plan.down_factor(), 2, "M should be 2");
    }

    // ── max_output_len ──────────────────────────────────────────────

    #[test]
    fn max_output_len_passthrough() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(44100.0, 44100.0, 3))
            .expect("plan");
        assert_eq!(plan.max_output_len(100), 100, "1:1 ratio");
    }

    #[test]
    fn max_output_len_2x_interpolation() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(22050.0, 44100.0, 5))
            .expect("plan");
        assert_eq!(plan.max_output_len(100), 200, "2x interpolation");
    }

    #[test]
    fn max_output_len_2x_decimation() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(44100.0, 22050.0, 5))
            .expect("plan");
        assert_eq!(plan.max_output_len(100), 50, "2x decimation");
    }

    #[test]
    fn max_output_len_44100_to_48000() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 5))
            .expect("plan");
        // ceil(1000 * 160 / 147) = ceil(1088.435...) = 1089
        let max = plan.max_output_len(1000);
        assert_eq!(max, 1089, "44100→48000 max output for 1000 samples");
    }

    #[test]
    fn max_output_len_zero_input() {
        let f = factory();
        let plan = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 5))
            .expect("plan");
        assert_eq!(plan.max_output_len(0), 0, "zero input → zero output");
    }

    #[test]
    fn max_output_len_is_upper_bound() {
        let f = factory();
        let mut plan = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 0))
            .expect("plan");
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
        let mut plan = f
            .create_proc::<f64>(&spec(44100.0, 44100.0, 3))
            .expect("plan");
        let input = vec![1.0; 100];
        let mut output = vec![0.0; plan.max_output_len(input.len())];
        let n = plan.process(&input, &mut output).expect("process");
        assert_eq!(n, 100, "passthrough should produce same length");
    }

    #[test]
    fn interpolation_2x_output_length() {
        let f = factory();
        let mut plan = f
            .create_proc::<f64>(&spec(22050.0, 44100.0, 5))
            .expect("plan");
        let input = vec![1.0; 100];
        let mut output = vec![0.0; plan.max_output_len(input.len())];
        let n = plan.process(&input, &mut output).expect("process");
        assert_eq!(n, 200, "2x interpolation should double length");
    }

    #[test]
    fn decimation_2x_output_length() {
        let f = factory();
        let mut plan = f
            .create_proc::<f64>(&spec(44100.0, 22050.0, 5))
            .expect("plan");
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
        let mut plan_single = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 0))
            .expect("plan");
        let input: Vec<f64> = (0..200).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let max_out = plan_single.max_output_len(input.len());
        let mut single_output = vec![0.0; max_out];
        let n_single = plan_single
            .process(&input, &mut single_output)
            .expect("single");

        // Process in chunks.
        let mut plan_chunks = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 0))
            .expect("plan");
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
        let mut plan = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 0))
            .expect("plan");
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
        let mut plan = f
            .create_proc::<f64>(&spec(22050.0, 44100.0, 3))
            .expect("plan");
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
        let mut plan = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 5))
            .expect("plan");
        let input: &[f64] = &[];
        let mut output = vec![0.0; 10];
        let n = plan.process(input, &mut output).expect("process");
        assert_eq!(n, 0, "empty input should produce no output");
    }

    // ── DC convergence ──────────────────────────────────────────────

    #[test]
    fn dc_input_converges_to_unity() {
        let f = factory();
        let mut plan = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 0))
            .expect("plan");
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
        let mut plan = f
            .create_proc::<f64>(&spec(22050.0, 44100.0, 5))
            .expect("plan");

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
        let plan = f
            .create_proc::<f64>(&spec(44100.0, 48000.0, 3))
            .expect("plan");
        let debug = format!("{plan:?}");
        assert!(
            debug.contains("OmniResampleProcessor"),
            "debug should contain type name"
        );
        assert!(
            debug.contains("up_factor"),
            "debug should contain up_factor"
        );
    }

    // ── Plan trait ──────────────────────────────────────────────────

    /// A generic consumer bound on `ResampleProcessor` compiles and runs, exercising
    /// `max_output_len`, `process`, and `reset` through the trait.
    #[test]
    fn implements_plan_trait() {
        fn check<P: ResampleProcessor<f64>>(plan: &mut P, input: &[f64]) -> usize {
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
            .create_proc::<f64>(&spec(22050.0, 44100.0, 3))
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
    /// `create_proc` — exercising the spec path rather than duplicating the
    /// polyphase decomposition.
    fn plan_from_prototype(
        prototype: &[f64],
        up_factor: usize,
        down_factor: usize,
    ) -> OmniResampleProcessor<f64> {
        let filter = FirFilter::new(prototype.to_vec(), FirMeta::unknown())
            .expect("non-empty prototype filter");
        let spec = ResampleSpec::new(filter, up_factor, down_factor).expect("valid resample spec");
        factory()
            .create_proc(&spec)
            .expect("create processor from prototype")
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
        // The golden vectors are the scipy reference truncated to the streaming
        // output length `⌈N·L/M⌉`, so drive the processor as a stream (`process`
        // only) and compare against them.  The full-convolution
        // `process(all) + finish` path is value-tested by
        // `finish_flushes_full_convolution_tail`.
        let mut plan = plan_from_prototype(proto, up, down);
        let mut output = vec![0.0; plan.max_output_len(input.len())];
        let n = plan.process(input, &mut output).expect(label);
        let compare_len = n.min(expected.len());
        assert!(
            compare_len == expected.len(),
            "{label}: reference has {} samples but we produced only {n}",
            expected.len(),
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
        // Process the long signal in varied-size chunks; the concatenated
        // streaming output matches the scipy reference (truncated to the
        // streaming length) sample-for-sample across chunk boundaries.
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

    // ── End-of-stream flush: `process(all) + finish` reproduces upfirdn ──
    //
    // There is no batch *mode* — batch is how you drive the processor.
    // `process(everything)` emits the streaming count, then `finish` flushes the
    // convolution ring-down tail; together they equal scipy's `upfirdn`
    // finite-convolution length `⌈((N−1)·L + H) / M⌉`.

    /// Drive a processor to end-of-stream: `process(all) + finish`, returning the
    /// concatenated output.
    fn drive_batch(plan: &mut OmniResampleProcessor<f64>, input: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; plan.max_output_len(input.len())];
        let n = plan.process(input, &mut out).expect("batch process");
        let mut tail = vec![0.0; plan.finish_output_len()];
        let t = plan.finish(&mut tail).expect("batch finish");
        out.truncate(n);
        out.extend_from_slice(&tail[..t]);
        out
    }

    #[test]
    fn finish_flushes_full_convolution_tail() {
        // A prototype that spans more than one phase (`H > L`, here 65 taps over
        // L = 2), so the streaming count never exceeds the finite convolution:
        // `process(all) + finish` reaches scipy `upfirdn`'s full length
        // `⌈((N−1)·L + H) / M⌉`, and its streaming prefix still matches the
        // reference.  (The 44↔48 cases use a single-tap-per-phase prototype where
        // `H < L` — a degenerate config that never reaches this path.)
        let mut plan = plan_from_prototype(RPOLY_22_44_PROTO, RPOLY_22_44_UP, RPOLY_22_44_DOWN);
        let out = drive_batch(&mut plan, RPOLY_22_44_INPUT);

        let upfirdn_len = ((RPOLY_22_44_INPUT.len() - 1) * RPOLY_22_44_UP
            + RPOLY_22_44_PROTO.len())
        .div_ceil(RPOLY_22_44_DOWN);
        assert_eq!(
            out.len(),
            upfirdn_len,
            "process(all) + finish should reach the upfirdn finite-convolution length",
        );
        assert_approx_eq(
            &out[..RPOLY_22_44_OUTPUT.len()],
            RPOLY_22_44_OUTPUT,
            1e-10,
            "finish streaming prefix",
        );
    }

    #[test]
    fn execute_matches_process_plus_finish() {
        // `execute` is the one-shot convenience for `process(all) + finish`.
        let mut a = plan_from_prototype(RPOLY_44_48_PROTO, RPOLY_44_48_UP, RPOLY_44_48_DOWN);
        let driven = drive_batch(&mut a, RPOLY_44_48_INPUT);

        let mut b = plan_from_prototype(RPOLY_44_48_PROTO, RPOLY_44_48_UP, RPOLY_44_48_DOWN);
        let cap = b.max_output_len(RPOLY_44_48_INPUT.len()) + b.finish_output_len();
        let mut out = vec![0.0; cap];
        let n = b.execute(RPOLY_44_48_INPUT, &mut out).expect("execute");
        assert_eq!(
            n,
            driven.len(),
            "execute should write the same count as process(all) + finish"
        );
        assert_approx_eq(&out[..n], &driven, 1e-12, "execute vs process+finish");
    }

    #[test]
    fn batch_empty_input() {
        let mut plan = plan_from_prototype(RPOLY_44_48_PROTO, RPOLY_44_48_UP, RPOLY_44_48_DOWN);
        assert_eq!(
            plan.max_output_len(0),
            0,
            "zero input should produce zero streaming output"
        );
        // A bare `finish` on a fresh (zero-state) processor flushes nothing of
        // substance: the empty-input batch produces no output.
        let out = drive_batch(&mut plan, &[]);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "empty-input batch should produce no signal"
        );
    }
}
