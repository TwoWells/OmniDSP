// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Streaming, newest-anchored multirate CQT (ticket 22).
//!
//! [`OmniCqtStreamPlan`] is the streaming analogue of the batch
//! [`OmniCqtPlan`](super::batch::OmniCqtPlan): a stateful `&mut self` plan that
//! mirrors [`ResamplePlan`] exactly —
//! no `Send + Sync`, a [`reset`](CqtStreamPlan::reset), and a variable-count
//! [`process`](CqtStreamPlan::process) that returns how many feature columns it
//! emitted (the column analogue of the resampler's sample count).  Feed any
//! chunk of samples; get back the `0..=max_output_columns` hop-boundary columns
//! the input crossed.
//!
//! # Newest-anchoring
//!
//! The batch CQT anchors every bin's kernel at the **oldest** sample of the
//! analysis frame (index 0), so its output trails the frame's old edge by the
//! whole window length.  Newest-anchoring is the inverse: each bin's output is
//! referenced to the **most recent** sample, collapsing its latency to the
//! Gabor floor `Q/f` (treble becomes effectively instant; only deep bass stays
//! inherently slow).  Anchoring is a pure time shift, so it changes **only
//! phase, not magnitude**: a newest-anchored column equals the batch CQT of the
//! same window, with each bin de-rotated by the per-bin linear phase
//! `exp(-j·2π·f_k·Δ/sr)`, where `Δ = fft_length − 1` is the window-start →
//! window-end shift.
//!
//! # This slice (22a): naive-but-correct
//!
//! This implementation rings the raw input at full rate and, on each
//! top-octave hop boundary the input crosses, runs the **batch** per-octave
//! analysis over the `fft_length`-sample window ending at "now," then applies
//! the per-bin newest-anchoring phase.  It is correct and newest-anchored but
//! deliberately unoptimized — the inner batch plan resets its decimator per
//! column.  Ticket 22b replaces the internals (continuous decimator, per-octave
//! rings, a true fast path) behind this same surface; the surface and its
//! conformance check do not change.

use std::ops::{AddAssign, MulAssign};

use num_complex::Complex;

use crate::create::CreatePlan;
use crate::design::cqt::CqtSpec;
use crate::design::resample::ResampleSpec;
use crate::error::{Error, Result};
use crate::modules::cqt::batch::OmniCqt;
use crate::modules::resample::ResamplePlan;
use crate::traits::dft::{DftR2c, DftR2cPlan};
use crate::traits::vecops::VecOps;
use crate::types::DspFloat;

// ─── Spec ──────────────────────────────────────────────────────────────

/// Streaming CQT specification.
///
/// Thin newtype wrapper over a batch [`CqtSpec`]: the streaming and batch paths
/// describe the **same** transform (same bins, windows, sample rate, FFT
/// length) and differ only in *state*, not math.  Wrapping rather than aliasing
/// keeps `CreatePlan<CqtStreamSpec>` a distinct dispatch route from
/// `CreatePlan<CqtSpec>` (wired in ticket 22c), so a backend can build either
/// plan from the same configuration.
///
/// The top-octave hop length ([`CqtSpec::hop_length`]) is the streaming
/// column-emission clock: [`process`](CqtStreamPlan::process) emits one column
/// each time the input crosses a hop boundary.
#[derive(Debug, Clone)]
pub struct CqtStreamSpec<T> {
    spec: CqtSpec<T>,
}

impl<T> CqtStreamSpec<T> {
    /// Wrap a batch [`CqtSpec`] as a streaming spec.
    #[must_use]
    pub const fn new(spec: CqtSpec<T>) -> Self {
        Self { spec }
    }

    /// The underlying batch [`CqtSpec`].
    #[must_use]
    pub const fn spec(&self) -> &CqtSpec<T> {
        &self.spec
    }

    /// Consume the wrapper, returning the underlying [`CqtSpec`].
    #[must_use]
    pub fn into_spec(self) -> CqtSpec<T> {
        self.spec
    }
}

// ─── Plan trait ───────────────────────────────────────────────────────

/// Execution object for a configured **streaming** Constant-Q Transform.
///
/// The named, stateful plan trait for the streaming CQT, mirroring
/// [`ResamplePlan`] exactly (the locked
/// `&mut self` streaming-plan category, SL-09 / ADR-006 §2 / ADR-007 §6): it is
/// mutable, carries **no** `Send + Sync` supertrait, exposes a
/// [`reset`](Self::reset), and has a variable-count [`process`](Self::process)
/// paired with a [`max_output_columns`](Self::max_output_columns) sizing method.
///
/// Where `ResamplePlan` is a stream→stream *filter* (`T → T`), this is a
/// stream→feature-frame *analyzer* (`T → Complex<T>` column) emitted on the
/// top-octave hop clock — a different cadence than the input.
/// `max_output_columns` resolves that mismatch the way the resampler's
/// `max_output_len` does: feed any chunk, get the hop-boundary columns it
/// crossed.  Output is complex; the magnitude convenience
/// ([`process_magnitude`](OmniCqtStreamPlan::process_magnitude)) stays inherent
/// on [`OmniCqtStreamPlan`].
pub trait CqtStreamPlan<T> {
    /// Number of frequency bins per column.
    fn num_bins(&self) -> usize;

    /// Per-bin center frequencies in Hz (pinned low → high).
    fn bin_frequencies(&self) -> &[f64];

    /// Upper bound on columns [`process`](Self::process) may emit for
    /// `input_len` new samples — size `out` to
    /// `max_output_columns(input_len) * num_bins`.  The
    /// [`ResamplePlan::max_output_len`] analogue.
    fn max_output_columns(&self, input_len: usize) -> usize;

    /// Feed new samples; write the newest-anchored columns whose hop boundaries
    /// the input crossed (each `num_bins` wide, low → high) and return the
    /// column count.  State persists across calls.
    ///
    /// `out` must hold at least
    /// [`max_output_columns(input.len())`](Self::max_output_columns) columns —
    /// that many times [`num_bins`](Self::num_bins) elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `out` is too short, or if execution fails.
    fn process(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<usize>;

    /// Reset all state (the input ring and hop phase) without recreating the
    /// plan — the [`ResamplePlan::reset`] analogue.
    fn reset(&mut self);
}

// ─── Plan ──────────────────────────────────────────────────────────────

/// Streaming, newest-anchored multirate CQT plan.
///
/// Created by [`OmniCqt::create_stream_plan`].  Mutable — it rings the raw
/// input and tracks the hop phase across calls so successive
/// [`process`](Self::process) calls form a continuous stream.  This slice (22a)
/// is naive-but-correct: each emitted column runs the wrapped batch plan over
/// the window ending at "now," then applies the per-bin newest-anchoring phase.
///
/// Type parameters mirror [`OmniCqtPlan`](super::batch::OmniCqtPlan): `RP` the
/// per-octave [`DftR2cPlan`], `V` the [`VecOps`], `ResP` the routed decimator
/// [`ResamplePlan`].
pub struct OmniCqtStreamPlan<T, RP, V, ResP> {
    /// The wrapped batch plan — the per-octave oracle this slice runs verbatim.
    batch: super::batch::OmniCqtPlan<T, RP, V, ResP>,
    /// `VecOps` handle for the magnitude convenience.
    vecops: V,
    /// Per-bin newest-anchoring rotors `exp(-j·2π·f_k·Δ/sr)`, `Δ = fft_len−1`.
    rotors: Vec<Complex<T>>,
    /// Required analysis-window length (the spec's FFT length).
    fft_length: usize,
    /// Top-octave hop length — the column-emission clock.
    hop_length: usize,
    /// Number of frequency bins per column.
    num_bins: usize,
    /// Input ring of the most-recent `fft_length` samples (chronological order
    /// reconstructed via `ring_pos`).
    ring: Vec<T>,
    /// Write head into `ring`; after a write it points to the oldest sample.
    ring_pos: usize,
    /// Samples accumulated toward the next hop boundary, in `[0, hop_length)`.
    pending: usize,
    /// Scratch window buffer (length `fft_length`) for the batch analysis.
    window: Vec<T>,
    /// Scratch column buffer (length `num_bins`) for one batch result.
    column: Vec<Complex<T>>,
}

impl<T, RP, V, ResP> std::fmt::Debug for OmniCqtStreamPlan<T, RP, V, ResP> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OmniCqtStreamPlan")
            .field("num_bins", &self.num_bins)
            .field("fft_length", &self.fft_length)
            .field("hop_length", &self.hop_length)
            .finish_non_exhaustive()
    }
}

impl<T, RP, V, ResP> OmniCqtStreamPlan<T, RP, V, ResP>
where
    T: DspFloat + AddAssign + MulAssign,
    RP: DftR2cPlan<T>,
    V: VecOps<T>,
    ResP: ResamplePlan<T>,
{
    /// Feed new samples; write the newest-anchored columns whose hop boundaries
    /// the input crossed and return the column count.  See
    /// [`CqtStreamPlan::process`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if `out` holds fewer than
    /// [`max_output_columns(input.len())`](Self::max_output_columns) columns
    /// (that many times [`num_bins`](Self::num_bins) elements), or propagates
    /// batch-analysis execution failures.
    pub fn process(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<usize> {
        let need = self.max_output_columns(input.len()) * self.num_bins;
        if out.len() < need {
            return Err(Error::BufferMismatch {
                expected: need,
                actual: out.len(),
            });
        }

        let mut columns = 0usize;
        for &sample in input {
            self.push_sample(sample);
            self.pending += 1;
            if self.pending == self.hop_length {
                self.pending = 0;
                let start = columns * self.num_bins;
                let dst = &mut out[start..start + self.num_bins];
                self.emit_column(dst)?;
                columns += 1;
            }
        }
        Ok(columns)
    }

    /// Compute the newest-anchored **magnitude** columns.
    ///
    /// Convenience wrapper around [`process`](Self::process) that writes
    /// `|CQT[k]|` per bin (the visualiser's path).  `out` must hold at least
    /// [`max_output_columns(input.len())`](Self::max_output_columns) columns —
    /// that many times [`num_bins`](Self::num_bins) elements; returns the column
    /// count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if `out` is too short, or propagates
    /// execution failures.
    pub fn process_magnitude(&mut self, input: &[T], out: &mut [T]) -> Result<usize> {
        let need = self.max_output_columns(input.len()) * self.num_bins;
        if out.len() < need {
            return Err(Error::BufferMismatch {
                expected: need,
                actual: out.len(),
            });
        }

        let mut columns = 0usize;
        let zero = Complex::new(T::zero(), T::zero());
        let mut col = vec![zero; self.num_bins];
        for &sample in input {
            self.push_sample(sample);
            self.pending += 1;
            if self.pending == self.hop_length {
                self.pending = 0;
                self.emit_column(&mut col)?;
                let start = columns * self.num_bins;
                self.vecops
                    .mag(&col, &mut out[start..start + self.num_bins])?;
                columns += 1;
            }
        }
        Ok(columns)
    }

    /// Reset the input ring and hop phase without recreating the plan.
    pub fn reset(&mut self) {
        for x in &mut self.ring {
            *x = T::zero();
        }
        self.ring_pos = 0;
        self.pending = 0;
    }

    /// Upper bound on columns emitted for `input_len` new samples.
    ///
    /// A column fires every `hop_length` samples crossed; with at most
    /// `hop_length − 1` samples already pending, feeding `input_len` samples
    /// crosses at most `⌈input_len / hop_length⌉` boundaries.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "div_ceil is not const on stable Rust"
    )]
    pub fn max_output_columns(&self, input_len: usize) -> usize {
        input_len.div_ceil(self.hop_length)
    }

    /// Number of frequency bins per column.
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Required analysis-window length (the spec's FFT length).
    #[must_use]
    pub const fn fft_length(&self) -> usize {
        self.fft_length
    }

    /// Top-octave hop length (the column-emission clock).
    #[must_use]
    pub const fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Per-bin center frequencies in Hz (pinned low → high).
    #[must_use]
    pub fn bin_frequencies(&self) -> &[f64] {
        self.batch.bin_frequencies()
    }

    /// Push one sample into the input ring (newest sample, overwriting oldest).
    fn push_sample(&mut self, sample: T) {
        self.ring[self.ring_pos] = sample;
        self.ring_pos = (self.ring_pos + 1) % self.fft_length;
    }

    /// Run the batch analysis over the window ending at "now," apply the
    /// per-bin newest-anchoring rotor, and write the column to `dst`.
    ///
    /// After [`push_sample`](Self::push_sample), `ring_pos` points at the oldest
    /// of the `fft_length` retained samples, so the chronological window is
    /// `ring[ring_pos..] ++ ring[..ring_pos]`.
    fn emit_column(&mut self, dst: &mut [Complex<T>]) -> Result<()> {
        let split = self.fft_length - self.ring_pos;
        self.window[..split].copy_from_slice(&self.ring[self.ring_pos..]);
        self.window[split..].copy_from_slice(&self.ring[..self.ring_pos]);

        // Batch (oldest-anchored) analysis of the window, then de-rotate each
        // bin to the newest-anchored phase reference.
        self.batch.process(&self.window, &mut self.column)?;
        for ((d, &c), &rot) in dst.iter_mut().zip(&self.column).zip(&self.rotors) {
            *d = c * rot;
        }
        Ok(())
    }
}

// ─── Trait impl ───────────────────────────────────────────────────────

impl<T, RP, V, ResP> CqtStreamPlan<T> for OmniCqtStreamPlan<T, RP, V, ResP>
where
    T: DspFloat + AddAssign + MulAssign,
    RP: DftR2cPlan<T>,
    V: VecOps<T>,
    ResP: ResamplePlan<T>,
{
    // Each method delegates to the inherent one.  Inherent methods take
    // precedence over trait methods in resolution, so these are not recursive.
    fn num_bins(&self) -> usize {
        self.num_bins()
    }

    fn bin_frequencies(&self) -> &[f64] {
        self.bin_frequencies()
    }

    fn max_output_columns(&self, input_len: usize) -> usize {
        self.max_output_columns(input_len)
    }

    fn process(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<usize> {
        self.process(input, out)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ─── Factory ──────────────────────────────────────────────────────────

impl<R, V> OmniCqt<R, V> {
    /// Create a streaming, newest-anchored CQT plan from a [`CqtStreamSpec`].
    ///
    /// Mirrors [`create_plan`](OmniCqt::create_plan): `resample_factory` (any
    /// `CreatePlan<ResampleSpec>`) builds the per-octave decimator sub-plan and
    /// is then dropped — the plan stores only the concrete decimator (option A,
    /// ADR-006 §2a).
    ///
    /// # Errors
    ///
    /// Returns an error if the per-bin newest-anchoring rotors cannot be
    /// represented in `T`, or if DFT / decimator sub-plan creation fails.
    #[allow(
        clippy::type_complexity,
        reason = "composite plan type names the routed decimator sub-plan; the \
                  dispatch layer aliases it via `type Plan`"
    )]
    #[allow(
        clippy::cast_precision_loss,
        reason = "FFT length is small enough that usize→f64 is exact"
    )]
    pub fn create_stream_plan<T, RF>(
        &self,
        spec: &CqtStreamSpec<T>,
        resample_factory: &RF,
    ) -> Result<OmniCqtStreamPlan<T, R::Plan, V, <RF as CreatePlan<ResampleSpec<T>>>::Plan>>
    where
        T: DspFloat + AddAssign + MulAssign,
        R: DftR2c<T>,
        V: VecOps<T>,
        RF: CreatePlan<ResampleSpec<T>>,
        <RF as CreatePlan<ResampleSpec<T>>>::Plan: ResamplePlan<T>,
    {
        let batch = self.create_plan(spec.spec(), resample_factory)?;
        let fft_length = batch.fft_length();
        let hop_length = spec.spec().hop_length().max(1);
        let num_bins = batch.num_bins();
        let sr = spec.spec().sample_rate();

        // Per-bin newest-anchoring rotor: exp(-j·2π·f_k·Δ/sr), Δ = fft_len−1
        // (the window-start → window-end shift).  Anchoring is a pure time
        // shift, so this rotates phase only and leaves magnitude unchanged.
        let delta = (fft_length.saturating_sub(1)) as f64;
        let mut rotors = Vec::with_capacity(num_bins);
        for &f_k in batch.bin_frequencies() {
            let angle = -std::f64::consts::TAU * f_k * delta / sr;
            let re = T::from(angle.cos())
                .ok_or_else(|| Error::Internal("cannot represent rotor re in T".into()))?;
            let im = T::from(angle.sin())
                .ok_or_else(|| Error::Internal("cannot represent rotor im in T".into()))?;
            rotors.push(Complex::new(re, im));
        }

        let zero = Complex::new(T::zero(), T::zero());
        Ok(OmniCqtStreamPlan {
            batch,
            vecops: self.vecops().clone(),
            rotors,
            fft_length,
            hop_length,
            num_bins,
            ring: vec![T::zero(); fft_length],
            ring_pos: 0,
            pending: 0,
            window: vec![T::zero(); fft_length],
            column: vec![zero; num_bins],
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use std::f64::consts::TAU;

    use num_complex::Complex;

    use super::{CqtStreamPlan, CqtStreamSpec, OmniCqtStreamPlan};
    use crate::design::cqt::{self, CqtSpec};
    use crate::modules::cqt::batch::{OmniCqt, OmniCqtPlan};
    use crate::modules::resample::{OmniResample, OmniResamplePlan};
    use crate::test_utils::{TestDftR2c, TestVecOps};
    use crate::traits::dft::DftR2c;
    use crate::types::Window;

    type TestStreamPlan = OmniCqtStreamPlan<
        f64,
        <TestDftR2c as DftR2c<f64>>::Plan,
        TestVecOps,
        OmniResamplePlan<f64, TestVecOps>,
    >;
    type TestBatchPlan = OmniCqtPlan<
        f64,
        <TestDftR2c as DftR2c<f64>>::Plan,
        TestVecOps,
        OmniResamplePlan<f64, TestVecOps>,
    >;

    fn factory() -> OmniCqt<TestDftR2c, TestVecOps> {
        OmniCqt::new(TestDftR2c, TestVecOps)
    }

    fn make_stream(spec: &CqtSpec<f64>) -> TestStreamPlan {
        factory()
            .create_stream_plan(
                &CqtStreamSpec::new(spec.clone()),
                &OmniResample::new(TestVecOps),
            )
            .expect("stream plan creation should succeed")
    }

    fn make_batch(spec: &CqtSpec<f64>) -> TestBatchPlan {
        factory()
            .create_plan(spec, &OmniResample::new(TestVecOps))
            .expect("batch plan creation should succeed")
    }

    /// Three octaves at 16 kHz — exercises the decimation chain.
    fn multi_octave_spec() -> CqtSpec<f64> {
        cqt::design(16000.0, 125.0, 1000.0, 12, &Window::Hann).expect("valid design")
    }

    // ── Accessors ──────────────────────────────────────────────────

    #[test]
    fn accessors_match_spec() {
        let spec = multi_octave_spec();
        let plan = make_stream(&spec);
        assert_eq!(plan.num_bins(), spec.num_bins(), "num_bins matches spec");
        assert_eq!(
            plan.fft_length(),
            spec.fft_length(),
            "fft_length matches spec"
        );
        assert_eq!(
            plan.hop_length(),
            spec.hop_length().max(1),
            "hop_length matches spec"
        );
        assert_eq!(
            plan.bin_frequencies().len(),
            spec.num_bins(),
            "one frequency per bin"
        );
    }

    // ── max_output_columns bounds the actual count ─────────────────

    #[test]
    fn max_output_columns_bounds_count() {
        let spec = multi_octave_spec();
        let mut plan = make_stream(&spec);
        let hop = plan.hop_length();
        let nb = plan.num_bins();

        for &len in &[0usize, 1, hop - 1, hop, hop + 1, 3 * hop + 2] {
            let bound = plan.max_output_columns(len);
            let mut out = vec![Complex::new(0.0, 0.0); bound * nb];
            let input = vec![0.0_f64; len];
            let produced = plan.process(&input, &mut out).expect("process");
            assert!(
                produced <= bound,
                "produced {produced} columns exceeds bound {bound} for len {len}"
            );
        }
    }

    // ── process emits exactly the hop-boundary columns ─────────────

    #[test]
    fn process_emits_hop_boundary_columns() {
        let spec = multi_octave_spec();
        let mut plan = make_stream(&spec);
        let hop = plan.hop_length();
        let nb = plan.num_bins();

        // Feed exactly N·hop samples → exactly N columns, in N separate calls.
        let n_cols = 5;
        let mut total = 0usize;
        for _ in 0..n_cols {
            let input = vec![0.5_f64; hop];
            let mut out = vec![Complex::new(0.0, 0.0); plan.max_output_columns(hop) * nb];
            let produced = plan.process(&input, &mut out).expect("process");
            assert_eq!(produced, 1, "one hop's worth of samples → one column");
            total += produced;
        }
        assert_eq!(total, n_cols, "five hops → five columns");
    }

    // ── Undersized output errors ───────────────────────────────────

    #[test]
    fn process_rejects_undersized_output() {
        let spec = multi_octave_spec();
        let mut plan = make_stream(&spec);
        let hop = plan.hop_length();
        let input = vec![0.0_f64; hop];
        let mut tiny = vec![Complex::new(0.0, 0.0); 1];
        assert!(
            plan.process(&input, &mut tiny).is_err(),
            "undersized output must error"
        );
    }

    // ── Newest-anchoring equivalence to the batch oracle ───────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn settled_column_matches_batch_oracle() {
        // The core 22a guarantee: each settled streaming column equals the batch
        // CQT of the equivalent newest-anchored window — magnitude exactly,
        // complex up to the per-bin newest-anchoring phase exp(-j·2π·f_k·Δ/sr).
        let spec = multi_octave_spec();
        let mut stream = make_stream(&spec);
        let batch = make_batch(&spec);
        let nb = stream.num_bins();
        let fft = stream.fft_length();
        let hop = stream.hop_length();
        let sr = spec.sample_rate();
        let delta = (fft - 1) as f64;

        // A broadband signal long enough that several columns are settled.
        let total = fft + 6 * hop;
        let freqs = spec.bins().iter().map(|b| b.frequency).collect::<Vec<_>>();
        let signal: Vec<f64> = (0..total)
            .map(|i| {
                freqs
                    .iter()
                    .map(|&f| (TAU * f * i as f64 / sr).sin())
                    .sum::<f64>()
            })
            .collect();

        let mut out = vec![Complex::new(0.0, 0.0); stream.max_output_columns(total) * nb];
        let columns = stream.process(&signal, &mut out).expect("stream process");
        assert!(columns >= 1, "expected at least one column, got {columns}");

        let mut checked = 0;
        for c in 0..columns {
            // Column c fires after (c+1)·hop samples → newest sample index
            // (c+1)·hop − 1; its window is the fft samples ending there.
            let now = (c + 1) * hop;
            if now < fft {
                continue; // not yet settled (window would be zero-padded)
            }
            let start = now - fft;
            let window = &signal[start..now];

            let mut oracle = vec![Complex::new(0.0, 0.0); nb];
            batch.process(window, &mut oracle).expect("batch oracle");

            let col = &out[c * nb..(c + 1) * nb];
            for (k, (s, o)) in col.iter().zip(&oracle).enumerate() {
                // Magnitude must match exactly (anchoring is a pure phase shift).
                assert!(
                    (s.norm() - o.norm()).abs() < 1e-9,
                    "column {c} bin {k}: |stream| {} vs |batch| {}",
                    s.norm(),
                    o.norm()
                );
                // Complex must match after applying the newest-anchoring phase.
                let angle = -TAU * freqs[k] * delta / sr;
                let rot = Complex::new(angle.cos(), angle.sin());
                let expected = o * rot;
                assert!(
                    (s - expected).norm() < 1e-9,
                    "column {c} bin {k}: stream {s} vs de-rotated batch {expected}"
                );
            }
            checked += 1;
        }
        assert!(
            checked >= 1,
            "expected at least one settled column to check"
        );
    }

    // ── reset restores initial state ───────────────────────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn reset_restores_initial_state() {
        let spec = multi_octave_spec();
        let mut plan = make_stream(&spec);
        let hop = plan.hop_length();
        let nb = plan.num_bins();
        let fft = plan.fft_length();

        // Feed enough to fully populate the ring and emit settled columns.
        let total = fft + 2 * hop;
        let signal: Vec<f64> = (0..total).map(|i| (i as f64 * 0.01).sin()).collect();
        let mut out_a = vec![Complex::new(0.0, 0.0); plan.max_output_columns(total) * nb];
        let cols_a = plan.process(&signal, &mut out_a).expect("first run");

        // Without reset, the ring carries history → a fresh feed differs.
        plan.reset();
        let mut out_b = vec![Complex::new(0.0, 0.0); plan.max_output_columns(total) * nb];
        let cols_b = plan.process(&signal, &mut out_b).expect("post-reset run");

        assert_eq!(cols_a, cols_b, "reset restores the hop phase");
        for (k, (a, b)) in out_a[..cols_a * nb].iter().zip(&out_b).enumerate() {
            assert!(
                (a - b).norm() < 1e-12,
                "index {k}: post-reset output differs ({a} vs {b})"
            );
        }
    }

    // ── process_magnitude agrees with process ──────────────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn process_magnitude_matches_process() {
        let spec = multi_octave_spec();
        let fft = spec.fft_length();
        let hop = spec.hop_length().max(1);
        let total = fft + 3 * hop;
        let signal: Vec<f64> = (0..total).map(|i| (i as f64 * 0.013).sin()).collect();

        let mut plan_c = make_stream(&spec);
        let nb = plan_c.num_bins();
        let mut complex_out = vec![Complex::new(0.0, 0.0); plan_c.max_output_columns(total) * nb];
        let cols_c = plan_c.process(&signal, &mut complex_out).expect("process");

        let mut plan_m = make_stream(&spec);
        let mut mag_out = vec![0.0_f64; plan_m.max_output_columns(total) * nb];
        let cols_m = plan_m
            .process_magnitude(&signal, &mut mag_out)
            .expect("process_magnitude");

        assert_eq!(cols_c, cols_m, "same column count");
        for (k, (&m, c)) in mag_out[..cols_m * nb].iter().zip(&complex_out).enumerate() {
            assert!(
                (m - c.norm()).abs() < 1e-12,
                "index {k}: magnitude {m} vs |complex| {}",
                c.norm()
            );
        }
    }

    // ── Trait drives the plan generically ──────────────────────────

    #[test]
    fn implements_stream_plan_trait() {
        fn drive<P: CqtStreamPlan<f64>>(plan: &mut P, input: &[f64]) -> usize {
            let need = plan.max_output_columns(input.len()) * plan.num_bins();
            let mut out = vec![Complex::new(0.0, 0.0); need];
            plan.process(input, &mut out).expect("trait process")
        }

        let spec = multi_octave_spec();
        let mut plan = make_stream(&spec);
        let hop = plan.hop_length();
        let input = vec![0.25_f64; 2 * hop];
        let produced = drive(&mut plan, &input);
        assert_eq!(produced, 2, "two hops via the trait → two columns");
    }

    // ── Debug formatting ───────────────────────────────────────────

    #[test]
    fn debug_format_is_readable() {
        let plan = make_stream(&multi_octave_spec());
        let debug = format!("{plan:?}");
        assert!(
            debug.contains("OmniCqtStreamPlan"),
            "debug should contain the type name"
        );
    }
}
