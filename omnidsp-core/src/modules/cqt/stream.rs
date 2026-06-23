// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Streaming, newest-anchored multirate CQT.
//!
//! [`OmniCqtProcessor`] is the streaming analogue of the batch
//! [`OmniCqtPlan`](super::batch::OmniCqtPlan): a stateful `&mut self` processor
//! that mirrors
//! [`ResampleProcessor`] — no
//! `Send + Sync`, a [`reset`](CqtProcessor::reset), and a variable-count
//! [`process`](CqtProcessor::process) that returns how many feature columns it
//! emitted (the column analogue of the resampler's sample count).  Feed any
//! chunk of samples; get back the `0..=max_output_columns` hop-boundary columns
//! the input crossed.
//!
//! # Newest-anchoring (the crux)
//!
//! The batch CQT anchors every bin's kernel at the **oldest** sample of its
//! octave frame (index 0), and slices each octave's frame from the *old* edge —
//! so even a treble bin's short window analyses signal from ~a whole `fft_length`
//! ago, and its output trails by the window length.
//!
//! Newest-anchoring relocates **each octave's analysis frame to end at "now"**
//! (its newest `fft_len_o` decimated samples) and places **each bin's kernel at
//! the frame end** (`kernel::Anchor::End`), so each bin analyses only
//! its newest `N_k` samples.  Onset latency then collapses to the Gabor floor
//! `Q/f`: the top octave looks at ~`Q/f_top` seconds (treble near-instant), the
//! bottom octave at ~`Q/f_bot` (bass inherently slow — physics, not anchoring).
//!
//! **This relocates *which samples each bin analyses* — it is not a pure phase
//! shift.**  For a transient/onset the newest- and oldest-anchored magnitudes
//! differ in *timing*, and that earlier response is the responsiveness win.  The
//! pure-phase relationship (magnitude identical, a per-bin linear phase) holds
//! **only for signals stationary across the frame** — a steady-state
//! cross-check, not the general law (see `check_cqt_stream`).
//!
//! # State: persistent continuous decimation
//!
//! 1. **Continuous decimators.**  One [`ResampleProcessor`] per octave transition
//!    (`o−1 → o`), advanced as samples arrive and **never reset per frame**.
//!    Each octave's decimated stream is produced once and flows continuously.
//! 2. **Per-octave rings.**  Each octave's decimated stream is kept in a ring so
//!    any frame ending at "now" can be re-analysed without re-decimating.
//! 3. **Newest-anchored placement with per-octave cadence.**  A column is pulled
//!    every top-octave hop; deep octaves update only at their (decimated) rate
//!    and **hold** their last value between updates.
//!
//! ## The newest-anchored oracle and the warm-up window
//!
//! `check_cqt_stream` verifies each settled column against an **independent**
//! newest-anchored reference: a decimation-free single-FFT CQT with end-placed
//! kernels over the newest `fft_length` samples (mirroring how the batch
//! path is validated against the single-FFT oracle).  The streaming path matches
//! it because:
//!
//! - **Group-delay-compensated frame end.**  Octave `o`'s frame ends at the
//!   decimated sample whose top-rate instant is "now", found by adding the
//!   accumulated decimator group delay (the band's `offset`) to `now >> o`.  The
//!   top octave (no decimation) matches the single-FFT reference complex; deeper
//!   octaves match in magnitude, with a small per-octave decimation-phase
//!   residual in the complex value (the same class of residual the multirate
//!   batch carries against its own single-FFT oracle).
//! - **Window taper + warm-up.**  The continuous decimator carries real history
//!   where a from-scratch decimation would see zeros; that difference lands at
//!   the frame's *old* edge, where the per-bin Hann window tapers to ≈ 0, so it
//!   is negligible once warmed up.  `check_cqt_stream` verifies only **settled**
//!   columns (`now ≥ 2·fft_length`, with "now" snapped to the deepest octave's
//!   grid).  A steady-state magnitude/phase cross-check against the oldest-
//!   anchored batch rides alongside it.

use std::f64::consts::TAU;
use std::ops::{AddAssign, MulAssign};

use num_complex::Complex;

use crate::create::CreateProc;
use crate::design::cqt::CqtSpec;
use crate::design::resample::ResampleSpec;
use crate::dispatch::Backend;
use crate::error::{Error, Result};
use crate::modules::cqt::batch::OmniCqt;
use crate::modules::cqt::kernel::{self, OctaveBand};
use crate::modules::resample::ResampleProcessor;
use crate::traits::dft::{DftR2c, DftR2cPlan};
use crate::traits::reconfigure::Reconfigure;
use crate::traits::vecops::VecOps;
use crate::types::DspFloat;
use crate::window::Window;

// ─── Processor trait ──────────────────────────────────────────────────

/// Execution object for a configured **streaming** Constant-Q Transform — a
/// stateful **Processor**.
///
/// The named processor trait for the streaming CQT, mirroring
/// [`ResampleProcessor`]: it is
/// mutable, carries **no** `Send + Sync` supertrait, exposes a
/// [`reset`](Self::reset), and has a variable-count [`process`](Self::process)
/// paired with a [`max_output_columns`](Self::max_output_columns) sizing method
/// and a [`finish`](Self::finish) end-of-stream flush.
///
/// Where a resample processor is a stream→stream *filter* (`T → T`), this is a
/// stream→feature-frame *analyzer* (`T → Complex<T>` column) emitted on the
/// top-octave hop clock — a different cadence than the input.
/// `max_output_columns` resolves that mismatch the way the resampler's
/// `max_output_len` does: feed any chunk, get the hop-boundary columns it
/// crossed.  Output is complex; the magnitude convenience
/// ([`process_magnitude`](Self::process_magnitude)) is on the trait too,
/// delegating to the concrete processor's efficient inherent implementation.
pub trait CqtProcessor<T> {
    /// Number of frequency bins per column.
    fn num_bins(&self) -> usize;

    /// Per-bin center frequencies in Hz (pinned low → high).
    fn bin_frequencies(&self) -> &[f64];

    /// Per-bin newest-anchored latency in **top-rate input samples** (pinned
    /// low → high, parallel to [`bin_frequencies`](Self::bin_frequencies)).
    ///
    /// The newest-anchored value at "now" reflects the signal at
    /// `now − latency`, where `latency = window_len/2 + decimation group delay`.
    /// Used for display latency compensation (de-warping a swept tone).
    fn bin_latencies(&self) -> &[f64];

    /// Upper bound on columns [`process`](Self::process) may emit for
    /// `input_len` new samples — size `out` to
    /// `max_output_columns(input_len) * num_bins`.
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

    /// Signal end-of-stream.  A column fires only on a full hop boundary, so a
    /// residual partial hop produces no column: `finish` writes nothing, returns
    /// `0`, and resets the processor.
    ///
    /// # Errors
    ///
    /// Infallible in practice; the `Result` keeps the Processor contract uniform.
    fn finish(&mut self, out: &mut [Complex<T>]) -> Result<usize>;

    /// Feed new samples and write the newest-anchored **magnitude** columns
    /// (`|CQT[k]|` per bin) whose hop boundaries the input crossed; returns the
    /// column count.  Same buffer contract as [`process`](Self::process): `out`
    /// must hold at least
    /// [`max_output_columns(input.len())`](Self::max_output_columns) columns,
    /// each [`num_bins`](Self::num_bins) wide.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if `out` is too short, or propagates
    /// execution failures.
    fn process_magnitude(&mut self, input: &[T], out: &mut [T]) -> Result<usize>;

    /// Reset all state (decimator delay lines, per-octave rings) without
    /// recreating the processor.
    fn reset(&mut self);
}

// ─── Per-octave ring ───────────────────────────────────────────────────

/// A fixed-capacity, global-index-addressed ring of one octave's decimated
/// stream.
///
/// Sample `g` (0-based global index since plan creation / last reset) lives at
/// `data[g % capacity]` while it is still retained — i.e. while
/// `count − capacity ≤ g < count`.  The capacity is sized so the ring always
/// retains a full analysis frame's lookback at a hop boundary (see
/// [`new_octave_rings`]).
struct OctaveRing<T> {
    /// Backing storage of length `capacity`.
    data: Vec<T>,
    /// Total samples ever pushed (the next sample's global index).
    count: usize,
}

impl<T: DspFloat> OctaveRing<T> {
    /// A ring of the given capacity (must be ≥ 1).
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::zero(); capacity.max(1)],
            count: 0,
        }
    }

    /// Append `samples` to the stream, advancing the global count.
    fn extend(&mut self, samples: &[T]) {
        let cap = self.data.len();
        for &s in samples {
            self.data[self.count % cap] = s;
            self.count += 1;
        }
    }

    /// The sample at global index `g`, or zero if `g` is past the newest sample
    /// produced (`g ≥ count`).  Indices below the retained window must not be
    /// requested — the ring is sized so they never are at a hop boundary.
    fn get_or_zero(&self, g: usize) -> T {
        if g >= self.count {
            T::zero()
        } else {
            self.data[g % self.data.len()]
        }
    }

    /// Clear the stream (zero the storage and reset the count).
    fn reset(&mut self) {
        for x in &mut self.data {
            *x = T::zero();
        }
        self.count = 0;
    }
}

/// Build the per-octave rings, sized so each retains a full analysis frame's
/// lookback at a hop boundary.
///
/// At a hop boundary the newest octave-`o` sample has global index
/// `count_o − 1 ≈ now / 2^o`, and the oldest the analysis reads is
/// `g0 = (now − fft_length) / 2^o + offset_o`.  The retained span is therefore
/// `count_o − g0 ≈ fft_length / 2^o − offset_o`.  We size the ring to
/// `⌈fft_length / 2^o⌉ + fft_len_o + offset_o + 4` — comfortably above that span
/// (the extra `fft_len_o` covers the window itself, `offset_o` the group-delay
/// skip, and `+4` absorbs decimator-phase/rounding slack), and independent of
/// the feed chunk size because columns are emitted at hop boundaries before any
/// further input is consumed.
fn new_octave_rings<T, RP, RS>(
    layout: &kernel::StreamOctaveLayout<T, RP, RS>,
    fft_length: usize,
) -> Vec<OctaveRing<T>>
where
    T: DspFloat,
{
    layout
        .octaves
        .iter()
        .enumerate()
        .map(|(o, band)| {
            let frame_span = fft_length.div_ceil(1usize << o);
            let capacity = frame_span + band.fft_len + band.offset + 4;
            OctaveRing::new(capacity)
        })
        .collect()
}

// ─── Processor ─────────────────────────────────────────────────────────

/// Streaming, newest-anchored multirate CQT processor.
///
/// Created by [`OmniCqt::create_proc`].  Stateful — it runs one continuous
/// ×2 decimator per octave transition, rings each octave's decimated stream, and
/// tracks the hop phase across calls so successive [`process`](Self::process)
/// calls form a continuous stream.  On each hop boundary it analyses the frame
/// ending at "now" directly from the rings (group-delay-compensated per octave,
/// no re-decimation), then applies the per-bin newest-anchoring phase.
///
/// Type parameters mirror [`OmniCqtPlan`](super::batch::OmniCqtPlan): `RP` the
/// per-octave [`DftR2cPlan`], `V` the [`VecOps`], `ResP` the routed decimator
/// [`ResampleProcessor`].
pub struct OmniCqtProcessor<T, RP, V, ResP> {
    /// The recipe this processor was built from, retained so a window swap can
    /// re-materialize every bin's kernel at its existing length (see
    /// [`Reconfigure<Window>`](crate::traits::reconfigure::Reconfigure)).  It
    /// carries the per-bin frequencies / kernel lengths, the sample rate, and the
    /// decimator quality — everything the kernel math needs except the window,
    /// which the swap supplies.
    spec: CqtSpec,
    /// Per-octave bands (kernels, r2c plan, group-delay offset), top → bottom.
    octaves: Vec<OctaveBand<T, RP>>,
    /// Continuous ×2 decimators, one per octave transition (`o−1 → o`); element
    /// `o−1` produces octave `o`.  Empty for a single-octave spec.
    decimators: Vec<ResP>,
    /// Per-octave ring of the decimated stream (`rings[0]` is the raw input).
    rings: Vec<OctaveRing<T>>,
    /// `VecOps` handle for the per-bin complex dot product and the magnitude
    /// convenience.
    vecops: V,
    /// Per-bin center frequencies in Hz, pinned low → high.
    bin_frequencies: Vec<f64>,
    /// Per-bin newest-anchored analysis-center latency in top-rate input
    /// samples, pinned low → high (parallel to `bin_frequencies`).  See
    /// [`bin_latencies`](Self::bin_latencies).
    bin_latencies: Vec<f64>,
    /// Top (input) sample rate in Hz — used to phase-compensate the per-octave
    /// decimator group delay the causal stream cannot reach (see
    /// [`emit_column`](Self::emit_column)).
    sample_rate: f64,
    /// Required analysis-window length (the spec's FFT length).
    fft_length: usize,
    /// Top-octave hop length — the column-emission clock.
    hop_length: usize,
    /// Deepest-octave sample stride `2^(O−1)`: "now" is quantised down to this
    /// grid so every octave's analysis window aligns to its decimated sample
    /// grid (see [`emit_column`](Self::emit_column)).
    align: usize,
    /// Number of frequency bins per column.
    num_bins: usize,
    /// Total top-rate samples fed since creation / last reset (== `rings[0]`'s
    /// global count; the newest sample is global index `n_total − 1`).
    n_total: usize,
    /// Samples accumulated toward the next hop boundary, in `[0, hop_length)`.
    pending: usize,
    /// Scratch r2c input segment (length `max_fft`).
    seg: Vec<T>,
    /// Scratch r2c output half-spectrum (length `max_fft/2 + 1`).
    half: Vec<Complex<T>>,
    /// Scratch decimator input buffer (one stage's per-piece input).
    dec_in: Vec<T>,
    /// Scratch decimator output buffer (one stage's per-piece output).
    dec_out: Vec<T>,
}

impl<T, RP, V, ResP> std::fmt::Debug for OmniCqtProcessor<T, RP, V, ResP> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OmniCqtProcessor")
            .field("num_bins", &self.num_bins)
            .field("fft_length", &self.fft_length)
            .field("hop_length", &self.hop_length)
            .finish_non_exhaustive()
    }
}

impl<T, RP, V, ResP> OmniCqtProcessor<T, RP, V, ResP>
where
    T: DspFloat + AddAssign + MulAssign,
    RP: DftR2cPlan<T>,
    V: VecOps<T>,
    ResP: ResampleProcessor<T>,
{
    /// Feed new samples; write the newest-anchored columns whose hop boundaries
    /// the input crossed and return the column count.  See
    /// [`CqtProcessor::process`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if `out` holds fewer than
    /// [`max_output_columns(input.len())`](Self::max_output_columns) columns
    /// (that many times [`num_bins`](Self::num_bins) elements), or propagates
    /// decimator / DFT execution failures.
    pub fn process(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<usize> {
        let need = self.max_output_columns(input.len()) * self.num_bins;
        if out.len() < need {
            return Err(Error::BufferMismatch {
                expected: need,
                actual: out.len(),
            });
        }

        let mut columns = 0usize;
        let mut pos = 0usize;
        while pos < input.len() {
            // Advance only up to the next hop boundary, so the rings hold
            // exactly up-to-"now" when a column fires; this bounds the ring
            // lookback to one frame regardless of the chunk size.
            let to_boundary = self.hop_length - self.pending;
            let take = to_boundary.min(input.len() - pos);
            self.feed(&input[pos..pos + take])?;
            pos += take;
            self.pending += take;
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
        let mut pos = 0usize;
        while pos < input.len() {
            let to_boundary = self.hop_length - self.pending;
            let take = to_boundary.min(input.len() - pos);
            self.feed(&input[pos..pos + take])?;
            pos += take;
            self.pending += take;
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

    /// Signal end-of-stream.  A column fires only when the input crosses a full
    /// hop boundary, so any residual `pending < hop_length` samples cannot
    /// complete one: `finish` writes nothing, returns `0`, and resets the
    /// processor.
    ///
    /// # Errors
    ///
    /// Infallible in practice; the `Result` keeps the Processor contract uniform.
    pub fn finish(&mut self, _out: &mut [Complex<T>]) -> Result<usize> {
        self.reset();
        Ok(0)
    }

    /// One-shot convenience: feed a complete `input` on a fresh stream —
    /// `reset`, then `process(input)`, then `finish` (which adds no column).
    /// Returns the column count and leaves the processor clean.  `out` must hold
    /// at least [`max_output_columns(input.len())`](Self::max_output_columns)
    /// columns, each [`num_bins`](Self::num_bins) wide.
    ///
    /// # Errors
    ///
    /// Returns an error if `out` is too short, or if execution fails.
    pub fn execute(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<usize> {
        self.reset();
        let n = self.process(input, out)?;
        let extra = self.finish(&mut out[n * self.num_bins..])?;
        Ok(n + extra)
    }

    /// Reset all state: decimator delay lines, per-octave rings, sample count,
    /// and hop phase.  Does not recreate the plan.
    pub fn reset(&mut self) {
        for d in &mut self.decimators {
            d.reset();
        }
        for r in &mut self.rings {
            r.reset();
        }
        self.n_total = 0;
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
        &self.bin_frequencies
    }

    /// Per-bin newest-anchored latency in **top-rate input samples** (pinned
    /// low → high, parallel to [`bin_frequencies`](Self::bin_frequencies)).
    ///
    /// The streaming CQT is newest-anchored: each bin's analysis window ends at
    /// "now", so a bin's value at "now" actually reflects the signal at
    /// `now − latency`.  The latency is the analysis-center delay
    /// `window_len/2 + decimation group delay` — the smooth window-center bow
    /// (`∝ 1/f`) plus the per-octave cascaded-decimator group delay (the
    /// staircase) — projected to top-rate input samples.  It decreases with
    /// frequency (treble near-instant, deep bass slow).
    ///
    /// For display latency compensation: shifting each bin's history left by its
    /// latency de-warps an exponential sweep into a straight diagonal while
    /// keeping the live edge newest-anchored (treble snaps, bass blooms).
    #[must_use]
    pub fn bin_latencies(&self) -> &[f64] {
        &self.bin_latencies
    }

    /// The deepest-octave sample stride `2^(O−1)` "now" is quantised to.
    ///
    /// Test-only accessor (not part of the locked 22a surface); the equivalence
    /// tests need it to snap the oracle window to the same instant the plan
    /// analyses.
    #[cfg(test)]
    pub(crate) const fn align(&self) -> usize {
        self.align
    }

    /// Advance every octave's continuous decimator and ring by one piece of new
    /// top-rate samples (a run no longer than the gap to the next hop boundary).
    fn feed(&mut self, piece: &[T]) -> Result<()> {
        if piece.is_empty() {
            return Ok(());
        }
        // Octave 0 is the raw input.
        self.rings[0].extend(piece);
        self.n_total += piece.len();

        // Cascade ×2 decimation: octave o's newly produced samples feed the
        // o → o+1 decimator.  `dec_in` holds the current stage's new samples.
        let n_octaves = self.octaves.len();
        self.dec_in[..piece.len()].copy_from_slice(piece);
        let mut in_len = piece.len();
        for o in 1..n_octaves {
            let dec = &mut self.decimators[o - 1];
            let max_out = dec.max_output_len(in_len);
            if max_out > self.dec_out.len() {
                // Defensive: `next_cap` bounds a full-frame piece, and pieces
                // never exceed one frame, so this is unreachable in practice.
                self.dec_out.resize(max_out, T::zero());
            }
            let produced = dec.process(&self.dec_in[..in_len], &mut self.dec_out[..max_out])?;
            self.rings[o].extend(&self.dec_out[..produced]);
            self.dec_in[..produced].copy_from_slice(&self.dec_out[..produced]);
            in_len = produced;
        }
        Ok(())
    }

    /// Analyse each octave's frame **ending at "now"** (`n_total`) directly from
    /// the rings, with each bin's end-anchored kernel, and write the column to
    /// `dst`.
    ///
    /// This is the newest-anchoring crux.  For octave `o` the frame
    /// is the `fft_len_o` decimated samples whose newest sample corresponds to
    /// the top-rate instant "now": it **ends** at the group-delay-compensated
    /// decimated index
    ///
    /// ```text
    /// g_end = (now / 2^o) + offset_o          (one past the newest sample)
    /// g_start = g_end − fft_len_o             (oldest sample of the frame)
    /// ```
    ///
    /// The kernels are placed at the frame *end* (`kernel::Anchor::End`), so each
    /// bin analyses the newest `N_k` decimated samples of this frame — its analysis
    /// window ends at the newest produced sample and its onset latency collapses to
    /// `Q/f`.  The kernel placement carries the per-bin newest-anchoring phase, so
    /// the complex coefficient is referenced to that frame end directly.
    ///
    /// A deep octave's newest produced sample lags "now" by the decimator group
    /// delay (`offset` decimated samples the causal continuous decimator has not
    /// emitted yet).  A small per-bin **phase advance** `exp(+j·2π·f_k·Δ/rate_o)`
    /// (`Δ` = the lag, `rate_o` = `sr/2^o`) re-references each deep bin to "now" —
    /// the correct realisation of the group-delay compensation the `+ offset`
    /// frame-end projection used to attempt by reading non-existent (zero) samples
    /// (the deep-octave skirt).
    ///
    /// Contrast the batch path, whose frame *starts* at the (group-delay-skipped)
    /// frame origin with kernels at index 0 — analysing the **oldest** samples,
    /// the window-bound latency this fix removes.
    #[allow(
        clippy::cast_precision_loss,
        reason = "offset and octave indices are small non-negative values exact in f64"
    )]
    fn emit_column(&mut self, dst: &mut [Complex<T>]) -> Result<()> {
        // Quantise "now" down to the deepest octave's sample grid so `now` is a
        // multiple of every 2^o (the batch's integer alignment), making each
        // octave's `now >> o` an exact decimated index.  The drop is at most
        // `align − 1` top-rate samples — below the deepest octave's own update
        // period, so it costs no real responsiveness; deep octaves can only
        // update on this grid anyway.
        let now = self.n_total - (self.n_total % self.align);

        for (o, band) in self.octaves.iter().enumerate() {
            let fft_len = band.fft_len;
            let half_len = fft_len / 2 + 1;
            let ring = &self.rings[o];

            // The decimated index one past this octave's newest sample whose
            // top-rate instant is "now".  `now >> o` is the batch's exact integer
            // alignment of "now" into this octave's sample grid (`now` is a
            // multiple of 2^o by the grid snap above).
            //
            // The `+ band.offset` group-delay projection the batch path applies to
            // *skip* the decimator's start-up transient from the frame's **old**
            // edge must NOT extend the frame's **new** edge here: the continuous
            // decimator has produced only `ring.count == now >> o` samples (it lags
            // the input by its group delay), so `(now >> o) + offset` points
            // `offset` samples past the newest sample that exists.  With the
            // newest-anchored kernel sitting on the frame end, those non-existent
            // samples read as zero and truncate the analysis window by `offset`,
            // broadening the deepest octaves' bins (the deep-octave skirt: ticket
            // 26).  Clamp the frame end to the newest produced sample so the
            // end-anchored kernel analyses real signal, not a group-delay phantom;
            // the `lag` it falls short of "now" is then compensated by a per-bin
            // phase advance below.
            let want_end = (now >> o) + band.offset;
            let g_end = want_end.min(ring.count);
            let lag = want_end - g_end; // decimated samples short of "now"
            // The frame ends at `g_end`; gather its `fft_len_o` samples, zero-
            // padding any below the ring's origin (warm-up only — settled frames
            // are fully retained, see `new_octave_rings`).
            let g_start = g_end.saturating_sub(fft_len);

            for (i, s) in self.seg[..fft_len].iter_mut().enumerate() {
                *s = ring.get_or_zero(g_start + i);
            }

            if !band.kernels.is_empty() {
                band.r2c
                    .execute(&mut self.seg[..fft_len], &mut self.half[..half_len])?;
                // Re-reference each bin to "now": the frame ends `lag` decimated
                // samples (at rate `sr/2^o`) before "now", so a tone at `f_k`
                // accrues a phase lag `2π·f_k·lag/rate_o`; advance it back.
                let rate_o = self.sample_rate / (1usize << o) as f64;
                for (j, kernel) in band.kernels.iter().enumerate() {
                    let coeff = self.vecops.cdot(&self.half[..half_len], kernel)?;
                    let out = band.out_start + j;
                    dst[out] = if lag == 0 {
                        coeff
                    } else {
                        let phase = TAU * self.bin_frequencies[out] * lag as f64 / rate_o;
                        let rotor = Complex::new(
                            T::from(phase.cos()).unwrap_or_else(T::zero),
                            T::from(phase.sin()).unwrap_or_else(T::zero),
                        );
                        coeff * rotor
                    };
                }
            }
        }
        Ok(())
    }
}

// ─── Trait impl ───────────────────────────────────────────────────────

impl<T, RP, V, ResP> CqtProcessor<T> for OmniCqtProcessor<T, RP, V, ResP>
where
    T: DspFloat + AddAssign + MulAssign,
    RP: DftR2cPlan<T>,
    V: VecOps<T>,
    ResP: ResampleProcessor<T>,
{
    // Each method delegates to the inherent one.  Inherent methods take
    // precedence over trait methods in resolution, so these are not recursive.
    fn num_bins(&self) -> usize {
        self.num_bins()
    }

    fn bin_frequencies(&self) -> &[f64] {
        self.bin_frequencies()
    }

    fn bin_latencies(&self) -> &[f64] {
        self.bin_latencies()
    }

    fn max_output_columns(&self, input_len: usize) -> usize {
        self.max_output_columns(input_len)
    }

    fn process(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<usize> {
        self.process(input, out)
    }

    fn finish(&mut self, out: &mut [Complex<T>]) -> Result<usize> {
        self.finish(out)
    }

    fn process_magnitude(&mut self, input: &[T], out: &mut [T]) -> Result<usize> {
        self.process_magnitude(input, out)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ─── Reconfigure ──────────────────────────────────────────────────────

impl<T, RP, V, ResP> Reconfigure<Window> for OmniCqtProcessor<T, RP, V, ResP>
where
    T: DspFloat + AddAssign + MulAssign,
{
    /// Swap the analysis window in place, re-materializing every bin's
    /// frequency-domain kernel from the new window at its existing length, and
    /// preserving all stream state (decimators, per-octave rings, hop phase).
    ///
    /// The window is **orthogonal** to the kernel sizes: a different window
    /// changes only the kernel coefficient *values*, never any `kernel_len`,
    /// `fft_len`, octave partition, or decimator.  So this never changes the
    /// layout and never returns
    /// [`StructuralMismatch`](crate::error::Error::StructuralMismatch) — the
    /// running stream simply continues analysing its history through the new
    /// window (the visualiser's glitch-free window swap).  The kernels are
    /// materialized in f64 and cast to `T` once, exactly as `create_proc` does.
    fn reconfigure(&mut self, window: &Window) -> Result<()> {
        kernel::rematerialize_kernels(&mut self.octaves, &self.spec, window)?;
        // Record the new window so a subsequent swap (or any future recipe read)
        // reflects the current state.
        self.spec = self.spec.clone().with_window(*window);
        Ok(())
    }
}

// ─── Factory ──────────────────────────────────────────────────────────

impl<R, V> OmniCqt<R, V> {
    /// Create a streaming, newest-anchored CQT processor from a [`CqtSpec`].
    ///
    /// The streaming peer of [`create_plan`](OmniCqt::create_plan) over the
    /// **same** spec: `resample_factory` (any `CreateProc<ResampleSpec>` that is
    /// also a [`Backend<T>`](crate::dispatch::Backend)) builds one continuous
    /// decimator sub-processor per octave transition and is then dropped — the
    /// processor stores only the concrete decimators (option A).
    ///
    /// # Errors
    ///
    /// Returns an error if DFT / decimator sub-plan creation fails.
    #[allow(
        clippy::type_complexity,
        reason = "composite processor type names the routed decimator sub-processor; \
                  the dispatch layer aliases it via `type Proc`"
    )]
    pub fn create_proc<T, RF>(
        &self,
        spec: &CqtSpec,
        resample_factory: &RF,
    ) -> Result<OmniCqtProcessor<T, R::Plan, V, <RF as CreateProc<ResampleSpec>>::Proc<T>>>
    where
        T: DspFloat + AddAssign + MulAssign,
        R: DftR2c<T>,
        V: VecOps<T>,
        RF: CreateProc<ResampleSpec> + Backend<T>,
        <RF as CreateProc<ResampleSpec>>::Proc<T>: ResampleProcessor<T>,
    {
        let cqt_spec = spec;
        let layout =
            kernel::build_octaves_streaming::<T, R, RF>(self.dftr2c(), cqt_spec, resample_factory)?;

        let fft_length = layout.fft_length;
        let hop_length = cqt_spec.hop_length().max(1);
        let num_bins = layout.num_bins;

        // The per-bin newest-anchoring phase is carried by the end-anchored
        // kernel placement ([`Anchor::End`], built in `build_octaves_streaming`),
        // so each octave's frame is gathered to end at its newest produced sample
        // and its end-placed kernels reference that sample directly.  A deep
        // octave's newest produced sample lags "now" by the decimator group delay
        // (`offset` decimated samples); `emit_column` phase-compensates that lag
        // per bin so the deep octaves stay referenced to "now".

        let rings = new_octave_rings(&layout, fft_length);

        // "now" is quantised down to the deepest octave's sample stride
        // (2^(O−1)) so every octave's analysis frame ends on a multiple of
        // 2^o — the batch path's exact integer alignment.
        let n_octaves = layout.octaves.len();
        let align = 1usize << n_octaves.saturating_sub(1);

        let zero = Complex::new(T::zero(), T::zero());
        Ok(OmniCqtProcessor {
            spec: cqt_spec.clone(),
            octaves: layout.octaves,
            decimators: layout.decimators,
            rings,
            vecops: self.vecops().clone(),
            bin_frequencies: layout.bin_frequencies,
            bin_latencies: layout.bin_latencies,
            sample_rate: cqt_spec.sample_rate(),
            fft_length,
            hop_length,
            align,
            num_bins,
            n_total: 0,
            pending: 0,
            seg: vec![T::zero(); layout.max_fft],
            half: vec![zero; layout.max_fft / 2 + 1],
            dec_in: vec![T::zero(); fft_length.max(1)],
            dec_out: vec![T::zero(); layout.next_cap.max(1)],
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use std::f64::consts::TAU;

    use num_complex::Complex;

    use super::{CqtProcessor, OmniCqtProcessor};
    use crate::design::cqt::{self, CqtSpec};
    use crate::modules::cqt::batch::{OmniCqt, OmniCqtPlan};
    use crate::modules::resample::OmniResampleProcessor;
    use crate::test_utils::{TestBackend, TestDftR2c, TestVecOps};
    use crate::traits::dft::{DftNorm, DftR2c, DftR2cPlan, DftR2cSpec};
    use crate::traits::vecops::VecOps;
    use crate::window::Window;

    type TestStreamPlan = OmniCqtProcessor<
        f64,
        <TestDftR2c as DftR2c<f64>>::Plan,
        TestVecOps,
        OmniResampleProcessor<f64>,
    >;
    type TestBatchPlan =
        OmniCqtPlan<f64, <TestDftR2c as DftR2c<f64>>::Plan, TestVecOps, OmniResampleProcessor<f64>>;

    fn factory() -> OmniCqt<TestDftR2c, TestVecOps> {
        OmniCqt::new(TestDftR2c, TestVecOps)
    }

    fn make_stream(spec: &CqtSpec) -> TestStreamPlan {
        factory()
            .create_proc(spec, &TestBackend)
            .expect("stream processor creation should succeed")
    }

    fn make_batch(spec: &CqtSpec) -> TestBatchPlan {
        factory()
            .create_plan(spec, &TestBackend)
            .expect("batch plan creation should succeed")
    }

    /// Three octaves at 16 kHz — exercises the continuous decimation chain.
    fn multi_octave_spec() -> CqtSpec {
        cqt::design(16000.0, 125.0, 1000.0, 12, &Window::Hann).expect("valid design")
    }

    // ── Independent newest-anchored single-FFT reference (the oracle) ──
    //
    // A decimation-free single-FFT CQT with **end-placed** kernels over the
    // newest `fft_length` samples — the same anchoring the streaming plan
    // produces, computed without the streaming plan, its rings, or its
    // decimators (mirroring how the batch path is validated against the
    // single-FFT oracle).  Each bin's full-rate kernel `(1/N_k)·w[n]·
    // exp(+j·2π·f_k·n/sr)` is placed at indices `fft_length−N_k … fft_length−1`
    // of an `fft_length`-point r2c frame, so the bin analyses the newest `N_k`
    // samples and its analysis ends at the window's last sample ("now").

    /// Conjugated, scaled, end-placed half-spectrum kernels (one per bin),
    /// independent of the streaming plan.
    struct NewestRef {
        kernels: Vec<Vec<Complex<f64>>>,
        plan: <TestDftR2c as DftR2c<f64>>::Plan,
        fft_length: usize,
    }

    impl NewestRef {
        #[allow(
            clippy::cast_precision_loss,
            reason = "kernel and FFT lengths are small enough for f64"
        )]
        fn new(spec: &CqtSpec) -> Self {
            let fft_length = spec.fft_length();
            let sr = spec.sample_rate();
            let half_len = fft_length / 2 + 1;
            let r2c_spec =
                DftR2cSpec::new(fft_length, DftNorm::None).expect("valid r2c reference spec");
            let plan = TestDftR2c
                .create_plan(&r2c_spec)
                .expect("reference r2c plan");

            let inv_n = 1.0 / fft_length as f64;
            let mut kernels = Vec::with_capacity(spec.num_bins());
            for bin in spec.bins() {
                // Materialize the bin's window in f64 from the spec's recipe at its
                // own kernel length — the reference is decimation-free, so it uses
                // the full-rate window directly.
                let window = spec
                    .window()
                    .coefficients::<f64>(bin.kernel_len)
                    .expect("reference window coefficients");
                let nk = bin.kernel_len;
                let kernel_start = fft_length - nk; // end placement
                let inv_nk = 1.0 / nk as f64;
                let mut k = vec![Complex::new(0.0, 0.0); half_len];
                for (m, slot) in k.iter_mut().enumerate() {
                    let mut acc = Complex::new(0.0, 0.0);
                    for (n, &wn) in window.iter().enumerate() {
                        let p = (kernel_start + n) as f64;
                        let angle = TAU
                            * (bin.frequency * n as f64 / sr - (m as f64 * p) / fft_length as f64);
                        let amp = wn * inv_nk;
                        acc += Complex::new(amp * angle.cos(), amp * angle.sin());
                    }
                    // Stored kernel = conj(DFT)/fft_length, matching the plan.
                    *slot = Complex::new(acc.re * inv_n, -(acc.im * inv_n));
                }
                kernels.push(k);
            }

            Self {
                kernels,
                plan,
                fft_length,
            }
        }

        /// The newest-anchored CQT column of `window` (length `fft_length`).
        fn column(&self, window: &[f64]) -> Vec<Complex<f64>> {
            assert_eq!(
                window.len(),
                self.fft_length,
                "reference window must be one fft_length"
            );
            let half_len = self.fft_length / 2 + 1;
            let mut seg = window.to_vec();
            let mut half = vec![Complex::new(0.0, 0.0); half_len];
            self.plan
                .execute(&mut seg, &mut half)
                .expect("reference r2c process");
            let ops = TestVecOps;
            self.kernels
                .iter()
                .map(|k| ops.cdot(&half, k).expect("reference cdot"))
                .collect()
        }
    }

    /// Settled equivalence tolerance for the magnitude / top-octave-complex
    /// checks against the independent newest-anchored reference (mixed abs/rel).
    ///
    /// The streaming path is multirate; the reference is decimation-free.  For
    /// the **top octave** (no decimation: identical frame end, exact kernel
    /// length) they agree to the half-spectrum-leakage floor, so the complex
    /// values match within this bound.  For **deeper octaves** the half-band
    /// decimator introduces a per-octave phase/group-delay residual (the same
    /// class the multirate batch carries against its single-FFT oracle), so
    /// only the **magnitude** is asserted at this bound there; the full complex
    /// residual is measured and bounded separately by [`DEEP_COMPLEX_TOL`].
    const REF_TOL: f64 = 1e-3;

    /// Magnitude tolerance for deeper-octave bins against the decimation-free
    /// reference.  The multirate path passes those bins through 1–2 windowed
    /// Kaiser ×2 decimation stages (asymmetric: passband at the next
    /// octave's top, stopband planted at `fs/4`) whose passband response is not
    /// perfectly flat, so its magnitude would droop toward the band edge — except
    /// the per-bin gain compensation (retained) bakes the analytic
    /// inverse `1/G_k` of that cascaded passband response into the kernel,
    /// flattening the octave seam.  **Before** compensation the deepest bin
    /// drooped to ≈ 0.24 (relative); **after**, the seam is flat.
    ///
    /// A deep octave's newest produced sample lags "now" by the decimator group
    /// delay, so its analysis window ends `offset·2^o` top-rate samples before the
    /// decimation-free reference's window does.  `emit_column`
    /// phase-compensates that lag (keeping the *complex* residual at the
    /// decimator-phase floor, [`DEEP_COMPLEX_TOL`]), but the window covers a
    /// slightly different span of the signal, so a small **magnitude** residual
    /// remains — the genuine, irreducible difference between a causal multirate
    /// path and a decimation-free oracle that sees `offset·2^o` more recent
    /// samples.  **Measured** worst-case ≈ 3.6e-2 (the deepest bin, largest lag).
    /// The top octave (no decimation, no lag) still matches to ≈ 2.7e-5, the proof
    /// this is a real group-delay residual, not a loosening.
    const DEEP_MAG_TOL: f64 = 5e-2;

    /// Complex tolerance for deeper-octave bins against the decimation-free
    /// reference.  The Kaiser ×2 decimator is not phase-exact, so a
    /// decimation-free reference and the multirate streaming path differ by a
    /// per-octave residual in the complex value.  Gain compensation corrects the
    /// **magnitude** droop (see [`DEEP_MAG_TOL`]) but is orthogonal to the
    /// decimator's group-delay *phase*; with `emit_column`'s per-bin group-delay
    /// phase compensation re-referencing each deep bin to "now", the
    /// remaining complex residual stays at the decimator-phase floor.
    /// **Measured** worst-case here ≈ 7.0e-2 (relative, asymmetric windowed Kaiser
    /// decimator).  Not a loosening: the streaming path is genuinely
    /// multirate and the reference is genuinely decimation-free, so this residual
    /// is real.
    const DEEP_COMPLEX_TOL: f64 = 8e-2;

    /// The deepest-octave sample stride `2^(O−1)` "now" is quantised to.
    fn align_of(plan: &TestStreamPlan) -> usize {
        plan.align()
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
        // 125–1000 Hz spans ≥3 octaves, so "now" snaps to a stride ≥ 2^2 = 4.
        assert!(plan.align() >= 4, "≥3 octaves → align stride ≥4");
        assert_eq!(
            plan.bin_frequencies().len(),
            spec.num_bins(),
            "one frequency per bin"
        );
    }

    // ── Per-bin newest-anchored latency ───────────────────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "bin counts and window lengths are small enough for f64"
    )]
    fn bin_latencies_decrease_with_frequency() {
        // The demo regime (48 kHz, 110 Hz … 16 kHz, 12 bins/oct, Hann): the
        // per-bin newest-anchored latency must be one value per bin, all positive,
        // and monotonically *decreasing* with frequency (low bins look further
        // back — `window/2 + decimation group delay`, both ∝ deeper octave).
        let spec = cqt::design(48000.0, 110.0, 16000.0, 12, &Window::Hann)
            .expect("valid demo-regime design");
        let plan = make_stream(&spec);
        let nb = plan.num_bins();
        let lat = plan.bin_latencies();
        let sr = spec.sample_rate();

        assert_eq!(lat.len(), nb, "one latency per bin");
        for (k, &l) in lat.iter().enumerate() {
            assert!(l > 0.0, "bin {k} latency {l} must be positive");
        }
        // bins ascend in frequency low → high, so latency must descend.
        for k in 1..nb {
            assert!(
                lat[k] <= lat[k - 1],
                "latency must not increase with frequency: bin {k} ({}) > bin {} ({})",
                lat[k],
                k - 1,
                lat[k - 1]
            );
        }

        // The deepest bin (≈ 110 Hz) sits in a sane display-latency range: its
        // window-center delay alone is window_len/2 / sr, and the decimation
        // group delay adds the staircase on top.  Measured ≈ 209 ms; bound it
        // generously (150–260 ms) so a regression in either term shows.
        let deep_ms = lat[0] / sr * 1000.0;
        assert!(
            (150.0..=260.0).contains(&deep_ms),
            "deepest (≈110 Hz) bin latency {deep_ms:.1} ms out of the 150–260 ms band"
        );
        // The window/2 term is the floor; the full latency must exceed it (the
        // decimation group delay is strictly positive for any decimated octave).
        let window_half = spec.bins()[0].kernel_len as f64 / 2.0;
        assert!(
            lat[0] > window_half,
            "deepest latency {} must exceed its window-center floor {window_half}",
            lat[0]
        );

        // The top bin (≈ 16 kHz, top octave, no decimation) is just its
        // window-center delay — far smaller, only a few ms.
        let top_ms = lat[nb - 1] / sr * 1000.0;
        assert!(
            top_ms < 5.0,
            "top (≈16 kHz) bin latency {top_ms:.1} ms should be near-instant (< 5 ms)"
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

    // ── Chunked feed equals single-shot feed ───────────────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn chunked_feed_matches_single_shot() {
        // The continuous decimators + rings must be chunk-size-independent: feed
        // the same signal in one shot and in ragged chunks; the columns agree.
        let spec = multi_octave_spec();
        let fft = spec.fft_length();
        let hop = spec.hop_length().max(1);
        let nb = spec.num_bins();
        let total = fft + 30 * hop;
        let signal: Vec<f64> = (0..total).map(|i| (i as f64 * 0.017).sin()).collect();

        let mut single = make_stream(&spec);
        let mut out_single = vec![Complex::new(0.0, 0.0); single.max_output_columns(total) * nb];
        let cols_single = single
            .process(&signal, &mut out_single)
            .expect("single-shot");

        let mut chunked = make_stream(&spec);
        let mut out_chunked = Vec::new();
        let chunks = [1usize, 7, hop, hop + 3, 2 * hop, 50, 200, 13];
        let mut pos = 0;
        let mut ci = 0;
        while pos < total {
            let len = chunks[ci % chunks.len()].min(total - pos);
            ci += 1;
            let mut buf = vec![Complex::new(0.0, 0.0); chunked.max_output_columns(len) * nb];
            let produced = chunked
                .process(&signal[pos..pos + len], &mut buf)
                .expect("chunk");
            out_chunked.extend_from_slice(&buf[..produced * nb]);
            pos += len;
        }

        assert_eq!(
            out_chunked.len(),
            cols_single * nb,
            "chunked and single-shot produce the same column count"
        );
        for (k, (a, b)) in out_single[..cols_single * nb]
            .iter()
            .zip(&out_chunked)
            .enumerate()
        {
            assert!(
                (a - b).norm() < 1e-12,
                "index {k}: chunked {b} vs single-shot {a}"
            );
        }
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

    // ── Newest-anchoring equivalence to the independent reference ──

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn settled_column_matches_newest_reference() {
        // The 22b guarantee against real internals: each settled streaming
        // column equals the **independent** newest-anchored single-FFT reference
        // (`NewestRef`) of the newest `fft_length` samples — magnitude to
        // `REF_TOL`, and complex to `REF_TOL` for the top octave (no decimation),
        // with deeper octaves carrying a small measured decimation-phase residual
        // in the complex value (bounded by `DEEP_COMPLEX_TOL`).  "now" is
        // quantised to the deepest octave's grid (`align`), so the reference
        // window is `signal[now-fft..now]` at the snapped hop instant.
        let spec = multi_octave_spec();
        let mut stream = make_stream(&spec);
        let reference = NewestRef::new(&spec);
        let nb = stream.num_bins();
        let fft = stream.fft_length();
        let hop = stream.hop_length();
        let align = align_of(&stream);
        let sr = spec.sample_rate();
        // The top-octave bins (octave 0) are the high-frequency tail of the bins
        // array; they share the reference's exact frame end and kernel length.
        let top_octave_lo = nb - 12.min(nb);

        // Broadband signal long enough to settle several columns past warm-up.
        let total = 4 * fft;
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

        let mut checked = 0;
        let mut worst_top_complex = 0.0_f64;
        let mut worst_top_mag = 0.0_f64;
        let mut worst_deep_complex = 0.0_f64;
        let mut worst_deep_mag = 0.0_f64;
        for c in 0..columns {
            // Snap the hop instant to the deepest octave's grid, exactly as
            // `emit_column` does, then require a full settled frame.
            let raw = (c + 1) * hop;
            let now = raw - (raw % align);
            if now < 2 * fft {
                continue; // warm-up: continuous decimators + first frame
            }
            let window = &signal[now - fft..now];
            let oracle = reference.column(window);

            let col = &out[c * nb..(c + 1) * nb];
            for (k, (s, o)) in col.iter().zip(&oracle).enumerate() {
                let mag = (s.norm() - o.norm()).abs() / (1.0 + o.norm());
                let cplx = (s - o).norm() / (1.0 + o.norm());
                if k >= top_octave_lo {
                    // Top octave: no decimation → magnitude *and* complex match
                    // the reference to the half-spectrum-leakage floor.
                    worst_top_mag = worst_top_mag.max(mag);
                    worst_top_complex = worst_top_complex.max(cplx);
                    assert!(
                        mag < REF_TOL,
                        "top-octave column {c} bin {k}: |stream| {} vs |reference| {}",
                        s.norm(),
                        o.norm()
                    );
                    assert!(
                        cplx < REF_TOL,
                        "top-octave column {c} bin {k}: stream {s} vs reference {o}"
                    );
                } else {
                    // Deeper octaves: the multirate path differs from the
                    // decimation-free reference by the half-band decimator's
                    // (imperfect) response — a per-octave residual in *both*
                    // magnitude and complex.  Magnitude is still close; the
                    // complex phase residual is larger.  Both are measured and
                    // bounded (see DEEP_MAG_TOL / DEEP_COMPLEX_TOL).
                    worst_deep_mag = worst_deep_mag.max(mag);
                    worst_deep_complex = worst_deep_complex.max(cplx);
                    assert!(
                        mag < DEEP_MAG_TOL,
                        "deep column {c} bin {k}: |stream| {} vs |reference| {} (mag residual {mag})",
                        s.norm(),
                        o.norm()
                    );
                    assert!(
                        cplx < DEEP_COMPLEX_TOL,
                        "deep column {c} bin {k}: stream {s} vs reference {o} (residual {cplx})"
                    );
                }
            }
            checked += 1;
        }
        assert!(
            checked >= 1,
            "expected at least one settled column to check, got {checked}"
        );
        // Surface the measured residuals so a regression is visible in -- --nocapture.
        report_residuals(
            worst_top_mag,
            worst_top_complex,
            worst_deep_mag,
            worst_deep_complex,
        );
    }

    /// Demo-regime streaming guard: a pure tone in the **top** octave
    /// must not leak into the **deep** octaves.
    ///
    /// `multi_octave_spec` keeps `f_max` ≈ 0.06·Nyquist (far below the decimator
    /// transition) and so never exercises octave-aliasing; this uses `f_max` ≈
    /// 0.67·Nyquist like the WASM demo — the regime where the earlier equiripple
    /// half-band produced full-amplitude octave-aliasing phantoms (an 11 kHz tone
    /// folding down to ≈ 1 kHz, four octaves below). The windowed Kaiser
    /// decimator has the rolloff to keep the deep octaves clean.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn demo_regime_top_tone_does_not_leak_into_deep_octaves() {
        let spec = cqt::design(48000.0, 250.0, 16000.0, 12, &Window::Hann)
            .expect("valid demo-like design");
        let mut stream = make_stream(&spec);
        let nb = stream.num_bins();
        let fft = stream.fft_length();
        let sr = spec.sample_rate();

        // A pure tone in the top octave (8–16 kHz) — exactly where the half-band
        // leaked; 11 kHz folds to ≈ 1 kHz through the decimation cascade.
        let ftone = 11000.0;
        let total = 4 * fft;
        let signal: Vec<f64> = (0..total)
            .map(|i| (TAU * ftone * i as f64 / sr).sin())
            .collect();

        let mut out = vec![0.0_f64; stream.max_output_columns(total) * nb];
        let columns = stream
            .process_magnitude(&signal, &mut out)
            .expect("stream process");
        assert!(columns >= 1, "expected at least one settled column");

        // The last column: the peak sits in the top octave (near 11 kHz); the
        // deep octaves (every bin below the top four) must stay quiet.
        let last = &out[(columns - 1) * nb..columns * nb];
        let peak = last.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
        let deep_hi = nb.saturating_sub(48); // exclude the top four octaves
        let mut worst_deep = 0.0_f64;
        let mut worst_bin = 0usize;
        for (k, &m) in last[..deep_hi].iter().enumerate() {
            if m > worst_deep {
                worst_deep = m;
                worst_bin = k;
            }
        }
        assert!(
            worst_deep < 0.05 * peak,
            "demo-regime: 11 kHz top-octave tone leaked into deep bin {worst_bin} \
             (|{worst_deep}| vs peak |{peak}|) — an octave-aliasing reflection"
        );
    }

    /// Demo-regime deep-octave near-skirt conformance: the streaming
    /// plan's sub-fundamental main-lobe shoulder must be as clean as the batch
    /// plan's at the **deepest** octaves, in the regime the WASM demo runs.
    ///
    /// This is the regression guard for the streaming deep-octave bug: in the
    /// demo regime (`f_max ≈ 0.67·Nyquist`) the streaming `emit_column` projected
    /// each octave's frame end forward by the decimator group delay `offset` —
    /// `offset` samples *past* the newest decimated sample the continuous
    /// decimator had produced.  With the newest-anchored kernel sitting on the
    /// frame end, those non-existent samples read as zero and truncated the
    /// analysis window, broadening the deepest octaves' bins.  The leak was
    /// depth-dependent (≈ −20 dB one octave below a 1 kHz tone, deep; ≈ −54 dB
    /// below a 4 kHz tone, shallow) and did not converge with warm-up.  The
    /// `multi_octave_spec` conformance never caught it: at `f_max ≈ 0.06·Nyquist`
    /// the decimator transition is wide, its filter short, so `offset` is tiny.
    ///
    /// The "near skirt" is the worst bin in `(peak−12 ..= peak−3)` — the main-lobe
    /// shoulder ≤ 1 octave below the peak, skipping the two main-lobe-adjacent
    /// bins — reported as `20·log10(worst/peak)` dB.  A correct constant-Q
    /// transform has a frequency-independent skirt, so the streaming skirt must
    /// (a) match the batch skirt to a tight margin and (b) sit well below the bug
    /// (≤ −45 dB; measured ≈ −54 dB after the fix, for both a deep 1 kHz tone and
    /// a shallow 4 kHz tone).
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn demo_regime_deep_octave_skirt_matches_batch() {
        let spec =
            cqt::design(48000.0, 100.0, 16000.0, 12, &Window::Hann).expect("valid demo design");
        let nb = spec.num_bins();
        let fft = spec.fft_length();
        let sr = spec.sample_rate();

        // The worst main-lobe-shoulder bin (peak−12 ..= peak−3), in dB rel. peak.
        let near_skirt_db = |col: &[f64]| -> f64 {
            let peak_bin = col
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN"))
                .expect("at least one bin")
                .0;
            let peak = col[peak_bin].max(1e-30);
            let lo = peak_bin.saturating_sub(12);
            let hi = peak_bin.saturating_sub(3);
            let worst = col[lo..=hi.min(col.len() - 1)]
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            20.0 * (worst / peak).log10()
        };

        // A deep tone (1 kHz, several octaves below f_max — where the bug bit) and
        // a shallow tone (4 kHz — always clean) bracket the depth dependence.
        for ftone in [1000.0_f64, 4000.0] {
            // Batch skirt of one fft-length frame: the clean reference (≈ −54 dB).
            let batch = make_batch(&spec);
            let frame: Vec<f64> = (0..fft)
                .map(|i| (TAU * ftone * i as f64 / sr).sin())
                .collect();
            let mut batch_col = vec![0.0_f64; nb];
            batch
                .process_magnitude(&frame, &mut batch_col)
                .expect("batch process");
            let batch_skirt = near_skirt_db(&batch_col);

            // Streaming skirt of a settled column, warmed well past any transient.
            let mut stream = make_stream(&spec);
            let total = 16 * fft; // ≫ 2·fft, the settled threshold
            let signal: Vec<f64> = (0..total)
                .map(|i| (TAU * ftone * i as f64 / sr).sin())
                .collect();
            let mut out = vec![0.0_f64; stream.max_output_columns(total) * nb];
            let columns = stream
                .process_magnitude(&signal, &mut out)
                .expect("stream process");
            assert!(columns >= 1, "expected at least one settled column");
            let last = &out[(columns - 1) * nb..columns * nb];
            let stream_skirt = near_skirt_db(last);

            // (a) The streaming skirt sits well below the −20 dB bug (and the
            // window's own ≈ −54 dB floor): no deep-octave broadening.
            assert!(
                stream_skirt <= -45.0,
                "demo-regime streaming near skirt {stream_skirt:.1} dB at {ftone} Hz exceeds \
                 −45 dB — the deep-octave window is truncated"
            );
            // (b) Streaming matches batch (the constant-Q transform is correct in
            // batch): the two skirts agree to a tight margin, so streaming is not
            // merely "low" but reproduces the batch/oracle result.
            assert!(
                (stream_skirt - batch_skirt).abs() <= 6.0,
                "demo-regime streaming near skirt {stream_skirt:.1} dB at {ftone} Hz diverges \
                 from batch {batch_skirt:.1} dB by > 6 dB — streaming/batch deep-octave \
                 conformance gap"
            );
        }
    }

    /// Report the measured top-/deep-octave magnitude and complex residuals.
    ///
    /// `std::println!` is denied in production; in `#[cfg(test)]` it is the
    /// idiomatic way to surface a measured value under `--nocapture` without
    /// affecting the pass/fail decision.
    #[allow(
        clippy::print_stdout,
        reason = "test-only diagnostic surfaced under --nocapture"
    )]
    fn report_residuals(top_mag: f64, top_cplx: f64, deep_mag: f64, deep_cplx: f64) {
        std::println!(
            "newest-ref residual (rel): top mag {top_mag:.3e} cplx {top_cplx:.3e}, \
             deep mag {deep_mag:.3e} cplx {deep_cplx:.3e}"
        );
    }

    // ── Gain compensation flattens the octave seam (the 23d evidence) ──

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "sample indices, bin counts, and octave indices are small \
                  non-negative values exact in f64"
    )]
    fn gain_compensation_flattens_octave_seam() {
        // Every octave's mean ratio |stream|/|oracle| must sit within this band
        // of 1 — i.e. no staircase.  (Uncompensated, the deepest octave's mean
        // is ≈ 0.5–0.8; this 6% bound would fail loudly there.)
        const SEAM_TOL: f64 = 0.06;

        // Direct evidence the per-bin gain compensation removes the octave
        // staircase.  Drive a broadband input with an *equal* tone
        // at every bin frequency; against the decimation-free oracle each bin
        // should read ≈ the same magnitude (the input is flat across the
        // spectrum), so the per-octave mean of |stream|/|oracle| must sit at ≈ 1
        // in **every** octave.  Without compensation the decimator passband
        // droop pulls the deep octaves' ratio well below 1 (the seam); with it,
        // the staircase collapses and the per-octave means agree to a tight band.
        let spec = multi_octave_spec();
        let mut stream = make_stream(&spec);
        let reference = NewestRef::new(&spec);
        let nb = stream.num_bins();
        let fft = stream.fft_length();
        let hop = stream.hop_length();
        let align = align_of(&stream);
        let sr = spec.sample_rate();

        // Bin → octave (0 = top), mirroring `build_bands`' partition.
        let f_max = spec.bins().last().expect("spec has bins").frequency;
        let octave_of =
            |f: f64| -> usize { ((f_max / f).max(1.0).log2() + 1e-9).floor().max(0.0) as usize };
        let n_octaves = spec
            .bins()
            .iter()
            .map(|b| octave_of(b.frequency))
            .max()
            .expect("at least one bin")
            + 1;
        assert!(n_octaves >= 3, "need ≥3 octaves to exercise a deep seam");

        // Equal-amplitude tone at every bin → a flat target across the spectrum.
        let total = 4 * fft;
        let freqs: Vec<f64> = spec.bins().iter().map(|b| b.frequency).collect();
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

        // Per-octave accumulated ratio |stream|/|oracle| over the settled
        // columns; a flat input + correct compensation → every octave's mean ≈ 1.
        let mut sum = vec![0.0_f64; n_octaves];
        let mut cnt = vec![0usize; n_octaves];
        let mut checked = 0;
        for c in 0..columns {
            let raw = (c + 1) * hop;
            let now = raw - (raw % align);
            if now < 2 * fft {
                continue;
            }
            let window = &signal[now - fft..now];
            let oracle = reference.column(window);
            let col = &out[c * nb..(c + 1) * nb];
            for (k, (s, o)) in col.iter().zip(&oracle).enumerate() {
                let on = o.norm();
                if on < 1e-6 {
                    continue; // skip bins with negligible reference energy
                }
                let o_idx = octave_of(freqs[k]);
                sum[o_idx] += s.norm() / on;
                cnt[o_idx] += 1;
            }
            checked += 1;
        }
        assert!(
            checked >= 1,
            "expected ≥1 settled column for the seam check"
        );

        // Every octave's mean ratio must sit within a tight band of 1 — no
        // staircase.
        for o in 0..n_octaves {
            if cnt[o] == 0 {
                continue;
            }
            let mean = sum[o] / cnt[o] as f64;
            assert!(
                (mean - 1.0).abs() < SEAM_TOL,
                "octave {o} seam: mean |stream|/|oracle| {mean:.4} should be ≈1 (no staircase)"
            );
        }
    }

    // ── Decimator quality is a free knob (≥2 levels vs the reference) ──

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn decimator_quality_levels_track_reference() {
        // The decimator quality is a *free* knob on `CqtSpec`; both
        // a low and the default-high level must converge to the decimation-free
        // oracle in magnitude (the equiripple stopband + gain compensation hold
        // at every quality, the deeper stopband simply lowers the residual).
        use crate::design::cqt::DecimatorQuality;

        let base = multi_octave_spec();
        let fft = base.fft_length();
        let hop = base.hop_length().max(1);
        let nb = base.num_bins();
        let sr = base.sample_rate();
        let total = 4 * fft;
        let freqs: Vec<f64> = base.bins().iter().map(|b| b.frequency).collect();
        let signal: Vec<f64> = (0..total)
            .map(|i| {
                freqs
                    .iter()
                    .map(|&f| (TAU * f * i as f64 / sr).sin())
                    .sum::<f64>()
            })
            .collect();
        let reference = NewestRef::new(&base);

        for q in [0u8, DecimatorQuality::MAX] {
            let spec = base
                .clone()
                .with_decimator_quality(DecimatorQuality::new(q).expect("valid quality"));
            let mut stream = make_stream(&spec);
            let align = align_of(&stream);
            let mut out = vec![Complex::new(0.0, 0.0); stream.max_output_columns(total) * nb];
            let columns = stream.process(&signal, &mut out).expect("process");

            let mut worst_mag = 0.0_f64;
            let mut checked = 0;
            for c in 0..columns {
                let raw = (c + 1) * hop;
                let now = raw - (raw % align);
                if now < 2 * fft {
                    continue;
                }
                let window = &signal[now - fft..now];
                let oracle = reference.column(window);
                let col = &out[c * nb..(c + 1) * nb];
                for (s, o) in col.iter().zip(&oracle) {
                    let mag = (s.norm() - o.norm()).abs() / (1.0 + o.norm());
                    worst_mag = worst_mag.max(mag);
                }
                checked += 1;
            }
            assert!(checked >= 1, "quality {q}: expected ≥1 settled column");
            // Gain compensation keeps every quality's magnitude close to the
            // oracle; the bound covers both the lowest quality's shorter
            // (shallower-stopband) decimator and the deepest bin's group-delay
            // window-position residual against the decimation-free oracle (ticket
            // 26 — see `DEEP_MAG_TOL`).  Measured worst-case ≈ 3.6e-2.
            assert!(
                worst_mag < 5e-2,
                "quality {q}: worst deep magnitude residual {worst_mag:.3e} should track the oracle"
            );
        }
    }

    // ── Stationary cross-check against the oldest-anchored batch ──────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn stationary_magnitude_matches_oldest_batch() {
        // On a steady tone (stationary across the frame), newest- and oldest-
        // anchoring cover the same signal, so |streaming| matches |batch| and the
        // complex values differ by the per-bin linear phase exp(-j·2π·f_k·Δ/sr).
        // This is the one thing the oldest-anchored batch is still good for —
        // a steady-state cross-check, never the transient oracle.
        let spec = multi_octave_spec();
        let mut stream = make_stream(&spec);
        let batch = make_batch(&spec);
        let nb = stream.num_bins();
        let fft = stream.fft_length();
        let hop = stream.hop_length();
        let align = align_of(&stream);
        let sr = spec.sample_rate();
        let top_octave_lo = nb - 12.min(nb);

        // A single steady tone in the top octave (undecimated, so the phase
        // relationship is exact there).
        let target = nb - 6;
        let f = spec.bins()[target].frequency;
        let total = 4 * fft;
        let signal: Vec<f64> = (0..total)
            .map(|i| (TAU * f * i as f64 / sr).sin())
            .collect();

        let mut out = vec![Complex::new(0.0, 0.0); stream.max_output_columns(total) * nb];
        let columns = stream.process(&signal, &mut out).expect("process");

        assert!(target >= top_octave_lo, "target tone is in the top octave");
        let mut checked = 0;
        // The per-bin phase relationship between oldest- and newest-anchoring on
        // a stationary signal: streaming and batch differ by a **pure, fixed
        // per-bin phase** (equal magnitude).  We verify this on the tone bin by
        // (a) magnitude equality and (b) the ratio `stream/batch` being a unit
        // complex that is *constant* across settled columns — i.e. a single
        // time-invariant rotation, the very thing "the relationship holds" means.
        let mut ratio_ref: Option<Complex<f64>> = None;
        for c in 0..columns {
            let raw = (c + 1) * hop;
            let now = raw - (raw % align);
            if now < 2 * fft {
                continue;
            }
            let window = &signal[now - fft..now];
            let mut batch_col = vec![Complex::new(0.0, 0.0); nb];
            batch.execute(window, &mut batch_col).expect("batch");
            let col = &out[c * nb..(c + 1) * nb];

            for (k, (s, b)) in col.iter().zip(&batch_col).enumerate() {
                // |streaming| == |batch| in every octave for the stationary tone
                // — anchoring preserves magnitude when the signal is stationary
                // across the frame (the only thing the oldest batch is good for).
                assert!(
                    (s.norm() - b.norm()).abs() < REF_TOL * (1.0 + b.norm()),
                    "stationary column {c} bin {k}: |stream| {} vs |batch| {}",
                    s.norm(),
                    b.norm()
                );
            }

            // Tone bin: stream / batch is a unit-magnitude, constant rotation.
            let ratio = col[target] / batch_col[target];
            assert!(
                (ratio.norm() - 1.0).abs() < REF_TOL,
                "stationary tone bin {target} col {c}: |stream/batch| {} should be 1 (pure phase)",
                ratio.norm()
            );
            match ratio_ref {
                None => ratio_ref = Some(ratio),
                Some(r0) => assert!(
                    (ratio - r0).norm() < 1e-2,
                    "stationary tone bin {target} col {c}: phase rotation drifted ({ratio} vs {r0})"
                ),
            }
            checked += 1;
        }
        assert!(checked >= 1, "expected ≥1 settled stationary column");
    }

    // ── Per-octave Q/f ONSET latency (the responsiveness win) ──────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices and magnitudes are small enough for f64"
    )]
    fn per_octave_onset_latency_tracks_q_over_f() {
        // The decisive proof of newest-anchoring.  Switch on a tone at one centre
        // bin per octave and measure the bin's **onset** latency — the time from
        // switch-on until its magnitude first crosses a fixed fraction of steady
        // state.  With per-bin end-placement + windows ending at "now", a bin's
        // analysis ends at the most recent sample, so its onset latency is its
        // Gabor floor `Q/f`: treble near-instant, strictly increasing toward
        // bass, and clearly NOT a uniform window-length delay (which would be the
        // same ~`fft_length` samples for every bin under the old oldest-anchored
        // placement).
        let spec = multi_octave_spec();
        let sr = spec.sample_rate();
        let hop = spec.hop_length();
        let nb = spec.num_bins();
        let fft = spec.fft_length();
        let freqs: Vec<f64> = spec.bins().iter().map(|b| b.frequency).collect();
        let q = cqt::quality_factor(12);

        // Switch the tone on well past warm-up so the onset is clean.
        let onset = 2 * fft;
        let total = onset + 6 * fft;

        // One centre bin per octave: treble (oct 0), mid (oct 1), bass (oct 2).
        let probes = [(30usize, "treble"), (18, "mid"), (6, "bass")];
        let mut latency: Vec<f64> = Vec::new();
        let mut qf_samples: Vec<f64> = Vec::new();
        for &(target, label) in &probes {
            let f = freqs[target];
            let mut s = make_stream(&spec);
            let signal: Vec<f64> = (0..total)
                .map(|i| {
                    if i >= onset {
                        (TAU * f * i as f64 / sr).sin()
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut out = vec![Complex::new(0.0, 0.0); s.max_output_columns(total) * nb];
            let cols = s.process(&signal, &mut out).expect("process");
            let steady = out[(cols - 1) * nb + target].norm();
            assert!(
                steady > 1e-3,
                "bin {target} ({label}) should reach steady state"
            );

            // Onset latency: first hop instant past switch-on where the bin
            // crosses 50% of steady state.
            let mut t_onset = None;
            for c in 0..cols {
                let now = (c + 1) * hop;
                if now <= onset {
                    continue;
                }
                if out[c * nb + target].norm() >= 0.5 * steady {
                    t_onset = Some((now - onset) as f64);
                    break;
                }
            }
            let t = t_onset.expect("bin should reach 50% of steady state");
            latency.push(t);
            qf_samples.push(q * sr / f);
            report_onset(label, f, t, q * sr / f);
        }

        // Onset latency is strictly ordered treble < mid < bass — the `Q/f`
        // signature.  Crucially treble is far below a uniform window-length
        // (`fft`) delay: newest-anchoring relocated the analysis to "now".
        assert!(
            latency[0] < latency[1] && latency[1] < latency[2],
            "onset latency must increase treble→bass (Q/f), got {latency:?}"
        );
        assert!(
            latency[0] < fft as f64 / 2.0,
            "treble onset {} must be well below a window-length delay ({fft})",
            latency[0]
        );
        // Each octave down doubles `Q/f`; onset should grow toward 2× per octave
        // (a wide band absorbs hop quantisation and the 50% threshold).
        for w in [(0usize, 1usize), (1, 2)] {
            let ratio = latency[w.1] / latency[w.0].max(1.0);
            assert!(
                (1.3..=3.5).contains(&ratio),
                "onset ratio {ratio} (Q/f ratio {:.2}) out of band {latency:?}",
                qf_samples[w.1] / qf_samples[w.0]
            );
        }
    }

    /// Report a measured per-octave onset latency (samples and ms) plus its
    /// `Q/f` floor, surfaced under `--nocapture`.
    #[allow(
        clippy::print_stdout,
        reason = "test-only diagnostic surfaced under --nocapture"
    )]
    fn report_onset(label: &str, freq: f64, latency_samples: f64, qf_samples: f64) {
        let ms = latency_samples / 16.0; // 16 kHz → samples/16 = ms
        std::println!(
            "onset[{label}] f={freq:.0}Hz: latency {latency_samples:.0} samp ({ms:.1} ms), \
             Q/f floor {qf_samples:.0} samp"
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

        let total = 3 * fft;
        let signal: Vec<f64> = (0..total).map(|i| (i as f64 * 0.01).sin()).collect();
        let mut out_a = vec![Complex::new(0.0, 0.0); plan.max_output_columns(total) * nb];
        let cols_a = plan.process(&signal, &mut out_a).expect("first run");

        // reset must clear the continuous decimator state, rings, and phase so a
        // fresh feed reproduces the run exactly.
        plan.reset();
        let mut out_b = vec![Complex::new(0.0, 0.0); plan.max_output_columns(total) * nb];
        let cols_b = plan.process(&signal, &mut out_b).expect("post-reset run");

        assert_eq!(cols_a, cols_b, "reset restores the hop cadence");
        for (k, (a, b)) in out_a[..cols_a * nb].iter().zip(&out_b).enumerate() {
            assert!(
                (a - b).norm() < 1e-12,
                "index {k}: post-reset output differs ({a} vs {b})"
            );
        }
        let _ = hop;
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
        fn drive<P: CqtProcessor<f64>>(plan: &mut P, input: &[f64]) -> usize {
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

    // ── Reconfigure (live window swap) ─────────────────────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn reconfigure_window_matches_in_place_swap() {
        use crate::traits::reconfigure::Reconfigure;

        // Behavioural contract: after a mid-stream window swap, the streaming
        // processor's columns must reflect the NEW window over the PRESERVED
        // stream state.  The reference `q` is built with the target window and
        // fed the identical input history, so its decimator/ring state matches
        // the reconfigured processor's exactly — the in-place-swap reference, not
        // a fresh-built processor with no history.
        let spec_a = cqt::design(16000.0, 125.0, 1000.0, 12, &Window::Hann).expect("valid design");
        let spec_b =
            cqt::design(16000.0, 125.0, 1000.0, 12, &Window::Blackman).expect("valid design");

        let mut p = make_stream(&spec_a); // will be reconfigured A → B
        let mut q = make_stream(&spec_b); // reference, always window B
        let nb = p.num_bins();
        let fft = p.fft_length();
        let hop = p.hop_length();

        // Feed an identical warm-up history to a hop boundary so both share the
        // same decimator / ring state.
        let warm = (3 * fft / hop) * hop; // a whole number of hops, well past warm-up
        let warm_sig: Vec<f64> = (0..warm).map(|i| (i as f64 * 0.019).sin()).collect();
        let mut scratch_p = vec![Complex::new(0.0, 0.0); p.max_output_columns(warm) * nb];
        let mut scratch_q = vec![Complex::new(0.0, 0.0); q.max_output_columns(warm) * nb];
        p.process(&warm_sig, &mut scratch_p).expect("warm p");
        q.process(&warm_sig, &mut scratch_q).expect("warm q");

        // Swap p's window to B in place (never structural for the CQT window).
        p.reconfigure(&Window::Blackman).expect("reconfigure ok");

        // Feed both the same next chunk; the emitted columns must now agree.
        let chunk = 6 * hop;
        let next: Vec<f64> = (0..chunk)
            .map(|i| ((warm + i) as f64 * 0.019).sin())
            .collect();
        let mut out_p = vec![Complex::new(0.0, 0.0); p.max_output_columns(chunk) * nb];
        let mut out_q = vec![Complex::new(0.0, 0.0); q.max_output_columns(chunk) * nb];
        let cols_p = p.process(&next, &mut out_p).expect("p process");
        let cols_q = q.process(&next, &mut out_q).expect("q process");

        assert_eq!(
            cols_p, cols_q,
            "reconfigured and reference emit the same column count"
        );
        assert!(cols_p >= 1, "expected at least one post-swap column");
        for (k, (a, b)) in out_p[..cols_p * nb].iter().zip(&out_q).enumerate() {
            assert!(
                (a - b).norm() < 1e-10,
                "post-swap column index {k}: reconfigured {a} vs in-place-swap reference {b}"
            );
        }
    }

    #[test]
    fn reconfigure_window_never_structural() {
        use crate::traits::reconfigure::Reconfigure;

        // The window is orthogonal to the kernel sizes, so a window swap is
        // always Ok — it never returns StructuralMismatch.
        let spec = multi_octave_spec();
        let mut plan = make_stream(&spec);
        for window in [
            Window::Blackman,
            Window::Hamming,
            Window::Kaiser(8.0),
            Window::Hann,
        ] {
            assert!(
                plan.reconfigure(&window).is_ok(),
                "window swap to {window:?} must succeed (never structural)"
            );
        }
    }

    // ── Debug formatting ───────────────────────────────────────────

    #[test]
    fn debug_format_is_readable() {
        let plan = make_stream(&multi_octave_spec());
        let debug = format!("{plan:?}");
        assert!(
            debug.contains("OmniCqtProcessor"),
            "debug should contain the type name"
        );
    }
}
