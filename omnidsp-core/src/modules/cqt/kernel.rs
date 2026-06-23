// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Shared per-octave kernel design and octave partitioning for the CQT.
//!
//! Both the batch ([`batch`](super::batch)) and streaming
//! ([`stream`](super::stream)) CQT paths analyse the same octave-recursive
//! decomposition: bins are partitioned into octave bands by frequency relative
//! to the top bin, each band carries a per-octave r2c plan plus conjugated
//! half-spectrum kernels, and the signal is decimated ×2 between octaves via a
//! routed [`ResampleProcessor`]
//! sub-processor.  This module owns that construction — `build_octaves` — so
//! neither path duplicates the kernel math.
//!
//! # Kernel time-axis convention and anchoring
//!
//! A bin's kernel occupies `N_k` samples of its octave's `fft_len`-point frame;
//! *where* in that frame it sits is the **anchor**, and it is the one thing the
//! batch and streaming paths place differently:
//!
//! - `Anchor::Start` (batch, oldest-anchored) — kernel at indices `0 … N_k−1`:
//!
//!   ```text
//!   kernel[n] = (1/N_k) · window[n] · exp(j·2π·f_k·n / rate),  n = 0 … N_k−1
//!   ```
//!
//!   so each bin analyses the **oldest** `N_k` samples of its octave frame.  This
//!   matches the single-FFT reference and is the batch capstone's placement —
//!   it must not change (the stationary cross-check and shipped path).
//!
//! - `Anchor::End` (streaming, newest-anchored) — the same kernel shifted to
//!   indices `fft_len−N_k … fft_len−1`, so each bin analyses the **newest** `N_k`
//!   samples of its octave frame.  With the streaming `emit_column` gathering
//!   each octave's frame to *end at "now"*, a bin's analysis then ends at the
//!   most recent sample and its onset latency collapses to its Gabor floor
//!   `Q/f`.  This relocates *which samples each bin analyses*; it is
//!   not a pure phase shift, and it is **streaming-only**.
//!
//! Both anchors share the identical window/frequency math (same `window`, `f_k`,
//! `rate`, `N_k`, `fft_len`); only the time-domain placement index differs.  The
//! kernel is built in the time domain at its anchored position, transformed, and
//! stored as a conjugated half-spectrum (`N/2+1` bins) scaled by `1/fft_len`, so
//! the per-bin coefficient is a single complex dot product of the r2c
//! half-spectrum with the half-kernel.  Because the analytic kernel concentrates
//! in the positive half, the half-spectrum sum equals the full-spectrum
//! correlation up to negligible negative-frequency leakage (no factor-of-two:
//! dropped terms, not a one-sided power sum).  The `Anchor::Start` placement
//! differs from librosa's centered placement by a benign half-kernel time offset
//! (see the single-FFT reference).

use std::f64::consts::TAU;
use std::ops::{AddAssign, MulAssign};

use num_complex::Complex;

use crate::create::CreateProc;
use crate::design::cqt::{CqtSpec, DecimatorQuality};
use crate::design::fir::{self, FirMethod};
use crate::design::resample::ResampleSpec;
use crate::dispatch::Backend;
use crate::error::{Error, Result};
use crate::modules::resample::ResampleProcessor;
use crate::traits::dft::{DftNorm, DftR2c, DftR2cSpec};
use crate::types::{DspFloat, FilterType};
use crate::window::{self, Window};

/// Where a bin's kernel sits inside its octave's `fft_len`-point frame.
///
/// The single thing the two CQT paths place differently: the batch path anchors
/// at the frame **start** (oldest samples, the batch capstone placement), the
/// streaming path at the frame **end** (newest samples — newest-anchoring).
/// Every other term of the kernel math is shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Anchor {
    /// Kernel at indices `0 … N_k−1` — analyses the oldest `N_k` samples.
    Start,
    /// Kernel at indices `fft_len−N_k … fft_len−1` — analyses the newest `N_k`
    /// samples.
    End,
}

/// One octave band: its r2c plan, half-spectrum kernels, and placement.
pub(super) struct OctaveBand<T, RP> {
    /// Forward real-DFT plan for this band's `fft_len`.
    pub r2c: RP,
    /// Conjugated, scaled half-spectrum kernels (`fft_len/2 + 1` bins each),
    /// one per bin in this band, ordered low → high frequency.
    pub kernels: Vec<Vec<Complex<T>>>,
    /// Global output index of this band's lowest-frequency bin.
    pub out_start: usize,
    /// Per-octave FFT length (power of two ≥ the band's longest kernel).
    pub fft_len: usize,
    /// Group-delay offset into the (decimated) current-rate signal: the index
    /// at which this band's aligned content begins.
    pub offset: usize,
}

/// The shared octave decomposition of a [`CqtSpec`]: per-octave bands plus the
/// routed decimator sub-plan and the derived geometry both paths need.
///
/// `RP` is the per-octave [`DftR2cPlan`](crate::traits::dft::DftR2cPlan); `ResP`
/// is the concrete decimator [`ResampleProcessor`].
pub(super) struct OctaveLayout<T, RP, ResP> {
    /// Per-octave bands, ordered top (octave 0, full rate) → bottom.
    pub octaves: Vec<OctaveBand<T, RP>>,
    /// The ×2 half-band decimator (option-A routed sub-plan).
    pub decimator: ResP,
    /// Per-bin center frequencies in Hz, pinned low → high.
    pub bin_frequencies: Vec<f64>,
    /// Total number of frequency bins.
    pub num_bins: usize,
    /// Required input frame length (the spec's FFT length).
    pub fft_length: usize,
    /// Largest per-octave FFT length across the bands.
    pub max_fft: usize,
    /// Decimator output capacity for a full frame (`≥ 1`).
    pub next_cap: usize,
}

/// The streaming octave decomposition: the same per-octave bands as
/// [`OctaveLayout`], but with **one continuous decimator per octave transition**
/// (`octaves.len() − 1`, indexed by destination octave `1..O`) instead of a
/// single per-frame-reset decimator.
///
/// The streaming CQT runs each transition's decimator continuously, so the
/// decimators cannot be shared the way the batch path reuses one (reset per
/// stage); each octave's decimated stream advances independently.
pub(super) struct StreamOctaveLayout<T, RP, ResP> {
    /// Per-octave bands, ordered top (octave 0, full rate) → bottom.
    pub octaves: Vec<OctaveBand<T, RP>>,
    /// Continuous ×2 decimators, one per octave transition (`o−1 → o`), indexed
    /// by destination octave minus one (`decimators[o-1]` produces octave `o`).
    /// Empty for a single-octave spec.
    pub decimators: Vec<ResP>,
    /// Per-bin center frequencies in Hz, pinned low → high.
    pub bin_frequencies: Vec<f64>,
    /// Per-bin newest-anchored analysis-center latency in **top-rate input
    /// samples**, pinned low → high (parallel to `bin_frequencies`): the
    /// newest-anchored value at "now" reflects the signal at `now − latency`,
    /// `latency = window_len/2 + decimation group delay`.
    pub bin_latencies: Vec<f64>,
    /// Total number of frequency bins.
    pub num_bins: usize,
    /// Required input frame length (the spec's FFT length).
    pub fft_length: usize,
    /// Largest per-octave FFT length across the bands.
    pub max_fft: usize,
    /// Decimator output capacity (per call) for a full-frame input chunk
    /// (`≥ 1`); upper-bounds one decimator's per-chunk advance.
    pub next_cap: usize,
}

/// The per-octave bands plus the derived geometry, shared by both the batch
/// ([`build_octaves`]) and streaming ([`build_octaves_streaming`]) layouts.
struct OctaveBands<T, RP> {
    octaves: Vec<OctaveBand<T, RP>>,
    bin_frequencies: Vec<f64>,
    /// Per-bin newest-anchored analysis-center latency in **top-rate input
    /// samples**, pinned low → high (parallel to `bin_frequencies`).  Used by
    /// the streaming layout only (display latency compensation).
    bin_latencies: Vec<f64>,
    num_bins: usize,
    fft_length: usize,
    max_fft: usize,
    /// Number of octave bands.
    octaves_count: usize,
    /// The derived ×2 half-band decimation spec (shared prototype for every
    /// transition).
    decimate_spec: ResampleSpec,
}

/// Build the per-octave bands and the shared decimation spec for a [`CqtSpec`].
///
/// This is the kernel/partition construction common to both CQT paths; it does
/// **not** build any decimator (the batch and streaming layouts differ only in
/// how many decimators they instantiate and whether they reset per frame).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "kernel lengths, octave indices, and offsets are small \
              non-negative values exact in f64"
)]
fn build_bands<T, R>(dftr2c: &R, spec: &CqtSpec, anchor: Anchor) -> Result<OctaveBands<T, R::Plan>>
where
    T: DspFloat + AddAssign + MulAssign,
    R: DftR2c<T>,
{
    let sr = spec.sample_rate();
    let fft_length = spec.fft_length();
    let bins = spec.bins();
    let num_bins = bins.len();
    let f_max = bins
        .last()
        .ok_or_else(|| Error::Internal("CqtSpec has no bins".into()))?
        .frequency;

    // Octave index per bin (0 = top); bands are contiguous because bins
    // ascend in frequency.  An epsilon nudges exact boundaries downward.
    let octave_of = |freq: f64| -> usize {
        let ratio = (f_max / freq).max(1.0);
        (ratio.log2() + 1e-9).floor().max(0.0) as usize
    };
    let octaves_count = bins
        .iter()
        .map(|b| octave_of(b.frequency))
        .max()
        .unwrap_or(0)
        + 1;

    // The ×2 windowed-Kaiser decimation spec (option A), at the spec's
    // decimator quality.  Its passband must reach the *next* (lower) octave's top
    // bin — ≈ `f_max/2` normalized to the current rate, plus an ~8% margin for
    // that bin's bandwidth — while its stopband is planted at `fs/4` (the new
    // Nyquist).  By self-similarity this single normalized passband edge serves
    // every ×2 stage.  The prototype's linear-phase group delay is (proto_len-1)/4
    // output samples per ×2 stage.
    let passband_edge = (f_max / (2.0 * sr)) * 1.08;
    let decimate_spec = derive_decimate_spec(spec.decimator_quality(), passband_edge)?;
    let proto_len = decimate_spec.prototype_filter().len();
    let proto_delay = (proto_len.saturating_sub(1)) as f64 / 4.0;

    // Decimator coefficients (already f64) for the per-bin gain compensation: a
    // bin in octave `o` has passed through `o` cascaded ×2 half-band decimators,
    // so its cumulative passband gain is the product of the decimator's magnitude
    // response at the bin's normalized frequency at each stage's input rate.
    // Baking the analytic inverse `1/G_k` into the kernel flattens the octave
    // staircase — the exact inverse of the known
    // filter's known response, no fudge factor.  The half-band's DC gain is ~1
    // and the ×2 decimator applies no L-scaling (up = 1), so `G_k` is the true
    // end-to-end per-octave passband gain.
    let decim_coeffs: Vec<f64> = decimate_spec.prototype_filter().to_vec();

    let bin_frequencies: Vec<f64> = bins.iter().map(|b| b.frequency).collect();
    // Per-bin newest-anchored analysis-center latency, top-rate samples, written
    // into each bin's global index.  See the accumulation below.
    let mut bin_latencies = vec![0.0_f64; num_bins];

    let mut octaves = Vec::with_capacity(octaves_count);
    let mut scale = 1.0_f64; // 1 / 2^o
    let mut offset_f = 0.0_f64;
    let mut max_fft = 1usize;

    for o in 0..octaves_count {
        let rate_o = sr * scale;

        // Bins in this octave (a contiguous slice of the ascending array).
        let band_idx: Vec<usize> = (0..num_bins)
            .filter(|&i| octave_of(bins[i].frequency) == o)
            .collect();
        let out_start = band_idx.first().copied().unwrap_or(0);

        // Per-octave FFT length: power of two ≥ the band's longest kernel at
        // the decimated rate.  Empty bands (only possible for irregular
        // hand-rolled specs) get a dummy length-1 transform.
        let fft_len = if band_idx.is_empty() {
            1
        } else {
            band_idx
                .iter()
                .map(|&i| kernel_len_at(bins[i].kernel_len, scale))
                .max()
                .unwrap_or(1)
                .next_power_of_two()
        };
        max_fft = max_fft.max(fft_len);

        let r2c_spec = DftR2cSpec::new(fft_len, DftNorm::None)?;
        let r2c = dftr2c.create_plan(&r2c_spec)?;

        // Per-bin newest-anchored latency, in top-rate input
        // samples: a bin's newest-anchored value at "now" reflects the signal at
        // `now − latency`.  Two terms, both projected to the top rate:
        //   • `window_len/2` — the smooth window-center delay (the bin's full-rate
        //     time-domain window length, halved).  ∝ 1/f, the smooth bow.
        //   • `offset_f · 2^o` — the cascaded decimator group delay (`offset_f` is
        //     this octave's accumulated *decimated*-sample offset, `2^o = 1/scale`
        //     projects it to the top rate).  The per-octave staircase.
        // Compensating both de-warps the demo's exponential sweep.  Written into
        // the bin's global index (`band_idx` gives the global low→high indices).
        let offset_top = offset_f / scale;
        for &i in &band_idx {
            bin_latencies[i] = bins[i].kernel_len as f64 / 2.0 + offset_top;
        }

        let mut kernels = Vec::with_capacity(band_idx.len());
        for &i in &band_idx {
            // Materialize this bin's full-rate window directly in f64 (the design
            // precision) from the spec's recipe — the kernel is accumulated in
            // f64 and cast to `T` once, never round-tripped through `T`.
            let window = spec.window().coefficients::<f64>(bins[i].kernel_len)?;
            let nk = kernel_len_at(bins[i].kernel_len, scale);
            // Placement of this bin's `nk`-sample kernel inside the `fft_len`
            // frame: the frame start (oldest) for the batch path, the frame end
            // (newest) for the streaming path.
            let kernel_start = match anchor {
                Anchor::Start => 0,
                Anchor::End => fft_len - nk,
            };
            let mut kernel = build_half_kernel::<T>(
                &window,
                bins[i].frequency,
                rate_o,
                nk,
                fft_len,
                kernel_start,
            )?;
            // Bake the analytic inverse of the cascaded decimator passband
            // response into the kernel: a bin in octave `o` accumulated gain
            // G_k = ∏_{j=1}^{o} |H(ω_j)|, with ω_j = 2π·f_k·2^(j-1)/sr the bin's
            // normalized frequency at stage j's input rate sr/2^(j-1).  Scaling
            // the kernel coefficients by 1/G_k removes the octave droop.  The
            // top octave (o = 0) has an empty product → 1.0, so it is untouched.
            let comp = compensation_factor::<T>(&decim_coeffs, bins[i].frequency, sr, o)?;
            for c in &mut kernel {
                *c = *c * comp;
            }
            kernels.push(kernel);
        }

        octaves.push(OctaveBand {
            r2c,
            kernels,
            out_start,
            fft_len,
            offset: offset_f.round() as usize,
        });

        // Advance to the next-lower octave: halve the rate and accumulate
        // the decimator group delay (off_{o+1} = off_o / 2 + proto_delay).
        scale /= 2.0;
        offset_f = offset_f / 2.0 + proto_delay;
    }

    Ok(OctaveBands {
        octaves,
        bin_frequencies,
        bin_latencies,
        num_bins,
        fft_length,
        max_fft,
        octaves_count,
        decimate_spec,
    })
}

/// Build the per-octave bands and routed decimator for a [`CqtSpec`].
///
/// `dftr2c` builds each band's forward real-DFT plan; `resample_factory` (any
/// `CreatePlan<ResampleSpec>`) builds the ×2 decimator sub-plan and is then
/// **dropped** — only the concrete decimator is returned (option A).
///
/// # Errors
///
/// Returns an error if the spec has no bins, or if DFT / decimator sub-plan
/// creation fails.
#[allow(
    clippy::type_complexity,
    reason = "the layout names the routed decimator sub-plan; callers alias it"
)]
pub(super) fn build_octaves<T, R, RF>(
    dftr2c: &R,
    spec: &CqtSpec,
    resample_factory: &RF,
) -> Result<OctaveLayout<T, R::Plan, <RF as CreateProc<ResampleSpec>>::Proc<T>>>
where
    T: DspFloat + AddAssign + MulAssign,
    R: DftR2c<T>,
    RF: CreateProc<ResampleSpec> + Backend<T>,
    <RF as CreateProc<ResampleSpec>>::Proc<T>: ResampleProcessor<T>,
{
    // Batch is oldest-anchored: kernels at the frame start.
    let bands = build_bands::<T, R>(dftr2c, spec, Anchor::Start)?;

    // The single per-frame-reset decimator (option A): built through the routed
    // factory, then the factory is dropped.
    let decimator = resample_factory.create_proc::<T>(&bands.decimate_spec)?;
    let next_cap = decimator.max_output_len(bands.fft_length).max(1);

    Ok(OctaveLayout {
        octaves: bands.octaves,
        decimator,
        bin_frequencies: bands.bin_frequencies,
        num_bins: bands.num_bins,
        fft_length: bands.fft_length,
        max_fft: bands.max_fft,
        next_cap,
    })
}

/// Build the streaming layout: the same per-octave bands as [`build_octaves`],
/// plus **one continuous decimator per octave transition**.
///
/// `resample_factory` builds `octaves_count − 1` decimator sub-plans (one per
/// `o−1 → o` transition) and is then **dropped** — the layout stores only the
/// concrete decimators (option A).  Unlike the batch path, these
/// decimators are advanced continuously and never reset per frame, so each must
/// be a distinct plan with its own polyphase delay line.
///
/// # Errors
///
/// Returns an error if the spec has no bins, or if DFT / decimator sub-plan
/// creation fails.
#[allow(
    clippy::type_complexity,
    reason = "the layout names the routed decimator sub-plan; callers alias it"
)]
pub(super) fn build_octaves_streaming<T, R, RF>(
    dftr2c: &R,
    spec: &CqtSpec,
    resample_factory: &RF,
) -> Result<StreamOctaveLayout<T, R::Plan, <RF as CreateProc<ResampleSpec>>::Proc<T>>>
where
    T: DspFloat + AddAssign + MulAssign,
    R: DftR2c<T>,
    RF: CreateProc<ResampleSpec> + Backend<T>,
    <RF as CreateProc<ResampleSpec>>::Proc<T>: ResampleProcessor<T>,
{
    // Streaming is newest-anchored: kernels at the frame end, so
    // each bin analyses its newest `N_k` samples.  `emit_column` then gathers
    // each octave's frame to end at "now", giving per-bin latency ≈ `Q/f`.
    let bands = build_bands::<T, R>(dftr2c, spec, Anchor::End)?;

    // One continuous decimator per octave transition (`o−1 → o`).
    let n_transitions = bands.octaves_count.saturating_sub(1);
    let mut decimators = Vec::with_capacity(n_transitions);
    let mut next_cap = 1usize;
    for _ in 0..n_transitions {
        let dec = resample_factory.create_proc::<T>(&bands.decimate_spec)?;
        // A per-call chunk is bounded by one frame's worth of input at this
        // stage's rate; the top stage sees the most (`fft_length`).
        next_cap = next_cap.max(dec.max_output_len(bands.fft_length));
        decimators.push(dec);
    }

    Ok(StreamOctaveLayout {
        octaves: bands.octaves,
        decimators,
        bin_frequencies: bands.bin_frequencies,
        bin_latencies: bands.bin_latencies,
        num_bins: bands.num_bins,
        fft_length: bands.fft_length,
        max_fft: bands.max_fft,
        next_cap: next_cap.max(1),
    })
}

/// Kernel length at a decimated rate: the full-rate `len` scaled by `scale =
/// 1/2^o`, at least one sample.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "kernel lengths are small non-negative values exact in f64"
)]
fn kernel_len_at(len: usize, scale: f64) -> usize {
    ((len as f64 * scale).round() as usize).max(1)
}

/// The ×2 decimator's stopband edge (normalized): the new Nyquist after
/// downsampling, `fs/4` of the input rate.  Everything at or above this must be
/// rejected so it cannot fold back into the kept band.
const DECIMATOR_STOPBAND: f64 = 0.25;

/// Derive the ×2 **windowed Kaiser** decimation spec (`up = 1`, `down = 2`) for
/// the requested [`DecimatorQuality`] and `passband_edge` (supersedes the
/// equiripple half-band).
///
/// Designs an **asymmetric** Kaiser-windowed lowpass via [`fir::design`] — the
/// standard Schörkhuber–Klapuri / librosa octave-decimation anti-alias filter.
/// The band edges are the CQT-correct ones:
///
/// - **passband** to `passband_edge` (the highest center frequency of the bins in
///   the *next* — target lower — octave plus half its bandwidth, normalized to the
///   current rate), so the lower octaves' content survives the decimation;
/// - **stopband** from [`DECIMATOR_STOPBAND`] (`fs/4`, the new Nyquist), so every
///   alias that would fold back into the kept band is rejected to the requested
///   `quality` attenuation.
///
/// The transition is the "don't care" gap between them (the *current* octave's
/// lower content, already analysed).  This is the distinction the half-band could
/// not honour: a half-band's skirt is symmetric about `fs/4`, so its stopband
/// only begins at `fs/4 + δ` and content in `[fs/4, fs/4 + δ]` aliased.
///
/// The filter is designed at nominal rate `1`, so its cutoff `(passband_edge +
/// fs/4)/2` is already normalized; the cutoff is `< fs/4`, satisfying the
/// `1/(2·max(1, 2))` polyphase anti-alias bound the cross-spec invariant checks.
/// The coefficients are rate-independent; the resampler uses only the `1:2`
/// factors.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the Kaiser order is a small positive integer exact in f64"
)]
fn derive_decimate_spec(quality: DecimatorQuality, passband_edge: f64) -> Result<ResampleSpec> {
    // Asymmetric transition [passband_edge, fs/4]; clamp to keep a sane width.
    let fp = passband_edge.clamp(0.02, DECIMATOR_STOPBAND - 0.02);
    let transition = DECIMATOR_STOPBAND - fp;
    let cutoff_norm = f64::midpoint(fp, DECIMATOR_STOPBAND);

    // Stopband attenuation target → Kaiser order (Kaiser's order formula over the
    // actual transition width); the same target solves the window's β via
    // `window::kaiser::attenuation`, so kernel and decimator share one
    // spec without a hand-converted β.
    let atten_db = quality.stop_atten_db();
    let order_est = (atten_db - 7.95) / (2.285 * 2.0 * std::f64::consts::PI * transition);
    // Even order (Type-I linear-phase FIR), clamped to a sane range.
    let order = ((order_est.ceil() as usize).clamp(16, 1024) + 1) & !1;

    // The decimator is designed at nominal rate 1 (f64); its coefficients stay
    // f64 in the spec and cast to `T` at the resampler's create edge.
    let filter = fir::design(
        FilterType::Lowpass,
        order,
        1.0,
        cutoff_norm,
        None,
        &FirMethod::Windowed {
            window: Window::Kaiser(window::kaiser::attenuation(atten_db)),
        },
    )?;
    ResampleSpec::new(filter, 1, 2)
}

/// The per-bin gain-compensation factor `1/G_k` for a bin at `freq` Hz in
/// octave `octave` (0 = top), given the decimator coefficients `coeffs` and the
/// top sample rate `sr`.
///
/// The bin passed through `octave` cascaded ×2 half-band decimators.  At stage
/// `j` (1..=octave) the input rate is `sr / 2^(j-1)`, so the bin sat at
/// normalized angular frequency `ω_j = 2π · freq · 2^(j-1) / sr`, where the
/// decimator's magnitude response is `|H(ω_j)|`.  The cumulative passband gain
/// is `G_k = ∏_{j=1}^{octave} |H(ω_j)|` (an empty product, `1`, for the top
/// octave), and the compensation is its exact inverse `1/G_k`.
///
/// Returns an error only if `1/G_k` cannot be represented in `T`; `G_k` is a
/// product of passband magnitudes (each ≈ 1) so it is bounded away from zero
/// for any bin below the decimator's `fs/4` cutoff.
#[allow(
    clippy::cast_precision_loss,
    reason = "octave/stage indices are small non-negative values exact in f64"
)]
fn compensation_factor<T: DspFloat>(
    coeffs: &[f64],
    freq: f64,
    sr: f64,
    octave: usize,
) -> Result<T> {
    let mut gain = 1.0_f64;
    for j in 1..=octave {
        // Stage j's input rate is sr / 2^(j-1); the bin's normalized angular
        // frequency there is 2π · freq / (sr / 2^(j-1)) = 2π · freq · 2^(j-1)/sr.
        let stage_scale = (1usize << (j - 1)) as f64;
        let omega = TAU * freq * stage_scale / sr;
        gain *= dtft_magnitude(coeffs, omega);
    }
    // 1/G_k — the analytic inverse of the known filter's known response.
    T::from(1.0 / gain)
        .ok_or_else(|| Error::Internal("cannot represent gain-compensation factor in T".into()))
}

/// Magnitude of the FIR frequency response `H(ω) = Σ_n h[n] e^{-jωn}` at
/// normalized angular frequency `ω` (the DTFT of the decimator coefficients).
#[allow(
    clippy::cast_precision_loss,
    reason = "coefficient indices are small enough that usize→f64 is exact"
)]
fn dtft_magnitude(coeffs: &[f64], omega: f64) -> f64 {
    let (re, im) = coeffs
        .iter()
        .enumerate()
        .fold((0.0, 0.0), |(re, im), (n, &h)| {
            let phase = omega * n as f64;
            (h.mul_add(phase.cos(), re), h.mul_add(-phase.sin(), im))
        });
    re.hypot(im)
}

/// Build one conjugated half-spectrum kernel via a direct DFT.
///
/// `window` is the bin's full-rate window; for a decimated band it is
/// linearly resampled to `nk` samples (the kernel length at the band's rate).
/// The complex kernel `(1/nk)·w'[n]·exp(+j·2π·f_k·n/rate)` (`n = 0 … nk−1`) is
/// placed at frame positions `kernel_start … kernel_start+nk−1` — `0` for the
/// oldest-anchored ([`Anchor::Start`]) batch path, `fft_len−nk` for the
/// newest-anchored ([`Anchor::End`]) streaming path — then transformed and
/// conjugated/scaled by `1/fft_len`, keeping only the `fft_len/2 + 1`
/// half-spectrum (the analytic kernel concentrates in the positive half, so the
/// dropped upper terms are negligible — no factor of two).
///
/// The DFT phase uses each sample's **frame position** `p = kernel_start + n`,
/// so the stored coefficient references the kernel's own placement: the time
/// shift `kernel_start` it introduces is exactly the per-bin newest-anchoring
/// phase the streaming path relies on (it falls out of the placement rather than
/// being applied as a separate rotor).
#[allow(
    clippy::cast_precision_loss,
    reason = "kernel and FFT lengths are small enough that usize→f64 is exact"
)]
fn build_half_kernel<T: DspFloat>(
    window: &[f64],
    f_k: f64,
    rate: f64,
    nk: usize,
    fft_len: usize,
    kernel_start: usize,
) -> Result<Vec<Complex<T>>> {
    let half_len = fft_len / 2 + 1;
    let w = resample_window(window, nk);
    let inv_nk = 1.0 / nk as f64;
    let inv_n = 1.0 / fft_len as f64;

    let mut kernel = Vec::with_capacity(half_len);
    for m in 0..half_len {
        let mut acc_re = 0.0_f64;
        let mut acc_im = 0.0_f64;
        for (n, &wn) in w.iter().enumerate() {
            // The sample sits at frame position p = kernel_start + n; its
            // contribution to half-spectrum bin m carries the kernel phase
            // exp(+j·2π·f_k·n/rate) and the DFT phase exp(-j·2π·m·p/fft_len):
            //   2π·(f_k·n/rate − m·p/fft_len)
            let p = (kernel_start + n) as f64;
            let angle = TAU * (f_k * n as f64 / rate - (m as f64 * p) / fft_len as f64);
            let amp = wn * inv_nk;
            acc_re += amp * angle.cos();
            acc_im += amp * angle.sin();
        }
        // Stored kernel = conj(DFT) / fft_len.
        let re: T = T::from(acc_re * inv_n)
            .ok_or_else(|| Error::Internal("cannot represent kernel re in T".into()))?;
        let im: T = T::from(-acc_im * inv_n)
            .ok_or_else(|| Error::Internal("cannot represent kernel im in T".into()))?;
        kernel.push(Complex::new(re, im));
    }
    Ok(kernel)
}

/// Resample an f64 window vector to `target` points by linear interpolation.
///
/// `target ≤ window.len()` always holds (a decimated band's kernel is shorter),
/// so this only ever downsamples a smooth window.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "window lengths are small and indices stay in bounds"
)]
fn resample_window(window: &[f64], target: usize) -> Vec<f64> {
    let src_len = window.len();

    if target == src_len {
        return window.to_vec();
    }
    if target <= 1 || src_len <= 1 {
        return vec![window[0]; target];
    }

    let step = (src_len - 1) as f64 / (target - 1) as f64;
    (0..target)
        .map(|m| {
            let pos = m as f64 * step;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(src_len - 1);
            let frac = pos - lo as f64;
            let a = window[lo];
            let b = window[hi];
            (b - a).mul_add(frac, a)
        })
        .collect()
}
