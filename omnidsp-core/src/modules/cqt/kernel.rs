// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Shared per-octave kernel design and octave partitioning for the CQT.
//!
//! Both the batch ([`batch`](super::batch)) and streaming
//! ([`stream`](super::stream)) CQT paths analyse the same octave-recursive
//! decomposition: bins are partitioned into octave bands by frequency relative
//! to the top bin, each band carries a per-octave r2c plan plus conjugated
//! half-spectrum kernels, and the signal is decimated ×2 between octaves via a
//! routed [`ResamplePlan`] sub-plan.  This module owns that construction —
//! `build_octaves` — so neither path duplicates the kernel math.
//!
//! # Kernel time-axis convention
//!
//! Kernels are placed causally (starting at index 0), as in the single-FFT
//! reference:
//!
//! ```text
//! kernel[n] = (1/N_k) · window[n] · exp(j·2π·f_k·n / rate),  n = 0 … N_k−1
//! ```
//!
//! at the **band's working rate**, stored as a conjugated half-spectrum
//! (`N/2+1` bins) so the per-bin coefficient is a single complex dot product of
//! the r2c half-spectrum with the half-kernel.  Because the analytic kernel
//! concentrates in the positive half, the half-spectrum sum equals the
//! full-spectrum correlation up to negligible negative-frequency leakage (no
//! factor-of-two: dropped terms, not a one-sided power sum).  This differs from
//! librosa's centered placement by a benign half-kernel time offset (see the
//! single-FFT reference; ticket-10 lesson).

use std::f64::consts::TAU;
use std::ops::{AddAssign, MulAssign};

use num_complex::Complex;

use crate::create::CreatePlan;
use crate::design::cqt::CqtSpec;
use crate::design::resample::{
    self, DEFAULT_MAX_PHASES, ResampleMode, ResampleQuality, ResampleSpec,
};
use crate::error::{Error, Result};
use crate::modules::resample::ResamplePlan;
use crate::traits::dft::{DftNorm, DftR2c, DftR2cSpec};
use crate::types::{DspFloat, Window};

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
/// is the concrete decimator [`ResamplePlan`].
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

/// Build the per-octave bands and routed decimator for a [`CqtSpec`].
///
/// `dftr2c` builds each band's forward real-DFT plan; `resample_factory` (any
/// `CreatePlan<ResampleSpec>`) builds the ×2 decimator sub-plan and is then
/// **dropped** — only the concrete decimator is returned (option A, ADR-006
/// §2a).
///
/// # Errors
///
/// Returns an error if the spec has no bins, or if DFT / decimator sub-plan
/// creation fails.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "kernel lengths, octave indices, and offsets are small \
              non-negative values exact in f64"
)]
#[allow(
    clippy::type_complexity,
    reason = "the layout names the routed decimator sub-plan; callers alias it"
)]
pub(super) fn build_octaves<T, R, RF>(
    dftr2c: &R,
    spec: &CqtSpec<T>,
    resample_factory: &RF,
) -> Result<OctaveLayout<T, R::Plan, <RF as CreatePlan<ResampleSpec<T>>>::Plan>>
where
    T: DspFloat + AddAssign + MulAssign,
    R: DftR2c<T>,
    RF: CreatePlan<ResampleSpec<T>>,
    <RF as CreatePlan<ResampleSpec<T>>>::Plan: ResamplePlan<T>,
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

    // The ×2 half-band decimator sub-plan (option A).  Built through the
    // routed factory, then dropped — only its plan is stored.
    let decimate_spec = derive_decimate_spec::<T>()?;
    let proto_len = decimate_spec.prototype_filter().len();
    let decimator = resample_factory.create_plan(&decimate_spec)?;
    // Group delay of the linear-phase prototype at the stage's input rate is
    // (proto_len-1)/2; under ×2 decimation that is (proto_len-1)/4 output
    // samples per stage.
    let proto_delay = (proto_len.saturating_sub(1)) as f64 / 4.0;

    let bin_frequencies: Vec<f64> = bins.iter().map(|b| b.frequency).collect();

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
                .map(|&i| kernel_len_at(bins[i].window.len(), scale))
                .max()
                .unwrap_or(1)
                .next_power_of_two()
        };
        max_fft = max_fft.max(fft_len);

        let r2c_spec = DftR2cSpec::new(fft_len, DftNorm::None)?;
        let r2c = dftr2c.create_plan(&r2c_spec)?;

        let mut kernels = Vec::with_capacity(band_idx.len());
        for &i in &band_idx {
            let nk = kernel_len_at(bins[i].window.len(), scale);
            kernels.push(build_half_kernel::<T>(
                &bins[i].window,
                bins[i].frequency,
                rate_o,
                nk,
                fft_len,
            )?);
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

    let next_cap = decimator.max_output_len(fft_length).max(1);

    Ok(OctaveLayout {
        octaves,
        decimator,
        bin_frequencies,
        num_bins,
        fft_length,
        max_fft,
        next_cap,
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

/// Derive the ×2 half-band decimation spec (`up = 1`, `down = 2`) from
/// `design::resample` — a 2:1 input/output rate ratio yields the half-band
/// polyphase prototype (cutoff at a quarter of the input rate).
fn derive_decimate_spec<T: DspFloat>() -> Result<ResampleSpec<T>> {
    let two: T =
        T::from(2.0_f64).ok_or_else(|| Error::Internal("cannot represent 2.0 in T".into()))?;
    let one: T =
        T::from(1.0_f64).ok_or_else(|| Error::Internal("cannot represent 1.0 in T".into()))?;
    // Quality 5 keeps the transition band comfortably above every band's bins
    // (which sit well below the f_s/4 cutoff) without an over-long filter.
    let quality = ResampleQuality::new(5)?;
    resample::design(
        two,
        one,
        quality,
        &Window::<T>::Hamming,
        DEFAULT_MAX_PHASES,
        ResampleMode::Streaming,
    )
}

/// Build one conjugated half-spectrum kernel via a direct DFT.
///
/// `window` is the bin's full-rate window; for a decimated band it is
/// linearly resampled to `nk` samples (the kernel length at the band's rate).
/// The causal complex kernel `(1/nk)·w'[n]·exp(+j·2π·f_k·n/rate)` is
/// transformed and conjugated/scaled by `1/fft_len`, keeping only the
/// `fft_len/2 + 1` half-spectrum (the analytic kernel concentrates in the
/// positive half, so the dropped upper terms are negligible — no factor of two).
#[allow(
    clippy::cast_precision_loss,
    reason = "kernel and FFT lengths are small enough that usize→f64 is exact"
)]
fn build_half_kernel<T: DspFloat>(
    window: &[T],
    f_k: f64,
    rate: f64,
    nk: usize,
    fft_len: usize,
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
            // Phase of kernel[n] · exp(-j·2π·m·n/fft_len):
            //   2π·(f_k·n/rate − m·n/fft_len)
            let angle = TAU * (f_k * n as f64 / rate - (m as f64 * n as f64) / fft_len as f64);
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

/// Resample a window vector to `target` points by linear interpolation,
/// returning `f64` values for the kernel accumulation.
///
/// `target ≤ window.len()` always holds (a decimated band's kernel is shorter),
/// so this only ever downsamples a smooth window.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "window lengths are small and indices stay in bounds"
)]
fn resample_window<T: DspFloat>(window: &[T], target: usize) -> Vec<f64> {
    let src_len = window.len();
    let to_f64 = |x: T| -> f64 { x.to_f64().unwrap_or(0.0) };

    if target == src_len {
        return window.iter().copied().map(to_f64).collect();
    }
    if target <= 1 || src_len <= 1 {
        return (0..target).map(|_| to_f64(window[0])).collect();
    }

    let step = (src_len - 1) as f64 / (target - 1) as f64;
    (0..target)
        .map(|m| {
            let pos = m as f64 * step;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(src_len - 1);
            let frac = pos - lo as f64;
            let a = to_f64(window[lo]);
            let b = to_f64(window[hi]);
            (b - a).mul_add(frac, a)
        })
        .collect()
}
