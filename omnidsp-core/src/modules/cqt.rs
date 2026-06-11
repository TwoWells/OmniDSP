// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Constant-Q Transform module — octave-recursive multirate analysis.
//!
//! [`OmniCqt`] is the public CQT factory, generic over a real-to-complex DFT
//! ([`DftR2c`]) and [`VecOps`].  It builds [`OmniCqtPlan`]s from a [`CqtSpec`]
//! by **octave-recursive multirate decomposition** (Brown-Puckette /
//! Schörkhuber-Klapuri): the highest octave is analysed at the input rate with
//! a small per-octave FFT, then the signal is lowpassed and decimated ×2 and
//! the next-lower octave is analysed at half the rate, and so on.  Roughly an
//! order of magnitude less compute than sizing one FFT for the lowest bin — the
//! property that makes the CQT comfortable in a `wasm32` real-time visualiser
//! (the founding feature, ADR-006 §2a / ADR-009 §6, surface-lock capstone 5L).
//!
//! # Sub-plan routing (option A)
//!
//! The ×2 decimator is an [`OmniResample`](crate::modules::resample::OmniResample)
//! half-band sub-plan (`up = 1`, `down = 2`), built through a
//! `CreatePlan<ResampleSpec>` factory passed
//! to [`create_plan`](OmniCqt::create_plan) and **dropped immediately** — the
//! plan stores only the concrete decimator plan, never the factory (ADR-006
//! §2a).  On a vendor backend the decimator drops onto the native resampler;
//! on the `RustBackend` floor it is
//! [`OmniResample`](crate::modules::resample::OmniResample) over
//! [`ScalarVecOps`](crate::scalar::ScalarVecOps).  This is the first consumer of
//! sub-plan routing — it validates the surface before any native backend exists.
//!
//! # Octave partitioning
//!
//! Bins are partitioned into octave bands purely by frequency relative to the
//! highest bin `f_max`: band 0 is `(f_max/2, f_max]`, band 1 is `(f_max/4,
//! f_max/2]`, and so on.  No `bins_per_octave` field is required — the band of
//! a bin is a function of its frequency ratio to the top.  When bins are
//! octave-regular (the [`design`](crate::design::cqt::design) case) each band
//! holds the same relative kernel set at successively halved rates; an
//! arbitrary [`CqtSpec::new`] layout still partitions correctly.  A single-bin
//! spec degenerates to one band with no decimation.
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
//!
//! Internal scratch buffers — including the stateful decimator sub-plan — live
//! behind a [`Mutex`] so the plan satisfies `Send + Sync` on `&self`.  The lock
//! is uncontended in the common single-threaded case.

use std::f64::consts::TAU;
use std::fmt;
use std::ops::{AddAssign, MulAssign};
use std::sync::Mutex;

use num_complex::Complex;

use crate::create::CreatePlan;
use crate::design::cqt::CqtSpec;
use crate::design::resample::{
    self, DEFAULT_MAX_PHASES, ResampleMode, ResampleQuality, ResampleSpec,
};
use crate::error::{Error, Result};
use crate::modules::resample::ResamplePlan;
use crate::traits::dft::{DftNorm, DftR2c, DftR2cPlan, DftR2cSpec};
use crate::traits::vecops::VecOps;
use crate::types::{DspFloat, Window};

// The single-FFT reference (`SingleFftCqt`) is complex-to-complex; its imports
// are gated to the test / `bench` builds that compile it.
#[cfg(any(test, feature = "bench"))]
use crate::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec};
#[cfg(any(test, feature = "bench"))]
use crate::types::Direction;

// ─── Public types ──────────────────────────────────────────────────────

/// Generic multirate CQT factory backed by [`DftR2c`] and [`VecOps`].
///
/// Creates [`OmniCqtPlan`]s for specific CQT configurations.  The factory owns
/// the real-DFT factory and `VecOps` instance; plans own their sub-plans
/// (per-octave r2c plans and the decimator), never the factory.
#[derive(Debug, Clone)]
pub struct OmniCqt<R, V> {
    dftr2c: R,
    vecops: V,
}

impl<R, V> OmniCqt<R, V> {
    /// Create a new multirate CQT factory.
    #[must_use]
    pub const fn new(dftr2c: R, vecops: V) -> Self {
        Self { dftr2c, vecops }
    }
}

/// Execution plan for an octave-recursive multirate Constant-Q Transform.
///
/// Created by [`OmniCqt::create_plan`].  `Send + Sync`; each call to
/// [`process`](Self::process) computes the CQT of one frame independently — the
/// stateful decimator is reset per frame, so no state is carried between calls
/// (batch v1; cross-frame octave state is a later extension).
///
/// Type parameters: `RP` the per-octave [`DftR2cPlan`], `V` the [`VecOps`],
/// `ResP` the concrete decimator [`ResamplePlan`] (the routed sub-plan).
pub struct OmniCqtPlan<T, RP, V, ResP> {
    /// Per-octave bands, ordered top (octave 0, full rate) → bottom.
    octaves: Vec<OctaveBand<T, RP>>,
    /// `VecOps` handle for the per-bin complex dot product.
    vecops: V,
    /// Required input frame length (the spec's FFT length, sized to the lowest
    /// bin) — the lowest octave needs the full frame, decimated down.
    fft_length: usize,
    /// Hop length (recommended advance between frames; advisory).
    hop_length: usize,
    /// Total number of frequency bins.
    num_bins: usize,
    /// Per-bin center frequencies in Hz, pinned low → high.
    bin_frequencies: Vec<f64>,
    /// Scratch buffers and the stateful decimator behind a `Mutex`.
    scratch: Mutex<CqtScratch<T, ResP>>,
}

/// One octave band: its r2c plan, half-spectrum kernels, and placement.
struct OctaveBand<T, RP> {
    /// Forward real-DFT plan for this band's `fft_len`.
    r2c: RP,
    /// Conjugated, scaled half-spectrum kernels (`fft_len/2 + 1` bins each),
    /// one per bin in this band, ordered low → high frequency.
    kernels: Vec<Vec<Complex<T>>>,
    /// Global output index of this band's lowest-frequency bin.
    out_start: usize,
    /// Per-octave FFT length (power of two ≥ the band's longest kernel).
    fft_len: usize,
    /// Group-delay offset into the (decimated) current-rate signal: the index
    /// at which this band's aligned content begins.
    offset: usize,
}

impl<T, RP, V, ResP> fmt::Debug for OmniCqtPlan<T, RP, V, ResP> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniCqtPlan")
            .field("num_bins", &self.num_bins)
            .field("num_octaves", &self.octaves.len())
            .field("fft_length", &self.fft_length)
            .field("hop_length", &self.hop_length)
            .finish_non_exhaustive()
    }
}

/// Scratch buffers plus the stateful decimator sub-plan.
struct CqtScratch<T, ResP> {
    /// The ×2 half-band decimator (option-A routed sub-plan).  Reset per frame.
    decimator: ResP,
    /// Current-rate working signal (max length = `fft_length`).
    cur: Vec<T>,
    /// Decimator output buffer (max length = `ceil(fft_length / 2)`).
    next: Vec<T>,
    /// r2c input segment, consumed by the transform (max length = largest
    /// per-octave FFT length).
    seg: Vec<T>,
    /// r2c output half-spectrum (max length = largest `fft_len/2 + 1`).
    half: Vec<Complex<T>>,
}

// ─── Plan methods ─────────────────────────────────────────────────────

impl<T, RP, V, ResP> OmniCqtPlan<T, RP, V, ResP>
where
    T: DspFloat + AddAssign + MulAssign,
    RP: DftR2cPlan<T>,
    V: VecOps<T>,
    ResP: ResamplePlan<T>,
{
    /// Compute the CQT of one frame.
    ///
    /// `input` must have length [`fft_length`](Self::fft_length).
    /// `output` must have length [`num_bins`](Self::num_bins).  Output bins are
    /// in the pinned low → high frequency order.  Each output element is the
    /// complex CQT coefficient for that bin.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if buffer lengths are wrong, or
    /// propagates DFT / decimator execution failures.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire octave-recursive pipeline"
    )]
    pub fn process(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
        if input.len() != self.fft_length {
            return Err(Error::BufferMismatch {
                expected: self.fft_length,
                actual: input.len(),
            });
        }
        if output.len() != self.num_bins {
            return Err(Error::BufferMismatch {
                expected: self.num_bins,
                actual: output.len(),
            });
        }

        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;
        let CqtScratch {
            decimator,
            cur,
            next,
            seg,
            half,
        } = &mut *guard;

        // Load the frame at the full (top-octave) rate.
        cur[..input.len()].copy_from_slice(input);
        let mut cur_len = input.len();
        let n_octaves = self.octaves.len();

        for (o, band) in self.octaves.iter().enumerate() {
            let fft_len = band.fft_len;
            let half_len = fft_len / 2 + 1;

            // Extract the group-delay-aligned segment of the current-rate
            // signal, zero-padding past its end.
            for (i, s) in seg[..fft_len].iter_mut().enumerate() {
                let src = band.offset + i;
                *s = if src < cur_len { cur[src] } else { T::zero() };
            }

            if !band.kernels.is_empty() {
                // r2c consumes `seg`; correlate the half-spectrum with each
                // half-kernel (a single complex dot product per bin).
                band.r2c
                    .process(&mut seg[..fft_len], &mut half[..half_len])?;
                for (j, kernel) in band.kernels.iter().enumerate() {
                    output[band.out_start + j] = self.vecops.cdot(&half[..half_len], kernel)?;
                }
            }

            // Decimate ×2 for the next-lower octave (reset for per-frame
            // independence — batch v1).
            if o + 1 < n_octaves {
                decimator.reset();
                let max_out = decimator.max_output_len(cur_len);
                let n = decimator.process(&cur[..cur_len], &mut next[..max_out])?;
                cur[..n].copy_from_slice(&next[..n]);
                cur_len = n;
            }
        }

        Ok(())
    }

    /// Compute the CQT magnitude spectrum of one frame.
    ///
    /// Convenience wrapper around [`process`](Self::process) that writes
    /// `|CQT[k]|` to `output`.
    ///
    /// `input` must have length [`fft_length`](Self::fft_length).
    /// `output` must have length [`num_bins`](Self::num_bins).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if buffer lengths are wrong, or
    /// propagates execution failures.
    pub fn process_magnitude(&self, input: &[T], output: &mut [T]) -> Result<()> {
        let zero = Complex::new(T::zero(), T::zero());
        let mut coeffs = vec![zero; self.num_bins];
        self.process(input, &mut coeffs)?;
        self.vecops.mag(&coeffs, output)?;
        Ok(())
    }

    /// Number of frequency bins.
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Required input frame length (the spec's FFT length).
    #[must_use]
    pub const fn fft_length(&self) -> usize {
        self.fft_length
    }

    /// Hop length (recommended advance between frames).
    #[must_use]
    pub const fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Number of octave bands the transform decomposes into.
    #[must_use]
    pub const fn num_octaves(&self) -> usize {
        self.octaves.len()
    }

    /// Per-bin center frequencies in Hz (pinned low → high).
    #[must_use]
    pub fn bin_frequencies(&self) -> &[f64] {
        &self.bin_frequencies
    }
}

// ─── Plan trait ───────────────────────────────────────────────────────

/// Execution object for a configured Constant-Q Transform.
///
/// The named, `Send + Sync` plan trait for the CQT module, mirroring
/// [`ConvPlan`](crate::traits::conv::ConvPlan) /
/// [`DctPlan`](crate::traits::dct::DctPlan) (ADR-006 §2 eager plan traits,
/// ADR-007 §6).  It lets the per-frame `process` be called generically (e.g.
/// by the shared conformance suite covering the multirate CQT) without naming
/// the concrete plan.  Implemented by [`OmniCqtPlan`]; vendor overrides may
/// implement it directly.  The magnitude convenience
/// ([`process_magnitude`](OmniCqtPlan::process_magnitude)) stays inherent.
pub trait CqtPlan<T>: Send + Sync {
    /// Compute the complex CQT coefficients of one frame.
    ///
    /// `input` must have length `fft_length`; `output` must have length
    /// `num_bins`, in the pinned low → high order.
    ///
    /// # Errors
    ///
    /// Returns an error if either buffer length is wrong, or if execution
    /// fails.
    fn process(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()>;
}

impl<T, RP, V, ResP> CqtPlan<T> for OmniCqtPlan<T, RP, V, ResP>
where
    T: DspFloat + AddAssign + MulAssign,
    RP: DftR2cPlan<T>,
    V: VecOps<T>,
    ResP: ResamplePlan<T> + Send,
{
    fn process(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
        // Delegate to the inherent `process`.  Inherent methods take precedence
        // over trait methods in resolution, so this is not recursive.
        self.process(input, output)
    }
}

// ─── Factory ──────────────────────────────────────────────────────────

impl<R, V> OmniCqt<R, V> {
    /// Create a multirate CQT plan from a [`CqtSpec`].
    ///
    /// `resample_factory` is the backend (any `CreatePlan<ResampleSpec>`); it
    /// builds the per-octave ×2 decimator sub-plan and is then **dropped** —
    /// the plan stores only the concrete decimator (option A, ADR-006 §2a).
    ///
    /// The spec provides per-bin center frequencies and window coefficients.
    /// Construct one via [`design`](crate::design::cqt::design) or
    /// [`CqtSpec::new`] for full control.
    ///
    /// # Errors
    ///
    /// Returns an error if DFT or decimator sub-plan creation fails.  The spec
    /// invariants are enforced by [`CqtSpec::new`], so they are not re-checked.
    #[allow(
        clippy::type_complexity,
        reason = "composite plan type names the routed decimator sub-plan; the \
                  dispatch layer aliases it via `type Plan`"
    )]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "kernel lengths, octave indices, and offsets are small \
                  non-negative values exact in f64"
    )]
    pub fn create_plan<T, RF>(
        &self,
        spec: &CqtSpec<T>,
        resample_factory: &RF,
    ) -> Result<OmniCqtPlan<T, R::Plan, V, <RF as CreatePlan<ResampleSpec<T>>>::Plan>>
    where
        T: DspFloat + AddAssign + MulAssign,
        R: DftR2c<T>,
        V: VecOps<T>,
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

        let zero = Complex::new(T::zero(), T::zero());
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
            let r2c = self.dftr2c.create_plan(&r2c_spec)?;

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
        let scratch = CqtScratch {
            decimator,
            cur: vec![T::zero(); fft_length],
            next: vec![T::zero(); next_cap],
            seg: vec![T::zero(); max_fft],
            half: vec![zero; max_fft / 2 + 1],
        };

        Ok(OmniCqtPlan {
            octaves,
            vecops: self.vecops.clone(),
            fft_length,
            hop_length: spec.hop_length(),
            num_bins,
            bin_frequencies,
            scratch: Mutex::new(scratch),
        })
    }
}

// ─── Construction helpers ──────────────────────────────────────────────

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

// ─── Single-FFT reference (test / `bench` feature only) ────────────────

/// Single-FFT Constant-Q Transform — the simple reference the multirate
/// [`OmniCqt`] is validated and benchmarked against.
///
/// Sizes one FFT for the lowest bin and correlates every bin against that one
/// spectrum (`Σ X[m]·conj(K[m])`).  It is **not** part of the shipped surface —
/// surface-lock 5L resolved the CQT to a single public type, the multirate
/// [`OmniCqt`] — so it is compiled only under `cfg(test)` or the `bench`
/// feature, serving as the equivalence oracle and as the naive baseline the
/// `cqt` benchmark times the multirate path against.
#[cfg(any(test, feature = "bench"))]
#[derive(Debug, Clone)]
pub struct SingleFftCqt<D, V> {
    dft: D,
    vecops: V,
}

#[cfg(any(test, feature = "bench"))]
impl<D, V> SingleFftCqt<D, V> {
    /// Create a single-FFT CQT factory.
    #[must_use]
    pub const fn new(dft: D, vecops: V) -> Self {
        Self { dft, vecops }
    }
}

/// Execution plan for the single-FFT reference CQT (see [`SingleFftCqt`]).
#[cfg(any(test, feature = "bench"))]
pub struct SingleFftCqtPlan<T, P, V> {
    kernels: Vec<Vec<Complex<T>>>,
    fwd: P,
    vecops: V,
    fft_length: usize,
    num_bins: usize,
    scratch: Mutex<SingleFftScratch<T>>,
}

#[cfg(any(test, feature = "bench"))]
struct SingleFftScratch<T> {
    buf_input: Vec<Complex<T>>,
    buf_spectrum: Vec<Complex<T>>,
}

#[cfg(any(test, feature = "bench"))]
impl<T, P, V> fmt::Debug for SingleFftCqtPlan<T, P, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleFftCqtPlan")
            .field("num_bins", &self.num_bins)
            .field("fft_length", &self.fft_length)
            .finish_non_exhaustive()
    }
}

#[cfg(any(test, feature = "bench"))]
impl<T, P, V> SingleFftCqtPlan<T, P, V>
where
    T: DspFloat + AddAssign + MulAssign,
    P: DftC2cPlan<T>,
    V: VecOps<T>,
{
    /// Compute the CQT of one frame.
    ///
    /// `input` must have length [`fft_length`](Self::fft_length); `output` must
    /// have length [`num_bins`](Self::num_bins).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] on a length mismatch, or propagates FFT
    /// execution failures.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire FFT + correlation pipeline"
    )]
    pub fn process(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
        if input.len() != self.fft_length {
            return Err(Error::BufferMismatch {
                expected: self.fft_length,
                actual: input.len(),
            });
        }
        if output.len() != self.num_bins {
            return Err(Error::BufferMismatch {
                expected: self.num_bins,
                actual: output.len(),
            });
        }

        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;
        let SingleFftScratch {
            buf_input,
            buf_spectrum,
        } = &mut *guard;

        self.vecops.real_to_complex(input, buf_input)?;
        self.fwd.process(buf_input, buf_spectrum)?;
        for (k, kernel) in self.kernels.iter().enumerate() {
            output[k] = self.vecops.cdot(buf_spectrum, kernel)?;
        }
        Ok(())
    }

    /// Number of frequency bins.
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Required input frame length.
    #[must_use]
    pub const fn fft_length(&self) -> usize {
        self.fft_length
    }
}

#[cfg(any(test, feature = "bench"))]
impl<D, V> SingleFftCqt<D, V> {
    /// Build a single-FFT CQT plan from a [`CqtSpec`].
    ///
    /// # Errors
    ///
    /// Returns an error if DFT plan creation fails.
    #[allow(
        clippy::cast_precision_loss,
        reason = "kernel lengths and FFT sizes are small enough for f64"
    )]
    #[allow(
        clippy::type_complexity,
        reason = "the plan names its concrete forward DFT plan type"
    )]
    pub fn create_plan<T>(&self, spec: &CqtSpec<T>) -> Result<SingleFftCqtPlan<T, D::Plan, V>>
    where
        T: DspFloat + AddAssign + MulAssign,
        D: DftC2c<T>,
        V: VecOps<T>,
    {
        let fft_length = spec.fft_length();
        let fwd_spec = DftC2cSpec::new(fft_length, Direction::Forward, DftNorm::None)?;
        let fwd = self.dft.create_plan(&fwd_spec)?;

        let inv_n = T::from(fft_length)
            .ok_or_else(|| Error::Internal("cannot convert FFT length to T".into()))?
            .recip();
        let zero = Complex::new(T::zero(), T::zero());
        let mut kern_time = vec![zero; fft_length];
        let mut kern_freq = vec![zero; fft_length];

        let bins = spec.bins();
        let mut kernels = Vec::with_capacity(bins.len());
        for bin in bins {
            let n_k = bin.window.len();
            let inv_nk = T::from(n_k)
                .ok_or_else(|| Error::Internal("cannot convert kernel length to T".into()))?
                .recip();
            kern_time.fill(zero);
            for (n, &w) in bin.window.iter().enumerate() {
                let angle = TAU * bin.frequency * n as f64 / spec.sample_rate();
                let cos_v = T::from(angle.cos())
                    .ok_or_else(|| Error::Internal("cannot convert cos to T".into()))?;
                let sin_v = T::from(angle.sin())
                    .ok_or_else(|| Error::Internal("cannot convert sin to T".into()))?;
                let scaled = w * inv_nk;
                kern_time[n] = Complex::new(scaled * cos_v, scaled * sin_v);
            }
            fwd.process(&kern_time, &mut kern_freq)?;
            kernels.push(
                kern_freq
                    .iter()
                    .map(|c| Complex::new(c.re * inv_n, -(c.im * inv_n)))
                    .collect(),
            );
        }

        let num_bins = kernels.len();
        Ok(SingleFftCqtPlan {
            kernels,
            fwd,
            vecops: self.vecops.clone(),
            fft_length,
            num_bins,
            scratch: Mutex::new(SingleFftScratch {
                buf_input: vec![zero; fft_length],
                buf_spectrum: vec![zero; fft_length],
            }),
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use std::f64::consts::TAU;

    use num_complex::Complex;

    use super::{CqtPlan, OmniCqt, OmniCqtPlan, SingleFftCqt};
    use crate::design::cqt::{self, CqtSpec};
    use crate::modules::resample::{OmniResample, OmniResamplePlan};
    use crate::test_utils::{TestDftC2c, TestDftR2c, TestVecOps};
    use crate::traits::dft::{DftC2c, DftR2c};
    use crate::types::Window;

    // ── The multirate plan under test ──────────────────────────────

    type TestCqtPlan = OmniCqtPlan<
        f64,
        <TestDftR2c as DftR2c<f64>>::Plan,
        TestVecOps,
        OmniResamplePlan<f64, TestVecOps>,
    >;

    /// Build a multirate CQT plan from a spec, routing the decimator through an
    /// [`OmniResample`] factory (the floor's `CreatePlan<ResampleSpec>`).
    fn make_plan(spec: &CqtSpec<f64>) -> TestCqtPlan {
        let factory = OmniCqt::new(TestDftR2c, TestVecOps);
        factory
            .create_plan(spec, &OmniResample::new(TestVecOps))
            .expect("multirate plan creation should succeed")
    }

    /// One octave, 12 bins — degenerate (no decimation).
    fn one_octave_spec() -> CqtSpec<f64> {
        cqt::design(44100.0, 440.0, 880.0, 12, &Window::Hann).expect("valid design")
    }

    /// Three octaves at 16 kHz — exercises the decimation chain.  `f_max` sits
    /// at ~0.06·Nyquist, far below every decimation cutoff.
    fn multi_octave_spec() -> CqtSpec<f64> {
        cqt::design(16000.0, 125.0, 1000.0, 12, &Window::Hann).expect("valid design")
    }

    // ── The single-FFT oracle (reference only) ─────────────────────
    //
    // The simple, numpy-validated single-FFT CQT ([`SingleFftCqt`]) retained
    // ONLY as the test oracle the multirate path is checked against (surface-lock
    // 5L: one public type — the multirate `OmniCqt`).  It is c2c and computes
    // each bin as the full-spectrum correlation `Σ X[m]·conj(K[m])`.

    fn oracle_plan(
        spec: &CqtSpec<f64>,
    ) -> super::SingleFftCqtPlan<f64, <TestDftC2c as DftC2c<f64>>::Plan, TestVecOps> {
        SingleFftCqt::new(TestDftC2c, TestVecOps)
            .create_plan(spec)
            .expect("oracle plan creation should succeed")
    }

    fn oracle_magnitudes(spec: &CqtSpec<f64>, input: &[f64]) -> Vec<f64> {
        let plan = oracle_plan(spec);
        let mut out = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(input, &mut out).expect("oracle process");
        out.iter().map(|c| c.norm()).collect()
    }

    // ── Accessors ──────────────────────────────────────────────────

    #[test]
    fn accessors_match_spec() {
        let spec = one_octave_spec();
        let plan = make_plan(&spec);
        assert_eq!(plan.num_bins(), spec.num_bins(), "num_bins matches spec");
        assert_eq!(
            plan.fft_length(),
            spec.fft_length(),
            "fft_length matches spec"
        );
        assert_eq!(
            plan.hop_length(),
            spec.hop_length(),
            "hop_length matches spec"
        );
        assert_eq!(
            plan.bin_frequencies().len(),
            spec.num_bins(),
            "one frequency per bin"
        );
        assert_eq!(plan.num_octaves(), 1, "one octave for a one-octave design");
    }

    #[test]
    fn multi_octave_decomposes() {
        let spec = multi_octave_spec();
        let plan = make_plan(&spec);
        assert!(
            plan.num_octaves() >= 3,
            "125–1000 Hz should span at least 3 octaves, got {}",
            plan.num_octaves()
        );
    }

    // ── Buffer validation ──────────────────────────────────────────

    #[test]
    fn process_rejects_wrong_input_length() {
        let plan = make_plan(&one_octave_spec());
        let input = vec![0.0_f64; plan.fft_length() + 1];
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "wrong input length should be rejected"
        );
    }

    #[test]
    fn process_rejects_wrong_output_length() {
        let plan = make_plan(&one_octave_spec());
        let input = vec![0.0_f64; plan.fft_length()];
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins() + 1];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "wrong output length should be rejected"
        );
    }

    // ── Zero input ─────────────────────────────────────────────────

    #[test]
    fn zero_input_produces_zero_output() {
        let plan = make_plan(&multi_octave_spec());
        let input = vec![0.0_f64; plan.fft_length()];
        let mut output = vec![Complex::new(9.0, 9.0); plan.num_bins()];
        plan.process(&input, &mut output).expect("process");
        for (k, c) in output.iter().enumerate() {
            assert!(
                c.norm() < 1e-10,
                "bin {k} should be zero for zero input, got {}",
                c.norm()
            );
        }
    }

    // ── Oracle equivalence (top octave, no decimation) ─────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "FFT length is small enough for f64"
    )]
    fn matches_oracle_one_octave() {
        // A one-octave spec has no decimation: the multirate path reduces to the
        // half-spectrum correlation of the same frame the oracle FFTs in full,
        // so they agree up to negligible negative-frequency leakage.
        let spec = one_octave_spec();
        let plan = make_plan(&spec);
        let n = plan.fft_length();
        let sr = spec.sample_rate();

        // Broadband input: sum of all bin-center tones.
        let input: Vec<f64> = (0..n)
            .map(|i| {
                plan.bin_frequencies()
                    .iter()
                    .map(|&f| (TAU * f * i as f64 / sr).sin())
                    .sum()
            })
            .collect();

        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut output).expect("process");
        let got: Vec<f64> = output.iter().map(|c| c.norm()).collect();
        let want = oracle_magnitudes(&spec, &input);

        // The only difference is the dropped negative-frequency terms of the
        // half-spectrum correlation — a few ×1e-5 on these ~0.1 magnitudes.
        let tol = 2e-4;
        for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < tol,
                "bin {k}: multirate |{g}| vs oracle |{w}| differ by {}",
                (g - w).abs()
            );
        }
    }

    // ── Pure tone peaks in its octave's center bin ─────────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn pure_tone_peaks_per_octave() {
        let spec = multi_octave_spec();
        let plan = make_plan(&spec);
        let n = plan.fft_length();
        let sr = spec.sample_rate();
        let num_octaves = plan.num_octaves();
        let bins_per_octave = 12;

        // Pick one mid-band bin per octave and confirm a tone there peaks in it.
        for o in 0..num_octaves {
            // Octave o occupies the high `o`-th block; pick a bin near its
            // center, clamped into range.
            let center = plan.num_bins().saturating_sub((o * bins_per_octave) + 6);
            let target = center.min(plan.num_bins() - 1);
            let freq = plan.bin_frequencies()[target];

            let input: Vec<f64> = (0..n).map(|i| (TAU * freq * i as f64 / sr).sin()).collect();
            let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
            plan.process(&input, &mut output).expect("process");

            let peak = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).expect("no NaN"))
                .expect("at least one bin")
                .0;
            // The peak should land in the same bin (or an immediate neighbour —
            // octave-edge bins sit nearer the decimation transition).
            assert!(
                peak.abs_diff(target) <= 1,
                "tone at {freq:.1} Hz (bin {target}, octave {o}) peaked at bin {peak}"
            );
        }
    }

    // ── Sub-plan routing actually runs the decimator ───────────────

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn sub_plan_routing_runs_decimation() {
        // A multi-octave transform must route through the OmniResample(1,2)
        // sub-plan for every octave below the top, producing finite, non-trivial
        // coefficients in the lower octaves (which only exist via decimation).
        let spec = multi_octave_spec();
        let plan = make_plan(&spec);
        assert!(
            plan.num_octaves() >= 3,
            "need decimation to exercise routing"
        );
        let n = plan.fft_length();
        let sr = spec.sample_rate();

        // A low tone in the bottom octave.
        let freq = plan.bin_frequencies()[2];
        let input: Vec<f64> = (0..n).map(|i| (TAU * freq * i as f64 / sr).sin()).collect();
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut output).expect("process");

        assert!(
            output.iter().all(|c| c.re.is_finite() && c.im.is_finite()),
            "all coefficients must be finite"
        );
        // The bottom-octave bins (reached only through decimation) must carry
        // real energy for a tone living there.
        let bottom_energy: f64 = output[..6].iter().map(|c| c.norm()).sum();
        assert!(
            bottom_energy > 1e-6,
            "bottom-octave bins should respond to a bottom-octave tone, got {bottom_energy}"
        );
    }

    // ── Plan reuse is deterministic ────────────────────────────────

    #[test]
    fn plan_reuse_is_deterministic() {
        let plan = make_plan(&multi_octave_spec());
        let n = plan.fft_length();
        let mut input = vec![0.0_f64; n];
        input[0] = 1.0;

        let mut out1 = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        let mut out2 = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut out1).expect("first");
        plan.process(&input, &mut out2).expect("second");

        for (k, (a, b)) in out1.iter().zip(&out2).enumerate() {
            assert!(
                (a - b).norm() < 1e-12,
                "bin {k} differs between identical calls: {a} vs {b}"
            );
        }
    }

    // ── process_magnitude agrees with process ──────────────────────

    #[test]
    fn process_magnitude_matches_process() {
        let plan = make_plan(&multi_octave_spec());
        let n = plan.fft_length();
        let mut input = vec![0.0_f64; n];
        input[0] = 1.0;

        let mut complex_out = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut complex_out).expect("process");
        let mut mag_out = vec![0.0_f64; plan.num_bins()];
        plan.process_magnitude(&input, &mut mag_out)
            .expect("process_magnitude");

        for (k, (&m, c)) in mag_out.iter().zip(&complex_out).enumerate() {
            assert!(
                (m - c.norm()).abs() < 1e-12,
                "bin {k} magnitude mismatch: {m} vs {}",
                c.norm()
            );
        }
    }

    // ── Debug formatting ───────────────────────────────────────────

    #[test]
    fn debug_format_is_readable() {
        let plan = make_plan(&one_octave_spec());
        let debug = format!("{plan:?}");
        assert!(
            debug.contains("OmniCqtPlan"),
            "debug should contain the type name"
        );
        assert!(
            debug.contains("num_octaves"),
            "debug should report the octave count"
        );
    }

    // ── CqtPlan trait ──────────────────────────────────────────────

    #[test]
    fn implements_plan_trait() {
        fn check<P: CqtPlan<f64>>(plan: &P, input: &[f64], output: &mut [Complex<f64>]) {
            plan.process(input, output).expect("trait process");
        }

        let plan = make_plan(&one_octave_spec());
        let mut input = vec![0.0_f64; plan.fft_length()];
        input[0] = 1.0;
        let mut via_trait = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        check(&plan, &input, &mut via_trait);

        let mut direct = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(&input, &mut direct).expect("inherent process");
        for (k, (t, d)) in via_trait.iter().zip(&direct).enumerate() {
            assert!(
                (t - d).norm() < 1e-15,
                "trait vs inherent mismatch at bin {k}"
            );
        }
    }

    // ── Single-bin degenerate spec ─────────────────────────────────

    #[test]
    fn single_bin_spec_works() {
        // The integration smoke-test shape: one bin, no octave structure.
        let bins = vec![crate::design::cqt::CqtBinSpec {
            frequency: 440.0,
            window: vec![0.0_f64, 0.5, 1.0, 0.5, 0.0],
        }];
        let spec = CqtSpec::new(44100.0, 8, 2, bins).expect("spec");
        let plan = make_plan(&spec);
        assert_eq!(plan.num_octaves(), 1, "single bin → one octave");

        let input = vec![0.0_f64; 8];
        let mut output = vec![Complex::new(0.0, 0.0); 1];
        plan.process(&input, &mut output).expect("process");
        assert!(output[0].norm() < 1e-10, "zero input → zero output");
    }

    // ── Oracle's own numpy validation (reference correctness) ──────
    //
    // Generated by scripts/gen_cqt_process_reference.py.
    // Regenerate with: make gen-cqt-process-reference
    //
    // These pin the single-FFT oracle to numpy so it remains a trustworthy
    // reference for the multirate equivalence checks above.

    include!(testdata!("cqt_process_numpy.rs"));

    fn numpy_spec() -> CqtSpec<f64> {
        cqt::design(
            CQT_PROC_SAMPLE_RATE,
            CQT_PROC_MIN_FREQ,
            CQT_PROC_MAX_FREQ,
            CQT_PROC_BINS_PER_OCTAVE,
            &Window::Hann,
        )
        .expect("valid design")
    }

    fn assert_complex_approx_eq(
        actual: &[Complex<f64>],
        expected: &[(f64, f64)],
        tol: f64,
        label: &str,
    ) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (a, &(re, im))) in actual.iter().zip(expected).enumerate() {
            let diff = (Complex::new(a.re - re, a.im - im)).norm();
            assert!(
                diff < tol,
                "{label}: bin {i} mismatch: got ({}, {}), expected ({re}, {im}), diff={diff}",
                a.re,
                a.im
            );
        }
    }

    #[test]
    fn oracle_numpy_all_tones() {
        let spec = numpy_spec();
        assert_eq!(spec.fft_length(), CQT_PROC_FFT_LENGTH, "FFT length");
        assert_eq!(spec.num_bins(), CQT_PROC_NUM_BINS, "bin count");

        let plan = oracle_plan(&spec);
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(CQT_PROC_ALL_TONES_INPUT, &mut output)
            .expect("oracle process");
        assert_complex_approx_eq(
            &output,
            CQT_PROC_ALL_TONES_OUTPUT,
            1e-10,
            "oracle all-tones",
        );
    }

    #[test]
    fn oracle_numpy_pure_tone() {
        let spec = numpy_spec();
        let plan = oracle_plan(&spec);
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(CQT_PROC_TONE_INPUT, &mut output)
            .expect("oracle process");
        assert_complex_approx_eq(&output, CQT_PROC_TONE_OUTPUT, 1e-10, "oracle pure-tone");

        let peak = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).expect("no NaN"))
            .expect("at least one bin")
            .0;
        assert_eq!(peak, CQT_PROC_TONE_BIN, "oracle peak bin");
    }

    #[test]
    fn oracle_numpy_two_tone() {
        let spec = numpy_spec();
        let plan = oracle_plan(&spec);
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.process(CQT_PROC_TWO_TONE_INPUT, &mut output)
            .expect("oracle process");
        assert_complex_approx_eq(&output, CQT_PROC_TWO_TONE_OUTPUT, 1e-10, "oracle two-tone");
    }
}
