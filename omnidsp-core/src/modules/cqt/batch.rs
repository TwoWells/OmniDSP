// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Batch (per-frame) multirate CQT — the multirate capstone.
//!
//! [`OmniCqt`] is the public CQT factory, generic over a real-to-complex DFT
//! ([`DftR2c`]) and [`VecOps`].  It builds [`OmniCqtPlan`]s from a
//! [`CqtSpec`] by **octave-recursive multirate
//! decomposition** (Brown-Puckette / Schörkhuber-Klapuri): the highest octave
//! is analysed at the input rate with a small per-octave FFT, then the signal
//! is lowpassed and decimated ×2 and the next-lower octave is analysed at half
//! the rate, and so on.  Roughly an order of magnitude less compute than sizing
//! one FFT for the lowest bin — the property that makes the CQT comfortable in a
//! `wasm32` real-time visualiser (the founding feature).
//!
//! The per-octave kernel design and octave partitioning live in the shared
//! [`kernel`] submodule; this batch file owns the per-frame `process`.
//! It anchors every bin's kernel at the **oldest** sample of the frame
//! (`kernel::Anchor::Start`); it is the streaming CQT's **stationary
//! cross-check** ([`stream`](super::stream)) — on a steady tone the
//! streaming magnitude matches this batch transform, and the per-bin phase
//! relationship holds — while the streaming path's primary oracle is an
//! independent newest-anchored reference.
//!
//! # Sub-plan routing (option A)
//!
//! The ×2 decimator is an [`OmniResample`](crate::modules::resample::OmniResample)
//! half-band sub-plan (`up = 1`, `down = 2`), built through a
//! `CreatePlan<ResampleSpec>` factory passed
//! to [`create_plan`](OmniCqt::create_plan) and **dropped immediately** — the
//! plan stores only the concrete decimator plan, never the factory.  On a
//! vendor backend the decimator drops onto the native resampler;
//! on the `RustBackend` floor it is
//! [`OmniResample`](crate::modules::resample::OmniResample) over
//! [`ScalarVecOps`](crate::scalar::ScalarVecOps).  This is the first consumer of
//! sub-plan routing — it validates the surface before any native backend exists.
//!
//! Internal scratch buffers — including the stateful decimator sub-plan — live
//! behind a [`Mutex`] so the plan satisfies `Send + Sync` on `&self`.  The lock
//! is uncontended in the common single-threaded case.

use std::fmt;
use std::sync::Mutex;

use num_complex::Complex;

use crate::create::CreateProc;
use crate::design::cqt::CqtSpec;
use crate::design::resample::ResampleSpec;
use crate::dispatch::Backend;
use crate::error::{Error, Result};
use crate::modules::cqt::kernel::{self, OctaveBand};
use crate::modules::resample::ResampleProcessor;
use crate::traits::dft::{DftR2c, DftR2cPlan};
use crate::traits::vecops::VecOps;
use crate::types::DspFloat;

// The single-FFT reference (`SingleFftCqt`) is complex-to-complex; its imports
// are gated to the test / `bench` builds that compile it.
#[cfg(any(test, feature = "bench"))]
use crate::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec, DftNorm};
#[cfg(any(test, feature = "bench"))]
use crate::types::Direction;
#[cfg(any(test, feature = "bench"))]
use std::f64::consts::TAU;

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

    /// The factory's `VecOps` handle (shared by the streaming path's
    /// magnitude convenience).
    pub(super) const fn vecops(&self) -> &V {
        &self.vecops
    }

    /// The factory's real-DFT handle (shared by the streaming path's octave
    /// layout builder).
    pub(super) const fn dftr2c(&self) -> &R {
        &self.dftr2c
    }
}

/// Execution plan for an octave-recursive multirate Constant-Q Transform.
///
/// Created by [`OmniCqt::create_plan`].  `Send + Sync`; each call to
/// [`execute`](Self::execute) computes the CQT of one frame independently — the
/// stateful decimator is reset per frame, so no state is carried between calls
/// (batch v1; cross-frame octave state is the streaming extension).
///
/// Type parameters: `RP` the per-octave [`DftR2cPlan`], `V` the [`VecOps`],
/// `ResP` the concrete decimator [`ResampleProcessor`] (the routed sub-processor).
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
    T: DspFloat,
    RP: DftR2cPlan<T>,
    V: VecOps<T>,
    ResP: ResampleProcessor<T>,
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
    pub fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
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
                    .execute(&mut seg[..fft_len], &mut half[..half_len])?;
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
    /// Convenience wrapper around [`execute`](Self::execute) that writes
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
        self.execute(input, &mut coeffs)?;
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
/// The named, `Send + Sync` plan trait for the CQT module, mirroring the eager
/// plan traits [`ConvPlan`](crate::traits::conv::ConvPlan) /
/// [`DctPlan`](crate::traits::dct::DctPlan).  It lets the per-frame `process`
/// be called generically (e.g. by the shared conformance suite covering the
/// multirate CQT) without naming the concrete plan.  Beyond `process` it
/// carries the buffer-sizing facts a
/// generic caller needs ([`num_bins`](Self::num_bins),
/// [`fft_length`](Self::fft_length) — the CQT's `process` has a size
/// precondition the length-symmetric `ConvPlan`/`DctPlan` do not), the per-bin
/// center frequencies, and the magnitude convenience.  Each method delegates to
/// the inherent implementation on [`OmniCqtPlan`]; vendor overrides may
/// implement the trait directly.
pub trait CqtPlan<T>: Send + Sync {
    /// Number of frequency bins per frame — the required `output` length for
    /// [`execute`](Self::execute) / [`process_magnitude`](Self::process_magnitude).
    fn num_bins(&self) -> usize;

    /// Required input frame length (the spec's FFT length).
    fn fft_length(&self) -> usize;

    /// Per-bin center frequencies in Hz (pinned low → high).
    fn bin_frequencies(&self) -> &[f64];

    /// Compute the complex CQT coefficients of one frame.
    ///
    /// `input` must have length [`fft_length`](Self::fft_length); `output` must
    /// have length [`num_bins`](Self::num_bins), in the pinned low → high order.
    ///
    /// # Errors
    ///
    /// Returns an error if either buffer length is wrong, or if execution
    /// fails.
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()>;

    /// Compute the CQT **magnitude** spectrum of one frame, writing `|CQT[k]|`
    /// to `output` (same buffer contract as [`execute`](Self::execute)).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if buffer lengths are wrong, or
    /// propagates execution failures.
    fn process_magnitude(&self, input: &[T], output: &mut [T]) -> Result<()>;
}

impl<T, RP, V, ResP> CqtPlan<T> for OmniCqtPlan<T, RP, V, ResP>
where
    T: DspFloat,
    RP: DftR2cPlan<T>,
    V: VecOps<T>,
    ResP: ResampleProcessor<T> + Send,
{
    // Each method delegates to the inherent one.  Inherent methods take
    // precedence over trait methods in resolution, so these are not recursive.
    fn num_bins(&self) -> usize {
        self.num_bins()
    }

    fn fft_length(&self) -> usize {
        self.fft_length()
    }

    fn bin_frequencies(&self) -> &[f64] {
        self.bin_frequencies()
    }

    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
        self.execute(input, output)
    }

    fn process_magnitude(&self, input: &[T], output: &mut [T]) -> Result<()> {
        self.process_magnitude(input, output)
    }
}

// ─── Factory ──────────────────────────────────────────────────────────

impl<R, V> OmniCqt<R, V> {
    /// Create a multirate CQT plan from a [`CqtSpec`].
    ///
    /// `resample_factory` is the backend (any `CreateProc<ResampleSpec>` that is
    /// also a [`Backend<T>`](crate::dispatch::Backend)); it builds the per-octave
    /// ×2 decimator sub-processor and is then **dropped** — the plan stores only
    /// the concrete decimator (option A).
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
        reason = "composite plan type names the routed decimator sub-processor; \
                  the dispatch layer aliases it via `type Plan`"
    )]
    pub fn create_plan<T, RF>(
        &self,
        spec: &CqtSpec,
        resample_factory: &RF,
    ) -> Result<OmniCqtPlan<T, R::Plan, V, <RF as CreateProc<ResampleSpec>>::Proc<T>>>
    where
        T: DspFloat,
        R: DftR2c<T>,
        V: VecOps<T>,
        RF: CreateProc<ResampleSpec> + Backend<T>,
        <RF as CreateProc<ResampleSpec>>::Proc<T>: ResampleProcessor<T>,
    {
        let layout = kernel::build_octaves::<T, R, RF>(&self.dftr2c, spec, resample_factory)?;
        let zero = Complex::new(T::zero(), T::zero());
        let scratch = CqtScratch {
            cur: vec![T::zero(); layout.fft_length],
            next: vec![T::zero(); layout.next_cap],
            seg: vec![T::zero(); layout.max_fft],
            half: vec![zero; layout.max_fft / 2 + 1],
            decimator: layout.decimator,
        };

        Ok(OmniCqtPlan {
            octaves: layout.octaves,
            vecops: self.vecops.clone(),
            fft_length: layout.fft_length,
            hop_length: spec.hop_length(),
            num_bins: layout.num_bins,
            bin_frequencies: layout.bin_frequencies,
            scratch: Mutex::new(scratch),
        })
    }
}

// ─── Single-FFT reference (test / `bench` feature only) ────────────────

/// Single-FFT Constant-Q Transform — the simple reference the multirate
/// [`OmniCqt`] is validated and benchmarked against.
///
/// Sizes one FFT for the lowest bin and correlates every bin against that one
/// spectrum (`Σ X[m]·conj(K[m])`).  It is **not** part of the shipped surface —
/// the public API exposes a single CQT type, the multirate
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
    T: DspFloat,
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
    pub fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
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
        self.fwd.execute(buf_input, buf_spectrum)?;
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
    pub fn create_plan<T>(&self, spec: &CqtSpec) -> Result<SingleFftCqtPlan<T, D::Plan, V>>
    where
        T: DspFloat,
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
            // Materialize this bin's window in f64 from the spec's recipe at the
            // bin's own kernel length — the single-FFT reference does not decimate,
            // so it uses the full-rate window directly.
            let window = spec.window().coefficients::<f64>(bin.kernel_len)?;
            let n_k = bin.kernel_len;
            let inv_nk = T::from(n_k)
                .ok_or_else(|| Error::Internal("cannot convert kernel length to T".into()))?
                .recip();
            kern_time.fill(zero);
            for (n, &w) in window.iter().enumerate() {
                let angle = TAU * bin.frequency * n as f64 / spec.sample_rate();
                let cos_v = T::from(angle.cos())
                    .ok_or_else(|| Error::Internal("cannot convert cos to T".into()))?;
                let sin_v = T::from(angle.sin())
                    .ok_or_else(|| Error::Internal("cannot convert sin to T".into()))?;
                let w_t = T::from(w)
                    .ok_or_else(|| Error::Internal("cannot convert window sample to T".into()))?;
                let scaled = w_t * inv_nk;
                kern_time[n] = Complex::new(scaled * cos_v, scaled * sin_v);
            }
            fwd.execute(&kern_time, &mut kern_freq)?;
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
    use crate::modules::resample::OmniResampleProcessor;
    use crate::test_utils::{TestBackend, TestDftC2c, TestDftR2c, TestVecOps};
    use crate::traits::dft::{DftC2c, DftR2c};
    use crate::window::Window;

    // ── The multirate plan under test ──────────────────────────────

    type TestCqtPlan =
        OmniCqtPlan<f64, <TestDftR2c as DftR2c<f64>>::Plan, TestVecOps, OmniResampleProcessor<f64>>;

    /// Build a multirate CQT plan from a spec, routing the decimator through a
    /// [`TestBackend`] (the foundational `Backend` + `CreateProc<ResampleSpec>`).
    fn make_plan(spec: &CqtSpec) -> TestCqtPlan {
        let factory = OmniCqt::new(TestDftR2c, TestVecOps);
        factory
            .create_plan(spec, &TestBackend)
            .expect("multirate plan creation should succeed")
    }

    /// One octave, 12 bins — degenerate (no decimation).
    fn one_octave_spec() -> CqtSpec {
        cqt::design(44100.0, 440.0, 880.0, 12, &Window::Hann).expect("valid design")
    }

    /// Three octaves at 16 kHz — exercises the decimation chain.  `f_max` sits
    /// at ~0.06·Nyquist, far below every decimation cutoff.
    fn multi_octave_spec() -> CqtSpec {
        cqt::design(16000.0, 125.0, 1000.0, 12, &Window::Hann).expect("valid design")
    }

    // ── The single-FFT oracle (reference only) ─────────────────────
    //
    // The simple, numpy-validated single-FFT CQT ([`SingleFftCqt`]) retained
    // ONLY as the test oracle the multirate path is checked against (one public
    // type — the multirate `OmniCqt`).  It is c2c and computes
    // each bin as the full-spectrum correlation `Σ X[m]·conj(K[m])`.

    fn oracle_plan(
        spec: &CqtSpec,
    ) -> super::SingleFftCqtPlan<f64, <TestDftC2c as DftC2c<f64>>::Plan, TestVecOps> {
        SingleFftCqt::new(TestDftC2c, TestVecOps)
            .create_plan(spec)
            .expect("oracle plan creation should succeed")
    }

    fn oracle_magnitudes(spec: &CqtSpec, input: &[f64]) -> Vec<f64> {
        let plan = oracle_plan(spec);
        let mut out = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.execute(input, &mut out).expect("oracle process");
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
            plan.execute(&input, &mut output).is_err(),
            "wrong input length should be rejected"
        );
    }

    #[test]
    fn process_rejects_wrong_output_length() {
        let plan = make_plan(&one_octave_spec());
        let input = vec![0.0_f64; plan.fft_length()];
        let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins() + 1];
        assert!(
            plan.execute(&input, &mut output).is_err(),
            "wrong output length should be rejected"
        );
    }

    // ── Zero input ─────────────────────────────────────────────────

    #[test]
    fn zero_input_produces_zero_output() {
        let plan = make_plan(&multi_octave_spec());
        let input = vec![0.0_f64; plan.fft_length()];
        let mut output = vec![Complex::new(9.0, 9.0); plan.num_bins()];
        plan.execute(&input, &mut output).expect("process");
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
        plan.execute(&input, &mut output).expect("process");
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

    // ── Sweep: no octave-aliasing image below the fundamental ──────
    //
    // The demo plays a pure exponential sweep; a rising tone must not paint an
    // aliased image into the octaves *below* it.  We compare the multirate path
    // to the decimation-free single-FFT oracle on a sweep frame: the multirate
    // low-octave energy must stay at (or below) the oracle's own inherent kernel
    // leakage, and far below any visible floor.  (The 23d half-band put a
    // full-amplitude phantom here — +1 dB; this guards the sweep case the
    // static-tone tests miss.)
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "sweep sample indices are small enough for f64"
    )]
    fn sweep_adds_no_low_octave_aliasing() {
        let spec = cqt::design(48000.0, 100.0, 16000.0, 12, &Window::Hann).expect("design");
        let plan = make_plan(&spec);
        let n = plan.fft_length();
        let sr = 48000.0;
        let (f_lo, f_hi, tsweep) = (100.0_f64, 16000.0_f64, 6.0_f64);
        let total = (tsweep * sr) as usize;
        let kk = (f_hi / f_lo).ln();
        let tc = tsweep * (1000.0_f64 / f_lo).ln() / kk; // frame centred where f≈1000 Hz
        let start = ((tc * sr) as usize).saturating_sub(n / 2);
        let mut phase = 0.0_f64;
        let mut frame = vec![0.0_f64; n];
        for i in 0..(start + n) {
            let f = f_lo * (kk * (i as f64) / (total as f64)).exp();
            phase += 2.0 * std::f64::consts::PI * f / sr;
            if i >= start {
                frame[i - start] = 0.3 * phase.sin();
            }
        }
        let mut out = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.execute(&frame, &mut out).expect("process");
        let mr: Vec<f64> = out.iter().map(|c| c.norm()).collect();
        let oracle = oracle_magnitudes(&spec, &frame);
        let peak_bin = mr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN"))
            .expect("bins")
            .0;
        let peak = mr[peak_bin].max(1e-12);
        let lo = peak_bin.saturating_sub(12); // bins > 1 octave below the peak
        let worst_mr = mr[..lo].iter().copied().fold(0.0_f64, f64::max);
        let worst_oracle = oracle[..lo].iter().copied().fold(0.0_f64, f64::max);
        let db = |x: f64| 20.0 * (x / peak).log10();
        assert!(
            db(worst_mr) < -50.0,
            "sweep low-octave leakage {:.1} dB exceeds −50 dB — an octave-aliasing image",
            db(worst_mr)
        );
        assert!(
            worst_mr <= worst_oracle * 2.0,
            "multirate low-octave leakage {worst_mr:.6} exceeds 2× the oracle's inherent \
             {worst_oracle:.6} — a decimation artifact, not just kernel leakage",
        );
    }

    // ── Demo-regime stress: f_max near Nyquist (what the WASM demo runs) ──
    //
    // `multi_octave_spec` keeps f_max ≈ 0.06·Nyquist — far below every
    // decimation cutoff — so it never exercises the decimator transition. The
    // demo sweeps 20 Hz–16 kHz with f_max a large fraction of Nyquist, so the
    // top bins of each octave sit at the decimator's fs/4 transition, where
    // stopband leakage aliases between octaves. Compare the multirate path
    // against the decimation-free oracle across the whole range.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices are small enough for f64"
    )]
    fn matches_oracle_demo_regime() {
        let spec = cqt::design(48000.0, 250.0, 16000.0, 12, &Window::Hann)
            .expect("valid demo-like design");
        let plan = make_plan(&spec);
        let n = plan.fft_length();
        let sr = spec.sample_rate();

        let mut worst = 0.0_f64;
        let mut at = (0.0_f64, 0usize, 0.0_f64, 0.0_f64);
        for &ftone in &[
            700.0, 1500.0, 3000.0, 6000.0, 9000.0, 11000.0, 11500.0, 12000.0, 12500.0, 13000.0,
            14000.0,
        ] {
            let input: Vec<f64> = (0..n)
                .map(|i| (TAU * ftone * i as f64 / sr).sin())
                .collect();
            let mut output = vec![Complex::new(0.0, 0.0); plan.num_bins()];
            plan.execute(&input, &mut output).expect("process");
            let got: Vec<f64> = output.iter().map(|c| c.norm()).collect();
            let want = oracle_magnitudes(&spec, &input);
            let peak = want.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let rel = (g - w).abs() / peak;
                if rel > worst {
                    worst = rel;
                    at = (ftone, k, g, w);
                }
            }
        }
        assert!(
            worst < 2e-2,
            "demo-regime multirate vs decimation-free oracle: worst relative deviation {worst:.4} \
             at tone {} Hz, bin {} (multirate |{:.5}| vs oracle |{:.5}|) — aliasing/leakage the \
             static-tone oracle test (multi_octave_spec) does not exercise",
            at.0,
            at.1,
            at.2,
            at.3
        );
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
            plan.execute(&input, &mut output).expect("process");

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
        plan.execute(&input, &mut output).expect("process");

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
        plan.execute(&input, &mut out1).expect("first");
        plan.execute(&input, &mut out2).expect("second");

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
        plan.execute(&input, &mut complex_out).expect("process");
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
            plan.execute(input, output).expect("trait process");
        }

        let plan = make_plan(&one_octave_spec());
        let mut input = vec![0.0_f64; plan.fft_length()];
        input[0] = 1.0;
        let mut via_trait = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        check(&plan, &input, &mut via_trait);

        let mut direct = vec![Complex::new(0.0, 0.0); plan.num_bins()];
        plan.execute(&input, &mut direct).expect("inherent process");
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
            kernel_len: 5,
        }];
        let spec = CqtSpec::new(44100.0, 8, 2, bins, Window::Hann).expect("spec");
        let plan = make_plan(&spec);
        assert_eq!(plan.num_octaves(), 1, "single bin → one octave");

        let input = vec![0.0_f64; 8];
        let mut output = vec![Complex::new(0.0, 0.0); 1];
        plan.execute(&input, &mut output).expect("process");
        assert!(output[0].norm() < 1e-10, "zero input → zero output");
    }

    // ── Oracle's own numpy validation (reference correctness) ──────
    //
    // Generated by scripts/gen_cqt_process_reference.py.
    // Regenerate with: make gen-cqt-process-reference
    //
    // These pin the single-FFT oracle to numpy so it remains a trustworthy
    // reference for the multirate equivalence checks above.

    #[allow(
        clippy::wildcard_imports,
        reason = "bulk golden-vector import in tests"
    )]
    use omnidsp_testdata::cqt_process_numpy::*;

    fn numpy_spec() -> CqtSpec {
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
        plan.execute(CQT_PROC_ALL_TONES_INPUT, &mut output)
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
        plan.execute(CQT_PROC_TONE_INPUT, &mut output)
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
        plan.execute(CQT_PROC_TWO_TONE_INPUT, &mut output)
            .expect("oracle process");
        assert_complex_approx_eq(&output, CQT_PROC_TWO_TONE_OUTPUT, 1e-10, "oracle two-tone");
    }
}
