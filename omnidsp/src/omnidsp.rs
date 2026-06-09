// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` struct and `RustBackend` — top-level DSP engine with pluggable backends.

use omnidsp_core::design::cqt::CqtSpec;
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::error::Result;
use omnidsp_core::modules::hilbert::HilbertSpec;
use omnidsp_core::modules::xcorr::CrossCorrSpec;
use omnidsp_core::scalar::ScalarVecOps;
use omnidsp_core::traits::conv::ConvSpec;
use omnidsp_core::traits::dct::DctSpec;
use omnidsp_core::traits::dft::DftC2cSpec;
use omnidsp_core::traits::fir::FirSpec;
use omnidsp_core::traits::iir::IirSpec;
use omnidsp_rustfft::{RustDftC2c, RustDftC2r, RustDftR2c};

use omnidsp_core::create::CreatePlan;

// ─── RustBackend ───────────────────────────────────────────────────────

/// Pure Rust fallback backend.
///
/// Combines [`RustDftC2c`] (wrapping `RustFFT`) and [`RustDftR2c`] /
/// [`RustDftC2r`] (wrapping `realfft`) with [`ScalarVecOps`] (scalar loops,
/// LLVM auto-vectorized).  Builds on every platform with no external
/// dependencies.
#[derive(Debug, Clone, Copy)]
pub struct RustBackend {
    /// Complex-to-complex DFT factory (`RustFFT` wrapper).
    pub(crate) dft: RustDftC2c,
    /// Real-to-complex DFT factory (`realfft` wrapper).
    #[allow(
        dead_code,
        reason = "wired into the real-input module CreatePlan impls by the macro in 05f / item 4"
    )]
    pub(crate) dftr2c: RustDftR2c,
    /// Complex-to-real DFT factory (`realfft` wrapper).
    #[allow(
        dead_code,
        reason = "wired into the real-input module CreatePlan impls by the macro in 05f / item 4"
    )]
    pub(crate) dftc2r: RustDftC2r,
    /// Vector operations (scalar fallback).
    pub(crate) vecops: ScalarVecOps,
}

impl RustBackend {
    /// Create a new Rust fallback backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dft: RustDftC2c,
            dftr2c: RustDftR2c,
            dftc2r: RustDftC2r,
            vecops: ScalarVecOps,
        }
    }
}

impl Default for RustBackend {
    fn default() -> Self {
        Self::new()
    }
}

omnidsp_macros::impl_generic_backend! {
    backend: RustBackend,
    dft: RustDftC2c,
    vecops: ScalarVecOps,
}

// ─── OmniDSP struct ─────────────────────────────────────────────────

/// Top-level DSP engine with a pluggable backend.
///
/// `OmniDSP<B>` wraps a backend `B` and provides a universal
/// [`create_plan`](Self::create_plan) method plus convenience wrappers
/// for each module.  The spec type drives compile-time dispatch.
///
/// # Examples
///
/// ```
/// use omnidsp::OmniDSP;
/// use omnidsp::traits::conv::{ConvSpec, ConvMethod};
///
/// let dsp = OmniDSP::rust();
/// let plan = dsp.create_plan(&ConvSpec::<f64>::new(3, 2, ConvMethod::Direct)).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OmniDSP<B> {
    backend: B,
}

impl<B> OmniDSP<B> {
    /// Create an engine wrapping the given backend.
    #[must_use]
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Create a plan from the given specification.
    ///
    /// The spec type `S` drives dispatch — the backend must implement
    /// `CreatePlan<S>` for the spec type.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn create_plan<S>(&self, spec: &S) -> Result<<B as CreatePlan<S>>::Plan>
    where
        B: CreatePlan<S>,
    {
        self.backend.create_plan(spec)
    }

    /// Create a convolution plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn conv<T>(&self, spec: &ConvSpec<T>) -> Result<<B as CreatePlan<ConvSpec<T>>>::Plan>
    where
        B: CreatePlan<ConvSpec<T>>,
    {
        self.create_plan(spec)
    }

    /// Create a FIR filter plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn fir<T>(&self, spec: &FirSpec<T>) -> Result<<B as CreatePlan<FirSpec<T>>>::Plan>
    where
        B: CreatePlan<FirSpec<T>>,
    {
        self.create_plan(spec)
    }

    /// Create a resampler plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn resample<T>(
        &self,
        spec: &ResampleSpec<T>,
    ) -> Result<<B as CreatePlan<ResampleSpec<T>>>::Plan>
    where
        B: CreatePlan<ResampleSpec<T>>,
    {
        self.create_plan(spec)
    }

    /// Create a CQT plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn cqt<T>(&self, spec: &CqtSpec<T>) -> Result<<B as CreatePlan<CqtSpec<T>>>::Plan>
    where
        B: CreatePlan<CqtSpec<T>>,
    {
        self.create_plan(spec)
    }

    /// Create a DFT plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn dft<T>(&self, spec: &DftC2cSpec<T>) -> Result<<B as CreatePlan<DftC2cSpec<T>>>::Plan>
    where
        B: CreatePlan<DftC2cSpec<T>>,
    {
        self.create_plan(spec)
    }

    /// Create an IIR filter plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn iir<T>(&self, spec: &IirSpec<T>) -> Result<<B as CreatePlan<IirSpec<T>>>::Plan>
    where
        B: CreatePlan<IirSpec<T>>,
    {
        self.create_plan(spec)
    }

    /// Create a Hilbert transform (analytic signal) plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn hilbert<T>(
        &self,
        spec: &HilbertSpec<T>,
    ) -> Result<<B as CreatePlan<HilbertSpec<T>>>::Plan>
    where
        B: CreatePlan<HilbertSpec<T>>,
    {
        self.create_plan(spec)
    }

    /// Create a DCT plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn dct<T>(&self, spec: &DctSpec<T>) -> Result<<B as CreatePlan<DctSpec<T>>>::Plan>
    where
        B: CreatePlan<DctSpec<T>>,
    {
        self.create_plan(spec)
    }

    /// Create a cross-correlation plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn xcorr<T>(
        &self,
        spec: &CrossCorrSpec<T>,
    ) -> Result<<B as CreatePlan<CrossCorrSpec<T>>>::Plan>
    where
        B: CreatePlan<CrossCorrSpec<T>>,
    {
        self.create_plan(spec)
    }
}

// ─── Named constructors ─────────────────────────────────────────────

impl OmniDSP<RustBackend> {
    /// Create an engine using the pure Rust fallback backend.
    ///
    /// Uses [`RustDftC2c`] (`RustFFT`) and [`ScalarVecOps`] (scalar loops,
    /// LLVM auto-vectorized).
    #[must_use]
    pub const fn rust() -> Self {
        Self::new(RustBackend::new())
    }
}

impl crate::Auto {
    /// Create an engine using the best compiled-in backend.
    ///
    /// Selects the best backend at compile time via the [`Best`](crate::Best)
    /// type alias.  Currently [`RustBackend`]; updated when vendor features
    /// (IPP, Accelerate, oneMKL) are enabled.
    #[must_use]
    pub const fn auto() -> Self {
        Self::new(crate::Best::new())
    }
}

impl Default for crate::Auto {
    fn default() -> Self {
        Self::auto()
    }
}
