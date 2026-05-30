// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` struct — top-level DSP engine with pluggable backends.

use omnidsp_core::design::cqt::CqtSpec;
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::error::Result;
use omnidsp_core::scalar::ScalarVecOps;
use omnidsp_core::traits::conv::ConvSpec;
use omnidsp_core::traits::dft::DftSpec;
use omnidsp_core::traits::fir::FirSpec;
use omnidsp_core::traits::iir::IirSpec;
use omnidsp_rustfft::RustDft;

use crate::create::{CreateConv, CreateCqt, CreateDft, CreateFir, CreateIir, CreateResampler};
use crate::generic::Generic;

// ─── OmniDSP struct ─────────────────────────────────────────────────

/// Top-level DSP engine with a pluggable backend.
///
/// `OmniDSP<B>` wraps a backend `B` and provides named methods for
/// creating module plans.  Each method bounds on the specific `Create*`
/// trait it needs — backends only implement the traits they support.
///
/// # Examples
///
/// ```
/// use omnidsp::OmniDSP;
/// use omnidsp::traits::conv::{ConvSpec, ConvMethod};
///
/// let dsp = OmniDSP::rust();
/// let plan = dsp.conv(&ConvSpec::<f64>::new(3, 2, ConvMethod::Direct)).unwrap();
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

    /// Create a convolution plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn conv<T>(&self, spec: &ConvSpec<T>) -> Result<B::Conv>
    where
        B: CreateConv<T>,
    {
        self.backend.create_conv(spec)
    }

    /// Create a FIR filter plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn fir<T>(&self, spec: &FirSpec<T>) -> Result<B::Fir>
    where
        B: CreateFir<T>,
    {
        self.backend.create_fir(spec)
    }

    /// Create a resampler plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn resample<T>(&self, spec: &ResampleSpec<T>) -> Result<B::Resampler>
    where
        B: CreateResampler<T>,
    {
        self.backend.create_resampler(spec)
    }

    /// Create a CQT plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn cqt<T>(&self, spec: &CqtSpec<T>) -> Result<B::Cqt>
    where
        B: CreateCqt<T>,
    {
        self.backend.create_cqt(spec)
    }

    /// Create a DFT plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn dft<T>(&self, spec: &DftSpec<T>) -> Result<B::Dft>
    where
        B: CreateDft<T>,
    {
        self.backend.create_dft(spec)
    }

    /// Create an IIR filter plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn iir<T>(&self, spec: &IirSpec<T>) -> Result<B::Iir>
    where
        B: CreateIir<T>,
    {
        self.backend.create_iir(spec)
    }
}

// ─── Named constructors ─────────────────────────────────────────────

impl OmniDSP<Generic<RustDft, ScalarVecOps>> {
    /// Create an engine using the pure Rust fallback backend.
    ///
    /// Uses [`RustDft`] (`RustFFT`) and [`ScalarVecOps`] (scalar loops,
    /// LLVM auto-vectorized).
    #[must_use]
    pub const fn rust() -> Self {
        Self::new(Generic(RustDft, ScalarVecOps))
    }
}

impl Default for OmniDSP<Generic<RustDft, ScalarVecOps>> {
    fn default() -> Self {
        Self::rust()
    }
}
