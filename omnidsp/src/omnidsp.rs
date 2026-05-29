// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` struct — top-level DSP engine with pluggable backends.

use omnidsp_core::design::cqt::CqtSpec;
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::error::Result;
use omnidsp_core::traits::conv::ConvSpec;
use omnidsp_core::traits::fir::FirSpec;
use omnidsp_rust::{RustDft, RustVecOps};

use crate::create::{CreateConv, CreateCqt, CreateFir, CreateResampler};
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
}

// ─── Named constructors ─────────────────────────────────────────────

impl OmniDSP<Generic<RustDft, RustVecOps>> {
    /// Create an engine using the pure Rust fallback backend.
    ///
    /// Uses [`RustDft`] (`RustFFT`) and [`RustVecOps`] (scalar loops,
    /// LLVM auto-vectorized).
    #[must_use]
    pub const fn rust() -> Self {
        Self::new(Generic(RustDft, RustVecOps))
    }
}

impl Default for OmniDSP<Generic<RustDft, RustVecOps>> {
    fn default() -> Self {
        Self::rust()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use omnidsp_core::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
    use omnidsp_core::traits::fir::{FirPlan, FirSpec};
    use omnidsp_rust::{RustDft, RustVecOps};

    use super::*;
    use crate::{Auto, CreateConv};

    #[test]
    fn rust_constructor_conv() {
        let dsp = OmniDSP::rust();
        let spec = ConvSpec::<f64>::new(3, 2, ConvMethod::Direct);
        let plan = dsp.conv(&spec).expect("conv plan");

        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 1.0];
        let mut output = [0.0; 4];
        plan.process(&a, &b, &mut output).expect("convolve");

        assert!(
            (output[0] - 1.0).abs() < 1e-12
                && (output[1] - 3.0).abs() < 1e-12
                && (output[2] - 5.0).abs() < 1e-12
                && (output[3] - 3.0).abs() < 1e-12,
            "conv output should be [1, 3, 5, 3], got {output:?}",
        );
    }

    #[test]
    fn auto_default() {
        let dsp = Auto::default();
        let spec = ConvSpec::<f64>::new(2, 2, ConvMethod::Direct);
        let _plan = dsp.conv(&spec).expect("conv plan via Auto::default()");
    }

    #[test]
    fn new_with_generic() {
        let dsp = OmniDSP::new(Generic(RustDft, RustVecOps));
        let spec = ConvSpec::<f32>::new(2, 2, ConvMethod::Direct);
        let _plan = dsp.conv(&spec).expect("conv plan via new(Generic)");
    }

    #[test]
    fn fir_method() {
        let dsp = OmniDSP::rust();
        let coeffs = vec![0.25_f64, 0.5, 0.25];
        let spec = FirSpec::new(coeffs.clone());
        let mut plan = dsp.fir(&spec).expect("FIR plan");

        let mut input = vec![0.0_f64; 8];
        input[0] = 1.0;
        let mut output = vec![0.0; 8];
        plan.process(&input, &mut output).expect("FIR process");

        for (i, &c) in coeffs.iter().enumerate() {
            assert!(
                (output[i] - c).abs() < 1e-12,
                "impulse response[{i}] should be {c}, got {}",
                output[i],
            );
        }
    }

    // ─── Macro test ─────────────────────────────────────────────────

    struct MacroTestBackend {
        dft: RustDft,
        vecops: RustVecOps,
    }

    crate::impl_generic_backend! {
        backend: MacroTestBackend,
        dft: RustDft,
        vecops: RustVecOps,
    }

    #[test]
    fn macro_generates_create_conv() {
        let backend = MacroTestBackend {
            dft: RustDft,
            vecops: RustVecOps,
        };
        let spec = ConvSpec::<f64>::new(3, 2, ConvMethod::Direct);
        let plan = CreateConv::create_conv(&backend, &spec).expect("macro conv plan");

        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 1.0];
        let mut output = [0.0; 4];
        plan.process(&a, &b, &mut output).expect("convolve");

        assert!(
            (output[0] - 1.0).abs() < 1e-12,
            "macro-generated conv should produce correct output",
        );
    }
}
