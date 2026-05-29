// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` struct — top-level DSP engine with pluggable backends.

use omnidsp_core::design::cqt::CqtSpec;
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::error::Result;
use omnidsp_core::modules::conv::OmniConv;
use omnidsp_core::modules::cqt::OmniCqt;
use omnidsp_core::modules::fir::OmniFir;
use omnidsp_core::modules::resample::OmniResample;
use omnidsp_core::traits::conv::{Conv, ConvSpec};
use omnidsp_core::traits::dft::{Dft, DftSpec};
use omnidsp_core::traits::fir::{Fir, FirSpec};
use omnidsp_core::traits::iir::{Iir, IirSpec};

use crate::config::Config;
use crate::create_plan::CreatePlan;
use crate::dispatch::{DynDft, DynDftPlan, DynIir, DynIirPlan, DynVecOps};
use crate::resolve::{resolve_dft, resolve_iir, resolve_vecops};

// ─── Type aliases ────────────────────────────────────────────────────

// Re-export plan type aliases at crate root via lib.rs.
use omnidsp_core::modules::conv::OmniConvPlan;
use omnidsp_core::modules::cqt::OmniCqtPlan;
use omnidsp_core::modules::fir::OmniFirPlan;
use omnidsp_core::modules::resample::OmniResamplePlan;

/// DFT plan for `f32` signals.
pub type DftPlan32 = DynDftPlan<f32>;
/// DFT plan for `f64` signals.
pub type DftPlan64 = DynDftPlan<f64>;

/// Convolution plan for `f32` signals.
pub type ConvPlan32 = OmniConvPlan<f32, DynDftPlan<f32>, DynVecOps>;
/// Convolution plan for `f64` signals.
pub type ConvPlan64 = OmniConvPlan<f64, DynDftPlan<f64>, DynVecOps>;

/// FIR filter plan for `f32` signals.
pub type FirPlan32 = OmniFirPlan<f32, DynDftPlan<f32>, DynVecOps>;
/// FIR filter plan for `f64` signals.
pub type FirPlan64 = OmniFirPlan<f64, DynDftPlan<f64>, DynVecOps>;

/// IIR filter plan for `f32` signals.
pub type IirPlan32 = DynIirPlan<f32>;
/// IIR filter plan for `f64` signals.
pub type IirPlan64 = DynIirPlan<f64>;

/// Resampling plan for `f32` signals.
pub type ResamplePlan32 = OmniResamplePlan<f32, DynVecOps>;
/// Resampling plan for `f64` signals.
pub type ResamplePlan64 = OmniResamplePlan<f64, DynVecOps>;

/// CQT plan for `f32` signals.
pub type CqtPlan32 = OmniCqtPlan<f32, DynDftPlan<f32>, DynVecOps>;
/// CQT plan for `f64` signals.
pub type CqtPlan64 = OmniCqtPlan<f64, DynDftPlan<f64>, DynVecOps>;

// ─── OmniDSP struct ──────────────────────────────────────────────────

/// Top-level DSP engine with pluggable backends.
///
/// Resolves backend selection once at construction time. All
/// subsequent `create_plan` calls use the resolved primitives.
///
/// # Examples
///
/// ```
/// use omnidsp::OmniDSP;
/// use omnidsp::traits::dft::{DftSpec, DftNorm};
/// use omnidsp::types::Direction;
///
/// let dsp = OmniDSP::new();
/// let plan = dsp.create_plan(&DftSpec::<f64>::new(1024, Direction::Forward, DftNorm::Inverse)).unwrap();
/// ```
#[derive(Debug)]
pub struct OmniDSP {
    dft: DynDft,
    vecops: DynVecOps,
    iir: DynIir,
}

impl OmniDSP {
    /// Create an engine with the default backend configuration.
    ///
    /// Equivalent to `OmniDSP::with_config(Config::default())`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(&Config::default())
    }

    /// Create an engine with a custom backend configuration.
    ///
    /// Backends are resolved during this call — the first available
    /// backend in each priority list wins. If no backend in a list
    /// is available, the Rust fallback is used.
    #[must_use]
    pub fn with_config(config: &Config) -> Self {
        Self {
            dft: resolve_dft(&config.dft),
            vecops: resolve_vecops(&config.vecops),
            iir: resolve_iir(&config.iir),
        }
    }

    /// Create a plan for the given specification.
    ///
    /// The spec type determines which module and backend are used.
    /// The compiler infers the correct `CreatePlan` impl from `S`.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation
    /// fails.
    pub fn create_plan<S>(&self, spec: &S) -> Result<<Self as CreatePlan<S>>::Plan>
    where
        Self: CreatePlan<S>,
    {
        <Self as CreatePlan<S>>::create_plan(self, spec)
    }
}

impl Default for OmniDSP {
    fn default() -> Self {
        Self::new()
    }
}

// ─── CreatePlan impls — primitives ───────────────────────────────────

macro_rules! impl_create_dft {
    ($t:ty) => {
        impl CreatePlan<DftSpec<$t>> for OmniDSP {
            type Plan = DynDftPlan<$t>;

            fn create_plan(&self, spec: &DftSpec<$t>) -> Result<DynDftPlan<$t>> {
                Dft::<$t>::create_plan(&self.dft, spec)
            }
        }
    };
}

impl_create_dft!(f32);
impl_create_dft!(f64);

macro_rules! impl_create_iir {
    ($t:ty) => {
        impl CreatePlan<IirSpec<$t>> for OmniDSP {
            type Plan = DynIirPlan<$t>;

            fn create_plan(&self, spec: &IirSpec<$t>) -> Result<DynIirPlan<$t>> {
                Iir::<$t>::create_plan(&self.iir, spec)
            }
        }
    };
}

impl_create_iir!(f32);
impl_create_iir!(f64);

// ─── CreatePlan impls — composed modules ─────────────────────────────

macro_rules! impl_create_conv {
    ($t:ty) => {
        impl CreatePlan<ConvSpec<$t>> for OmniDSP {
            type Plan = OmniConvPlan<$t, DynDftPlan<$t>, DynVecOps>;

            fn create_plan(&self, spec: &ConvSpec<$t>) -> Result<Self::Plan> {
                let factory = OmniConv::new(self.dft, self.vecops);
                Conv::<$t>::create_plan(&factory, spec)
            }
        }
    };
}

impl_create_conv!(f32);
impl_create_conv!(f64);

macro_rules! impl_create_fir {
    ($t:ty) => {
        impl CreatePlan<FirSpec<$t>> for OmniDSP {
            type Plan = OmniFirPlan<$t, DynDftPlan<$t>, DynVecOps>;

            fn create_plan(&self, spec: &FirSpec<$t>) -> Result<Self::Plan> {
                let factory = OmniFir::new(self.dft, self.vecops);
                Fir::<$t>::create_plan(&factory, spec)
            }
        }
    };
}

impl_create_fir!(f32);
impl_create_fir!(f64);

macro_rules! impl_create_resample {
    ($t:ty) => {
        impl CreatePlan<ResampleSpec<$t>> for OmniDSP {
            type Plan = OmniResamplePlan<$t, DynVecOps>;

            fn create_plan(&self, spec: &ResampleSpec<$t>) -> Result<Self::Plan> {
                let factory = OmniResample::new(self.vecops);
                factory.create_plan(spec)
            }
        }
    };
}

impl_create_resample!(f32);
impl_create_resample!(f64);

macro_rules! impl_create_cqt {
    ($t:ty) => {
        impl CreatePlan<CqtSpec<$t>> for OmniDSP {
            type Plan = OmniCqtPlan<$t, DynDftPlan<$t>, DynVecOps>;

            fn create_plan(&self, spec: &CqtSpec<$t>) -> Result<Self::Plan> {
                let factory = OmniCqt::new(self.dft, self.vecops);
                factory.create_plan(spec)
            }
        }
    };
}

impl_create_cqt!(f32);
impl_create_cqt!(f64);

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use num_complex::Complex;
    use omnidsp_core::design::cqt;
    use omnidsp_core::design::resample::{ResampleQuality, ResampleSpec};
    use omnidsp_core::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
    use omnidsp_core::traits::dft::{DftNorm, DftPlan, DftSpec};
    use omnidsp_core::traits::fir::{FirPlan, FirSpec};
    use omnidsp_core::traits::iir::{IirPlan, IirSpec};
    use omnidsp_core::types::{BiquadSection, Direction, Window};

    use super::*;

    // ─── Unified API (f64) ───────────────────────────────────────────

    #[test]
    fn create_dft_plan_f64() {
        let dsp = OmniDSP::new();
        let spec = DftSpec::<f64>::new(8, Direction::Forward, DftNorm::Inverse);
        let fwd = dsp.create_plan(&spec).expect("forward plan");

        let input: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let mut output = vec![Complex::default(); 8];
        fwd.process(&input, &mut output).expect("forward transform");

        // DC bin should equal sum of inputs = 0+1+2+...+7 = 28
        assert!(
            (output[0].re - 28.0).abs() < 1e-10,
            "DC bin should be 28.0, got {}",
            output[0].re,
        );
    }

    #[test]
    fn create_conv_plan_f64() {
        let dsp = OmniDSP::new();
        let spec = ConvSpec::<f64>::new(3, 2, ConvMethod::Direct);
        let plan = dsp.create_plan(&spec).expect("conv plan");

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
    fn create_fir_plan_f64() {
        let dsp = OmniDSP::new();
        let coeffs: Vec<f64> = vec![0.25, 0.5, 0.25];
        let spec = FirSpec::new(coeffs.clone());
        let mut plan = dsp.create_plan(&spec).expect("FIR plan");

        // Feed an impulse and verify the response matches coefficients.
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

    #[test]
    fn create_iir_plan_f64() {
        let dsp = OmniDSP::new();
        // Simple lowpass: passthrough numerator, mild feedback.
        let section = BiquadSection {
            b0: 0.2_f64,
            b1: 0.2,
            b2: 0.0,
            a1: -0.6,
            a2: 0.0,
        };
        let spec = IirSpec::new(vec![section]);
        let mut plan = dsp.create_plan(&spec).expect("IIR plan");

        // Feed DC (all ones) and verify convergence.
        let input = [1.0_f64; 64];
        let mut output = [0.0; 64];
        plan.process(&input, &mut output).expect("IIR process");

        // The last sample should be close to the DC gain:
        // H(z=1) = (b0 + b1) / (1 + a1) = 0.4 / 0.4 = 1.0
        assert!(
            (output[63] - 1.0).abs() < 1e-6,
            "IIR DC output should converge to 1.0, got {}",
            output[63],
        );
    }

    #[test]
    fn create_resample_plan_f64() {
        let dsp = OmniDSP::new();
        let spec = ResampleSpec::new(
            44100.0_f64,
            48000.0,
            ResampleQuality::new(5).expect("quality"),
            Window::Hamming,
        )
        .expect("resample spec");
        let mut plan = dsp.create_plan(&spec).expect("resample plan");

        // Resample a DC signal — output amplitude should be close to 1.0.
        let input = vec![1.0_f64; 441];
        let mut output = vec![0.0; 480];
        plan.process(&input, &mut output).expect("resample");

        // Check that the middle samples are close to 1.0 (skip edges
        // where filter transients may occur).
        let mid = output.len() / 2;
        assert!(
            (output[mid] - 1.0).abs() < 0.05,
            "resampled DC should be ~1.0 in the middle, got {}",
            output[mid],
        );
    }

    #[test]
    fn create_cqt_plan_f64() {
        let dsp = OmniDSP::new();
        let spec = cqt::design(44100.0, 55.0, 110.0, 12, &Window::<f64>::Hann).expect("CQT design");
        let plan = dsp.create_plan(&spec).expect("CQT plan");

        // Process a frame of silence — output length should equal the
        // number of bins.
        let frame = vec![0.0_f64; spec.fft_length()];
        let mut output = vec![Complex::default(); spec.num_bins()];
        plan.process(&frame, &mut output).expect("CQT process");

        assert_eq!(
            output.len(),
            spec.num_bins(),
            "CQT output length should match number of bins",
        );
    }

    // ─── Type inference ──────────────────────────────────────────────

    #[test]
    fn type_inference_no_turbofish() {
        let dsp = OmniDSP::new();

        // If these compile without type annotations, type inference works.
        let _dft = dsp
            .create_plan(&DftSpec::<f64>::new(
                16,
                Direction::Forward,
                DftNorm::Inverse,
            ))
            .expect("dft");

        let _conv = dsp
            .create_plan(&ConvSpec::<f64>::new(4, 4, ConvMethod::Auto))
            .expect("conv");

        let _fir = dsp
            .create_plan(&FirSpec::new(vec![1.0_f64, 0.5]))
            .expect("fir");

        let _iir = dsp
            .create_plan(&IirSpec::new(vec![BiquadSection {
                b0: 1.0_f64,
                b1: 0.0,
                b2: 0.0,
                a1: 0.0,
                a2: 0.0,
            }]))
            .expect("iir");

        let _resample = dsp
            .create_plan(
                &ResampleSpec::new(
                    48000.0_f64,
                    16000.0,
                    ResampleQuality::new(3).expect("quality"),
                    Window::Hann,
                )
                .expect("spec"),
            )
            .expect("resample");

        let _cqt = dsp
            .create_plan(
                &cqt::design(44100.0, 100.0, 200.0, 12, &Window::<f64>::Hann).expect("cqt design"),
            )
            .expect("cqt");
    }

    // ─── Config variants ─────────────────────────────────────────────

    #[test]
    fn omnidsp_default() {
        let dsp = OmniDSP::new();
        let plan = dsp
            .create_plan(&DftSpec::<f64>::new(
                4,
                Direction::Forward,
                DftNorm::Inverse,
            ))
            .expect("plan from new()");

        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output = [Complex::default(); 4];
        plan.process(&input, &mut output).expect("process");

        assert!((output[0].re - 10.0).abs() < 1e-10, "DC bin should be 10.0");
    }

    #[test]
    fn omnidsp_with_config() {
        let dsp = OmniDSP::with_config(&Config::default());
        let plan = dsp
            .create_plan(&DftSpec::<f64>::new(
                4,
                Direction::Forward,
                DftNorm::Inverse,
            ))
            .expect("plan from with_config()");

        let input = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut output = [Complex::default(); 4];
        plan.process(&input, &mut output).expect("process");

        assert!((output[0].re - 10.0).abs() < 1e-10, "DC bin should be 10.0");
    }

    // ─── f32 coverage ────────────────────────────────────────────────

    #[test]
    fn create_dft_plan_f32() {
        let dsp = OmniDSP::new();
        let fwd = dsp
            .create_plan(&DftSpec::<f32>::new(
                4,
                Direction::Forward,
                DftNorm::Inverse,
            ))
            .expect("fwd");
        let inv = dsp
            .create_plan(&DftSpec::<f32>::new(
                4,
                Direction::Inverse,
                DftNorm::Inverse,
            ))
            .expect("inv");

        let input = [
            Complex::new(1.0_f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd.process(&input, &mut freq).expect("forward");
        inv.process(&freq, &mut recovered).expect("inverse");

        for (i, (got, want)) in recovered.iter().zip(&input).enumerate() {
            assert!(
                (got.re - want.re).abs() < 1e-5 && (got.im - want.im).abs() < 1e-5,
                "round-trip mismatch at {i}: got ({}, {}), want ({}, {})",
                got.re,
                got.im,
                want.re,
                want.im,
            );
        }
    }

    #[test]
    fn create_fir_plan_f32() {
        let dsp = OmniDSP::new();
        let coeffs = vec![0.5_f32, 0.5];
        let spec = FirSpec::new(coeffs.clone());
        let mut plan = dsp.create_plan(&spec).expect("FIR plan f32");

        let mut input = vec![0.0_f32; 8];
        input[0] = 1.0;
        let mut output = vec![0.0; 8];
        plan.process(&input, &mut output).expect("FIR process f32");

        for (i, &c) in coeffs.iter().enumerate() {
            assert!(
                (output[i] - c).abs() < 1e-5,
                "f32 impulse response[{i}] should be {c}, got {}",
                output[i],
            );
        }
    }

    // ─── Default impl ────────────────────────────────────────────────

    #[test]
    fn omnidsp_implements_default() {
        let dsp = OmniDSP::default();
        let _plan = dsp
            .create_plan(&DftSpec::<f64>::new(
                8,
                Direction::Forward,
                DftNorm::Inverse,
            ))
            .expect("default should create plans");
    }
}
