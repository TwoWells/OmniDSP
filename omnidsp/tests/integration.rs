// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Integration tests for the generic `OmniDSP<B>` dispatch API.
//!
//! Exercises every module method through `OmniDSP::rust()`,
//! `OmniDSP::auto()`, and `Auto::default()`, for both `f32` and `f64`.
//! Also verifies that `impl_generic_backend!` produces working
//! `CreatePlan<S>` implementations.

#![allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]

use std::f64::consts::TAU;

use num_complex::Complex;
use num_traits::Float;

use omnidsp::{Auto, OmniDSP, RustBackend};
use omnidsp_core::create::{CreatePlan, CreateProc};
use omnidsp_core::design::cqt::{self, CqtBinSpec, CqtSpec};
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::dispatch::Backend;
use omnidsp_core::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
use omnidsp_core::traits::dft::{
    DftC2cPlan, DftC2cSpec, DftC2rPlan, DftC2rSpec, DftNorm, DftR2cPlan, DftR2cSpec,
};
use omnidsp_core::traits::fir::{FirFilter, FirMeta, FirProcessor, FirSpec, FirStrategy};
use omnidsp_core::traits::iir::{IirProcessor, IirSpec};
use omnidsp_core::types::{BiquadSection, Direction};
use omnidsp_core::window::Window;
use omnidsp_rustfft::{RustDftC2c, RustDftC2r, RustDftR2c};

// ─── Helpers ───────────────────────────────────────────────────────────

fn assert_approx<T: Float + std::fmt::Debug>(actual: &[T], expected: &[T], tol: T, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: got {a:?}, expected {e:?}",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DFT
// ═══════════════════════════════════════════════════════════════════════

/// Forward-then-inverse round-trip should recover the original signal.
fn dft_round_trip<T, B>(dsp: &OmniDSP<B>, tol: T)
where
    T: Float
        + Send
        + Sync
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign,
    B: CreatePlan<DftC2cSpec> + Backend<T>,
    B::Plan<T>: DftC2cPlan<T>,
{
    let n = 8;
    let fwd_spec = DftC2cSpec::new(n, Direction::Forward, DftNorm::Inverse).expect("forward spec");
    let fwd = dsp.dft_c2c(&fwd_spec).expect("forward DFT plan");
    let inv_spec = DftC2cSpec::new(n, Direction::Inverse, DftNorm::Inverse).expect("inverse spec");
    let inv = dsp.dft_c2c(&inv_spec).expect("inverse DFT plan");

    let zero = Complex::new(T::zero(), T::zero());
    let input: Vec<Complex<T>> = (0..n)
        .map(|i| Complex::new(T::from(i).expect("from usize"), T::zero()))
        .collect();

    let mut spectrum = vec![zero; n];
    fwd.execute(&input, &mut spectrum).expect("forward FFT");

    let mut recovered = vec![zero; n];
    inv.execute(&spectrum, &mut recovered).expect("inverse FFT");

    let actual: Vec<T> = recovered.iter().map(|c| c.re).collect();
    let expected: Vec<T> = input.iter().map(|c| c.re).collect();
    assert_approx(&actual, &expected, tol, "DFT round-trip");
}

#[test]
fn rust_dft_f64() {
    dft_round_trip(&OmniDSP::rust(), 1e-12);
}

#[test]
fn rust_dft_f32() {
    dft_round_trip::<f32, _>(&OmniDSP::rust(), 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════
// Convolution
// ═══════════════════════════════════════════════════════════════════════

/// Convolve [1,2,3] * [1,1] = [1,3,5,3].
fn conv_verify<T, B>(dsp: &OmniDSP<B>, tol: T)
where
    T: Float
        + Send
        + Sync
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign,
    B: CreatePlan<ConvSpec> + Backend<T>,
    B::Plan<T>: ConvPlan<T>,
{
    let spec = ConvSpec::new(3, 2, ConvMethod::Direct).expect("valid conv spec");
    let plan = dsp.conv(&spec).expect("conv plan");

    let one = T::one();
    let two = one + one;
    let three = two + one;
    let five = three + two;

    let a = [one, two, three];
    let b = [one, one];
    let mut output = [T::zero(); 4];
    plan.execute(&a, &b, &mut output).expect("convolve");

    assert_approx(&output, &[one, three, five, three], tol, "conv");
}

#[test]
fn rust_conv_f64() {
    conv_verify(&OmniDSP::rust(), 1e-12);
}

#[test]
fn rust_conv_f32() {
    conv_verify::<f32, _>(&OmniDSP::rust(), 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════
// FIR filter
// ═══════════════════════════════════════════════════════════════════════

/// Impulse response of a 3-tap FIR should match the coefficients.
fn fir_impulse<T, B>(dsp: &OmniDSP<B>, tol: T)
where
    T: Float
        + Send
        + Sync
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign,
    B: CreateProc<FirSpec> + Backend<T>,
    B::Proc<T>: FirProcessor<T>,
{
    let quarter = T::from(0.25).expect("0.25");
    let half = T::from(0.5).expect("0.5");
    let coeffs = [quarter, half, quarter];
    let filter =
        FirFilter::new(vec![0.25, 0.5, 0.25], FirMeta::unknown()).expect("valid fir filter");
    let spec = FirSpec::new(filter, FirStrategy::Auto);
    let mut plan = dsp.fir(&spec).expect("FIR plan");

    let mut input = vec![T::zero(); 8];
    input[0] = T::one();
    let mut output = vec![T::zero(); 8];
    plan.process(&input, &mut output).expect("FIR process");

    for (i, &c) in coeffs.iter().enumerate() {
        assert!(
            (output[i] - c).abs() < tol,
            "FIR impulse[{i}]: got {:?}, expected {:?}",
            output[i],
            c,
        );
    }
}

#[test]
fn rust_fir_f64() {
    fir_impulse(&OmniDSP::rust(), 1e-12);
}

#[test]
fn rust_fir_f32() {
    fir_impulse::<f32, _>(&OmniDSP::rust(), 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════
// IIR filter
// ═══════════════════════════════════════════════════════════════════════

/// Unity passthrough biquad: y[n] = x[n].
fn iir_passthrough<T, B>(dsp: &OmniDSP<B>, tol: T)
where
    T: Float
        + Send
        + Sync
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign,
    B: CreateProc<IirSpec> + Backend<T>,
    B::Proc<T>: IirProcessor<T>,
{
    let spec = IirSpec::new(vec![BiquadSection {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    }])
    .expect("valid iir spec");
    let mut plan = dsp.iir(&spec).expect("IIR plan");

    let input: Vec<T> = (0..8).map(|i| T::from(i).expect("from usize")).collect();
    let mut output = vec![T::zero(); 8];
    plan.process(&input, &mut output).expect("IIR process");

    assert_approx(&output, &input, tol, "IIR passthrough");
}

/// First-order lowpass: H(z) = 0.5(1 + z⁻¹).
/// Impulse response: [0.5, 0.5, 0, 0, …].
fn iir_first_order<T, B>(dsp: &OmniDSP<B>, tol: T)
where
    T: Float
        + Send
        + Sync
        + std::fmt::Debug
        + num_traits::FromPrimitive
        + 'static
        + std::ops::AddAssign
        + std::ops::MulAssign,
    B: CreateProc<IirSpec> + Backend<T>,
    B::Proc<T>: IirProcessor<T>,
{
    let half = T::from(0.5).expect("0.5");
    let spec = IirSpec::new(vec![BiquadSection {
        b0: 0.5,
        b1: 0.5,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    }])
    .expect("valid iir spec");
    let mut plan = dsp.iir(&spec).expect("IIR plan");

    let mut input = vec![T::zero(); 8];
    input[0] = T::one();
    let mut output = vec![T::zero(); 8];
    plan.process(&input, &mut output).expect("IIR process");

    assert!(
        (output[0] - half).abs() < tol,
        "IIR first-order[0]: got {:?}, expected {:?}",
        output[0],
        half,
    );
    assert!(
        (output[1] - half).abs() < tol,
        "IIR first-order[1]: got {:?}, expected {:?}",
        output[1],
        half,
    );
    for (i, &val) in output.iter().enumerate().skip(2) {
        assert!(
            val.abs() < tol,
            "IIR first-order[{i}]: expected ~0, got {val:?}",
        );
    }
}

#[test]
fn rust_iir_passthrough_f64() {
    iir_passthrough(&OmniDSP::rust(), 1e-12);
}

#[test]
fn rust_iir_passthrough_f32() {
    iir_passthrough::<f32, _>(&OmniDSP::rust(), 1e-5);
}

#[test]
fn rust_iir_first_order_f64() {
    iir_first_order(&OmniDSP::rust(), 1e-12);
}

#[test]
fn rust_iir_first_order_f32() {
    iir_first_order::<f32, _>(&OmniDSP::rust(), 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════
// Resampling
// ═══════════════════════════════════════════════════════════════════════

/// 2× upsample with a simple prototype filter: output count should be
/// exactly 2× input count (streaming mode).
fn resample_upsample_f64(dsp: &OmniDSP<RustBackend>) {
    let filter =
        FirFilter::new(vec![0.5, 1.0, 0.5], FirMeta::unknown()).expect("non-empty prototype");
    let spec = ResampleSpec::new(filter, 2, 1).expect("valid resample spec");
    let mut plan = dsp.resample(&spec).expect("resample plan");

    let input: Vec<f64> = (0..16).map(f64::from).collect();
    let mut output = vec![0.0; 64];
    let written = plan.process(&input, &mut output).expect("resample");
    assert_eq!(written, 32, "2× upsample should produce 2× samples");
}

#[allow(
    clippy::cast_precision_loss,
    reason = "small test indices fit in f32 mantissa"
)]
fn resample_upsample_f32(dsp: &OmniDSP<RustBackend>) {
    let filter =
        FirFilter::new(vec![0.5, 1.0, 0.5], FirMeta::unknown()).expect("non-empty prototype");
    let spec = ResampleSpec::new(filter, 2, 1).expect("valid resample spec");
    let mut plan = dsp.resample(&spec).expect("resample plan f32");

    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut output = vec![0.0_f32; 64];
    let written = plan.process(&input, &mut output).expect("resample");
    assert_eq!(written, 32, "2× upsample f32 should produce 2× samples");
}

#[test]
fn rust_resample_f64() {
    resample_upsample_f64(&OmniDSP::rust());
}

#[test]
fn rust_resample_f32() {
    resample_upsample_f32(&OmniDSP::rust());
}

// ═══════════════════════════════════════════════════════════════════════
// CQT
// ═══════════════════════════════════════════════════════════════════════

/// Construct a minimal CQT plan and process a zero-input frame.
fn cqt_smoke_f64(dsp: &OmniDSP<RustBackend>) {
    let bins = vec![CqtBinSpec {
        frequency: 440.0,
        kernel_len: 5,
    }];
    let spec = CqtSpec::new(44100.0, 8, 2, bins, Window::Hann).expect("CQT spec");
    let plan = dsp.cqt(&spec).expect("CQT plan");

    let input = vec![0.0_f64; 8];
    let mut output = vec![Complex::new(0.0, 0.0); 1];
    plan.execute(&input, &mut output).expect("CQT process");

    assert!(
        output[0].re.abs() < 1e-10,
        "CQT of zero input should be ~zero, got {:?}",
        output[0],
    );
}

fn cqt_smoke_f32(dsp: &OmniDSP<RustBackend>) {
    let bins = vec![CqtBinSpec {
        frequency: 440.0,
        kernel_len: 5,
    }];
    let spec = CqtSpec::new(44100.0, 8, 2, bins, Window::Hann).expect("CQT spec");
    let plan = dsp.cqt(&spec).expect("CQT plan");

    let input = vec![0.0_f32; 8];
    let mut output = vec![Complex::new(0.0_f32, 0.0); 1];
    plan.execute(&input, &mut output).expect("CQT process");

    assert!(
        output[0].re.abs() < 1e-4,
        "CQT f32 of zero input should be ~zero, got {:?}",
        output[0],
    );
}

/// CQT of a pure tone should produce its strongest response in the
/// bin closest to that frequency.  Uses `cqt::design()` for proper
/// kernel lengths and windowing.
#[allow(
    clippy::cast_precision_loss,
    reason = "sample indices are small enough for f64"
)]
fn cqt_tone_detect_f64(dsp: &OmniDSP<RustBackend>) {
    let sr = 8000.0;
    let freq = 440.0;
    // Design covers one octave around the tone (220–880 Hz).
    let spec = cqt::design(sr, 220.0, 880.0, 12, &Window::Hann).expect("CQT design");
    let plan = dsp.cqt(&spec).expect("CQT plan");
    let fft_len = plan.fft_length();
    let num_bins = plan.num_bins();

    let input: Vec<f64> = (0..fft_len)
        .map(|n| (TAU * freq * n as f64 / sr).sin())
        .collect();
    let mut output = vec![Complex::new(0.0, 0.0); num_bins];
    plan.execute(&input, &mut output).expect("CQT process");

    // Find the bin with the strongest magnitude.
    let (peak_bin, _) = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.re.hypot(a.im)
                .partial_cmp(&b.re.hypot(b.im))
                .expect("no NaN")
        })
        .expect("at least one bin");

    let peak_freq = plan.bin_frequencies()[peak_bin];
    // The peak bin should be within one semitone of 440 Hz.
    let ratio = peak_freq / freq;
    assert!(
        (0.94..=1.06).contains(&ratio),
        "peak bin at {peak_freq} Hz is too far from {freq} Hz (ratio {ratio})",
    );
}

/// Same tone detection test for f32.
#[allow(
    clippy::cast_possible_truncation,
    reason = "sin() values are in [-1, 1], safe for f32"
)]
#[allow(
    clippy::cast_precision_loss,
    reason = "sample indices are small enough for f64"
)]
fn cqt_tone_detect_f32(dsp: &OmniDSP<RustBackend>) {
    let sr = 8000.0;
    let freq = 440.0;
    let spec = cqt::design(sr, 220.0, 880.0, 12, &Window::Hann).expect("CQT design");
    let plan = dsp.cqt(&spec).expect("CQT plan");
    let fft_len = plan.fft_length();
    let num_bins = plan.num_bins();

    let input: Vec<f32> = (0..fft_len)
        .map(|n| (TAU * freq * n as f64 / sr).sin() as f32)
        .collect();
    let mut output = vec![Complex::new(0.0_f32, 0.0); num_bins];
    plan.execute(&input, &mut output).expect("CQT process");

    let (peak_bin, _) = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.re.hypot(a.im)
                .partial_cmp(&b.re.hypot(b.im))
                .expect("no NaN")
        })
        .expect("at least one bin");

    let peak_freq = plan.bin_frequencies()[peak_bin];
    let ratio = peak_freq / freq;
    assert!(
        (0.94..=1.06).contains(&ratio),
        "peak bin at {peak_freq} Hz is too far from {freq} Hz (ratio {ratio})",
    );
}

#[test]
fn rust_cqt_smoke_f64() {
    cqt_smoke_f64(&OmniDSP::rust());
}

#[test]
fn rust_cqt_smoke_f32() {
    cqt_smoke_f32(&OmniDSP::rust());
}

#[test]
fn rust_cqt_tone_f64() {
    cqt_tone_detect_f64(&OmniDSP::rust());
}

#[test]
fn rust_cqt_tone_f32() {
    cqt_tone_detect_f32(&OmniDSP::rust());
}

// ═══════════════════════════════════════════════════════════════════════
// create_plan — universal method
// ═══════════════════════════════════════════════════════════════════════

/// `create_plan` should work for all spec types via the universal method.
#[test]
fn create_plan_all_modules_f64() {
    let dsp = OmniDSP::rust();

    dft_round_trip(&dsp, 1e-12);
    conv_verify(&dsp, 1e-12);
    fir_impulse(&dsp, 1e-12);
    iir_passthrough(&dsp, 1e-12);
    iir_first_order(&dsp, 1e-12);
    resample_upsample_f64(&dsp);
    cqt_smoke_f64(&dsp);
    cqt_tone_detect_f64(&dsp);
}

#[test]
fn create_plan_all_modules_f32() {
    let dsp = OmniDSP::rust();

    dft_round_trip::<f32, _>(&dsp, 1e-5);
    conv_verify::<f32, _>(&dsp, 1e-5);
    fir_impulse::<f32, _>(&dsp, 1e-5);
    iir_passthrough::<f32, _>(&dsp, 1e-5);
    iir_first_order::<f32, _>(&dsp, 1e-5);
    resample_upsample_f32(&dsp);
    cqt_smoke_f32(&dsp);
    cqt_tone_detect_f32(&dsp);
}

// ═══════════════════════════════════════════════════════════════════════
// Auto type alias
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn auto_default_conv() {
    let dsp = Auto::default();
    conv_verify(&dsp, 1e-12);
}

#[test]
fn auto_default_dft() {
    let dsp = Auto::default();
    dft_round_trip(&dsp, 1e-12);
}

#[test]
fn auto_default_iir() {
    let dsp = Auto::default();
    iir_passthrough(&dsp, 1e-12);
}

#[test]
fn auto_default_fir() {
    let dsp = Auto::default();
    fir_impulse(&dsp, 1e-12);
}

#[test]
fn auto_default_resample() {
    resample_upsample_f64(&Auto::default());
}

#[test]
fn auto_default_cqt() {
    cqt_smoke_f64(&Auto::default());
}

#[test]
fn auto_constructor() {
    let dsp = OmniDSP::auto();
    conv_verify(&dsp, 1e-12);
    dft_round_trip(&dsp, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════
// impl_generic_backend! macro
// ═══════════════════════════════════════════════════════════════════════

/// A test backend that exercises the macro code path.  The macro
/// generates all `CreatePlan<S>` impls by delegating to `omnidsp-core`
/// modules.
#[derive(Clone)]
struct MacroTestBackend {
    dftc2c: RustDftC2c,
    dftr2c: RustDftR2c,
    dftc2r: RustDftC2r,
}

impl omnidsp_core::traits::vecops::VecOps<f32> for MacroTestBackend {}
impl omnidsp_core::traits::vecops::VecOps<f64> for MacroTestBackend {}

// Omits `iir:` — the macro materializes the `ScalarIir` default inline, with
// no backing struct field.  The backend supplies `VecOps` via its own empty
// impls above.
omnidsp_macros::impl_generic_backend! {
    backend: MacroTestBackend,
    dftc2c: RustDftC2c,
    dftr2c: RustDftR2c,
    dftc2r: RustDftC2r,
}

const fn macro_backend() -> MacroTestBackend {
    MacroTestBackend {
        dftc2c: RustDftC2c,
        dftr2c: RustDftR2c,
        dftc2r: RustDftC2r,
    }
}

#[test]
fn macro_create_conv_f64() {
    let b = macro_backend();
    let spec = ConvSpec::new(3, 2, ConvMethod::Direct).expect("valid conv spec");
    let plan = CreatePlan::create_plan::<f64>(&b, &spec).expect("macro conv plan");

    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 1.0];
    let mut output = [0.0; 4];
    plan.execute(&a, &b, &mut output).expect("convolve");
    assert_approx(&output, &[1.0, 3.0, 5.0, 3.0], 1e-12, "macro conv");
}

#[test]
fn macro_create_conv_f32() {
    let b = macro_backend();
    let spec = ConvSpec::new(3, 2, ConvMethod::Direct).expect("valid conv spec");
    let plan = CreatePlan::create_plan::<f32>(&b, &spec).expect("macro conv plan f32");

    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.0_f32, 1.0];
    let mut output = [0.0_f32; 4];
    plan.execute(&a, &b, &mut output).expect("convolve");
    assert_approx(&output, &[1.0, 3.0, 5.0, 3.0], 1e-5, "macro conv f32");
}

#[test]
fn macro_create_dft_f64() {
    let b = macro_backend();
    let dsp = OmniDSP::new(b);
    dft_round_trip(&dsp, 1e-12);
}

#[test]
fn macro_create_dft_f32() {
    let b = macro_backend();
    let dsp = OmniDSP::new(b);
    dft_round_trip::<f32, _>(&dsp, 1e-5);
}

#[test]
fn macro_create_fir_f64() {
    let b = macro_backend();
    let filter =
        FirFilter::new(vec![0.25_f64, 0.5, 0.25], FirMeta::unknown()).expect("valid fir filter");
    let spec = FirSpec::new(filter, FirStrategy::Auto);
    let mut plan = CreateProc::create_proc::<f64>(&b, &spec).expect("macro FIR plan");

    let mut input = vec![0.0; 8];
    input[0] = 1.0;
    let mut output = vec![0.0; 8];
    plan.process(&input, &mut output).expect("FIR process");
    assert_approx(&output[..3], &[0.25, 0.5, 0.25], 1e-12, "macro FIR");
}

#[test]
fn macro_create_fir_f32() {
    let b = macro_backend();
    let filter =
        FirFilter::new(vec![0.25_f64, 0.5, 0.25], FirMeta::unknown()).expect("valid fir filter");
    let spec = FirSpec::new(filter, FirStrategy::Auto);
    let mut plan = CreateProc::create_proc::<f32>(&b, &spec).expect("macro FIR plan f32");

    let mut input = vec![0.0_f32; 8];
    input[0] = 1.0;
    let mut output = vec![0.0_f32; 8];
    plan.process(&input, &mut output).expect("FIR process");
    assert_approx(&output[..3], &[0.25, 0.5, 0.25], 1e-5, "macro FIR f32");
}

#[test]
fn macro_create_iir_f64() {
    let b = macro_backend();
    let spec = IirSpec::new(vec![BiquadSection {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    }])
    .expect("valid iir spec");
    let mut plan = CreateProc::create_proc::<f64>(&b, &spec).expect("macro IIR plan");

    let input: Vec<f64> = (0..8).map(f64::from).collect();
    let mut output = vec![0.0; 8];
    plan.process(&input, &mut output).expect("IIR process");
    assert_approx(&output, &input, 1e-12, "macro IIR passthrough");
}

#[test]
#[allow(
    clippy::cast_precision_loss,
    reason = "small test indices fit in f32 mantissa"
)]
fn macro_create_iir_f32() {
    let b = macro_backend();
    let spec = IirSpec::new(vec![BiquadSection {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    }])
    .expect("valid iir spec");
    let mut plan = CreateProc::create_proc::<f32>(&b, &spec).expect("macro IIR plan f32");

    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut output = vec![0.0_f32; 8];
    plan.process(&input, &mut output).expect("IIR process");
    assert_approx(&output, &input, 1e-5, "macro IIR passthrough f32");
}

#[test]
fn macro_create_resample_f64() {
    let b = macro_backend();
    let filter =
        FirFilter::new(vec![0.5, 1.0, 0.5], FirMeta::unknown()).expect("non-empty prototype");
    let spec = ResampleSpec::new(filter, 2, 1).expect("valid resample spec");
    let mut plan = CreateProc::create_proc::<f64>(&b, &spec).expect("macro resample plan");

    let input: Vec<f64> = (0..16).map(f64::from).collect();
    let mut output = vec![0.0; 64];
    let written = plan.process(&input, &mut output).expect("resample");
    assert_eq!(written, 32, "macro resample should produce 2× samples");
}

#[test]
fn macro_create_cqt_f64() {
    let b = macro_backend();
    let bins = vec![CqtBinSpec {
        frequency: 440.0,
        kernel_len: 5,
    }];
    let spec = CqtSpec::new(44100.0, 8, 2, bins, Window::Hann).expect("CQT spec");
    let plan = CreatePlan::create_plan::<f64>(&b, &spec).expect("macro CQT plan");

    let input = vec![0.0; 8];
    let mut output = vec![Complex::new(0.0, 0.0); 1];
    plan.execute(&input, &mut output).expect("CQT process");
    assert!(
        output[0].re.abs() < 1e-10,
        "macro CQT of zero input should be ~zero",
    );
}

// ═══════════════════════════════════════════════════════════════════════
// impl_generic_backend! with skip
// ═══════════════════════════════════════════════════════════════════════

/// A test backend that skips conv, proving the macro's `skip` list works.
/// The backend can then hand-write `CreatePlan<ConvSpec>` without
/// conflicting with macro-generated impls.
#[derive(Clone)]
struct SkipConvBackend {
    dftc2c: RustDftC2c,
    dftr2c: RustDftR2c,
    dftc2r: RustDftC2r,
}

impl omnidsp_core::traits::vecops::VecOps<f32> for SkipConvBackend {}
impl omnidsp_core::traits::vecops::VecOps<f64> for SkipConvBackend {}

omnidsp_macros::impl_generic_backend! {
    backend: SkipConvBackend,
    dftc2c: RustDftC2c,
    dftr2c: RustDftR2c,
    dftc2r: RustDftC2r,
    skip: [conv],
}

/// The composite plan type the hand-written conv override returns — the same
/// shape the macro would emit (r2c forward + Hermitian-shaped c2r inverse).
/// The `Conv` factory trait was dropped, so the type is spelled out.
type SkipConvPlan<T> = omnidsp_core::modules::conv::OmniConvPlan<
    T,
    <<SkipConvBackend as omnidsp_core::dispatch::RawDft<T>>::R2c
        as omnidsp_core::traits::dft::DftR2c<T>>::Plan,
    omnidsp_core::hermitian::HermitianC2rPlan<
        <<SkipConvBackend as omnidsp_core::dispatch::RawDft<T>>::C2r
            as omnidsp_core::traits::dft::DftC2r<T>>::Plan,
    >,
    SkipConvBackend,
>;

// Hand-written override for ConvSpec — proves no conflict.
impl CreatePlan<ConvSpec> for SkipConvBackend {
    type Plan<T>
        = SkipConvPlan<T>
    where
        Self: omnidsp_core::dispatch::Backend<T>;

    fn create_plan<T>(&self, spec: &ConvSpec) -> omnidsp_core::error::Result<Self::Plan<T>>
    where
        Self: omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + std::ops::AddAssign + std::ops::MulAssign,
    {
        let factory = omnidsp_core::modules::conv::OmniConv::new(
            omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
            omnidsp_core::dispatch::RawDft::<T>::raw_dftc2r(self),
            ::core::clone::Clone::clone(self),
        );
        factory.create_plan::<T>(spec)
    }
}

#[test]
fn skip_conv_backend_hand_written_conv() {
    let b = SkipConvBackend {
        dftc2c: RustDftC2c,
        dftr2c: RustDftR2c,
        dftc2r: RustDftC2r,
    };
    let spec = ConvSpec::new(3, 2, ConvMethod::Direct).expect("valid conv spec");
    let plan = CreatePlan::create_plan::<f64>(&b, &spec).expect("skip-conv hand-written plan");

    let a = [1.0, 2.0, 3.0];
    let bv = [1.0, 1.0];
    let mut output = [0.0; 4];
    plan.execute(&a, &bv, &mut output).expect("convolve");
    assert_approx(&output, &[1.0, 3.0, 5.0, 3.0], 1e-12, "skip-conv override");
}

#[test]
fn skip_conv_backend_other_modules_work() {
    let b = SkipConvBackend {
        dftc2c: RustDftC2c,
        dftr2c: RustDftR2c,
        dftc2r: RustDftC2r,
    };
    let dsp = OmniDSP::new(b);
    dft_round_trip(&dsp, 1e-12);
    fir_impulse(&dsp, 1e-12);
    iir_passthrough(&dsp, 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════
// impl_generic_backend! minimal — only `dftc2c:`, everything else defaulted
// ═══════════════════════════════════════════════════════════════════════

/// A backend declaring **only** `dftc2c:` — `dftr2c`/`dftc2r`/`iir`
/// take their realfft/scalar defaults, and the backend supplies `VecOps`
/// via its own empty impls below.  It carries a
/// single field; the macro materializes the omitted floors inline, so no
/// backing struct field is needed for them.
#[derive(Clone)]
struct MinimalBackend {
    dftc2c: RustDftC2c,
}

impl omnidsp_core::traits::vecops::VecOps<f32> for MinimalBackend {}
impl omnidsp_core::traits::vecops::VecOps<f64> for MinimalBackend {}

omnidsp_macros::impl_generic_backend! {
    backend: MinimalBackend,
    dftc2c: RustDftC2c,
}

#[test]
fn minimal_backend_defaults_materialize() {
    let dsp = OmniDSP::new(MinimalBackend { dftc2c: RustDftC2c });

    // c2c via the bare blanket; the rest compose the defaulted
    // realfft floor, the backend's own VecOps impls, and ScalarIir.
    dft_round_trip(&dsp, 1e-12);
    conv_verify(&dsp, 1e-12);
    fir_impulse(&dsp, 1e-12);
    iir_passthrough(&dsp, 1e-12);
}

#[test]
fn minimal_backend_shaped_direct_r2c_c2r() {
    let dsp = OmniDSP::new(MinimalBackend { dftc2c: RustDftC2c });

    // The `dft_r2c` / `dft_c2r` convenience methods dispatch through the core
    // blanket: shaped plans over the defaulted realfft floor.
    let original = vec![1.0_f64, -2.0, 3.0, 0.5, -1.5, 2.0, 0.0, 4.0];
    let n = original.len();
    let r2c_spec = DftR2cSpec::new(n, DftNorm::Inverse).expect("r2c spec");
    let r2c = dsp.dft_r2c::<f64>(&r2c_spec).expect("r2c plan");
    let c2r_spec = DftC2rSpec::new(n, DftNorm::Inverse).expect("c2r spec");
    let c2r = dsp.dft_c2r::<f64>(&c2r_spec).expect("c2r plan");

    let mut time = original.clone();
    let mut spectrum = vec![Complex::new(0.0, 0.0); n / 2 + 1];
    r2c.execute(&mut time, &mut spectrum).expect("r2c process");
    let mut recovered = vec![0.0_f64; n];
    c2r.execute(&mut spectrum, &mut recovered)
        .expect("c2r process");
    assert_approx(
        &recovered,
        &original,
        1e-12,
        "minimal shaped r2c→c2r round-trip",
    );
}
