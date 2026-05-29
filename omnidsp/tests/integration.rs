// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Integration tests for the `omnidsp` dispatch layer.
//!
//! These tests exercise the full user-facing workflow:
//! design → spec → plan → process, importing `omnidsp` as an external
//! crate exactly like downstream code would.

#![allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
#![allow(
    clippy::cast_precision_loss,
    reason = "test indices and sample counts are small enough for f32/f64"
)]
#![allow(
    clippy::cast_lossless,
    reason = "explicit From calls add noise in test signal generation"
)]

use num_complex::Complex;
use omnidsp::OmniDSP;
use omnidsp::design::cqt;
use omnidsp::design::iir::{self, FilterFamily};
use omnidsp::design::resample::{
    self as resample, DEFAULT_MAX_PHASES, ResampleMode, ResampleQuality,
};
use omnidsp::error::Error;
use omnidsp::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
use omnidsp::traits::dft::{DftNorm, DftPlan, DftSpec};
use omnidsp::traits::fir::{FirPlan, FirSpec};
use omnidsp::traits::iir::{IirPlan, IirSpec};
use omnidsp::types::{Direction, FilterType, Window};

const TOLERANCE_F64: f64 = 1e-10;
const TOLERANCE_F32: f32 = 1e-4;

// ─── DFT ────────────────────────────────────────────────────────────

#[test]
fn dft_round_trip_f64() {
    let dsp = OmniDSP::new();
    let fwd_spec = DftSpec::<f64>::new(128, Direction::Forward, DftNorm::Inverse);
    let inv_spec = DftSpec::<f64>::new(128, Direction::Inverse, DftNorm::Inverse);
    let fwd = dsp.create_plan(&fwd_spec).expect("forward plan");
    let inv = dsp.create_plan(&inv_spec).expect("inverse plan");

    let mut input = vec![Complex::new(0.0, 0.0); 128];
    input[0] = Complex::new(1.0, 0.0);
    let mut spectrum = vec![Complex::default(); 128];
    let mut recovered = vec![Complex::default(); 128];

    fwd.process(&input, &mut spectrum).expect("forward");
    inv.process(&spectrum, &mut recovered).expect("inverse");

    for (i, (a, b)) in input.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (a.re - b.re).abs() < TOLERANCE_F64 && (a.im - b.im).abs() < TOLERANCE_F64,
            "round-trip mismatch at {i}: got ({}, {}), want ({}, {})",
            b.re,
            b.im,
            a.re,
            a.im,
        );
    }
}

#[test]
fn dft_round_trip_f32() {
    let dsp = OmniDSP::new();
    let fwd_spec = DftSpec::<f32>::new(64, Direction::Forward, DftNorm::Inverse);
    let inv_spec = DftSpec::<f32>::new(64, Direction::Inverse, DftNorm::Inverse);
    let fwd = dsp.create_plan(&fwd_spec).expect("forward plan");
    let inv = dsp.create_plan(&inv_spec).expect("inverse plan");

    let input: Vec<Complex<f32>> = (0..64)
        .map(|i| Complex::new(i as f32, -(i as f32)))
        .collect();
    let mut spectrum = vec![Complex::default(); 64];
    let mut recovered = vec![Complex::default(); 64];

    fwd.process(&input, &mut spectrum).expect("forward");
    inv.process(&spectrum, &mut recovered).expect("inverse");

    for (i, (a, b)) in input.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (a.re - b.re).abs() < TOLERANCE_F32 && (a.im - b.im).abs() < TOLERANCE_F32,
            "f32 round-trip mismatch at {i}: got ({}, {}), want ({}, {})",
            b.re,
            b.im,
            a.re,
            a.im,
        );
    }
}

// ─── Convolution ────────────────────────────────────────────────────

#[test]
fn conv_direct_f64() {
    let dsp = OmniDSP::new();
    let spec = ConvSpec::<f64>::new(3, 2, ConvMethod::Direct);
    let plan = dsp.create_plan(&spec).expect("conv plan");

    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 1.0];
    let mut output = vec![0.0; 4];

    plan.process(&a, &b, &mut output).expect("convolve");

    let expected = [1.0, 3.0, 5.0, 3.0];
    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < TOLERANCE_F64,
            "conv direct output[{i}]: got {got}, want {want}",
        );
    }
}

#[test]
fn conv_fft_f64() {
    let dsp = OmniDSP::new();
    let spec = ConvSpec::<f64>::new(3, 2, ConvMethod::Fft);
    let plan = dsp.create_plan(&spec).expect("conv plan fft");

    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 1.0];
    let mut output = vec![0.0; 4];

    plan.process(&a, &b, &mut output).expect("convolve fft");

    let expected = [1.0, 3.0, 5.0, 3.0];
    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < TOLERANCE_F64,
            "conv fft output[{i}]: got {got}, want {want}",
        );
    }
}

#[test]
fn conv_auto_f64() {
    let dsp = OmniDSP::new();
    let spec = ConvSpec::<f64>::new(3, 2, ConvMethod::Auto);
    let plan = dsp.create_plan(&spec).expect("conv plan auto");

    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 1.0];
    let mut output = vec![0.0; 4];

    plan.process(&a, &b, &mut output).expect("convolve auto");

    let expected = [1.0, 3.0, 5.0, 3.0];
    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < TOLERANCE_F64,
            "conv auto output[{i}]: got {got}, want {want}",
        );
    }
}

#[test]
fn conv_basic_f32() {
    let dsp = OmniDSP::new();
    let spec = ConvSpec::<f32>::new(3, 2, ConvMethod::Auto);
    let plan = dsp.create_plan(&spec).expect("conv plan f32");

    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.0_f32, 1.0];
    let mut output = vec![0.0_f32; 4];

    plan.process(&a, &b, &mut output).expect("convolve f32");

    let expected = [1.0_f32, 3.0, 5.0, 3.0];
    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < TOLERANCE_F32,
            "conv f32 output[{i}]: got {got}, want {want}",
        );
    }
}

// ─── FIR ────────────────────────────────────────────────────────────

#[test]
fn fir_impulse_response_f64() {
    let dsp = OmniDSP::new();
    let coeffs: Vec<f64> = vec![0.25, 0.5, 0.25];
    let spec = FirSpec::new(coeffs.clone());
    let mut plan = dsp.create_plan(&spec).expect("FIR plan");

    let input = vec![1.0_f64, 0.0, 0.0, 0.0, 0.0];
    let mut output = vec![0.0; 5];
    plan.process(&input, &mut output).expect("FIR process");

    for (i, &c) in coeffs.iter().enumerate() {
        assert!(
            (output[i] - c).abs() < TOLERANCE_F64,
            "impulse response[{i}]: got {}, want {c}",
            output[i],
        );
    }
}

#[test]
fn fir_impulse_f32() {
    let dsp = OmniDSP::new();
    let coeffs = vec![0.5_f32, 0.3, 0.2];
    let spec = FirSpec::new(coeffs.clone());
    let mut plan = dsp.create_plan(&spec).expect("FIR plan f32");

    let input = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0];
    let mut output = vec![0.0_f32; 5];
    plan.process(&input, &mut output).expect("FIR process f32");

    for (i, &c) in coeffs.iter().enumerate() {
        assert!(
            (output[i] - c).abs() < TOLERANCE_F32,
            "f32 impulse response[{i}]: got {}, want {c}",
            output[i],
        );
    }
}

#[test]
fn fir_streaming_continuity_f64() {
    let dsp = OmniDSP::new();
    let coeffs = vec![0.25_f64, 0.5, 0.25];
    let spec = FirSpec::new(coeffs);

    // Single-pass reference
    let mut plan_ref = dsp.create_plan(&spec).expect("FIR plan ref");
    let full_input: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
    let mut full_output = vec![0.0; 64];
    plan_ref
        .process(&full_input, &mut full_output)
        .expect("FIR ref process");

    // Two-chunk streaming
    let mut plan_stream = dsp.create_plan(&spec).expect("FIR plan stream");
    let (chunk1, chunk2) = full_input.split_at(32);
    let mut out1 = vec![0.0; 32];
    let mut out2 = vec![0.0; 32];
    plan_stream.process(chunk1, &mut out1).expect("chunk 1");
    plan_stream.process(chunk2, &mut out2).expect("chunk 2");

    let streamed: Vec<f64> = out1.into_iter().chain(out2).collect();
    for (i, (a, b)) in full_output.iter().zip(streamed.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOLERANCE_F64,
            "streaming mismatch at sample {i}: single-pass {a}, streamed {b}",
        );
    }
}

// ─── IIR ────────────────────────────────────────────────────────────

#[test]
fn iir_dc_lowpass_f64() {
    let dsp = OmniDSP::new();
    let spec = iir::design::<f64>(
        FilterFamily::Butterworth,
        FilterType::Lowpass,
        2,
        8000.0,
        1000.0,
        None,
    )
    .expect("IIR design");
    let mut plan = dsp.create_plan(&spec).expect("IIR plan");

    let dc = vec![1.0_f64; 256];
    let mut output = vec![0.0; 256];
    plan.process(&dc, &mut output).expect("IIR process");

    assert!(
        (output[255] - 1.0).abs() < 0.01,
        "DC should pass through lowpass, got {}",
        output[255],
    );
}

#[test]
fn iir_highpass_rejects_dc_f64() {
    let dsp = OmniDSP::new();
    let spec = iir::design::<f64>(
        FilterFamily::Butterworth,
        FilterType::Highpass,
        2,
        8000.0,
        1000.0,
        None,
    )
    .expect("IIR highpass design");
    let mut plan = dsp.create_plan(&spec).expect("IIR highpass plan");

    let dc = vec![1.0_f64; 256];
    let mut output = vec![0.0; 256];
    plan.process(&dc, &mut output)
        .expect("IIR highpass process");

    assert!(
        output[255].abs() < 0.01,
        "DC should be rejected by highpass, got {}",
        output[255],
    );
}

#[test]
fn iir_dc_f32() {
    let dsp = OmniDSP::new();
    let spec = iir::design::<f32>(
        FilterFamily::Butterworth,
        FilterType::Lowpass,
        2,
        8000.0,
        1000.0,
        None,
    )
    .expect("IIR f32 design");
    let mut plan = dsp.create_plan(&spec).expect("IIR f32 plan");

    let dc = vec![1.0_f32; 256];
    let mut output = vec![0.0_f32; 256];
    plan.process(&dc, &mut output).expect("IIR f32 process");

    assert!(
        (output[255] - 1.0).abs() < 0.05,
        "f32 DC should pass through lowpass, got {}",
        output[255],
    );
}

// ─── Resampling ─────────────────────────────────────────────────────

#[test]
fn resample_dc_preservation_f64() {
    let dsp = OmniDSP::new();
    let spec = resample::design(
        44100.0_f64,
        48000.0,
        ResampleQuality::new(5).expect("quality"),
        &Window::Hamming,
        DEFAULT_MAX_PHASES,
        ResampleMode::Streaming,
    )
    .expect("resample spec");
    let mut plan = dsp.create_plan(&spec).expect("resample plan");

    let input = vec![1.0_f64; 1024];
    let max_out = plan.max_output_len(input.len());
    let mut output = vec![0.0; max_out];
    let written = plan.process(&input, &mut output).expect("resample");

    // After the filter transient, output should be ~1.0
    let steady_start = 100.min(written);
    for (i, &v) in output[steady_start..written].iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 0.05,
            "resampled DC sample {}: got {v}, want ~1.0",
            steady_start + i,
        );
    }
}

// ─── CQT ────────────────────────────────────────────────────────────

#[test]
fn cqt_bin_detection_f64() {
    let dsp = OmniDSP::new();
    let cqt_spec = cqt::design(44100.0, 55.0, 880.0, 24, &Window::<f64>::Hann).expect("CQT design");
    let plan = dsp.create_plan(&cqt_spec).expect("CQT plan");

    let fft_len = cqt_spec.fft_length();
    let signal: Vec<f64> = (0..fft_len)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
        .collect();

    let num_bins = cqt_spec.num_bins();
    let mut output = vec![Complex::new(0.0, 0.0); num_bins];
    plan.process(&signal, &mut output).expect("CQT process");

    let magnitudes: Vec<f64> = output.iter().map(|c| c.norm()).collect();
    let peak_bin = magnitudes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("no NaN"))
        .map(|(i, _)| i)
        .expect("at least one bin");

    // 440 Hz with fmin=55, 24 bins/octave:
    // octaves above fmin = log2(440/55) = 3
    // expected bin ≈ 3 * 24 = 72
    let expected_bin = 72;
    assert!(
        peak_bin.abs_diff(expected_bin) <= 1,
        "peak at bin {peak_bin}, expected ~{expected_bin}",
    );
}

// ─── Window convenience ─────────────────────────────────────────────

#[test]
fn window_convenience_reexport() {
    let w: Vec<f64> = Window::hann(1024).expect("hann");
    assert_eq!(w.len(), 1024, "window length should match");

    assert!(
        w[0].abs() < TOLERANCE_F64,
        "first coeff should be ~0, got {}",
        w[0],
    );
    assert!(
        w[1023].abs() < TOLERANCE_F64,
        "last coeff should be ~0, got {}",
        w[1023],
    );
    // Even-length Hann window peaks between samples, so the
    // nearest sample is close to but not exactly 1.0.
    assert!(
        (w[512] - 1.0).abs() < 1e-5,
        "midpoint should be ~1, got {}",
        w[512],
    );

    let k: Vec<f64> = Window::kaiser(5.0, 256).expect("kaiser");
    assert_eq!(k.len(), 256, "kaiser length should match");
}

// ─── Error propagation ──────────────────────────────────────────────

#[test]
fn dft_zero_length_fails() {
    let dsp = OmniDSP::new();
    let spec = DftSpec::<f64>::new(0, Direction::Forward, DftNorm::Inverse);
    let result = dsp.create_plan(&spec);
    assert!(
        matches!(&result, Err(Error::InvalidSpec(_))),
        "zero-length DFT should fail with InvalidSpec, got {result:?}",
    );
}

#[test]
fn conv_zero_length_fails() {
    let dsp = OmniDSP::new();
    let spec = ConvSpec::<f64>::new(0, 5, ConvMethod::Auto);
    let result = dsp.create_plan(&spec);
    assert!(
        matches!(&result, Err(Error::InvalidSpec(_))),
        "zero-length conv should fail with InvalidSpec, got {result:?}",
    );
}

#[test]
fn fir_empty_coefficients_fails() {
    let dsp = OmniDSP::new();
    let spec = FirSpec::<f64>::new(vec![]);
    let result = dsp.create_plan(&spec);
    assert!(
        matches!(&result, Err(Error::InvalidSpec(_))),
        "empty FIR coefficients should fail with InvalidSpec, got {result:?}",
    );
}

#[test]
fn iir_empty_sections_fails() {
    let dsp = OmniDSP::new();
    let spec = IirSpec::<f64>::new(vec![]);
    let result = dsp.create_plan(&spec);
    assert!(
        matches!(&result, Err(Error::InvalidSpec(_))),
        "empty IIR sections should fail with InvalidSpec, got {result:?}",
    );
}
