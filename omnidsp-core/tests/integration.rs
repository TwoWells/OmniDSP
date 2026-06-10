// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! End-to-end integration tests: design layer → module factory → plan → process.
//!
//! These tests exercise the full pipeline using real primitive implementations
//! (`RustDftC2c` from `omnidsp-rustfft`, `ScalarVecOps` and `ScalarIir` from
//! `omnidsp-core::scalar`), validating the public API surface as an external
//! consumer would.

#![allow(clippy::expect_used, reason = "tests use expect for clarity")]
#![allow(
    clippy::missing_assert_message,
    reason = "assert messages are on non-trivial assertions"
)]

use std::f64::consts::TAU;

use num_complex::Complex;

use omnidsp_core::design::{cqt, fir, iir, resample};
use omnidsp_core::modules::cqt::OmniCqt;
use omnidsp_core::modules::fir::OmniFir;
use omnidsp_core::modules::resample::OmniResample;
use omnidsp_core::modules::xcorr::{CrossCorrSpec, OmniCrossCorr};
use omnidsp_core::scalar::{ScalarIir, ScalarVecOps};
use omnidsp_core::traits::dft::{DftC2c, DftC2cPlan, DftC2cSpec, DftNorm};
use omnidsp_core::traits::fir::{FirPlan, FirStrategy};
use omnidsp_core::traits::iir::{Iir, IirPlan};
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_core::types::{Direction, FilterType, Window};
use omnidsp_rustfft::{RustDftC2c, RustDftC2r, RustDftR2c};

// ─── Test data ─────────────────────────────────────────────────────────

/// Resolve a path relative to `testdata/` in the `omnidsp-core` crate root.
macro_rules! testdata {
    ($file:expr) => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/", $file)
    };
}

// ─── Helpers ───────────────────────────────────────────────────────────

fn assert_approx_eq(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}: mismatch at index {i}: got {a}, expected {e} (diff={})",
            (a - e).abs()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 1. FIR filter: design + apply
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code, reason = "not all scipy reference constants are used")]
mod fir_integration {
    use super::*;

    include!(testdata!("fir_lfilter_scipy.rs"));

    /// End-to-end: design LP FIR → create plan with RustDftC2c+ScalarVecOps → process → compare scipy.
    #[test]
    fn design_and_apply_lowpass() {
        // Design: lowpass, order 30, Hamming, fc=1000 Hz, fs=44100
        // (matches scipy: firwin(31, 1000, window='hamming', pass_zero=True, fs=44100))
        let spec = fir::design(
            FilterType::Lowpass,
            30,
            44100.0_f64,
            1000.0,
            None,
            &Window::Hamming,
        )
        .expect("FIR design");

        // Verify our design matches scipy's taps.
        assert_approx_eq(
            &spec.coefficients,
            LFILTER_LP30_HAMMING_TAPS,
            1e-14,
            "FIR taps vs scipy firwin",
        );

        // Create plan with real primitives and process.
        let factory = OmniFir::new(RustDftR2c, RustDftC2r, ScalarVecOps);

        for &strategy in &[FirStrategy::Direct, FirStrategy::OverlapSave] {
            let plan_spec = spec.clone().with_strategy(strategy);
            let mut plan = factory.create_plan(&plan_spec).expect("FIR plan creation");

            let mut output = vec![0.0; LFILTER_INPUT.len()];
            plan.process(LFILTER_INPUT, &mut output)
                .expect("FIR process");

            assert_approx_eq(
                &output,
                LFILTER_LP30_HAMMING_OUTPUT,
                1e-10,
                &format!("FIR LP30 design+apply ({strategy:?})"),
            );
        }
    }

    /// End-to-end: design HP FIR → create plan → process → verify properties.
    ///
    /// Our highpass gain normalization differs from scipy's `firwin` (same
    /// frequency response, different individual tap signs), so we verify
    /// output properties rather than exact sample comparison.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "signal indices are small enough for f64"
    )]
    fn design_and_apply_highpass() {
        let spec = fir::design(
            FilterType::Highpass,
            30,
            44100.0_f64,
            10000.0,
            None,
            &Window::Hann,
        )
        .expect("FIR design");

        let factory = OmniFir::new(RustDftR2c, RustDftC2r, ScalarVecOps);

        for &strategy in &[FirStrategy::Direct, FirStrategy::OverlapSave] {
            let plan_spec = spec.clone().with_strategy(strategy);
            let mut plan = factory.create_plan(&plan_spec).expect("FIR plan creation");

            // Input: 400 Hz (below cutoff, should be rejected) + 15 kHz (above cutoff, should pass).
            let n = 512;
            let sr = 44100.0;
            let input: Vec<f64> = (0..n)
                .map(|i| {
                    (TAU * 400.0 * i as f64 / sr).sin() + (TAU * 15000.0 * i as f64 / sr).sin()
                })
                .collect();

            let mut output = vec![0.0; n];
            plan.process(&input, &mut output).expect("FIR process");

            // After the transient, the 400 Hz component should be attenuated
            // and the 15 kHz component should pass.
            let skip = n / 2;
            let chunk = &output[skip..];
            let rms = (chunk.iter().map(|x| x * x).sum::<f64>() / chunk.len() as f64).sqrt();

            // 15 kHz sine alone has RMS ~0.707. With some transition band
            // leakage, expect moderate energy.
            assert!(
                rms > 0.3 && rms < 1.5,
                "HP filter should pass high frequencies, got RMS {rms} ({strategy:?})"
            );
        }
    }

    /// End-to-end: design LP FIR → process long signal in chunks → compare scipy.
    ///
    /// Tests design + module + streaming with real primitives against a 1024-sample
    /// scipy lfilter reference.
    #[test]
    fn design_and_stream_long_signal() {
        let spec = fir::design(
            FilterType::Lowpass,
            30,
            44100.0_f64,
            1000.0,
            None,
            &Window::Hamming,
        )
        .expect("FIR design");

        let factory = OmniFir::new(RustDftR2c, RustDftC2r, ScalarVecOps);

        for &strategy in &[FirStrategy::Direct, FirStrategy::OverlapSave] {
            let plan_spec = spec.clone().with_strategy(strategy);
            let mut plan = factory.create_plan(&plan_spec).expect("FIR plan creation");

            let mut output = vec![0.0; LFILTER_LONG_INPUT.len()];

            // Process in varied-size chunks to stress streaming.
            let chunks = [37, 100, 63, 200, 11, 256, 357];
            let mut pos = 0;
            for &chunk in &chunks {
                let end = (pos + chunk).min(output.len());
                if pos >= end {
                    break;
                }
                plan.process(&LFILTER_LONG_INPUT[pos..end], &mut output[pos..end])
                    .expect("chunk process");
                pos = end;
            }

            assert_approx_eq(
                &output,
                LFILTER_LONG_OUTPUT,
                1e-10,
                &format!("FIR LP30 long streaming ({strategy:?})"),
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. IIR filter: design + apply
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code, reason = "not all scipy reference constants are used")]
mod iir_integration {
    use super::*;

    include!(testdata!("iir_sosfilt_scipy.rs"));

    /// End-to-end: design LP Butterworth → `ScalarIir` → process → compare scipy sosfilt.
    ///
    /// Our SOS sections may be ordered differently from scipy's `butter`,
    /// but the cascade transfer function is identical, so the processed
    /// output should match scipy's `sosfilt` within floating-point tolerance.
    #[test]
    fn design_and_apply_lowpass() {
        // Design: Butterworth LP, order 4, fc=1000 Hz, fs=44100
        let spec = iir::design::<f64>(
            iir::FilterFamily::Butterworth,
            FilterType::Lowpass,
            4,
            44100.0,
            1000.0,
            None,
        )
        .expect("IIR design");

        // Section count should match (ceil(order/2) = 2 for LP order 4).
        assert_eq!(
            spec.sections.len(),
            SOSFILT_LP4_SOS.len(),
            "IIR LP4: section count mismatch"
        );

        // Create plan and process — compare output against scipy's sosfilt.
        let iir_factory = ScalarIir::new();
        let mut plan = Iir::<f64>::create_plan(&iir_factory, &spec).expect("IIR plan creation");

        let mut output = vec![0.0; SOSFILT_INPUT.len()];
        plan.process(SOSFILT_INPUT, &mut output)
            .expect("IIR process");

        assert_approx_eq(&output, SOSFILT_LP4_OUTPUT, 1e-8, "IIR LP4 design+apply");
    }

    /// End-to-end: design HP Butterworth → `ScalarIir` → process → compare scipy.
    #[test]
    fn design_and_apply_highpass() {
        let spec = iir::design::<f64>(
            iir::FilterFamily::Butterworth,
            FilterType::Highpass,
            4,
            44100.0,
            5000.0,
            None,
        )
        .expect("IIR design");

        assert_eq!(
            spec.sections.len(),
            SOSFILT_HP4_SOS.len(),
            "IIR HP4: section count mismatch"
        );

        let iir_factory = ScalarIir::new();
        let mut plan = Iir::<f64>::create_plan(&iir_factory, &spec).expect("IIR plan creation");

        let mut output = vec![0.0; SOSFILT_INPUT.len()];
        plan.process(SOSFILT_INPUT, &mut output)
            .expect("IIR process");

        assert_approx_eq(&output, SOSFILT_HP4_OUTPUT, 1e-8, "IIR HP4 design+apply");
    }

    /// End-to-end: design BP Butterworth → `ScalarIir` → process → compare scipy.
    #[test]
    fn design_and_apply_bandpass() {
        let spec = iir::design::<f64>(
            iir::FilterFamily::Butterworth,
            FilterType::Bandpass,
            4,
            44100.0,
            1000.0,
            Some(5000.0),
        )
        .expect("IIR design");

        assert_eq!(
            spec.sections.len(),
            SOSFILT_BP4_SOS.len(),
            "IIR BP4: section count mismatch"
        );

        let iir_factory = ScalarIir::new();
        let mut plan = Iir::<f64>::create_plan(&iir_factory, &spec).expect("IIR plan creation");

        let mut output = vec![0.0; SOSFILT_INPUT.len()];
        plan.process(SOSFILT_INPUT, &mut output)
            .expect("IIR process");

        assert_approx_eq(&output, SOSFILT_BP4_OUTPUT, 1e-8, "IIR BP4 design+apply");
    }

    /// End-to-end: design LP IIR → process long signal in chunks → compare scipy.
    #[test]
    fn design_and_stream_long_signal() {
        let spec = iir::design::<f64>(
            iir::FilterFamily::Butterworth,
            FilterType::Lowpass,
            4,
            44100.0,
            1000.0,
            None,
        )
        .expect("IIR design");

        let iir_factory = ScalarIir::new();
        let mut plan = Iir::<f64>::create_plan(&iir_factory, &spec).expect("IIR plan creation");

        let mut output = vec![0.0; SOSFILT_LONG_INPUT.len()];

        // Process in varied-size chunks.
        let chunks = [37, 100, 63, 200, 11, 256, 357];
        let mut pos = 0;
        for &chunk in &chunks {
            let end = (pos + chunk).min(output.len());
            if pos >= end {
                break;
            }
            plan.process(&SOSFILT_LONG_INPUT[pos..end], &mut output[pos..end])
                .expect("chunk process");
            pos = end;
        }

        assert_approx_eq(&output, SOSFILT_LONG_OUTPUT, 1e-8, "IIR LP4 long streaming");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Resampling: design + apply
// ═══════════════════════════════════════════════════════════════════════

mod resample_integration {
    use super::*;
    use omnidsp_core::design::resample::{DEFAULT_MAX_PHASES, ResampleMode, ResampleQuality};

    /// End-to-end: design 44100→48000 resampler → process → verify properties.
    ///
    /// Exact sample-by-sample comparison against scipy is not possible because
    /// `resample_poly` uses a different prototype filter.  Instead we verify
    /// key properties: correct output length, DC convergence, and energy
    /// preservation for a pure tone.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "output length is small enough for exact f64"
    )]
    fn design_and_apply_44100_to_48000() {
        let spec = resample::design(
            44100.0_f64,
            48000.0,
            ResampleQuality::new(5).expect("valid quality"),
            &Window::Hamming,
            DEFAULT_MAX_PHASES,
            ResampleMode::Streaming,
        )
        .expect("resample design");

        assert_eq!(spec.up_factor(), 160, "L should be 160");
        assert_eq!(spec.down_factor(), 147, "M should be 147");

        let factory = OmniResample::new(ScalarVecOps);
        let mut plan = factory.create_plan(&spec).expect("resample plan");

        // Generate a 400 Hz sine at 44100 Hz.
        let n_in = 1000;
        let sr_in = 44100.0;
        let freq = 400.0;
        let input: Vec<f64> = (0..n_in)
            .map(|i| (TAU * freq * f64::from(i) / sr_in).sin())
            .collect();

        let max_out = plan.max_output_len(input.len());
        let mut output = vec![0.0; max_out];
        let n_out = plan.process(&input, &mut output).expect("resample process");

        // Verify output length: ceil(1000 * 160 / 147) = 1089
        assert_eq!(n_out, 1089, "output length for 44100→48000");

        // Verify energy preservation: skip group delay transient edges.
        let skip = n_out / 4;
        let chunk = &output[skip..n_out - skip];
        let energy: f64 = chunk.iter().map(|x| x * x).sum::<f64>() / chunk.len() as f64;
        assert!(
            (energy - 0.5).abs() < 0.05,
            "sine energy should be ~0.5 (unit amplitude), got {energy}"
        );
    }

    /// End-to-end: DC input converges to unity after group delay.
    #[test]
    fn design_and_apply_dc_convergence() {
        let spec = resample::design(
            44100.0_f64,
            48000.0,
            ResampleQuality::new(5).expect("valid quality"),
            &Window::Hamming,
            DEFAULT_MAX_PHASES,
            ResampleMode::Streaming,
        )
        .expect("resample design");

        let factory = OmniResample::new(ScalarVecOps);
        let mut plan = factory.create_plan(&spec).expect("resample plan");

        let input = vec![1.0_f64; 500];
        let max = plan.max_output_len(input.len());
        let mut output = vec![0.0; max];
        let n = plan.process(&input, &mut output).expect("resample process");

        // After the group delay, output should converge to 1.0.
        let steady_start = n / 2;
        for (i, &y) in output[steady_start..n].iter().enumerate() {
            assert!(
                (y - 1.0).abs() < 0.01,
                "DC output should converge to 1.0, sample {}: {y}",
                steady_start + i
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Streaming equivalence
// ═══════════════════════════════════════════════════════════════════════

mod streaming_equivalence {
    use super::*;
    use omnidsp_core::design::resample::{DEFAULT_MAX_PHASES, ResampleMode, ResampleQuality};

    /// FIR: single-shot vs chunked processing must produce identical output.
    #[test]
    fn fir_streaming_equivalence() {
        let spec = fir::design(
            FilterType::Lowpass,
            64,
            44100.0_f64,
            2000.0,
            None,
            &Window::Hamming,
        )
        .expect("FIR design");

        let factory = OmniFir::new(RustDftR2c, RustDftC2r, ScalarVecOps);
        let input: Vec<f64> = (0..1024)
            .map(|i| (TAU * 400.0 * f64::from(i) / 44100.0).sin())
            .collect();

        for &strategy in &[FirStrategy::Direct, FirStrategy::OverlapSave] {
            // Single-shot.
            let plan_spec = spec.clone().with_strategy(strategy);
            let mut plan_single = factory.create_plan(&plan_spec).expect("single plan");
            let mut single_out = vec![0.0; input.len()];
            plan_single
                .process(&input, &mut single_out)
                .expect("single process");

            // Chunked with varied sizes.
            let mut plan_chunks = factory.create_plan(&plan_spec).expect("chunks plan");
            let mut chunks_out = vec![0.0; input.len()];
            let chunk_sizes = [37, 100, 63, 200, 11, 256, 357];
            let mut pos = 0;
            for &chunk in &chunk_sizes {
                let end = (pos + chunk).min(input.len());
                if pos >= end {
                    break;
                }
                plan_chunks
                    .process(&input[pos..end], &mut chunks_out[pos..end])
                    .expect("chunk process");
                pos = end;
            }

            assert_approx_eq(
                &chunks_out,
                &single_out,
                1e-10,
                &format!("FIR streaming equivalence ({strategy:?})"),
            );
        }
    }

    /// IIR: single-shot vs chunked processing must produce identical output.
    #[test]
    fn iir_streaming_equivalence() {
        let spec = iir::design::<f64>(
            iir::FilterFamily::Butterworth,
            FilterType::Lowpass,
            4,
            44100.0,
            1000.0,
            None,
        )
        .expect("IIR design");

        let iir_factory = ScalarIir::new();
        let input: Vec<f64> = (0..1024)
            .map(|i| (TAU * 400.0 * f64::from(i) / 44100.0).sin())
            .collect();

        // Single-shot.
        let mut plan_single = Iir::<f64>::create_plan(&iir_factory, &spec).expect("single plan");
        let mut single_out = vec![0.0; input.len()];
        plan_single
            .process(&input, &mut single_out)
            .expect("single process");

        // Chunked with varied sizes.
        let mut plan_chunks = Iir::<f64>::create_plan(&iir_factory, &spec).expect("chunks plan");
        let mut chunks_out = vec![0.0; input.len()];
        let chunk_sizes = [37, 100, 63, 200, 11, 256, 357];
        let mut pos = 0;
        for &chunk in &chunk_sizes {
            let end = (pos + chunk).min(input.len());
            if pos >= end {
                break;
            }
            plan_chunks
                .process(&input[pos..end], &mut chunks_out[pos..end])
                .expect("chunk process");
            pos = end;
        }

        assert_approx_eq(&chunks_out, &single_out, 1e-12, "IIR streaming equivalence");
    }

    /// Resample: single-shot vs chunked processing must produce identical output.
    #[test]
    fn resample_streaming_equivalence() {
        let spec = resample::design(
            44100.0_f64,
            48000.0,
            ResampleQuality::new(5).expect("valid quality"),
            &Window::Hamming,
            DEFAULT_MAX_PHASES,
            ResampleMode::Streaming,
        )
        .expect("resample design");

        let factory = OmniResample::new(ScalarVecOps);
        let input: Vec<f64> = (0..512)
            .map(|i| (TAU * 400.0 * f64::from(i) / 44100.0).sin())
            .collect();

        // Single-shot.
        let mut plan_single = factory.create_plan(&spec).expect("single plan");
        let max_single = plan_single.max_output_len(input.len());
        let mut single_out = vec![0.0; max_single];
        let n_single = plan_single
            .process(&input, &mut single_out)
            .expect("single process");

        // Chunked.
        let mut plan_chunks = factory.create_plan(&spec).expect("chunks plan");
        let mut chunks_out = Vec::new();
        let chunk_sizes = [37, 100, 63, 200, 112];
        let mut pos = 0;
        for &chunk in &chunk_sizes {
            let end = (pos + chunk).min(input.len());
            if pos >= end {
                break;
            }
            let max = plan_chunks.max_output_len(end - pos);
            let mut buf = vec![0.0; max];
            let n = plan_chunks
                .process(&input[pos..end], &mut buf)
                .expect("chunk process");
            chunks_out.extend_from_slice(&buf[..n]);
            pos = end;
        }

        assert_eq!(
            chunks_out.len(),
            n_single,
            "chunked should produce same total as single-shot"
        );
        assert_approx_eq(
            &chunks_out,
            &single_out[..n_single],
            1e-10,
            "resample streaming equivalence",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Pipeline composition
// ═══════════════════════════════════════════════════════════════════════

mod pipeline {
    use super::*;
    use omnidsp_core::design::resample::{DEFAULT_MAX_PHASES, ResampleMode, ResampleQuality};

    /// Resample 44100→48000 then lowpass FIR at 8000 Hz.
    ///
    /// Verifies that module outputs are type-compatible and composable.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "output length is small enough for exact f64"
    )]
    fn resample_then_fir_lowpass() {
        // Stage 1: resample 44100 → 48000.
        let resample_spec = resample::design(
            44100.0_f64,
            48000.0,
            ResampleQuality::new(3).expect("valid quality"),
            &Window::Hamming,
            DEFAULT_MAX_PHASES,
            ResampleMode::Streaming,
        )
        .expect("resample design");

        let resample_factory = OmniResample::new(ScalarVecOps);
        let mut resample_plan = resample_factory
            .create_plan(&resample_spec)
            .expect("resample plan");

        let n_in = 512;
        let input: Vec<f64> = (0..n_in)
            .map(|i| (TAU * 1000.0 * f64::from(i) / 44100.0).sin())
            .collect();

        let max_resamp = resample_plan.max_output_len(input.len());
        let mut resampled = vec![0.0; max_resamp];
        let n_resamp = resample_plan
            .process(&input, &mut resampled)
            .expect("resample process");
        let resampled = &resampled[..n_resamp];

        // Stage 2: lowpass FIR at 8000 Hz, 48000 Hz sample rate.
        let fir_spec = fir::design(
            FilterType::Lowpass,
            32,
            48000.0_f64,
            8000.0,
            None,
            &Window::Hamming,
        )
        .expect("FIR design");

        let fir_factory = OmniFir::new(RustDftR2c, RustDftC2r, ScalarVecOps);
        let mut fir_plan = fir_factory.create_plan(&fir_spec).expect("FIR plan");

        let mut filtered = vec![0.0; resampled.len()];
        fir_plan
            .process(resampled, &mut filtered)
            .expect("FIR process");

        // A 1 kHz tone is well below the 8 kHz cutoff, so energy should
        // be preserved (after the transient).
        let skip = filtered.len() / 4;
        let chunk = &filtered[skip..filtered.len() - skip];
        let energy: f64 = chunk.iter().map(|x| x * x).sum::<f64>() / chunk.len() as f64;
        assert!(
            energy > 0.1,
            "1 kHz tone should survive LP 8 kHz filter, got energy {energy}"
        );
    }

    /// IIR highpass at 100 Hz then IIR lowpass at 8000 Hz (bandpass via cascade).
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample indices and lengths are small enough for f64"
    )]
    #[allow(
        clippy::similar_names,
        reason = "hp_plan/lp_plan are distinct filter stages"
    )]
    fn iir_cascade_bandpass() {
        let iir_factory = ScalarIir::new();

        // Stage 1: highpass at 100 Hz.
        let hp_spec = iir::design::<f64>(
            iir::FilterFamily::Butterworth,
            FilterType::Highpass,
            4,
            44100.0,
            100.0,
            None,
        )
        .expect("IIR HP design");

        let mut hp_plan = Iir::<f64>::create_plan(&iir_factory, &hp_spec).expect("IIR HP plan");

        // Stage 2: lowpass at 8000 Hz.
        let lp_spec = iir::design::<f64>(
            iir::FilterFamily::Butterworth,
            FilterType::Lowpass,
            4,
            44100.0,
            8000.0,
            None,
        )
        .expect("IIR LP design");

        let mut lp_plan = Iir::<f64>::create_plan(&iir_factory, &lp_spec).expect("IIR LP plan");

        // Signal: sum of 50 Hz (below HP), 1000 Hz (in passband), 15000 Hz (above LP).
        let n: usize = 4096;
        let sr = 44100.0;
        let input: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (TAU * 50.0 * t).sin() + (TAU * 1000.0 * t).sin() + (TAU * 15000.0 * t).sin()
            })
            .collect();

        let mut after_hp = vec![0.0; n];
        hp_plan.process(&input, &mut after_hp).expect("HP process");

        let mut after_lp = vec![0.0; n];
        lp_plan
            .process(&after_hp, &mut after_lp)
            .expect("LP process");

        // After group delay transient, only the 1 kHz component should remain.
        // Measure RMS of the steady-state region.
        let skip = n / 2;
        let chunk = &after_lp[skip..];
        let rms = (chunk.iter().map(|x| x * x).sum::<f64>() / chunk.len() as f64).sqrt();

        // 1 kHz sine has RMS ~0.707. Some loss from filter transition bands is OK.
        assert!(
            rms > 0.3 && rms < 1.0,
            "bandpass cascade RMS should be moderate (1 kHz passes), got {rms}"
        );
    }

    /// Window → FFT → magnitude spectrum (verify Parseval's theorem).
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "FFT length is small enough for exact f64"
    )]
    fn window_fft_parseval() {
        use omnidsp_core::types::Window;

        let n = 1024;

        // Generate a sine signal.
        let sr = 44100.0;
        let freq = 1000.0;
        let signal: Vec<f64> = (0..n).map(|i| (TAU * freq * i as f64 / sr).sin()).collect();

        // Apply Hann window.
        let coeffs: Vec<f64> = Window::hann(n).expect("window creation");
        let vecops = ScalarVecOps;
        let mut windowed = signal;
        vecops
            .mul_inplace(&mut windowed, &coeffs)
            .expect("window apply");

        // Forward FFT with unitary normalization.
        let dft_spec = DftC2cSpec::new(n, Direction::Forward, DftNorm::Ortho).expect("valid spec");
        let dft = RustDftC2c;
        let plan = DftC2c::<f64>::create_plan(&dft, &dft_spec).expect("DFT plan");

        let input_c: Vec<Complex<f64>> = windowed.iter().map(|&r| Complex::new(r, 0.0)).collect();
        let mut spectrum = vec![Complex::new(0.0, 0.0); n];
        plan.process(&input_c, &mut spectrum).expect("FFT process");

        // Parseval's theorem with ortho normalization:
        // sum(|x[n]|²) == sum(|X[k]|²)
        let time_energy: f64 = windowed.iter().map(|x| x * x).sum();
        let freq_energy: f64 = spectrum.iter().map(Complex::norm_sqr).sum();

        let rel_error = (time_energy - freq_energy).abs() / time_energy;
        assert!(
            rel_error < 1e-10,
            "Parseval's theorem: time_energy={time_energy}, freq_energy={freq_energy}, \
             relative error={rel_error}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. CQT: design + compute
// ═══════════════════════════════════════════════════════════════════════

mod cqt_integration {
    use super::*;

    include!(testdata!("cqt_librosa.rs"));

    /// Helper: create a CQT plan with real primitives.
    fn make_plan(
        spec: &omnidsp_core::design::cqt::CqtSpec<f64>,
    ) -> omnidsp_core::modules::cqt::OmniCqtPlan<f64, <RustDftC2c as DftC2c<f64>>::Plan, ScalarVecOps>
    {
        let factory = OmniCqt::new(RustDftC2c, ScalarVecOps);
        factory.create_plan(spec).expect("CQT plan creation")
    }

    fn librosa_spec() -> omnidsp_core::design::cqt::CqtSpec<f64> {
        cqt::design(
            CQT_LIB_SAMPLE_RATE,
            CQT_LIB_MIN_FREQ,
            CQT_LIB_MAX_FREQ,
            CQT_LIB_BINS_PER_OCTAVE,
            &Window::<f64>::Hann,
        )
        .expect("CQT design")
    }

    /// End-to-end: design CQT → create plan with RustDftC2c+ScalarVecOps → process.
    ///
    /// Verifies that a pure tone at A440 produces a peak at the correct bin.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "FFT length is small enough for f64"
    )]
    fn pure_tone_peaks_at_correct_bin() {
        let spec = librosa_spec();
        let plan = make_plan(&spec);

        let n = plan.fft_length();
        assert_eq!(n, CQT_LIB_FFT_LENGTH, "FFT length should match reference");
        assert_eq!(
            plan.num_bins(),
            CQT_LIB_NUM_BINS,
            "bin count should match reference"
        );

        // Generate a pure sine at A440.
        let sr = 44100.0;
        let freq = 440.0;
        let input: Vec<f64> = (0..n).map(|i| (TAU * freq * i as f64 / sr).sin()).collect();

        let mut magnitudes = vec![0.0; plan.num_bins()];
        plan.process_magnitude(&input, &mut magnitudes)
            .expect("CQT process");

        // Bin 0 should be A440 — it should have the largest magnitude.
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("no NaN"))
            .expect("has bins")
            .0;

        assert_eq!(
            peak_bin, 0,
            "A440 should peak at bin 0, got bin {peak_bin} (magnitudes: {magnitudes:?})"
        );
    }

    /// Compare our CQT magnitudes against librosa-validated reference for a single tone.
    ///
    /// Reference data uses librosa-validated design parameters (frequencies,
    /// kernel lengths, Q factor) with our kernel construction algorithm.
    /// See `gen_cqt_librosa_reference.py` for root cause of the previous
    /// ~2e-4 offset (librosa's centered time axis).
    #[test]
    fn librosa_single_tone() {
        let spec = librosa_spec();
        let plan = make_plan(&spec);

        let mut magnitudes = vec![0.0; plan.num_bins()];
        plan.process_magnitude(CQT_LIB_TONE_INPUT, &mut magnitudes)
            .expect("CQT process");

        // Verify peak bin matches reference.
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("no NaN"))
            .expect("has bins")
            .0;
        assert_eq!(
            peak_bin, CQT_LIB_TONE_BIN,
            "single tone should peak at expected bin"
        );

        assert_approx_eq(
            &magnitudes,
            CQT_LIB_TONE_MAG,
            1e-10,
            "CQT single-tone magnitudes vs librosa-validated reference",
        );
    }

    /// Compare our CQT magnitudes against librosa-validated reference for all tones.
    #[test]
    fn librosa_all_tones() {
        let spec = librosa_spec();
        let plan = make_plan(&spec);

        let mut magnitudes = vec![0.0; plan.num_bins()];
        plan.process_magnitude(CQT_LIB_ALL_TONES_INPUT, &mut magnitudes)
            .expect("CQT process");

        assert_approx_eq(
            &magnitudes,
            CQT_LIB_ALL_TONES_MAG,
            1e-10,
            "CQT all-tones magnitudes vs librosa-validated reference",
        );
    }

    /// Compare our CQT magnitudes against librosa-validated reference for two tones.
    #[test]
    fn librosa_two_tones() {
        let spec = librosa_spec();
        let plan = make_plan(&spec);

        let mut magnitudes = vec![0.0; plan.num_bins()];
        plan.process_magnitude(CQT_LIB_TWO_TONE_INPUT, &mut magnitudes)
            .expect("CQT process");

        assert_approx_eq(
            &magnitudes,
            CQT_LIB_TWO_TONE_MAG,
            1e-10,
            "CQT two-tone magnitudes vs librosa-validated reference",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Cross-correlation: scipy.signal.correlate reference
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code, reason = "not all scipy reference constants are used")]
mod xcorr_integration {
    use super::*;

    include!(testdata!("xcorr_scipy.rs"));

    const fn make_factory() -> OmniCrossCorr<RustDftR2c, RustDftC2r, ScalarVecOps> {
        OmniCrossCorr::new(RustDftR2c, RustDftC2r, ScalarVecOps)
    }

    /// Short asymmetric signals (len=16 × len=8) vs scipy.signal.correlate.
    #[test]
    fn scipy_short_asymmetric() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(XCORR_SHORT_A.len(), XCORR_SHORT_B.len());
        let plan = factory.create_plan(&spec).expect("plan creation");

        let mut output = vec![0.0; spec.output_len()];
        plan.process(XCORR_SHORT_A, XCORR_SHORT_B, &mut output)
            .expect("xcorr process");

        assert_approx_eq(
            &output,
            XCORR_SHORT_RESULT,
            1e-10,
            "short xcorr vs scipy.signal.correlate",
        );
    }

    /// Equal-length signals (len=32 × len=32) vs scipy.signal.correlate.
    #[test]
    fn scipy_equal_length() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(XCORR_EQUAL_A.len(), XCORR_EQUAL_B.len());
        let plan = factory.create_plan(&spec).expect("plan creation");

        let mut output = vec![0.0; spec.output_len()];
        plan.process(XCORR_EQUAL_A, XCORR_EQUAL_B, &mut output)
            .expect("xcorr process");

        assert_approx_eq(
            &output,
            XCORR_EQUAL_RESULT,
            1e-10,
            "equal-length xcorr vs scipy.signal.correlate",
        );
    }

    /// Sinusoidal delay signals (len=128 × len=128) vs scipy.signal.correlate.
    #[test]
    fn scipy_delay_sinusoids() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(XCORR_DELAY_A.len(), XCORR_DELAY_B.len());
        let plan = factory.create_plan(&spec).expect("plan creation");

        let mut output = vec![0.0; spec.output_len()];
        plan.process(XCORR_DELAY_A, XCORR_DELAY_B, &mut output)
            .expect("xcorr process");

        assert_approx_eq(
            &output,
            XCORR_DELAY_RESULT,
            1e-8,
            "delay xcorr vs scipy.signal.correlate",
        );
    }
}
