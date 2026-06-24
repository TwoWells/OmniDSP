// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Oracle checks for the native oneMKL VS convolution / cross-correlation
//! overrides.
//!
//! Each native VS plan ([`OneMklConvPlan`] / [`OneMklCrossCorrPlan`]) is compared
//! against the pure-Rust floor — the generic `OmniConv` / `OmniCrossCorr`
//! modules built over the `rustfft`/`realfft` real-DFT factories
//! ([`RustDftR2c`] / [`RustDftC2r`]) and [`ScalarVecOps`] — for **every**
//! [`ConvMethod`] / [`CorrMethod`] (`Auto`, `Fft`, `Direct`) over small, medium,
//! and large sizes, in both `f32` and `f64`.  The floor is the reference the
//! shared conformance suite pins to scipy, so floor agreement transitively
//! pins the VS overrides to the same convention.
//!
//! All checks link `mkl_rt`, so they only run where MKL is installed.

#![allow(
    clippy::expect_used,
    reason = "test harness: expect makes failures legible"
)]
#![allow(
    clippy::cast_precision_loss,
    reason = "index->float test-signal synthesis is exact at the small lengths used"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "small loop indices are well within the target float range"
)]
#![allow(
    clippy::cast_sign_loss,
    reason = "loop indices 0..N are non-negative, so the index->unsigned/float cast cannot lose a sign"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "test-signal formulas favor readability over fused multiply-add"
)]

use omnidsp_core::create::CreatePlan;
use omnidsp_core::modules::conv::OmniConv;
use omnidsp_core::modules::xcorr::{CorrMethod, CrossCorrPlan, CrossCorrSpec, OmniCrossCorr};
use omnidsp_core::scalar::ScalarVecOps;
use omnidsp_core::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
use omnidsp_core::types::DspFloat;
use omnidsp_onemkl::OneMklBackend;
use omnidsp_rustfft::{RustDftC2r, RustDftR2c};

/// Per-precision agreement tolerance against the floor reference.
trait OracleFloat: DspFloat {
    /// Maximum absolute deviation tolerated against the floor result.
    const TOL: f64;
    /// Width label for assertion messages.
    const WIDTH: &'static str;
}

impl OracleFloat for f32 {
    const TOL: f64 = 1e-3;
    const WIDTH: &'static str = "f32";
}

impl OracleFloat for f64 {
    const TOL: f64 = 1e-9;
    const WIDTH: &'static str = "f64";
}

/// Small/medium/large `(a_len, b_len)` size pairs exercised for every method.
const SIZES: &[(usize, usize)] = &[(3, 2), (17, 5), (64, 33), (300, 200), (1024, 257)];

/// Deterministic, well-conditioned test signal of length `n`.
fn signal<T: DspFloat>(n: usize, phase: f64) -> Vec<T> {
    (0..n)
        .map(|i| {
            let x = (i as f64).mul_add(0.17, phase).sin() + 0.3 * (i as f64 * 0.05 - 1.0);
            T::from_f64(x).expect("signal sample fits the float type")
        })
        .collect()
}

/// Assert two real buffers agree elementwise within `tol`.
fn assert_close<T: DspFloat>(got: &[T], want: &[T], tol: f64, context: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{context}: length mismatch ({} vs {})",
        got.len(),
        want.len(),
    );
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let gf = g.to_f64().expect("got sample to f64");
        let wf = w.to_f64().expect("want sample to f64");
        let scale = wf.abs().max(1.0);
        assert!(
            (gf - wf).abs() <= tol * scale,
            "{context}: mismatch at {i}: got {gf}, want {wf} (tol {tol})",
        );
    }
}

/// Compute the floor convolution reference for `a` ∗ `b` with `method`.
fn floor_conv<T: DspFloat>(method: ConvMethod, a: &[T], b: &[T]) -> Vec<T> {
    let spec = ConvSpec::new(a.len(), b.len(), method).expect("valid floor conv spec");
    let plan = OmniConv::new(RustDftR2c, RustDftC2r, ScalarVecOps)
        .create_plan::<T>(&spec)
        .expect("floor conv plan");
    let mut out = vec![T::zero(); a.len() + b.len() - 1];
    plan.execute(a, b, &mut out).expect("floor conv execute");
    out
}

/// Compute the floor cross-correlation reference for `a`, `b` with `method`.
fn floor_xcorr<T: DspFloat>(method: CorrMethod, a: &[T], b: &[T]) -> Vec<T> {
    let spec = CrossCorrSpec::new(a.len(), b.len(), method).expect("valid floor xcorr spec");
    let plan = OmniCrossCorr::new(RustDftR2c, RustDftC2r, ScalarVecOps)
        .create_plan::<T>(&spec)
        .expect("floor xcorr plan");
    let mut out = vec![T::zero(); spec.output_len()];
    plan.execute(a, b, &mut out).expect("floor xcorr execute");
    out
}

/// Native VS convolution matches the floor for every method and size.
fn check_conv<T: OracleFloat>() {
    let mkl = OneMklBackend::new();
    for &method in &[ConvMethod::Auto, ConvMethod::Fft, ConvMethod::Direct] {
        for &(a_len, b_len) in SIZES {
            let a = signal::<T>(a_len, 0.0);
            let b = signal::<T>(b_len, 1.3);

            let spec = ConvSpec::new(a_len, b_len, method).expect("valid mkl conv spec");
            let plan = CreatePlan::<ConvSpec>::create_plan::<T>(&mkl, &spec)
                .expect("mkl conv plan");
            let mut got = vec![T::zero(); a_len + b_len - 1];
            ConvPlan::execute(&plan, &a, &b, &mut got).expect("mkl conv execute");

            let want = floor_conv(method, &a, &b);
            assert_close(
                &got,
                &want,
                T::TOL,
                &format!("conv {} {method:?} ({a_len}x{b_len})", T::WIDTH),
            );
        }
    }
}

/// Native VS cross-correlation matches the floor for every method and size.
fn check_xcorr<T: OracleFloat>() {
    let mkl = OneMklBackend::new();
    for &method in &[CorrMethod::Auto, CorrMethod::Fft, CorrMethod::Direct] {
        for &(a_len, b_len) in SIZES {
            let a = signal::<T>(a_len, 0.5);
            let b = signal::<T>(b_len, 2.1);

            let spec =
                CrossCorrSpec::new(a_len, b_len, method).expect("valid mkl xcorr spec");
            let plan = CreatePlan::<CrossCorrSpec>::create_plan::<T>(&mkl, &spec)
                .expect("mkl xcorr plan");
            let mut got = vec![T::zero(); spec.output_len()];
            CrossCorrPlan::execute(&plan, &a, &b, &mut got).expect("mkl xcorr execute");

            let want = floor_xcorr(method, &a, &b);
            assert_close(
                &got,
                &want,
                T::TOL,
                &format!("xcorr {} {method:?} ({a_len}x{b_len})", T::WIDTH),
            );
        }
    }
}

#[test]
fn vs_conv_matches_floor_f32() {
    check_conv::<f32>();
}

#[test]
fn vs_conv_matches_floor_f64() {
    check_conv::<f64>();
}

#[test]
fn vs_xcorr_matches_floor_f32() {
    check_xcorr::<f32>();
}

#[test]
fn vs_xcorr_matches_floor_f64() {
    check_xcorr::<f64>();
}
