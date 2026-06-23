// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Light oneMKL-vs-floor oracle checks for fast localized diagnosis.
//!
//! The `conformance` test is the comprehensive acceptance gate; this file is a quick
//! triage layer: a c2c forward+inverse round-trip, an r2c→c2r round-trip, and
//! `dot` / `cmul` parity with the scalar reference.  All checks link `mkl_rt`,
//! so they only run where MKL is installed.

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
    clippy::similar_names,
    reason = "DFT tests use the systematic c2c / r2c / c2r short names"
)]
#![allow(
    clippy::cast_sign_loss,
    reason = "loop indices 0..N are non-negative, so the index->unsigned/float cast cannot lose a sign"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "test-signal formulas favor readability over fused multiply-add"
)]

use num_complex::Complex;

use omnidsp_core::scalar::ScalarVecOps;
use omnidsp_core::traits::dft::{
    DftC2c, DftC2cPlan, DftC2cSpec, DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan,
    DftR2cSpec,
};
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_core::types::Direction;
use omnidsp_onemkl::{OneMklBackend, OneMklDftC2c, OneMklDftC2r, OneMklDftR2c};

const EPS_F32: f32 = 1e-4;
const EPS_F64: f64 = 1e-11;

/// c2c forward then inverse with `Inverse` normalization recovers the input.
#[test]
fn c2c_round_trip_f64() {
    let n = 64_usize;
    let fwd = DftC2cSpec::new(n, Direction::Forward, DftNorm::Inverse).expect("fwd spec");
    let inv = DftC2cSpec::new(n, Direction::Inverse, DftNorm::Inverse).expect("inv spec");
    let fwd_plan = DftC2c::<f64>::create_plan(&OneMklDftC2c, &fwd).expect("fwd plan");
    let inv_plan = DftC2c::<f64>::create_plan(&OneMklDftC2c, &inv).expect("inv plan");

    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(f64::from(i as u32) * 0.5 - 3.0, f64::from(i as u32) * 0.25))
        .collect();
    let mut freq = vec![Complex::default(); n];
    let mut recovered = vec![Complex::default(); n];

    fwd_plan.execute(&input, &mut freq).expect("fwd execute");
    inv_plan
        .execute(&freq, &mut recovered)
        .expect("inv execute");

    for (i, (got, want)) in recovered.iter().zip(&input).enumerate() {
        assert!(
            (got.re - want.re).abs() < EPS_F64 && (got.im - want.im).abs() < EPS_F64,
            "c2c round-trip mismatch at {i}: got ({}, {}), want ({}, {})",
            got.re,
            got.im,
            want.re,
            want.im,
        );
    }
}

/// r2c then c2r with `Inverse` normalization recovers the real input.
#[test]
fn r2c_c2r_round_trip_f32() {
    let n = 63_usize;
    let bins = n / 2 + 1;
    let r2c_spec = DftR2cSpec::new(n, DftNorm::Inverse).expect("r2c spec");
    let c2r_spec = DftC2rSpec::new(n, DftNorm::Inverse).expect("c2r spec");
    let r2c_plan = DftR2c::<f32>::create_plan(&OneMklDftR2c, &r2c_spec).expect("r2c plan");
    let c2r_plan = DftC2r::<f32>::create_plan(&OneMklDftC2r, &c2r_spec).expect("c2r plan");

    let input: Vec<f32> = (0..n).map(|i| (i as f32).mul_add(0.1, 1.0)).collect();
    let mut scratch = input.clone();
    let mut spectrum = vec![Complex::default(); bins];
    r2c_plan
        .execute(&mut scratch, &mut spectrum)
        .expect("r2c execute");

    let mut recovered = vec![0.0_f32; n];
    c2r_plan
        .execute(&mut spectrum, &mut recovered)
        .expect("c2r execute");

    for (i, (got, want)) in recovered.iter().zip(&input).enumerate() {
        assert!(
            (got - want).abs() < EPS_F32,
            "r2c->c2r round-trip mismatch at {i}: got {got}, want {want}",
        );
    }
}

/// `dot` matches the scalar reference.
#[test]
fn dot_matches_scalar_f64() {
    let a: Vec<f64> = (0..128).map(|i| f64::from(i as u32) * 0.3 - 5.0).collect();
    let b: Vec<f64> = (0..128)
        .map(|i| f64::from(i as u32).mul_add(-0.2, 2.0))
        .collect();

    let mkl = OneMklBackend::new();
    let got = VecOps::<f64>::dot(&mkl, &a, &b).expect("mkl dot");
    let want = VecOps::<f64>::dot(&ScalarVecOps, &a, &b).expect("scalar dot");
    assert!(
        (got - want).abs() < EPS_F64 * want.abs().max(1.0),
        "dot mismatch: got {got}, want {want}",
    );
}

/// `cmul` matches the scalar reference.
#[test]
fn cmul_matches_scalar_f32() {
    let a: Vec<Complex<f32>> = (0..96)
        .map(|i| Complex::new(i as f32 * 0.1, 1.0 - i as f32 * 0.05))
        .collect();
    let b: Vec<Complex<f32>> = (0..96)
        .map(|i| Complex::new(2.0 - i as f32 * 0.03, i as f32 * 0.07))
        .collect();

    let mkl = OneMklBackend::new();
    let mut got = vec![Complex::default(); 96];
    let mut want = vec![Complex::default(); 96];
    VecOps::<f32>::cmul(&mkl, &a, &b, &mut got).expect("mkl cmul");
    VecOps::<f32>::cmul(&ScalarVecOps, &a, &b, &mut want).expect("scalar cmul");

    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g.re - w.re).abs() < EPS_F32 && (g.im - w.im).abs() < EPS_F32,
            "cmul mismatch at {i}: got ({}, {}), want ({}, {})",
            g.re,
            g.im,
            w.re,
            w.im,
        );
    }
}
