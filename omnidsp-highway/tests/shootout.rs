// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Quick performance comparison: Highway vs Rust fallback.
//!
//! Run with: `make test T=shootout`
//! All tests are `#[ignore = "benchmark — run manually with make test T=shootout"]` so they don't run in `make check`.

#![allow(clippy::expect_used, reason = "benchmark harness")]
#![allow(clippy::print_stderr, reason = "benchmark output goes to stderr")]

use std::hint::black_box;
use std::time::Instant;

use num_complex::Complex;

use omnidsp_core::traits::dft::{Dft, DftNorm, DftPlan, DftSpec};
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_core::types::Direction;

use omnidsp_highway::{HwyDft, HwyVecOps};
use omnidsp_rust::{RustDft, RustVecOps};

const WARMUP: usize = 10;
const ITERS: usize = 1000;

fn bench<F: FnMut()>(label: &str, mut f: F) -> std::time::Duration {
    // Warmup.
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let elapsed = start.elapsed();
    #[allow(clippy::cast_possible_truncation, reason = "ITERS is a small constant")]
    let per_iter = elapsed / ITERS as u32;
    eprintln!("  {label:>20}: {per_iter:>10?}  ({ITERS} iters)");
    elapsed
}

// ─── DFT shootout ───────────────────────────────────────────────────

fn dft_shootout(n: usize) {
    eprintln!("\n=== DFT forward f64, N={n} ===");

    let spec = DftSpec::new(n, Direction::Forward, DftNorm::Inverse);
    let input: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss, reason = "bench value")]
            let v = (i as f64) * 0.01;
            Complex::new(v.sin(), v.cos())
        })
        .collect();
    let mut output = vec![Complex::default(); n];

    let hwy_plan = Dft::<f64>::create_plan(&HwyDft, &spec).expect("HwyDft plan should succeed");
    let rust_plan = Dft::<f64>::create_plan(&RustDft, &spec).expect("RustDft plan should succeed");

    let hwy_time = bench("HwyDft", || {
        hwy_plan
            .process(&input, &mut output)
            .expect("process should succeed");
        black_box(&output);
    });
    let rust_time = bench("RustDft (RustFFT)", || {
        rust_plan
            .process(&input, &mut output)
            .expect("process should succeed");
        black_box(&output);
    });

    let ratio = hwy_time.as_secs_f64() / rust_time.as_secs_f64();
    eprintln!("  ratio (Hwy/Rust): {ratio:.2}x  (< 1.0 = Highway wins)");
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn dft_shootout_64() {
    dft_shootout(64);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn dft_shootout_256() {
    dft_shootout(256);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn dft_shootout_1024() {
    dft_shootout(1024);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn dft_shootout_4096() {
    dft_shootout(4096);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn dft_shootout_8192() {
    dft_shootout(8192);
}

// ─── VecOps shootout ────────────────────────────────────────────────

fn vecops_dot_shootout(n: usize) {
    eprintln!("\n=== VecOps dot f64, N={n} ===");

    let a: Vec<f64> = (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss, reason = "bench value")]
            let v = (i as f64) * 0.001;
            v
        })
        .collect();
    let b: Vec<f64> = a.iter().map(|x| x + 1.0).collect();

    bench("HwyVecOps", || {
        black_box(VecOps::<f64>::dot(&HwyVecOps, &a, &b).expect("dot should succeed"));
    });
    bench("RustVecOps", || {
        black_box(VecOps::<f64>::dot(&RustVecOps, &a, &b).expect("dot should succeed"));
    });
}

fn vecops_cmul_shootout(n: usize) {
    eprintln!("\n=== VecOps cmul f64, N={n} ===");

    let a: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss, reason = "bench value")]
            let v = (i as f64) * 0.001;
            Complex::new(v, v + 0.5)
        })
        .collect();
    let b: Vec<Complex<f64>> = a.iter().map(|c| c + Complex::new(1.0, -1.0)).collect();
    let mut out = vec![Complex::default(); n];

    bench("HwyVecOps", || {
        VecOps::<f64>::cmul(&HwyVecOps, &a, &b, &mut out).expect("cmul should succeed");
        black_box(&out);
    });
    bench("RustVecOps", || {
        VecOps::<f64>::cmul(&RustVecOps, &a, &b, &mut out).expect("cmul should succeed");
        black_box(&out);
    });
}

fn vecops_mul_shootout(n: usize) {
    eprintln!("\n=== VecOps mul f64, N={n} ===");

    let a: Vec<f64> = (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss, reason = "bench value")]
            let v = (i as f64) * 0.001;
            v
        })
        .collect();
    let b: Vec<f64> = a.iter().map(|x| x + 1.0).collect();
    let mut out = vec![0.0_f64; n];

    bench("HwyVecOps", || {
        VecOps::<f64>::mul(&HwyVecOps, &a, &b, &mut out).expect("mul should succeed");
        black_box(&out);
    });
    bench("RustVecOps", || {
        VecOps::<f64>::mul(&RustVecOps, &a, &b, &mut out).expect("mul should succeed");
        black_box(&out);
    });
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn vecops_dot_1024() {
    vecops_dot_shootout(1024);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn vecops_dot_8192() {
    vecops_dot_shootout(8192);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn vecops_cmul_1024() {
    vecops_cmul_shootout(1024);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn vecops_cmul_8192() {
    vecops_cmul_shootout(8192);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn vecops_mul_1024() {
    vecops_mul_shootout(1024);
}

#[test]
#[ignore = "benchmark — run manually with make test T=shootout"]
fn vecops_mul_8192() {
    vecops_mul_shootout(8192);
}
