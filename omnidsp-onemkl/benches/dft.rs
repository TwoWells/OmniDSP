// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! DFT throughput: Intel oneMKL DFTI vs the pure-Rust `rustfft` / `realfft`
//! floor, across a range of transform lengths.
//!
//! Both engines implement the same `OmniDSP` primitive traits, so this is an
//! apples-to-apples `execute` comparison: `OneMklDftC2c` vs `RustDftC2c`
//! (complex FFT) and `OneMklDftR2c` vs `RustDftR2c` (real FFT, the path the
//! convolution / FIR / CQT modules actually use).  The plan is built once and
//! reused; only the per-call `execute` is timed — the realistic streaming case.
//!
//! Run with `make onemkl-bench` (or `cargo bench --manifest-path
//! omnidsp-onemkl/Cargo.toml`).  Links `libmkl_rt`, so it runs only where Intel
//! oneMKL is installed (the `omnidsp-ci` image).  FFT runtime is data-independent
//! for these algorithms, so buffers are reused without per-iteration refills.
//!
//! This is a coarse wall-clock harness (warmup + median), not a statistical
//! benchmark; on a shared CI runner the absolute numbers drift, but the
//! same-machine same-run *ratio* is the meaningful figure.

#![allow(
    clippy::expect_used,
    reason = "a benchmark should abort loudly if setup fails"
)]
#![allow(
    clippy::print_stdout,
    reason = "a benchmark reports its measurements to stdout"
)]
#![allow(
    clippy::cast_precision_loss,
    reason = "timing arithmetic on small counts is exact enough in f64"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "the synthetic test signal is clearer written as plain arithmetic"
)]
#![allow(
    clippy::similar_names,
    reason = "the bench uses the systematic c2c / r2c short names for paired bindings"
)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex;

use omnidsp_core::traits::dft::{
    DftC2c, DftC2cPlan, DftC2cSpec, DftNorm, DftR2c, DftR2cPlan, DftR2cSpec,
};
use omnidsp_core::types::Direction;
use omnidsp_onemkl::{OneMklDftC2c, OneMklDftR2c};
use omnidsp_rustfft::{RustDftC2c, RustDftR2c};

/// Median wall-clock of `f` over `iters` runs, after a short warmup.
fn median<F: FnMut()>(mut f: F, iters: usize) -> Duration {
    for _ in 0..5 {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// A deterministic broadband-ish real signal: a few summed sinusoids.
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            0.5 * (0.002 * x).sin() + 0.3 * (0.017 * x).sin() + 0.2 * (0.061 * x).sin()
        })
        .collect()
}

/// Iterations scaled to the transform size so each row measures a sane span.
fn iters_for(n: usize) -> usize {
    (40_000_000 / n).clamp(50, 4000)
}

/// Median per-call `execute` time (µs) of a complex-forward DFT factory.
fn time_c2c<F: DftC2c<f64>>(factory: &F, n: usize) -> f64 {
    let spec = DftC2cSpec::new(n, Direction::Forward, DftNorm::Inverse).expect("c2c spec");
    let plan = factory.create_plan(&spec).expect("c2c plan");
    let input: Vec<Complex<f64>> = signal(n)
        .into_iter()
        .map(|v| Complex::new(v, 0.0))
        .collect();
    let mut output = vec![Complex::new(0.0, 0.0); n];
    let dur = median(
        || {
            plan.execute(black_box(&input), black_box(&mut output))
                .expect("c2c execute");
        },
        iters_for(n),
    );
    dur.as_secs_f64() * 1e6
}

/// Median per-call `execute` time (µs) of a real-forward DFT factory.
///
/// FFT runtime is data-independent, so the input buffer (consumed as scratch by
/// the `realfft` floor) is reused across iterations without a refill.
fn time_r2c<F: DftR2c<f64>>(factory: &F, n: usize) -> f64 {
    let spec = DftR2cSpec::new(n, DftNorm::Inverse).expect("r2c spec");
    let plan = factory.create_plan(&spec).expect("r2c plan");
    let mut input = signal(n);
    let mut output = vec![Complex::new(0.0, 0.0); n / 2 + 1];
    let dur = median(
        || {
            plan.execute(black_box(&mut input), black_box(&mut output))
                .expect("r2c execute");
        },
        iters_for(n),
    );
    dur.as_secs_f64() * 1e6
}

fn row(n: usize) {
    let rust_c2c = time_c2c(&RustDftC2c, n);
    let mkl_c2c = time_c2c(&OneMklDftC2c, n);
    let rust_r2c = time_r2c(&RustDftR2c, n);
    let mkl_r2c = time_r2c(&OneMklDftR2c, n);
    println!(
        "{n:>7}  |  {rust_c2c:>9.2} {mkl_c2c:>9.2} {:>6.2}x  |  {rust_r2c:>9.2} {mkl_r2c:>9.2} {:>6.2}x",
        rust_c2c / mkl_c2c,
        rust_r2c / mkl_r2c,
    );
}

fn main() {
    println!("DFT per-call execute throughput (f64): rustfft/realfft floor vs Intel oneMKL DFTI");
    println!("lower µs is better; speedup = rust / mkl (>1 means MKL faster)\n");
    println!(
        "{:>7}  |  {:>9} {:>9} {:>7}  |  {:>9} {:>9} {:>7}",
        "len", "rust c2c", "mkl c2c", "speedup", "rust r2c", "mkl r2c", "speedup",
    );
    println!("{}", "-".repeat(74));
    for &n in &[1024, 4096, 16384, 65536, 262_144] {
        row(n);
    }
    println!("  (non-power-of-two lengths — mixed-radix / Bluestein territory)");
    for &n in &[1000, 6000, 44100] {
        row(n);
    }
}
