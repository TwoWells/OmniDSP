// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! CQT throughput: the octave-recursive multirate [`OmniCqt`] vs the single-FFT
//! [`SingleFftCqt`] baseline, across frequency ranges of growing octave span.
//!
//! Run with `make bench` (or `cargo bench -p omnidsp-core --features bench`).
//! Prints, per range, the median per-frame `process` time of each path and the
//! speedup.  Both run on the `RustBackend` floor (`RustFFT` / `realfft` +
//! `ScalarVecOps`) — the same primitives the WASM demo ships.
//!
//! This is a coarse wall-clock harness (warmup + median), not a statistical
//! benchmark; it is meant to confirm the order-of-magnitude compute argument,
//! not to micro-optimise.

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

use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex;
use omnidsp_core::design::cqt;
use omnidsp_core::modules::cqt::{OmniCqt, SingleFftCqt};
use omnidsp_core::modules::resample::OmniResample;
use omnidsp_core::scalar::ScalarVecOps;
use omnidsp_core::window;
use omnidsp_rustfft::{RustDftC2c, RustDftR2c};

/// Median wall-clock of `f` over `iters` runs, after a short warmup.
fn median<F: FnMut()>(mut f: F, iters: usize) -> Duration {
    for _ in 0..3 {
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

/// A deterministic broadband-ish frame: a few summed sinusoids.
fn frame(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            0.5 * (0.002 * x).sin() + 0.3 * (0.017 * x).sin() + 0.2 * (0.061 * x).sin()
        })
        .collect()
}

fn run(label: &str, sr: f64, fmin: f64, fmax: f64, bpo: u32) {
    let spec = cqt::design::<f64>(sr, fmin, fmax, bpo, &window::hann()).expect("cqt design");
    let n = spec.fft_length();
    let bins = spec.num_bins();

    let naive = SingleFftCqt::new(RustDftC2c, ScalarVecOps)
        .create_plan(&spec)
        .expect("single-fft plan");
    let multi = OmniCqt::new(RustDftR2c, ScalarVecOps)
        .create_plan(&spec, &OmniResample::new(ScalarVecOps))
        .expect("multirate plan");
    let octaves = multi.num_octaves();

    let input = frame(n);
    let mut out_naive = vec![Complex::new(0.0, 0.0); bins];
    let mut out_multi = vec![Complex::new(0.0, 0.0); bins];

    // Scale iterations to the frame size so each path measures for a sane span.
    let iters = (4_000_000 / n).clamp(20, 400);

    let t_naive = median(
        || {
            naive
                .process(black_box(&input), black_box(&mut out_naive))
                .expect("naive process");
        },
        iters,
    );
    let t_multi = median(
        || {
            multi
                .process(black_box(&input), black_box(&mut out_multi))
                .expect("multirate process");
        },
        iters,
    );

    let us_naive = t_naive.as_secs_f64() * 1e6;
    let us_multi = t_multi.as_secs_f64() * 1e6;
    let speedup = us_naive / us_multi;
    println!(
        "{label:<14} fft={n:>6}  bins={bins:>3}  oct={octaves:>2}  \
         naive={us_naive:>9.1} us  multi={us_multi:>9.1} us  speedup={speedup:>5.2}x"
    );
}

fn main() {
    println!("CQT per-frame throughput: single-FFT (naive) vs octave-recursive multirate");
    println!("44.1 kHz, 12 bins/octave, RustFFT/realfft + ScalarVecOps floor\n");
    run("2 octaves", 44100.0, 220.0, 880.0, 12);
    run("4 octaves", 44100.0, 110.0, 1760.0, 12);
    run("6 octaves", 44100.0, 55.0, 3520.0, 12);
    run("8 octaves", 44100.0, 40.0, 10000.0, 12);
    run("audio 20-16k", 44100.0, 20.0, 16000.0, 12);
}
