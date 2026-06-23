// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `VecOps` throughput: Intel oneMKL VM / BLAS L1 vs the scalar (LLVM
//! auto-vectorized) floor, across a range of vector lengths.
//!
//! `OneMklBackend` overrides the accelerable `VecOps` methods with oneMKL VM
//! (`v?Mul`, `v?Add`, conjugate, complex multiply) and BLAS L1 (`?dot`, `?scal`,
//! `?axpy`); `ScalarVecOps` is the per-element floor that `RustBackend` ships.
//! Both implement `VecOps<f64>`, so this times the same trait calls head-to-head.
//!
//! Benched ops: `mul` (real elementwise), `cmul` (complex elementwise — the
//! spectral-multiply hot path in conv / FIR / CQT / xcorr / Hilbert), and `dot`
//! (the FIR-direct / resample reduction).
//!
//! These ops are ns-scale, so each measurement times a **batch** of calls and
//! divides — a single-call `Instant` would be swamped by timer overhead. Built
//! with `target-cpu=native` (see `make onemkl-bench`) so the Rust floor gets the
//! same vector ISA oneMKL dispatches to; `MKL_THREADING_LAYER=SEQUENTIAL` in the
//! CI image keeps MKL single-threaded, matching the single-thread scalar floor.
//!
//! Run with `make onemkl-bench`. Links `libmkl_rt` (the `omnidsp-ci` image).

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
    reason = "index->float synthesis and call-count arithmetic are exact at these sizes"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "the synthetic test signal is clearer written as plain arithmetic"
)]
#![allow(
    clippy::similar_names,
    reason = "the bench uses systematic scal_/mkl_ paired binding names"
)]

use std::hint::black_box;

use num_complex::Complex;

use omnidsp_core::scalar::ScalarVecOps;
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_onemkl::OneMklBackend;

/// Median per-call time (nanoseconds) of `f`, timed in batches so the result is
/// not dominated by `Instant` overhead.  `batch` calls are timed together and
/// divided; the median of `samples` such batches is returned.
fn per_call_ns<F: FnMut()>(mut f: F, n: usize) -> f64 {
    let batch = (100_000 / n.max(1)).clamp(4, 20_000);
    let samples = 100;
    for _ in 0..batch.max(16) {
        f();
    }
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = std::time::Instant::now();
        for _ in 0..batch {
            f();
        }
        times.push(start.elapsed().as_secs_f64() / batch as f64);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2] * 1e9
}

fn rdata(n: usize, phase: f64) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 0.001 + phase).sin()).collect()
}

fn cdata(n: usize, phase: f64) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let x = i as f64 * 0.001;
            Complex::new((x + phase).sin(), (x + phase).cos())
        })
        .collect()
}

fn row(n: usize) {
    let mkl = OneMklBackend::new();
    let a = rdata(n, 0.0);
    let b = rdata(n, 1.0);
    let mut out = vec![0.0_f64; n];
    let ca = cdata(n, 0.0);
    let cb = cdata(n, 1.0);
    let mut cout = vec![Complex::new(0.0, 0.0); n];

    let scal_mul = per_call_ns(
        || {
            ScalarVecOps
                .mul(black_box(&a), black_box(&b), black_box(&mut out))
                .expect("scalar mul");
        },
        n,
    );
    let mkl_mul = per_call_ns(
        || {
            mkl.mul(black_box(&a), black_box(&b), black_box(&mut out))
                .expect("mkl mul");
        },
        n,
    );

    let scal_cmul = per_call_ns(
        || {
            ScalarVecOps
                .cmul(black_box(&ca), black_box(&cb), black_box(&mut cout))
                .expect("scalar cmul");
        },
        n,
    );
    let mkl_cmul = per_call_ns(
        || {
            mkl.cmul(black_box(&ca), black_box(&cb), black_box(&mut cout))
                .expect("mkl cmul");
        },
        n,
    );

    let scal_dot = per_call_ns(
        || {
            black_box(
                ScalarVecOps
                    .dot(black_box(&a), black_box(&b))
                    .expect("scalar dot"),
            );
        },
        n,
    );
    let mkl_dot = per_call_ns(
        || {
            black_box(mkl.dot(black_box(&a), black_box(&b)).expect("mkl dot"));
        },
        n,
    );

    println!(
        "{n:>8}  |  {scal_mul:>9.1} {mkl_mul:>9.1} {:>6.2}x  |  \
         {scal_cmul:>9.1} {mkl_cmul:>9.1} {:>6.2}x  |  {scal_dot:>9.1} {mkl_dot:>9.1} {:>6.2}x",
        scal_mul / mkl_mul,
        scal_cmul / mkl_cmul,
        scal_dot / mkl_dot,
    );
}

fn main() {
    println!("VecOps per-call execute (f64, ns/call): scalar floor vs Intel oneMKL VM/BLAS L1");
    println!(
        "target-cpu=native, single-threaded; lower ns is better; speedup = scalar / mkl \
         (>1 means MKL faster)\n"
    );
    println!(
        "{:>8}  |  {:>9} {:>9} {:>7}  |  {:>9} {:>9} {:>7}  |  {:>9} {:>9} {:>7}",
        "len",
        "scal mul",
        "mkl mul",
        "speedup",
        "scal cmul",
        "mkl cmul",
        "speedup",
        "scal dot",
        "mkl dot",
        "speedup",
    );
    println!("{}", "-".repeat(104));
    for &n in &[64, 256, 1024, 4096, 16384, 65536, 262_144, 1_048_576] {
        row(n);
    }
}
