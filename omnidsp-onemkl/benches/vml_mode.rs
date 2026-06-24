// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Experiment: does oneMKL's VM accuracy/performance mode change `cmul` cost?
//!
//! Complex multiply is exact, so HA (high accuracy) / LA (low) / EP (enhanced
//! performance) cannot change the *result* — this isolates whether the default
//! High-Accuracy mode carries per-call frontend overhead (special-value handling
//! / error checks) that EP sheds. The bench prints `vmlGetMode()` after each
//! switch to prove the mode actually changed, since a *working* switch may
//! legitimately show no speedup for an exact op. If EP ≈ HA, the cost is the VM
//! frontend + kernel, not the accuracy mode. The scalar floor (no VM frontend)
//! anchors the comparison.
//!
//! `target-cpu=native`, single-threaded (`MKL_THREADING_LAYER=SEQUENTIAL`),
//! batched median timer. Run via `make onemkl-bench`; links `libmkl_rt`.

#![allow(
    unsafe_code,
    reason = "the experiment switches VM mode by calling vmlSetMode/vmlGetMode directly"
)]
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
    reason = "index/count arithmetic is exact at these sizes"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "the synthetic test signal is clearer written as plain arithmetic"
)]
#![allow(
    clippy::similar_names,
    reason = "the experiment uses systematic ha/la/ep paired binding names"
)]

use std::hint::black_box;

use num_complex::Complex;

use omnidsp_core::scalar::ScalarVecOps;
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_onemkl::OneMklBackend;
use omnidsp_onemkl_sys::{VML_EP, VML_HA, VML_LA, vmlGetMode, vmlSetMode};

/// Median per-call time (nanoseconds) of `f`, timed in batches.
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

fn cdata(n: usize, phase: f64) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let x = i as f64 * 0.001;
            Complex::new((x + phase).sin(), (x + phase).cos())
        })
        .collect()
}

fn cmul_ns<V: VecOps<f64>>(v: &V, n: usize) -> f64 {
    let a = cdata(n, 0.0);
    let b = cdata(n, 1.0);
    let mut out = vec![Complex::new(0.0, 0.0); n];
    per_call_ns(
        || {
            v.cmul(black_box(&a), black_box(&b), black_box(&mut out))
                .expect("cmul");
        },
        n,
    )
}

/// Set the VM mode and read it back to confirm the switch took effect.
fn switch_mode(mode: u32) -> u32 {
    unsafe { vmlSetMode(mode) };
    unsafe { vmlGetMode() }
}

fn main() {
    let mkl = OneMklBackend::new();

    let default_mode = unsafe { vmlGetMode() };
    let confirm_ha = switch_mode(VML_HA);
    let confirm_la = switch_mode(VML_LA);
    let confirm_ep = switch_mode(VML_EP);

    println!("oneMKL VM mode experiment — f64 cmul (ns/call), target-cpu=native, single-thread");
    println!(
        "mode confirm: default={default_mode:#010x}  HA={confirm_ha:#010x}  \
         LA={confirm_la:#010x}  EP={confirm_ep:#010x}"
    );
    println!("(cmul is exact: HA/LA/EP change cost, never the result)\n");
    println!(
        "{:>8}  |  {:>10}  |  {:>10} {:>10} {:>10}",
        "len", "scalar", "mkl HA", "mkl LA", "mkl EP"
    );
    println!("{}", "-".repeat(60));
    for &n in &[1024, 8192, 65536, 262_144] {
        let scalar = cmul_ns(&ScalarVecOps, n);
        switch_mode(VML_HA);
        let mkl_ha = cmul_ns(&mkl, n);
        switch_mode(VML_LA);
        let mkl_la = cmul_ns(&mkl, n);
        switch_mode(VML_EP);
        let mkl_ep = cmul_ns(&mkl, n);
        println!("{n:>8}  |  {scalar:>10.1}  |  {mkl_ha:>10.1} {mkl_la:>10.1} {mkl_ep:>10.1}");
    }
}
