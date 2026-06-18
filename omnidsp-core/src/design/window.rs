// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Window coefficient generation.
//!
//! This module contains the pure-math implementation behind
//! [`Window::coefficients`](crate::types::Window::coefficients).  All formulas
//! compute in f64 internally and convert to `T` at the end.

// Window math operates on sample indices that never exceed practical FFT sizes
// (well within f64's 52-bit mantissa), and the formulas read more clearly with
// the standard arithmetic operators than with mul_add chains.
#![allow(
    clippy::cast_precision_loss,
    reason = "sample indices are small enough for f64"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "standard window formulas are clearer without mul_add"
)]

use std::f64::consts::PI;

use num_traits::Float;

use crate::error::{Error, Result};
use crate::types::Window;

// ─── Public entry point ───────────────────────────────────────────────

/// Compute window coefficients for the given function and length.
///
/// Called by [`Window::coefficients`](crate::types::Window::coefficients).
///
/// # Errors
///
/// Returns [`Error::InvalidSpec`] for zero length or invalid parameters.
pub fn compute<T: Float>(window: &Window<T>, length: usize) -> Result<Vec<T>> {
    if length == 0 {
        return Err(Error::InvalidSpec("window length must be non-zero".into()));
    }
    if length == 1 {
        return Ok(vec![from_f64(1.0)?]);
    }

    match *window {
        Window::Bartlett => bartlett(length),
        Window::Blackman => blackman(length),
        Window::BlackmanHarris => blackman_harris(length),
        Window::FlatTop => flat_top(length),
        Window::Gaussian(sigma) => gaussian(length, to_f64(sigma)?),
        Window::Hamming => hamming(length),
        Window::Hann => hann(length),
        Window::Kaiser(beta) => kaiser(length, to_f64(beta)?),
        Window::Nuttall => nuttall(length),
        Window::Rectangular => rectangular(length),
        Window::Triangular => triangular(length),
        Window::Tukey(alpha) => tukey(length, to_f64(alpha)?),
    }
}

// ─── Conversion helpers ───────────────────────────────────────────────

fn to_f64<T: Float>(val: T) -> Result<f64> {
    val.to_f64()
        .ok_or_else(|| Error::Internal("failed to convert parameter to f64".into()))
}

fn from_f64<T: Float>(val: f64) -> Result<T> {
    T::from(val).ok_or_else(|| Error::Internal("failed to convert f64 to target type".into()))
}

/// Build a `Vec<T>` from an iterator of f64 values, converting each to `T`.
fn collect_f64<T: Float>(iter: impl Iterator<Item = f64>) -> Result<Vec<T>> {
    iter.map(from_f64).collect()
}

// ─── Window functions ─────────────────────────────────────────────────

/// Bartlett (triangular with zero endpoints).
///
/// `w[n] = 1 - |2n/(N-1) - 1|`
fn bartlett<T: Float>(n: usize) -> Result<Vec<T>> {
    let nm1 = (n - 1) as f64;
    collect_f64((0..n).map(|i| 1.0 - (2.0 * i as f64 / nm1 - 1.0).abs()))
}

/// Blackman window.
///
/// `w[n] = 0.42 - 0.5·cos(2πn/(N-1)) + 0.08·cos(4πn/(N-1))`
fn blackman<T: Float>(n: usize) -> Result<Vec<T>> {
    let nm1 = (n - 1) as f64;
    collect_f64((0..n).map(|i| {
        let x = 2.0 * PI * i as f64 / nm1;
        0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
    }))
}

/// Blackman–Harris window (4-term cosine sum, ≈ −92 dB peak sidelobes).
///
/// `w[n] = a0 − a1·cos(x) + a2·cos(2x) − a3·cos(3x)` where `x = 2πn/(N−1)`.
fn blackman_harris<T: Float>(n: usize) -> Result<Vec<T>> {
    const A0: f64 = 0.358_75;
    const A1: f64 = 0.488_29;
    const A2: f64 = 0.141_28;
    const A3: f64 = 0.011_68;

    let nm1 = (n - 1) as f64;
    collect_f64((0..n).map(|i| {
        let x = 2.0 * PI * i as f64 / nm1;
        A0 - A1 * x.cos() + A2 * (2.0 * x).cos() - A3 * (3.0 * x).cos()
    }))
}

/// Flat-top window (SFT5F / ISO 18431-2 coefficients).
///
/// 5-term cosine sum.
fn flat_top<T: Float>(n: usize) -> Result<Vec<T>> {
    const A0: f64 = 0.215_578_95;
    const A1: f64 = 0.416_631_58;
    const A2: f64 = 0.277_263_158;
    const A3: f64 = 0.083_578_947;
    const A4: f64 = 0.006_947_368;

    let nm1 = (n - 1) as f64;
    collect_f64((0..n).map(|i| {
        let x = 2.0 * PI * i as f64 / nm1;
        A0 - A1 * x.cos() + A2 * (2.0 * x).cos() - A3 * (3.0 * x).cos() + A4 * (4.0 * x).cos()
    }))
}

/// Gaussian window.
///
/// `w[n] = exp(-0.5·((n - (N-1)/2) / (σ·(N-1)/2))²)`
///
/// `sigma` is relative to half the window length.
fn gaussian<T: Float>(n: usize, sigma: f64) -> Result<Vec<T>> {
    if sigma <= 0.0 {
        return Err(Error::InvalidSpec("Gaussian sigma must be positive".into()));
    }
    let nm1 = (n - 1) as f64;
    let half = nm1 / 2.0;
    let denom = sigma * half;
    collect_f64((0..n).map(|i| {
        let t = (i as f64 - half) / denom;
        (-0.5 * t * t).exp()
    }))
}

/// Hamming window.
///
/// `w[n] = 0.54 - 0.46·cos(2πn/(N-1))`
fn hamming<T: Float>(n: usize) -> Result<Vec<T>> {
    let nm1 = (n - 1) as f64;
    collect_f64((0..n).map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / nm1).cos()))
}

/// Hann (raised cosine) window.
///
/// `w[n] = 0.5·(1 - cos(2πn/(N-1)))`
fn hann<T: Float>(n: usize) -> Result<Vec<T>> {
    let nm1 = (n - 1) as f64;
    collect_f64((0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / nm1).cos())))
}

/// Kaiser window.
///
/// `w[n] = I₀(β·√(1 - ((2n/(N-1)) - 1)²)) / I₀(β)`
fn kaiser<T: Float>(n: usize, beta: f64) -> Result<Vec<T>> {
    if beta < 0.0 {
        return Err(Error::InvalidSpec(
            "Kaiser beta must be non-negative".into(),
        ));
    }
    let nm1 = (n - 1) as f64;
    let i0_beta = bessel_i0(beta);
    collect_f64((0..n).map(|i| {
        let t = 2.0 * i as f64 / nm1 - 1.0;
        let arg = beta * (1.0 - t * t).max(0.0).sqrt();
        bessel_i0(arg) / i0_beta
    }))
}

/// Nuttall window (scipy's minimum 4-term cosine sum, ≈ −93 dB peak sidelobes).
///
/// `w[n] = a0 − a1·cos(x) + a2·cos(2x) − a3·cos(3x)` where `x = 2πn/(N−1)`.
fn nuttall<T: Float>(n: usize) -> Result<Vec<T>> {
    const A0: f64 = 0.363_581_9;
    const A1: f64 = 0.489_177_5;
    const A2: f64 = 0.136_599_5;
    const A3: f64 = 0.010_641_1;

    let nm1 = (n - 1) as f64;
    collect_f64((0..n).map(|i| {
        let x = 2.0 * PI * i as f64 / nm1;
        A0 - A1 * x.cos() + A2 * (2.0 * x).cos() - A3 * (3.0 * x).cos()
    }))
}

/// Rectangular (all ones) window.
fn rectangular<T: Float>(n: usize) -> Result<Vec<T>> {
    let one = from_f64(1.0)?;
    Ok(vec![one; n])
}

/// Triangular (non-zero endpoints) window.
///
/// Matches scipy's `triang`: for even N uses denominator N, for odd N
/// uses denominator N+1.
fn triangular<T: Float>(n: usize) -> Result<Vec<T>> {
    let nf = n as f64;
    if n.is_multiple_of(2) {
        // Even: w[i] = (2·min(i, N-1-i) + 1) / N
        collect_f64((0..n).map(|i| (2.0 * i.min(n - 1 - i) as f64 + 1.0) / nf))
    } else {
        // Odd: w[i] = 2·(min(i, N-1-i) + 1) / (N+1)
        collect_f64((0..n).map(|i| 2.0 * (i.min(n - 1 - i) as f64 + 1.0) / (nf + 1.0)))
    }
}

/// Tukey (tapered cosine) window.
///
/// `alpha = 0` gives rectangular, `alpha = 1` gives Hann.
fn tukey<T: Float>(n: usize, alpha: f64) -> Result<Vec<T>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(Error::InvalidSpec("Tukey alpha must be in [0, 1]".into()));
    }
    if alpha <= 0.0 {
        return rectangular(n);
    }
    if alpha >= 1.0 {
        return hann(n);
    }
    let nm1 = (n - 1) as f64;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "alpha * nm1 / 2 is non-negative and fits in usize for any practical window"
    )]
    let width = (alpha * nm1 / 2.0).floor() as usize;
    collect_f64((0..n).map(|i| {
        if i <= width {
            0.5 * (1.0 + (PI * (2.0 * i as f64 / (alpha * nm1) - 1.0)).cos())
        } else if i >= n - 1 - width {
            0.5 * (1.0 + (PI * (2.0 * i as f64 / (alpha * nm1) - 2.0 / alpha + 1.0)).cos())
        } else {
            1.0
        }
    }))
}

// ─── Bessel function ──────────────────────────────────────────────────

/// Modified Bessel function of the first kind, order zero.
///
/// Uses the power series: `I₀(x) = Σ ((x/2)^k / k!)²` for k = 0, 1, 2, ...
///
/// Converges within ~25 terms for all practical arguments (beta up to ~40).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=50 {
        let factor = x / (2.0 * f64::from(k));
        term *= factor * factor;
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;

    /// Tolerance for f64 comparisons.
    const TOL: f64 = 1e-10;

    /// Assert two slices are element-wise equal within tolerance.
    fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch: got {}, expected {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < TOL,
                "{label}[{i}]: got {a}, expected {e}, diff {}",
                (a - e).abs()
            );
        }
    }

    /// Helper to compute f64 window coefficients.
    fn win(winfn: &Window<f64>, length: usize) -> Vec<f64> {
        compute::<f64>(winfn, length).expect("window computation should succeed")
    }

    // ── Length edge cases ─────────────────────────────────────────

    #[test]
    fn length_zero_is_error() {
        let result = compute::<f64>(&Window::Hann, 0);
        assert!(result.is_err(), "length 0 should return error");
    }

    #[test]
    fn length_one_returns_one_for_all_types() {
        let fns: Vec<Window<f64>> = vec![
            Window::Bartlett,
            Window::Blackman,
            Window::BlackmanHarris,
            Window::FlatTop,
            Window::Gaussian(0.4),
            Window::Hamming,
            Window::Hann,
            Window::Kaiser(5.0),
            Window::Nuttall,
            Window::Rectangular,
            Window::Triangular,
            Window::Tukey(0.5),
        ];
        for winfn in &fns {
            let c = win(winfn, 1);
            assert_eq!(c.len(), 1, "{winfn:?}: length should be 1");
            assert!(
                (c[0] - 1.0).abs() < TOL,
                "{winfn:?}: value should be 1.0, got {}",
                c[0]
            );
        }
    }

    // ── Rectangular ───────────────────────────────────────────────

    #[test]
    fn rectangular_all_ones() {
        let c = win(&Window::Rectangular, 8);
        assert_close(&c, &[1.0; 8], "rectangular(8)");
    }

    // ── Bartlett ──────────────────────────────────────────────────

    #[test]
    fn bartlett_5() {
        let c = win(&Window::Bartlett, 5);
        assert_close(&c, &[0.0, 0.5, 1.0, 0.5, 0.0], "bartlett(5)");
    }

    #[test]
    fn bartlett_endpoints_are_zero() {
        let c = win(&Window::Bartlett, 64);
        assert!(c[0].abs() < TOL, "bartlett left endpoint should be 0");
        assert!(c[63].abs() < TOL, "bartlett right endpoint should be 0");
    }

    // ── Triangular ───────────────────────────────────────────────

    #[test]
    fn triangular_5() {
        // Odd N: w[n] = 2*(min(n,N-1-n)+1)/(N+1)
        let c = win(&Window::Triangular, 5);
        let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0];
        assert_close(&c, &expected, "triangular(5)");
    }

    #[test]
    fn triangular_4() {
        // Even N: w[n] = (2*min(n,N-1-n)+1)/N
        let c = win(&Window::Triangular, 4);
        assert_close(&c, &[0.25, 0.75, 0.75, 0.25], "triangular(4)");
    }

    #[test]
    fn triangular_endpoints_nonzero() {
        let c = win(&Window::Triangular, 64);
        assert!(c[0] > 0.0, "triangular left endpoint should be > 0");
        assert!(c[63] > 0.0, "triangular right endpoint should be > 0");
    }

    // ── Hann ─────────────────────────────────────────────────────

    #[test]
    fn hann_5() {
        let c = win(&Window::Hann, 5);
        assert_close(&c, &[0.0, 0.5, 1.0, 0.5, 0.0], "hann(5)");
    }

    #[test]
    fn hann_symmetric() {
        let c = win(&Window::Hann, 64);
        for i in 0..32 {
            assert!(
                (c[i] - c[63 - i]).abs() < TOL,
                "hann(64) not symmetric at index {i}"
            );
        }
    }

    // ── Hamming ──────────────────────────────────────────────────

    #[test]
    fn hamming_5() {
        let c = win(&Window::Hamming, 5);
        assert_close(&c, &[0.08, 0.54, 1.0, 0.54, 0.08], "hamming(5)");
    }

    #[test]
    fn hamming_endpoints_nonzero() {
        let c = win(&Window::Hamming, 64);
        assert!(
            (c[0] - 0.08).abs() < TOL,
            "hamming left endpoint should be ≈0.08"
        );
    }

    // ── Blackman ─────────────────────────────────────────────────

    #[test]
    fn blackman_5() {
        // w[n] = 0.42 - 0.5*cos(2πn/4) + 0.08*cos(4πn/4)
        // n=0: 0.42 - 0.5 + 0.08 = 0.0
        // n=1: 0.42 - 0.5*cos(π/2) + 0.08*cos(π) = 0.42 - 0 - 0.08 = 0.34
        // n=2: 0.42 + 0.5 + 0.08 = 1.0
        let c = win(&Window::Blackman, 5);
        assert_close(&c, &[0.0, 0.34, 1.0, 0.34, 0.0], "blackman(5)");
    }

    // ── Blackman–Harris ──────────────────────────────────────────

    #[test]
    fn blackman_harris_5() {
        // scipy.signal.windows.blackmanharris(5):
        // [0.00006, 0.21747, 1.0, 0.21747, 0.00006].
        // At N=5, x=πn/2: n=0 → a0−a1+a2−a3, n=1 → a0−a2, n=2 → a0+a1+a2+a3.
        let c = win(&Window::BlackmanHarris, 5);
        let a0 = 0.358_75;
        let a1 = 0.488_29;
        let a2 = 0.141_28;
        let a3 = 0.011_68;
        let expected = [
            a0 - a1 + a2 - a3,
            a0 - a2,
            a0 + a1 + a2 + a3,
            a0 - a2,
            a0 - a1 + a2 - a3,
        ];
        assert_close(&c, &expected, "blackman_harris(5)");
        // Sanity against the published golden rounded to 5 digits.
        assert!(
            (c[0] - 0.00006).abs() < 1e-5,
            "blackman_harris(5)[0] should be ≈0.00006, got {}",
            c[0]
        );
        assert!(
            (c[1] - 0.21747).abs() < 1e-5,
            "blackman_harris(5)[1] should be ≈0.21747, got {}",
            c[1]
        );
    }

    #[test]
    fn blackman_harris_symmetric() {
        let c = win(&Window::BlackmanHarris, 64);
        for i in 0..32 {
            assert!(
                (c[i] - c[63 - i]).abs() < TOL,
                "blackman_harris(64) not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn blackman_harris_peak_near_one() {
        // The peak of a Blackman–Harris window is 1.0 at the center.
        let c = win(&Window::BlackmanHarris, 65);
        assert!(
            (c[32] - 1.0).abs() < 1e-6,
            "blackman_harris(65) center should be ≈1.0, got {}",
            c[32]
        );
    }

    // ── Nuttall ──────────────────────────────────────────────────

    #[test]
    fn nuttall_5() {
        // scipy.signal.windows.nuttall(5):
        // [0.0003628, 0.2269824, 1.0, 0.2269824, 0.0003628].
        // At N=5, x=πn/2: n=0 → a0−a1+a2−a3, n=1 → a0−a2, n=2 → a0+a1+a2+a3.
        let c = win(&Window::Nuttall, 5);
        let a0 = 0.363_581_9;
        let a1 = 0.489_177_5;
        let a2 = 0.136_599_5;
        let a3 = 0.010_641_1;
        let expected = [
            a0 - a1 + a2 - a3,
            a0 - a2,
            a0 + a1 + a2 + a3,
            a0 - a2,
            a0 - a1 + a2 - a3,
        ];
        assert_close(&c, &expected, "nuttall(5)");
        // Sanity against the published golden rounded to 7 digits.
        assert!(
            (c[0] - 0.000_362_8).abs() < 1e-7,
            "nuttall(5)[0] should be ≈0.0003628, got {}",
            c[0]
        );
        assert!(
            (c[1] - 0.226_982_4).abs() < 1e-7,
            "nuttall(5)[1] should be ≈0.2269824, got {}",
            c[1]
        );
    }

    #[test]
    fn nuttall_symmetric() {
        let c = win(&Window::Nuttall, 64);
        for i in 0..32 {
            assert!(
                (c[i] - c[63 - i]).abs() < TOL,
                "nuttall(64) not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn nuttall_peak_near_one() {
        // The peak of a Nuttall window is 1.0 at the center.
        let c = win(&Window::Nuttall, 65);
        assert!(
            (c[32] - 1.0).abs() < 1e-6,
            "nuttall(65) center should be ≈1.0, got {}",
            c[32]
        );
    }

    // ── FlatTop ──────────────────────────────────────────────────

    #[test]
    fn flat_top_peak_near_one() {
        // The peak of a flat-top window is 1.0 at the center.
        let c = win(&Window::FlatTop, 65);
        assert!(
            (c[32] - 1.0).abs() < 1e-6,
            "flat_top(65) center should be ≈1.0, got {}",
            c[32]
        );
    }

    #[test]
    fn flat_top_has_negative_sidelobes() {
        // Flat-top windows characteristically go negative near the edges.
        let c = win(&Window::FlatTop, 64);
        assert!(
            c.iter().any(|&v| v < 0.0),
            "flat_top should have negative values"
        );
    }

    // ── Gaussian ─────────────────────────────────────────────────

    #[test]
    fn gaussian_peak_at_center() {
        // Odd length: center sample is exactly at (N-1)/2, so exp(0) = 1.
        let c = win(&Window::Gaussian(0.4), 65);
        assert!(
            (c[32] - 1.0).abs() < TOL,
            "gaussian(65) center should be 1.0, got {}",
            c[32]
        );
    }

    #[test]
    fn gaussian_symmetric() {
        let c = win(&Window::Gaussian(0.4), 65);
        for i in 0..32 {
            assert!(
                (c[i] - c[64 - i]).abs() < TOL,
                "gaussian(65) not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn gaussian_invalid_sigma() {
        assert!(
            compute::<f64>(&Window::Gaussian(0.0), 64).is_err(),
            "sigma=0 should error"
        );
        assert!(
            compute::<f64>(&Window::Gaussian(-1.0), 64).is_err(),
            "sigma<0 should error"
        );
    }

    // ── Kaiser ───────────────────────────────────────────────────

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        // I₀(0) = 1 for all n, so Kaiser with β=0 is rectangular.
        let c = win(&Window::Kaiser(0.0), 8);
        assert_close(&c, &[1.0; 8], "kaiser(8, beta=0)");
    }

    #[test]
    fn kaiser_symmetric() {
        let c = win(&Window::Kaiser(5.0), 64);
        for i in 0..32 {
            assert!(
                (c[i] - c[63 - i]).abs() < TOL,
                "kaiser(64, beta=5) not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn kaiser_peak_at_center() {
        // Center sample: t=0, arg=beta, I₀(β)/I₀(β) = 1.
        let c = win(&Window::Kaiser(14.0), 5);
        assert!(
            (c[2] - 1.0).abs() < TOL,
            "kaiser(5, beta=14) center should be 1.0, got {}",
            c[2]
        );
    }

    #[test]
    fn kaiser_endpoints_equal() {
        // Endpoints: t = ±1, arg = 0, I₀(0)/I₀(β) = 1/I₀(β) — both equal.
        let c = win(&Window::Kaiser(14.0), 5);
        assert!(
            (c[0] - c[4]).abs() < TOL,
            "kaiser(5, beta=14) endpoints should be equal"
        );
        // Endpoints should be small for high beta.
        assert!(
            c[0] < 0.001,
            "kaiser(5, beta=14) endpoint should be small, got {}",
            c[0]
        );
    }

    #[test]
    fn kaiser_invalid_beta() {
        assert!(
            compute::<f64>(&Window::Kaiser(-1.0), 64).is_err(),
            "negative beta should error"
        );
    }

    // ── Tukey ────────────────────────────────────────────────────

    #[test]
    fn tukey_alpha_zero_is_rectangular() {
        let c = win(&Window::Tukey(0.0), 8);
        assert_close(&c, &[1.0; 8], "tukey(8, alpha=0)");
    }

    #[test]
    fn tukey_alpha_one_is_hann() {
        let tukey_w = win(&Window::Tukey(1.0), 64);
        let hann_w = win(&Window::Hann, 64);
        assert_close(&tukey_w, &hann_w, "tukey(64, alpha=1) vs hann(64)");
    }

    #[test]
    fn tukey_symmetric() {
        let c = win(&Window::Tukey(0.5), 64);
        for i in 0..32 {
            assert!(
                (c[i] - c[63 - i]).abs() < TOL,
                "tukey(64, alpha=0.5) not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn tukey_invalid_alpha() {
        assert!(
            compute::<f64>(&Window::Tukey(-0.1), 64).is_err(),
            "alpha<0 should error"
        );
        assert!(
            compute::<f64>(&Window::Tukey(1.1), 64).is_err(),
            "alpha>1 should error"
        );
    }

    // ── Bessel I₀ ────────────────────────────────────────────────

    #[test]
    fn bessel_i0_known_values() {
        // I₀(0) = 1
        assert!((bessel_i0(0.0) - 1.0).abs() < TOL, "I₀(0) should be 1");
        // I₀(1) ≈ 1.2660658777520084
        assert!(
            (bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-12,
            "I₀(1) mismatch"
        );
        // I₀(5) ≈ 27.239871823604442
        assert!(
            (bessel_i0(5.0) - 27.239_871_823_604_442).abs() < 1e-9,
            "I₀(5) mismatch"
        );
    }

    // ── f32 path ─────────────────────────────────────────────────

    #[test]
    fn f32_hann_matches_f64() {
        let c32 = compute::<f32>(&Window::Hann, 5).expect("f32 hann should succeed");
        let c64 = win(&Window::Hann, 5);
        assert_eq!(c32.len(), c64.len(), "f32/f64 length mismatch");
        for (i, (a, b)) in c32.iter().zip(&c64).enumerate() {
            assert!(
                (f64::from(*a) - b).abs() < 1e-6,
                "f32 hann[{i}]: got {a}, expected {b}"
            );
        }
    }
}
