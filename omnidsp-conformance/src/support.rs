// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Shared float abstraction, tolerance constants, and comparison helpers.
//!
//! Every conformance check is generic over [`ConformanceFloat`], implemented for
//! `f64` (the tight-tolerance reference width) and `f32` (the primary DSP width,
//! carrying the looser tolerance).
//!
//! All tolerances are **documented shared constants** on the trait, applied with
//! the mixed absolute/relative rule
//! `|actual − expected| ≤ tol · (1 + |expected|)`, so a single per-operation
//! knob behaves sensibly for both near-zero and large-magnitude reference
//! values.  A per-backend loosening would be a reviewed, documented exception —
//! none exist today.

use std::fmt::Debug;
use std::ops::AddAssign;

use num_complex::Complex;
use num_traits::{Float, FromPrimitive};

/// A float width the conformance suite runs against.
///
/// Bundles the numeric bounds every module's `process` needs with the
/// per-operation tolerance constants and a width label for assertion messages.
/// Implemented only for `f32` and `f64`.
pub trait ConformanceFloat:
    Float + FromPrimitive + AddAssign + Debug + Send + Sync + 'static
{
    /// Width label for assertion messages (`"f32"` / `"f64"`).
    const WIDTH: &'static str;

    /// Tolerance for DFT round-trips and known-spectrum checks (all three
    /// primitives).
    const DFT_TOL: Self;
    /// Tolerance for convolution golden vectors and analytic cases.
    const CONV_TOL: Self;
    /// Tolerance for FIR `lfilter` golden vectors.
    const FIR_TOL: Self;
    /// Tolerance for IIR `sosfilt` golden vectors.
    const IIR_TOL: Self;
    /// Tolerance for DCT golden vectors (every type × norm).
    const DCT_TOL: Self;
    /// Tolerance for Hilbert analytic-signal golden vectors.
    const HILBERT_TOL: Self;
    /// Tolerance for cross-correlation golden vectors.
    const XCORR_TOL: Self;
    /// Tolerance for resampler `upfirdn` golden vectors.
    const RESAMPLE_TOL: Self;
    /// Tolerance for the multirate-CQT magnitude comparison against numpy.
    const CQT_TOL: Self;

    /// Narrow an `f64` reference value to this width.
    ///
    /// Golden vectors are stored as `f64`; this is the single conversion point
    /// to the width under test.
    fn lit(x: f64) -> Self;
}

impl ConformanceFloat for f64 {
    const WIDTH: &'static str = "f64";
    const DFT_TOL: Self = 1e-11;
    const CONV_TOL: Self = 1e-10;
    const FIR_TOL: Self = 1e-9;
    const IIR_TOL: Self = 1e-9;
    const DCT_TOL: Self = 1e-9;
    const HILBERT_TOL: Self = 1e-10;
    const XCORR_TOL: Self = 1e-7;
    const RESAMPLE_TOL: Self = 1e-7;
    const CQT_TOL: Self = 1e-3;

    fn lit(x: f64) -> Self {
        x
    }
}

impl ConformanceFloat for f32 {
    const WIDTH: &'static str = "f32";
    const DFT_TOL: Self = 1e-4;
    const CONV_TOL: Self = 1e-3;
    const FIR_TOL: Self = 1e-3;
    const IIR_TOL: Self = 5e-3;
    const DCT_TOL: Self = 1e-3;
    const HILBERT_TOL: Self = 1e-4;
    const XCORR_TOL: Self = 1e-3;
    const RESAMPLE_TOL: Self = 1e-3;
    const CQT_TOL: Self = 1e-2;

    fn lit(x: f64) -> Self {
        x as Self
    }
}

/// Mixed absolute/relative closeness: `|a − e| ≤ tol · (1 + |e|)`.
fn close<T: ConformanceFloat>(a: T, e: T, tol: T) -> bool {
    (a - e).abs() <= tol * (T::one() + e.abs())
}

/// Map an `f64` reference slice to the width under test.
pub fn to_vec<T: ConformanceFloat>(data: &[f64]) -> Vec<T> {
    data.iter().copied().map(T::lit).collect()
}

/// Assert a computed real buffer matches an `f64` reference within `tol`.
#[track_caller]
pub fn assert_real<T: ConformanceFloat>(actual: &[T], expected: &[f64], tol: T, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} [{}]: length {} != reference {}",
        T::WIDTH,
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e_ref)) in actual.iter().zip(expected).enumerate() {
        let e = T::lit(e_ref);
        assert!(
            close(a, e, tol),
            "{label} [{}]: index {i}: got {a:?}, expected {e:?}",
            T::WIDTH,
        );
    }
}

/// Assert two computed real buffers (same width) match within `tol`.
///
/// Used where the reference is itself computed at width `T` — e.g. the c2r
/// drift case compares the drifted-input output against the clean-input output.
#[track_caller]
pub fn assert_real_t<T: ConformanceFloat>(actual: &[T], expected: &[T], tol: T, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} [{}]: length {} != {}",
        T::WIDTH,
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            close(a, e, tol),
            "{label} [{}]: index {i}: got {a:?}, expected {e:?}",
            T::WIDTH,
        );
    }
}

/// Assert a computed complex buffer matches an `f64` `(re, im)` reference.
#[track_caller]
pub fn assert_complex<T: ConformanceFloat>(
    actual: &[Complex<T>],
    expected: &[(f64, f64)],
    tol: T,
    label: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} [{}]: length {} != reference {}",
        T::WIDTH,
        actual.len(),
        expected.len(),
    );
    for (i, (a, &(re, im))) in actual.iter().zip(expected).enumerate() {
        let e = Complex::new(T::lit(re), T::lit(im));
        let diff = (*a - e).norm();
        assert!(
            diff <= tol * (T::one() + e.norm()),
            "{label} [{}]: bin {i}: got ({:?}, {:?}), expected ({re}, {im}), diff {diff:?}",
            T::WIDTH,
            a.re,
            a.im,
        );
    }
}

/// Assert computed complex magnitudes match an `f64` `(re, im)` reference's
/// magnitudes.
///
/// Used by the multirate CQT, whose half-spectrum correlation drops
/// negative-frequency leakage and so matches the full-FFT reference in
/// magnitude rather than exact phase.
#[track_caller]
pub fn assert_magnitude<T: ConformanceFloat>(
    actual: &[Complex<T>],
    expected: &[(f64, f64)],
    tol: T,
    label: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} [{}]: length {} != reference {}",
        T::WIDTH,
        actual.len(),
        expected.len(),
    );
    for (i, (a, &(re, im))) in actual.iter().zip(expected).enumerate() {
        let e_mag = T::lit(re.hypot(im));
        assert!(
            close(a.norm(), e_mag, tol),
            "{label} [{}]: bin {i}: got |{:?}|, expected magnitude {e_mag:?}",
            T::WIDTH,
            a.norm(),
        );
    }
}
