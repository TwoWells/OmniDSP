// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Conformance checks for the three DFT primitives (`DftC2c` / `DftR2c` /
//! `DftC2r`, ADR-009) and the Hermitian-shaping contract (ADR-010).
//!
//! These exercise the **output-affecting** spec fields — direction
//! (forward/inverse) and normalization (`None`/`Inverse`/`Ortho`) — across
//! representative lengths (even, odd, prime, `len == 1`), plus the
//! drift-tolerant c2r contract (ADR-009 §4).

use num_complex::Complex;

use omnidsp_core::create::CreatePlan;
use omnidsp_core::traits::dft::{
    DftC2cPlan, DftC2cSpec, DftC2rPlan, DftC2rSpec, DftNorm, DftR2cPlan, DftR2cSpec,
};
use omnidsp_core::types::Direction;

use crate::support::{ConformanceFloat, assert_complex, assert_real, assert_real_t, to_vec};

/// Round-trip normalization factor: `IFFT(FFT(x)) = scale · x`.
const fn round_trip_scale(norm: DftNorm, n: usize) -> f64 {
    match norm {
        DftNorm::None => n as f64,
        DftNorm::Inverse | DftNorm::Ortho => 1.0,
    }
}

const NORMS: [DftNorm; 3] = [DftNorm::None, DftNorm::Inverse, DftNorm::Ortho];

/// Conformance for the complex-to-complex DFT ([`DftC2c`](omnidsp_core::traits::dft::DftC2c)).
///
/// Covers forward→inverse round-trips over every `DftNorm` and representative
/// lengths, a known constant→DC spectrum, and a buffer-length error case.
///
/// # Panics
///
/// Panics if any transform deviates from the analytic result beyond
/// [`ConformanceFloat::DFT_TOL`], or if an invalid call fails to error.
pub fn check_dft_c2c<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<DftC2cSpec<T>>,
    B::Plan: DftC2cPlan<T>,
{
    let zero = Complex::new(T::zero(), T::zero());

    for &n in &[1usize, 2, 4, 5, 7, 8] {
        let re: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let im: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 1.0).collect();
        let input: Vec<Complex<T>> = re
            .iter()
            .zip(&im)
            .map(|(&r, &i)| Complex::new(T::lit(r), T::lit(i)))
            .collect();

        for &norm in &NORMS {
            let fwd_spec =
                DftC2cSpec::<T>::new(n, Direction::Forward, norm).expect("valid c2c forward spec");
            let inv_spec =
                DftC2cSpec::<T>::new(n, Direction::Inverse, norm).expect("valid c2c inverse spec");
            let fwd = b.create_plan(&fwd_spec).expect("c2c forward plan");
            let inv = b.create_plan(&inv_spec).expect("c2c inverse plan");

            let mut spectrum = vec![zero; n];
            fwd.process(&input, &mut spectrum)
                .expect("c2c forward process");
            let mut recovered = vec![zero; n];
            inv.process(&spectrum, &mut recovered)
                .expect("c2c inverse process");

            let scale = round_trip_scale(norm, n);
            let expected: Vec<(f64, f64)> = re
                .iter()
                .zip(&im)
                .map(|(&r, &i)| (r * scale, i * scale))
                .collect();
            assert_complex(
                &recovered,
                &expected,
                T::DFT_TOL,
                &format!("c2c round-trip n={n} {norm:?}"),
            );
        }
    }

    // Known spectrum: forward (unnormalized) of a constant concentrates all
    // energy in the DC bin.
    let n = 8;
    let c = 2.0;
    let constant = vec![Complex::new(T::lit(c), T::zero()); n];
    let spec = DftC2cSpec::<T>::new(n, Direction::Forward, DftNorm::None).expect("valid spec");
    let plan = b.create_plan(&spec).expect("c2c plan");
    let mut spectrum = vec![zero; n];
    plan.process(&constant, &mut spectrum)
        .expect("c2c forward process");
    let mut expected = vec![(0.0, 0.0); n];
    expected[0] = (n as f64 * c, 0.0);
    assert_complex(&spectrum, &expected, T::DFT_TOL, "c2c constant→DC spectrum");

    // Error case: output length must match the plan length.
    let mut wrong = vec![zero; n - 1];
    assert!(
        plan.process(&constant, &mut wrong).is_err(),
        "c2c [{}]: output-length mismatch must error",
        T::WIDTH,
    );
}

/// Conformance for the real-to-complex DFT ([`DftR2c`](omnidsp_core::traits::dft::DftR2c)).
///
/// Covers a known constant→DC half-spectrum, the Hermitian-shaping contract
/// (ADR-010 §3: bit-exactly-real DC and even-`N` Nyquist), and an error case,
/// over even and odd lengths.
///
/// # Panics
///
/// Panics if the half-spectrum deviates from the analytic result, if DC/Nyquist
/// imaginary parts are not bit-exactly zero, or if an invalid call fails to
/// error.
pub fn check_dft_r2c<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<DftR2cSpec<T>>,
    B::Plan: DftR2cPlan<T>,
{
    let zero = Complex::new(T::zero(), T::zero());

    for &n in &[8usize, 7] {
        let bins = n / 2 + 1;

        // Known: constant input → DC bin = N·c, all other bins zero.
        let c = 1.5;
        let spec = DftR2cSpec::<T>::new(n, DftNorm::None).expect("valid r2c spec");
        let plan = b.create_plan(&spec).expect("r2c plan");
        let mut constant = vec![T::lit(c); n];
        let mut spectrum = vec![zero; bins];
        plan.process(&mut constant, &mut spectrum)
            .expect("r2c process");
        let mut expected = vec![(0.0, 0.0); bins];
        expected[0] = (n as f64 * c, 0.0);
        assert_complex(
            &spectrum,
            &expected,
            T::DFT_TOL,
            &format!("r2c constant→DC spectrum n={n}"),
        );

        // Hermitian shaping (ADR-010 §3): a generic real input's DC — and, for
        // even N, Nyquist — imaginary parts come out bit-exactly zero.
        let mut signal = to_vec::<T>(
            &(0..n)
                .map(|i| (i as f64 * 0.7).sin() + 0.3)
                .collect::<Vec<_>>(),
        );
        let mut spectrum = vec![zero; bins];
        plan.process(&mut signal, &mut spectrum)
            .expect("r2c process");
        assert!(
            spectrum[0].im.is_zero(),
            "r2c [{}]: DC imaginary must be bit-exactly zero (ADR-010 §3)",
            T::WIDTH,
        );
        if n % 2 == 0 {
            assert!(
                spectrum[bins - 1].im.is_zero(),
                "r2c [{}]: even-N Nyquist imaginary must be bit-exactly zero (ADR-010 §3)",
                T::WIDTH,
            );
        }

        // Error case: half-spectrum output must have N/2 + 1 bins.
        let mut wrong = vec![zero; bins - 1];
        let mut input = vec![T::lit(c); n];
        assert!(
            plan.process(&mut input, &mut wrong).is_err(),
            "r2c [{}]: output-length mismatch must error",
            T::WIDTH,
        );
    }
}

/// Conformance for the complex-to-real DFT ([`DftC2r`](omnidsp_core::traits::dft::DftC2r)),
/// including the drift-tolerant DC/Nyquist contract (ADR-009 §4).
///
/// The shaped c2r (ADR-010 §2) projects its half-spectrum input onto the nearest
/// valid Hermitian spectrum before transforming, so a DC/Nyquist bin carrying
/// the ~1e-15 imaginary drift the r2c → multiply → c2r chain produces in
/// practice must neither error nor change the output.
///
/// # Panics
///
/// Panics if the drifted-input output differs from the clean-input output, if
/// the drifted input errors, or if an invalid call fails to error.
pub fn check_dft_c2r<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<DftC2rSpec<T>>,
    B::Plan: DftC2rPlan<T>,
{
    let n = 8;
    let bins = n / 2 + 1;
    let spec = DftC2rSpec::<T>::new(n, DftNorm::Inverse).expect("valid c2r spec");
    let plan = b.create_plan(&spec).expect("c2r plan");

    // A clean Hermitian half-spectrum: real DC and (even N) Nyquist.
    let clean: Vec<Complex<T>> = [
        (3.0, 0.0),
        (1.0, -2.0),
        (-0.5, 0.75),
        (2.0, 0.25),
        (1.5, 0.0),
    ]
    .iter()
    .map(|&(re, im)| Complex::new(T::lit(re), T::lit(im)))
    .collect();
    assert_eq!(clean.len(), bins, "drift case half-spectrum size");

    let mut clean_in = clean.clone();
    let mut out_clean = vec![T::zero(); n];
    plan.process(&mut clean_in, &mut out_clean)
        .expect("c2r clean process");

    // Inject the ~1e-15 DC/Nyquist imaginary drift the convolution chain leaks.
    let mut drift = clean;
    drift[0].im += T::lit(1e-15);
    drift[bins - 1].im += T::lit(1e-15);
    let mut out_drift = vec![T::zero(); n];
    let result = plan.process(&mut drift, &mut out_drift);
    assert!(
        result.is_ok(),
        "c2r [{}]: near-zero DC/Nyquist imaginary drift must not error (ADR-009 §4)",
        T::WIDTH,
    );
    assert_real_t(
        &out_drift,
        &out_clean,
        T::DFT_TOL,
        "c2r DC/Nyquist drift vs clean",
    );

    // Error case: input must have N/2 + 1 bins.
    let mut wrong = vec![Complex::new(T::zero(), T::zero()); bins + 1];
    let mut out = vec![T::zero(); n];
    assert!(
        plan.process(&mut wrong, &mut out).is_err(),
        "c2r [{}]: input-length mismatch must error",
        T::WIDTH,
    );
}

/// Conformance for the real-DFT round-trip identity: `c2r(r2c(x)) = x` under a
/// matching normalization.
///
/// Exercises both real primitives together over even and odd lengths and every
/// `DftNorm`, the way every real-input module (Conv/FIR/DCT/CrossCorr) composes
/// them.
///
/// # Panics
///
/// Panics if the recovered signal deviates from the (scaled) input beyond
/// [`ConformanceFloat::DFT_TOL`].
pub fn check_dft_real_round_trip<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<DftR2cSpec<T>> + CreatePlan<DftC2rSpec<T>>,
    <B as CreatePlan<DftR2cSpec<T>>>::Plan: DftR2cPlan<T>,
    <B as CreatePlan<DftC2rSpec<T>>>::Plan: DftC2rPlan<T>,
{
    let zero = Complex::new(T::zero(), T::zero());

    for &n in &[8usize, 9] {
        let bins = n / 2 + 1;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.9).cos() * 2.0 - 0.5).collect();

        for &norm in &NORMS {
            let fwd = b
                .create_plan(&DftR2cSpec::<T>::new(n, norm).expect("r2c spec"))
                .expect("r2c plan");
            let inv = b
                .create_plan(&DftC2rSpec::<T>::new(n, norm).expect("c2r spec"))
                .expect("c2r plan");

            let mut time = to_vec::<T>(&signal);
            let mut spectrum = vec![zero; bins];
            fwd.process(&mut time, &mut spectrum).expect("r2c process");
            let mut recovered = vec![T::zero(); n];
            inv.process(&mut spectrum, &mut recovered)
                .expect("c2r process");

            let scale = round_trip_scale(norm, n);
            let expected: Vec<f64> = signal.iter().map(|&x| x * scale).collect();
            assert_real(
                &recovered,
                &expected,
                T::DFT_TOL,
                &format!("real round-trip n={n} {norm:?}"),
            );
        }
    }
}
