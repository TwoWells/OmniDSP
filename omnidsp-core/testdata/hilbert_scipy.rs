// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Reference data for Hilbert transform tests.
// Regenerate with: make gen-hilbert-reference
//
// Hand-computed test case (N=4, even):
//   input = [1, 2, 3, 4]
//   FFT = [10, -2+2j, -2, -2-2j]
//   mask = [1, 2, 1, 0]
//   masked = [10, -4+4j, -2, 0]
//   IFFT = [1+j, 2-j, 3-j, 4+j]

const HAND_N4_INPUT: &[f64] = &[1.0, 2.0, 3.0, 4.0];
const HAND_N4_EXPECTED: &[(f64, f64)] = &[
    (1.0, 1.0),
    (2.0, -1.0),
    (3.0, -1.0),
    (4.0, 1.0),
];

// Hand-computed test case (N=8, even):
//   input = [1, 0, -1, 0, 1, 0, -1, 0]  (cosine at k=2, i.e. f=2 cycles/N)
//   This is cos(2π·2·n/8) = cos(πn/2)
//   Analytic signal = cos(πn/2) + j·sin(πn/2) = exp(jπn/2)
const HAND_N8_COS2_INPUT: &[f64] = &[1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
const HAND_N8_COS2_EXPECTED: &[(f64, f64)] = &[
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
];

#[test]
fn hand_computed_n4() {
    let factory = make_factory();
    let spec = HilbertSpec::<f64>::new(HAND_N4_INPUT.len()).expect("valid hilbert spec");
    let plan = factory.create_plan(&spec).expect("plan creation should succeed");

    let mut output = vec![Complex::new(0.0, 0.0); HAND_N4_INPUT.len()];
    plan.process(HAND_N4_INPUT, &mut output).expect("process should succeed");

    let tol = 1e-12;
    for (i, (z, &(re, im))) in output.iter().zip(HAND_N4_EXPECTED).enumerate() {
        assert!(
            (z.re - re).abs() < tol,
            "n4 real mismatch at {i}: got {}, expected {re}",
            z.re
        );
        assert!(
            (z.im - im).abs() < tol,
            "n4 imag mismatch at {i}: got {}, expected {im}",
            z.im
        );
    }
}

#[test]
fn hand_computed_n8_cos2() {
    let factory = make_factory();
    let spec = HilbertSpec::<f64>::new(HAND_N8_COS2_INPUT.len()).expect("valid hilbert spec");
    let plan = factory.create_plan(&spec).expect("plan creation should succeed");

    let mut output = vec![Complex::new(0.0, 0.0); HAND_N8_COS2_INPUT.len()];
    plan.process(HAND_N8_COS2_INPUT, &mut output).expect("process should succeed");

    let tol = 1e-12;
    for (i, (z, &(re, im))) in output.iter().zip(HAND_N8_COS2_EXPECTED).enumerate() {
        assert!(
            (z.re - re).abs() < tol,
            "n8_cos2 real mismatch at {i}: got {}, expected {re}",
            z.re
        );
        assert!(
            (z.im - im).abs() < tol,
            "n8_cos2 imag mismatch at {i}: got {}, expected {im}",
            z.im
        );
    }
}
