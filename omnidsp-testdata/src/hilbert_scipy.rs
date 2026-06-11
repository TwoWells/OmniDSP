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

pub const HAND_N4_INPUT: &[f64] = &[1.0, 2.0, 3.0, 4.0];
pub const HAND_N4_EXPECTED: &[(f64, f64)] = &[(1.0, 1.0), (2.0, -1.0), (3.0, -1.0), (4.0, 1.0)];

// Hand-computed test case (N=8, even):
//   input = [1, 0, -1, 0, 1, 0, -1, 0]  (cosine at k=2, i.e. f=2 cycles/N)
//   This is cos(2π·2·n/8) = cos(πn/2)
//   Analytic signal = cos(πn/2) + j·sin(πn/2) = exp(jπn/2)
pub const HAND_N8_COS2_INPUT: &[f64] = &[1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
pub const HAND_N8_COS2_EXPECTED: &[(f64, f64)] = &[
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
];
