// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Dynamic `VecOps` dispatch enum.

use num_complex::Complex;

use omnidsp_core::error::Result;
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_rust::RustVecOps;

/// Dispatch enum wrapping concrete `VecOps` implementations.
#[derive(Debug, Clone, Copy)]
pub enum DynVecOps {
    /// Pure Rust backend (scalar loops, LLVM auto-vectorized).
    Rust(RustVecOps),
}

macro_rules! impl_dyn_vecops {
    ($t:ty) => {
        impl VecOps<$t> for DynVecOps {
            fn mul(&self, a: &[$t], b: &[$t], out: &mut [$t]) -> Result<()> {
                match self {
                    Self::Rust(v) => v.mul(a, b, out),
                }
            }

            fn add(&self, a: &[$t], b: &[$t], out: &mut [$t]) -> Result<()> {
                match self {
                    Self::Rust(v) => v.add(a, b, out),
                }
            }

            fn scale(&self, data: &mut [$t], scalar: $t) {
                match self {
                    Self::Rust(v) => v.scale(data, scalar),
                }
            }

            fn dot(&self, a: &[$t], b: &[$t]) -> Result<$t> {
                match self {
                    Self::Rust(v) => v.dot(a, b),
                }
            }

            fn cmul(
                &self,
                a: &[Complex<$t>],
                b: &[Complex<$t>],
                out: &mut [Complex<$t>],
            ) -> Result<()> {
                match self {
                    Self::Rust(v) => v.cmul(a, b, out),
                }
            }

            fn mul_inplace(&self, data: &mut [$t], other: &[$t]) -> Result<()> {
                match self {
                    Self::Rust(v) => v.mul_inplace(data, other),
                }
            }

            fn add_inplace(&self, data: &mut [$t], other: &[$t]) -> Result<()> {
                match self {
                    Self::Rust(v) => v.add_inplace(data, other),
                }
            }

            fn cmul_inplace(&self, data: &mut [Complex<$t>], other: &[Complex<$t>]) -> Result<()> {
                match self {
                    Self::Rust(v) => v.cmul_inplace(data, other),
                }
            }
        }
    };
}

impl_dyn_vecops!(f32);
impl_dyn_vecops!(f64);

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
#[allow(clippy::float_cmp, reason = "test values are exact small integers")]
#[allow(clippy::suboptimal_flops, reason = "clarity over performance in tests")]
mod tests {
    use super::*;

    #[test]
    fn dyn_vecops_mul() {
        let ops = DynVecOps::Rust(RustVecOps);
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 4];
        VecOps::mul(&ops, &a, &b, &mut out).expect("mul should succeed");
        assert_eq!(out, [5.0, 12.0, 21.0, 32.0], "element-wise product");
    }

    #[test]
    fn dyn_vecops_dot() {
        let ops = DynVecOps::Rust(RustVecOps);
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let result = VecOps::dot(&ops, &a, &b).expect("dot should succeed");
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        assert!(
            (result - expected).abs() < 1e-12,
            "dot product: got {result}, expected {expected}"
        );
    }

    #[test]
    fn dyn_vecops_all_methods_f64() {
        let ops = DynVecOps::Rust(RustVecOps);
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut out = [0.0; 3];

        // mul
        VecOps::mul(&ops, &a, &b, &mut out).expect("mul");
        assert_eq!(out, [4.0, 10.0, 18.0], "mul result");

        // add
        VecOps::add(&ops, &a, &b, &mut out).expect("add");
        assert_eq!(out, [5.0, 7.0, 9.0], "add result");

        // scale
        let mut data = [2.0, 4.0, 6.0];
        VecOps::scale(&ops, &mut data, 0.5);
        assert_eq!(data, [1.0, 2.0, 3.0], "scale result");

        // dot
        let d = VecOps::dot(&ops, &a, &b).expect("dot");
        assert!((d - 32.0).abs() < 1e-12, "dot result: {d}");

        // cmul
        let ca = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let cb = [Complex::new(5.0_f64, 6.0), Complex::new(7.0, 8.0)];
        let mut cout = [Complex::<f64>::default(); 2];
        VecOps::cmul(&ops, &ca, &cb, &mut cout).expect("cmul");
        assert!(
            (cout[0].re - (1.0_f64 * 5.0 - 2.0 * 6.0)).abs() < 1e-12,
            "cmul[0].re"
        );

        // mul_inplace
        let mut mi = [1.0, 2.0, 3.0];
        VecOps::mul_inplace(&ops, &mut mi, &b).expect("mul_inplace");
        assert_eq!(mi, [4.0, 10.0, 18.0], "mul_inplace result");

        // add_inplace
        let mut ai = [1.0, 2.0, 3.0];
        VecOps::add_inplace(&ops, &mut ai, &b).expect("add_inplace");
        assert_eq!(ai, [5.0, 7.0, 9.0], "add_inplace result");

        // cmul_inplace
        let mut ci = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        VecOps::cmul_inplace(&ops, &mut ci, &cb).expect("cmul_inplace");
        assert!(
            (ci[0].re - (1.0_f64 * 5.0 - 2.0 * 6.0)).abs() < 1e-12,
            "cmul_inplace[0].re"
        );
    }
}
