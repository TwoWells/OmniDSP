// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Dynamic DFT dispatch enum.

use std::fmt;

use num_complex::Complex;
use num_traits::FromPrimitive;

use omnidsp_core::error::Result;
use omnidsp_core::traits::dft::{Dft, DftPlan, DftSpec};
use omnidsp_rust::{RustDft, RustDftPlan};

/// Dispatch enum wrapping concrete DFT factories.
#[derive(Debug, Clone, Copy)]
pub enum DynDft {
    /// Pure Rust backend (`RustFFT`).
    Rust(RustDft),
}

/// Dispatch enum wrapping concrete DFT plans.
pub enum DynDftPlan<T> {
    /// Pure Rust backend plan.
    Rust(RustDftPlan<T>),
}

impl<T> fmt::Debug for DynDftPlan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rust(_) => write!(f, "DynDftPlan::Rust(..)"),
        }
    }
}

macro_rules! impl_dyn_dft {
    ($t:ty) => {
        impl Dft<$t> for DynDft {
            type Plan = DynDftPlan<$t>;

            fn create_plan(&self, spec: &DftSpec<$t>) -> Result<DynDftPlan<$t>> {
                match self {
                    Self::Rust(d) => Ok(DynDftPlan::Rust(d.create_plan(spec)?)),
                }
            }
        }
    };
}

impl_dyn_dft!(f32);
impl_dyn_dft!(f64);

impl<T> DftPlan<T> for DynDftPlan<T>
where
    T: Copy + FromPrimitive + std::ops::Mul<Output = T> + Send + Sync + 'static,
{
    fn process(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> Result<()> {
        match self {
            Self::Rust(p) => p.process(input, output),
        }
    }
}

const _: () = {
    const fn _assert<T: Send + Sync>() {}
    let _ = _assert::<DynDftPlan<f32>>;
    let _ = _assert::<DynDftPlan<f64>>;
};

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;
    use omnidsp_core::traits::dft::DftNorm;
    use omnidsp_core::types::Direction;

    const EPSILON_F64: f64 = 1e-12;
    const EPSILON_F32: f32 = 1e-5;

    fn assert_complex_slice_eq<T: num_traits::Float + fmt::Display>(
        a: &[Complex<T>],
        b: &[Complex<T>],
        eps: T,
    ) {
        assert_eq!(a.len(), b.len(), "slice lengths differ");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x.re - y.re).abs() < eps && (x.im - y.im).abs() < eps,
                "mismatch at index {i}: got ({}, {}), expected ({}, {})",
                x.re,
                x.im,
                y.re,
                y.im,
            );
        }
    }

    #[test]
    fn dyn_dft_round_trip_f64() {
        let dft = DynDft::Rust(RustDft);
        let fwd =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(8, Direction::Forward, DftNorm::Inverse))
                .expect("forward plan");
        let inv =
            Dft::<f64>::create_plan(&dft, &DftSpec::new(8, Direction::Inverse, DftNorm::Inverse))
                .expect("inverse plan");

        let input: Vec<Complex<f64>> = (0..8)
            .map(|i| Complex::new(f64::from(i), -f64::from(i)))
            .collect();
        let mut freq = vec![Complex::default(); 8];
        let mut recovered = vec![Complex::default(); 8];

        fwd.process(&input, &mut freq).expect("forward");
        inv.process(&freq, &mut recovered).expect("inverse");

        assert_complex_slice_eq(&recovered, &input, EPSILON_F64);
    }

    #[test]
    fn dyn_dft_round_trip_f32() {
        let dft = DynDft::Rust(RustDft);
        let fwd =
            Dft::<f32>::create_plan(&dft, &DftSpec::new(4, Direction::Forward, DftNorm::Inverse))
                .expect("forward plan");
        let inv =
            Dft::<f32>::create_plan(&dft, &DftSpec::new(4, Direction::Inverse, DftNorm::Inverse))
                .expect("inverse plan");

        let input = [
            Complex::new(1.0_f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut freq = [Complex::default(); 4];
        let mut recovered = [Complex::default(); 4];

        fwd.process(&input, &mut freq).expect("forward");
        inv.process(&freq, &mut recovered).expect("inverse");

        assert_complex_slice_eq(&recovered, &input, EPSILON_F32);
    }

    #[test]
    fn dyn_dft_plan_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DynDftPlan<f64>>();
    }
}
