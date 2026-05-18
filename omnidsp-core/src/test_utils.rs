// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Test utilities: naive DFT and scalar `VecOps` for unit tests.
//!
//! Only compiled when `cfg(test)` is active.

use std::f64::consts::TAU;

use num_complex::Complex;

use crate::error::{Error, Result};
use crate::traits::dft::{Dft, DftPlan};
use crate::traits::vecops::VecOps;
use crate::types::Direction;

/// Naive O(N²) DFT factory for tests.
#[derive(Debug, Clone)]
pub struct TestDft;

/// Naive O(N²) DFT plan for tests.
#[derive(Debug)]
pub struct TestDftPlan {
    length: usize,
    direction: Direction,
}

#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
impl DftPlan<f64> for TestDftPlan {
    fn process(&self, input: &[Complex<f64>], output: &mut [Complex<f64>]) -> Result<()> {
        if input.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: input.len(),
            });
        }
        if output.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: output.len(),
            });
        }

        let n = self.length;
        let sign = match self.direction {
            Direction::Forward => -1.0,
            Direction::Inverse => 1.0,
        };

        for (k, out) in output.iter_mut().enumerate().take(n) {
            let mut sum = Complex::new(0.0, 0.0);
            for (j, &x) in input.iter().enumerate() {
                let angle = sign * TAU * (k * j) as f64 / n as f64;
                sum += x * Complex::new(angle.cos(), angle.sin());
            }
            *out = sum;
        }

        // Apply 1/N scaling for inverse transforms.
        if self.direction == Direction::Inverse {
            let scale = 1.0 / n as f64;
            for c in output.iter_mut() {
                *c = Complex::new(c.re * scale, c.im * scale);
            }
        }

        Ok(())
    }
}

impl Dft<f64> for TestDft {
    type Plan = TestDftPlan;

    fn create_plan(&self, length: usize, direction: Direction) -> Result<Self::Plan> {
        if length == 0 {
            return Err(Error::InvalidSpec("DFT length must be non-zero".to_owned()));
        }
        Ok(TestDftPlan { length, direction })
    }
}

/// Scalar loop `VecOps` for tests.
#[derive(Debug, Clone)]
pub struct TestVecOps;

impl VecOps<f64> for TestVecOps {
    fn mul(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<()> {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
        Ok(())
    }

    fn add(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<()> {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
        Ok(())
    }

    fn scale(&self, data: &mut [f64], scalar: f64) {
        for x in data.iter_mut() {
            *x *= scalar;
        }
    }

    fn dot(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
    }

    fn cmul(&self, a: &[Complex<f64>], b: &[Complex<f64>], out: &mut [Complex<f64>]) -> Result<()> {
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
        Ok(())
    }

    fn mul_inplace(&self, data: &mut [f64], other: &[f64]) -> Result<()> {
        for (x, &y) in data.iter_mut().zip(other) {
            *x *= y;
        }
        Ok(())
    }

    fn add_inplace(&self, data: &mut [f64], other: &[f64]) -> Result<()> {
        for (x, &y) in data.iter_mut().zip(other) {
            *x += y;
        }
        Ok(())
    }

    fn cmul_inplace(&self, data: &mut [Complex<f64>], other: &[Complex<f64>]) -> Result<()> {
        for (x, y) in data.iter_mut().zip(other) {
            *x *= y;
        }
        Ok(())
    }
}
