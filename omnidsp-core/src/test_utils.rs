// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Test utilities: naive DFT and scalar `VecOps` for unit tests.
//!
//! Only compiled when `cfg(test)` is active.

use std::f64::consts::TAU;

use num_complex::Complex;

use crate::error::{Error, Result};
use crate::traits::dft::{Dft, DftNorm, DftPlan, DftSpec};
use crate::traits::vecops::VecOps;
use crate::types::Direction;

/// DFT factory for tests.
///
/// Uses Cooley-Tukey radix-2 FFT for power-of-2 lengths and falls
/// back to an O(N²) direct DFT for other lengths.  No external
/// dependencies — just textbook algorithms.
#[derive(Debug, Clone)]
pub struct TestDft;

/// DFT plan for tests.
///
/// Respects [`DftNorm`] so that module tests exercising different
/// normalization conventions get correct (or correctly wrong) results.
#[derive(Debug)]
pub struct TestDftPlan {
    length: usize,
    direction: Direction,
    norm: DftNorm,
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

        let sign = match self.direction {
            Direction::Forward => -1.0,
            Direction::Inverse => 1.0,
        };

        output.copy_from_slice(input);

        if self.length.is_power_of_two() {
            fft_radix2(output, sign);
        } else {
            dft_naive(input, output, sign);
        }

        if let Some(scale) = self.compute_scale() {
            for c in output.iter_mut() {
                *c = Complex::new(c.re * scale, c.im * scale);
            }
        }

        Ok(())
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
impl TestDftPlan {
    /// Compute the scaling factor for the configured normalization.
    fn compute_scale(&self) -> Option<f64> {
        let n = self.length as f64;
        match self.norm {
            DftNorm::None => None,
            DftNorm::Inverse => match self.direction {
                Direction::Inverse => Some(1.0 / n),
                Direction::Forward => None,
            },
            DftNorm::Ortho => Some(1.0 / n.sqrt()),
        }
    }
}

impl Dft<f64> for TestDft {
    type Plan = TestDftPlan;

    fn create_plan(&self, spec: &DftSpec<f64>) -> Result<Self::Plan> {
        if spec.length == 0 {
            return Err(Error::InvalidSpec("DFT length must be non-zero".to_owned()));
        }
        Ok(TestDftPlan {
            length: spec.length,
            direction: spec.direction,
            norm: spec.norm,
        })
    }
}

// ─── FFT internals ───────────────────────────────────────────────────

/// O(N²) direct DFT — fallback for non-power-of-2 lengths.
#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
fn dft_naive(input: &[Complex<f64>], output: &mut [Complex<f64>], sign: f64) {
    let n = input.len();
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = Complex::new(0.0, 0.0);
        for (j, &x) in input.iter().enumerate() {
            let angle = sign * TAU * (k * j) as f64 / n as f64;
            sum += x * Complex::new(angle.cos(), angle.sin());
        }
        *out = sum;
    }
}

/// In-place Cooley-Tukey radix-2 FFT.  `data.len()` must be a power of 2.
#[allow(
    clippy::cast_precision_loss,
    reason = "test DFT operates on small sizes where usize→f64 is exact"
)]
fn fft_radix2(data: &mut [Complex<f64>], sign: f64) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly passes.
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle_step = sign * TAU / size as f64;
        for start in (0..n).step_by(size) {
            for k in 0..half {
                let angle = angle_step * k as f64;
                let twiddle = Complex::new(angle.cos(), angle.sin());
                let u = data[start + k];
                let t = twiddle * data[start + k + half];
                data[start + k] = u + t;
                data[start + k + half] = u - t;
            }
        }
        size *= 2;
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
