// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Window module — coefficient generation and [`VecOps`]-accelerated apply.
//!
//! [`OmniWindow`] implements the [`Window`](crate::traits::window::Window)
//! trait generically over any [`VecOps`] implementation.  Coefficient
//! generation covers all [`WindowType`] variants; the apply step uses
//! [`VecOps::mul_inplace`] for SIMD-accelerated element-wise multiply.

use num_traits::{Float, FloatConst};

use crate::error::{Error, Result};
use crate::traits::vecops::VecOps;
use crate::traits::window::{Window, WindowPlan};
use crate::types::WindowType;

// ─── Public types ──────────────────────────────────────────────────────

/// Generic window factory backed by a [`VecOps`] implementation.
///
/// Creates [`OmniWindowPlan`]s from [`WindowType`] specifications.
/// The factory itself is lightweight — all state lives in the plan.
#[derive(Debug, Clone)]
pub struct OmniWindow<V> {
    vecops: V,
}

impl<V> OmniWindow<V> {
    /// Create a new window factory backed by the given [`VecOps`].
    #[must_use]
    pub const fn new(vecops: V) -> Self {
        Self { vecops }
    }
}

/// Execution plan for a window operation.
///
/// Holds precomputed window coefficients and a [`VecOps`] handle for
/// accelerated apply.  Immutable and thread-safe (`Send + Sync`).
#[derive(Debug, Clone)]
pub struct OmniWindowPlan<T, V> {
    coefficients: Vec<T>,
    vecops: V,
}

// ─── Trait implementations ─────────────────────────────────────────────

impl<T, V> WindowPlan<T> for OmniWindowPlan<T, V>
where
    T: Send + Sync,
    V: VecOps<T>,
{
    fn apply(&self, data: &mut [T]) -> Result<()> {
        if data.len() != self.coefficients.len() {
            return Err(Error::BufferMismatch {
                expected: self.coefficients.len(),
                actual: data.len(),
            });
        }
        self.vecops.mul_inplace(data, &self.coefficients)
    }

    fn coefficients(&self) -> &[T] {
        &self.coefficients
    }
}

impl<T, V> Window<T> for OmniWindow<V>
where
    T: Float + FloatConst + Send + Sync,
    V: VecOps<T>,
{
    type Plan = OmniWindowPlan<T, V>;

    fn create_plan(&self, window_type: WindowType<T>) -> Result<Self::Plan> {
        let coefficients = generate(window_type)?;
        Ok(OmniWindowPlan {
            coefficients,
            vecops: self.vecops.clone(),
        })
    }
}

// ─── Coefficient generation (private) ──────────────────────────────────

/// Validate that a window length is non-zero.
fn validate_length(length: usize) -> Result<()> {
    if length == 0 {
        return Err(Error::InvalidSpec(
            "window length must be non-zero".to_owned(),
        ));
    }
    Ok(())
}

/// Cast a `usize` to the target float type.
fn cast_usize<T: Float>(n: usize) -> Result<T> {
    T::from(n).ok_or_else(|| Error::Internal(format!("usize {n} not representable as float")))
}

/// Cast an `f64` constant to the target float type.
fn cast_f64<T: Float>(x: f64) -> Result<T> {
    T::from(x).ok_or_else(|| Error::Internal(format!("f64 {x} not representable as target type")))
}

/// Generate window coefficients from a [`WindowType`] specification.
fn generate<T: Float + FloatConst>(window_type: WindowType<T>) -> Result<Vec<T>> {
    match window_type {
        WindowType::Bartlett { length } => bartlett(length),
        WindowType::Blackman { length } => {
            validate_length(length)?;
            cosine_sum(length, &[cast_f64(0.42)?, cast_f64(0.5)?, cast_f64(0.08)?])
        }
        WindowType::Custom(coefficients) => {
            if coefficients.is_empty() {
                return Err(Error::InvalidSpec(
                    "custom window must not be empty".to_owned(),
                ));
            }
            Ok(coefficients)
        }
        WindowType::FlatTop { length } => {
            validate_length(length)?;
            cosine_sum(
                length,
                &[
                    cast_f64(0.215_578_95)?,
                    cast_f64(0.416_631_58)?,
                    cast_f64(0.277_263_158)?,
                    cast_f64(0.083_578_947)?,
                    cast_f64(0.006_947_368)?,
                ],
            )
        }
        WindowType::Gaussian { length, sigma } => gaussian(length, sigma),
        WindowType::Hamming { length } => {
            validate_length(length)?;
            cosine_sum(length, &[cast_f64(0.54)?, cast_f64(0.46)?])
        }
        WindowType::Hann { length } => {
            validate_length(length)?;
            let half = T::one() / (T::one() + T::one());
            cosine_sum(length, &[half, half])
        }
        WindowType::Kaiser { length, beta } => kaiser(length, beta),
        WindowType::Rectangular { length } => {
            validate_length(length)?;
            Ok(vec![T::one(); length])
        }
        WindowType::Triangular { length } => triangular(length),
        WindowType::Tukey { length, alpha } => tukey(length, alpha),
    }
}

/// Generalized cosine-sum window: `w[n] = Σ (-1)^k · a[k] · cos(2πkn/(N-1))`.
fn cosine_sum<T: Float + FloatConst>(length: usize, a: &[T]) -> Result<Vec<T>> {
    if length == 1 {
        return Ok(vec![T::one()]);
    }
    let n_minus_1 = cast_usize::<T>(length - 1)?;
    let tau = T::TAU();
    let mut result = Vec::with_capacity(length);
    for i in 0..length {
        let x = cast_usize::<T>(i)? / n_minus_1;
        let mut w = a[0];
        let mut sign = -T::one();
        let mut k = T::one();
        for &ak in &a[1..] {
            w = w + sign * ak * (k * tau * x).cos();
            sign = -sign;
            k = k + T::one();
        }
        result.push(w);
    }
    Ok(result)
}

/// Bartlett (triangular with zero endpoints) window.
fn bartlett<T: Float>(length: usize) -> Result<Vec<T>> {
    validate_length(length)?;
    if length == 1 {
        return Ok(vec![T::one()]);
    }
    let n_minus_1 = cast_usize::<T>(length - 1)?;
    let two = T::one() + T::one();
    let mut result = Vec::with_capacity(length);
    for i in 0..length {
        let w = T::one() - (two * cast_usize::<T>(i)? / n_minus_1 - T::one()).abs();
        result.push(w);
    }
    Ok(result)
}

/// Triangular (non-zero endpoints) window.
///
/// Matches `scipy.signal.windows.triang`: endpoints are non-zero, peak is 1.0.
fn triangular<T: Float>(length: usize) -> Result<Vec<T>> {
    validate_length(length)?;
    if length == 1 {
        return Ok(vec![T::one()]);
    }
    let two = T::one() + T::one();
    let mut result = Vec::with_capacity(length);
    if length.is_multiple_of(2) {
        // Even N: w[n] = (2*(idx+1) - 1) / N, symmetric about center.
        let n = cast_usize::<T>(length)?;
        for i in 0..length {
            let idx = if i < length / 2 { i } else { length - 1 - i };
            let w = (two * cast_usize::<T>(idx + 1)? - T::one()) / n;
            result.push(w);
        }
    } else {
        // Odd N: w[n] = 2*(idx+1) / (N+1), symmetric about center.
        let n_plus_1 = cast_usize::<T>(length + 1)?;
        for i in 0..length {
            let idx = if i <= length / 2 { i } else { length - 1 - i };
            let w = two * cast_usize::<T>(idx + 1)? / n_plus_1;
            result.push(w);
        }
    }
    Ok(result)
}

/// Gaussian window.
///
/// `sigma` is relative to half the window length: `std = sigma * (N-1)/2`.
fn gaussian<T: Float>(length: usize, sigma: T) -> Result<Vec<T>> {
    validate_length(length)?;
    if length == 1 {
        return Ok(vec![T::one()]);
    }
    let two = T::one() + T::one();
    let half_len = cast_usize::<T>(length - 1)? / two;
    let std_dev = sigma * half_len;
    let two_var = two * std_dev * std_dev;
    let mut result = Vec::with_capacity(length);
    for i in 0..length {
        let n = cast_usize::<T>(i)? - half_len;
        let w = (-(n * n) / two_var).exp();
        result.push(w);
    }
    Ok(result)
}

/// Kaiser window using the modified Bessel function I₀.
fn kaiser<T: Float>(length: usize, beta: T) -> Result<Vec<T>> {
    validate_length(length)?;
    if length == 1 {
        return Ok(vec![T::one()]);
    }
    let two = T::one() + T::one();
    let alpha = cast_usize::<T>(length - 1)? / two;
    let denom = bessel_i0(beta);
    let mut result = Vec::with_capacity(length);
    for i in 0..length {
        let x = (cast_usize::<T>(i)? - alpha) / alpha;
        // Clamp to avoid sqrt of tiny negative due to floating-point rounding.
        let inner = (T::one() - x * x).max(T::zero());
        let w = bessel_i0(beta * inner.sqrt()) / denom;
        result.push(w);
    }
    Ok(result)
}

/// Modified Bessel function of the first kind, order 0.
///
/// Series expansion: `I₀(x) = Σ_{k=0}^{∞} ((x/2)^k / k!)²`
fn bessel_i0<T: Float>(x: T) -> T {
    let two = T::one() + T::one();
    let half_x = x / two;
    let half_x_sq = half_x * half_x;
    let mut sum = T::one();
    let mut term = T::one();
    let mut k = T::one();
    for _ in 0..100 {
        let k_sq = k * k;
        term = term * half_x_sq / k_sq;
        let new_sum = sum + term;
        if (new_sum - sum).abs() <= T::epsilon() * sum.abs() {
            break;
        }
        sum = new_sum;
        k = k + T::one();
    }
    sum
}

/// Tukey (tapered cosine) window.
fn tukey<T: Float + FloatConst>(length: usize, alpha: T) -> Result<Vec<T>> {
    validate_length(length)?;
    if length == 1 {
        return Ok(vec![T::one()]);
    }
    // alpha <= 0 degenerates to rectangular.
    if alpha <= T::zero() {
        return Ok(vec![T::one(); length]);
    }
    // alpha >= 1 degenerates to Hann.
    if alpha >= T::one() {
        let half = T::one() / (T::one() + T::one());
        return cosine_sum(length, &[half, half]);
    }
    let two = T::one() + T::one();
    let tau = T::TAU();
    let half_alpha = alpha / two;
    let n_minus_1 = cast_usize::<T>(length - 1)?;
    let mut result = Vec::with_capacity(length);
    for i in 0..length {
        let x = cast_usize::<T>(i)? / n_minus_1;
        let w = if x < half_alpha {
            (T::one() + (tau / alpha * (x - half_alpha)).cos()) / two
        } else if x > T::one() - half_alpha {
            (T::one() + (tau / alpha * (x - T::one() + half_alpha)).cos()) / two
        } else {
            T::one()
        };
        result.push(w);
    }
    Ok(result)
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;
    use crate::test_utils::TestVecOps;

    const EPSILON: f64 = 1e-10;

    fn assert_approx_eq(actual: &[f64], expected: &[f64], eps: f64) {
        assert_eq!(actual.len(), expected.len(), "slice lengths differ");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < eps,
                "mismatch at index {i}: got {a}, expected {e}"
            );
        }
    }

    // ── Coefficient generation tests ───────────────────────────────────

    #[test]
    fn hann_5() {
        let coeffs: Vec<f64> =
            generate(WindowType::Hann { length: 5 }).expect("generation should succeed");
        assert_approx_eq(&coeffs, &[0.0, 0.5, 1.0, 0.5, 0.0], EPSILON);
    }

    #[test]
    fn hamming_5() {
        let coeffs: Vec<f64> =
            generate(WindowType::Hamming { length: 5 }).expect("generation should succeed");
        assert_approx_eq(&coeffs, &[0.08, 0.54, 1.0, 0.54, 0.08], EPSILON);
    }

    #[test]
    fn blackman_5() {
        let coeffs: Vec<f64> =
            generate(WindowType::Blackman { length: 5 }).expect("generation should succeed");
        // 0.42 - 0.5*cos(0) + 0.08*cos(0) = 0.0 at endpoints
        // 0.42 - 0.5*cos(π/2) + 0.08*cos(π) = 0.34 at n=1,3
        assert_approx_eq(&coeffs, &[0.0, 0.34, 1.0, 0.34, 0.0], 1e-8);
    }

    #[test]
    fn rectangular_4() {
        let coeffs: Vec<f64> =
            generate(WindowType::Rectangular { length: 4 }).expect("generation should succeed");
        assert_approx_eq(&coeffs, &[1.0, 1.0, 1.0, 1.0], EPSILON);
    }

    #[test]
    fn bartlett_5() {
        let coeffs: Vec<f64> =
            generate(WindowType::Bartlett { length: 5 }).expect("generation should succeed");
        assert_approx_eq(&coeffs, &[0.0, 0.5, 1.0, 0.5, 0.0], EPSILON);
    }

    #[test]
    fn triangular_odd_5() {
        let coeffs: Vec<f64> =
            generate(WindowType::Triangular { length: 5 }).expect("generation should succeed");
        let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0];
        assert_approx_eq(&coeffs, &expected, EPSILON);
    }

    #[test]
    fn triangular_even_4() {
        let coeffs: Vec<f64> =
            generate(WindowType::Triangular { length: 4 }).expect("generation should succeed");
        assert_approx_eq(&coeffs, &[0.25, 0.75, 0.75, 0.25], EPSILON);
    }

    #[test]
    fn gaussian_5_sigma_1() {
        let coeffs: Vec<f64> = generate(WindowType::Gaussian {
            length: 5,
            sigma: 1.0,
        })
        .expect("generation should succeed");
        // std = 1.0 * (5-1)/2 = 2.0, w[n] = exp(-(n-2)^2 / 8)
        let expected = [
            (-0.5_f64).exp(),
            (-0.125_f64).exp(),
            1.0,
            (-0.125_f64).exp(),
            (-0.5_f64).exp(),
        ];
        assert_approx_eq(&coeffs, &expected, EPSILON);
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        let coeffs: Vec<f64> = generate(WindowType::Kaiser {
            length: 5,
            beta: 0.0,
        })
        .expect("generation should succeed");
        assert_approx_eq(&coeffs, &[1.0, 1.0, 1.0, 1.0, 1.0], EPSILON);
    }

    #[test]
    fn kaiser_symmetric() {
        let coeffs: Vec<f64> = generate(WindowType::Kaiser {
            length: 7,
            beta: 5.0,
        })
        .expect("generation should succeed");
        // Kaiser windows are symmetric
        assert_eq!(coeffs.len(), 7, "length should be 7");
        assert!(
            (coeffs[0] - coeffs[6]).abs() < EPSILON,
            "endpoints should be equal"
        );
        assert!(
            (coeffs[1] - coeffs[5]).abs() < EPSILON,
            "symmetric points should be equal"
        );
        assert!(
            (coeffs[2] - coeffs[4]).abs() < EPSILON,
            "symmetric points should be equal"
        );
        // Center should be the maximum (1.0 for Kaiser).
        assert!(
            (coeffs[3] - 1.0).abs() < EPSILON,
            "center should be 1.0, got {}",
            coeffs[3]
        );
    }

    #[test]
    fn tukey_alpha_zero_is_rectangular() {
        let coeffs: Vec<f64> = generate(WindowType::Tukey {
            length: 5,
            alpha: 0.0,
        })
        .expect("generation should succeed");
        assert_approx_eq(&coeffs, &[1.0, 1.0, 1.0, 1.0, 1.0], EPSILON);
    }

    #[test]
    fn tukey_alpha_one_is_hann() {
        let hann: Vec<f64> = generate(WindowType::Hann { length: 5 }).expect("hann should succeed");
        let tukey: Vec<f64> = generate(WindowType::Tukey {
            length: 5,
            alpha: 1.0,
        })
        .expect("tukey should succeed");
        assert_approx_eq(&tukey, &hann, EPSILON);
    }

    #[test]
    fn tukey_7_half() {
        let coeffs: Vec<f64> = generate(WindowType::Tukey {
            length: 7,
            alpha: 0.5,
        })
        .expect("generation should succeed");
        assert_approx_eq(&coeffs, &[0.0, 0.75, 1.0, 1.0, 1.0, 0.75, 0.0], EPSILON);
    }

    #[test]
    fn flat_top_5_center_is_one() {
        let coeffs: Vec<f64> =
            generate(WindowType::FlatTop { length: 5 }).expect("generation should succeed");
        assert_eq!(coeffs.len(), 5, "length should be 5");
        // Center value should be sum of all cosine coefficients ≈ 1.0
        assert!(
            (coeffs[2] - 1.0).abs() < 1e-6,
            "flat-top center should be ≈1.0, got {}",
            coeffs[2]
        );
        // Symmetric
        assert!(
            (coeffs[0] - coeffs[4]).abs() < EPSILON,
            "endpoints should be equal"
        );
        assert!(
            (coeffs[1] - coeffs[3]).abs() < EPSILON,
            "symmetric points should be equal"
        );
    }

    #[test]
    fn custom_passthrough() {
        let custom = vec![0.1, 0.2, 0.3, 0.4];
        let coeffs: Vec<f64> =
            generate(WindowType::Custom(custom.clone())).expect("generation should succeed");
        assert_approx_eq(&coeffs, &custom, EPSILON);
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn length_one_returns_one() {
        fn check(name: &str, wt: WindowType<f64>) {
            let coeffs = generate(wt).expect(name);
            assert_eq!(coeffs.len(), 1, "{name} should have length 1");
            assert!(
                (coeffs[0] - 1.0).abs() < EPSILON,
                "{name} coefficient should be 1.0, got {}",
                coeffs[0]
            );
        }

        check("hann", WindowType::Hann { length: 1 });
        check("hamming", WindowType::Hamming { length: 1 });
        check("blackman", WindowType::Blackman { length: 1 });
        check("rectangular", WindowType::Rectangular { length: 1 });
        check("bartlett", WindowType::Bartlett { length: 1 });
        check("triangular", WindowType::Triangular { length: 1 });
        check("flat_top", WindowType::FlatTop { length: 1 });
        check(
            "gaussian",
            WindowType::Gaussian {
                length: 1,
                sigma: 1.0,
            },
        );
        check(
            "kaiser",
            WindowType::Kaiser {
                length: 1,
                beta: 5.0,
            },
        );
        check(
            "tukey",
            WindowType::Tukey {
                length: 1,
                alpha: 0.5,
            },
        );
    }

    #[test]
    fn zero_length_returns_error() {
        let result: std::result::Result<Vec<f64>, _> = generate(WindowType::Hann { length: 0 });
        assert!(result.is_err(), "zero length should return error");
    }

    #[test]
    fn empty_custom_returns_error() {
        let result: std::result::Result<Vec<f64>, _> = generate(WindowType::Custom(vec![]));
        assert!(result.is_err(), "empty custom should return error");
    }

    // ── Factory + plan integration tests ───────────────────────────────

    #[test]
    fn apply_multiplies_by_coefficients() {
        let factory = OmniWindow::new(TestVecOps);
        let plan = Window::<f64>::create_plan(&factory, WindowType::Hann { length: 5 })
            .expect("plan creation should succeed");

        let mut data = [2.0_f64; 5];
        plan.apply(&mut data).expect("apply should succeed");

        let coeffs = plan.coefficients();
        for (i, (&d, &c)) in data.iter().zip(coeffs).enumerate() {
            let expected = c + c;
            assert!(
                (d - expected).abs() < EPSILON,
                "apply mismatch at index {i}: got {d}, expected {expected}",
            );
        }
    }

    #[test]
    fn apply_wrong_length_returns_error() {
        let factory = OmniWindow::new(TestVecOps);
        let plan = Window::<f64>::create_plan(&factory, WindowType::Hann { length: 5 })
            .expect("plan creation should succeed");

        let mut data = [1.0_f64; 3];
        assert!(
            plan.apply(&mut data).is_err(),
            "apply with wrong buffer length should return error"
        );
    }

    #[test]
    fn coefficients_length_matches_window() {
        let factory = OmniWindow::new(TestVecOps);
        let plan = Window::<f64>::create_plan(&factory, WindowType::Hamming { length: 128 })
            .expect("plan creation should succeed");

        assert_eq!(
            plan.coefficients().len(),
            128,
            "coefficients length should match window length"
        );
    }

    #[test]
    fn factory_create_plan_zero_length_error() {
        let factory = OmniWindow::new(TestVecOps);
        let result = Window::<f64>::create_plan(&factory, WindowType::Hann { length: 0 });
        assert!(result.is_err(), "zero length plan should fail");
    }
}
