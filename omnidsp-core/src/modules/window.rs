// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Window module — [`VecOps`]-accelerated window apply.
//!
//! [`OmniWindow`] is a lightweight factory that pairs a [`VecOps`]
//! implementation with window coefficients to produce [`OmniWindowPlan`]s.
//! Coefficient generation is handled by the design layer
//! ([`Window::from_fn`](crate::types::Window::from_fn)); this module just
//! applies the window via [`VecOps::mul_inplace`].

use num_traits::Float;

use crate::error::{Error, Result};
use crate::traits::vecops::VecOps;
use crate::types::{Window, WindowFn};

// ─── Public types ──────────────────────────────────────────────────────

/// Generic window factory backed by a [`VecOps`] implementation.
///
/// Creates [`OmniWindowPlan`]s from a [`WindowFn`] and length.
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
    window: Window<T>,
    vecops: V,
}

impl<T: Float, V: VecOps<T>> OmniWindowPlan<T, V> {
    /// Apply the window to `data` in-place via element-wise multiply.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferMismatch`] if `data.len()` does not match
    /// the window length.
    pub fn apply(&self, data: &mut [T]) -> Result<()> {
        if data.len() != self.window.len() {
            return Err(Error::BufferMismatch {
                expected: self.window.len(),
                actual: data.len(),
            });
        }
        self.vecops.mul_inplace(data, self.window.coefficients())
    }

    /// Access the precomputed window coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[T] {
        self.window.coefficients()
    }
}

impl<V> OmniWindow<V> {
    /// Create a plan from a [`WindowFn`] and length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if the length is zero or the window
    /// function parameters are invalid.
    pub fn create_plan<T: Float>(
        &self,
        winfn: &WindowFn<T>,
        length: usize,
    ) -> Result<OmniWindowPlan<T, V>>
    where
        V: VecOps<T>,
    {
        let window = Window::from_fn(winfn, length)?;
        Ok(OmniWindowPlan {
            window,
            vecops: self.vecops.clone(),
        })
    }

    /// Create a plan from pre-existing [`Window`] coefficients.
    pub fn create_plan_from_window<T: Float>(&self, window: Window<T>) -> OmniWindowPlan<T, V>
    where
        V: VecOps<T>,
    {
        OmniWindowPlan {
            window,
            vecops: self.vecops.clone(),
        }
    }
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

    // ── Factory + plan integration tests ───────────────────────────────

    #[test]
    fn apply_multiplies_by_coefficients() {
        let factory = OmniWindow::new(TestVecOps);
        let plan = factory
            .create_plan(&WindowFn::Hann, 5)
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
        let plan = factory
            .create_plan(&WindowFn::Hann, 5)
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
        let plan = factory
            .create_plan(&WindowFn::Hamming, 128)
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
        let result = factory.create_plan(&WindowFn::<f64>::Hann, 0);
        assert!(result.is_err(), "zero length plan should fail");
    }

    #[test]
    fn hann_5_coefficients() {
        let factory = OmniWindow::new(TestVecOps);
        let plan = factory
            .create_plan(&WindowFn::Hann, 5)
            .expect("plan creation should succeed");
        assert_approx_eq(plan.coefficients(), &[0.0, 0.5, 1.0, 0.5, 0.0], EPSILON);
    }

    #[test]
    fn from_window_uses_given_coefficients() {
        let factory = OmniWindow::new(TestVecOps);
        let window = Window::new(&[0.25, 0.5, 1.0, 0.5, 0.25]);
        let plan = factory.create_plan_from_window(window);

        assert_approx_eq(plan.coefficients(), &[0.25, 0.5, 1.0, 0.5, 0.25], EPSILON);

        let mut data = [2.0_f64; 5];
        plan.apply(&mut data).expect("apply should succeed");
        assert_approx_eq(&data, &[0.5, 1.0, 2.0, 1.0, 0.5], EPSILON);
    }
}
