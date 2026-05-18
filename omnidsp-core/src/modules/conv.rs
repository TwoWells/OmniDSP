// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Convolution module — FFT-based fast convolution.
//!
//! [`OmniConv`] implements the [`Conv`](crate::traits::conv::Conv) trait
//! generically over any [`Dft`] and [`VecOps`] implementation.  Uses the
//! standard FFT convolution algorithm: forward-FFT both inputs, complex
//! element-wise multiply, inverse FFT.
//!
//! Internal scratch buffers are behind a [`Mutex`] so that the plan
//! satisfies `Send + Sync` while taking `&self`.  The lock is uncontended
//! in the common single-threaded case.

use std::fmt;
use std::sync::Mutex;

use num_complex::Complex;
use num_traits::Float;

use crate::error::{Error, Result};
use crate::traits::conv::{Conv, ConvPlan};
use crate::traits::dft::{Dft, DftPlan};
use crate::traits::vecops::VecOps;
use crate::types::Direction;

// ─── Public types ──────────────────────────────────────────────────────

/// Generic convolution factory backed by [`Dft`] and [`VecOps`].
///
/// Creates [`OmniConvPlan`]s for specific input lengths.  The factory
/// owns the DFT factory and `VecOps` instance; plans own their sub-plans.
#[derive(Debug, Clone)]
pub struct OmniConv<D, V> {
    dft: D,
    vecops: V,
}

impl<D, V> OmniConv<D, V> {
    /// Create a new convolution factory.
    #[must_use]
    pub const fn new(dft: D, vecops: V) -> Self {
        Self { dft, vecops }
    }
}

/// Execution plan for a convolution operation.
///
/// Holds forward and inverse FFT plans plus scratch buffers behind a
/// [`Mutex`].  The plan is immutable and thread-safe (`Send + Sync`).
///
/// Output length is `a_len + b_len - 1` (full convolution).
pub struct OmniConvPlan<T, P, V> {
    fwd: P,
    inv: P,
    vecops: V,
    a_len: usize,
    b_len: usize,
    output_len: usize,
    scratch: Mutex<ConvScratch<T>>,
}

impl<T, P, V> fmt::Debug for OmniConvPlan<T, P, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniConvPlan")
            .field("a_len", &self.a_len)
            .field("b_len", &self.b_len)
            .field("output_len", &self.output_len)
            .finish_non_exhaustive()
    }
}

// ─── Private helpers ───────────────────────────────────────────────────

/// Scratch buffers for the FFT convolution pipeline.
#[allow(
    clippy::struct_field_names,
    reason = "buf_ prefix clarifies these are reusable buffers"
)]
struct ConvScratch<T> {
    /// Padding / FFT input / IFFT output.
    buf_a: Vec<Complex<T>>,
    /// FFT output for first input → holds frequency-domain product.
    buf_b: Vec<Complex<T>>,
    /// FFT output for second input.
    buf_c: Vec<Complex<T>>,
}

/// Zero-pad a real slice into a pre-allocated complex buffer.
///
/// First `real.len()` elements are set to `real[i] + 0i`; the remainder
/// is zeroed.
fn pad_real_to_complex<T: Float>(real: &[T], buf: &mut [Complex<T>]) {
    for (c, &r) in buf.iter_mut().zip(real) {
        *c = Complex::new(r, T::zero());
    }
    for c in &mut buf[real.len()..] {
        *c = Complex::new(T::zero(), T::zero());
    }
}

// ─── Trait implementations ─────────────────────────────────────────────

impl<T, P, V> ConvPlan<T> for OmniConvPlan<T, P, V>
where
    T: Float + Send + Sync,
    P: DftPlan<T>,
    V: VecOps<T>,
{
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire FFT pipeline"
    )]
    fn process(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        if a.len() != self.a_len {
            return Err(Error::BufferMismatch {
                expected: self.a_len,
                actual: a.len(),
            });
        }
        if b.len() != self.b_len {
            return Err(Error::BufferMismatch {
                expected: self.b_len,
                actual: b.len(),
            });
        }
        if output.len() != self.output_len {
            return Err(Error::BufferMismatch {
                expected: self.output_len,
                actual: output.len(),
            });
        }

        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;

        // Split-borrow the scratch buffers so we can pass disjoint
        // mutable/shared references to DFT and VecOps calls.
        let ConvScratch {
            buf_a,
            buf_b,
            buf_c,
        } = &mut *guard;

        // 1. Zero-pad a → buf_a, then forward FFT → buf_b.
        pad_real_to_complex(a, buf_a);
        self.fwd.process(buf_a, buf_b)?;

        // 2. Zero-pad b → buf_a (reuse), then forward FFT → buf_c.
        pad_real_to_complex(b, buf_a);
        self.fwd.process(buf_a, buf_c)?;

        // 3. Complex element-wise multiply: buf_b *= buf_c.
        self.vecops.cmul_inplace(buf_b, buf_c)?;

        // 4. Inverse FFT: buf_b → buf_a.
        self.inv.process(buf_b, buf_a)?;

        // 5. Extract real parts of the first output_len samples.
        for (o, c) in output.iter_mut().zip(&buf_a[..self.output_len]) {
            *o = c.re;
        }

        Ok(())
    }
}

impl<T, D, V> Conv<T> for OmniConv<D, V>
where
    T: Float + Send + Sync,
    D: Dft<T>,
    V: VecOps<T>,
{
    type Plan = OmniConvPlan<T, D::Plan, V>;

    fn create_plan(&self, a_len: usize, b_len: usize) -> Result<Self::Plan> {
        if a_len == 0 {
            return Err(Error::InvalidSpec(
                "convolution input a_len must be non-zero".to_owned(),
            ));
        }
        if b_len == 0 {
            return Err(Error::InvalidSpec(
                "convolution input b_len must be non-zero".to_owned(),
            ));
        }

        let output_len = a_len + b_len - 1;
        let fft_len = output_len
            .checked_next_power_of_two()
            .ok_or_else(|| Error::InvalidSpec("convolution FFT length overflow".to_owned()))?;

        let fwd = self.dft.create_plan(fft_len, Direction::Forward)?;
        let inv = self.dft.create_plan(fft_len, Direction::Inverse)?;

        let zero = Complex::new(T::zero(), T::zero());
        let scratch = ConvScratch {
            buf_a: vec![zero; fft_len],
            buf_b: vec![zero; fft_len],
            buf_c: vec![zero; fft_len],
        };

        Ok(OmniConvPlan {
            fwd,
            inv,
            vecops: self.vecops.clone(),
            a_len,
            b_len,
            output_len,
            scratch: Mutex::new(scratch),
        })
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;
    use crate::test_utils::{TestDft, TestVecOps};

    const EPSILON: f64 = 1e-8;

    fn assert_approx_eq(actual: &[f64], expected: &[f64], eps: f64) {
        assert_eq!(actual.len(), expected.len(), "slice lengths differ");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < eps,
                "mismatch at index {i}: got {a}, expected {e}"
            );
        }
    }

    fn make_factory() -> OmniConv<TestDft, TestVecOps> {
        OmniConv::new(TestDft, TestVecOps)
    }

    // ── Correctness tests ──────────────────────────────────────────────

    /// `[1, 2, 3] * [4, 5] = [4, 13, 22, 15]`
    #[test]
    fn basic_convolution() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 3, 2).expect("plan creation should succeed");
        let mut output = [0.0_f64; 4];
        plan.process(&[1.0, 2.0, 3.0], &[4.0, 5.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[4.0, 13.0, 22.0, 15.0], EPSILON);
    }

    /// Convolving with `[1]` is an identity operation.
    #[test]
    fn impulse_identity() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 1, 3).expect("plan creation should succeed");
        let mut output = [0.0_f64; 3];
        plan.process(&[1.0], &[2.0, 3.0, 4.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[2.0, 3.0, 4.0], EPSILON);
    }

    /// `[1, 1, 1] * [1, 1, 1] = [1, 2, 3, 2, 1]`
    #[test]
    fn symmetric_inputs() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 3, 3).expect("plan creation should succeed");
        let mut output = [0.0_f64; 5];
        plan.process(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[1.0, 2.0, 3.0, 2.0, 1.0], EPSILON);
    }

    /// `[3] * [7] = [21]`
    #[test]
    fn single_element() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 1, 1).expect("plan creation should succeed");
        let mut output = [0.0_f64; 1];
        plan.process(&[3.0], &[7.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[21.0], EPSILON);
    }

    /// A delayed impulse shifts the signal.
    #[test]
    fn delayed_impulse() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 3, 3).expect("plan creation should succeed");
        let mut output = [0.0_f64; 5];
        plan.process(&[0.0, 1.0, 0.0], &[2.0, 3.0, 4.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[0.0, 2.0, 3.0, 4.0, 0.0], EPSILON);
    }

    /// Convolution is commutative: `a * b == b * a`.
    #[test]
    fn commutative() {
        let factory = make_factory();
        let plan_ab = Conv::<f64>::create_plan(&factory, 3, 2).expect("plan ab should succeed");
        let plan_ba = Conv::<f64>::create_plan(&factory, 2, 3).expect("plan ba should succeed");

        let mut out_ab = [0.0_f64; 4];
        let mut out_ba = [0.0_f64; 4];

        plan_ab
            .process(&[1.0, 2.0, 3.0], &[4.0, 5.0], &mut out_ab)
            .expect("ab should succeed");
        plan_ba
            .process(&[4.0, 5.0], &[1.0, 2.0, 3.0], &mut out_ba)
            .expect("ba should succeed");

        assert_approx_eq(&out_ab, &out_ba, EPSILON);
    }

    /// A plan can be reused for multiple process calls.
    #[test]
    fn plan_reuse() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 2, 2).expect("plan creation should succeed");

        let mut out1 = [0.0_f64; 3];
        let mut out2 = [0.0_f64; 3];

        plan.process(&[1.0, 2.0], &[3.0, 4.0], &mut out1)
            .expect("first call should succeed");
        plan.process(&[5.0, 6.0], &[7.0, 8.0], &mut out2)
            .expect("second call should succeed");

        // [1,2]*[3,4] = [3, 10, 8]
        assert_approx_eq(&out1, &[3.0, 10.0, 8.0], EPSILON);
        // [5,6]*[7,8] = [35, 82, 48]
        assert_approx_eq(&out2, &[35.0, 82.0, 48.0], EPSILON);
    }

    // ── Validation tests ───────────────────────────────────────────────

    #[test]
    fn zero_a_len_returns_error() {
        let factory = make_factory();
        let result = Conv::<f64>::create_plan(&factory, 0, 3);
        assert!(result.is_err(), "zero a_len should return error");
    }

    #[test]
    fn zero_b_len_returns_error() {
        let factory = make_factory();
        let result = Conv::<f64>::create_plan(&factory, 3, 0);
        assert!(result.is_err(), "zero b_len should return error");
    }

    #[test]
    fn wrong_a_length_returns_error() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 3, 2).expect("plan creation should succeed");
        let mut output = [0.0_f64; 4];
        assert!(
            plan.process(&[1.0, 2.0], &[4.0, 5.0], &mut output).is_err(),
            "wrong a length should return error"
        );
    }

    #[test]
    fn wrong_b_length_returns_error() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 3, 2).expect("plan creation should succeed");
        let mut output = [0.0_f64; 4];
        assert!(
            plan.process(&[1.0, 2.0, 3.0], &[4.0], &mut output).is_err(),
            "wrong b length should return error"
        );
    }

    #[test]
    fn wrong_output_length_returns_error() {
        let factory = make_factory();
        let plan = Conv::<f64>::create_plan(&factory, 3, 2).expect("plan creation should succeed");
        let mut output = [0.0_f64; 3];
        assert!(
            plan.process(&[1.0, 2.0, 3.0], &[4.0, 5.0], &mut output)
                .is_err(),
            "wrong output length should return error"
        );
    }
}
