// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Native Intel IPP linear-convolution override.
//!
//! [`IppBackend`] lists `conv` in its `skip` set and hand-writes
//! [`CreatePlan<ConvSpec>`](omnidsp_core::create::CreatePlan) here, so
//! convolution runs IPP's `ippsConvolve` in a single optimized call instead of
//! the generic FFT → complex-multiply → inverse-FFT composition.  The user's
//! [`ConvMethod`](omnidsp_core::traits::conv::ConvMethod) drives IPP's own
//! `algType` selector:
//! [`Auto`](omnidsp_core::traits::conv::ConvMethod::Auto) → `ippAlgAuto`,
//! [`Fft`](omnidsp_core::traits::conv::ConvMethod::Fft) → `ippAlgFFT`,
//! [`Direct`](omnidsp_core::traits::conv::ConvMethod::Direct) → `ippAlgDirect`.
//! Without this override a `Direct` request would resolve to the scalar floor;
//! the override makes every method run an IPP kernel end-to-end.
//!
//! Convolution is commutative, so IPP's output matches `OmniDSP`'s lag/order
//! convention directly — no output reversal is needed (unlike a native
//! *correlation*).  The plan owns an IPP work buffer sized for the spec's
//! `a_len` / `b_len`; `ippsConvolve` mutates that scratch on every call, so the
//! engine is held behind a [`Mutex`] and freed with the plan.  This module is
//! entirely unsafe-free: every IPP call and the f32/f64 dispatch live in
//! [`crate::ffi`].

use std::marker::PhantomData;
use std::os::raw::c_int;
use std::sync::Mutex;

use omnidsp_ipp_sys::{IPP_ALG_AUTO, IPP_ALG_DIRECT, IPP_ALG_FFT};

use omnidsp_core::create::CreatePlan;
use omnidsp_core::dispatch::Backend;
use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
use omnidsp_core::types::DspFloat;

use crate::IppBackend;
use crate::ffi;

/// Translate a [`ConvMethod`] onto IPP's `algType` selector.
const fn ipp_alg(method: ConvMethod) -> c_int {
    match method {
        ConvMethod::Auto => IPP_ALG_AUTO,
        ConvMethod::Fft => IPP_ALG_FFT,
        ConvMethod::Direct => IPP_ALG_DIRECT,
    }
}

/// Execution plan for a convolution backed by IPP's native `ippsConvolve`.
///
/// Holds the IPP convolution engine (behind a [`Mutex`] — IPP writes scratch
/// into its work buffer on every call) and the input / output lengths the engine
/// was built for.  Immutable apart from the serialized engine access; freed on
/// `Drop` with the engine's owned buffer.
pub struct IppConvPlan<T> {
    engine: Mutex<ffi::Conv>,
    a_len: usize,
    b_len: usize,
    output_len: usize,
    _marker: PhantomData<T>,
}

impl<T: DspFloat> IppConvPlan<T> {
    /// Build a convolution plan for `spec` at precision `T`.
    ///
    /// `output_len = a_len + b_len - 1` (full linear convolution).
    fn new(spec: &ConvSpec) -> Result<Self> {
        let a_len = spec.a_len();
        let b_len = spec.b_len();
        // `ConvSpec::new` guarantees both lengths are non-zero, so the
        // subtraction cannot underflow.
        let output_len = a_len + b_len - 1;
        let engine = ffi::Conv::build::<T>(a_len, b_len, ipp_alg(spec.method()))?;
        Ok(Self {
            engine: Mutex::new(engine),
            a_len,
            b_len,
            output_len,
            _marker: PhantomData,
        })
    }
}

impl<T: DspFloat> ConvPlan<T> for IppConvPlan<T> {
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the engine lock must be held for the entire ippsConvolve call (IPP mutates the work buffer)"
    )]
    fn execute(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
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

        let mut engine = self
            .engine
            .lock()
            .map_err(|e| Error::Internal(format!("IPP conv engine lock poisoned: {e}")))?;
        engine.exec::<T>(a, b, output)
    }
}

impl CreatePlan<ConvSpec> for IppBackend {
    type Plan<T>
        = IppConvPlan<T>
    where
        Self: Backend<T>;

    fn create_plan<T>(&self, spec: &ConvSpec) -> Result<Self::Plan<T>>
    where
        Self: Backend<T>,
        T: DspFloat,
    {
        IppConvPlan::<T>::new(spec)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;

    /// Canonical full linear convolution, computed directly from the definition
    /// `out[i + j] += a[i] * b[j]`.  This is the ground truth the IPP kernel must
    /// match — it fixes both the values and the lag/order convention.
    fn naive_conv(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        out
    }

    fn to_f64<T: DspFloat>(data: &[T]) -> Vec<f64> {
        data.iter()
            .map(|x| x.to_f64().expect("value representable as f64"))
            .collect()
    }

    fn assert_approx_eq(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: slice lengths differ ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{label}: mismatch at index {i}: got {a}, expected {e} (diff={})",
                (a - e).abs()
            );
        }
    }

    /// Run `a ∗ b` through IPP at width `T` for one method and assert it matches
    /// the naive reference (computed in f64).
    fn check<T: DspFloat>(a_f64: &[f64], b_f64: &[f64], method: ConvMethod, tol: f64, label: &str)
    where
        IppBackend: Backend<T>,
    {
        let a: Vec<T> = a_f64
            .iter()
            .map(|&x| T::from_f64(x).expect("input representable in T"))
            .collect();
        let b: Vec<T> = b_f64
            .iter()
            .map(|&x| T::from_f64(x).expect("kernel representable in T"))
            .collect();

        let backend = IppBackend::new();
        let spec = ConvSpec::new(a.len(), b.len(), method).expect("valid conv spec");
        let plan = backend.create_plan::<T>(&spec).expect("ipp conv plan");

        let mut out = vec![T::from_f64(0.0).expect("zero representable"); a.len() + b.len() - 1];
        plan.execute(&a, &b, &mut out).expect("conv execute");

        let expected = naive_conv(a_f64, b_f64);
        assert_approx_eq(&to_f64(&out), &expected, tol, label);
    }

    /// Every `ConvMethod` over an asymmetric size, both widths — the canonical
    /// oracle that pins values *and* lag order.
    #[test]
    fn methods_match_naive() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [0.5, -1.0, 0.25];
        for &method in &[ConvMethod::Auto, ConvMethod::Direct, ConvMethod::Fft] {
            check::<f64>(&a, &b, method, 1e-9, &format!("asymmetric f64 {method:?}"));
            check::<f32>(&a, &b, method, 1e-4, &format!("asymmetric f32 {method:?}"));
        }
    }

    /// The known textbook result `[1,2,3] ∗ [1,1] = [1,3,5,3]`, matching the
    /// shared conformance check.
    #[test]
    fn known_small_result() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 1.0];
        let expected = [1.0, 3.0, 5.0, 3.0];
        for &method in &[ConvMethod::Auto, ConvMethod::Direct, ConvMethod::Fft] {
            let backend = IppBackend::new();
            let spec = ConvSpec::new(a.len(), b.len(), method).expect("valid conv spec");
            let plan = backend.create_plan::<f64>(&spec).expect("ipp conv plan");
            let mut out = [0.0; 4];
            plan.execute(&a, &b, &mut out).expect("conv execute");
            assert_approx_eq(&out, &expected, 1e-12, &format!("[1,2,3]*[1,1] {method:?}"));
        }
    }

    /// Edge cases: a length-1 kernel (impulse identity, scaled), equal lengths,
    /// and a kernel longer than the signal (convolution is commutative).
    #[test]
    fn edge_cases() {
        // Length-1 kernel scales the signal.
        check::<f64>(
            &[1.0, 2.0, 3.0],
            &[2.0],
            ConvMethod::Auto,
            1e-12,
            "length-1 kernel",
        );
        // Equal lengths.
        check::<f64>(
            &[1.0, -2.0, 3.0],
            &[4.0, 5.0, -6.0],
            ConvMethod::Direct,
            1e-12,
            "equal lengths",
        );
        // Kernel longer than the signal.
        check::<f64>(
            &[1.0, 2.0],
            &[1.0, 0.5, -0.25, 0.125],
            ConvMethod::Fft,
            1e-9,
            "kernel longer than signal",
        );
    }

    /// A wrong output length must be rejected before the IPP call.
    #[test]
    fn output_length_mismatch_errors() {
        let backend = IppBackend::new();
        let spec = ConvSpec::new(3, 2, ConvMethod::Direct).expect("valid conv spec");
        let plan = backend.create_plan::<f64>(&spec).expect("ipp conv plan");

        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 1.0];
        let mut wrong = [0.0; 3]; // should be 4
        let err = plan
            .execute(&a, &b, &mut wrong)
            .expect_err("output-length mismatch must error");
        assert!(
            matches!(err, Error::BufferMismatch { .. }),
            "output-length mismatch must yield BufferMismatch, got {err:?}"
        );
    }
}
