// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Native oneMKL VS cross-correlation override.
//!
//! [`OneMklBackend`](crate::OneMklBackend) lists `xcorr` in its `skip` set and
//! hand-writes [`CreatePlan`](omnidsp_core::create::CreatePlan)`<CrossCorrSpec>`
//! here so a user's [`CorrMethod`](omnidsp_core::modules::xcorr::CorrMethod)
//! drives oneMKL's native VS correlation kernel directly.  The method maps onto
//! the VS correlation mode:
//! [`Auto`](omnidsp_core::modules::xcorr::CorrMethod::Auto) →
//! `VSL_CORR_MODE_AUTO`, [`Fft`](omnidsp_core::modules::xcorr::CorrMethod::Fft) →
//! `VSL_CORR_MODE_FFT`,
//! [`Direct`](omnidsp_core::modules::xcorr::CorrMethod::Direct) →
//! `VSL_CORR_MODE_DIRECT`.  Without this override the generic FFT-domain
//! composition would resolve a `Direct` request to the scalar floor; the
//! override makes that path run an MKL kernel end-to-end.
//!
//! The plan owns a VS correlation task built for the spec's `a_len` / `b_len`
//! and full-correlation output `a_len + b_len - 1`, with `x = a`, `y = b` passed
//! to the executor directly (stride 1).  VS task execution mutates internal task
//! state, so the task pointer is held behind a [`Mutex`] and freed on `Drop`.
//! This module is entirely unsafe-free: every VS call and the f32/f64 dispatch
//! live in [`crate::ffi`].
//!
//! # Lag / order convention
//!
//! VS correlation of `x[0..a_len-1]`, `y[0..b_len-1]` yields `a_len + b_len - 1`
//! values for lags `[-(b_len-1), a_len-1]`.  The intended convention here is the
//! one the rustfft/realfft floor uses (and the shared conformance suite pins to
//! `scipy.signal.correlate`): the full linear cross-correlation
//! `output[k] = Σ_i a[i] · b[i - k + (b_len - 1)]`, with lag 0 landing at index
//! `b_len - 1` and the ordering running from the most-negative lag to the
//! most-positive.  This plan uses `x = a`, `y = b`, `zshape = a_len + b_len - 1`
//! and the **default** task `start` / `decimation`, which is expected to match
//! the floor's lag-0 position and output order.
//!
//! That lag/order convention is the one thing this code cannot validate without
//! Intel MKL — it is verified in CI by the oracle test comparing this plan to
//! the floor for every `CorrMethod`.  **If CI reveals the VS output reversed or
//! offset from the floor, the fix lever is the VS task `start` / `decimation`
//! parameters** (set them explicitly on the task at construction); the FFI
//! surface and this plan's structure stay the same.

use std::marker::PhantomData;
use std::sync::Mutex;

use omnidsp_onemkl_sys::{
    VSL_CORR_MODE_AUTO, VSL_CORR_MODE_DIRECT, VSL_CORR_MODE_FFT, VSLCorrTaskPtr,
};

use omnidsp_core::create::CreatePlan;
use omnidsp_core::dispatch::Backend;
use omnidsp_core::error::{Error, Result};
use omnidsp_core::modules::xcorr::{CorrMethod, CrossCorrPlan, CrossCorrSpec};
use omnidsp_core::types::DspFloat;

use crate::ffi;
use crate::OneMklBackend;

/// Translate a [`CorrMethod`] onto the VS correlation mode constant.
const fn vsl_corr_mode(method: CorrMethod) -> i32 {
    match method {
        CorrMethod::Auto => VSL_CORR_MODE_AUTO,
        CorrMethod::Fft => VSL_CORR_MODE_FFT,
        CorrMethod::Direct => VSL_CORR_MODE_DIRECT,
    }
}

/// Execution plan for a cross-correlation backed by a native oneMKL VS
/// correlation task.
///
/// Holds the VS correlation task (behind a [`Mutex`] — VS execution mutates
/// internal task state) and the input / output lengths the task was built for.
/// Immutable apart from the serialized task access; freed on `Drop`.
pub struct OneMklCrossCorrPlan<T> {
    task: Mutex<VSLCorrTaskPtr>,
    a_len: usize,
    b_len: usize,
    output_len: usize,
    _marker: PhantomData<T>,
}

impl<T> OneMklCrossCorrPlan<T> {
    /// Build a VS correlation task for `spec` at precision `T`.
    ///
    /// `xshape = a_len`, `yshape = b_len`, `zshape = a_len + b_len - 1`.
    fn new(spec: &CrossCorrSpec) -> Result<Self>
    where
        T: DspFloat,
    {
        let a_len = spec.a_len();
        let b_len = spec.b_len();
        let output_len = spec.output_len();
        let mode = vsl_corr_mode(spec.method());
        let task = ffi::corr_new_task::<T>(mode, a_len, b_len, output_len)?;
        Ok(Self {
            task: Mutex::new(task),
            a_len,
            b_len,
            output_len,
            _marker: PhantomData,
        })
    }
}

impl<T: DspFloat> CrossCorrPlan<T> for OneMklCrossCorrPlan<T> {
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the task lock must be held for the entire VS exec call (VS mutates internal task state)"
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

        let task = self
            .task
            .lock()
            .map_err(|e| Error::Internal(format!("VS corr task lock poisoned: {e}")))?;
        ffi::corr_exec::<T>(*task, a, b, output)
    }
}

impl<T> Drop for OneMklCrossCorrPlan<T> {
    fn drop(&mut self) {
        // `&mut self` is exclusive, so `get_mut` reads the task without locking;
        // it still returns the pointer through a poison-tolerant `Result`.
        if let Ok(task) = self.task.get_mut() {
            ffi::corr_delete_task(*task);
        }
    }
}

impl CreatePlan<CrossCorrSpec> for OneMklBackend {
    type Plan<T>
        = OneMklCrossCorrPlan<T>
    where
        Self: Backend<T>;

    fn create_plan<T>(&self, spec: &CrossCorrSpec) -> Result<Self::Plan<T>>
    where
        Self: Backend<T>,
        T: DspFloat,
    {
        OneMklCrossCorrPlan::<T>::new(spec)
    }
}
