// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Native oneMKL VS convolution override.
//!
//! [`OneMklBackend`](crate::OneMklBackend) lists `conv` in its `skip` set and
//! hand-writes [`CreatePlan`](omnidsp_core::create::CreatePlan)`<ConvSpec>` here
//! so a user's [`ConvMethod`](omnidsp_core::traits::conv::ConvMethod) drives
//! MKL's native VS convolution kernel directly.  The method maps onto the VS
//! convolution mode:
//! [`Auto`](omnidsp_core::traits::conv::ConvMethod::Auto) →
//! `VSL_CONV_MODE_AUTO`, [`Fft`](omnidsp_core::traits::conv::ConvMethod::Fft) →
//! `VSL_CONV_MODE_FFT`, [`Direct`](omnidsp_core::traits::conv::ConvMethod::Direct)
//! → `VSL_CONV_MODE_DIRECT`.  Without this override the generic FFT-domain
//! composition would resolve a `Direct` request to the scalar floor; the
//! override makes that path run an MKL kernel end-to-end.
//!
//! The plan owns a VS convolution task built for the spec's `a_len` / `b_len`
//! and full-convolution output `a_len + b_len - 1`.  VS task execution mutates
//! internal task state, so the task pointer is held behind a [`Mutex`] and
//! freed on `Drop`.  This module is entirely unsafe-free: every VS call and the
//! f32/f64 dispatch live in [`crate::ffi`].

use std::marker::PhantomData;
use std::sync::Mutex;

use omnidsp_onemkl_sys::{
    VSL_CONV_MODE_AUTO, VSL_CONV_MODE_DIRECT, VSL_CONV_MODE_FFT, VSLConvTaskPtr,
};

use omnidsp_core::create::CreatePlan;
use omnidsp_core::dispatch::Backend;
use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
use omnidsp_core::types::DspFloat;

use crate::ffi;
use crate::OneMklBackend;

/// Translate a [`ConvMethod`] onto the VS convolution mode constant.
const fn vsl_conv_mode(method: ConvMethod) -> i32 {
    match method {
        ConvMethod::Auto => VSL_CONV_MODE_AUTO,
        ConvMethod::Fft => VSL_CONV_MODE_FFT,
        ConvMethod::Direct => VSL_CONV_MODE_DIRECT,
    }
}

/// Execution plan for a convolution backed by a native oneMKL VS task.
///
/// Holds the VS convolution task (behind a [`Mutex`] — VS execution mutates
/// internal task state) and the input / output lengths the task was built for.
/// Immutable apart from the serialized task access; freed on `Drop`.
pub struct OneMklConvPlan<T> {
    task: Mutex<VSLConvTaskPtr>,
    a_len: usize,
    b_len: usize,
    output_len: usize,
    _marker: PhantomData<T>,
}

impl<T> OneMklConvPlan<T> {
    /// Build a VS convolution task for `spec` at precision `T`.
    ///
    /// `xshape = a_len`, `yshape = b_len`, `zshape = a_len + b_len - 1`.
    fn new(spec: &ConvSpec) -> Result<Self>
    where
        T: DspFloat,
    {
        let a_len = spec.a_len();
        let b_len = spec.b_len();
        let output_len = a_len + b_len - 1;
        let mode = vsl_conv_mode(spec.method());
        let task = ffi::conv_new_task::<T>(mode, a_len, b_len, output_len)?;
        Ok(Self {
            task: Mutex::new(task),
            a_len,
            b_len,
            output_len,
            _marker: PhantomData,
        })
    }
}

impl<T: DspFloat> ConvPlan<T> for OneMklConvPlan<T> {
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
            .map_err(|e| Error::Internal(format!("VS conv task lock poisoned: {e}")))?;
        ffi::conv_exec::<T>(*task, a, b, output)
    }
}

impl<T> Drop for OneMklConvPlan<T> {
    fn drop(&mut self) {
        // `&mut self` is exclusive, so `get_mut` reads the task without locking;
        // it still returns the pointer through a poison-tolerant `Result`.
        if let Ok(task) = self.task.get_mut() {
            ffi::conv_delete_task(*task);
        }
    }
}

impl CreatePlan<ConvSpec> for OneMklBackend {
    type Plan<T>
        = OneMklConvPlan<T>
    where
        Self: Backend<T>;

    fn create_plan<T>(&self, spec: &ConvSpec) -> Result<Self::Plan<T>>
    where
        Self: Backend<T>,
        T: DspFloat,
    {
        OneMklConvPlan::<T>::new(spec)
    }
}
