// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Spec-parameterized factory traits for the dispatch layer.
//!
//! [`CreatePlan`] and [`CreateProc`] are the two factory traits a backend
//! implements for each spec type it supports.  They are peers: [`CreatePlan`]
//! produces a stateless **Plan** (`execute(&self, …)`, parallelizable), and
//! [`CreateProc`] produces a stateful **Processor** (`process`/`finish`,
//! `&mut self`).  A module implements one, the other, or both — both only when
//! its batch and streaming forms carry **different state** (the CQT: a stateless
//! parallel batch Plan and a stateful streaming Processor).
//!
//! Both keep the spec type as the compile-time dispatch key and introduce the
//! precision `T` at the **method**, via a generic associated type — the spec is
//! non-generic and one spec serves every precision
//! (`dsp.create_plan::<f32>(&spec)`).  The `Self: Backend<T>` bound is
//! load-bearing: a backend is the closed foundational contract — the real-DFT
//! family (`C2c` / `R2c` / `C2r`) plus the vector-ops primitive — that every
//! composite module is built over, threaded through as the module's primitive
//! handle, so choosing a tuned backend accelerates every module's glue with no
//! extra API.
//!
//! Adding a new module means adding a new spec type and a new
//! `CreatePlan<NewSpec>` / `CreateProc<NewSpec>` impl — no new trait
//! definitions, no existing code breaks.

use std::ops::{AddAssign, MulAssign};

use crate::dispatch::Backend;
use crate::error::Result;
use crate::types::DspFloat;

/// Factory trait producing a stateless **Plan** for a spec type.
///
/// A backend implements `CreatePlan<ConvSpec>`, `CreatePlan<DftC2cSpec>`, etc.
/// The spec type drives compile-time dispatch — the compiler monomorphizes each
/// `OmniDSP<B>` instantiation — while the precision `T` is chosen at the call:
/// `dsp.create_plan::<f32>(&spec)`.  The `Self: Backend<T>` bound makes the
/// backend the foundational provider — real-DFT family plus vector ops — every
/// omni module composes over.
///
/// Use `omnidsp_macros::impl_generic_backend!` to generate impls for all known
/// spec types using generic composition from `omnidsp-core` modules.
pub trait CreatePlan<S> {
    /// The plan type produced by this factory for the given spec, at precision
    /// `T`.
    type Plan<T>
    where
        Self: Backend<T>;

    /// Create a plan from the given specification at precision `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    fn create_plan<T>(&self, spec: &S) -> Result<Self::Plan<T>>
    where
        Self: Backend<T>,
        T: DspFloat + AddAssign + MulAssign;
}

/// A backend that can route a **sub-processor** of spec type `S` into a
/// composing (routing) module.
///
/// This is the *composition* peer of [`CreateProc`], and the two are
/// equivalent: a blanket impl makes every `CreateProc<S>` backend a
/// `RoutesSubProc<S>`.  The split exists for one reason — the **diagnostic**.
/// Some modules are *composers*: they route another module as an internal
/// sub-processor (the multirate CQT routes a half-band resample decimator per
/// octave).  A composer's generated impl is bounded on `RoutesSubProc<TheSub>`
/// rather than `CreateProc<TheSub>` directly, so when a backend keeps the
/// composer but does not supply that sub-processor (e.g. the omni resampler was
/// dropped from the backend's module set and nothing replaced it), the compiler
/// reports the dependency in words — naming the missing capability and pointing
/// at the fix — instead of an opaque trait-bound error deep in the composer's
/// machinery.
///
/// The legitimate "drop the omni sub-module, hand-write a native one" path is
/// unaffected: providing a `CreateProc<S>` impl satisfies the blanket, so the
/// diagnostic never fires there.
///
/// [`CreateProc<S>`](CreateProc) is a supertrait, so a composer bounded on
/// `Self: RoutesSubProc<S>` can still name the routed processor type
/// `<Self as CreateProc<S>>::Proc<T>` — the marker adds the diagnostic without
/// hiding the factory.
#[diagnostic::on_unimplemented(
    message = "the backend `{Self}` does not provide a sub-processor for `{S}`",
    label = "this backend cannot route a `{S}` sub-processor",
    note = "a composing module (for example the multirate CQT, which routes a \
            half-band resample decimator) needs this sub-processor but the \
            backend does not implement `CreateProc<{S}>` for it",
    note = "either keep the omni sub-module that supplies it (do not list it in \
            the backend macro's `skip:` set), or hand-write an \
            `impl CreateProc<{S}>` for `{Self}`"
)]
pub trait RoutesSubProc<S>: CreateProc<S> {}

impl<S, B> RoutesSubProc<S> for B where B: CreateProc<S> {}

/// Factory trait producing a stateful **Processor** for a spec type.
///
/// The stateful peer of [`CreatePlan`]: same spec-keyed dispatch and same
/// `T`-at-the-method GAT shape, but the product carries stream state across
/// calls (`process`/`finish`, `&mut self`).  A module that streams (FIR, IIR,
/// resample, streaming CQT) is reached through this trait.
pub trait CreateProc<S> {
    /// The processor type produced by this factory for the given spec, at
    /// precision `T`.
    type Proc<T>
    where
        Self: Backend<T>;

    /// Create a processor from the given specification at precision `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or processor creation fails.
    fn create_proc<T>(&self, spec: &S) -> Result<Self::Proc<T>>
    where
        Self: Backend<T>,
        T: DspFloat + AddAssign + MulAssign;
}
