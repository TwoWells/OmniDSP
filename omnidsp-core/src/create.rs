// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Spec-parameterized factory trait for the dispatch layer.
//!
//! [`CreatePlan`] is the single trait that backends implement for each
//! spec type they support.  Adding a new module means adding a new spec
//! type and a new `CreatePlan<NewSpec>` impl — no new trait definitions,
//! no existing code breaks.

use crate::error::Result;

/// Factory trait parameterized by a spec type.
///
/// A backend implements `CreatePlan<ConvSpec<f32>>`,
/// `CreatePlan<DftSpec<f64>>`, etc.  The spec type drives compile-time
/// dispatch — the compiler monomorphizes each `OmniDSP<B>` instantiation.
///
/// Use `omnidsp_macros::impl_generic_backend!` to generate impls for
/// all known spec types using generic composition from `omnidsp-core`
/// modules.
pub trait CreatePlan<S> {
    /// The plan type produced by this factory for the given spec.
    type Plan;

    /// Create a plan from the given specification.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    fn create_plan(&self, spec: &S) -> Result<Self::Plan>;
}
