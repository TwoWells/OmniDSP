// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Dispatch trait that maps a spec type to its plan type.

use omnidsp_core::error::Result;

/// Dispatch trait that maps a spec type to its plan type.
///
/// [`OmniDSP`](crate::OmniDSP) implements this for every spec type. Users call
/// `dsp.create_plan(&spec)` and the compiler infers `S` from
/// the spec — no turbofish needed.
pub trait CreatePlan<S> {
    /// The plan type produced from this spec.
    type Plan;

    /// Create a plan for the given specification.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation
    /// fails.
    fn create_plan(&self, spec: &S) -> Result<Self::Plan>;
}
