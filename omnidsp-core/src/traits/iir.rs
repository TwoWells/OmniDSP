// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! IIR (Infinite Impulse Response) filter primitive traits.
//!
//! The [`Iir`] factory creates [`IirPlan`] execution objects configured from
//! an [`IirSpec`].  Plans are mutable — they maintain state across calls
//! and take `&mut self`.

use crate::error::{Error, Result};
use crate::types::BiquadSection;

// ─── Spec ────────────────────────────────────────────────────────────

/// IIR filter specification (biquad cascade).
///
/// Describes the second-order sections for a biquad cascade filter.
/// This is the contract between the design layer (which computes
/// biquad coefficients) and any backend (which executes the filter).
///
/// Construct via [`design::iir::design`](crate::design::iir::design) for
/// automatic coefficient generation, or directly via [`IirSpec::new`] with
/// pre-computed sections.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::iir::IirSpec;
/// use omnidsp_core::types::BiquadSection;
///
/// let spec = IirSpec::new(vec![
///     BiquadSection { b0: 0.5, b1: 0.5, b2: 0.0, a1: 0.0, a2: 0.0 },
/// ]).unwrap();
/// assert_eq!(spec.sections().len(), 1);
/// ```
///
/// The field is private and the spec is valid-by-construction (ADR-006 §4).
#[derive(Debug, Clone)]
pub struct IirSpec<T> {
    sections: Vec<BiquadSection<T>>,
}

impl<T> IirSpec<T> {
    /// Create a new IIR spec from biquad sections.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `sections` is empty.
    pub fn new(sections: Vec<BiquadSection<T>>) -> Result<Self> {
        if sections.is_empty() {
            return Err(Error::InvalidSpec(
                "IIR filter requires at least one biquad section".into(),
            ));
        }
        Ok(Self { sections })
    }

    /// Second-order sections applied in series, first to last.
    ///
    /// The output of `sections()[0]` feeds `sections()[1]`, and so on.
    /// The total filter order is `2 × sections().len()` (or less if
    /// any section is first-order with `b2 = a2 = 0`).
    #[must_use]
    pub fn sections(&self) -> &[BiquadSection<T>] {
        &self.sections
    }
}

// ─── Traits ──────────────────────────────────────────────────────────

/// Execution object for a configured IIR filter (biquad cascade).
///
/// A plan is created by an [`Iir`] factory and holds the biquad coefficients
/// and internal state.  Plans are mutable and take `&mut self` — each call to
/// [`process`](IirPlan::process) continues from where the previous call left
/// off.
pub trait IirPlan<T> {
    /// Filter `input`, writing the result to `output`.
    ///
    /// `output` must have the same length as `input`.  Internal state (biquad
    /// delays) is updated so successive calls form a continuous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer lengths do not match.
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<()>;

    /// Reset internal state (biquad delays) to zero without recreating the plan.
    fn reset(&mut self);
}

/// Factory for creating [`IirPlan`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system.  The associated [`Plan`](Iir::Plan) type lets each
/// implementor return a concrete plan — no `Box<dyn>`.
pub trait Iir<T> {
    /// The concrete plan type returned by this factory.
    type Plan: IirPlan<T>;

    /// Create a plan for an IIR filter described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the sections are empty or otherwise unsupported by
    /// the implementation.
    fn create_plan(&self, spec: &IirSpec<T>) -> Result<Self::Plan>;
}
