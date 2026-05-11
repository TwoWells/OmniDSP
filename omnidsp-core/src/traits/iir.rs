// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! IIR (Infinite Impulse Response) filter primitive traits.
//!
//! The [`Iir`] factory creates [`IirPlan`] execution objects that hold biquad
//! coefficients and filter state.  Plans are mutable — they maintain state
//! across calls and take `&mut self`.

use crate::error::Result;
use crate::types::BiquadSection;

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

    /// Create a plan for an IIR filter with the given biquad `sections`.
    ///
    /// The filter is a cascade of second-order sections applied in order.
    ///
    /// # Errors
    ///
    /// Returns an error if `sections` is empty or otherwise unsupported by
    /// the implementation.
    fn create_plan(&self, sections: &[BiquadSection<T>]) -> Result<Self::Plan>;
}
