// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FIR (Finite Impulse Response) filter primitive traits.
//!
//! The [`Fir`] factory creates [`FirPlan`] execution objects that hold filter
//! coefficients and a delay line.  Plans are mutable — they maintain state
//! across calls and take `&mut self`.

use crate::error::Result;

/// Execution object for a configured FIR filter.
///
/// A plan is created by a [`Fir`] factory and holds the filter coefficients
/// and internal delay line.  Plans are mutable and take `&mut self` — each
/// call to [`process`](FirPlan::process) continues from where the previous
/// call left off.
pub trait FirPlan<T> {
    /// Filter `input`, writing the result to `output`.
    ///
    /// `output` must have the same length as `input`.  Internal state (delay
    /// line) is updated so successive calls form a continuous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer lengths do not match.
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<()>;

    /// Reset internal state (delay line) to zero without recreating the plan.
    fn reset(&mut self);
}

/// Factory for creating [`FirPlan`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system.  The associated [`Plan`](Fir::Plan) type lets each
/// implementor return a concrete plan — no `Box<dyn>`.
///
/// The factory decides the implementation strategy — direct MAC for short
/// filters, FFT-based overlap-save for long filters.  The consumer does not
/// know or care which strategy is used.
pub trait Fir<T> {
    /// The concrete plan type returned by this factory.
    type Plan: FirPlan<T>;

    /// Create a plan for a FIR filter with the given `coefficients` (tap values).
    ///
    /// # Errors
    ///
    /// Returns an error if the coefficients slice is empty or otherwise
    /// unsupported by the implementation.
    fn create_plan(&self, coefficients: &[T]) -> Result<Self::Plan>;
}
