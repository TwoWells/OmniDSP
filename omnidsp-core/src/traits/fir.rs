// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FIR (Finite Impulse Response) filter primitive traits.
//!
//! The [`Fir`] factory creates [`FirPlan`] execution objects configured from
//! a [`FirSpec`].  Plans are mutable — they maintain state across calls
//! and take `&mut self`.

use crate::error::Result;

// ─── Spec ────────────────────────────────────────────────────────────

/// FIR filter implementation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirStrategy {
    /// Backend decides based on filter length.
    Auto,
    /// Time-domain direct convolution (MAC loop).
    Direct,
    /// Frequency-domain overlap-save.
    OverlapSave,
}

/// FIR filter specification.
///
/// Describes the filter coefficients and preferred implementation strategy.
/// This is the contract between the design layer (which computes
/// coefficients) and any backend (which executes the filter).
#[derive(Debug, Clone)]
pub struct FirSpec<T> {
    /// Filter tap coefficients.
    pub coefficients: Vec<T>,
    /// Preferred implementation strategy.
    pub strategy: FirStrategy,
}

impl<T> FirSpec<T> {
    /// Create a new FIR spec with the given coefficients and [`FirStrategy::Auto`].
    #[must_use]
    pub const fn new(coefficients: Vec<T>) -> Self {
        Self {
            coefficients,
            strategy: FirStrategy::Auto,
        }
    }

    /// Override the implementation strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: FirStrategy) -> Self {
        self.strategy = strategy;
        self
    }
}

// ─── Traits ──────────────────────────────────────────────────────────

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
pub trait Fir<T> {
    /// The concrete plan type returned by this factory.
    type Plan: FirPlan<T>;

    /// Create a plan for a FIR filter described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the coefficients are empty or otherwise
    /// unsupported by the implementation.
    fn create_plan(&self, spec: &FirSpec<T>) -> Result<Self::Plan>;
}
