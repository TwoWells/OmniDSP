// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Convolution primitive traits.
//!
//! The [`Conv`] factory creates [`ConvPlan`] execution objects configured for
//! specific input lengths.  Plans are immutable and take `&self`.  This is a
//! one-shot (full) convolution — for streaming convolution, see the FIR filter
//! primitive.

use crate::error::Result;

/// Execution object for a configured convolution operation.
///
/// A plan is created by a [`Conv`] factory and holds any preallocated state
/// (FFT plans, scratch buffers, etc.) needed to execute the convolution
/// efficiently.  Plans are immutable and take `&self`.
pub trait ConvPlan<T> {
    /// Convolve inputs `a` and `b`, writing the result to `output`.
    ///
    /// `a` and `b` must have the lengths the plan was created for.  `output`
    /// must have length `a.len() + b.len() - 1` (full convolution).
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer length does not match the expected size.
    fn process(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()>;
}

/// Factory for creating [`ConvPlan`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system.  The associated [`Plan`](Conv::Plan) type lets each
/// implementor return a concrete plan — no `Box<dyn>`.
pub trait Conv<T> {
    /// The concrete plan type returned by this factory.
    type Plan: ConvPlan<T>;

    /// Create a plan for convolving inputs of length `a_len` and `b_len`.
    ///
    /// The plan preallocates any internal buffers so that execution is
    /// allocation-free.
    ///
    /// # Errors
    ///
    /// Returns an error if either length is zero or otherwise unsupported by
    /// the implementation.
    fn create_plan(&self, a_len: usize, b_len: usize) -> Result<Self::Plan>;
}
