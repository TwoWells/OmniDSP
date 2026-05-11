// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! DFT (Discrete Fourier Transform) primitive traits.
//!
//! The [`Dft`] factory creates [`DftPlan`] execution objects configured for a
//! specific length and direction.  Plans are immutable — a single plan can be
//! reused across many calls and shared between threads.

use num_complex::Complex;

use crate::error::Result;
use crate::types::Direction;

/// Execution object for a configured DFT operation.
///
/// A plan is created by a [`Dft`] factory and holds any precomputed state
/// (twiddle factors, scratch buffers, etc.) needed to execute the transform
/// efficiently.  Plans are immutable and take `&self`.
pub trait DftPlan<T> {
    /// Execute the DFT on `input`, writing the result to `output`.
    ///
    /// Both buffers must have the length the plan was created for.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer lengths do not match the plan length.
    fn process(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> Result<()>;
}

/// Factory for creating [`DftPlan`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system: `impl Dft<f32>` and `impl Dft<f64>` are independent
/// capabilities.  The associated [`Plan`](Dft::Plan) type lets each implementor
/// return a concrete plan — no `Box<dyn>`.
pub trait Dft<T> {
    /// The concrete plan type returned by this factory.
    type Plan: DftPlan<T>;

    /// Create a plan for a DFT of the given `length` and `direction`.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero or otherwise unsupported by the
    /// implementation.
    fn create_plan(&self, length: usize, direction: Direction) -> Result<Self::Plan>;
}
