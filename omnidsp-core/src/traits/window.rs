// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Window function primitive traits.
//!
//! The [`Window`] factory creates [`WindowPlan`] execution objects that hold
//! precomputed window coefficients.  Plans are immutable and take `&self`.

use crate::error::Result;
use crate::types::WindowType;

/// Execution object for a configured window operation.
///
/// A plan is created by a [`Window`] factory and holds the precomputed window
/// coefficients.  Plans are immutable and take `&self`.
pub trait WindowPlan<T> {
    /// Apply the window to `data` in-place (element-wise multiply).
    ///
    /// `data` must have the length the plan was created for.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length does not match the plan length.
    fn apply(&self, data: &mut [T]) -> Result<()>;

    /// Return the raw window coefficients.
    ///
    /// Some modules need direct access to the window values (e.g. for
    /// normalization or analysis).
    fn coefficients(&self) -> &[T];
}

/// Factory for creating [`WindowPlan`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system.  The associated [`Plan`](Window::Plan) type lets each
/// implementor return a concrete plan — no `Box<dyn>`.
pub trait Window<T> {
    /// The concrete plan type returned by this factory.
    type Plan: WindowPlan<T>;

    /// Create a plan for the given window specification.
    ///
    /// The [`WindowType`] carries both the window shape and its length (or, for
    /// [`Custom`](WindowType::Custom), the coefficients themselves).
    ///
    /// # Errors
    ///
    /// Returns an error if the window length is zero or otherwise unsupported
    /// by the implementation.
    fn create_plan(&self, window_type: WindowType<T>) -> Result<Self::Plan>;
}
