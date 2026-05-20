// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! DFT (Discrete Fourier Transform) primitive traits.
//!
//! The [`Dft`] factory creates [`DftPlan`] execution objects configured from
//! a [`DftSpec`].  Plans are immutable — a single plan can be reused across
//! many calls and shared between threads.

use num_complex::Complex;

use crate::error::Result;
use crate::types::Direction;

// ─── Spec ────────────────────────────────────────────────────────────

/// Normalization convention for the DFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DftNorm {
    /// No normalization — round-trip scales by N.
    None,
    /// Divide by N on the inverse transform — round-trip is identity.
    Inverse,
    /// Divide by √N on both transforms — unitary (energy-preserving).
    Ortho,
}

/// DFT operation specification.
///
/// Describes the length, direction, and normalization convention for a
/// DFT plan.  Any backend can consume this spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DftSpec {
    /// Transform length (number of samples).
    pub length: usize,
    /// Transform direction (forward or inverse).
    pub direction: Direction,
    /// Normalization convention.
    pub norm: DftNorm,
}

impl DftSpec {
    /// Create a new DFT spec with the given length, direction, and normalization.
    #[must_use]
    pub const fn new(length: usize, direction: Direction, norm: DftNorm) -> Self {
        Self {
            length,
            direction,
            norm,
        }
    }
}

// ─── Traits ──────────────────────────────────────────────────────────

/// Execution object for a configured DFT operation.
///
/// A plan is created by a [`Dft`] factory and holds any precomputed state
/// (twiddle factors, scratch buffers, etc.) needed to execute the transform
/// efficiently.  Plans are immutable and take `&self`.
pub trait DftPlan<T>: Send + Sync {
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

    /// Create a plan for a DFT described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero or otherwise unsupported by the
    /// implementation.
    fn create_plan(&self, spec: &DftSpec) -> Result<Self::Plan>;
}
