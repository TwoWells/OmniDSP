// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! DFT (Discrete Fourier Transform) primitive traits.
//!
//! The [`DftC2c`] factory creates [`DftC2cPlan`] execution objects configured from
//! a [`DftC2cSpec`].  Plans are immutable — a single plan can be reused across
//! many calls and shared between threads.

use std::marker::PhantomData;

use num_complex::Complex;

use crate::error::Result;
use crate::types::Direction;

// ─── Spec ────────────────────────────────────────────────────────────

/// Normalization convention for the DFT.
///
/// Controls the scaling factor applied during forward and inverse transforms.
/// A forward+inverse round-trip must use the same convention to get
/// predictable results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DftNorm {
    /// No normalization on either direction — `IFFT(FFT(x)) = N·x`.
    None,
    /// Divide by N on the inverse transform — `IFFT(FFT(x)) = x`.
    ///
    /// This is the convention used by the convolution and FIR modules.
    Inverse,
    /// Divide by √N on both transforms — unitary (energy-preserving).
    Ortho,
}

/// DFT operation specification.
///
/// Describes the length, direction, and normalization convention for a
/// DFT plan.  Any backend can consume this spec.
///
/// The type parameter `T` ties the spec to a specific float type, making
/// specs fully self-describing for the dispatch layer's `CreatePlan<S>` trait.
/// `T` is carried via [`PhantomData`] and hidden behind the constructor.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::dft::{DftC2cSpec, DftNorm};
/// use omnidsp_core::types::Direction;
///
/// // 1024-point forward FFT with inverse normalization (round-trip = identity)
/// let spec = DftC2cSpec::<f64>::new(1024, Direction::Forward, DftNorm::Inverse);
/// assert_eq!(spec.length, 1024);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DftC2cSpec<T> {
    /// Number of complex samples for both input and output buffers.
    ///
    /// Determines the frequency resolution: each bin spans
    /// `sample_rate / length` Hz.
    pub length: usize,
    /// Transform direction (forward or inverse).
    pub direction: Direction,
    /// Normalization convention.
    ///
    /// Must be consistent between a forward/inverse pair for the
    /// round-trip to produce the expected scaling.
    pub norm: DftNorm,
    _marker: PhantomData<T>,
}

impl<T> PartialEq for DftC2cSpec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length && self.direction == other.direction && self.norm == other.norm
    }
}

impl<T> Eq for DftC2cSpec<T> {}

impl<T> DftC2cSpec<T> {
    /// Create a new DFT spec with the given length, direction, and normalization.
    #[must_use]
    pub const fn new(length: usize, direction: Direction, norm: DftNorm) -> Self {
        Self {
            length,
            direction,
            norm,
            _marker: PhantomData,
        }
    }
}

// ─── Traits ──────────────────────────────────────────────────────────

/// Execution object for a configured DFT operation.
///
/// A plan is created by a [`DftC2c`] factory and holds any precomputed state
/// (twiddle factors, scratch buffers, etc.) needed to execute the transform
/// efficiently.  Plans are immutable and take `&self`.
pub trait DftC2cPlan<T>: Send + Sync {
    /// Execute the DFT on `input`, writing the result to `output`.
    ///
    /// Both buffers must have the length the plan was created for.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer lengths do not match the plan length.
    fn process(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> Result<()>;
}

/// Factory for creating [`DftC2cPlan`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system: `impl DftC2c<f32>` and `impl DftC2c<f64>` are independent
/// capabilities.  The associated [`Plan`](DftC2c::Plan) type lets each implementor
/// return a concrete plan — no `Box<dyn>`.
pub trait DftC2c<T> {
    /// The concrete plan type returned by this factory.
    type Plan: DftC2cPlan<T>;

    /// Create a plan for a DFT described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero or otherwise unsupported by the
    /// implementation.
    fn create_plan(&self, spec: &DftC2cSpec<T>) -> Result<Self::Plan>;
}
