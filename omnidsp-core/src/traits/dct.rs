// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Discrete Cosine Transform (DCT) primitive traits.
//!
//! The [`Dct`] factory creates [`DctPlan`] execution objects configured from
//! a [`DctSpec`].  Plans are immutable and take `&self` — DCT is a stateless
//! transform with no delay line or streaming state.

use std::marker::PhantomData;

use crate::error::Result;

// ─── Types ──────────────────────────────────────────────────────────

/// Supported DCT variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctType {
    /// DCT-II: the "standard" DCT (JPEG, MP3, etc.).
    ///
    /// `X[k] = 2 * sum x[n] cos(π(2n+1)k/(2N))` (unnormalized).
    II,
    /// DCT-III: inverse of DCT-II (up to normalization).
    ///
    /// `X[k] = x[0] + 2 * sum_{n=1} x[n] cos(πn(2k+1)/(2N))` (unnormalized).
    III,
}

/// Normalization convention for the DCT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctNorm {
    /// No extra normalization (scipy default).
    ///
    /// `DCT-III(DCT-II(x)) = 2N * x`.
    None,
    /// Orthonormal: `DCT-III_ortho(DCT-II_ortho(x)) = x`.
    ///
    /// Makes DCT-II and DCT-III exact inverses of each other.
    Ortho,
}

// ─── Spec ────────────────────────────────────────────────────────────

/// DCT operation specification.
///
/// Describes the length, DCT type, and normalization convention for a
/// DCT plan.  Any backend can consume this spec.
///
/// The type parameter `T` ties the spec to a specific float type, making
/// specs fully self-describing for the dispatch layer's `CreatePlan<S>` trait.
/// `T` is carried via [`PhantomData`] and hidden behind the constructor.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::dct::{DctSpec, DctType, DctNorm};
///
/// // 1024-point DCT-II with orthonormal normalization
/// let spec = DctSpec::<f64>::new(1024, DctType::II, DctNorm::Ortho);
/// assert_eq!(spec.length, 1024);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DctSpec<T> {
    /// Number of real samples for both input and output.
    pub length: usize,
    /// DCT variant (II or III).
    pub dct_type: DctType,
    /// Normalization convention.
    pub norm: DctNorm,
    _marker: PhantomData<T>,
}

impl<T> PartialEq for DctSpec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length && self.dct_type == other.dct_type && self.norm == other.norm
    }
}

impl<T> Eq for DctSpec<T> {}

impl<T> DctSpec<T> {
    /// Create a new DCT spec with the given length, type, and normalization.
    #[must_use]
    pub const fn new(length: usize, dct_type: DctType, norm: DctNorm) -> Self {
        Self {
            length,
            dct_type,
            norm,
            _marker: PhantomData,
        }
    }
}

// ─── Traits ──────────────────────────────────────────────────────────

/// Execution object for a configured DCT operation.
///
/// A plan is created by a [`Dct`] factory and holds any precomputed state
/// (DFT sub-plans, twiddle factors, scratch buffers) needed to execute the
/// transform efficiently.  Plans are immutable and take `&self`.
pub trait DctPlan<T>: Send + Sync {
    /// Execute the DCT on `input`, writing the result to `output`.
    ///
    /// Both buffers must have the length the plan was created for.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer lengths do not match the plan length.
    fn process(&self, input: &[T], output: &mut [T]) -> Result<()>;
}

/// Factory for creating [`DctPlan`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system: `impl Dct<f32>` and `impl Dct<f64>` are independent
/// capabilities.  The associated [`Plan`](Dct::Plan) type lets each implementor
/// return a concrete plan — no `Box<dyn>`.
pub trait Dct<T> {
    /// The concrete plan type returned by this factory.
    type Plan: DctPlan<T>;

    /// Create a plan for a DCT described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the length is zero or otherwise unsupported by the
    /// implementation.
    fn create_plan(&self, spec: &DctSpec<T>) -> Result<Self::Plan>;
}
