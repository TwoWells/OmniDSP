// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Convolution primitive types.
//!
//! [`ConvPlan`] is the execution object for a one-shot (full) convolution,
//! configured from a [`ConvSpec`].  Plans are immutable and take `&self`.  The
//! generic [`OmniConv`](crate::modules::conv::OmniConv) module builds plans via
//! an inherent `create_plan` — the `Conv` factory trait was dropped because
//! nothing was generic over it.  For streaming convolution, see the FIR
//! filter primitive.

use crate::error::{Error, Result};

// ─── Spec ────────────────────────────────────────────────────────────

/// Convolution implementation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvMethod {
    /// Backend decides direct vs. FFT.
    ///
    /// How and when `Auto` resolves is a backend detail: the pure-Rust floor
    /// resolves it at plan creation from operation counts (`recommend_method`);
    /// a vendor backend may defer to its own internal auto-select mode.
    Auto,
    /// Frequency-domain (FFT-based) convolution — `O(N log N)`.
    Fft,
    /// Time-domain (direct) convolution — `O(a_len × b_len)`.
    Direct,
}

/// Convolution operation specification.
///
/// Describes the input sizes and preferred implementation method.  Any
/// backend can consume this spec.  The plan validates that
/// [`execute`](ConvPlan::execute) inputs match these lengths.
///
/// Output length is always `a_len + b_len - 1` (full linear convolution).
///
/// The spec is non-generic; precision is chosen at `create_plan::<T>`.  Fields
/// are private and the spec is valid-by-construction.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::conv::{ConvSpec, ConvMethod};
///
/// // Let the backend pick direct vs. FFT
/// let spec = ConvSpec::new(1024, 64, ConvMethod::Auto).unwrap();
/// assert_eq!(spec.a_len(), 1024);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvSpec {
    a_len: usize,
    b_len: usize,
    method: ConvMethod,
}

impl ConvSpec {
    /// Create a new convolution spec with the given input lengths and method.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if either `a_len` or `b_len` is zero.
    pub fn new(a_len: usize, b_len: usize, method: ConvMethod) -> Result<Self> {
        if a_len == 0 || b_len == 0 {
            return Err(Error::InvalidSpec(
                "convolution input lengths must be non-zero".into(),
            ));
        }
        Ok(Self {
            a_len,
            b_len,
            method,
        })
    }

    /// Length of the first input signal.
    #[must_use]
    pub const fn a_len(&self) -> usize {
        self.a_len
    }

    /// Length of the second input (often the shorter kernel).
    #[must_use]
    pub const fn b_len(&self) -> usize {
        self.b_len
    }

    /// Preferred implementation method.  [`ConvMethod::Auto`] is resolved
    /// at plan creation time; the plan itself always uses a concrete method.
    #[must_use]
    pub const fn method(&self) -> ConvMethod {
        self.method
    }
}

// ─── Traits ──────────────────────────────────────────────────────────

/// Execution object for a configured convolution operation.
///
/// A plan is created by the [`OmniConv`](crate::modules::conv::OmniConv) module
/// (or a vendor override) and holds any preallocated state
/// (FFT plans, scratch buffers, etc.) needed to execute the convolution
/// efficiently.  Plans are immutable and take `&self`.
pub trait ConvPlan<T>: Send + Sync {
    /// Convolve inputs `a` and `b`, writing the result to `output`.
    ///
    /// `a` and `b` must have the lengths the plan was created for.  `output`
    /// must have length `a.len() + b.len() - 1` (full convolution).
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer length does not match the expected size.
    fn execute(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()>;
}
