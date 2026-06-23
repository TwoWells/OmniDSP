// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! IIR (Infinite Impulse Response) filter primitive traits.
//!
//! The [`Iir`] factory creates [`IirProcessor`] execution objects configured
//! from an [`IirSpec`].  Processors are stateful — they maintain biquad delays
//! across calls and take `&mut self`.

use crate::error::{Error, Result};
use crate::types::BiquadSection;

// ─── Spec ────────────────────────────────────────────────────────────

/// IIR filter specification (biquad cascade).
///
/// Describes the second-order sections for a biquad cascade filter.
/// This is the contract between the design layer (which computes
/// biquad coefficients) and any backend (which executes the filter).
///
/// Construct via [`design::iir::design`](crate::design::iir::design) for
/// automatic coefficient generation, or directly via [`IirSpec::new`] with
/// pre-computed sections.
///
/// The coefficient cascade is the BYO-able artifact, so the spec carries its
/// sections in **f64** (the design precision) and is non-generic; the cast to
/// the operation's `T` happens at `create_proc::<T>`.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::iir::IirSpec;
/// use omnidsp_core::types::BiquadSection;
///
/// let spec = IirSpec::new(vec![
///     BiquadSection { b0: 0.5, b1: 0.5, b2: 0.0, a1: 0.0, a2: 0.0 },
/// ]).unwrap();
/// assert_eq!(spec.sections().len(), 1);
/// ```
///
/// The field is private and the spec is valid-by-construction.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "sections are f64, which is not Eq, so neither is IirSpec"
)]
pub struct IirSpec {
    sections: Vec<BiquadSection<f64>>,
}

impl IirSpec {
    /// Create a new IIR spec from f64 biquad sections.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `sections` is empty.
    pub fn new(sections: Vec<BiquadSection<f64>>) -> Result<Self> {
        if sections.is_empty() {
            return Err(Error::InvalidSpec(
                "IIR filter requires at least one biquad section".into(),
            ));
        }
        Ok(Self { sections })
    }

    /// Second-order sections applied in series, first to last (in the f64 design
    /// precision).
    ///
    /// The output of `sections()[0]` feeds `sections()[1]`, and so on.
    /// The total filter order is `2 × sections().len()` (or less if
    /// any section is first-order with `b2 = a2 = 0`).
    #[must_use]
    pub fn sections(&self) -> &[BiquadSection<f64>] {
        &self.sections
    }
}

// ─── Traits ──────────────────────────────────────────────────────────

/// Stateful execution object for a configured IIR filter — a **Processor**.
///
/// Created by an [`Iir`] factory and holds the biquad coefficients and internal
/// state.  Processors are mutable and take `&mut self` — each call to
/// [`process`](IirProcessor::process) continues from where the previous call
/// left off.
///
/// An IIR filter has an *infinite* impulse response, so there is no finite
/// ring-down tail to emit: [`finish`](IirProcessor::finish) writes nothing
/// (returns `0`) and only marks the stream's end.  Batch is still
/// `process(everything) + finish`; [`execute`](IirProcessor::execute) is the
/// one-shot convenience.
///
/// # Live retuning
///
/// Retuning the cascade coefficients mid-stream is the separate [`Reconfigure`]
/// capability (`Reconfigure<[BiquadSection<f64>]>`), deliberately *not* a method
/// on this trait — so a `dyn IirProcessor` cannot reach it.  Hold the concrete
/// processor type (or add the `Reconfigure` bound) when you need live retuning.
///
/// [`Reconfigure`]: crate::traits::reconfigure::Reconfigure
pub trait IirProcessor<T> {
    /// Filter the streaming `input`, writing the result to `output`; returns the
    /// number of samples written (equal to `input.len()`).
    ///
    /// `output` must have the same length as `input`.  Internal state (biquad
    /// delays) is updated so successive calls form a continuous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer lengths do not match.
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<usize>;

    /// Signal end-of-stream.  An IIR cascade has no finite tail, so this writes
    /// nothing, returns `0`, and clears the delay state.
    ///
    /// # Errors
    ///
    /// Infallible in practice; the `Result` keeps the Processor contract uniform.
    fn finish(&mut self, output: &mut [T]) -> Result<usize>;

    /// One-shot convenience: filter a complete `input` to a complete `output` on
    /// a fresh stream — `reset`, then `process(input)`, then `finish` (a no-op
    /// for IIR).  Returns the number of samples written (`input.len()`) and
    /// leaves the processor clean.
    ///
    /// `output` must have the same length as `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer lengths do not match.
    fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<usize>;

    /// Reset internal state (biquad delays) to zero without recreating the
    /// processor.
    fn reset(&mut self);
}

/// Factory for creating [`IirProcessor`] execution objects.
///
/// `T` is a type parameter on the factory trait so that capability is expressed
/// in the type system.  The associated [`Proc`](Iir::Proc) type lets each
/// implementor return a concrete processor — no `Box<dyn>`.
pub trait Iir<T> {
    /// The concrete processor type returned by this factory.
    type Proc: IirProcessor<T>;

    /// Create a processor for an IIR filter described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if the sections are empty or otherwise unsupported by
    /// the implementation.
    fn create_proc(&self, spec: &IirSpec) -> Result<Self::Proc>;
}
