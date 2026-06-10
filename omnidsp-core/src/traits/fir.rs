// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FIR (Finite Impulse Response) filter primitive types.
//!
//! [`FirPlan`] is the execution object configured from a [`FirSpec`].  Plans
//! are mutable — they maintain state across calls and take `&mut self`.  The
//! generic [`OmniFir`](crate::modules::fir::OmniFir) module builds plans via an
//! inherent `create_plan` — the `Fir` factory trait was dropped (ADR-006 §1;
//! nothing was generic over it).

use crate::error::{Error, Result};

// ─── Spec ────────────────────────────────────────────────────────────

/// FIR filter implementation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirStrategy {
    /// Backend decides based on filter length.
    ///
    /// Resolved at plan creation time — the plan always uses a concrete strategy.
    Auto,
    /// Time-domain direct convolution — one dot product per output sample.
    ///
    /// Lower overhead for short filters (typically < 64 taps).
    Direct,
    /// Frequency-domain overlap-save — block FFT processing.
    ///
    /// Amortizes the FFT cost over many output samples; wins for longer
    /// filters.
    OverlapSave,
}

/// FIR filter specification.
///
/// Describes the filter coefficients and preferred implementation strategy.
/// This is the contract between the design layer (which computes
/// coefficients) and any backend (which executes the filter).
///
/// Construct via [`design::fir::design`](crate::design::fir::design) for
/// automatic coefficient generation, or directly via [`FirSpec::new`] with
/// pre-computed taps.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::fir::{FirSpec, FirStrategy};
///
/// // 3-tap moving average, force direct strategy
/// let spec = FirSpec::new(vec![1.0 / 3.0; 3]).unwrap()
///     .with_strategy(FirStrategy::Direct);
/// assert_eq!(spec.coefficients().len(), 3);
/// ```
///
/// Fields are private and the spec is valid-by-construction (ADR-006 §4).
#[derive(Debug, Clone)]
pub struct FirSpec<T> {
    coefficients: Vec<T>,
    strategy: FirStrategy,
}

impl<T> FirSpec<T> {
    /// Create a new FIR spec with the given coefficients and [`FirStrategy::Auto`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `coefficients` is empty.
    pub fn new(coefficients: Vec<T>) -> Result<Self> {
        if coefficients.is_empty() {
            return Err(Error::InvalidSpec(
                "FIR filter requires at least one coefficient".into(),
            ));
        }
        Ok(Self {
            coefficients,
            strategy: FirStrategy::Auto,
        })
    }

    /// Override the implementation strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: FirStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Filter tap coefficients in time-domain order: `h[0], h[1], …, h[N-1]`.
    ///
    /// The number of taps (`coefficients().len()`) determines the filter order
    /// (`N - 1`), group delay (`(N - 1) / 2` samples for linear phase), and
    /// the direct/overlap-save crossover point.
    #[must_use]
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }

    /// Preferred implementation strategy.  [`FirStrategy::Auto`] is resolved
    /// at plan creation time based on the number of taps.
    #[must_use]
    pub const fn strategy(&self) -> FirStrategy {
        self.strategy
    }
}

// ─── Traits ──────────────────────────────────────────────────────────

/// Execution object for a configured FIR filter.
///
/// A plan is created by the [`OmniFir`](crate::modules::fir::OmniFir) module
/// (or a vendor override) and holds the filter coefficients
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
