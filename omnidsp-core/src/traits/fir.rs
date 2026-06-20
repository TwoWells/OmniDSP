// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FIR (Finite Impulse Response) filter primitive types.
//!
//! [`FirPlan`] is the execution object configured from a [`FirSpec`].  Plans
//! are mutable — they maintain state across calls and take `&mut self`.  The
//! generic [`OmniFir`](crate::modules::fir::OmniFir) module builds plans via an
//! inherent `create_plan` — the `Fir` factory trait was dropped (ADR-006 §1;
//! nothing was generic over it).
//!
//! The designed-filter / filtering-operation split (ADR-012 §2):
//!
//! - [`FirFilter`] is the **designed artifact** — coefficients plus minimal
//!   [`FirMeta`] (rate / normalized cutoff), the reusable unit produced by
//!   [`design::fir::design`](crate::design::fir::design) or constructed directly
//!   via [`FirFilter::new`] for bring-your-own coefficients.  It carries no
//!   execution strategy.
//! - [`FirSpec`] is the FIR **module's execution spec** — a [`FirFilter`] plus a
//!   [`FirStrategy`].

use crate::error::{Error, Result};

// ─── Designed filter ─────────────────────────────────────────────────

/// Metadata describing a designed FIR filter's frequency context.
///
/// Carries the sampling rate and normalized cutoff that produced the
/// coefficients, when known.  Both are `None` for bring-your-own coefficients
/// whose design context is unavailable.  This is the minimal data a composing
/// spec needs to validate cross-spec invariants (ADR-012 §2/§4) — for example,
/// that a resampler's prototype cutoff is compatible with its polyphase
/// factorization.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "fields are f64, which is not Eq, so neither is FirMeta"
)]
pub struct FirMeta {
    sample_rate: Option<f64>,
    normalized_cutoff: Option<f64>,
}

impl FirMeta {
    /// Metadata for a filter designed at a known `sample_rate` (Hz) and
    /// `normalized_cutoff` (cutoff frequency divided by the sample rate).
    ///
    /// The frequency context is `f64` regardless of the filter's coefficient
    /// type `T` (the surface-wide "Hz is f64" rule — see the `spec` module): the
    /// design math is f64 and the sole consumer compares against f64 polyphase
    /// bounds, so storing f64 avoids a round-trip back through `T`.
    #[must_use]
    pub const fn new(sample_rate: f64, normalized_cutoff: f64) -> Self {
        Self {
            sample_rate: Some(sample_rate),
            normalized_cutoff: Some(normalized_cutoff),
        }
    }

    /// Metadata for bring-your-own coefficients with no recorded design
    /// context (both fields `None`).
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            sample_rate: None,
            normalized_cutoff: None,
        }
    }

    /// Sampling rate (Hz) the filter was designed for, if known.
    #[must_use]
    pub const fn sample_rate(&self) -> Option<f64> {
        self.sample_rate
    }

    /// Normalized cutoff (cutoff frequency / sample rate), if known.
    #[must_use]
    pub const fn normalized_cutoff(&self) -> Option<f64> {
        self.normalized_cutoff
    }
}

impl Default for FirMeta {
    /// The same as [`FirMeta::unknown`] — no recorded design context.
    fn default() -> Self {
        Self::unknown()
    }
}

/// A designed FIR filter — tap coefficients plus minimal [`FirMeta`].
///
/// This is the **reusable artifact** of the design layer (ADR-012 §2): the
/// output of [`design::fir::design`](crate::design::fir::design) and a
/// directly-constructible struct ([`FirFilter::new`]) for bring-your-own
/// coefficients (a filter designed in scipy/MATLAB).  It carries no window
/// (consumed by the windowed method, or never existed for equiripple) and no
/// execution strategy — those belong to [`FirSpec`].  The same `FirFilter` is
/// valid for every backend, and is the unit composing specs reuse.
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::fir::{FirFilter, FirMeta};
///
/// // 3-tap moving average, bring-your-own coefficients.
/// let filter = FirFilter::new(vec![1.0 / 3.0; 3], FirMeta::unknown()).unwrap();
/// assert_eq!(filter.coefficients().len(), 3);
/// ```
///
/// Fields are private and the filter is valid-by-construction (ADR-006 §4).
#[derive(Debug, Clone)]
pub struct FirFilter<T> {
    coefficients: Vec<T>,
    meta: FirMeta,
}

impl<T> FirFilter<T> {
    /// Create a designed filter from coefficients and metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `coefficients` is empty.
    pub fn new(coefficients: Vec<T>, meta: FirMeta) -> Result<Self> {
        if coefficients.is_empty() {
            return Err(Error::InvalidSpec(
                "FIR filter requires at least one coefficient".into(),
            ));
        }
        Ok(Self { coefficients, meta })
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

    /// Design metadata (rate / normalized cutoff, when known).
    #[must_use]
    pub const fn meta(&self) -> &FirMeta {
        &self.meta
    }
}

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

/// FIR filter specification — the FIR **module's** execution spec.
///
/// Composes a designed [`FirFilter`] with a preferred [`FirStrategy`]
/// (ADR-012 §2).  This is the contract between the design layer (which computes
/// the [`FirFilter`]) and any backend (which executes it).
///
/// Construct via [`FirSpec::new`] from a [`FirFilter`] — obtained from
/// [`design::fir::design`](crate::design::fir::design) or built directly via
/// [`FirFilter::new`].
///
/// # Examples
///
/// ```
/// use omnidsp_core::traits::fir::{FirFilter, FirMeta, FirSpec, FirStrategy};
///
/// // 3-tap moving average, force direct strategy.
/// let filter = FirFilter::new(vec![1.0 / 3.0; 3], FirMeta::unknown()).unwrap();
/// let spec = FirSpec::new(filter, FirStrategy::Direct);
/// assert_eq!(spec.coefficients().len(), 3);
/// ```
///
/// Fields are private and the spec is valid-by-construction (ADR-006 §4) — the
/// non-empty-coefficients invariant is established by [`FirFilter::new`].
#[derive(Debug, Clone)]
pub struct FirSpec<T> {
    filter: FirFilter<T>,
    strategy: FirStrategy,
}

impl<T> FirSpec<T> {
    /// Create a FIR spec composing a designed `filter` with an execution
    /// `strategy`.
    #[must_use]
    pub const fn new(filter: FirFilter<T>, strategy: FirStrategy) -> Self {
        Self { filter, strategy }
    }

    /// Override the implementation strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: FirStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// The designed filter this spec executes.
    #[must_use]
    pub const fn filter(&self) -> &FirFilter<T> {
        &self.filter
    }

    /// Filter tap coefficients in time-domain order: `h[0], h[1], …, h[N-1]`.
    ///
    /// The number of taps (`coefficients().len()`) determines the filter order
    /// (`N - 1`), group delay (`(N - 1) / 2` samples for linear phase), and
    /// the direct/overlap-save crossover point.
    #[must_use]
    pub fn coefficients(&self) -> &[T] {
        self.filter.coefficients()
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
