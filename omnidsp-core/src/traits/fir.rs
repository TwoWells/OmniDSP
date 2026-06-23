// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FIR (Finite Impulse Response) filter primitive types.
//!
//! [`FirProcessor`] is the stateful execution object configured from a
//! [`FirSpec`] at the create edge.  Processors carry stream state across calls
//! and take `&mut self`.  The generic
//! [`OmniFir`](crate::modules::fir::OmniFir) module builds processors via an
//! inherent `create_proc` — the `Fir` factory trait was dropped because nothing
//! was generic over it.
//!
//! The designed-filter / filtering-operation split:
//!
//! - [`FirFilter`] is the **designed artifact** — coefficients plus minimal
//!   [`FirMeta`] (rate / normalized cutoff), the reusable unit produced by
//!   [`design::fir::design`](crate::design::fir::design) or constructed directly
//!   via [`FirFilter::new`] for bring-your-own coefficients.  It is the BYO-able
//!   coefficient artifact, so it carries its taps in **f64** (the design
//!   precision); the cast to the operation's `T` happens at the create edge.  It
//!   carries no execution strategy.
//! - [`FirSpec`] is the FIR **module's execution spec** — a [`FirFilter`] plus a
//!   [`FirStrategy`].

use crate::error::{Error, Result};

// ─── Designed filter ─────────────────────────────────────────────────

/// Metadata describing a designed FIR filter's frequency context.
///
/// Carries the sampling rate and normalized cutoff that produced the
/// coefficients, when known.  Both are `None` for bring-your-own coefficients
/// whose design context is unavailable.  This is the minimal data a composing
/// spec needs to validate cross-spec invariants — for example,
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
/// This is the **reusable artifact** of the design layer: the
/// output of [`design::fir::design`](crate::design::fir::design) and a
/// directly-constructible struct ([`FirFilter::new`]) for bring-your-own
/// coefficients (a filter designed in scipy/MATLAB).  It carries no window
/// (consumed by the windowed method, or never existed for equiripple) and no
/// execution strategy — those belong to [`FirSpec`].
///
/// Because hand-rolled coefficients are a first-class entry point, the filter
/// **is** the coefficient artifact — carried in **f64**, the design precision,
/// and cast to the operation's `T` only at the create edge.  The same
/// `FirFilter` is valid for every backend and every precision, and is the unit
/// composing specs reuse.
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
/// Fields are private and the filter is valid-by-construction.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "coefficients are f64, which is not Eq, so neither is FirFilter"
)]
pub struct FirFilter {
    coefficients: Vec<f64>,
    meta: FirMeta,
}

impl FirFilter {
    /// Create a designed filter from f64 coefficients and metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if `coefficients` is empty.
    pub fn new(coefficients: Vec<f64>, meta: FirMeta) -> Result<Self> {
        if coefficients.is_empty() {
            return Err(Error::InvalidSpec(
                "FIR filter requires at least one coefficient".into(),
            ));
        }
        Ok(Self { coefficients, meta })
    }

    /// Filter tap coefficients in time-domain order: `h[0], h[1], …, h[N-1]`,
    /// carried in the f64 design precision.
    ///
    /// The number of taps (`coefficients().len()`) determines the filter order
    /// (`N - 1`), group delay (`(N - 1) / 2` samples for linear phase), and
    /// the direct/overlap-save crossover point.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
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
/// Composes a designed [`FirFilter`] with a preferred [`FirStrategy`].
/// This is the contract between the design layer (which computes
/// the [`FirFilter`]) and any backend (which executes it).
///
/// Construct via [`FirSpec::new`] from a [`FirFilter`] — obtained from
/// [`design::fir::design`](crate::design::fir::design) or built directly via
/// [`FirFilter::new`].  The spec is non-generic (it carries the f64 filter
/// artifact); the operation's precision is chosen at `create_proc::<T>`.
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
/// Fields are private and the spec is valid-by-construction — the
/// non-empty-coefficients invariant is established by [`FirFilter::new`].
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "the composed FirFilter carries f64 coefficients, which are not Eq"
)]
pub struct FirSpec {
    filter: FirFilter,
    strategy: FirStrategy,
}

impl FirSpec {
    /// Create a FIR spec composing a designed `filter` with an execution
    /// `strategy`.
    #[must_use]
    pub const fn new(filter: FirFilter, strategy: FirStrategy) -> Self {
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
    pub const fn filter(&self) -> &FirFilter {
        &self.filter
    }

    /// Filter tap coefficients in time-domain order: `h[0], h[1], …, h[N-1]`
    /// (in the f64 design precision).
    ///
    /// The number of taps (`coefficients().len()`) determines the filter order
    /// (`N - 1`), group delay (`(N - 1) / 2` samples for linear phase), and
    /// the direct/overlap-save crossover point.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
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

/// Stateful execution object for a configured FIR filter — a **Processor**.
///
/// Created by the [`OmniFir`](crate::modules::fir::OmniFir) module (or a vendor
/// override) and holds the filter coefficients and internal delay line.
/// Processors are mutable and take `&mut self` — each call to
/// [`process`](FirProcessor::process) continues from where the previous call
/// left off, and [`finish`](FirProcessor::finish) flushes the filter's
/// ring-down tail once the input ends.
///
/// Drive it batch or streaming the same way: `process(chunk)` repeatedly, then
/// `finish` at the true end of the stream.  Batch is just
/// `process(everything) + finish`; [`execute`](FirProcessor::execute) is the
/// one-shot convenience for exactly that on a fresh stream.
///
/// # Live retuning
///
/// Retuning the taps mid-stream is the separate [`Reconfigure<FirFilter>`]
/// capability, deliberately *not* a method on this trait — so a `dyn
/// FirProcessor` cannot reach it.  Hold the concrete processor type (or add a
/// `Reconfigure<FirFilter>` bound) when you need live retuning.
///
/// [`Reconfigure<FirFilter>`]: crate::traits::reconfigure::Reconfigure
pub trait FirProcessor<T> {
    /// Filter the streaming `input`, writing the steady-state output to
    /// `output`; returns the number of samples written.
    ///
    /// For a same-rate FIR the steady-state count equals `input.len()`, so
    /// `output` must have the same length as `input`.  Internal state (the delay
    /// line) is updated so successive calls form a continuous stream; the
    /// `num_taps − 1` ring-down tail is emitted by [`finish`](Self::finish), not
    /// here.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer lengths do not match.
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<usize>;

    /// Signal end-of-stream: flush the filter's ring-down tail (the
    /// `num_taps − 1` trailing samples of the finite convolution) into `output`,
    /// returning the number written, and leave the delay line cleared.
    ///
    /// `process(everything) + finish` reproduces the full finite-convolution
    /// output (the impulse response runs out to its last tap).
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is shorter than the tail length.
    fn finish(&mut self, output: &mut [T]) -> Result<usize>;

    /// One-shot convenience: filter a complete `input` to a complete `output` on
    /// a fresh stream — `reset` (so prior state cannot leak), then
    /// `process(input)` for the steady output, then `finish` for the tail.
    /// Returns the total number of samples written, and leaves the processor
    /// clean (as after [`reset`](Self::reset)).
    ///
    /// `output` must hold `input.len() + (num_taps − 1)` samples (the finite
    /// convolution length).
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is too short, or if execution fails.
    fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<usize>;

    /// Reset internal state (delay line) to zero without recreating the
    /// processor.
    fn reset(&mut self);
}
