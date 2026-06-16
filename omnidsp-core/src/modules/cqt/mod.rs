// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Constant-Q Transform module — octave-recursive multirate analysis.
//!
//! Two execution paths share one octave decomposition:
//!
//! - [`batch`] — the per-frame multirate CQT (surface-lock capstone `05L`).
//!   [`OmniCqt`] / [`OmniCqtPlan`] anchor every bin's kernel at the **oldest**
//!   sample of the frame (causal, index 0).  It is the public batch CQT and the
//!   **conformance oracle** for the streaming path.
//! - [`stream`] — the **newest-anchored** streaming CQT (ticket 22).
//!   [`OmniCqtStreamPlan`] is a stateful `&mut self` analyzer mirroring
//!   [`ResamplePlan`](crate::modules::resample::ResamplePlan): feed any chunk of
//!   samples, get the hop-boundary feature columns it crossed, each referenced
//!   to the most recent sample so per-bin latency collapses to the Gabor floor
//!   `Q/f` rather than the whole window length.
//!
//! Both paths reuse the per-octave half-spectrum kernel design and octave
//! partitioning in the [`kernel`] submodule, so the kernel math lives in exactly
//! one place.
//!
//! # Newest- vs oldest-anchoring (the streaming crux)
//!
//! Anchoring is a pure time shift, so it affects **only phase, not magnitude**.
//! The batch (oldest-anchored) and streaming (newest-anchored) coefficients for
//! the same window differ by a known per-bin linear phase
//! `exp(-j·2π·f_k·Δ/sr)`, where `Δ` is the shift from window-start to
//! window-end.  Magnitudes are identical; complex outputs match after
//! de-rotating by that per-bin factor.  This equivalence is the correctness
//! gate the streaming conformance check enforces against the batch oracle.

pub mod batch;
pub mod kernel;
pub mod stream;

// Preserve the established public paths (`modules::cqt::OmniCqt`, …): the
// `gen_cqt` macro, dispatch, benches, and conformance reference these names
// directly under `modules::cqt`.
pub use batch::{CqtPlan, OmniCqt, OmniCqtPlan};
#[cfg(any(test, feature = "bench"))]
pub use batch::{SingleFftCqt, SingleFftCqtPlan};
pub use stream::{CqtStreamPlan, CqtStreamSpec, OmniCqtStreamPlan};
