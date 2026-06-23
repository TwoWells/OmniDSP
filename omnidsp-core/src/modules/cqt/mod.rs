// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Constant-Q Transform module — octave-recursive multirate analysis.
//!
//! Two execution paths share one octave decomposition:
//!
//! - [`batch`] — the per-frame multirate CQT (the multirate capstone).
//!   [`OmniCqt`] / [`OmniCqtPlan`] anchor every bin's kernel at the **oldest**
//!   sample of the frame (causal, index 0).  It is the public batch CQT and the
//!   streaming path's **stationary cross-check** (steady-state magnitude/phase);
//!   the streaming path's primary oracle is an independent newest-anchored
//!   reference (see [`stream`]).
//! - [`stream`] — the **newest-anchored** streaming CQT.
//!   [`OmniCqtProcessor`] is a stateful `&mut self`
//!   analyzer mirroring
//!   [`ResampleProcessor`](crate::modules::resample::ResampleProcessor): feed
//!   any chunk of samples, get the hop-boundary feature columns it crossed, each
//!   referenced to the most recent sample so per-bin latency collapses to the
//!   Gabor floor `Q/f` rather than the whole window length.  Batch and stream
//!   are reached over the **same** [`CqtSpec`](crate::design::cqt::CqtSpec) via
//!   `create_plan` (the parallel batch [`OmniCqtPlan`]) and `create_proc` (this
//!   stateful processor).
//!
//! Both paths reuse the per-octave half-spectrum kernel design and octave
//! partitioning in the [`kernel`] submodule, so the kernel math lives in exactly
//! one place.
//!
//! # Newest- vs oldest-anchoring (the streaming crux)
//!
//! Anchoring relocates **which samples each octave/bin analyses**, so it is
//! **not** a pure time shift.  The batch path slices each octave's frame from
//! the *old* edge with kernels at index 0 (oldest samples); the streaming path
//! relocates each octave's frame to **end at "now"** and places each bin's
//! kernel at the frame end (newest samples), so onset latency collapses to the
//! Gabor floor `Q/f` instead of the whole window length.  Because the two cover
//! **different** signal, for a transient/onset their magnitudes differ in
//! *timing* — that earlier response is the responsiveness win.  The pure-phase
//! relationship (magnitudes identical, a per-bin linear phase) holds **only for
//! signals stationary across the frame**: a steady-state cross-check, not the
//! general law.  The streaming correctness gate is therefore equivalence to an
//! *independent* newest-anchored reference (a decimation-free single-FFT CQT
//! with end-placed kernels), with the oldest-anchored batch retained as the
//! stationary magnitude/phase cross-check.

pub mod batch;
pub mod kernel;
pub mod stream;

// Preserve the established public paths (`modules::cqt::OmniCqt`, …): the
// `gen_cqt` macro, dispatch, benches, and conformance reference these names
// directly under `modules::cqt`.
pub use batch::{CqtPlan, OmniCqt, OmniCqtPlan};
#[cfg(any(test, feature = "bench"))]
pub use batch::{SingleFftCqt, SingleFftCqtPlan};
pub use stream::{CqtProcessor, OmniCqtProcessor};
