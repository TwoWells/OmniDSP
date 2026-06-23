// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! The [`Reconfigure`] capability — live retuning of a stateful Processor.
//!
//! Some Processors carry a **value-only parameter** — one that changes
//! coefficient *values* without changing the buffer / state *layout* — worth
//! retuning live without losing stream state.  A synth filter sweeps its
//! coefficients, an adaptive FIR updates its taps, a visualiser swaps its
//! analysis window — all mid-stream, all keeping the delay line / per-section
//! state / decimator history they have built up.
//!
//! [`Reconfigure`] is a **capability**, not a third execution kind: a
//! reconfigurable Processor is the same kind as any other Processor (it streams
//! the same way), differing only by this one method.  It applies to stateful
//! Processors with a value-only parameter — FIR (`Reconfigure<FirFilter>`), IIR
//! (`Reconfigure<[BiquadSection]>`), and the streaming CQT
//! (`Reconfigure<Window>`).  It does **not** apply to resampling — the knob a
//! resampler would retune is its rate, which is structural to the L/M polyphase
//! factorisation, so a live rate change is a different algorithm, not a
//! reconfiguration — nor to any Plan, which carries no state, so changing a
//! parameter is just a free rebuild.
//!
//! # Contract vs. quality
//!
//! The hard contract is **behavioural**: after `reconfigure(&p)` succeeds,
//! subsequent processing uses `p`.  Doing so glitch-free and without
//! re-allocating — the in-place, state-preserving update — is a
//! quality-of-implementation property (like throughput), not a guaranteed part
//! of the contract.  The omni Processors here do the in-place state-preserving
//! update, which is the whole point of the capability.

use crate::error::Result;

/// Re-derive a Processor's coefficients from a new parameter `P`, preserving its
/// stream state.
///
/// The parameter must be **value-only** — it may change coefficient values but
/// not the processor's buffer / state layout.  A parameter that would change the
/// layout (a FIR with a different tap count, an IIR with a different section
/// count) is rejected with [`Error::StructuralMismatch`]; the caller should
/// rebuild the processor from the new spec instead.  A parameter that cannot
/// change the layout (a CQT analysis window, which is orthogonal to the kernel
/// sizes) never fails on that account.
///
/// [`Error::StructuralMismatch`]: crate::error::Error::StructuralMismatch
pub trait Reconfigure<P: ?Sized> {
    /// Re-derive coefficients from `param`, preserving stream state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::StructuralMismatch`] if `param` would change the
    /// processor's buffer / state layout (the caller should rebuild), or
    /// propagates a coefficient-conversion failure.
    ///
    /// [`Error::StructuralMismatch`]: crate::error::Error::StructuralMismatch
    fn reconfigure(&mut self, param: &P) -> Result<()>;
}
