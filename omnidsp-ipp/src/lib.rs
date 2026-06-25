// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Intel IPP backend for `OmniDSP`.
//!
//! [`IppBackend`] combines three IPP-backed DFT factories ([`IppDftC2c`] /
//! [`IppDftR2c`] / [`IppDftC2r`]) with accelerated
//! [`VecOps`](omnidsp_core::traits::vecops::VecOps) implementations over Intel
//! IPP `ipps` vector ops.  The backend *is* its own vector-ops provider; the
//! `impl_generic_backend!` macro threads it as each composite module's
//! vector-ops handle, so every module (convolution, FIR, cross-correlation,
//! CQT, …) is accelerated transitively without any per-module override.
//!
//! Each DFT plan selects IPP's power-of-two FFT engine when the length allows
//! and the arbitrary-length DFT otherwise — the FFT is materially faster, and
//! the choice is internal to the plan.  This is a **primitives-first** landing:
//! there are no native module overrides yet (`skip` is empty), so convolution,
//! FIR, and the rest compose over the accelerated DFT + vector ops.  IPP's
//! native time-domain kernels (IIR, FIR, resampler, conv) land as overrides in
//! later workstream tickets.
//!
//! # Safety isolation
//!
//! The crate is `deny(unsafe_code)`.  Every `unsafe` operation — FFI calls,
//! pointer casts, and the `unsafe impl Send`/`Sync` for the DFT plans — is
//! confined to the [`ffi`] module, which alone carries `#![allow(unsafe_code)]`.
//! The DFT and vector-ops modules are entirely unsafe-free.
//!
//! # Linking
//!
//! This crate links dynamically against `ipps` / `ippvm` / `ippcore` through
//! `omnidsp-ipp-sys`.  If the crate is compiled, IPP must be available at link
//! and run time.

mod dft;
mod ffi;
mod vecops;

pub use dft::{IppDftC2c, IppDftC2cPlan, IppDftC2r, IppDftC2rPlan, IppDftR2c, IppDftR2cPlan};

/// Intel IPP backend.
///
/// Combines the three IPP DFT factories with accelerated `VecOps` over IPP
/// `ipps`.  It is a zero-sized value type (the heavyweight IPP specs live in the
/// plans, not the backend), so it is cheap to `Copy` and share.
#[derive(Debug, Clone, Copy)]
pub struct IppBackend {
    pub(crate) dftc2c: IppDftC2c,
    pub(crate) dftr2c: IppDftR2c,
    pub(crate) dftc2r: IppDftC2r,
}

impl IppBackend {
    /// Create a new IPP backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dftc2c: IppDftC2c,
            dftr2c: IppDftR2c,
            dftc2r: IppDftC2r,
        }
    }
}

impl Default for IppBackend {
    fn default() -> Self {
        Self::new()
    }
}

omnidsp_macros::impl_generic_backend! {
    backend: IppBackend,
    dftc2c: IppDftC2c,
    dftr2c: IppDftR2c,
    dftc2r: IppDftC2r,
}
