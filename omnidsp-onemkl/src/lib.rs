// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Intel oneMKL backend for `OmniDSP`.
//!
//! [`OneMklBackend`] combines three DFTI-backed DFT factories
//! ([`OneMklDftC2c`] / [`OneMklDftR2c`] / [`OneMklDftC2r`]) with accelerated
//! [`VecOps`](omnidsp_core::traits::vecops::VecOps) implementations over oneMKL
//! vector math (VM) and level-1 BLAS.  The backend *is* its own vector-ops
//! provider; the `impl_generic_backend!` macro threads it as each composite
//! module's vector-ops handle, so every module (convolution, FIR, CQT, …) is
//! accelerated transitively without any per-module override.
//!
//! Convolution and cross-correlation ride *free* over the accelerated DFT in
//! this build (no `skip`): they are the generic FFT-domain compositions from
//! `omnidsp-core`, now running over DFTI FFTs and VM complex multiplies.  Native
//! VS convolution / correlation overrides are a deliberate follow-up.
//!
//! # Safety isolation
//!
//! The crate is `deny(unsafe_code)`.  Every `unsafe` operation — FFI calls,
//! pointer casts, and the `unsafe impl Send`/`Sync` for the DFT plans — is
//! confined to the [`ffi`] module, which alone carries
//! `#![allow(unsafe_code)]`.  The DFT and vector-ops modules are entirely
//! unsafe-free.
//!
//! # Linking
//!
//! This crate links dynamically against `libmkl_rt` (the oneMKL Single Dynamic
//! Library) through `omnidsp-onemkl-sys`.  If the crate is compiled, MKL must be
//! available at link and run time.

mod dft;
mod ffi;
mod vecops;

pub use dft::{
    OneMklDftC2c, OneMklDftC2cPlan, OneMklDftC2r, OneMklDftC2rPlan, OneMklDftR2c, OneMklDftR2cPlan,
};

/// Intel oneMKL backend.
///
/// Combines the three DFTI DFT factories with accelerated `VecOps` over oneMKL
/// VM + BLAS L1.  It is a zero-sized value type (the heavyweight DFTI
/// descriptors live in the plans, not the backend), so it is cheap to `Copy`
/// and share.
#[derive(Debug, Clone, Copy)]
pub struct OneMklBackend {
    pub(crate) dftc2c: OneMklDftC2c,
    pub(crate) dftr2c: OneMklDftR2c,
    pub(crate) dftc2r: OneMklDftC2r,
}

impl OneMklBackend {
    /// Create a new oneMKL backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dftc2c: OneMklDftC2c,
            dftr2c: OneMklDftR2c,
            dftc2r: OneMklDftC2r,
        }
    }
}

impl Default for OneMklBackend {
    fn default() -> Self {
        Self::new()
    }
}

omnidsp_macros::impl_generic_backend! {
    backend: OneMklBackend,
    dftc2c: OneMklDftC2c,
    dftr2c: OneMklDftR2c,
    dftc2r: OneMklDftC2r,
}
