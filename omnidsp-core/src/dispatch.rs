// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Direct-primitive dispatch — shaped r2c/c2r, bare c2c (ADR-010 §5, ADR-011 §2).
//!
//! A backend exposes its three *raw* DFT factories through [`RawDft`]; this
//! module provides blanket [`CreatePlan`] impls over that accessor that wrap the
//! real transforms in the Hermitian boundary shaping
//! ([`HermitianR2c`] / [`HermitianC2r`], ADR-010 §3/§2) and pass the complex
//! transform through bare (ADR-010 §3b).
//!
//! The shaping is therefore written **once**, in core, and inherited by every
//! backend: a hand-written custom backend that implements only [`RawDft`] still
//! cannot ship an unshaped direct r2c/c2r plan (ADR-011 §2 "central wrapping").
//! The `impl_generic_backend!` macro emits the [`RawDft`] impl; the backend
//! never writes the `CreatePlan<Dft*Spec>` impls itself.

use crate::create::CreatePlan;
use crate::error::Result;
use crate::hermitian::{HermitianC2r, HermitianC2rPlan, HermitianR2c, HermitianR2cPlan};
use crate::traits::dft::{DftC2c, DftC2cSpec, DftC2r, DftC2rSpec, DftR2c, DftR2cSpec};
use crate::types::DspFloat;

/// A backend's raw (unshaped) DFT primitive factories.
///
/// Implemented per float type (`RawDft<f32>`, `RawDft<f64>`) — usually by the
/// `impl_generic_backend!` macro.  The factories returned are **raw**: the
/// Hermitian boundary shaping (ADR-010) lives in the blanket [`CreatePlan`]
/// impls in this module, never in the backend.
///
/// Accessors return the factory **by value** (not `&self`-borrowed) so the
/// macro can materialize a defaulted realfft floor (`RustDftR2c` / `RustDftC2r`)
/// inline for an omitted slot, with no backing struct field — the
/// "vendor omits; macro materializes" rule (ADR-011 §3).
pub trait RawDft<T> {
    /// The raw complex-to-complex factory (the required base, ADR-009 §3).
    type C2c: DftC2c<T>;
    /// The raw real-to-complex factory (native or the realfft floor).
    type R2c: DftR2c<T>;
    /// The raw complex-to-real factory (native or the realfft floor).
    type C2r: DftC2r<T>;

    /// The backend's raw c2c factory.
    fn raw_dftc2c(&self) -> Self::C2c;
    /// The backend's raw r2c factory (unshaped).
    fn raw_dftr2c(&self) -> Self::R2c;
    /// The backend's raw c2r factory (unshaped).
    fn raw_dftc2r(&self) -> Self::C2r;
}

// ─── Blanket dispatch over the raw accessor ──────────────────────────────

/// c2c — exposed **bare** (ADR-010 §3b): no Hermitian convention to enforce.
impl<T, B> CreatePlan<DftC2cSpec<T>> for B
where
    T: DspFloat,
    B: RawDft<T>,
{
    type Plan = <B::C2c as DftC2c<T>>::Plan;

    fn create_plan(&self, spec: &DftC2cSpec<T>) -> Result<Self::Plan> {
        self.raw_dftc2c().create_plan(spec)
    }
}

/// r2c — Hermitian-cleaned **output** (ADR-010 §3): bit-exactly-real DC/Nyquist.
impl<T, B> CreatePlan<DftR2cSpec<T>> for B
where
    T: DspFloat,
    B: RawDft<T>,
{
    type Plan = HermitianR2cPlan<<B::R2c as DftR2c<T>>::Plan>;

    fn create_plan(&self, spec: &DftR2cSpec<T>) -> Result<Self::Plan> {
        HermitianR2c::new(self.raw_dftr2c()).create_plan(spec)
    }
}

/// c2r — Hermitian-projected **input** (ADR-010 §2): the load-bearing drift guard.
impl<T, B> CreatePlan<DftC2rSpec<T>> for B
where
    T: DspFloat,
    B: RawDft<T>,
{
    type Plan = HermitianC2rPlan<<B::C2r as DftC2r<T>>::Plan>;

    fn create_plan(&self, spec: &DftC2rSpec<T>) -> Result<Self::Plan> {
        HermitianC2r::new(self.raw_dftc2r()).create_plan(spec)
    }
}
