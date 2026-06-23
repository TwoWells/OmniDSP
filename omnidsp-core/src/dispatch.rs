// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Direct-primitive dispatch — the raw DFT accessor the macro shapes.
//!
//! A backend exposes its three *raw* DFT factories through [`RawDft`]; the
//! `impl_generic_backend!` macro emits the per-backend `CreatePlan<Dft*Spec>`
//! impls over that accessor, wrapping the real transforms in the Hermitian
//! boundary shaping ([`HermitianR2c`](crate::hermitian::HermitianR2c) /
//! [`HermitianC2r`](crate::hermitian::HermitianC2r)) and passing the complex
//! transform through bare.
//!
//! The shaping wrappers live **once** in core ([`crate::hermitian`]); the macro
//! merely routes the backend's raw r2c/c2r plans through them, so a custom
//! backend that goes through the macro cannot ship an unshaped direct r2c/c2r
//! plan.  The `CreatePlan<Dft*Spec>` impls are emitted per-backend (not as a
//! core blanket) because the spec is now non-generic and its plan precision
//! arrives at the generic `create_plan::<T>` method: a per-backend impl keeps
//! `Self` concrete, so each `RawDft<f32>` / `RawDft<f64>` capability is in scope
//! when the plan type is named — which a blanket bounded only by the trait's
//! `Self: VecOps<T>` could not guarantee.

use std::ops::{AddAssign, MulAssign};

use num_traits::Float;

use crate::traits::dft::{DftC2c, DftC2r, DftR2c};
use crate::traits::vecops::VecOps;

/// A complete backend at precision `T`.
///
/// Provides the real-DFT family (`C2c` / `R2c` / `C2r`, via [`RawDft`]) and the
/// vector-ops primitive ([`VecOps`]).  This is the **entire** foundational
/// contract: every composite module is built over it, and it never grows — a new
/// module is a new [`CreatePlan`](crate::create::CreatePlan) /
/// [`CreateProc`](crate::create::CreateProc) impl, never a wider backend.
///
/// The three DFT family members are first-class because real-to-complex and
/// complex-to-real transforms are separately-tuned real-FFT primitives (the
/// realfft floor), never synthesized from the complex transform: real DFTs need
/// real per-hardware tuning (as the realfft, scipy, vDSP, and IPP real-FFT
/// kernels all are), so each family member is its own factory.
pub trait Backend<T>: RawDft<T> + VecOps<T>
where
    T: Float + AddAssign + MulAssign,
{
}

impl<T, B> Backend<T> for B
where
    T: Float + AddAssign + MulAssign,
    B: RawDft<T> + VecOps<T>,
{
}

/// A backend's raw (unshaped) DFT primitive factories.
///
/// Implemented per float type (`RawDft<f32>`, `RawDft<f64>`) — usually by the
/// `impl_generic_backend!` macro.  The factories returned are **raw**: the
/// Hermitian boundary shaping is applied by the macro-emitted `CreatePlan`
/// impls through the [`crate::hermitian`] wrappers, never in the backend.
///
/// Accessors return the factory **by value** (not `&self`-borrowed) so the
/// macro can materialize a defaulted realfft floor (`RustDftR2c` / `RustDftC2r`)
/// inline for an omitted slot, with no backing struct field — the
/// "vendor omits; macro materializes" rule.
pub trait RawDft<T> {
    /// The raw complex-to-complex factory (the required base).
    ///
    /// `Clone` because composite modules (FIR/conv/Hilbert/DCT/xcorr) clone the
    /// factory to wrap or reuse it; these factories are cheap value types.
    type C2c: DftC2c<T> + Clone;
    /// The raw real-to-complex factory (native or the realfft floor).
    type R2c: DftR2c<T> + Clone;
    /// The raw complex-to-real factory (native or the realfft floor).
    type C2r: DftC2r<T> + Clone;

    /// The backend's raw c2c factory.
    fn raw_dftc2c(&self) -> Self::C2c;
    /// The backend's raw r2c factory (unshaped).
    fn raw_dftr2c(&self) -> Self::R2c;
    /// The backend's raw c2r factory (unshaped).
    fn raw_dftc2r(&self) -> Self::C2r;
}
