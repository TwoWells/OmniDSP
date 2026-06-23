// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Shared backend-conformance suite for `OmniDSP`.
//!
//! One golden-vector test surface that **every** backend — the `RustBackend`
//! floor, future IPP / Accelerate / oneMKL crates, and any third-party backend
//! — is held to, so backends cannot silently drift apart.  The checks are a
//! public library (not test-only), generic over the backend `B` and the float
//! width `T`, so any backend crate can call them from its own `#[test]`:
//!
//! ```ignore
//! #[test]
//! fn conformance() {
//!     omnidsp_conformance::run_all(&MyBackend::new());
//! }
//! ```
//!
//! [`run_all`] runs every per-operation check over **both** `f32` and `f64`
//! ([`ConformanceFloat`]); `f32` is the primary DSP width and carries the looser
//! documented tolerance.  Each `check_*` is also public and independently
//! callable with only the bounds that one operation needs.
//!
//! The golden vectors live in the shared [`omnidsp_testdata`] crate, the single
//! source of truth that breaks the `core → conformance` dependency cycle.

#![allow(
    clippy::expect_used,
    reason = "the conformance suite is a test-assertion harness; expect/assert ARE its API"
)]
#![allow(
    clippy::cast_precision_loss,
    reason = "index→float test-signal synthesis is exact at the small lengths used"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "f64 golden vectors are intentionally narrowed to f32 for f32 conformance"
)]
#![allow(
    clippy::similar_names,
    reason = "DFT code uses the systematic c2c / r2c / c2r short names"
)]
#![allow(
    clippy::cast_lossless,
    reason = "small-index integer→float casts in test-signal synthesis are exact"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "test-signal formulas favor readability over fused multiply-add"
)]
#![allow(
    clippy::type_complexity,
    reason = "golden-vector case tables read clearest as inline tuple slices"
)]

mod modules;
mod primitives;
mod support;

use omnidsp_core::create::{CreatePlan, CreateProc};
use omnidsp_core::design::cqt::CqtSpec;
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::dispatch::Backend;
use omnidsp_core::modules::cqt::{CqtPlan, CqtProcessor};
use omnidsp_core::modules::hilbert::{HilbertPlan, HilbertSpec};
use omnidsp_core::modules::resample::ResampleProcessor;
use omnidsp_core::modules::xcorr::{CrossCorrPlan, CrossCorrSpec};
use omnidsp_core::traits::conv::{ConvPlan, ConvSpec};
use omnidsp_core::traits::dct::{DctPlan, DctSpec};
use omnidsp_core::traits::dft::{
    DftC2cPlan, DftC2cSpec, DftC2rPlan, DftC2rSpec, DftR2cPlan, DftR2cSpec,
};
use omnidsp_core::traits::fir::{FirFilter, FirProcessor, FirSpec};
use omnidsp_core::traits::iir::{IirProcessor, IirSpec};
use omnidsp_core::traits::reconfigure::Reconfigure;
use omnidsp_core::types::BiquadSection;
use omnidsp_core::window::Window;

pub use modules::{
    check_conv, check_cqt, check_cqt_stream, check_dct, check_fir, check_hilbert, check_iir,
    check_reconfigure_cqt_stream, check_reconfigure_fir, check_reconfigure_iir, check_resample,
    check_xcorr,
};
pub use primitives::{check_dft_c2c, check_dft_c2r, check_dft_r2c, check_dft_real_round_trip};
pub use support::ConformanceFloat;

/// A backend implementing the full `OmniDSP` surface for one float width `T`.
///
/// This aggregates every `CreatePlan` / `CreateProc` bound (with its plan- or
/// processor-trait requirement at precision `T`, via the GAT associated type)
/// into a single name so [`run_all`] reads cleanly.  [`Backend<T>`] is a
/// supertrait — the foundational contract (the real-DFT family plus vector ops)
/// the GAT dispatch keys on.  A blanket impl covers every backend that satisfies
/// the parts — including the macro-generated `RustBackend` — so backends never
/// name this trait themselves.
pub trait BackendUnderTest<T>:
    Backend<T>
    + CreatePlan<DftC2cSpec, Plan<T>: DftC2cPlan<T>>
    + CreatePlan<DftR2cSpec, Plan<T>: DftR2cPlan<T>>
    + CreatePlan<DftC2rSpec, Plan<T>: DftC2rPlan<T>>
    + CreatePlan<ConvSpec, Plan<T>: ConvPlan<T>>
    + CreateProc<FirSpec, Proc<T>: FirProcessor<T> + Reconfigure<FirFilter>>
    + CreateProc<IirSpec, Proc<T>: IirProcessor<T> + Reconfigure<[BiquadSection<f64>]>>
    + CreatePlan<DctSpec, Plan<T>: DctPlan<T>>
    + CreatePlan<HilbertSpec, Plan<T>: HilbertPlan<T>>
    + CreatePlan<CrossCorrSpec, Plan<T>: CrossCorrPlan<T>>
    + CreateProc<ResampleSpec, Proc<T>: ResampleProcessor<T>>
    + CreatePlan<CqtSpec, Plan<T>: CqtPlan<T>>
    + CreateProc<CqtSpec, Proc<T>: CqtProcessor<T> + Reconfigure<Window>>
where
    T: omnidsp_core::types::DspFloat + std::ops::AddAssign + std::ops::MulAssign,
{
}

impl<T, B> BackendUnderTest<T> for B
where
    T: omnidsp_core::types::DspFloat + std::ops::AddAssign + std::ops::MulAssign,
    B: Backend<T>
        + CreatePlan<DftC2cSpec, Plan<T>: DftC2cPlan<T>>
        + CreatePlan<DftR2cSpec, Plan<T>: DftR2cPlan<T>>
        + CreatePlan<DftC2rSpec, Plan<T>: DftC2rPlan<T>>
        + CreatePlan<ConvSpec, Plan<T>: ConvPlan<T>>
        + CreateProc<FirSpec, Proc<T>: FirProcessor<T> + Reconfigure<FirFilter>>
        + CreateProc<IirSpec, Proc<T>: IirProcessor<T> + Reconfigure<[BiquadSection<f64>]>>
        + CreatePlan<DctSpec, Plan<T>: DctPlan<T>>
        + CreatePlan<HilbertSpec, Plan<T>: HilbertPlan<T>>
        + CreatePlan<CrossCorrSpec, Plan<T>: CrossCorrPlan<T>>
        + CreateProc<ResampleSpec, Proc<T>: ResampleProcessor<T>>
        + CreatePlan<CqtSpec, Plan<T>: CqtPlan<T>>
        + CreateProc<CqtSpec, Proc<T>: CqtProcessor<T> + Reconfigure<Window>>,
{
}

/// Run every conformance check for one float width.
fn run_all_for<B, T>(b: &B)
where
    T: ConformanceFloat + std::ops::MulAssign,
    omnidsp_core::scalar::ScalarVecOps: omnidsp_core::traits::vecops::VecOps<T>,
    B: BackendUnderTest<T>,
{
    primitives::check_dft_c2c::<B, T>(b);
    primitives::check_dft_r2c::<B, T>(b);
    primitives::check_dft_c2r::<B, T>(b);
    primitives::check_dft_real_round_trip::<B, T>(b);
    modules::check_conv::<B, T>(b);
    modules::check_fir::<B, T>(b);
    modules::check_iir::<B, T>(b);
    modules::check_dct::<B, T>(b);
    modules::check_hilbert::<B, T>(b);
    modules::check_xcorr::<B, T>(b);
    modules::check_resample::<B, T>(b);
    modules::check_cqt::<B, T>(b);
    modules::check_cqt_stream::<B, T>(b);
    modules::check_reconfigure_fir::<B, T>(b);
    modules::check_reconfigure_iir::<B, T>(b);
    modules::check_reconfigure_cqt_stream::<B, T>(b);
}

/// Run the full conformance suite against `b` over both `f32` and `f64`.
///
/// One line per backend:
///
/// ```ignore
/// #[test]
/// fn conformance() {
///     omnidsp_conformance::run_all(&RustBackend::new());
/// }
/// ```
///
/// # Panics
///
/// Panics on the first deviation from the golden vectors beyond the documented
/// per-operation tolerance, or if any contracted error case fails to error.
pub fn run_all<B>(b: &B)
where
    B: BackendUnderTest<f32> + BackendUnderTest<f64>,
{
    run_all_for::<B, f64>(b);
    run_all_for::<B, f32>(b);
}
