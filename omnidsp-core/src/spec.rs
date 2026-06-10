// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Specification façade — the single import path for every operation spec
//! (ADR-006 §5).
//!
//! Every `*Spec` is valid-by-construction (private fields, fallible `new`,
//! accessors) and lives next to the trait or module it configures.  This
//! module re-exports them all under one path so callers can `use
//! omnidsp_core::spec::*` to build any spec directly, without tracking which
//! submodule each lives in.  The definition sites stay put — this is a façade,
//! not a relocation.
//!
//! The `design::*` helpers (FIR/IIR/resample/CQT) remain the convenient way to
//! compute coefficients; they return one of these specs, so every spec is also
//! constructible directly via its `new`.
//!
//! # Examples
//!
//! ```
//! use omnidsp_core::spec::{ConvSpec, DctSpec};
//! use omnidsp_core::traits::conv::ConvMethod;
//! use omnidsp_core::traits::dct::{DctNorm, DctType};
//!
//! let conv = ConvSpec::<f64>::new(1024, 64, ConvMethod::Auto).unwrap();
//! let dct = DctSpec::<f64>::new(512, DctType::II, DctNorm::Ortho).unwrap();
//! assert_eq!(conv.a_len(), 1024);
//! assert_eq!(dct.length(), 512);
//! ```

pub use crate::design::cqt::CqtSpec;
pub use crate::design::resample::ResampleSpec;
pub use crate::modules::hilbert::HilbertSpec;
pub use crate::modules::xcorr::{CrossCorrNorm, CrossCorrSpec};
pub use crate::traits::conv::ConvSpec;
pub use crate::traits::dct::DctSpec;
pub use crate::traits::dft::{DftC2cSpec, DftC2rSpec, DftR2cSpec};
pub use crate::traits::fir::FirSpec;
pub use crate::traits::iir::IirSpec;
// AutoCorrSpec / PsdSpec / DwtSpec join as they land (tickets 18/19/20).
