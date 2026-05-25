// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Highway SIMD implementations of `OmniDSP` primitives.
//!
//! [`HwyDft`] implements radix-2 Cooley-Tukey FFT with algorithm logic in Rust
//! and butterfly operations using Highway SIMD kernels.  [`HwyVecOps`] wraps
//! Highway SIMD kernels for element-wise vector operations.
//!
//! These two types are the only primitives `omnidsp-highway` needs to export.
//! Composite modules (Window, Conv, FIR, CQT, resampling) are generic over
//! `Dft<T>` + `VecOps<T>` and work with both `omnidsp-rust` and
//! `omnidsp-highway` interchangeably.

mod dft;
mod vecops;

pub use dft::{HwyDft, HwyDftPlan};
pub use vecops::HwyVecOps;
