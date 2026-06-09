// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `RustFFT`-backed DFT for `OmniDSP`.
//!
//! [`RustDftC2c`] wraps [RustFFT](https://crates.io/crates/rustfft) for
//! arbitrary-length complex DFTs; [`RustDftR2c`] / [`RustDftC2r`] wrap its
//! [realfft](https://crates.io/crates/realfft) companion for the real-input /
//! real-output halves (ADR-009).  Together they are the portable FFT floor —
//! no C++ toolchain, no FFI, builds everywhere Rust builds.
//!
//! Scalar primitives (`ScalarVecOps`, `ScalarIir`) live in
//! `omnidsp-core::scalar`.

mod dft;
mod dft_real;

pub use dft::{RustDftC2c, RustDftC2cPlan};
pub use dft_real::{RustDftC2r, RustDftC2rPlan, RustDftR2c, RustDftR2cPlan};
