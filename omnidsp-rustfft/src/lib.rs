// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `RustFFT`-backed DFT for `OmniDSP`.
//!
//! [`RustDft`] wraps [RustFFT](https://crates.io/crates/rustfft) for
//! arbitrary-length DFTs.  This is the portable FFT fallback — no C++
//! toolchain, no FFI, builds everywhere Rust builds.
//!
//! Scalar primitives (`ScalarVecOps`, `ScalarIir`) live in
//! `omnidsp-core::scalar`.

mod dft;

pub use dft::{RustDft, RustDftPlan};
