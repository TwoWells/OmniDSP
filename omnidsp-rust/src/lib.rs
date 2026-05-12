// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Pure Rust implementations of `OmniDSP` primitives.
//!
//! [`RustDft`] wraps [RustFFT](https://crates.io/crates/rustfft) for
//! arbitrary-length DFTs.  [`RustVecOps`] provides scalar loop
//! implementations of element-wise operations that LLVM auto-vectorizes.
//!
//! This crate is a portable fallback — no C++ toolchain, no FFI, builds
//! everywhere Rust builds.

mod dft;
mod vecops;

pub use dft::{RustDft, RustDftPlan};
pub use vecops::RustVecOps;
