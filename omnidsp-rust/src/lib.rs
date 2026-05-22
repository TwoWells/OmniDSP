// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Pure Rust implementations of `OmniDSP` primitives.
//!
//! [`RustDft`] wraps [RustFFT](https://crates.io/crates/rustfft) for
//! arbitrary-length DFTs.  [`RustVecOps`] provides scalar loop
//! implementations of element-wise operations that LLVM auto-vectorizes.
//! [`RustIir`] is a scalar DF2T biquad cascade.
//!
//! This crate is a portable fallback — no C++ toolchain, no FFI, builds
//! everywhere Rust builds.

/// Resolve a path relative to `testdata/` in the crate root.
///
/// ```ignore
/// include!(testdata!("iir_sosfilt_scipy.rs"));
/// ```
#[cfg(test)]
macro_rules! testdata {
    ($file:expr) => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/", $file)
    };
}

mod dft;
mod iir;
mod vecops;

pub use dft::{RustDft, RustDftPlan};
pub use iir::{RustIir, RustIirPlan};
pub use vecops::RustVecOps;
