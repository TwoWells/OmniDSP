// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Scalar fallback implementations of primitive traits.
//!
//! [`ScalarVecOps`] provides element-wise vector operations using scalar
//! loops that LLVM auto-vectorizes.  [`ScalarIir`] is a scalar DF2T biquad
//! cascade.  Both are pure math with no external dependencies — they
//! build everywhere Rust builds.

mod iir;
mod vecops;

pub use iir::{ScalarIir, ScalarIirProcessor};
pub use vecops::ScalarVecOps;
