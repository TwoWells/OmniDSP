// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` core library.

/// Resolve a path relative to `testdata/` in the crate root.
///
/// ```ignore
/// include!(testdata!("iir_scipy.rs"));
/// ```
#[cfg(test)]
macro_rules! testdata {
    ($file:expr) => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/", $file)
    };
}

pub mod design;
pub mod error;
pub mod traits;
pub mod types;
