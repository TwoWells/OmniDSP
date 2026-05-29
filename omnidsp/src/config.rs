// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Dispatch configuration.

use crate::backend::Backend;

/// Per-primitive backend priority lists.
///
/// Each field is a priority-ordered list of backends for a specific
/// primitive. The first available backend wins. If no backend in the
/// list is available, the Rust fallback is used.
#[derive(Debug, Clone)]
pub struct Config {
    /// Backend priority for DFT operations (first available wins).
    pub dft: Vec<Backend>,
    /// Backend priority for vector operations (first available wins).
    pub vecops: Vec<Backend>,
    /// Backend priority for IIR filtering (first available wins).
    pub iir: Vec<Backend>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dft: vec![Backend::Rust],
            vecops: vec![Backend::Rust],
            iir: vec![Backend::Rust],
        }
    }
}
