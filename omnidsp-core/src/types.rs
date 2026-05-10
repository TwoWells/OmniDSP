// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Core type aliases and enums.

pub use num_complex::{Complex32, Complex64};

/// Alias for `f32` — single-precision real.
pub type Real32 = f32;

/// Alias for `f64` — double-precision real.
pub type Real64 = f64;

/// Transform direction for DFT and related operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward transform.
    Forward,
    /// Inverse transform.
    Inverse,
}
