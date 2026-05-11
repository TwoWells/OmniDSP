// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Core type aliases and enums.

pub use num_complex::{Complex32, Complex64};

/// Transform direction for DFT and related operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward transform.
    Forward,
    /// Inverse transform.
    Inverse,
}

/// Window function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Hann (raised cosine) window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
}
