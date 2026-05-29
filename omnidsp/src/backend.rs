// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Backend selection for dispatch.

/// Available DSP backends.
///
/// Each variant represents a concrete implementation of one or more
/// primitive traits ([`Dft`](crate::traits::dft::Dft),
/// [`VecOps`](crate::traits::vecops::VecOps),
/// [`Iir`](crate::traits::iir::Iir)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Pure Rust fallback (`RustFFT` + scalar loops).
    Rust,
    // Future: Ipp, Accelerate, OneMkl
}

impl Backend {
    /// Is this backend compiled into the binary?
    pub(crate) const fn is_available(self) -> bool {
        match self {
            Self::Rust => true,
        }
    }
}
