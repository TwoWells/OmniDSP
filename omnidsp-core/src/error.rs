// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Error types for `omnidsp-core`.

use thiserror::Error;

/// Errors produced by core operations.
#[derive(Error, Debug)]
pub enum Error {
    /// A specification or parameter was invalid.
    #[error("invalid specification: {0}")]
    InvalidSpec(String),

    /// A buffer's length did not match the expected size.
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferMismatch {
        /// The size the operation required.
        expected: usize,
        /// The size that was provided.
        actual: usize,
    },

    /// An internal error not covered by other variants.
    #[error("{0}")]
    Internal(String),
}

/// Convenience alias for results with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
