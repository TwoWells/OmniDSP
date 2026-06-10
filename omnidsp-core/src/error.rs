// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Error types for `omnidsp-core`.

use thiserror::Error;

/// Errors produced by core operations.
///
/// This enum is `#[non_exhaustive]`: downstream crates matching on it must
/// include a wildcard arm, so new variants can be added without a breaking
/// change.
#[derive(Error, Debug)]
#[non_exhaustive]
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

    /// A runtime failure reported by a backend implementation.
    ///
    /// Covers genuine FFI runtime failures only (out-of-memory, an internal
    /// vendor error). Vendors cannot `impl From<_> for Error` (orphan rule),
    /// and [`Error::Internal`] is semantically wrong for a backend failure, so
    /// backends surface their status via [`Error::backend`].
    #[error("backend error (code {code}): {message}")]
    Backend {
        /// Backend-specific status or error code.
        code: i32,
        /// Human-readable description of the failure.
        message: String,
    },

    /// An internal error not covered by other variants.
    #[error("{0}")]
    Internal(String),
}

impl Error {
    /// Construct a [`Error::Backend`] from a backend status code and message.
    ///
    /// Backends use this to report genuine runtime failures (out-of-memory,
    /// an internal vendor error) across the FFI boundary.
    pub fn backend(code: i32, message: impl Into<String>) -> Self {
        Self::Backend {
            code,
            message: message.into(),
        }
    }
}

/// Convenience alias for results with [`enum@Error`].
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::Error;

    #[test]
    fn backend_constructor_round_trips_fields() {
        // `backend()` always yields `Error::Backend`, so the else arm is
        // unreachable; it exists only to bind the fields.
        let Error::Backend { code, message } = Error::backend(-42, "device out of memory") else {
            unreachable!("Error::backend must construct an Error::Backend variant")
        };
        assert_eq!(code, -42, "backend code should round-trip");
        assert_eq!(
            message, "device out of memory",
            "backend message should round-trip"
        );
    }

    #[test]
    fn backend_accepts_owned_and_borrowed_messages() {
        let from_str = Error::backend(1, "literal");
        let from_string = Error::backend(1, String::from("owned"));
        assert!(
            matches!(from_str, Error::Backend { code: 1, .. }),
            "constructor should accept &str messages"
        );
        assert!(
            matches!(from_string, Error::Backend { code: 1, .. }),
            "constructor should accept String messages"
        );
    }

    #[test]
    fn backend_display_reads_sensibly() {
        let err = Error::backend(7, "fft plan creation failed");
        assert_eq!(
            err.to_string(),
            "backend error (code 7): fft plan creation failed",
            "Display should surface both the code and message"
        );
    }
}
