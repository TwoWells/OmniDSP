// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` core library.
//!
//! Test reference vectors live in the [`omnidsp-testdata`] crate (a
//! dev-dependency) rather than inline here, so the `omnidsp-conformance` suite
//! can share the exact same golden data without a dependency cycle (ADR-007 §3).
//!
//! [`omnidsp-testdata`]: https://github.com/TwoWells/OmniDSP

pub mod create;
pub mod design;
pub mod dispatch;
pub mod error;
pub mod hermitian;
pub mod modules;
pub mod scalar;
pub mod spec;
pub mod traits;
pub mod types;
pub mod window;

#[cfg(test)]
pub(crate) mod test_utils;
