// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Backend conformance (ADR-007): the `RustBackend` floor is held to the shared
//! `omnidsp-conformance` golden-vector suite — one line, f32 + f64.
//!
//! Every future backend (IPP / Accelerate / oneMKL / third-party) adds the same
//! single-line test against its own backend, so none can silently drift from the
//! floor.

use omnidsp::RustBackend;

#[test]
fn rust_backend_conformance() {
    omnidsp_conformance::run_all(&RustBackend::new());
}
