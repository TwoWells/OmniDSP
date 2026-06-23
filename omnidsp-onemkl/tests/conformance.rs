// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! The acceptance gate: the shared `omnidsp_conformance` golden-vector suite,
//! run against `OneMklBackend` over both `f32` and `f64`.

#[test]
fn onemkl_backend_conformance() {
    omnidsp_conformance::run_all(&omnidsp_onemkl::OneMklBackend::new());
}
