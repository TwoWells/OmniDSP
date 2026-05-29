// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` — composable DSP with pluggable backends.
//!
//! This crate is the user-facing entry point. It re-exports the
//! `omnidsp-core` public API and adds a generic dispatch layer
//! that lets users select a backend at compile time.

// Re-export core public API
pub use omnidsp_core::design;
pub use omnidsp_core::error;
pub use omnidsp_core::modules;
pub use omnidsp_core::traits;
pub use omnidsp_core::types;

// Re-export Window at crate root for convenience
pub use omnidsp_core::types::Window;

pub mod create;
pub mod generic;
mod macros;
mod omnidsp;

pub use create::{CreateConv, CreateCqt, CreateDft, CreateFir, CreateIir, CreateResampler};
pub use generic::Generic;
pub use omnidsp::OmniDSP;

/// The best available backend, selected at compile time.
///
/// Defaults to [`Generic<RustDft, ScalarVecOps>`](Generic) (pure Rust
/// fallback).  Updated when vendor features (IPP, Accelerate, oneMKL)
/// are enabled.
pub type Best = Generic<omnidsp_rustfft::RustDft, omnidsp_core::scalar::ScalarVecOps>;

/// Convenience alias: [`OmniDSP`] with the best compiled-in backend.
///
/// `Auto::default()` creates an engine using the best backend without
/// naming it explicitly.
pub type Auto = OmniDSP<Best>;

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::Window;
    use super::traits::dft::{DftNorm, DftSpec};
    use super::types::Direction;

    #[test]
    fn reexport_window() {
        let w: Vec<f64> = Window::hann(64).expect("hann should succeed");
        assert_eq!(w.len(), 64, "window length should match");
    }

    #[test]
    fn reexport_dft_spec() {
        let spec = DftSpec::<f64>::new(1024, Direction::Forward, DftNorm::Inverse);
        assert_eq!(spec.length, 1024, "spec length should match");
    }
}
