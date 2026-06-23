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
pub use omnidsp_core::scalar;
pub use omnidsp_core::traits;
pub use omnidsp_core::types;
pub use omnidsp_core::window;

// Re-export Window at crate root for convenience
pub use omnidsp_core::window::Window;

pub mod create;
mod omnidsp;

pub use create::{CreatePlan, CreateProc};
pub use omnidsp::{OmniDSP, RustBackend};

/// The Intel oneMKL backend, re-exported when the `onemkl` feature is enabled.
#[cfg(feature = "onemkl")]
pub use omnidsp_onemkl::OneMklBackend;

/// The best available backend, selected at compile time.
///
/// Defaults to [`RustBackend`] (pure Rust fallback).  With the `onemkl` feature
/// enabled it resolves to [`OneMklBackend`].  As more vendor features land (IPP,
/// Accelerate) this becomes a `cfg` priority ladder.
#[cfg(feature = "onemkl")]
pub type Best = OneMklBackend;

/// The best available backend, selected at compile time (pure-Rust floor when no
/// vendor feature is enabled).
#[cfg(not(feature = "onemkl"))]
pub type Best = RustBackend;

/// Convenience alias: [`OmniDSP`] with the best compiled-in backend.
///
/// `Auto::default()` creates an engine using the best backend without
/// naming it explicitly.
pub type Auto = OmniDSP<Best>;

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::Window;
    use super::traits::dft::{DftC2cSpec, DftNorm};
    use super::types::Direction;

    #[test]
    fn reexport_window() {
        // The re-exported `Window` enum is reachable from the crate root.
        let recipe: Window = Window::Hann;
        let w: Vec<f64> = recipe.coefficients(64).expect("hann should succeed");
        assert_eq!(w.len(), 64, "window length should match");
    }

    #[test]
    fn reexport_dft_spec() {
        let spec = DftC2cSpec::new(1024, Direction::Forward, DftNorm::Inverse).expect("valid spec");
        assert_eq!(spec.length(), 1024, "spec length should match");
    }
}
