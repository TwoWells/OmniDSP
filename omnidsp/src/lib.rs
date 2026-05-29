// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` — composable DSP with pluggable backends.
//!
//! This crate is the user-facing entry point. It re-exports the
//! [`omnidsp-core`] public API and adds the dispatch layer that
//! selects the best available backend at construction time.

// Re-export core public API
pub use omnidsp_core::design;
pub use omnidsp_core::error;
pub use omnidsp_core::modules;
pub use omnidsp_core::traits;
pub use omnidsp_core::types;

// Re-export Window at crate root for convenience
pub use omnidsp_core::types::Window;

pub mod backend;
pub mod config;
pub mod create_plan;
pub mod dispatch;
mod omnidsp;
pub(crate) mod resolve;

pub use backend::Backend;
pub use config::Config;
pub use create_plan::CreatePlan;
pub use dispatch::{DynDft, DynDftPlan, DynIir, DynIirPlan, DynVecOps};
pub use omnidsp::{
    ConvPlan32, ConvPlan64, CqtPlan32, CqtPlan64, DftPlan32, DftPlan64, FirPlan32, FirPlan64,
    IirPlan32, IirPlan64, OmniDSP, ResamplePlan32, ResamplePlan64,
};

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
