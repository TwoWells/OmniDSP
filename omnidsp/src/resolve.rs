// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Backend resolution — selects the best available backend from a priority list.

use omnidsp_rust::{RustDft, RustIir, RustVecOps};

use crate::backend::Backend;
use crate::dispatch::{DynDft, DynIir, DynVecOps};

/// Resolve the best available DFT backend.
#[allow(dead_code, reason = "used by OmniDSP::with_config in ticket 02")]
pub fn resolve_dft(priority: &[Backend]) -> DynDft {
    if let Some(&backend) = priority.iter().find(|b| b.is_available()) {
        match backend {
            Backend::Rust => return DynDft::Rust(RustDft),
        }
    }
    DynDft::Rust(RustDft) // ultimate fallback
}

/// Resolve the best available `VecOps` backend.
#[allow(dead_code, reason = "used by OmniDSP::with_config in ticket 02")]
pub fn resolve_vecops(priority: &[Backend]) -> DynVecOps {
    if let Some(&backend) = priority.iter().find(|b| b.is_available()) {
        match backend {
            Backend::Rust => return DynVecOps::Rust(RustVecOps),
        }
    }
    DynVecOps::Rust(RustVecOps) // ultimate fallback
}

/// Resolve the best available IIR backend.
#[allow(dead_code, reason = "used by OmniDSP::with_config in ticket 02")]
pub fn resolve_iir(priority: &[Backend]) -> DynIir {
    if let Some(&backend) = priority.iter().find(|b| b.is_available()) {
        match backend {
            Backend::Rust => return DynIir::Rust(RustIir::new()),
        }
    }
    DynIir::Rust(RustIir::new()) // ultimate fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn resolve_default_returns_rust() {
        let config = Config::default();
        assert!(
            matches!(resolve_dft(&config.dft), DynDft::Rust(_)),
            "default DFT should be Rust"
        );
        assert!(
            matches!(resolve_vecops(&config.vecops), DynVecOps::Rust(_)),
            "default VecOps should be Rust"
        );
        assert!(
            matches!(resolve_iir(&config.iir), DynIir::Rust(_)),
            "default IIR should be Rust"
        );
    }

    #[test]
    fn resolve_empty_priority_falls_back() {
        assert!(
            matches!(resolve_dft(&[]), DynDft::Rust(_)),
            "empty DFT priority should fall back to Rust"
        );
        assert!(
            matches!(resolve_vecops(&[]), DynVecOps::Rust(_)),
            "empty VecOps priority should fall back to Rust"
        );
        assert!(
            matches!(resolve_iir(&[]), DynIir::Rust(_)),
            "empty IIR priority should fall back to Rust"
        );
    }
}
