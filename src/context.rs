// Note: This file was previously src/backend/mod.rs
// We must update the module paths to point to src/backends/
use crate::backends::{omni, ipp, accelerate};

use crate::core::config::Config;
use crate::core::error::{Result, OmniError};
use crate::core::types::{Complex32, Complex64};
use crate::traits::dft::{Dft, DftPlan, DftSpec};

/// The Manager that holds active implementations.
pub struct Context {
    config: Config,
    
    // DFT Backends for supported types
    dft_c32: Box<dyn Dft<Complex32>>,
    dft_c64: Box<dyn Dft<Complex64>>,
}

impl Context {
    pub fn new(config: Config) -> Result<Self> {
        let dft_c32 = Self::select_dft_c32(&config)?;
        let dft_c64 = Self::select_dft_c64(&config)?;

        Ok(Self {
            config,
            dft_c32,
            dft_c64,
        })
    }

    fn select_dft_c32(config: &Config) -> Result<Box<dyn Dft<Complex32>>> {
        for backend_name in config.get_preferred_backends("dft") {
            match backend_name.as_str() {
                "omni" => return Ok(Box::new(omni::dft::OmniDft::new())),
                "ipp" => {
                    // TODO: Check availability
                    log::warn!("IPP Backend for DFT (C32) requested but not yet implemented/available.");
                }
                "accelerate" => {
                    // TODO: Check availability
                    log::warn!("Accelerate Backend for DFT (C32) requested but not yet implemented/available.");
                }
                _ => log::warn!("Unknown backend '{}' requested for DFT.", backend_name),
            }
        }
        
        Err(OmniError::NoBackendAvailable("dft".to_string()))
    }

    fn select_dft_c64(config: &Config) -> Result<Box<dyn Dft<Complex64>>> {
        for backend_name in config.get_preferred_backends("dft") {
            match backend_name.as_str() {
                "omni" => return Ok(Box::new(omni::dft::OmniDft::new())),
                "ipp" => {
                    log::warn!("IPP Backend for DFT (C64) requested but not yet implemented/available.");
                }
                "accelerate" => {
                    log::warn!("Accelerate Backend for DFT (C64) requested but not yet implemented/available.");
                }
                _ => log::warn!("Unknown backend '{}' requested for DFT.", backend_name),
            }
        }

        Err(OmniError::NoBackendAvailable("dft".to_string()))
    }
}

// Implement the Dft trait for the Context itself.
// This allows the Context to act as a transparent proxy.

impl Dft<Complex32> for Context {
    fn create_plan(&self, spec: DftSpec) -> Result<Box<dyn DftPlan<Complex32>>> {
        self.dft_c32.create_plan(spec)
    }
}

impl Dft<Complex64> for Context {
    fn create_plan(&self, spec: DftSpec) -> Result<Box<dyn DftPlan<Complex64>>> {
        self.dft_c64.create_plan(spec)
    }
}
