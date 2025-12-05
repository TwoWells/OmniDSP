pub mod omni;
pub mod ipp;
pub mod accelerate;

use crate::core::config::Config;
use crate::core::error::Result;
use crate::core::types::{Complex32, Complex64};
use crate::traits::dft::{Dft, DftPlan, DftSpec};

/// The Manager that holds active implementations.
pub struct Backend {
    config: Config,
    
    // DFT Providers for supported types
    dft_c32: Box<dyn Dft<Complex32>>,
    dft_c64: Box<dyn Dft<Complex64>>,
}

impl Backend {
    pub fn new(config: Config) -> Self {
        let dft_c32 = Self::select_dft_c32(&config);
        let dft_c64 = Self::select_dft_c64(&config);

        Self {
            config,
            dft_c32,
            dft_c64,
        }
    }

    fn select_dft_c32(config: &Config) -> Box<dyn Dft<Complex32>> {
        for provider_name in config.get_preferred_providers("dft") {
            match provider_name.as_str() {
                "omni" => return Box::new(omni::dft::OmniDft::new()),
                "ipp" => {
                    // TODO: Check feature flag and availability
                    log::warn!("IPP Backend for DFT (C32) requested but not yet implemented/available.");
                }
                "accelerate" => {
                    // TODO: Check target_os and availability
                    log::warn!("Accelerate Backend for DFT (C32) requested but not yet implemented/available.");
                }
                _ => log::warn!("Unknown provider '{}' requested for DFT.", provider_name),
            }
        }
        
        log::warn!("No available DFT (C32) provider found in preferences. Falling back to Omni.");
        Box::new(omni::dft::OmniDft::new())
    }

    fn select_dft_c64(config: &Config) -> Box<dyn Dft<Complex64>> {
        for provider_name in config.get_preferred_providers("dft") {
            match provider_name.as_str() {
                "omni" => return Box::new(omni::dft::OmniDft::new()),
                "ipp" => {
                    log::warn!("IPP Backend for DFT (C64) requested but not yet implemented/available.");
                }
                "accelerate" => {
                    log::warn!("Accelerate Backend for DFT (C64) requested but not yet implemented/available.");
                }
                _ => log::warn!("Unknown provider '{}' requested for DFT.", provider_name),
            }
        }

        log::warn!("No available DFT (C64) provider found in preferences. Falling back to Omni.");
        Box::new(omni::dft::OmniDft::new())
    }
}

// Implement the Dft trait for the Backend itself.
// This allows the Backend to act as a transparent proxy.

impl Dft<Complex32> for Backend {
    fn create_plan(&self, spec: DftSpec) -> Result<Box<dyn DftPlan<Complex32>>> {
        self.dft_c32.create_plan(spec)
    }
}

impl Dft<Complex64> for Backend {
    fn create_plan(&self, spec: DftSpec) -> Result<Box<dyn DftPlan<Complex64>>> {
        self.dft_c64.create_plan(spec)
    }
}