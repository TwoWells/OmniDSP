use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::core::error::{OmniError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub default_providers: Vec<String>,
    pub module_overrides: HashMap<String, Vec<String>>,
    pub provider_settings: HashMap<String, toml::Table>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            default_providers: vec!["omni".to_string()],
            module_overrides: HashMap::new(),
            provider_settings: HashMap::new(),
        }
    }

    /// Returns the list of preferred providers for a specific module (e.g., "dft", "fir").
    /// Resolves overrides first, then falls back to global defaults.
    pub fn get_preferred_providers(&self, module: &str) -> &Vec<String> {
        if let Some(overrides) = self.module_overrides.get(module) {
            overrides
        } else {
            &self.default_providers
        }
    }
    
    // TODO: Add load methods (from file, string, etc.)
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}
