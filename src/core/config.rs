use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::core::error::{OmniError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub default_backends: Vec<String>,
    pub module_overrides: HashMap<String, Vec<String>>,
    pub backend_settings: HashMap<String, toml::Table>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            default_backends: vec!["omni".to_string()],
            module_overrides: HashMap::new(),
            backend_settings: HashMap::new(),
        }
    }

    /// Returns the list of preferred backends for a specific module (e.g., "dft", "fir").
    /// Resolves overrides first, then falls back to global defaults.
    pub fn get_preferred_backends(&self, module: &str) -> &Vec<String> {
        if let Some(overrides) = self.module_overrides.get(module) {
            overrides
        } else {
            &self.default_backends
        }
    }
    
    // TODO: Add load methods (from file, string, etc.)
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}