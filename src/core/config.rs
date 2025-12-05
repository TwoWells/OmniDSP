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
    
    // TODO: Add load methods (from file, string, etc.)
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}
