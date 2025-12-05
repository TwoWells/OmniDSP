# OmniDSP Configuration Specification

This document defines how the `OmniConfig` is structured, loaded, and resolved.

## 1. Philosophy
*   **Explicit over Implicit:** No "magic" auto-selection. Users define an explicit preference order.
*   **Single Point of Truth:** We do **NOT** cascade or merge configurations. The library loads exactly one configuration source.
*   **Programmatic First:** The filesystem is just one way to get a config. Applications can provide a JSON string or build the struct directly in code.
*   **Provider-Specific Tuning:** The config allows passing arbitrary keys/values (JSON/TOML Tables) to specific backends.

## 2. The Configuration File Format (TOML/JSON)
We support TOML for files (readability) and JSON for programmatic interchange.

```toml
# omnidsp.toml
default_providers = ["ipp", "omni"]

[modules]
dft = ["accelerate", "omni"]

[providers.ipp]
num_threads = 4

[providers.ipp.firsr]
# We pass raw keys directly to the IPP provider. 
# No standardization attempted here.
algType = "direct" 

[providers.accelerate]
# MacOS specific tuning
use_high_precision = true
```

## 3. Creation Logic

The user creates the config using one of these methods. All paths ultimately produce an `OmniConfig` struct.

```rust
// 1. Programmatic (Builder Pattern)
let conf = OmniConfig::new()
    .default_providers(vec!["ipp", "omni"])
    .provider_setting("ipp", "num_threads", 4)
    .build();

// 2. JSON String (Direct)
let json = r#"{ "default_providers": ["omni"] }"#;
let conf = OmniConfig::from_json(json)?;

// 3. Load from Default File (omnidsp.toml)
let conf = OmniConfig::load(ConfigSource::Default)?;

// 4. Load from Named File
// "custom" -> searches for "custom.toml" in paths.
// "/abs/path/to/custom.toml" -> loads directly.
let conf = OmniConfig::load(ConfigSource::Named("custom"))?;
```

### 3.1. File Search Paths (Resolution Order)
The library searches the following locations in order. **The first file found is used.**

*   **Default:** Searches for `omnidsp.toml`.
*   **Named:** Searches for `<name>.toml` (if name is not a path).

**Search Locations:**
1.  **Current Working Directory:** `./<filename>`
2.  **User Config (XDG):** `~/.config/omnidsp/<filename>` (platform specific)
3.  **System Config:** `/etc/omnidsp/<filename>` (platform specific)

### 3.2. No Cascading & Defaults
*   **Single Point of Truth:** If a named config (e.g., `custom.toml`) is loaded, `omnidsp.toml` is **ignored**.
*   **Hardcoded Fallbacks:** If the loaded config is missing fields (e.g., defines `dft` but not `fir`), the library falls back to **internal defaults** (OmniProvider), NOT to values in `omnidsp.toml`.

## 4. The In-Memory Structure

```rust
pub struct OmniConfig {
    pub default_providers: Vec<String>,
    pub module_overrides: HashMap<String, Vec<String>>,
    pub provider_settings: HashMap<String, toml::Table>,
}
```

## 5. Validation
*   The `Config` struct itself is just data. It does not validate if "ipp" is actually installed.
*   Validation happens when the `OmniBackend` attempts to initialize the providers based on this config.
