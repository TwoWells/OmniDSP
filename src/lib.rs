pub mod core;
pub mod traits;
pub mod backends;
pub mod context;
pub mod capi;

// Re-export key items for easier access
pub use core::config::Config;
pub use core::error::OmniError;
pub use context::Context;
