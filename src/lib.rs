pub mod core;
pub mod traits;
pub mod backend;

// Re-export key items for easier access
pub use core::config::Config;
pub use core::error::OmniError;
pub use backend::Backend;