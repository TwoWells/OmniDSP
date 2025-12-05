use thiserror::Error;

#[derive(Error, Debug)]
pub enum OmniError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Invalid specification: {0}")]
    InvalidSpec(String),

    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("No available backend found for module: {0}")]
    NoBackendAvailable(String),

    #[error("Backend initialization failed: {0}")]
    BackendInitFailed(String),

    #[error("Buffer size mismatch: expected {expected}, actual {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, OmniError>;