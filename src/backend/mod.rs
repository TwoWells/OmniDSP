pub mod omni;
pub mod ipp;
pub mod accelerate;

use crate::core::config::Config;
use crate::traits::dft::Dft;
use num_complex::Complex;

/// The Manager that holds active implementations.
pub struct Backend {
    // In a real implementation, these would be dynamic dispatch boxes
    // capable of handling multiple types. For now, we'll just show the structure.
    // We might need a "Registry" or just hardcoded fields for common types.
    
    // Simplification: For this stage, let's assume we have a way to hold
    // generic factories. Since we can't have `Box<dyn Dft<T>>` without knowing T,
    // the Backend usually holds specific boxes for f32/c32/etc, OR the Factory itself is generic-erased?
    
    // Actually, the Factory (Dft<T>) is typed. 
    // The Backend typically implements Dft<T> itself and delegates.
    // So it might hold:
    // dft_c32: Box<dyn Dft<Complex<f32>>>,
    // dft_c64: Box<dyn Dft<Complex<f64>>>,
    
    config: Config,
}

impl Backend {
    pub fn new(config: Config) -> Self {
        // Logic to load providers based on config
        Self { config }
    }
}
