use num_complex::Complex;

// Standard floating point types
pub type Real32 = f32;
pub type Real64 = f64;
// TODO: Add support for f16 (Half Precision) when the 'half' crate is integrated.
// pub type Real16 = half::f16; 

// Interleaved Complex types
pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;
// pub type Complex16 = Complex<Real16>;