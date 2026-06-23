// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! oneMKL DFTI-backed DFT primitives.
//!
//! Three peer factories — [`OneMklDftC2c`], [`OneMklDftR2c`], [`OneMklDftC2r`]
//! — mirror the real-DFT floor (`RustDftC2c` / `RustDftR2c` / `RustDftC2r`),
//! each building a committed DFTI descriptor and returning a plan that executes
//! transforms out-of-place.
//!
//! DFTI is *unnormalized* by default: the descriptor's forward/backward scales
//! are left at `1.0` and the [`DftNorm`] convention is applied in Rust by
//! [`norm_scale`], exactly as the floor does, so module round-trips compose
//! identically across backends.
//!
//! Real transforms use `COMPLEX_COMPLEX` conjugate-even storage, which writes /
//! reads exactly `N/2 + 1` interleaved-complex bins — `OmniDSP`'s half-spectrum
//! convention — so no scratch buffer or half-spectrum mirroring is needed.  The
//! c2r factory is the *raw* inverse: the macro wraps it in the Hermitian
//! shaping decorator, so it performs no DC/Nyquist projection here.

use std::marker::PhantomData;

use num_complex::Complex;
use omnidsp_onemkl_sys::{DftiConfigValue, DftiDescriptorHandle};

use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::dft::{
    DftC2c, DftC2cPlan, DftC2cSpec, DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan,
    DftR2cSpec,
};
use omnidsp_core::types::{Direction, DspFloat};

use crate::ffi;

// ─── Normalization (shared, applied in Rust) ─────────────────────────

/// Scaling factor a [`DftNorm`] convention applies to a transform of real
/// length `n` running in `direction`, or `None` when no scaling is needed.
///
/// Replicates the floor's `norm_scale`: r2c scales like a c2c **forward** plan,
/// c2r like a c2c **inverse** plan.  DFTI's descriptor scales stay at `1.0`, so
/// this is the only normalization applied.
fn norm_scale<T: DspFloat>(norm: DftNorm, direction: Direction, n: usize) -> Result<Option<T>> {
    #[allow(
        clippy::cast_precision_loss,
        reason = "DFT lengths are well within f64 mantissa range"
    )]
    let nf = n as f64;

    let factor = match norm {
        DftNorm::None => return Ok(None),
        DftNorm::Inverse => match direction {
            Direction::Inverse => 1.0 / nf,
            Direction::Forward => return Ok(None),
        },
        DftNorm::Ortho => 1.0 / nf.sqrt(),
    };

    let scale = T::from_f64(factor)
        .ok_or_else(|| Error::backend(0, "failed to convert DFT scale factor"))?;
    Ok(Some(scale))
}

/// Apply a real scale factor to every element of a complex buffer in place.
fn scale_complex<T: DspFloat>(output: &mut [Complex<T>], scale: T) {
    for bin in output.iter_mut() {
        *bin = Complex::new(bin.re * scale, bin.im * scale);
    }
}

/// Apply a real scale to every element of a real buffer in place.
fn scale_real<T: DspFloat>(output: &mut [T], scale: T) {
    for sample in output.iter_mut() {
        *sample *= scale;
    }
}

// ─── c2c: complex → complex ──────────────────────────────────────────

/// oneMKL complex-to-complex DFT factory.
///
/// Unit struct — no state.  The plan owns the committed DFTI descriptor.
#[derive(Debug, Clone, Copy)]
pub struct OneMklDftC2c;

/// Execution plan for a complex-to-complex DFT backed by a committed DFTI
/// descriptor.
///
/// The descriptor is configured for the plan's length and out-of-place
/// placement; forward versus inverse is chosen at `execute` time from
/// [`direction`](Self::direction).  Normalization is applied in Rust.
pub struct OneMklDftC2cPlan<T> {
    handle: DftiDescriptorHandle,
    length: usize,
    direction: Direction,
    norm: DftNorm,
    _marker: PhantomData<T>,
}

impl<T> OneMklDftC2cPlan<T> {
    /// Build and commit a complex-domain DFTI descriptor for `spec`.
    fn new(spec: &DftC2cSpec, precision: DftiConfigValue) -> Result<Self> {
        let length = spec.length();
        let handle = ffi::create_descriptor(precision, DftiConfigValue::Complex, length)?;
        if let Err(e) = configure_c2c(handle) {
            ffi::free_descriptor(handle);
            return Err(e);
        }
        Ok(Self {
            handle,
            length,
            direction: spec.direction(),
            norm: spec.norm(),
            _marker: PhantomData,
        })
    }
}

/// Set out-of-place placement and commit a complex-domain descriptor.
fn configure_c2c(handle: DftiDescriptorHandle) -> Result<()> {
    ffi::set_placement_not_inplace(handle)?;
    ffi::commit(handle)
}

impl<T: DspFloat> DftC2cPlan<T> for OneMklDftC2cPlan<T> {
    fn execute(&self, input: &[Complex<T>], output: &mut [Complex<T>]) -> Result<()> {
        if input.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: input.len(),
            });
        }
        if output.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: output.len(),
            });
        }

        let in_ptr = ffi::complex_ptr(input);
        let out_ptr = ffi::complex_ptr_mut(output);
        match self.direction {
            Direction::Forward => ffi::compute_forward(self.handle, in_ptr, out_ptr)?,
            Direction::Inverse => ffi::compute_backward(self.handle, in_ptr, out_ptr)?,
        }

        if let Some(scale) = norm_scale::<T>(self.norm, self.direction, self.length)? {
            scale_complex(output, scale);
        }
        Ok(())
    }
}

impl<T> Drop for OneMklDftC2cPlan<T> {
    fn drop(&mut self) {
        ffi::free_descriptor(self.handle);
    }
}

// ─── r2c: real → complex half-spectrum ───────────────────────────────

/// oneMKL real-to-complex DFT factory.
///
/// Forward-only; maps `length` real samples to a `length / 2 + 1` complex
/// half-spectrum via a `COMPLEX_COMPLEX`-storage real descriptor.
#[derive(Debug, Clone, Copy)]
pub struct OneMklDftR2c;

/// Execution plan for a real-to-complex DFT backed by a committed real DFTI
/// descriptor with `COMPLEX_COMPLEX` conjugate-even storage.
pub struct OneMklDftR2cPlan<T> {
    handle: DftiDescriptorHandle,
    length: usize,
    norm: DftNorm,
    _marker: PhantomData<T>,
}

impl<T> OneMklDftR2cPlan<T> {
    /// Build and commit a real-domain DFTI descriptor for `spec`.
    fn new(spec: &DftR2cSpec, precision: DftiConfigValue) -> Result<Self> {
        let length = spec.length();
        let handle = ffi::create_descriptor(precision, DftiConfigValue::Real, length)?;
        if let Err(e) = configure_real(handle) {
            ffi::free_descriptor(handle);
            return Err(e);
        }
        Ok(Self {
            handle,
            length,
            norm: spec.norm(),
            _marker: PhantomData,
        })
    }
}

/// Set out-of-place placement, `COMPLEX_COMPLEX` storage, and commit a
/// real-domain descriptor (shared by r2c and c2r).
fn configure_real(handle: DftiDescriptorHandle) -> Result<()> {
    ffi::set_placement_not_inplace(handle)?;
    ffi::set_cce_complex_complex(handle)?;
    ffi::commit(handle)
}

impl<T: DspFloat> DftR2cPlan<T> for OneMklDftR2cPlan<T> {
    fn execute(&self, input: &mut [T], output: &mut [Complex<T>]) -> Result<()> {
        let bins = self.length / 2 + 1;
        if input.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: input.len(),
            });
        }
        if output.len() != bins {
            return Err(Error::BufferMismatch {
                expected: bins,
                actual: output.len(),
            });
        }

        let in_ptr = ffi::real_ptr(input);
        let out_ptr = ffi::complex_ptr_mut(output);
        ffi::compute_forward(self.handle, in_ptr, out_ptr)?;

        // r2c is forward-only — it scales like a c2c forward plan.
        if let Some(scale) = norm_scale::<T>(self.norm, Direction::Forward, self.length)? {
            scale_complex(output, scale);
        }
        Ok(())
    }
}

impl<T> Drop for OneMklDftR2cPlan<T> {
    fn drop(&mut self) {
        ffi::free_descriptor(self.handle);
    }
}

// ─── c2r: complex half-spectrum → real ───────────────────────────────

/// oneMKL complex-to-real DFT factory.
///
/// Inverse-only; maps a `length / 2 + 1` complex half-spectrum back to `length`
/// real samples via a `COMPLEX_COMPLEX`-storage real descriptor.
///
/// This is the **raw** inverse — it performs no DC/Nyquist Hermitian
/// projection.  The `impl_generic_backend!` macro wraps it in the Hermitian
/// shaping decorator, which owns that projection.
#[derive(Debug, Clone, Copy)]
pub struct OneMklDftC2r;

/// Execution plan for a complex-to-real DFT backed by a committed real DFTI
/// descriptor with `COMPLEX_COMPLEX` conjugate-even storage.
pub struct OneMklDftC2rPlan<T> {
    handle: DftiDescriptorHandle,
    length: usize,
    norm: DftNorm,
    _marker: PhantomData<T>,
}

impl<T> OneMklDftC2rPlan<T> {
    /// Build and commit a real-domain DFTI descriptor for `spec`.
    fn new(spec: &DftC2rSpec, precision: DftiConfigValue) -> Result<Self> {
        let length = spec.length();
        let handle = ffi::create_descriptor(precision, DftiConfigValue::Real, length)?;
        if let Err(e) = configure_real(handle) {
            ffi::free_descriptor(handle);
            return Err(e);
        }
        Ok(Self {
            handle,
            length,
            norm: spec.norm(),
            _marker: PhantomData,
        })
    }
}

impl<T: DspFloat> DftC2rPlan<T> for OneMklDftC2rPlan<T> {
    fn execute(&self, input: &mut [Complex<T>], output: &mut [T]) -> Result<()> {
        let bins = self.length / 2 + 1;
        if input.len() != bins {
            return Err(Error::BufferMismatch {
                expected: bins,
                actual: input.len(),
            });
        }
        if output.len() != self.length {
            return Err(Error::BufferMismatch {
                expected: self.length,
                actual: output.len(),
            });
        }

        let in_ptr = ffi::complex_ptr(input);
        let out_ptr = ffi::real_ptr_mut(output);
        ffi::compute_backward(self.handle, in_ptr, out_ptr)?;

        // c2r is inverse-only — it scales like a c2c inverse plan.
        if let Some(scale) = norm_scale::<T>(self.norm, Direction::Inverse, self.length)? {
            scale_real(output, scale);
        }
        Ok(())
    }
}

impl<T> Drop for OneMklDftC2rPlan<T> {
    fn drop(&mut self) {
        ffi::free_descriptor(self.handle);
    }
}

// ─── Factory impls (concrete float widths) ───────────────────────────

/// Implement the three DFT factory traits for one concrete float width.
///
/// The only per-width difference is the DFTI precision config value; the plan
/// `execute` bodies are width-generic because DFTI compute is type-agnostic
/// (the precision is baked into the committed descriptor).
macro_rules! impl_onemkl_dft {
    ($t:ty, $precision:expr $(,)?) => {
        impl DftC2c<$t> for OneMklDftC2c {
            type Plan = OneMklDftC2cPlan<$t>;

            fn create_plan(&self, spec: &DftC2cSpec) -> Result<Self::Plan> {
                OneMklDftC2cPlan::new(spec, $precision)
            }
        }

        impl DftR2c<$t> for OneMklDftR2c {
            type Plan = OneMklDftR2cPlan<$t>;

            fn create_plan(&self, spec: &DftR2cSpec) -> Result<Self::Plan> {
                OneMklDftR2cPlan::new(spec, $precision)
            }
        }

        impl DftC2r<$t> for OneMklDftC2r {
            type Plan = OneMklDftC2rPlan<$t>;

            fn create_plan(&self, spec: &DftC2rSpec) -> Result<Self::Plan> {
                OneMklDftC2rPlan::new(spec, $precision)
            }
        }
    };
}

impl_onemkl_dft!(f32, DftiConfigValue::Single);
impl_onemkl_dft!(f64, DftiConfigValue::Double);
