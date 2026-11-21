// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! IPP-backed DFT primitives.
//!
//! Three peer factories — [`IppDftC2c`], [`IppDftR2c`], [`IppDftC2r`] — mirror
//! the real-DFT floor (`RustDftC2c` / `RustDftR2c` / `RustDftC2r`).  Each plan
//! owns a built IPP transform (see [`crate::ffi::Dft`]) that selects the
//! power-of-two FFT engine when it can and the arbitrary-length DFT otherwise —
//! the FFT is materially faster, and the switch is internal, exactly as a vendor
//! DFT auto-selects its radix path.
//!
//! IPP runs the transform *unnormalized* (the `IPP_FFT_NODIV_BY_ANY` flag), so
//! the [`DftNorm`] convention is applied in Rust by [`norm_scale`] — identical to
//! the floor, so module round-trips compose identically across backends.
//!
//! Real transforms use IPP's CCS packed format, which is bit-identical to
//! `OmniDSP`'s `N/2 + 1` interleaved-complex half-spectrum, so the complex
//! buffer is handed straight to IPP with no scratch or mirroring.  The c2r
//! factory is the *raw* inverse: the `impl_generic_backend!` macro wraps it in
//! the Hermitian shaping decorator, so it performs no DC/Nyquist projection here.

use std::marker::PhantomData;
use std::sync::Mutex;

use num_complex::Complex;

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
/// c2r like a c2c **inverse** plan.  IPP runs unnormalized, so this is the only
/// normalization applied.
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

/// IPP complex-to-complex DFT factory.
///
/// Unit struct — no state.  The plan owns the built IPP transform.
#[derive(Debug, Clone, Copy)]
pub struct IppDftC2c;

/// Execution plan for a complex-to-complex DFT backed by a built IPP transform.
///
/// The transform is held behind a [`Mutex`] (IPP mutates its work scratch on
/// every compute call); forward versus inverse is chosen at `execute` time from
/// [`direction`](Self::direction).  Normalization is applied in Rust.
pub struct IppDftC2cPlan<T> {
    inner: Mutex<ffi::Dft>,
    length: usize,
    direction: Direction,
    norm: DftNorm,
    _marker: PhantomData<T>,
}

impl<T: DspFloat> IppDftC2cPlan<T> {
    /// Build a complex IPP transform for `spec` at precision `T`.
    fn new(spec: &DftC2cSpec) -> Result<Self> {
        let length = spec.length();
        let dft = ffi::Dft::build_complex::<T>(length)?;
        Ok(Self {
            inner: Mutex::new(dft),
            length,
            direction: spec.direction(),
            norm: spec.norm(),
            _marker: PhantomData,
        })
    }
}

impl<T: DspFloat> DftC2cPlan<T> for IppDftC2cPlan<T> {
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

        let inverse = self.direction == Direction::Inverse;
        {
            let mut dft = self
                .inner
                .lock()
                .map_err(|e| Error::Internal(format!("IPP DFT task lock poisoned: {e}")))?;
            dft.exec_complex::<T>(input, output, inverse)?;
        }

        if let Some(scale) = norm_scale::<T>(self.norm, self.direction, self.length)? {
            scale_complex(output, scale);
        }
        Ok(())
    }
}

// ─── r2c: real → complex half-spectrum ───────────────────────────────

/// IPP real-to-complex DFT factory.
///
/// Forward-only; maps `length` real samples to a `length / 2 + 1` complex
/// half-spectrum via IPP's CCS packed format.
#[derive(Debug, Clone, Copy)]
pub struct IppDftR2c;

/// Execution plan for a real-to-complex DFT backed by a built IPP transform.
pub struct IppDftR2cPlan<T> {
    inner: Mutex<ffi::Dft>,
    length: usize,
    norm: DftNorm,
    _marker: PhantomData<T>,
}

impl<T: DspFloat> IppDftR2cPlan<T> {
    /// Build a real-domain IPP transform for `spec` at precision `T`.
    fn new(spec: &DftR2cSpec) -> Result<Self> {
        let length = spec.length();
        let dft = ffi::Dft::build_real::<T>(length)?;
        Ok(Self {
            inner: Mutex::new(dft),
            length,
            norm: spec.norm(),
            _marker: PhantomData,
        })
    }
}

impl<T: DspFloat> DftR2cPlan<T> for IppDftR2cPlan<T> {
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

        {
            let mut dft = self
                .inner
                .lock()
                .map_err(|e| Error::Internal(format!("IPP DFT task lock poisoned: {e}")))?;
            dft.exec_r2c::<T>(input, output)?;
        }

        // r2c is forward-only — it scales like a c2c forward plan.
        if let Some(scale) = norm_scale::<T>(self.norm, Direction::Forward, self.length)? {
            scale_complex(output, scale);
        }
        Ok(())
    }
}

// ─── c2r: complex half-spectrum → real ───────────────────────────────

/// IPP complex-to-real DFT factory.
///
/// Inverse-only; maps a `length / 2 + 1` complex half-spectrum back to `length`
/// real samples via IPP's CCS packed format.
///
/// This is the **raw** inverse — it performs no DC/Nyquist Hermitian
/// projection.  The `impl_generic_backend!` macro wraps it in the Hermitian
/// shaping decorator, which owns that projection.
#[derive(Debug, Clone, Copy)]
pub struct IppDftC2r;

/// Execution plan for a complex-to-real DFT backed by a built IPP transform.
pub struct IppDftC2rPlan<T> {
    inner: Mutex<ffi::Dft>,
    length: usize,
    norm: DftNorm,
    _marker: PhantomData<T>,
}

impl<T: DspFloat> IppDftC2rPlan<T> {
    /// Build a real-domain IPP transform for `spec` at precision `T`.
    fn new(spec: &DftC2rSpec) -> Result<Self> {
        let length = spec.length();
        let dft = ffi::Dft::build_real::<T>(length)?;
        Ok(Self {
            inner: Mutex::new(dft),
            length,
            norm: spec.norm(),
            _marker: PhantomData,
        })
    }
}

impl<T: DspFloat> DftC2rPlan<T> for IppDftC2rPlan<T> {
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

        {
            let mut dft = self
                .inner
                .lock()
                .map_err(|e| Error::Internal(format!("IPP DFT task lock poisoned: {e}")))?;
            dft.exec_c2r::<T>(input, output)?;
        }

        // c2r is inverse-only — it scales like a c2c inverse plan.
        if let Some(scale) = norm_scale::<T>(self.norm, Direction::Inverse, self.length)? {
            scale_real(output, scale);
        }
        Ok(())
    }
}

// ─── Factory impls (concrete float widths) ───────────────────────────

/// Implement the three DFT factory traits for one concrete float width.
///
/// The plan `new` / `execute` bodies are width-generic (precision is threaded
/// through `ffi::Dft`'s `TypeId` dispatch), so each width impl only names the
/// concrete plan type.
macro_rules! impl_ipp_dft {
    ($t:ty $(,)?) => {
        impl DftC2c<$t> for IppDftC2c {
            type Plan = IppDftC2cPlan<$t>;

            fn create_plan(&self, spec: &DftC2cSpec) -> Result<Self::Plan> {
                IppDftC2cPlan::new(spec)
            }
        }

        impl DftR2c<$t> for IppDftR2c {
            type Plan = IppDftR2cPlan<$t>;

            fn create_plan(&self, spec: &DftR2cSpec) -> Result<Self::Plan> {
                IppDftR2cPlan::new(spec)
            }
        }

        impl DftC2r<$t> for IppDftC2r {
            type Plan = IppDftC2rPlan<$t>;

            fn create_plan(&self, spec: &DftC2rSpec) -> Result<Self::Plan> {
                IppDftC2rPlan::new(spec)
            }
        }
    };
}

impl_ipp_dft!(f32);
impl_ipp_dft!(f64);
