// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Safe wrappers over the raw `omnidsp-ipp-sys` FFI surface.
//!
//! This is the **only** module in the crate that contains `unsafe` code: every
//! FFI call, every pointer cast, and the `unsafe impl Send`/`Sync` for the plan
//! types live here, each guarded by a safety comment.  The DFT and vector-ops
//! modules call exclusively into these wrappers and stay entirely unsafe-free.
//!
//! Two pieces of machinery dominate:
//!
//! - **Precision dispatch.** IPP's entry points are per-width (`*_32f` /
//!   `*_64f`, `*_32fc` / `*_64fc`), but the plan types are generic over a
//!   `T: DspFloat`.  Each wrapper branches on [`TypeId`] and reinterprets the
//!   slice pointers to the matching IPP layout (`Complex<f32>` ⇄ `[f32; 2]` is a
//!   `#[repr(C)]` layout identity).  Anything other than `f32` / `f64` is an
//!   unreachable backend error.
//! - **The [`Dft`] engine.** A built transform owns its IPP spec and work
//!   buffers and remembers whether it uses the power-of-two FFT engine or the
//!   arbitrary-length DFT engine.  IPP writes scratch into the work buffer on
//!   every compute call, so the owning plan keeps the whole [`Dft`] behind a
//!   `Mutex` and the marker traits are asserted here.

#![allow(
    unsafe_code,
    reason = "FFI calls to omnidsp-ipp-sys: this is the single isolated unsafe module"
)]

use std::any::TypeId;
use std::os::raw::{c_int, c_void};
use std::ptr;

use num_complex::Complex;
use omnidsp_ipp_sys as sys;
use omnidsp_ipp_sys::{IPP_ALG_HINT_NONE, IPP_FFT_NODIV_BY_ANY, IPP_STS_NO_ERR, IppStatus};

use omnidsp_core::error::{Error, Result};
use omnidsp_core::types::DspFloat;

use crate::dft::{IppDftC2cPlan, IppDftC2rPlan, IppDftR2cPlan};

// ─── Status / conversion helpers ─────────────────────────────────────

/// Map an IPP status code to a [`Result`]; `ippStsNoErr` (0) is success.
fn check_ipp(status: IppStatus, context: &'static str) -> Result<()> {
    if status == IPP_STS_NO_ERR {
        Ok(())
    } else {
        Err(Error::backend(status, context))
    }
}

/// Convert a `usize` length to the `c_int` IPP entry points take, mapping
/// overflow to a backend error rather than truncating.
fn to_ipp_int(len: usize, context: &'static str) -> Result<c_int> {
    c_int::try_from(len).map_err(|_| Error::backend(-1, context))
}

/// Fallible `usize → c_int` for infallible call sites (`scale`), which fall back
/// to a scalar loop on the astronomically unlikely overflow.
#[must_use]
pub fn try_ipp_int(len: usize) -> Option<c_int> {
    c_int::try_from(len).ok()
}

/// `true` when `T` is `f32`.
fn is_f32<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// `true` when `T` is `f64`.
fn is_f64<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// The error returned when a generic `T` is neither `f32` nor `f64` (unreachable
/// in practice — the factory impls only exist for those two widths).
fn unsupported_width() -> Error {
    Error::backend(-1, "IPP backend supports only f32 and f64")
}

// ─── Complex-slice layout casts (Complex<T> ⇄ [T; 2]) ────────────────

/// Cast a complex slice to IPP's `*const [f32; 2]` (`Ipp32fc`) layout.
const fn cplx32(slice: &[Complex<f32>]) -> *const [f32; 2] {
    slice.as_ptr().cast::<[f32; 2]>()
}

/// Cast a complex slice to IPP's `*const [f64; 2]` (`Ipp64fc`) layout.
const fn cplx64(slice: &[Complex<f64>]) -> *const [f64; 2] {
    slice.as_ptr().cast::<[f64; 2]>()
}

/// Cast a mutable complex slice to IPP's `*mut [f32; 2]` layout.
const fn cplx32_mut(slice: &mut [Complex<f32>]) -> *mut [f32; 2] {
    slice.as_mut_ptr().cast::<[f32; 2]>()
}

/// Cast a mutable complex slice to IPP's `*mut [f64; 2]` layout.
const fn cplx64_mut(slice: &mut [Complex<f64>]) -> *mut [f64; 2] {
    slice.as_mut_ptr().cast::<[f64; 2]>()
}

// ─── VecOps wrappers (VM-style elementwise) ──────────────────────────

macro_rules! real_binop {
    ($name:ident, $sys_fn:ident, $float:ty $(,)?) => {
        #[doc = concat!("Elementwise `", stringify!($sys_fn), "`: `out = a OP b`.")]
        pub fn $name(a: &[$float], b: &[$float], out: &mut [$float]) -> Result<()> {
            let n = to_ipp_int(
                out.len(),
                concat!(stringify!($sys_fn), " length exceeds IPP range"),
            )?;
            // SAFETY: `a`, `b`, `out` all have `out.len()` elements (length
            // equality validated by the caller); `n` is that length as c_int.
            let status =
                unsafe { sys::vecops::$sys_fn(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n) };
            check_ipp(status, stringify!($sys_fn))
        }
    };
}

real_binop!(vmul_f32, ippsMul_32f, f32);
real_binop!(vmul_f64, ippsMul_64f, f64);
real_binop!(vadd_f32, ippsAdd_32f, f32);
real_binop!(vadd_f64, ippsAdd_64f, f64);

macro_rules! real_binop_inplace {
    ($name:ident, $sys_fn:ident, $float:ty $(,)?) => {
        #[doc = concat!("In-place elementwise `", stringify!($sys_fn), "`: `data OP= other`.")]
        pub fn $name(data: &mut [$float], other: &[$float]) -> Result<()> {
            let n = to_ipp_int(
                data.len(),
                concat!(stringify!($sys_fn), " length exceeds IPP range"),
            )?;
            // SAFETY: `other` and `data` both have `data.len()` elements
            // (validated by the caller). IPP's `_I` form is `(pSrc, pSrcDst,
            // len)`: `src_dst OP= src`.
            let status = unsafe { sys::vecops::$sys_fn(other.as_ptr(), data.as_mut_ptr(), n) };
            check_ipp(status, stringify!($sys_fn))
        }
    };
}

real_binop_inplace!(mul_inplace_f32, ippsMul_32f_I, f32);
real_binop_inplace!(mul_inplace_f64, ippsMul_64f_I, f64);
real_binop_inplace!(add_inplace_f32, ippsAdd_32f_I, f32);
real_binop_inplace!(add_inplace_f64, ippsAdd_64f_I, f64);

macro_rules! real_dot {
    ($name:ident, $sys_fn:ident, $float:ty $(,)?) => {
        #[doc = concat!("Real dot product via `", stringify!($sys_fn), "`.")]
        pub fn $name(a: &[$float], b: &[$float]) -> Result<$float> {
            let n = to_ipp_int(
                a.len(),
                concat!(stringify!($sys_fn), " length exceeds IPP range"),
            )?;
            let mut result: $float = 0.0;
            // SAFETY: `a` and `b` both have `a.len()` elements (validated by the
            // caller); `result` is a valid out-pointer for the scalar product.
            let status =
                unsafe { sys::vecops::$sys_fn(a.as_ptr(), b.as_ptr(), n, &raw mut result) };
            check_ipp(status, stringify!($sys_fn))?;
            Ok(result)
        }
    };
}

real_dot!(dot_f32, ippsDotProd_32f, f32);
real_dot!(dot_f64, ippsDotProd_64f, f64);

macro_rules! real_scale {
    ($name:ident, $sys_fn:ident, $float:ty $(,)?) => {
        #[doc = concat!("In-place scale by a constant via `", stringify!($sys_fn), "`.")]
        pub fn $name(data: &mut [$float], scalar: $float, n: c_int) {
            // SAFETY: the caller passes `n == data.len() as c_int`, so `data`
            // has `n` elements. The status (success for valid args) is ignored
            // because `scale` is infallible at the trait level.
            unsafe {
                sys::vecops::$sys_fn(scalar, data.as_mut_ptr(), n);
            }
        }
    };
}

real_scale!(scale_f32, ippsMulC_32f_I, f32);
real_scale!(scale_f64, ippsMulC_64f_I, f64);

macro_rules! complex_binop {
    ($name:ident, $sys_fn:ident, $float:ty, $cast:ident, $cast_mut:ident $(,)?) => {
        #[doc = concat!("Elementwise complex `", stringify!($sys_fn), "`: `out = a * b`.")]
        pub fn $name(
            a: &[Complex<$float>],
            b: &[Complex<$float>],
            out: &mut [Complex<$float>],
        ) -> Result<()> {
            let n = to_ipp_int(
                out.len(),
                concat!(stringify!($sys_fn), " length exceeds IPP range"),
            )?;
            // SAFETY: `a`, `b`, `out` have `out.len()` complex elements
            // (validated by the caller); the `[$float; 2]` casts are sound
            // because `Complex<$float>` is `#[repr(C)]` with the same layout.
            let status = unsafe { sys::vecops::$sys_fn($cast(a), $cast(b), $cast_mut(out), n) };
            check_ipp(status, stringify!($sys_fn))
        }
    };
}

complex_binop!(cmul_f32, ippsMul_32fc, f32, cplx32, cplx32_mut);
complex_binop!(cmul_f64, ippsMul_64fc, f64, cplx64, cplx64_mut);

macro_rules! complex_binop_inplace {
    ($name:ident, $sys_fn:ident, $float:ty, $cast:ident, $cast_mut:ident $(,)?) => {
        #[doc = concat!("In-place complex `", stringify!($sys_fn), "`: `data *= other`.")]
        pub fn $name(data: &mut [Complex<$float>], other: &[Complex<$float>]) -> Result<()> {
            let n = to_ipp_int(
                data.len(),
                concat!(stringify!($sys_fn), " length exceeds IPP range"),
            )?;
            // SAFETY: `other` and `data` have `data.len()` complex elements
            // (validated by the caller); the `[$float; 2]` casts are layout
            // identities. IPP's `_I` form is `(pSrc, pSrcDst, len)`.
            let status = unsafe { sys::vecops::$sys_fn($cast(other), $cast_mut(data), n) };
            check_ipp(status, stringify!($sys_fn))
        }
    };
}

complex_binop_inplace!(cmul_inplace_f32, ippsMul_32fc_I, f32, cplx32, cplx32_mut);
complex_binop_inplace!(cmul_inplace_f64, ippsMul_64fc_I, f64, cplx64, cplx64_mut);

macro_rules! complex_dot {
    ($name:ident, $sys_fn:ident, $float:ty, $cast:ident $(,)?) => {
        #[doc = concat!("Unconjugated complex dot product via `", stringify!($sys_fn), "`.")]
        pub fn $name(a: &[Complex<$float>], b: &[Complex<$float>]) -> Result<Complex<$float>> {
            let n = to_ipp_int(
                a.len(),
                concat!(stringify!($sys_fn), " length exceeds IPP range"),
            )?;
            let mut result: [$float; 2] = [0.0, 0.0];
            // SAFETY: `a` and `b` have `a.len()` complex elements (validated by
            // the caller); `result` is a valid `[$float; 2]` out-pointer.
            let status = unsafe { sys::vecops::$sys_fn($cast(a), $cast(b), n, &raw mut result) };
            check_ipp(status, stringify!($sys_fn))?;
            Ok(Complex::new(result[0], result[1]))
        }
    };
}

complex_dot!(cdot_f32, ippsDotProd_32fc, f32, cplx32);
complex_dot!(cdot_f64, ippsDotProd_64fc, f64, cplx64);

macro_rules! complex_conj {
    ($name:ident, $sys_fn:ident, $float:ty, $cast_mut:ident $(,)?) => {
        #[doc = concat!("In-place complex conjugate via `", stringify!($sys_fn), "`.")]
        pub fn $name(data: &mut [Complex<$float>]) -> Result<()> {
            let n = to_ipp_int(
                data.len(),
                concat!(stringify!($sys_fn), " length exceeds IPP range"),
            )?;
            // SAFETY: `data` has `data.len()` complex elements (validated by the
            // caller); the `[$float; 2]` cast is a layout identity.
            let status = unsafe { sys::vecops::$sys_fn($cast_mut(data), n) };
            check_ipp(status, stringify!($sys_fn))
        }
    };
}

complex_conj!(conj_inplace_f32, ippsConj_32fc_I, f32, cplx32_mut);
complex_conj!(conj_inplace_f64, ippsConj_64fc_I, f64, cplx64_mut);

// ─── DFT / FFT engine ────────────────────────────────────────────────

/// FFT normalization flag: compute the raw transform; the [`DftNorm`]
/// convention is applied in Rust (matching the floor).
///
/// [`DftNorm`]: omnidsp_core::traits::dft::DftNorm
const FLAG: c_int = IPP_FFT_NODIV_BY_ANY;

/// Algorithm hint — deprecated for the signal-processing domain, always "none".
const HINT: c_int = IPP_ALG_HINT_NONE;

/// IPP picks the materially faster FFT engine for power-of-two lengths and the
/// arbitrary-length DFT otherwise. Length 1 routes to the DFT (FFT order 0 is
/// rejected by IPP).
const fn use_fft(length: usize) -> bool {
    length.is_power_of_two() && length > 1
}

/// FFT order (`log2(length)`) for a power-of-two length; meaningless and unused
/// when [`use_fft`] is false.
fn order_of(length: usize) -> c_int {
    c_int::try_from(length.trailing_zeros()).unwrap_or(0)
}

/// Allocate the three IPP buffers from the sizes `GetSize` returned (a negative
/// size — never produced on success — clamps to empty).
fn alloc_buffers(
    spec_size: c_int,
    init_size: c_int,
    work_size: c_int,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let spec = vec![0u8; usize::try_from(spec_size).unwrap_or(0)];
    let init = vec![0u8; usize::try_from(init_size).unwrap_or(0)];
    let work = vec![0u8; usize::try_from(work_size).unwrap_or(0)];
    (spec, init, work)
}

/// A null-or-data pointer for an IPP byte buffer (IPP tolerates null when the
/// corresponding size is zero).
const fn buf_ptr(buf: &mut [u8]) -> *mut u8 {
    if buf.is_empty() {
        ptr::null_mut()
    } else {
        buf.as_mut_ptr()
    }
}

/// A built IPP transform — FFT (power-of-two) or DFT (arbitrary length), for one
/// precision and domain.
///
/// Owns its spec and work buffers. The work buffer is mutated on every compute
/// call, so the owning plan holds the whole `Dft` behind a `Mutex`; the marker
/// traits for the plan types are asserted at the bottom of this module.
///
/// The struct does not encode the precision `T` — the owning plan is monomorphic
/// in `T` and always calls `build_*::<T>` and `exec_*::<T>` with the same `T`,
/// so the spec built for one width is only ever executed at that width.
pub struct Dft {
    /// `true` when the power-of-two FFT engine is used; `false` for DFT.
    fft: bool,
    /// The spec allocation; the IPP spec lives in this buffer.
    ///
    /// Never read directly — it owns the memory `spec_ptr` references and is
    /// kept alive (and freed on `Drop`) for the plan's lifetime.
    #[allow(
        dead_code,
        reason = "RAII keep-alive: owns the allocation spec_ptr references; freed on Drop"
    )]
    spec: Vec<u8>,
    /// The initialized spec pointer. For FFT, the pointer IPP wrote into `spec`
    /// during `Init` (possibly offset within it); for DFT, `spec.as_mut_ptr()`.
    /// Stays valid because `spec` is kept alive and a `Vec`'s heap allocation is
    /// stable across moves of the `Vec` header.
    spec_ptr: *mut c_void,
    /// Reusable work scratch passed to every `Fwd`/`Inv` call.
    work: Vec<u8>,
}

impl Dft {
    /// Build a complex-to-complex transform of `length` points at width `T`.
    pub fn build_complex<T: DspFloat>(length: usize) -> Result<Self> {
        let fft = use_fft(length);
        let order = order_of(length);
        let len = to_ipp_int(length, "DFT length exceeds IPP c_int range")?;

        let mut spec_size: c_int = 0;
        let mut init_size: c_int = 0;
        let mut work_size: c_int = 0;
        let sizes = (&raw mut spec_size, &raw mut init_size, &raw mut work_size);

        // SAFETY: the size out-pointers are valid; `order`/`len` are valid for
        // the selected engine. Dispatch picks the matching-width entry point.
        let status = unsafe {
            if is_f32::<T>() {
                if fft {
                    sys::fft::ippsFFTGetSize_C_32fc(order, FLAG, HINT, sizes.0, sizes.1, sizes.2)
                } else {
                    sys::fft::ippsDFTGetSize_C_32fc(len, FLAG, HINT, sizes.0, sizes.1, sizes.2)
                }
            } else if is_f64::<T>() {
                if fft {
                    sys::fft::ippsFFTGetSize_C_64fc(order, FLAG, HINT, sizes.0, sizes.1, sizes.2)
                } else {
                    sys::fft::ippsDFTGetSize_C_64fc(len, FLAG, HINT, sizes.0, sizes.1, sizes.2)
                }
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP complex transform GetSize")?;

        let (mut spec, mut init, work) = alloc_buffers(spec_size, init_size, work_size);
        let mut spec_ptr: *mut c_void = ptr::null_mut();
        let init_ptr = buf_ptr(&mut init);

        // SAFETY: `spec`/`init` are sized per `GetSize`; FFT `Init` writes the
        // spec pointer into `spec_ptr`, DFT `Init` uses `spec` as the spec.
        let status = unsafe {
            if is_f32::<T>() {
                if fft {
                    sys::fft::ippsFFTInit_C_32fc(
                        &raw mut spec_ptr,
                        order,
                        FLAG,
                        HINT,
                        spec.as_mut_ptr(),
                        init_ptr,
                    )
                } else {
                    spec_ptr = spec.as_mut_ptr().cast::<c_void>();
                    sys::fft::ippsDFTInit_C_32fc(len, FLAG, HINT, spec_ptr, init_ptr)
                }
            } else if fft {
                sys::fft::ippsFFTInit_C_64fc(
                    &raw mut spec_ptr,
                    order,
                    FLAG,
                    HINT,
                    spec.as_mut_ptr(),
                    init_ptr,
                )
            } else {
                spec_ptr = spec.as_mut_ptr().cast::<c_void>();
                sys::fft::ippsDFTInit_C_64fc(len, FLAG, HINT, spec_ptr, init_ptr)
            }
        };
        check_ipp(status, "IPP complex transform Init")?;

        Ok(Self {
            fft,
            spec,
            spec_ptr,
            work,
        })
    }

    /// Build a real-domain transform of real length `length` at width `T`.
    ///
    /// One spec serves both the r2c (`RToCCS`) and c2r (`CCSToR`) directions.
    pub fn build_real<T: DspFloat>(length: usize) -> Result<Self> {
        let fft = use_fft(length);
        let order = order_of(length);
        let len = to_ipp_int(length, "DFT length exceeds IPP c_int range")?;

        let mut spec_size: c_int = 0;
        let mut init_size: c_int = 0;
        let mut work_size: c_int = 0;
        let sizes = (&raw mut spec_size, &raw mut init_size, &raw mut work_size);

        // SAFETY: as in `build_complex`, but the real-domain (`_R_`) entry points.
        let status = unsafe {
            if is_f32::<T>() {
                if fft {
                    sys::fft::ippsFFTGetSize_R_32f(order, FLAG, HINT, sizes.0, sizes.1, sizes.2)
                } else {
                    sys::fft::ippsDFTGetSize_R_32f(len, FLAG, HINT, sizes.0, sizes.1, sizes.2)
                }
            } else if is_f64::<T>() {
                if fft {
                    sys::fft::ippsFFTGetSize_R_64f(order, FLAG, HINT, sizes.0, sizes.1, sizes.2)
                } else {
                    sys::fft::ippsDFTGetSize_R_64f(len, FLAG, HINT, sizes.0, sizes.1, sizes.2)
                }
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP real transform GetSize")?;

        let (mut spec, mut init, work) = alloc_buffers(spec_size, init_size, work_size);
        let mut spec_ptr: *mut c_void = ptr::null_mut();
        let init_ptr = buf_ptr(&mut init);

        // SAFETY: as in `build_complex`, real-domain `Init`.
        let status = unsafe {
            if is_f32::<T>() {
                if fft {
                    sys::fft::ippsFFTInit_R_32f(
                        &raw mut spec_ptr,
                        order,
                        FLAG,
                        HINT,
                        spec.as_mut_ptr(),
                        init_ptr,
                    )
                } else {
                    spec_ptr = spec.as_mut_ptr().cast::<c_void>();
                    sys::fft::ippsDFTInit_R_32f(len, FLAG, HINT, spec_ptr, init_ptr)
                }
            } else if fft {
                sys::fft::ippsFFTInit_R_64f(
                    &raw mut spec_ptr,
                    order,
                    FLAG,
                    HINT,
                    spec.as_mut_ptr(),
                    init_ptr,
                )
            } else {
                spec_ptr = spec.as_mut_ptr().cast::<c_void>();
                sys::fft::ippsDFTInit_R_64f(len, FLAG, HINT, spec_ptr, init_ptr)
            }
        };
        check_ipp(status, "IPP real transform Init")?;

        Ok(Self {
            fft,
            spec,
            spec_ptr,
            work,
        })
    }

    /// Execute a complex-to-complex transform (`inverse` selects `Inv`).
    pub fn exec_complex<T: DspFloat>(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        inverse: bool,
    ) -> Result<()> {
        let spec = self.spec_ptr.cast_const();
        let work = buf_ptr(&mut self.work);

        // SAFETY: `input`/`output` are the plan's length (validated by the
        // caller); the `[$float; 2]` casts are layout identities; `spec`/`work`
        // belong to this `Dft` and outlive the call. The engine/precision branch
        // picks the entry point matching how the spec was built.
        let status = unsafe {
            if is_f32::<T>() {
                let src = input.as_ptr().cast::<[f32; 2]>();
                let dst = output.as_mut_ptr().cast::<[f32; 2]>();
                match (self.fft, inverse) {
                    (true, false) => sys::fft::ippsFFTFwd_CToC_32fc(src, dst, spec, work),
                    (true, true) => sys::fft::ippsFFTInv_CToC_32fc(src, dst, spec, work),
                    (false, false) => sys::fft::ippsDFTFwd_CToC_32fc(src, dst, spec, work),
                    (false, true) => sys::fft::ippsDFTInv_CToC_32fc(src, dst, spec, work),
                }
            } else if is_f64::<T>() {
                let src = input.as_ptr().cast::<[f64; 2]>();
                let dst = output.as_mut_ptr().cast::<[f64; 2]>();
                match (self.fft, inverse) {
                    (true, false) => sys::fft::ippsFFTFwd_CToC_64fc(src, dst, spec, work),
                    (true, true) => sys::fft::ippsFFTInv_CToC_64fc(src, dst, spec, work),
                    (false, false) => sys::fft::ippsDFTFwd_CToC_64fc(src, dst, spec, work),
                    (false, true) => sys::fft::ippsDFTInv_CToC_64fc(src, dst, spec, work),
                }
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP complex transform exec")
    }

    /// Execute a forward real-to-complex transform; `output` receives the
    /// CCS-packed half-spectrum (bit-identical to `N/2 + 1` `Complex<T>` bins).
    pub fn exec_r2c<T: DspFloat>(&mut self, input: &[T], output: &mut [Complex<T>]) -> Result<()> {
        let spec = self.spec_ptr.cast_const();
        let work = buf_ptr(&mut self.work);

        // SAFETY: `input` has the real length and `output` the `N/2+1` bins
        // (validated by the caller); the complex output reinterprets to the
        // CCS real destination IPP writes. Engine/precision branch as built.
        let status = unsafe {
            if is_f32::<T>() {
                let src = input.as_ptr().cast::<f32>();
                let dst = output.as_mut_ptr().cast::<f32>();
                if self.fft {
                    sys::fft::ippsFFTFwd_RToCCS_32f(src, dst, spec, work)
                } else {
                    sys::fft::ippsDFTFwd_RToCCS_32f(src, dst, spec, work)
                }
            } else if is_f64::<T>() {
                let src = input.as_ptr().cast::<f64>();
                let dst = output.as_mut_ptr().cast::<f64>();
                if self.fft {
                    sys::fft::ippsFFTFwd_RToCCS_64f(src, dst, spec, work)
                } else {
                    sys::fft::ippsDFTFwd_RToCCS_64f(src, dst, spec, work)
                }
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP r2c transform exec")
    }

    /// Execute an inverse complex-to-real transform; `input` is the CCS-packed
    /// half-spectrum, `output` the `length` real samples.
    pub fn exec_c2r<T: DspFloat>(&mut self, input: &[Complex<T>], output: &mut [T]) -> Result<()> {
        let spec = self.spec_ptr.cast_const();
        let work = buf_ptr(&mut self.work);

        // SAFETY: `input` has the `N/2+1` bins (CCS source) and `output` the
        // real length (validated by the caller). Engine/precision branch as built.
        let status = unsafe {
            if is_f32::<T>() {
                let src = input.as_ptr().cast::<f32>();
                let dst = output.as_mut_ptr().cast::<f32>();
                if self.fft {
                    sys::fft::ippsFFTInv_CCSToR_32f(src, dst, spec, work)
                } else {
                    sys::fft::ippsDFTInv_CCSToR_32f(src, dst, spec, work)
                }
            } else if is_f64::<T>() {
                let src = input.as_ptr().cast::<f64>();
                let dst = output.as_mut_ptr().cast::<f64>();
                if self.fft {
                    sys::fft::ippsFFTInv_CCSToR_64f(src, dst, spec, work)
                } else {
                    sys::fft::ippsDFTInv_CCSToR_64f(src, dst, spec, work)
                }
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP c2r transform exec")
    }
}

// ─── IIR biquad-cascade engine ───────────────────────────────────────

/// A built IPP biquad-cascade IIR filter state, for one precision.
///
/// Owns the IPP state buffer; the cascade coefficients and the running delay
/// line both live inside it. IPP advances that delay line on every [`apply`]
/// call, so the owning processor takes `&mut self` for execution (there is no
/// `Mutex`, unlike the read-only DFT plans).
///
/// The struct does not encode the precision `T` — the owning processor is
/// monomorphic in `T` and always calls `build::<T>` / `apply::<T>` / … with the
/// same `T`, so a state built for one width is only ever driven at that width.
///
/// [`apply`]: Iir::apply
pub struct Iir {
    /// The state allocation; IPP's biquad state (coefficients + delay line) lives
    /// in this buffer.  It owns the memory `state_ptr` references — kept alive
    /// (and freed on `Drop`) for the processor's lifetime — and is re-initialized
    /// in place by [`re_init`](Iir::re_init) when retuning or resetting.
    state: Vec<u8>,
    /// The initialized state pointer `Init` returned (an offset within `state`).
    /// Stays valid because `state` is kept alive and a `Vec`'s heap allocation is
    /// stable across moves of the `Vec` header.
    state_ptr: *mut c_void,
    /// Number of biquad sections — the delay line is `2 * num_sections` values.
    num_sections: usize,
}

impl Iir {
    /// Initialize (or re-initialize) a biquad-cascade state into `state` from the
    /// flat `taps` (`{b0, b1, b2, a0, a1, a2}` per section, `a0 = 1`) and an
    /// optional delay line (`2 * num_sections` values; `None` ⇒ a zeroed delay
    /// line). Returns the state pointer IPP produced (an offset within `state`).
    fn init<T: DspFloat>(
        state: &mut [u8],
        taps: &[T],
        num_sections: usize,
        dly_line: Option<&[T]>,
    ) -> Result<*mut c_void> {
        let num_bq = to_ipp_int(num_sections, "IIR section count exceeds IPP c_int range")?;
        let mut state_ptr: *mut c_void = ptr::null_mut();

        // SAFETY: `state` is sized per `GetStateSize` for `num_bq` sections; `taps`
        // holds `6 * num_sections` values and `dly_line` (when `Some`) holds
        // `2 * num_sections`, matching what IPP reads. The `Init` double-pointer
        // form writes the state pointer into `state_ptr`. The width branch picks
        // the matching-width entry point.
        let status = unsafe {
            if is_f32::<T>() {
                let taps_ptr = taps.as_ptr().cast::<f32>();
                let dly_ptr = dly_line.map_or(ptr::null(), |d| d.as_ptr().cast::<f32>());
                sys::iir::ippsIIRInit_BiQuad_32f(
                    &raw mut state_ptr,
                    taps_ptr,
                    num_bq,
                    dly_ptr,
                    state.as_mut_ptr(),
                )
            } else if is_f64::<T>() {
                let taps_ptr = taps.as_ptr().cast::<f64>();
                let dly_ptr = dly_line.map_or(ptr::null(), |d| d.as_ptr().cast::<f64>());
                sys::iir::ippsIIRInit_BiQuad_64f(
                    &raw mut state_ptr,
                    taps_ptr,
                    num_bq,
                    dly_ptr,
                    state.as_mut_ptr(),
                )
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP IIR Init")?;
        Ok(state_ptr)
    }

    /// Build a biquad cascade of `num_sections` sections from the flat `taps`
    /// (`6 * num_sections` values), with a zeroed delay line, at width `T`.
    pub fn build<T: DspFloat>(taps: &[T], num_sections: usize) -> Result<Self> {
        let num_bq = to_ipp_int(num_sections, "IIR section count exceeds IPP c_int range")?;

        let mut state_size: c_int = 0;
        // SAFETY: the size out-pointer is valid; `num_bq` is the section count.
        let status = unsafe {
            if is_f32::<T>() {
                sys::iir::ippsIIRGetStateSize_BiQuad_32f(num_bq, &raw mut state_size)
            } else if is_f64::<T>() {
                sys::iir::ippsIIRGetStateSize_BiQuad_64f(num_bq, &raw mut state_size)
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP IIR GetStateSize")?;

        let mut state = vec![0u8; usize::try_from(state_size).unwrap_or(0)];
        let state_ptr = Self::init::<T>(&mut state, taps, num_sections, None)?;
        Ok(Self {
            state,
            state_ptr,
            num_sections,
        })
    }

    /// Number of biquad sections this state was built for.
    pub const fn num_sections(&self) -> usize {
        self.num_sections
    }

    /// Apply the cascade to a block, advancing the internal delay line so
    /// successive calls stream continuously.
    pub fn apply<T: DspFloat>(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        let len = to_ipp_int(input.len(), "IIR length exceeds IPP c_int range")?;

        // SAFETY: `input`/`output` have `len` elements (validated by the caller);
        // `state_ptr` belongs to this `Iir` and is valid for the call. The width
        // branch picks the entry point matching how the state was built.
        let status = unsafe {
            if is_f32::<T>() {
                let src = input.as_ptr().cast::<f32>();
                let dst = output.as_mut_ptr().cast::<f32>();
                sys::iir::ippsIIR_32f(src, dst, len, self.state_ptr)
            } else if is_f64::<T>() {
                let src = input.as_ptr().cast::<f64>();
                let dst = output.as_mut_ptr().cast::<f64>();
                sys::iir::ippsIIR_64f(src, dst, len, self.state_ptr)
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP IIR exec")
    }

    /// Read the running delay line (`2 * num_sections` values) into `dly`.
    pub fn get_dly_line<T: DspFloat>(&self, dly: &mut [T]) -> Result<()> {
        // SAFETY: `dly` holds `2 * num_sections` elements (the caller sizes it);
        // `state_ptr` is this `Iir`'s state, read (not mutated) here. Width branch.
        let status = unsafe {
            if is_f32::<T>() {
                sys::iir::ippsIIRGetDlyLine_32f(
                    self.state_ptr.cast_const(),
                    dly.as_mut_ptr().cast::<f32>(),
                )
            } else if is_f64::<T>() {
                sys::iir::ippsIIRGetDlyLine_64f(
                    self.state_ptr.cast_const(),
                    dly.as_mut_ptr().cast::<f64>(),
                )
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP IIR GetDlyLine")
    }

    /// Re-initialize the cascade in place with new `taps`, optionally carrying an
    /// existing delay line over (`Some`) or zeroing it (`None`). The section
    /// count is unchanged, so the existing state buffer is reused.
    pub fn re_init<T: DspFloat>(&mut self, taps: &[T], dly_line: Option<&[T]>) -> Result<()> {
        self.state_ptr = Self::init::<T>(&mut self.state, taps, self.num_sections, dly_line)?;
        Ok(())
    }
}

// ─── Convolution engine ──────────────────────────────────────────────

/// A built IPP linear-convolution engine, for one precision and algorithm.
///
/// Owns the external work buffer IPP needs for its internal calculations (sized
/// by `ippsConvolveGetBufferSize`). IPP writes scratch into that buffer on every
/// [`exec`](Conv::exec) call, so the owning plan holds the whole `Conv` behind a
/// `Mutex` (like the [`Dft`] engine) and `exec` takes `&mut self`.
///
/// The struct holds no raw pointers — only owned `Vec<u8>` scratch and `c_int`
/// state — so it is `Send + Sync` by auto-derivation, and the convolution plan
/// needs no hand-written marker impls.
///
/// The struct does not encode the precision `T` — the owning plan is monomorphic
/// in `T` and always calls `build::<T>` / `exec::<T>` with the same `T`, so a
/// buffer sized for one width is only ever driven at that width.
pub struct Conv {
    /// Reusable work scratch passed to every `ippsConvolve` call (may be empty
    /// when the chosen algorithm needs none — e.g. the direct method).
    work: Vec<u8>,
    /// The IPP algorithm selector (`ippAlg{Auto,Direct,FFT}`).
    alg: c_int,
}

impl Conv {
    /// Size the work buffer for a convolution of `a_len`-by-`b_len` inputs with
    /// algorithm `alg` at width `T`, and build the engine.
    pub fn build<T: DspFloat>(a_len: usize, b_len: usize, alg: c_int) -> Result<Self> {
        let src1 = to_ipp_int(a_len, "conv a_len exceeds IPP c_int range")?;
        let src2 = to_ipp_int(b_len, "conv b_len exceeds IPP c_int range")?;
        let data_type = if is_f32::<T>() {
            sys::IPP_32F
        } else if is_f64::<T>() {
            sys::IPP_64F
        } else {
            return Err(unsupported_width());
        };

        let mut buf_size: c_int = 0;
        // SAFETY: the size out-pointer is valid; `data_type` and `alg` are valid
        // IPP enum values; the lengths are positive as `c_int`.
        let status = unsafe {
            sys::conv::ippsConvolveGetBufferSize(src1, src2, data_type, alg, &raw mut buf_size)
        };
        check_ipp(status, "IPP ConvolveGetBufferSize")?;

        let work = vec![0u8; usize::try_from(buf_size).unwrap_or(0)];
        Ok(Self { work, alg })
    }

    /// Run the full linear convolution `out = a ∗ b`.
    ///
    /// `a` / `b` must have the lengths the engine was built for and `out` must
    /// hold `a.len() + b.len() - 1` samples (validated by the caller).
    pub fn exec<T: DspFloat>(&mut self, a: &[T], b: &[T], out: &mut [T]) -> Result<()> {
        let src1 = to_ipp_int(a.len(), "conv a_len exceeds IPP c_int range")?;
        let src2 = to_ipp_int(b.len(), "conv b_len exceeds IPP c_int range")?;
        let alg = self.alg;
        let buf = buf_ptr(&mut self.work);

        // SAFETY: `a`/`b` have `src1`/`src2` elements and `out` has
        // `src1 + src2 - 1` (validated by the caller); `work` was sized by
        // `ippsConvolveGetBufferSize` for these lengths and `alg`; the width
        // branch picks the matching-width entry point. `buf` is null only when
        // the sized buffer is empty, which IPP tolerates.
        let status = unsafe {
            if is_f32::<T>() {
                sys::conv::ippsConvolve_32f(
                    a.as_ptr().cast::<f32>(),
                    src1,
                    b.as_ptr().cast::<f32>(),
                    src2,
                    out.as_mut_ptr().cast::<f32>(),
                    alg,
                    buf,
                )
            } else if is_f64::<T>() {
                sys::conv::ippsConvolve_64f(
                    a.as_ptr().cast::<f64>(),
                    src1,
                    b.as_ptr().cast::<f64>(),
                    src2,
                    out.as_mut_ptr().cast::<f64>(),
                    alg,
                    buf,
                )
            } else {
                return Err(unsupported_width());
            }
        };
        check_ipp(status, "IPP Convolve exec")
    }
}

// ─── Plan thread-safety markers ──────────────────────────────────────
//
// An IPP transform writes scratch into its work buffer on every compute call,
// so a `Dft` is not concurrency-safe the way a read-only spec would be. Each
// plan keeps the whole `Dft` behind a `Mutex` and touches it only while that
// lock is held (`execute` locks, runs one transform, unlocks; `Drop` is the
// `Vec`s freeing themselves with no live borrows). Serializing all `Dft` access
// behind the plan's own `Mutex` makes sharing `&plan` across threads sound, so
// the marker traits are asserted here.

// `Dft` is `!Send` only because of its raw `spec_ptr` field (the `Vec`s and the
// `bool` are `Send`). The pointer is an owned interior reference into `spec`,
// which moves with the `Dft`, so sending the whole `Dft` to another thread is
// sound. Asserting `Send` here makes `Mutex<Dft>` `Send`, so each plan's
// `Mutex<Dft>` field carries no other-thread hazard and the plan markers below
// guard only the work-scratch mutation.
//
// SAFETY: the raw `spec_ptr` references `Dft`'s own `spec` allocation and moves
// with it; no other thread observes the `Dft` except through the owning plan.
unsafe impl Send for Dft {}

// SAFETY: the `Dft` is only touched behind the plan's `Mutex`, so it never
// crosses a thread boundary unguarded.
unsafe impl<T> Send for IppDftC2cPlan<T> {}
// SAFETY: see above — all `Dft` access is serialized behind the plan's `Mutex`.
unsafe impl<T> Sync for IppDftC2cPlan<T> {}

// SAFETY: see above.
unsafe impl<T> Send for IppDftR2cPlan<T> {}
// SAFETY: see above.
unsafe impl<T> Sync for IppDftR2cPlan<T> {}

// SAFETY: see above.
unsafe impl<T> Send for IppDftC2rPlan<T> {}
// SAFETY: see above.
unsafe impl<T> Sync for IppDftC2rPlan<T> {}

// The IIR engine owns its state buffer; the raw `state_ptr` is an interior
// reference into that buffer and moves with the `Iir`, so the whole `Iir` may
// cross a thread boundary (like `Dft`). It carries no `Mutex` because the
// processor drives it through `&mut self`; the marker traits flow up to the
// owning `IppIirProcessor` by auto-derivation.

// SAFETY: `state_ptr` references `Iir`'s own `state` allocation and moves with
// it; no other thread observes the `Iir` except through the owning processor.
unsafe impl Send for Iir {}

// SAFETY: the only `&self` method (`get_dly_line`) reads the IPP state without
// mutating it; every mutation (`apply`, `re_init`) takes `&mut self`, so a shared
// `&Iir` cannot race. Sharing `&Iir` across threads is therefore sound.
unsafe impl Sync for Iir {}
