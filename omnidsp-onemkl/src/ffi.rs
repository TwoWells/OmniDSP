// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Safe wrappers over the raw `omnidsp-onemkl-sys` FFI surface.
//!
//! This is the **only** module in the crate that contains `unsafe` code: every
//! FFI call, every pointer cast, and the `unsafe impl Send`/`Sync` for the plan
//! types live here, each guarded by a safety comment.  The DFT and vector-ops
//! modules call exclusively into these wrappers and stay entirely unsafe-free.
//!
//! Each wrapper validates what it can in safe Rust (descriptor handles, slice
//! lengths via the caller, integer-width conversions), performs the FFI call,
//! and maps any non-success status to [`Error::backend`].

#![allow(
    unsafe_code,
    reason = "FFI calls to omnidsp-onemkl-sys: this is the single isolated unsafe module"
)]

use std::any::TypeId;
use std::os::raw::c_void;
use std::sync::Once;

use num_complex::Complex;
use omnidsp_onemkl_sys as sys;
use omnidsp_onemkl_sys::{
    DFTI_NO_ERROR, DftiConfigParam, DftiConfigValue, DftiDescriptorHandle, MklInt, MklLong,
    VSL_STATUS_OK, VSLConvTaskPtr, VSLCorrTaskPtr,
};

use omnidsp_core::error::{Error, Result};
use omnidsp_core::types::DspFloat;

use crate::conv::OneMklConvPlan;
use crate::dft::{OneMklDftC2cPlan, OneMklDftC2rPlan, OneMklDftR2cPlan};
use crate::xcorr::OneMklCrossCorrPlan;

// ─── Status helpers ──────────────────────────────────────────────────

/// Map a DFTI status code to a [`Result`].
///
/// DFTI returns `DFTI_NO_ERROR` (0) on success and a non-zero `MKL_LONG`
/// otherwise.
fn check_dfti(status: MklLong, context: &'static str) -> Result<()> {
    if status == DFTI_NO_ERROR {
        Ok(())
    } else {
        // DFTI status codes are small; the saturating conversion avoids any
        // truncation while never panicking.
        let code = i32::try_from(status).unwrap_or(-1);
        Err(Error::backend(code, context))
    }
}

/// Convert a `usize` length to the `MKL_INT` (i32) the BLAS/VM entry points
/// take, mapping overflow to a backend error rather than truncating.
fn to_mkl_int(len: usize, context: &'static str) -> Result<MklInt> {
    MklInt::try_from(len).map_err(|_| Error::backend(-1, context))
}

// ─── DFTI descriptor lifecycle ───────────────────────────────────────

/// Create a 1-D DFTI descriptor of the given precision and domain.
///
/// `precision` is [`DftiConfigValue::Single`] or [`DftiConfigValue::Double`];
/// `domain` is [`DftiConfigValue::Complex`] or [`DftiConfigValue::Real`].
pub fn create_descriptor(
    precision: DftiConfigValue,
    domain: DftiConfigValue,
    length: usize,
) -> Result<DftiDescriptorHandle> {
    let len = MklLong::try_from(length)
        .map_err(|_| Error::backend(-1, "DFT length exceeds MKL_LONG range"))?;
    let mut handle: DftiDescriptorHandle = std::ptr::null_mut();

    // SAFETY: `handle` is a valid out-pointer; `precision`/`domain` are valid
    // DFTI config-value enums; dimension is 1 and `len` is a positive length.
    // DFTI writes the new descriptor pointer into `handle` and returns a status.
    let status = unsafe { sys::DftiCreateDescriptor(&raw mut handle, precision, domain, 1, len) };
    check_dfti(status, "DftiCreateDescriptor failed")?;

    if handle.is_null() {
        return Err(Error::backend(
            -1,
            "DftiCreateDescriptor returned a null descriptor",
        ));
    }
    Ok(handle)
}

/// Set a DFTI config parameter whose value is a config-value enum.
///
/// The variadic `DftiSetValue` reads the trailing argument with the type the
/// parameter expects; for `PLACEMENT` and `CONJUGATE_EVEN_STORAGE` that type is
/// `MKL_LONG`, so the enum is passed as its `MklLong` discriminant.
fn set_value_enum(
    handle: DftiDescriptorHandle,
    param: DftiConfigParam,
    value: DftiConfigValue,
) -> Result<()> {
    let raw = value as MklLong;
    // SAFETY: `handle` is a committed-or-uncommitted valid descriptor; `param`
    // is a valid config parameter whose value type is `MKL_LONG`, and `raw` is
    // passed as the matching `MklLong` through the variadic argument.
    let status = unsafe { sys::DftiSetValue(handle, param, raw) };
    check_dfti(status, "DftiSetValue failed")
}

/// Configure the descriptor for out-of-place transforms (distinct I/O buffers).
pub fn set_placement_not_inplace(handle: DftiDescriptorHandle) -> Result<()> {
    set_value_enum(
        handle,
        DftiConfigParam::Placement,
        DftiConfigValue::NotInplace,
    )
}

/// Configure conjugate-even storage as interleaved complex (`COMPLEX_COMPLEX`).
///
/// This makes the real forward transform write exactly `N/2 + 1` complex bins
/// and the real backward transform consume `N/2 + 1` complex bins, matching
/// `OmniDSP`'s half-spectrum convention with no manual mirroring.
pub fn set_cce_complex_complex(handle: DftiDescriptorHandle) -> Result<()> {
    set_value_enum(
        handle,
        DftiConfigParam::ConjugateEvenStorage,
        DftiConfigValue::ComplexComplex,
    )
}

/// Commit a descriptor, making it ready for compute calls.
pub fn commit(handle: DftiDescriptorHandle) -> Result<()> {
    // SAFETY: `handle` is a valid, fully configured descriptor.
    let status = unsafe { sys::DftiCommitDescriptor(handle) };
    check_dfti(status, "DftiCommitDescriptor failed")
}

/// Free a DFTI descriptor.
///
/// Called from the plans' `Drop`; the status is checked but a failed free is
/// non-recoverable at drop time, so it is dropped silently apart from the
/// returned `Result` the caller may ignore.
pub fn free_descriptor(mut handle: DftiDescriptorHandle) {
    // SAFETY: `handle` was produced by `create_descriptor` and is freed exactly
    // once (from the owning plan's `Drop`).  `DftiFreeDescriptor` nulls the
    // handle through the out-pointer.
    let _status = unsafe { sys::DftiFreeDescriptor(&raw mut handle) };
}

// ─── DFTI compute ────────────────────────────────────────────────────

/// Run a committed forward transform out-of-place.
///
/// `input` and `output` must point to buffers of the layout the committed
/// descriptor expects.  DFTI does not mutate the input for out-of-place
/// transforms, but the C signature is `void*`, so the input pointer is passed
/// as `*mut c_void`.
pub fn compute_forward(
    handle: DftiDescriptorHandle,
    input: *mut c_void,
    output: *mut c_void,
) -> Result<()> {
    // SAFETY: `handle` is a committed descriptor; `input`/`output` point to
    // distinct caller-owned buffers sized per the descriptor's domain and
    // length (validated by the caller before this call).
    let status = unsafe { sys::DftiComputeForward(handle, input, output) };
    check_dfti(status, "DftiComputeForward failed")
}

/// Run a committed backward transform out-of-place.
///
/// Same buffer/aliasing contract as [`compute_forward`].
pub fn compute_backward(
    handle: DftiDescriptorHandle,
    input: *mut c_void,
    output: *mut c_void,
) -> Result<()> {
    // SAFETY: `handle` is a committed descriptor; `input`/`output` point to
    // distinct caller-owned buffers sized per the descriptor's domain and
    // length (validated by the caller before this call).
    let status = unsafe { sys::DftiComputeBackward(handle, input, output) };
    check_dfti(status, "DftiComputeBackward failed")
}

/// Reinterpret a real slice as a `*mut c_void` for an out-of-place DFTI input.
///
/// The pointee is not mutated for out-of-place transforms (see
/// [`compute_forward`]); the `*const → *mut` cast only satisfies the C
/// signature.
#[must_use]
pub const fn real_ptr<T>(slice: &[T]) -> *mut c_void {
    slice.as_ptr().cast::<c_void>().cast_mut()
}

/// Reinterpret a mutable real slice as a `*mut c_void` for a DFTI output.
#[must_use]
pub const fn real_ptr_mut<T>(slice: &mut [T]) -> *mut c_void {
    slice.as_mut_ptr().cast::<c_void>()
}

/// Reinterpret a complex slice as a `*mut c_void` for an out-of-place DFTI
/// input.  `num_complex::Complex<T>` is `#[repr(C)]` with `re, im`, so its
/// memory layout is `[T; 2]`, which DFTI's `COMPLEX_COMPLEX` storage expects.
#[must_use]
pub const fn complex_ptr<T>(slice: &[Complex<T>]) -> *mut c_void {
    slice.as_ptr().cast::<c_void>().cast_mut()
}

/// Reinterpret a mutable complex slice as a `*mut c_void` for a DFTI output.
#[must_use]
pub const fn complex_ptr_mut<T>(slice: &mut [Complex<T>]) -> *mut c_void {
    slice.as_mut_ptr().cast::<c_void>()
}

// ─── VM accuracy-mode configuration ──────────────────────────────────

/// Process-global guard for the one-time VM accuracy-mode configuration.
static VML_MODE_INIT: Once = Once::new();

/// Select oneMKL's enhanced-performance VM mode (`VML_EP`) once per process.
///
/// Every VM op the backend uses (`vsMul` / `vcMul` / `vcConj` / …) is exact:
/// there are no transcendentals whose accuracy `EP` would trade away, so `EP`
/// loses nothing while shedding the default mode's per-element special-value
/// checks (which otherwise roughly double the `cmul` cost).  This is a documented
/// configuration knob, not per-size tuning.
///
/// `vmlSetMode` is process-global and thread-safe; the [`Once`] makes the
/// configuration idempotent and lazy — the backend's `new` is `const` and cannot
/// run FFI, so the mode is set on first VM use instead.
fn ensure_vml_ep() {
    VML_MODE_INIT.call_once(|| {
        // SAFETY: `vmlSetMode` is a thread-safe oneMKL entry point; it takes a
        // mode constant and returns the previous mode, which we discard.
        unsafe {
            sys::vmlSetMode(sys::VML_EP);
        }
    });
}

// ─── VM / BLAS vector-op wrappers ────────────────────────────────────
//
// Each wrapper validates the `usize → MKL_INT` conversion and casts complex
// slices to the layout-compatible `[T; 2]` pointers MKL expects.  Real-only
// callers (mul/add/scale/dot/axpy) pass plain `*const`/`*mut` pointers.
//
// Every VM wrapper (the `vsMul` / `vcMul` / `vcConj` family) calls
// `ensure_vml_ep` first so the enhanced-performance mode is configured before
// any VM work; the BLAS wrappers (dot / scal / axpy) are unaffected by VM mode.

/// Cast a complex slice to MKL's `*const [f32; 2]` (`MKL_Complex8`) layout.
///
/// `Complex<f32>` is `#[repr(C)] { re: f32, im: f32 }`, identical in layout to
/// `[f32; 2]`, so the reinterpretation is sound.
const fn cplx32(slice: &[Complex<f32>]) -> *const [f32; 2] {
    slice.as_ptr().cast::<[f32; 2]>()
}

/// Cast a complex slice to MKL's `*const [f64; 2]` (`MKL_Complex16`) layout.
const fn cplx64(slice: &[Complex<f64>]) -> *const [f64; 2] {
    slice.as_ptr().cast::<[f64; 2]>()
}

/// Cast a mutable complex slice to MKL's `*mut [f32; 2]` layout.
const fn cplx32_mut(slice: &mut [Complex<f32>]) -> *mut [f32; 2] {
    slice.as_mut_ptr().cast::<[f32; 2]>()
}

/// Cast a mutable complex slice to MKL's `*mut [f64; 2]` layout.
const fn cplx64_mut(slice: &mut [Complex<f64>]) -> *mut [f64; 2] {
    slice.as_mut_ptr().cast::<[f64; 2]>()
}

macro_rules! real_binop {
    ($name:ident, $sys_fn:ident, $float:ty, $ctx:literal $(,)?) => {
        #[doc = concat!("Elementwise `", stringify!($sys_fn), "`: `out = a OP b` for `", stringify!($float), "`.")]
        pub fn $name(a: &[$float], b: &[$float], out: &mut [$float]) -> Result<()> {
            ensure_vml_ep();
            let n = to_mkl_int(out.len(), $ctx)?;
            // SAFETY: `a`, `b`, `out` all have `out.len()` elements (length
            // equality validated by the caller); `n` is that length as MKL_INT.
            unsafe {
                sys::$sys_fn(n, a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
            }
            Ok(())
        }
    };
}

real_binop!(vmul_f32, vsMul, f32, "vsMul length exceeds MKL_INT range");
real_binop!(vmul_f64, vdMul, f64, "vdMul length exceeds MKL_INT range");
real_binop!(vadd_f32, vsAdd, f32, "vsAdd length exceeds MKL_INT range");
real_binop!(vadd_f64, vdAdd, f64, "vdAdd length exceeds MKL_INT range");

macro_rules! complex_mul {
    ($name:ident, $sys_fn:ident, $float:ty, $cast:ident, $cast_mut:ident, $ctx:literal $(,)?) => {
        #[doc = concat!("Elementwise complex `", stringify!($sys_fn), "`: `out = a * b`.")]
        pub fn $name(
            a: &[Complex<$float>],
            b: &[Complex<$float>],
            out: &mut [Complex<$float>],
        ) -> Result<()> {
            ensure_vml_ep();
            let n = to_mkl_int(out.len(), $ctx)?;
            // SAFETY: `a`, `b`, `out` have `out.len()` complex elements (length
            // equality validated by the caller); the `[<$float>; 2]` casts are
            // sound because `Complex<$float>` is `#[repr(C)]` with the same
            // layout (see `cplx*`).
            unsafe {
                sys::$sys_fn(n, $cast(a), $cast(b), $cast_mut(out));
            }
            Ok(())
        }
    };
}

complex_mul!(
    cmul_f32,
    vcMul,
    f32,
    cplx32,
    cplx32_mut,
    "vcMul length exceeds MKL_INT range"
);
complex_mul!(
    cmul_f64,
    vzMul,
    f64,
    cplx64,
    cplx64_mut,
    "vzMul length exceeds MKL_INT range"
);

/// In-place elementwise complex multiply: `data *= other` via `vcMul`.
///
/// MKL's VM permits the result pointer to alias an input pointer.
pub fn cmul_inplace_f32(data: &mut [Complex<f32>], other: &[Complex<f32>]) -> Result<()> {
    ensure_vml_ep();
    let n = to_mkl_int(data.len(), "vcMul length exceeds MKL_INT range")?;
    let out = cplx32_mut(data);
    // SAFETY: `out` aliases `data`'s storage; `other` has `data.len()` elements
    // (validated by the caller).  VM allows the in-place alias.
    unsafe {
        sys::vcMul(n, out.cast_const(), cplx32(other), out);
    }
    Ok(())
}

/// In-place elementwise complex multiply: `data *= other` via `vzMul`.
pub fn cmul_inplace_f64(data: &mut [Complex<f64>], other: &[Complex<f64>]) -> Result<()> {
    ensure_vml_ep();
    let n = to_mkl_int(data.len(), "vzMul length exceeds MKL_INT range")?;
    let out = cplx64_mut(data);
    // SAFETY: `out` aliases `data`'s storage; `other` has `data.len()` elements
    // (validated by the caller).  VM allows the in-place alias.
    unsafe {
        sys::vzMul(n, out.cast_const(), cplx64(other), out);
    }
    Ok(())
}

/// In-place elementwise real multiply: `data *= other` via `vsMul`.
pub fn mul_inplace_f32(data: &mut [f32], other: &[f32]) -> Result<()> {
    ensure_vml_ep();
    let n = to_mkl_int(data.len(), "vsMul length exceeds MKL_INT range")?;
    let ptr = data.as_mut_ptr();
    // SAFETY: `ptr` aliases `data`; `other` has `data.len()` elements
    // (validated by the caller).  VM allows the in-place alias.
    unsafe {
        sys::vsMul(n, ptr.cast_const(), other.as_ptr(), ptr);
    }
    Ok(())
}

/// In-place elementwise real multiply: `data *= other` via `vdMul`.
pub fn mul_inplace_f64(data: &mut [f64], other: &[f64]) -> Result<()> {
    ensure_vml_ep();
    let n = to_mkl_int(data.len(), "vdMul length exceeds MKL_INT range")?;
    let ptr = data.as_mut_ptr();
    // SAFETY: `ptr` aliases `data`; `other` has `data.len()` elements
    // (validated by the caller).  VM allows the in-place alias.
    unsafe {
        sys::vdMul(n, ptr.cast_const(), other.as_ptr(), ptr);
    }
    Ok(())
}

/// Conjugate a complex slice in place via `vcConj`.
pub fn conj_inplace_f32(data: &mut [Complex<f32>]) -> Result<()> {
    ensure_vml_ep();
    let n = to_mkl_int(data.len(), "vcConj length exceeds MKL_INT range")?;
    let ptr = cplx32_mut(data);
    // SAFETY: `ptr` aliases `data`; `vcConj` allows the in-place alias.
    unsafe {
        sys::vcConj(n, ptr.cast_const(), ptr);
    }
    Ok(())
}

/// Conjugate a complex slice in place via `vzConj`.
pub fn conj_inplace_f64(data: &mut [Complex<f64>]) -> Result<()> {
    ensure_vml_ep();
    let n = to_mkl_int(data.len(), "vzConj length exceeds MKL_INT range")?;
    let ptr = cplx64_mut(data);
    // SAFETY: `ptr` aliases `data`; `vzConj` allows the in-place alias.
    unsafe {
        sys::vzConj(n, ptr.cast_const(), ptr);
    }
    Ok(())
}

macro_rules! blas_dot {
    ($name:ident, $sys_fn:ident, $float:ty, $ctx:literal $(,)?) => {
        #[doc = concat!("BLAS L1 `", stringify!($sys_fn), "` dot product (stride 1).")]
        pub fn $name(a: &[$float], b: &[$float]) -> Result<$float> {
            let n = to_mkl_int(a.len(), $ctx)?;
            // SAFETY: `a`, `b` have `a.len()` elements (validated by the
            // caller); strides are 1.
            Ok(unsafe { sys::$sys_fn(n, a.as_ptr(), 1, b.as_ptr(), 1) })
        }
    };
}

blas_dot!(
    dot_f32,
    cblas_sdot,
    f32,
    "cblas_sdot length exceeds MKL_INT range"
);
blas_dot!(
    dot_f64,
    cblas_ddot,
    f64,
    "cblas_ddot length exceeds MKL_INT range"
);

macro_rules! complex_dot {
    ($name:ident, $sys_fn:ident, $float:ty, $cast:ident, $ctx:literal $(,)?) => {
        #[doc = concat!("BLAS L1 `", stringify!($sys_fn), "` unconjugated complex dot (stride 1).")]
        pub fn $name(a: &[Complex<$float>], b: &[Complex<$float>]) -> Result<Complex<$float>> {
            let n = to_mkl_int(a.len(), $ctx)?;
            let mut dotu: [$float; 2] = [0.0, 0.0];
            // SAFETY: `a`, `b` have `a.len()` complex elements (validated by the
            // caller); `dotu` is a valid 2-element out-buffer matching
            // `MKL_Complex`; the `[<$float>; 2]` casts are sound (see `cplx*`).
            unsafe {
                sys::$sys_fn(n, $cast(a), 1, $cast(b), 1, &mut dotu);
            }
            Ok(Complex::new(dotu[0], dotu[1]))
        }
    };
}

complex_dot!(
    cdot_f32,
    cblas_cdotu_sub,
    f32,
    cplx32,
    "cblas_cdotu_sub length exceeds MKL_INT range",
);
complex_dot!(
    cdot_f64,
    cblas_zdotu_sub,
    f64,
    cplx64,
    "cblas_zdotu_sub length exceeds MKL_INT range",
);

/// BLAS L1 in-place scale `data *= alpha` via `cblas_sscal` (stride 1).
///
/// Infallible: `n` is pre-validated by the caller (which falls back to a scalar
/// loop when the length exceeds `MKL_INT`), so this is only reached with a
/// representable length.
pub fn scale_f32(data: &mut [f32], alpha: f32, n: MklInt) {
    // SAFETY: `data` has at least `n` elements; stride is 1.
    unsafe {
        sys::cblas_sscal(n, alpha, data.as_mut_ptr(), 1);
    }
}

/// BLAS L1 in-place scale `data *= alpha` via `cblas_dscal` (stride 1).
pub fn scale_f64(data: &mut [f64], alpha: f64, n: MklInt) {
    // SAFETY: `data` has at least `n` elements; stride is 1.
    unsafe {
        sys::cblas_dscal(n, alpha, data.as_mut_ptr(), 1);
    }
}

/// BLAS L1 `data += other` via `cblas_saxpy` (alpha = 1.0, stride 1).
pub fn axpy_f32(data: &mut [f32], other: &[f32]) -> Result<()> {
    let n = to_mkl_int(data.len(), "cblas_saxpy length exceeds MKL_INT range")?;
    // SAFETY: `other` and `data` have `data.len()` elements (validated by the
    // caller); strides are 1.
    unsafe {
        sys::cblas_saxpy(n, 1.0, other.as_ptr(), 1, data.as_mut_ptr(), 1);
    }
    Ok(())
}

/// BLAS L1 `data += other` via `cblas_daxpy` (alpha = 1.0, stride 1).
pub fn axpy_f64(data: &mut [f64], other: &[f64]) -> Result<()> {
    let n = to_mkl_int(data.len(), "cblas_daxpy length exceeds MKL_INT range")?;
    // SAFETY: `other` and `data` have `data.len()` elements (validated by the
    // caller); strides are 1.
    unsafe {
        sys::cblas_daxpy(n, 1.0, other.as_ptr(), 1, data.as_mut_ptr(), 1);
    }
    Ok(())
}

/// Try to convert a length to `MKL_INT`, returning `None` on overflow.
///
/// Used by the infallible `VecOps::scale` to decide between the BLAS call and a
/// scalar fallback without panicking.
#[must_use]
pub fn try_mkl_int(len: usize) -> Option<MklInt> {
    MklInt::try_from(len).ok()
}

// ─── VS convolution task wrappers ────────────────────────────────────
//
// These wrap oneMKL's native VS *convolution* task API: build a task
// (`vsl{s,d}ConvNewTask1D`), execute it (`vsl{s,d}ConvExec1D`, stride 1), and
// destroy it (`vslConvDeleteTask`).  Cross-correlation has its own peer wrappers
// below over the VS *correlation* API, so these are convolution-only.
//
// f32/f64 flavor selection (`vsls*` vs `vsld*`) happens here, inside the
// generic dispatch, because the `create_plan::<T>` signature cannot carry a
// `T: f32 | f64` bound.  Dispatch keys on `TypeId::of::<T>()`; once the concrete
// width is known, the `&[T]` execution slices are reinterpreted as `&[f32]` /
// `&[f64]` via a layout-identity transmute (sound because `DspFloat: 'static`
// and `T` is only ever `f32` or `f64` — the same identity the rest of the crate
// relies on for its complex-layout casts).  A non-`f32`/`f64` `T` returns a
// backend error; it is unreachable in practice.  The correlation wrappers reuse
// the same `as_f{32,64}_slice*` reinterpret helpers.

/// Map a VS status code to a [`Result`].
///
/// VS returns `VSL_STATUS_OK` (0) on success and a non-zero `int` otherwise.
fn check_vsl(status: i32, context: &'static str) -> Result<()> {
    if status == VSL_STATUS_OK {
        Ok(())
    } else {
        Err(Error::backend(status, context))
    }
}

/// Construct a 1-D single-precision VS **convolution** task.
///
/// `mode` is one of `VSL_CONV_MODE_{AUTO,FFT,DIRECT}`; the three shapes are the
/// input lengths and `xshape + yshape - 1` for the full-convolution output.
fn conv_new_task_f32(
    mode: MklInt,
    xshape: usize,
    yshape: usize,
    zshape: usize,
) -> Result<VSLConvTaskPtr> {
    let x = to_mkl_int(xshape, "vslsConvNewTask1D xshape exceeds MKL_INT range")?;
    let y = to_mkl_int(yshape, "vslsConvNewTask1D yshape exceeds MKL_INT range")?;
    let z = to_mkl_int(zshape, "vslsConvNewTask1D zshape exceeds MKL_INT range")?;
    let mut task: VSLConvTaskPtr = std::ptr::null_mut();
    // SAFETY: `task` is a valid out-pointer; `mode` is a valid VS conv-mode
    // constant; the shapes are positive lengths as MKL_INT.  VS writes the new
    // task pointer into `task` and returns a status.
    let status = unsafe { sys::vslsConvNewTask1D(&raw mut task, mode, x, y, z) };
    check_vsl(status, "vslsConvNewTask1D failed")?;
    if task.is_null() {
        return Err(Error::backend(-1, "vslsConvNewTask1D returned a null task"));
    }
    Ok(task)
}

/// Construct a 1-D double-precision VS **convolution** task.
///
/// Double-precision peer of [`conv_new_task_f32`].
fn conv_new_task_f64(
    mode: MklInt,
    xshape: usize,
    yshape: usize,
    zshape: usize,
) -> Result<VSLConvTaskPtr> {
    let x = to_mkl_int(xshape, "vsldConvNewTask1D xshape exceeds MKL_INT range")?;
    let y = to_mkl_int(yshape, "vsldConvNewTask1D yshape exceeds MKL_INT range")?;
    let z = to_mkl_int(zshape, "vsldConvNewTask1D zshape exceeds MKL_INT range")?;
    let mut task: VSLConvTaskPtr = std::ptr::null_mut();
    // SAFETY: `task` is a valid out-pointer; `mode` is a valid VS conv-mode
    // constant; the shapes are positive lengths as MKL_INT.  VS writes the new
    // task pointer into `task` and returns a status.
    let status = unsafe { sys::vsldConvNewTask1D(&raw mut task, mode, x, y, z) };
    check_vsl(status, "vsldConvNewTask1D failed")?;
    if task.is_null() {
        return Err(Error::backend(-1, "vsldConvNewTask1D returned a null task"));
    }
    Ok(task)
}

/// Build a VS convolution task for precision `T`, dispatching on the concrete
/// float width.
///
/// `T` is `f32` or `f64` in practice; any other type returns a backend error.
pub fn conv_new_task<T: DspFloat>(
    mode: MklInt,
    xshape: usize,
    yshape: usize,
    zshape: usize,
) -> Result<VSLConvTaskPtr> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        conv_new_task_f32(mode, xshape, yshape, zshape)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        conv_new_task_f64(mode, xshape, yshape, zshape)
    } else {
        Err(Error::backend(-1, "VS convolution task requires f32 or f64 precision"))
    }
}

/// Reinterpret a `&[T]` as `&[f32]` once `T` is known to be `f32`.
///
/// SAFETY contract is the caller's: this is only reached after a
/// `TypeId::of::<T>() == TypeId::of::<f32>()` guard, so `T` and `f32` are the
/// same type and the reinterpretation is a layout identity.
fn as_f32_slice<T: DspFloat>(slice: &[T]) -> &[f32] {
    // SAFETY: reached only when `T` is `f32` (caller checks `TypeId`), so `T`
    // and `f32` have identical layout; the element count is preserved, so the
    // resulting slice has the same length.  `DspFloat: 'static` makes the
    // `TypeId` comparison meaningful.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<f32>(), slice.len()) }
}

/// Reinterpret a `&mut [T]` as `&mut [f32]` once `T` is known to be `f32`.
fn as_f32_slice_mut<T: DspFloat>(slice: &mut [T]) -> &mut [f32] {
    let len = slice.len();
    // SAFETY: reached only when `T` is `f32` (caller checks `TypeId`), so `T`
    // and `f32` have identical layout and the element count is preserved.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<f32>(), len) }
}

/// Reinterpret a `&[T]` as `&[f64]` once `T` is known to be `f64`.
fn as_f64_slice<T: DspFloat>(slice: &[T]) -> &[f64] {
    // SAFETY: reached only when `T` is `f64` (caller checks `TypeId`), so `T`
    // and `f64` have identical layout and the element count is preserved.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<f64>(), slice.len()) }
}

/// Reinterpret a `&mut [T]` as `&mut [f64]` once `T` is known to be `f64`.
fn as_f64_slice_mut<T: DspFloat>(slice: &mut [T]) -> &mut [f64] {
    let len = slice.len();
    // SAFETY: reached only when `T` is `f64` (caller checks `TypeId`), so `T`
    // and `f64` have identical layout and the element count is preserved.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<f64>(), len) }
}

/// Execute a single-precision VS convolution task with stride 1.
fn conv_exec_f32(task: VSLConvTaskPtr, x: &[f32], y: &[f32], z: &mut [f32]) -> Result<()> {
    // SAFETY: `task` is a live conv task; `x`/`y`/`z` are caller-owned buffers
    // sized to the task's `xshape`/`yshape`/`zshape` (validated by the caller);
    // the stride-1 access stays within each slice.
    let status =
        unsafe { sys::vslsConvExec1D(task, x.as_ptr(), 1, y.as_ptr(), 1, z.as_mut_ptr(), 1) };
    check_vsl(status, "vslsConvExec1D failed")
}

/// Execute a double-precision VS convolution task with stride 1.
fn conv_exec_f64(task: VSLConvTaskPtr, x: &[f64], y: &[f64], z: &mut [f64]) -> Result<()> {
    // SAFETY: `task` is a live conv task; `x`/`y`/`z` are caller-owned buffers
    // sized to the task's `xshape`/`yshape`/`zshape` (validated by the caller);
    // the stride-1 access stays within each slice.
    let status =
        unsafe { sys::vsldConvExec1D(task, x.as_ptr(), 1, y.as_ptr(), 1, z.as_mut_ptr(), 1) };
    check_vsl(status, "vsldConvExec1D failed")
}

/// Execute a VS convolution task for precision `T`, dispatching on the concrete
/// float width.
///
/// The buffers are reinterpreted to the concrete `f32`/`f64` slice the matching
/// executor takes (a layout-identity transmute guarded by the same `TypeId`
/// check the task constructor used).  `x`, `y`, and `z` must already match the
/// task's `xshape`, `yshape`, and `zshape`.  A non-`f32`/`f64` `T` returns a
/// backend error; it is unreachable in practice.
pub fn conv_exec<T: DspFloat>(
    task: VSLConvTaskPtr,
    x: &[T],
    y: &[T],
    z: &mut [T],
) -> Result<()> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        conv_exec_f32(task, as_f32_slice(x), as_f32_slice(y), as_f32_slice_mut(z))
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        conv_exec_f64(task, as_f64_slice(x), as_f64_slice(y), as_f64_slice_mut(z))
    } else {
        Err(Error::backend(-1, "VS convolution exec requires f32 or f64 precision"))
    }
}

/// Destroy a VS convolution task and free its resources.
///
/// Called from the plans' `Drop`; the status is checked but a failed free is
/// non-recoverable at drop time, so the returned status is discarded.
pub fn conv_delete_task(mut task: VSLConvTaskPtr) {
    // SAFETY: `task` was produced by `conv_new_task*` and is freed exactly once
    // (from the owning plan's `Drop`).  `vslConvDeleteTask` takes the task by
    // out-pointer.
    let _status = unsafe { sys::vslConvDeleteTask(&raw mut task) };
}

// ─── VS correlation task wrappers ────────────────────────────────────
//
// Cross-correlation wraps oneMKL's native VS *correlation* task API
// (`vsl{s,d}CorrNewTask1D` / `vsl{s,d}CorrExec1D` / `vslCorrDeleteTask`), the
// literal vendor kernel — not a convolution of a reversed input.  The same
// `TypeId` dispatch and layout-identity slice reinterpretation the conv
// wrappers use carry the f32/f64 selection (see the conv-wrapper module note).
//
// The full linear correlation is built with `xshape = a_len`, `yshape = b_len`,
// `zshape = a_len + b_len - 1`, default `start`/`decimation` — see the
// lag/order note in `crate::xcorr`.

/// Construct a 1-D single-precision VS **correlation** task.
///
/// `mode` is one of `VSL_CORR_MODE_{AUTO,FFT,DIRECT}`; the three shapes are the
/// input lengths and `xshape + yshape - 1` for the full-correlation output.
fn corr_new_task_f32(
    mode: MklInt,
    xshape: usize,
    yshape: usize,
    zshape: usize,
) -> Result<VSLCorrTaskPtr> {
    let x = to_mkl_int(xshape, "vslsCorrNewTask1D xshape exceeds MKL_INT range")?;
    let y = to_mkl_int(yshape, "vslsCorrNewTask1D yshape exceeds MKL_INT range")?;
    let z = to_mkl_int(zshape, "vslsCorrNewTask1D zshape exceeds MKL_INT range")?;
    let mut task: VSLCorrTaskPtr = std::ptr::null_mut();
    // SAFETY: `task` is a valid out-pointer; `mode` is a valid VS corr-mode
    // constant; the shapes are positive lengths as MKL_INT.  VS writes the new
    // task pointer into `task` and returns a status.
    let status = unsafe { sys::vslsCorrNewTask1D(&raw mut task, mode, x, y, z) };
    check_vsl(status, "vslsCorrNewTask1D failed")?;
    if task.is_null() {
        return Err(Error::backend(-1, "vslsCorrNewTask1D returned a null task"));
    }
    Ok(task)
}

/// Construct a 1-D double-precision VS **correlation** task.
///
/// Double-precision peer of [`corr_new_task_f32`].
fn corr_new_task_f64(
    mode: MklInt,
    xshape: usize,
    yshape: usize,
    zshape: usize,
) -> Result<VSLCorrTaskPtr> {
    let x = to_mkl_int(xshape, "vsldCorrNewTask1D xshape exceeds MKL_INT range")?;
    let y = to_mkl_int(yshape, "vsldCorrNewTask1D yshape exceeds MKL_INT range")?;
    let z = to_mkl_int(zshape, "vsldCorrNewTask1D zshape exceeds MKL_INT range")?;
    let mut task: VSLCorrTaskPtr = std::ptr::null_mut();
    // SAFETY: `task` is a valid out-pointer; `mode` is a valid VS corr-mode
    // constant; the shapes are positive lengths as MKL_INT.  VS writes the new
    // task pointer into `task` and returns a status.
    let status = unsafe { sys::vsldCorrNewTask1D(&raw mut task, mode, x, y, z) };
    check_vsl(status, "vsldCorrNewTask1D failed")?;
    if task.is_null() {
        return Err(Error::backend(-1, "vsldCorrNewTask1D returned a null task"));
    }
    Ok(task)
}

/// Build a VS correlation task for precision `T`, dispatching on the concrete
/// float width.
///
/// `T` is `f32` or `f64` in practice; any other type returns a backend error.
pub fn corr_new_task<T: DspFloat>(
    mode: MklInt,
    xshape: usize,
    yshape: usize,
    zshape: usize,
) -> Result<VSLCorrTaskPtr> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        corr_new_task_f32(mode, xshape, yshape, zshape)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        corr_new_task_f64(mode, xshape, yshape, zshape)
    } else {
        Err(Error::backend(-1, "VS correlation task requires f32 or f64 precision"))
    }
}

/// Execute a single-precision VS correlation task with stride 1.
fn corr_exec_f32(task: VSLCorrTaskPtr, x: &[f32], y: &[f32], z: &mut [f32]) -> Result<()> {
    // SAFETY: `task` is a live corr task; `x`/`y`/`z` are caller-owned buffers
    // sized to the task's `xshape`/`yshape`/`zshape` (validated by the caller);
    // the stride-1 access stays within each slice.
    let status =
        unsafe { sys::vslsCorrExec1D(task, x.as_ptr(), 1, y.as_ptr(), 1, z.as_mut_ptr(), 1) };
    check_vsl(status, "vslsCorrExec1D failed")
}

/// Execute a double-precision VS correlation task with stride 1.
fn corr_exec_f64(task: VSLCorrTaskPtr, x: &[f64], y: &[f64], z: &mut [f64]) -> Result<()> {
    // SAFETY: `task` is a live corr task; `x`/`y`/`z` are caller-owned buffers
    // sized to the task's `xshape`/`yshape`/`zshape` (validated by the caller);
    // the stride-1 access stays within each slice.
    let status =
        unsafe { sys::vsldCorrExec1D(task, x.as_ptr(), 1, y.as_ptr(), 1, z.as_mut_ptr(), 1) };
    check_vsl(status, "vsldCorrExec1D failed")
}

/// Execute a VS correlation task for precision `T`, dispatching on the concrete
/// float width.
///
/// The buffers are reinterpreted to the concrete `f32`/`f64` slice the matching
/// executor takes (a layout-identity transmute guarded by the same `TypeId`
/// check the task constructor used).  `x`, `y`, and `z` must already match the
/// task's `xshape`, `yshape`, and `zshape`.  A non-`f32`/`f64` `T` returns a
/// backend error; it is unreachable in practice.
pub fn corr_exec<T: DspFloat>(task: VSLCorrTaskPtr, x: &[T], y: &[T], z: &mut [T]) -> Result<()> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        corr_exec_f32(task, as_f32_slice(x), as_f32_slice(y), as_f32_slice_mut(z))
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        corr_exec_f64(task, as_f64_slice(x), as_f64_slice(y), as_f64_slice_mut(z))
    } else {
        Err(Error::backend(-1, "VS correlation exec requires f32 or f64 precision"))
    }
}

/// Destroy a VS correlation task and free its resources.
///
/// Called from the plan's `Drop`; the status is checked but a failed free is
/// non-recoverable at drop time, so the returned status is discarded.
pub fn corr_delete_task(mut task: VSLCorrTaskPtr) {
    // SAFETY: `task` was produced by `corr_new_task*` and is freed exactly once
    // (from the owning plan's `Drop`).  `vslCorrDeleteTask` takes the task by
    // out-pointer.
    let _status = unsafe { sys::vslCorrDeleteTask(&raw mut task) };
}

// ─── Thread-safety markers ───────────────────────────────────────────
//
// A committed DFTI descriptor is thread-safe for concurrent out-of-place
// compute: each `DftiComputeForward`/`DftiComputeBackward` call operates only on
// the caller-supplied I/O buffers and reads (does not mutate) the committed
// descriptor, per the Intel oneMKL DFTI thread-safety documentation.  The plan
// is otherwise immutable (it only frees the descriptor on `Drop`), so sharing
// `&plan` across threads is sound.

// SAFETY: see the module note above — the committed descriptor is read-only
// during compute and freed once at drop; the raw handle carries no other state.
unsafe impl<T> Send for OneMklDftC2cPlan<T> {}
// SAFETY: see above — concurrent out-of-place compute through a shared `&plan`
// is thread-safe per the DFTI documentation.
unsafe impl<T> Sync for OneMklDftC2cPlan<T> {}

// SAFETY: see the module note above.
unsafe impl<T> Send for OneMklDftR2cPlan<T> {}
// SAFETY: see above.
unsafe impl<T> Sync for OneMklDftR2cPlan<T> {}

// SAFETY: see the module note above.
unsafe impl<T> Send for OneMklDftC2rPlan<T> {}
// SAFETY: see above.
unsafe impl<T> Sync for OneMklDftC2rPlan<T> {}

// ─── VS plan thread-safety markers ───────────────────────────────────
//
// A VS conv/corr task carries mutable internal state across its `…Exec1D` calls
// (it caches FFT plans / scratch), so it is *not* concurrency-safe the way a
// committed DFTI descriptor is.  Both VS plans therefore keep the raw task
// pointer (`VSLConvTaskPtr` / `VSLCorrTaskPtr`) behind a `Mutex` and touch it
// only while that lock is held: `execute` locks, runs exactly one stride-1 exec,
// and unlocks; `Drop` frees the task with no live borrows.  Serializing every
// task access behind the plan's own `Mutex` makes sharing `&plan` across threads
// sound, so the marker traits are asserted here (the raw pointer alone is
// neither `Send` nor `Sync`).

// SAFETY: the raw task pointer is only dereferenced behind the plan's `Mutex`
// (see the module note above), so it never crosses a thread boundary unguarded.
unsafe impl<T> Send for OneMklConvPlan<T> {}
// SAFETY: see above — all task access is serialized behind the plan's `Mutex`.
unsafe impl<T> Sync for OneMklConvPlan<T> {}

// SAFETY: see above — the cross-correlation plan likewise guards its native VS
// correlation task behind a `Mutex`.
unsafe impl<T> Send for OneMklCrossCorrPlan<T> {}
// SAFETY: see above.
unsafe impl<T> Sync for OneMklCrossCorrPlan<T> {}
