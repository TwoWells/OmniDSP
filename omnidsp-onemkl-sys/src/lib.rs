// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Raw FFI declarations for Intel oneMKL.
//!
//! This `-sys` crate declares the ~30 oneMKL entry points needed by
//! `omnidsp-onemkl`, spanning three MKL domains:
//!
//! - **DFTI** — discrete Fourier transform interface (complex and
//!   real-to-complex FFTs).
//! - **VM + BLAS L1** — vector math (elementwise multiply/add/conjugate) and
//!   level-1 BLAS (dot, scale, axpy).
//! - **VS** — convolution and correlation tasks.
//!
//! Linking is **dynamic**, against `libmkl_rt` (the oneMKL Single Dynamic
//! Library). MKL is used in **LP64 mode**, so `MKL_INT` is a 32-bit integer
//! (see [`MklInt`]); ILP64 is not supported here. MKL threading is the
//! caller's responsibility via the `MKL_THREADING_LAYER` environment variable.
//!
//! All declarations are raw and unsafe to call: callers must uphold MKL's
//! contracts (valid pointers, correct buffer sizes, committed descriptors).
//! Complex arguments use `[f32; 2]` / `[f64; 2]`, which are layout-compatible
//! with `MKL_Complex8` / `MKL_Complex16` and with `#[repr(C)]`
//! `num_complex::Complex<T>`; the safe wrapper in `omnidsp-onemkl` performs
//! the slice-pointer reinterpretation.

use std::os::raw::c_void;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// MKL uses 32-bit integers in LP64 mode (`MKL_INT`).
pub type MklInt = i32;

/// MKL uses 64-bit integers for DFTI status codes and config values
/// (`MKL_LONG`).
pub type MklLong = i64;

/// Opaque type behind the DFTI descriptor pointer.
///
/// The C API declares the handle as `DFTI_DESCRIPTOR*`; the pointee is an
/// opaque MKL-internal structure, modeled here as a zero-sized opaque type.
#[repr(C)]
pub struct DftiDescriptor {
    _private: [u8; 0],
}

/// Opaque DFTI descriptor handle (`DFTI_DESCRIPTOR_HANDLE`).
///
/// A raw pointer to an opaque [`DftiDescriptor`].
pub type DftiDescriptorHandle = *mut DftiDescriptor;

// ---------------------------------------------------------------------------
// DFTI config enums
//
// Discriminants are from oneMKL 2024.2 `mkl_dfti.h` (the `DFTI_CONFIG_VALUE`
// and `DFTI_CONFIG_PARAM` enumerations). MKL stores these as `MKL_LONG`, so
// the Rust enums use `#[repr(i64)]`.
// ---------------------------------------------------------------------------

/// DFTI configuration **values** (`DFTI_CONFIG_VALUE`).
///
/// Discriminants from oneMKL 2024.2 `mkl_dfti.h`. `Real` domain enables
/// r2c/c2r transforms; `ComplexComplex` conjugate-even storage stores the
/// real-transform output as interleaved complex values (layout-compatible
/// with `Complex<T>`), matching `OmniDSP`'s r2c output convention of
/// `N/2 + 1` complex bins.
#[repr(i64)]
pub enum DftiConfigValue {
    // Precision
    /// Single precision (`f32`).
    Single = 35,
    /// Double precision (`f64`).
    Double = 36,
    // Domain
    /// Complex forward domain.
    Complex = 32,
    /// Real forward domain (enables r2c/c2r transforms).
    Real = 33,
    // Placement
    /// Out-of-place transform (distinct input and output buffers).
    NotInplace = 44,
    // Conjugate-even storage
    /// Store conjugate-even output as interleaved complex values.
    ComplexComplex = 39,
}

/// DFTI configuration **parameters** (`DFTI_CONFIG_PARAM`).
///
/// Discriminants from oneMKL 2024.2 `mkl_dfti.h`.
#[repr(i64)]
pub enum DftiConfigParam {
    /// Forward-transform domain (`DFTI_FORWARD_DOMAIN`).
    ForwardDomain = 0,
    /// Transform dimension / rank (`DFTI_DIMENSION`).
    Dimension = 1,
    /// Transform lengths (`DFTI_LENGTHS`).
    Lengths = 2,
    /// Precision (`DFTI_PRECISION`).
    Precision = 3,
    /// Forward-transform scale factor (`DFTI_FORWARD_SCALE`).
    ForwardScale = 4,
    /// Backward-transform scale factor (`DFTI_BACKWARD_SCALE`).
    BackwardScale = 5,
    /// In-place vs. out-of-place placement (`DFTI_PLACEMENT`).
    Placement = 11,
    /// Conjugate-even storage layout (`DFTI_CONJUGATE_EVEN_STORAGE`).
    ///
    /// Value `10` per `mkl_dfti.h` — note `18` is `DFTI_ORDERING`, a different
    /// parameter; routing the `COMPLEX_COMPLEX` value there yields
    /// `DFTI_INCONSISTENT_CONFIGURATION` at `DftiSetValue`.
    ConjugateEvenStorage = 10,
}

// ---------------------------------------------------------------------------
// DFTI functions (6)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Allocate and initialize a DFTI descriptor.
    pub fn DftiCreateDescriptor(
        handle: *mut DftiDescriptorHandle,
        precision: DftiConfigValue,
        domain: DftiConfigValue,
        dimension: MklLong,
        length: MklLong,
    ) -> MklLong;

    /// Set a DFTI configuration value.
    ///
    /// This is a variadic C function: the caller casts the trailing config
    /// argument to the appropriate type (`f32`, `f64`, `i64`, or a config
    /// enum) for the given [`DftiConfigParam`].
    pub fn DftiSetValue(
        handle: DftiDescriptorHandle,
        param: DftiConfigParam,
        ... // variadic — MKL uses this for heterogeneous config
    ) -> MklLong;

    /// Commit a descriptor, making it ready for compute calls.
    pub fn DftiCommitDescriptor(handle: DftiDescriptorHandle) -> MklLong;

    /// Compute a forward transform.
    pub fn DftiComputeForward(
        handle: DftiDescriptorHandle,
        input: *mut c_void,
        output: *mut c_void,
    ) -> MklLong;

    /// Compute a backward transform.
    pub fn DftiComputeBackward(
        handle: DftiDescriptorHandle,
        input: *mut c_void,
        output: *mut c_void,
    ) -> MklLong;

    /// Free a DFTI descriptor and null out the handle.
    pub fn DftiFreeDescriptor(handle: *mut DftiDescriptorHandle) -> MklLong;
}

// ---------------------------------------------------------------------------
// VM (Vector Math) functions (8)
//
// Complex arguments use `[f32; 2]` (`MKL_Complex8`) and `[f64; 2]`
// (`MKL_Complex16`).
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Elementwise multiply of two `f32` vectors: `r = a * b`.
    pub fn vsMul(n: MklInt, a: *const f32, b: *const f32, r: *mut f32);
    /// Elementwise multiply of two `f64` vectors: `r = a * b`.
    pub fn vdMul(n: MklInt, a: *const f64, b: *const f64, r: *mut f64);

    /// Elementwise add of two `f32` vectors: `r = a + b`.
    pub fn vsAdd(n: MklInt, a: *const f32, b: *const f32, r: *mut f32);
    /// Elementwise add of two `f64` vectors: `r = a + b`.
    pub fn vdAdd(n: MklInt, a: *const f64, b: *const f64, r: *mut f64);

    /// Elementwise multiply of two complex-`f32` vectors: `r = a * b`.
    pub fn vcMul(n: MklInt, a: *const [f32; 2], b: *const [f32; 2], r: *mut [f32; 2]);
    /// Elementwise multiply of two complex-`f64` vectors: `r = a * b`.
    pub fn vzMul(n: MklInt, a: *const [f64; 2], b: *const [f64; 2], r: *mut [f64; 2]);

    /// Elementwise conjugate of a complex-`f32` vector: `r = conj(a)`.
    pub fn vcConj(n: MklInt, a: *const [f32; 2], r: *mut [f32; 2]);
    /// Elementwise conjugate of a complex-`f64` vector: `r = conj(a)`.
    pub fn vzConj(n: MklInt, a: *const [f64; 2], r: *mut [f64; 2]);
}

// ---------------------------------------------------------------------------
// VM computation mode (mkl_vml_defines.h / mkl_vml_functions.h, oneMKL 2024.x)
//
// Accuracy modes for the VM (vector math) functions. Multiply/add/conjugate are
// exact, so these do not change their *result* — they are a tuning knob: a
// lower-accuracy / looser mode can shed per-call frontend overhead.
// ---------------------------------------------------------------------------

/// VM high-accuracy mode (the oneMKL default).
pub const VML_HA: u32 = 0x0000_0001;
/// VM low-accuracy mode.
pub const VML_LA: u32 = 0x0000_0002;
/// VM enhanced-performance (lowest-accuracy) mode.
pub const VML_EP: u32 = 0x0000_0003;

unsafe extern "C" {
    /// Set the VM computation mode, returning the previous mode.
    pub fn vmlSetMode(mode: u32) -> u32;
    /// Return the current VM computation mode.
    pub fn vmlGetMode() -> u32;
}

// ---------------------------------------------------------------------------
// BLAS Level-1 functions (6)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Dot product of two `f32` vectors.
    pub fn cblas_sdot(n: MklInt, x: *const f32, incx: MklInt, y: *const f32, incy: MklInt) -> f32;
    /// Dot product of two `f64` vectors.
    pub fn cblas_ddot(n: MklInt, x: *const f64, incx: MklInt, y: *const f64, incy: MklInt) -> f64;

    /// Scale an `f32` vector in place: `x = alpha * x`.
    pub fn cblas_sscal(n: MklInt, alpha: f32, x: *mut f32, incx: MklInt);
    /// Scale an `f64` vector in place: `x = alpha * x`.
    pub fn cblas_dscal(n: MklInt, alpha: f64, x: *mut f64, incx: MklInt);

    /// AXPY for `f32`: `y += alpha * x`.
    pub fn cblas_saxpy(
        n: MklInt,
        alpha: f32,
        x: *const f32,
        incx: MklInt,
        y: *mut f32,
        incy: MklInt,
    );
    /// AXPY for `f64`: `y += alpha * x`.
    pub fn cblas_daxpy(
        n: MklInt,
        alpha: f64,
        x: *const f64,
        incx: MklInt,
        y: *mut f64,
        incy: MklInt,
    );

    /// Unconjugated complex-`f32` dot product, result via out-pointer.
    pub fn cblas_cdotu_sub(
        n: MklInt,
        x: *const [f32; 2],
        incx: MklInt,
        y: *const [f32; 2],
        incy: MklInt,
        dotu: *mut [f32; 2],
    );
    /// Unconjugated complex-`f64` dot product, result via out-pointer.
    pub fn cblas_zdotu_sub(
        n: MklInt,
        x: *const [f64; 2],
        incx: MklInt,
        y: *const [f64; 2],
        incy: MklInt,
        dotu: *mut [f64; 2],
    );
}

// ---------------------------------------------------------------------------
// VS (convolution / correlation) opaque task types
// ---------------------------------------------------------------------------

/// Opaque convolution task descriptor.
#[repr(C)]
pub struct VslConvTask {
    _private: [u8; 0],
}

/// Opaque correlation task descriptor.
#[repr(C)]
pub struct VslCorrTask {
    _private: [u8; 0],
}

/// Pointer to a convolution task descriptor (`VSLConvTaskPtr`).
pub type VSLConvTaskPtr = *mut VslConvTask;

/// Pointer to a correlation task descriptor (`VSLCorrTaskPtr`).
pub type VSLCorrTaskPtr = *mut VslCorrTask;

// ---------------------------------------------------------------------------
// VS mode and status constants
//
// Values from oneMKL 2024.2 `mkl_vsl_defines.h`.
// ---------------------------------------------------------------------------

/// Convolution: compute via fast Fourier transform (`VSL_CONV_MODE_FFT`).
pub const VSL_CONV_MODE_FFT: MklInt = 0;
/// Convolution: compute via the direct algorithm (`VSL_CONV_MODE_DIRECT`).
pub const VSL_CONV_MODE_DIRECT: MklInt = 1;
/// Convolution: automatically choose direct or FFT (`VSL_CONV_MODE_AUTO`).
pub const VSL_CONV_MODE_AUTO: MklInt = 2;

/// Correlation: compute via fast Fourier transform (`VSL_CORR_MODE_FFT`).
pub const VSL_CORR_MODE_FFT: MklInt = 0;
/// Correlation: compute via the direct algorithm (`VSL_CORR_MODE_DIRECT`).
pub const VSL_CORR_MODE_DIRECT: MklInt = 1;
/// Correlation: automatically choose direct or FFT (`VSL_CORR_MODE_AUTO`).
pub const VSL_CORR_MODE_AUTO: MklInt = 2;

/// VS operation completed successfully (`VSL_STATUS_OK`).
pub const VSL_STATUS_OK: i32 = 0;

// ---------------------------------------------------------------------------
// VS convolution functions (4)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Construct a 1-D single-precision convolution task.
    pub fn vslsConvNewTask1D(
        task: *mut VSLConvTaskPtr,
        mode: MklInt,
        xshape: MklInt,
        yshape: MklInt,
        zshape: MklInt,
    ) -> i32;

    /// Construct a 1-D double-precision convolution task.
    pub fn vsldConvNewTask1D(
        task: *mut VSLConvTaskPtr,
        mode: MklInt,
        xshape: MklInt,
        yshape: MklInt,
        zshape: MklInt,
    ) -> i32;

    /// Execute a 1-D single-precision convolution task.
    pub fn vslsConvExec1D(
        task: VSLConvTaskPtr,
        x: *const f32,
        xstride: MklInt,
        y: *const f32,
        ystride: MklInt,
        z: *mut f32,
        zstride: MklInt,
    ) -> i32;

    /// Execute a 1-D double-precision convolution task.
    pub fn vsldConvExec1D(
        task: VSLConvTaskPtr,
        x: *const f64,
        xstride: MklInt,
        y: *const f64,
        ystride: MklInt,
        z: *mut f64,
        zstride: MklInt,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// VS correlation functions (4)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Construct a 1-D single-precision correlation task.
    pub fn vslsCorrNewTask1D(
        task: *mut VSLCorrTaskPtr,
        mode: MklInt,
        xshape: MklInt,
        yshape: MklInt,
        zshape: MklInt,
    ) -> i32;

    /// Construct a 1-D double-precision correlation task.
    pub fn vsldCorrNewTask1D(
        task: *mut VSLCorrTaskPtr,
        mode: MklInt,
        xshape: MklInt,
        yshape: MklInt,
        zshape: MklInt,
    ) -> i32;

    /// Execute a 1-D single-precision correlation task.
    pub fn vslsCorrExec1D(
        task: VSLCorrTaskPtr,
        x: *const f32,
        xstride: MklInt,
        y: *const f32,
        ystride: MklInt,
        z: *mut f32,
        zstride: MklInt,
    ) -> i32;

    /// Execute a 1-D double-precision correlation task.
    pub fn vsldCorrExec1D(
        task: VSLCorrTaskPtr,
        x: *const f64,
        xstride: MklInt,
        y: *const f64,
        ystride: MklInt,
        z: *mut f64,
        zstride: MklInt,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// VS destructors (2)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Destroy a convolution task and free its resources.
    pub fn vslConvDeleteTask(task: *mut VSLConvTaskPtr) -> i32;
    /// Destroy a correlation task and free its resources.
    pub fn vslCorrDeleteTask(task: *mut VSLCorrTaskPtr) -> i32;
}

// ---------------------------------------------------------------------------
// Status constants
// ---------------------------------------------------------------------------

/// DFTI status: operation completed successfully (`DFTI_NO_ERROR`).
///
/// Value from oneMKL 2024.2 `mkl_dfti.h`.
pub const DFTI_NO_ERROR: MklLong = 0;

// ---------------------------------------------------------------------------
// Smoke tests
//
// These link against `mkl_rt`, so they only run where MKL is installed; on a
// machine without MKL the link step fails and the tests do not run (the tests
// are effectively self-gating).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        DFTI_NO_ERROR, DftiConfigValue, DftiCreateDescriptor, DftiDescriptorHandle,
        DftiFreeDescriptor, VSL_CONV_MODE_AUTO, VSL_STATUS_OK, VSLConvTaskPtr, vslConvDeleteTask,
        vslsConvNewTask1D,
    };

    #[test]
    fn dfti_create_and_free() {
        unsafe {
            let mut handle: DftiDescriptorHandle = std::ptr::null_mut();
            let status = DftiCreateDescriptor(
                &raw mut handle,
                DftiConfigValue::Double,
                DftiConfigValue::Complex,
                1, // dimension
                8, // length
            );
            assert_eq!(status, DFTI_NO_ERROR, "DftiCreateDescriptor should succeed");
            assert!(!handle.is_null(), "handle should be non-null");

            let status = DftiFreeDescriptor(&raw mut handle);
            assert_eq!(status, DFTI_NO_ERROR, "DftiFreeDescriptor should succeed");
        }
    }

    #[test]
    fn vs_conv_create_and_free() {
        unsafe {
            let mut task: VSLConvTaskPtr = std::ptr::null_mut();
            let status = vslsConvNewTask1D(
                &raw mut task,
                VSL_CONV_MODE_AUTO,
                4, // xshape
                3, // yshape
                6, // zshape = 4 + 3 - 1
            );
            assert_eq!(status, VSL_STATUS_OK, "vslsConvNewTask1D should succeed");
            assert!(!task.is_null(), "task should be non-null");

            let status = vslConvDeleteTask(&raw mut task);
            assert_eq!(status, VSL_STATUS_OK, "vslConvDeleteTask should succeed");
        }
    }
}
