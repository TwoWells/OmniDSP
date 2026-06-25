// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Raw FFI declarations for Intel IPP's signal-processing domain (`ipps`).
//!
//! This `-sys` crate declares only the IPP entry points needed by
//! `omnidsp-ipp`, grouped by domain:
//!
//! - [`fft`] — FFT (power-of-two) and DFT (arbitrary length) transforms, both
//!   complex-to-complex and real-to-complex (CCS-packed), f32 and f64.
//! - [`vecops`] — elementwise vector arithmetic and complex utilities backing
//!   the `VecOps` trait surface.
//! - [`iir`] — biquad-cascade IIR (state allocation, init, apply).
//!
//! Linking is **dynamic**, against `ipps` + `ippvm` + `ippcore` (see
//! `build.rs`). IPP performs automatic runtime CPU dispatch (SSE4.2 / AVX2 /
//! AVX-512) inside `ippcore`.
//!
//! All declarations are raw and unsafe to call: callers must uphold IPP's
//! contracts (valid pointers, correct buffer sizes, initialized spec/state
//! structures). Complex arguments use `[f32; 2]` / `[f64; 2]`, which are
//! layout-compatible with IPP's `Ipp32fc` / `Ipp64fc` and with `#[repr(C)]`
//! `num_complex::Complex<T>`; the safe wrapper in `omnidsp-ipp` performs the
//! slice-pointer reinterpretation (mirroring the sibling `omnidsp-onemkl-sys`).
//! Real transforms emit their spectrum in IPP's CCS (complex-conjugate-
//! symmetric) packed format as a plain real array, so those signatures use
//! `f32` / `f64`, not the complex aliases.

use std::os::raw::{c_char, c_int};

pub mod fft;
pub mod iir;
pub mod vecops;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// IPP status / error code (`IppStatus`); `0` is success, negatives are errors.
pub type IppStatus = i32;

/// IPP work-buffer byte (`Ipp8u`). Buffers are sized via `*GetSize` /
/// `*GetStateSize` and allocated by the caller as `Vec<u8>`.
pub type Ipp8u = u8;

/// IPP single-precision real scalar (`Ipp32f`).
pub type Ipp32f = f32;

/// IPP double-precision real scalar (`Ipp64f`).
pub type Ipp64f = f64;

/// IPP algorithm hint (`IppHintAlgorithm`), passed to FFT/DFT `GetSize`/`Init`.
/// Deprecated in modern IPP — always pass [`IPP_ALG_HINT_NONE`].
pub type IppHintAlgorithm = c_int;

// ---------------------------------------------------------------------------
// Status codes (subset; values from the IPP 2021.x `ippbase.h` status enum)
// ---------------------------------------------------------------------------

/// Operation completed successfully (`ippStsNoErr`).
pub const IPP_STS_NO_ERR: IppStatus = 0;
/// A required pointer argument was null (`ippStsNullPtrErr`).
pub const IPP_STS_NULL_PTR_ERR: IppStatus = -8;
/// An argument had an invalid size (`ippStsSizeErr`).
pub const IPP_STS_SIZE_ERR: IppStatus = -6;
/// Memory allocation failed (`ippStsMemAllocErr`).
pub const IPP_STS_MEM_ALLOC_ERR: IppStatus = -4;
/// A generic bad argument was supplied (`ippStsBadArgErr`).
pub const IPP_STS_BAD_ARG_ERR: IppStatus = -1;
/// Invalid FFT order (`ippStsFftOrderErr`).
pub const IPP_STS_FFT_ORDER_ERR: IppStatus = -36;
/// Invalid FFT normalization flag (`ippStsFftFlagErr`).
pub const IPP_STS_FFT_FLAG_ERR: IppStatus = -37;
/// Spec/state structure does not match the call (`ippStsContextMatchErr`).
pub const IPP_STS_CONTEXT_MATCH_ERR: IppStatus = -17;

// ---------------------------------------------------------------------------
// FFT / DFT normalization flags (`ippsFFTInit` / `ippsDFTInit` flag argument)
// ---------------------------------------------------------------------------

/// Divide the forward transform result by `N` (`IPP_FFT_DIV_FWD_BY_N`).
pub const IPP_FFT_DIV_FWD_BY_N: c_int = 1;
/// Divide the inverse transform result by `N` (`IPP_FFT_DIV_INV_BY_N`).
pub const IPP_FFT_DIV_INV_BY_N: c_int = 2;
/// Divide both transforms by `sqrt(N)` (`IPP_FFT_DIV_BY_SQRTN`).
pub const IPP_FFT_DIV_BY_SQRTN: c_int = 4;
/// Apply no normalization in either direction (`IPP_FFT_NODIV_BY_ANY`).
pub const IPP_FFT_NODIV_BY_ANY: c_int = 8;

/// No algorithm hint (`ippAlgHintNone`) — the only non-deprecated value.
pub const IPP_ALG_HINT_NONE: IppHintAlgorithm = 0;

// ---------------------------------------------------------------------------
// Convolution / correlation algorithm selector (`IppEnum algType`)
//
// Passed to `ippsConvolve` / `ippsCrossCorrNorm` (native module overrides land
// in later tickets); declared here so the constants live with the other IPP
// enums. The low bits select the method.
// ---------------------------------------------------------------------------

/// Let IPP choose direct vs. FFT (`ippAlgAuto`).
pub const IPP_ALG_AUTO: c_int = 0;
/// Force the direct (time-domain) algorithm (`ippAlgDirect`).
pub const IPP_ALG_DIRECT: c_int = 1;
/// Force the FFT-based algorithm (`ippAlgFFT`).
pub const IPP_ALG_FFT: c_int = 2;

// ---------------------------------------------------------------------------
// Library version query
// ---------------------------------------------------------------------------

/// IPP library version descriptor (`IppLibraryVersion`).
///
/// `#[repr(C)]` matches IPP's struct layout; the Rust field names are
/// snake-cased but the order and types are faithful. The trailing `*const
/// c_char` fields are NUL-terminated C strings owned by IPP (do not free).
#[repr(C)]
pub struct IppLibraryVersion {
    /// Major version number.
    pub major: c_int,
    /// Minor version number.
    pub minor: c_int,
    /// Major build number (`majorBuild`).
    pub major_build: c_int,
    /// Build number.
    pub build: c_int,
    /// Target CPU tag (`targetCpu`), four characters.
    pub target_cpu: [c_char; 4],
    /// Library name string (`Name`).
    pub name: *const c_char,
    /// Version string (`Version`).
    pub version: *const c_char,
    /// Build date string (`BuildDate`).
    pub build_date: *const c_char,
}

unsafe extern "C" {
    /// Return a pointer to the `ipps` library version descriptor.
    ///
    /// The pointee is static, owned by IPP, and valid for the process lifetime.
    pub fn ippsGetLibVersion() -> *const IppLibraryVersion;

    /// Return the human-readable message for an [`IppStatus`] code.
    ///
    /// The returned C string is static and owned by IPP.
    pub fn ippGetStatusString(sts_code: IppStatus) -> *const c_char;
}

// ---------------------------------------------------------------------------
// Smoke test
//
// Links against `ipps`, so it only runs where IPP is installed; on a machine
// without IPP the link step fails and the test does not run (self-gating).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::ippsGetLibVersion;

    #[test]
    fn lib_version_is_available() {
        // SAFETY: `ippsGetLibVersion` takes no arguments and returns a pointer
        // to a static, process-lifetime descriptor; reading `major` through it
        // is sound once we confirm it is non-null.
        unsafe {
            let version = ippsGetLibVersion();
            assert!(
                !version.is_null(),
                "ippsGetLibVersion should return non-null"
            );
            let major = (*version).major;
            assert!(
                major > 0,
                "IPP major version should be positive, got {major}"
            );
        }
    }
}
