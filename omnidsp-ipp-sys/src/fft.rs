// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! FFT (power-of-two) and DFT (arbitrary length) transform declarations.
//!
//! IPP exposes two engines with parallel surfaces:
//!
//! - **FFT** — `order`-based (length `2^order`). `Init` takes a **double
//!   pointer** (`ppFFTSpec`): IPP writes the spec pointer into the caller's
//!   spec buffer at an internal offset.
//! - **DFT** — arbitrary `length`. `Init` takes a **single pointer**
//!   (`pDFTSpec`): the caller's buffer *is* the spec.
//!
//! Both follow the three-buffer pattern: `GetSize` returns `spec_size`,
//! `spec_buffer_size` (a temporary needed only during `Init`), and
//! `buffer_size` (the work buffer reused across `Fwd`/`Inv` calls). The spec
//! and work buffers are kept alive for the plan's lifetime; the init buffer is
//! freed once `Init` returns.
//!
//! Complex transforms (`*_C_*`, `*_CToC_*`) use the interleaved complex aliases
//! `[f32; 2]` / `[f64; 2]`. Real transforms (`*_R_*`) emit / consume their
//! spectrum in IPP's CCS packed format — a plain real array of `N + 2` values
//! holding `N/2 + 1` interleaved complex bins, layout-compatible with
//! `[Complex<T>; N/2 + 1]` — so their `Fwd`/`Inv` signatures use `f32` / `f64`.
//!
//! The opaque spec structures are modeled as `c_void`; precision/domain
//! correctness is the safe wrapper's responsibility.

use std::os::raw::{c_int, c_void};

use crate::{Ipp8u, Ipp32f, Ipp64f, IppHintAlgorithm, IppStatus};

// ---------------------------------------------------------------------------
// FFT — complex-to-complex (power-of-two, `order`)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Single-precision complex FFT: query the three buffer sizes.
    pub fn ippsFFTGetSize_C_32fc(
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec_size: *mut c_int,
        p_spec_buffer_size: *mut c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Single-precision complex FFT: initialize the spec (double-pointer form).
    pub fn ippsFFTInit_C_32fc(
        pp_fft_spec: *mut *mut c_void,
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec: *mut Ipp8u,
        p_spec_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Single-precision complex FFT: forward transform.
    pub fn ippsFFTFwd_CToC_32fc(
        p_src: *const [Ipp32f; 2],
        p_dst: *mut [Ipp32f; 2],
        p_fft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Single-precision complex FFT: inverse transform.
    pub fn ippsFFTInv_CToC_32fc(
        p_src: *const [Ipp32f; 2],
        p_dst: *mut [Ipp32f; 2],
        p_fft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision complex FFT: query the three buffer sizes.
    pub fn ippsFFTGetSize_C_64fc(
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec_size: *mut c_int,
        p_spec_buffer_size: *mut c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Double-precision complex FFT: initialize the spec (double-pointer form).
    pub fn ippsFFTInit_C_64fc(
        pp_fft_spec: *mut *mut c_void,
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec: *mut Ipp8u,
        p_spec_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision complex FFT: forward transform.
    pub fn ippsFFTFwd_CToC_64fc(
        p_src: *const [Ipp64f; 2],
        p_dst: *mut [Ipp64f; 2],
        p_fft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision complex FFT: inverse transform.
    pub fn ippsFFTInv_CToC_64fc(
        p_src: *const [Ipp64f; 2],
        p_dst: *mut [Ipp64f; 2],
        p_fft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;
}

// ---------------------------------------------------------------------------
// FFT — real-to-CCS / CCS-to-real (power-of-two, `order`)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Single-precision real FFT: query the three buffer sizes.
    pub fn ippsFFTGetSize_R_32f(
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec_size: *mut c_int,
        p_spec_buffer_size: *mut c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Single-precision real FFT: initialize the spec (double-pointer form).
    pub fn ippsFFTInit_R_32f(
        pp_fft_spec: *mut *mut c_void,
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec: *mut Ipp8u,
        p_spec_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Single-precision real FFT: forward transform (real → CCS-packed).
    pub fn ippsFFTFwd_RToCCS_32f(
        p_src: *const Ipp32f,
        p_dst: *mut Ipp32f,
        p_fft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Single-precision real FFT: inverse transform (CCS-packed → real).
    pub fn ippsFFTInv_CCSToR_32f(
        p_src: *const Ipp32f,
        p_dst: *mut Ipp32f,
        p_fft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision real FFT: query the three buffer sizes.
    pub fn ippsFFTGetSize_R_64f(
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec_size: *mut c_int,
        p_spec_buffer_size: *mut c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Double-precision real FFT: initialize the spec (double-pointer form).
    pub fn ippsFFTInit_R_64f(
        pp_fft_spec: *mut *mut c_void,
        order: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec: *mut Ipp8u,
        p_spec_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision real FFT: forward transform (real → CCS-packed).
    pub fn ippsFFTFwd_RToCCS_64f(
        p_src: *const Ipp64f,
        p_dst: *mut Ipp64f,
        p_fft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision real FFT: inverse transform (CCS-packed → real).
    pub fn ippsFFTInv_CCSToR_64f(
        p_src: *const Ipp64f,
        p_dst: *mut Ipp64f,
        p_fft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;
}

// ---------------------------------------------------------------------------
// DFT — complex-to-complex (arbitrary `length`)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Single-precision complex DFT: query the three buffer sizes.
    pub fn ippsDFTGetSize_C_32fc(
        length: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec_size: *mut c_int,
        p_spec_buffer_size: *mut c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Single-precision complex DFT: initialize the spec (single-pointer form).
    pub fn ippsDFTInit_C_32fc(
        length: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_dft_spec: *mut c_void,
        p_mem_init: *mut Ipp8u,
    ) -> IppStatus;

    /// Single-precision complex DFT: forward transform.
    pub fn ippsDFTFwd_CToC_32fc(
        p_src: *const [Ipp32f; 2],
        p_dst: *mut [Ipp32f; 2],
        p_dft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Single-precision complex DFT: inverse transform.
    pub fn ippsDFTInv_CToC_32fc(
        p_src: *const [Ipp32f; 2],
        p_dst: *mut [Ipp32f; 2],
        p_dft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision complex DFT: query the three buffer sizes.
    pub fn ippsDFTGetSize_C_64fc(
        length: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec_size: *mut c_int,
        p_spec_buffer_size: *mut c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Double-precision complex DFT: initialize the spec (single-pointer form).
    pub fn ippsDFTInit_C_64fc(
        length: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_dft_spec: *mut c_void,
        p_mem_init: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision complex DFT: forward transform.
    pub fn ippsDFTFwd_CToC_64fc(
        p_src: *const [Ipp64f; 2],
        p_dst: *mut [Ipp64f; 2],
        p_dft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision complex DFT: inverse transform.
    pub fn ippsDFTInv_CToC_64fc(
        p_src: *const [Ipp64f; 2],
        p_dst: *mut [Ipp64f; 2],
        p_dft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;
}

// ---------------------------------------------------------------------------
// DFT — real-to-CCS / CCS-to-real (arbitrary `length`)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Single-precision real DFT: query the three buffer sizes.
    pub fn ippsDFTGetSize_R_32f(
        length: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec_size: *mut c_int,
        p_spec_buffer_size: *mut c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Single-precision real DFT: initialize the spec (single-pointer form).
    pub fn ippsDFTInit_R_32f(
        length: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_dft_spec: *mut c_void,
        p_mem_init: *mut Ipp8u,
    ) -> IppStatus;

    /// Single-precision real DFT: forward transform (real → CCS-packed).
    pub fn ippsDFTFwd_RToCCS_32f(
        p_src: *const Ipp32f,
        p_dst: *mut Ipp32f,
        p_dft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Single-precision real DFT: inverse transform (CCS-packed → real).
    pub fn ippsDFTInv_CCSToR_32f(
        p_src: *const Ipp32f,
        p_dst: *mut Ipp32f,
        p_dft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision real DFT: query the three buffer sizes.
    pub fn ippsDFTGetSize_R_64f(
        length: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_spec_size: *mut c_int,
        p_spec_buffer_size: *mut c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Double-precision real DFT: initialize the spec (single-pointer form).
    pub fn ippsDFTInit_R_64f(
        length: c_int,
        flag: c_int,
        hint: IppHintAlgorithm,
        p_dft_spec: *mut c_void,
        p_mem_init: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision real DFT: forward transform (real → CCS-packed).
    pub fn ippsDFTFwd_RToCCS_64f(
        p_src: *const Ipp64f,
        p_dst: *mut Ipp64f,
        p_dft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Double-precision real DFT: inverse transform (CCS-packed → real).
    pub fn ippsDFTInv_CCSToR_64f(
        p_src: *const Ipp64f,
        p_dst: *mut Ipp64f,
        p_dft_spec: *const c_void,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;
}
