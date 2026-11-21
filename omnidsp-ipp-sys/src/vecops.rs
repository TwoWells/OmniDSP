// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Elementwise vector arithmetic and complex utilities.
//!
//! These declarations back the `VecOps` trait surface (`add`, `mul`, `scale`,
//! `dot`, `cmul`, `cdot`, the in-place variants, `conj_inplace`, `mag_sq`,
//! `mag`, `real_to_complex`, `extract_real`).
//!
//! Real operands use `f32` / `f64`; complex operands use the interleaved
//! `[f32; 2]` / `[f64; 2]` aliases. The trailing `len` is the element count.
//! Each returns an [`IppStatus`]; the safe wrapper maps non-zero codes to an
//! error.

use std::os::raw::c_int;

use crate::{Ipp32f, Ipp64f, IppStatus};

// ---------------------------------------------------------------------------
// Real elementwise arithmetic
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// `dst = src1 + src2` (f32).
    pub fn ippsAdd_32f(
        p_src1: *const Ipp32f,
        p_src2: *const Ipp32f,
        p_dst: *mut Ipp32f,
        len: c_int,
    ) -> IppStatus;
    /// `dst = src1 + src2` (f64).
    pub fn ippsAdd_64f(
        p_src1: *const Ipp64f,
        p_src2: *const Ipp64f,
        p_dst: *mut Ipp64f,
        len: c_int,
    ) -> IppStatus;

    /// `src_dst += src` in place (f32).
    pub fn ippsAdd_32f_I(p_src: *const Ipp32f, p_src_dst: *mut Ipp32f, len: c_int) -> IppStatus;
    /// `src_dst += src` in place (f64).
    pub fn ippsAdd_64f_I(p_src: *const Ipp64f, p_src_dst: *mut Ipp64f, len: c_int) -> IppStatus;

    /// `dst = src1 * src2` (f32).
    pub fn ippsMul_32f(
        p_src1: *const Ipp32f,
        p_src2: *const Ipp32f,
        p_dst: *mut Ipp32f,
        len: c_int,
    ) -> IppStatus;
    /// `dst = src1 * src2` (f64).
    pub fn ippsMul_64f(
        p_src1: *const Ipp64f,
        p_src2: *const Ipp64f,
        p_dst: *mut Ipp64f,
        len: c_int,
    ) -> IppStatus;

    /// `src_dst *= src` in place (f32).
    pub fn ippsMul_32f_I(p_src: *const Ipp32f, p_src_dst: *mut Ipp32f, len: c_int) -> IppStatus;
    /// `src_dst *= src` in place (f64).
    pub fn ippsMul_64f_I(p_src: *const Ipp64f, p_src_dst: *mut Ipp64f, len: c_int) -> IppStatus;

    /// `dst = src * val` (scale by a constant, f32).
    pub fn ippsMulC_32f(
        p_src: *const Ipp32f,
        val: Ipp32f,
        p_dst: *mut Ipp32f,
        len: c_int,
    ) -> IppStatus;
    /// `dst = src * val` (scale by a constant, f64).
    pub fn ippsMulC_64f(
        p_src: *const Ipp64f,
        val: Ipp64f,
        p_dst: *mut Ipp64f,
        len: c_int,
    ) -> IppStatus;

    /// `src_dst *= val` in place (scale by a constant, f32).
    pub fn ippsMulC_32f_I(val: Ipp32f, p_src_dst: *mut Ipp32f, len: c_int) -> IppStatus;
    /// `src_dst *= val` in place (scale by a constant, f64).
    pub fn ippsMulC_64f_I(val: Ipp64f, p_src_dst: *mut Ipp64f, len: c_int) -> IppStatus;

    /// Real dot product `Σ src1·src2`, result via out-pointer (f32).
    pub fn ippsDotProd_32f(
        p_src1: *const Ipp32f,
        p_src2: *const Ipp32f,
        len: c_int,
        p_dp: *mut Ipp32f,
    ) -> IppStatus;
    /// Real dot product `Σ src1·src2`, result via out-pointer (f64).
    pub fn ippsDotProd_64f(
        p_src1: *const Ipp64f,
        p_src2: *const Ipp64f,
        len: c_int,
        p_dp: *mut Ipp64f,
    ) -> IppStatus;
}

// ---------------------------------------------------------------------------
// Complex elementwise arithmetic
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// `dst = src1 * src2` (complex f32).
    pub fn ippsMul_32fc(
        p_src1: *const [Ipp32f; 2],
        p_src2: *const [Ipp32f; 2],
        p_dst: *mut [Ipp32f; 2],
        len: c_int,
    ) -> IppStatus;
    /// `dst = src1 * src2` (complex f64).
    pub fn ippsMul_64fc(
        p_src1: *const [Ipp64f; 2],
        p_src2: *const [Ipp64f; 2],
        p_dst: *mut [Ipp64f; 2],
        len: c_int,
    ) -> IppStatus;

    /// `src_dst *= src` in place (complex f32).
    pub fn ippsMul_32fc_I(
        p_src: *const [Ipp32f; 2],
        p_src_dst: *mut [Ipp32f; 2],
        len: c_int,
    ) -> IppStatus;
    /// `src_dst *= src` in place (complex f64).
    pub fn ippsMul_64fc_I(
        p_src: *const [Ipp64f; 2],
        p_src_dst: *mut [Ipp64f; 2],
        len: c_int,
    ) -> IppStatus;

    /// Unconjugated complex dot product `Σ src1·src2`, result via out-pointer
    /// (complex f32).
    pub fn ippsDotProd_32fc(
        p_src1: *const [Ipp32f; 2],
        p_src2: *const [Ipp32f; 2],
        len: c_int,
        p_dp: *mut [Ipp32f; 2],
    ) -> IppStatus;
    /// Unconjugated complex dot product `Σ src1·src2`, result via out-pointer
    /// (complex f64).
    pub fn ippsDotProd_64fc(
        p_src1: *const [Ipp64f; 2],
        p_src2: *const [Ipp64f; 2],
        len: c_int,
        p_dp: *mut [Ipp64f; 2],
    ) -> IppStatus;

    /// `src_dst = conj(src_dst)` in place (complex f32).
    pub fn ippsConj_32fc_I(p_src_dst: *mut [Ipp32f; 2], len: c_int) -> IppStatus;
    /// `src_dst = conj(src_dst)` in place (complex f64).
    pub fn ippsConj_64fc_I(p_src_dst: *mut [Ipp64f; 2], len: c_int) -> IppStatus;
}

// ---------------------------------------------------------------------------
// Complex ↔ real utilities
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// `dst = |src|²` — power spectrum / squared magnitude (complex f32 → f32).
    pub fn ippsPowerSpectr_32fc(
        p_src: *const [Ipp32f; 2],
        p_dst: *mut Ipp32f,
        len: c_int,
    ) -> IppStatus;
    /// `dst = |src|²` — power spectrum / squared magnitude (complex f64 → f64).
    pub fn ippsPowerSpectr_64fc(
        p_src: *const [Ipp64f; 2],
        p_dst: *mut Ipp64f,
        len: c_int,
    ) -> IppStatus;

    /// `dst = |src|` — magnitude (complex f32 → f32).
    pub fn ippsMagnitude_32fc(
        p_src: *const [Ipp32f; 2],
        p_dst: *mut Ipp32f,
        len: c_int,
    ) -> IppStatus;
    /// `dst = |src|` — magnitude (complex f64 → f64).
    pub fn ippsMagnitude_64fc(
        p_src: *const [Ipp64f; 2],
        p_dst: *mut Ipp64f,
        len: c_int,
    ) -> IppStatus;

    /// Combine real + imaginary parts into a complex vector (f32). `p_src_im`
    /// may be null for a purely real input.
    pub fn ippsRealToCplx_32f(
        p_src_re: *const Ipp32f,
        p_src_im: *const Ipp32f,
        p_dst: *mut [Ipp32f; 2],
        len: c_int,
    ) -> IppStatus;
    /// Combine real + imaginary parts into a complex vector (f64). `p_src_im`
    /// may be null for a purely real input.
    pub fn ippsRealToCplx_64f(
        p_src_re: *const Ipp64f,
        p_src_im: *const Ipp64f,
        p_dst: *mut [Ipp64f; 2],
        len: c_int,
    ) -> IppStatus;

    /// Extract the real part of a complex vector (complex f32 → f32).
    pub fn ippsReal_32fc(p_src: *const [Ipp32f; 2], p_dst_re: *mut Ipp32f, len: c_int)
    -> IppStatus;
    /// Extract the real part of a complex vector (complex f64 → f64).
    pub fn ippsReal_64fc(p_src: *const [Ipp64f; 2], p_dst_re: *mut Ipp64f, len: c_int)
    -> IppStatus;
}
