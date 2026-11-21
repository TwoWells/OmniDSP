// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Linear convolution declarations (`ippsConvolve*`).
//!
//! `ippsConvolve` performs a full linear convolution of two 1-D signals in a
//! single call, writing `src1Len + src2Len - 1` output samples. The algorithm
//! (direct vs. FFT) is chosen by the `alg_type` selector
//! ([`IPP_ALG_AUTO`](crate::IPP_ALG_AUTO) /
//! [`IPP_ALG_DIRECT`](crate::IPP_ALG_DIRECT) / [`IPP_ALG_FFT`](crate::IPP_ALG_FFT)).
//!
//! The lifecycle is two-step, like the FFT/DFT engines:
//!
//! 1. `ippsConvolveGetBufferSize` returns the byte size of the external work
//!    buffer for the given input lengths, element type (`data_type`, one of
//!    [`IPP_32F`](crate::IPP_32F) / [`IPP_64F`](crate::IPP_64F)), and algorithm.
//! 2. `ippsConvolve_{32f,64f}` runs the convolution using a caller-allocated
//!    buffer of that size. The buffer is scratch — IPP writes into it on every
//!    call, so a shared plan must serialize access to it.

use std::os::raw::c_int;

use crate::{Ipp8u, Ipp32f, Ipp64f, IppStatus};

unsafe extern "C" {
    /// Byte size of the work buffer `ippsConvolve` needs for inputs of
    /// `src1_len` / `src2_len`, element type `data_type`
    /// ([`IPP_32F`](crate::IPP_32F) / [`IPP_64F`](crate::IPP_64F)), and algorithm
    /// `alg_type`.
    pub fn ippsConvolveGetBufferSize(
        src1_len: c_int,
        src2_len: c_int,
        data_type: c_int,
        alg_type: c_int,
        p_buffer_size: *mut c_int,
    ) -> IppStatus;

    /// Full linear convolution of two real f32 signals: `pDst = pSrc1 ∗ pSrc2`,
    /// `src1Len + src2Len - 1` output samples. `p_buffer` is scratch sized by
    /// `ippsConvolveGetBufferSize`.
    pub fn ippsConvolve_32f(
        p_src1: *const Ipp32f,
        src1_len: c_int,
        p_src2: *const Ipp32f,
        src2_len: c_int,
        p_dst: *mut Ipp32f,
        alg_type: c_int,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;

    /// Full linear convolution of two real f64 signals: `pDst = pSrc1 ∗ pSrc2`,
    /// `src1Len + src2Len - 1` output samples. `p_buffer` is scratch sized by
    /// `ippsConvolveGetBufferSize`.
    pub fn ippsConvolve_64f(
        p_src1: *const Ipp64f,
        src1_len: c_int,
        p_src2: *const Ipp64f,
        src2_len: c_int,
        p_dst: *mut Ipp64f,
        alg_type: c_int,
        p_buffer: *mut Ipp8u,
    ) -> IppStatus;
}
