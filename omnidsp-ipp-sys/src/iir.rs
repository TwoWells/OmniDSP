// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Biquad-cascade IIR declarations (`ippsIIR*_BiQuad*`).
//!
//! The lifecycle mirrors the FFT/DFT pattern:
//!
//! 1. `GetStateSize_BiQuad` returns the byte size of the state buffer for a
//!    cascade of `num_bq` second-order sections.
//! 2. `Init_BiQuad` initializes a state into the caller's buffer (double-pointer
//!    form, like FFT `Init`). `p_taps` holds six coefficients per section in
//!    `{b0, b1, b2, a0, a1, a2}` order (`a0` normalized to 1); `p_dly_line` is
//!    the initial delay line (two values per section) or null for zeros.
//! 3. `IIR` / `IIR_I` apply the cascade to a block, advancing the delay line in
//!    the state so successive blocks stream continuously.
//!
//! The opaque `IppsIIRState_{32f,64f}` is modeled as `c_void`.

use std::os::raw::{c_int, c_void};

use crate::{Ipp8u, Ipp32f, Ipp64f, IppStatus};

unsafe extern "C" {
    /// Byte size of an `num_bq`-section biquad-cascade state (f32).
    pub fn ippsIIRGetStateSize_BiQuad_32f(num_bq: c_int, p_buffer_size: *mut c_int) -> IppStatus;
    /// Byte size of an `num_bq`-section biquad-cascade state (f64).
    pub fn ippsIIRGetStateSize_BiQuad_64f(num_bq: c_int, p_buffer_size: *mut c_int) -> IppStatus;

    /// Initialize a biquad-cascade state into `p_buf` (double-pointer form, f32).
    pub fn ippsIIRInit_BiQuad_32f(
        pp_state: *mut *mut c_void,
        p_taps: *const Ipp32f,
        num_bq: c_int,
        p_dly_line: *const Ipp32f,
        p_buf: *mut Ipp8u,
    ) -> IppStatus;
    /// Initialize a biquad-cascade state into `p_buf` (double-pointer form, f64).
    pub fn ippsIIRInit_BiQuad_64f(
        pp_state: *mut *mut c_void,
        p_taps: *const Ipp64f,
        num_bq: c_int,
        p_dly_line: *const Ipp64f,
        p_buf: *mut Ipp8u,
    ) -> IppStatus;

    /// Apply the biquad cascade to a block: `dst = H(src)` (f32).
    pub fn ippsIIR_32f(
        p_src: *const Ipp32f,
        p_dst: *mut Ipp32f,
        len: c_int,
        p_state: *mut c_void,
    ) -> IppStatus;
    /// Apply the biquad cascade to a block: `dst = H(src)` (f64).
    pub fn ippsIIR_64f(
        p_src: *const Ipp64f,
        p_dst: *mut Ipp64f,
        len: c_int,
        p_state: *mut c_void,
    ) -> IppStatus;

    /// Apply the biquad cascade in place: `src_dst = H(src_dst)` (f32).
    pub fn ippsIIR_32f_I(p_src_dst: *mut Ipp32f, len: c_int, p_state: *mut c_void) -> IppStatus;
    /// Apply the biquad cascade in place: `src_dst = H(src_dst)` (f64).
    pub fn ippsIIR_64f_I(p_src_dst: *mut Ipp64f, len: c_int, p_state: *mut c_void) -> IppStatus;
}
