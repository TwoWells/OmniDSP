// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Design layer — pure math functions that produce coefficients.
//!
//! This module contains filter design functions, window coefficient generation,
//! and related utilities.  Nothing here depends on runtime primitives
//! ([`DftC2c`](crate::traits::dft::DftC2c), [`VecOps`](crate::traits::vecops::VecOps))
//! or any implementation crate.  All computation is plain arithmetic on
//! `f32`/`f64` values, suitable for use at plan-creation time.

pub mod conv;
pub mod cqt;
pub mod dft;
pub mod fir;
pub mod iir;
pub(crate) mod remez;
pub mod resample;
pub(crate) mod window;
