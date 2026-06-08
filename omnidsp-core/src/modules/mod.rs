// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Composite modules (Layer 2).
//!
//! Modules are generic over [`DftC2c`](crate::traits::dft::DftC2c) and
//! [`VecOps`](crate::traits::vecops::VecOps).  A vendor that provides just
//! these two primitives gets every module for free.

pub mod conv;
pub mod cqt;
pub mod dct;
pub mod fir;
pub mod hilbert;
pub mod resample;
pub mod xcorr;
