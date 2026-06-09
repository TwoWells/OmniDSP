// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Composite modules (Layer 2).
//!
//! Modules are generic over the DFT primitives and
//! [`VecOps`](crate::traits::vecops::VecOps).  The real-input modules (Conv,
//! FIR, `CrossCorr`, DCT) compose [`DftR2c`](crate::traits::dft::DftR2c) +
//! [`DftC2r`](crate::traits::dft::DftC2r) (ADR-009 §6); Hilbert mixes
//! [`DftR2c`](crate::traits::dft::DftR2c) + [`DftC2c`](crate::traits::dft::DftC2c)
//! (complex analytic output); CQT composes [`DftC2c`](crate::traits::dft::DftC2c)
//! (the single-FFT oracle, pending the multirate capstone).  A vendor that
//! provides these primitives plus `VecOps` gets every module for free.

pub mod conv;
pub mod cqt;
pub mod dct;
pub mod fir;
pub mod hilbert;
pub mod resample;
pub mod xcorr;
