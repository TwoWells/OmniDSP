// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Dispatch enums that wrap concrete backend implementations.

mod dft;
mod iir;
mod vecops;

pub use dft::{DynDft, DynDftPlan};
pub use iir::{DynIir, DynIirPlan};
pub use vecops::DynVecOps;
