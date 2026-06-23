// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Primitive trait definitions (Layer 1).
//!
//! Each primitive follows the factory+plan pattern: a factory trait creates
//! configured plan objects, and plans are the execution objects.  [`VecOps`](vecops::VecOps)
//! is the exception — it is stateless and uses direct methods instead.
//!
//! Window does not have a trait — coefficient generation is pure math
//! ([`Window::coefficients`](crate::window::Window::coefficients)) and applying
//! a window is [`VecOps::mul_inplace`](vecops::VecOps::mul_inplace).
//!
//! See the [architecture docs](https://github.com/TwoWells/OmniDSP) for details.

pub mod conv;
pub mod dct;
pub mod dft;
pub mod fir;
pub mod iir;
pub mod reconfigure;
pub mod vecops;
