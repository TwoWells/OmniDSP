// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Re-export of the dispatch factory traits [`CreatePlan`] and [`CreateProc`]
//! from `omnidsp-core`.
//!
//! They are the stateless-Plan and stateful-Processor halves of the backend
//! surface, named symmetrically so a backend author or `dyn` consumer reaches
//! both from the facade.

pub use omnidsp_core::create::{CreatePlan, CreateProc};
