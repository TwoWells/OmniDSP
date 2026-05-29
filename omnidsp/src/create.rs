// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Per-module factory traits for the dispatch layer.
//!
//! Each module has its own `Create*` trait with an associated plan type.
//! Adding a new module means adding a new trait — no existing trait
//! changes, no vendor breakage.

use omnidsp_core::design::cqt::CqtSpec;
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::error::Result;
use omnidsp_core::traits::conv::ConvSpec;
use omnidsp_core::traits::fir::FirSpec;

/// Factory trait for creating convolution plans.
///
/// Implementations provide a specific convolution plan type and a method
/// to create it from a [`ConvSpec`].
pub trait CreateConv<T> {
    /// The convolution plan type produced by this factory.
    type Conv;

    /// Create a convolution plan from the given specification.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    fn create_conv(&self, spec: &ConvSpec<T>) -> Result<Self::Conv>;
}

/// Factory trait for creating FIR filter plans.
///
/// Implementations provide a specific FIR plan type and a method to
/// create it from a [`FirSpec`].
pub trait CreateFir<T> {
    /// The FIR filter plan type produced by this factory.
    type Fir;

    /// Create a FIR filter plan from the given specification.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    fn create_fir(&self, spec: &FirSpec<T>) -> Result<Self::Fir>;
}

/// Factory trait for creating resampler plans.
///
/// Implementations provide a specific resampler plan type and a method
/// to create it from a [`ResampleSpec`].
pub trait CreateResampler<T> {
    /// The resampler plan type produced by this factory.
    type Resampler;

    /// Create a resampler plan from the given specification.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    fn create_resampler(&self, spec: &ResampleSpec<T>) -> Result<Self::Resampler>;
}

/// Factory trait for creating CQT (Constant-Q Transform) plans.
///
/// Implementations provide a specific CQT plan type and a method to
/// create it from a [`CqtSpec`].
pub trait CreateCqt<T> {
    /// The CQT plan type produced by this factory.
    type Cqt;

    /// Create a CQT plan from the given specification.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    fn create_cqt(&self, spec: &CqtSpec<T>) -> Result<Self::Cqt>;
}
