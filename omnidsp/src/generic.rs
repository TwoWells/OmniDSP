// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Generic composition wrapper for [`Dft`] + [`VecOps`] pairs.
//!
//! [`Generic`] implements all [`Create*`](crate::CreateConv) traits using
//! the composed modules from `omnidsp-core` ([`OmniConv`], [`OmniFir`],
//! [`OmniResample`], [`OmniCqt`]).  Any `Dft<T>` + `VecOps<T>` pair gets
//! every module for free through composition.

use num_traits::Float;

use omnidsp_core::design::cqt::CqtSpec;
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::error::Result;
use omnidsp_core::modules::conv::{OmniConv, OmniConvPlan};
use omnidsp_core::modules::cqt::{OmniCqt, OmniCqtPlan};
use omnidsp_core::modules::fir::{OmniFir, OmniFirPlan};
use omnidsp_core::modules::resample::{OmniResample, OmniResamplePlan};
use omnidsp_core::scalar::{ScalarIir, ScalarIirPlan};
use omnidsp_core::traits::conv::{Conv, ConvSpec};
use omnidsp_core::traits::dft::{Dft, DftSpec};
use omnidsp_core::traits::fir::{Fir, FirSpec};
use omnidsp_core::traits::iir::{Iir, IirSpec};
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_core::types::DspFloat;

use crate::create::{CreateConv, CreateCqt, CreateDft, CreateFir, CreateIir, CreateResampler};

/// Generic composition wrapper over a [`Dft`] + [`VecOps`] pair.
///
/// Holds any `Dft` + `VecOps` implementation and implements all `Create*`
/// traits by delegating to the composed modules from `omnidsp-core`.
/// This is the universal fallback path — any backend that provides
/// `Dft<T>` + `VecOps<T>` gets every module for free.
///
/// # Examples
///
/// ```
/// use omnidsp::{Generic, OmniDSP};
/// use omnidsp_core::scalar::ScalarVecOps;
/// use omnidsp_rustfft::RustDft;
///
/// let dsp = OmniDSP::new(Generic(RustDft, ScalarVecOps));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Generic<D, V>(pub D, pub V);

impl<T, D, V> CreateConv<T> for Generic<D, V>
where
    T: Float + Send + Sync,
    D: Dft<T> + Clone,
    V: VecOps<T>,
{
    type Conv = OmniConvPlan<T, D::Plan, V>;

    fn create_conv(&self, spec: &ConvSpec<T>) -> Result<Self::Conv> {
        let factory = OmniConv::new(self.0.clone(), self.1.clone());
        Conv::<T>::create_plan(&factory, spec)
    }
}

impl<T, D, V> CreateFir<T> for Generic<D, V>
where
    T: Float + Send + Sync,
    D: Dft<T> + Clone,
    V: VecOps<T>,
{
    type Fir = OmniFirPlan<T, D::Plan, V>;

    fn create_fir(&self, spec: &FirSpec<T>) -> Result<Self::Fir> {
        let factory = OmniFir::new(self.0.clone(), self.1.clone());
        Fir::<T>::create_plan(&factory, spec)
    }
}

impl<T, D, V> CreateResampler<T> for Generic<D, V>
where
    T: Float + Send + Sync,
    V: VecOps<T>,
{
    type Resampler = OmniResamplePlan<T, V>;

    fn create_resampler(&self, spec: &ResampleSpec<T>) -> Result<Self::Resampler> {
        let factory = OmniResample::new(self.1.clone());
        factory.create_plan(spec)
    }
}

impl<T, D, V> CreateCqt<T> for Generic<D, V>
where
    T: Float + Send + Sync,
    D: Dft<T> + Clone,
    V: VecOps<T>,
{
    type Cqt = OmniCqtPlan<T, D::Plan, V>;

    fn create_cqt(&self, spec: &CqtSpec<T>) -> Result<Self::Cqt> {
        let factory = OmniCqt::new(self.0.clone(), self.1.clone());
        factory.create_plan(spec)
    }
}

impl<T, D, V> CreateDft<T> for Generic<D, V>
where
    D: Dft<T>,
{
    type Dft = D::Plan;

    fn create_dft(&self, spec: &DftSpec<T>) -> Result<Self::Dft> {
        self.0.create_plan(spec)
    }
}

impl<T, D, V> CreateIir<T> for Generic<D, V>
where
    T: DspFloat,
{
    type Iir = ScalarIirPlan<T>;

    fn create_iir(&self, spec: &IirSpec<T>) -> Result<Self::Iir> {
        Iir::<T>::create_plan(&ScalarIir, spec)
    }
}
