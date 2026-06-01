// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Backend macro for generating [`CreatePlan<S>`](crate::CreatePlan) implementations.

/// Generate [`CreatePlan<S>`](crate::CreatePlan) implementations for all
/// known spec types using generic composition.
///
/// The backend struct must have `dft` and `vecops` fields of the specified
/// types.  The macro delegates to `omnidsp-core` modules internally,
/// producing composition-based plans.
///
/// Vendor backends use this macro for the baseline generic impls, then
/// hand-write `CreatePlan<S>` impls for specs where native implementations
/// outperform generic composition.
///
/// # Example
///
/// ```ignore
/// omnidsp::impl_generic_backend! {
///     backend: IppBackend,
///     dft: IppDft,
///     vecops: IppVecOps,
/// }
/// ```
#[macro_export]
macro_rules! impl_generic_backend {
    (backend: $backend:ty, dft: $dft:ty, vecops: $vecops:ty $(,)?) => {
        $crate::impl_generic_backend!(@conv $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@conv $backend, $dft, $vecops, f64);
        $crate::impl_generic_backend!(@fir $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@fir $backend, $dft, $vecops, f64);
        $crate::impl_generic_backend!(@resampler $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@resampler $backend, $dft, $vecops, f64);
        $crate::impl_generic_backend!(@cqt $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@cqt $backend, $dft, $vecops, f64);
        $crate::impl_generic_backend!(@dft $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@dft $backend, $dft, $vecops, f64);
        $crate::impl_generic_backend!(@iir $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@iir $backend, $dft, $vecops, f64);
        $crate::impl_generic_backend!(@hilbert $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@hilbert $backend, $dft, $vecops, f64);
        $crate::impl_generic_backend!(@dct $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@dct $backend, $dft, $vecops, f64);
        $crate::impl_generic_backend!(@xcorr $backend, $dft, $vecops, f32);
        $crate::impl_generic_backend!(@xcorr $backend, $dft, $vecops, f64);
    };

    (@conv $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::traits::conv::ConvSpec<$float>> for $backend {
            type Plan = $crate::modules::conv::OmniConvPlan<
                $float,
                <$dft as $crate::traits::dft::Dft<$float>>::Plan,
                $vecops,
            >;

            fn create_plan(
                &self,
                spec: &$crate::traits::conv::ConvSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                let factory = $crate::modules::conv::OmniConv::new(
                    self.dft.clone(),
                    self.vecops.clone(),
                );
                $crate::traits::conv::Conv::<$float>::create_plan(&factory, spec)
            }
        }
    };

    (@fir $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::traits::fir::FirSpec<$float>> for $backend {
            type Plan = $crate::modules::fir::OmniFirPlan<
                $float,
                <$dft as $crate::traits::dft::Dft<$float>>::Plan,
                $vecops,
            >;

            fn create_plan(
                &self,
                spec: &$crate::traits::fir::FirSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                let factory = $crate::modules::fir::OmniFir::new(
                    self.dft.clone(),
                    self.vecops.clone(),
                );
                $crate::traits::fir::Fir::<$float>::create_plan(&factory, spec)
            }
        }
    };

    (@resampler $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::design::resample::ResampleSpec<$float>> for $backend {
            type Plan = $crate::modules::resample::OmniResamplePlan<$float, $vecops>;

            fn create_plan(
                &self,
                spec: &$crate::design::resample::ResampleSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                let factory = $crate::modules::resample::OmniResample::new(
                    self.vecops.clone(),
                );
                factory.create_plan(spec)
            }
        }
    };

    (@cqt $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::design::cqt::CqtSpec<$float>> for $backend {
            type Plan = $crate::modules::cqt::OmniCqtPlan<
                $float,
                <$dft as $crate::traits::dft::Dft<$float>>::Plan,
                $vecops,
            >;

            fn create_plan(
                &self,
                spec: &$crate::design::cqt::CqtSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                let factory = $crate::modules::cqt::OmniCqt::new(
                    self.dft.clone(),
                    self.vecops.clone(),
                );
                factory.create_plan(spec)
            }
        }
    };

    (@dft $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::traits::dft::DftSpec<$float>> for $backend {
            type Plan = <$dft as $crate::traits::dft::Dft<$float>>::Plan;

            fn create_plan(
                &self,
                spec: &$crate::traits::dft::DftSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                $crate::traits::dft::Dft::<$float>::create_plan(&self.dft, spec)
            }
        }
    };

    (@iir $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::traits::iir::IirSpec<$float>> for $backend {
            type Plan = $crate::scalar::ScalarIirPlan<$float>;

            fn create_plan(
                &self,
                spec: &$crate::traits::iir::IirSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                $crate::traits::iir::Iir::<$float>::create_plan(
                    &$crate::scalar::ScalarIir,
                    spec,
                )
            }
        }
    };

    (@hilbert $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::modules::hilbert::HilbertSpec<$float>> for $backend {
            type Plan = $crate::modules::hilbert::OmniHilbertPlan<
                $float,
                <$dft as $crate::traits::dft::Dft<$float>>::Plan,
                $vecops,
            >;

            fn create_plan(
                &self,
                spec: &$crate::modules::hilbert::HilbertSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                let factory = $crate::modules::hilbert::OmniHilbert::new(
                    self.dft.clone(),
                    self.vecops.clone(),
                );
                factory.create_plan(spec)
            }
        }
    };

    (@dct $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::traits::dct::DctSpec<$float>> for $backend {
            type Plan = $crate::modules::dct::OmniDctPlan<
                $float,
                <$dft as $crate::traits::dft::Dft<$float>>::Plan,
                $vecops,
            >;

            fn create_plan(
                &self,
                spec: &$crate::traits::dct::DctSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                let factory = $crate::modules::dct::OmniDct::new(
                    self.dft.clone(),
                    self.vecops.clone(),
                );
                $crate::traits::dct::Dct::<$float>::create_plan(&factory, spec)
            }
        }
    };

    (@xcorr $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreatePlan<$crate::modules::xcorr::CrossCorrSpec<$float>> for $backend {
            type Plan = $crate::modules::xcorr::OmniCrossCorrPlan<
                $float,
                <$dft as $crate::traits::dft::Dft<$float>>::Plan,
                $vecops,
            >;

            fn create_plan(
                &self,
                spec: &$crate::modules::xcorr::CrossCorrSpec<$float>,
            ) -> $crate::error::Result<Self::Plan> {
                let factory = $crate::modules::xcorr::OmniCrossCorr::new(
                    self.dft.clone(),
                    self.vecops.clone(),
                );
                factory.create_plan(spec)
            }
        }
    };
}
