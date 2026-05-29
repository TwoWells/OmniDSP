// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Backend macro for generating `Create*` trait implementations.

/// Generate all `Create*` trait implementations for a backend type
/// using generic composition.
///
/// The backend struct must have `dft` and `vecops` fields of the
/// specified types.  The macro delegates to [`Generic`](crate::Generic)
/// internally, producing the same composition-based plans.
///
/// Vendor backends use this macro for the baseline generic impls,
/// then hand-write `Create*` impls for modules where native
/// implementations outperform generic composition.
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
    };

    (@conv $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreateConv<$float> for $backend {
            type Conv =
                <$crate::Generic<$dft, $vecops> as $crate::CreateConv<$float>>::Conv;

            fn create_conv(
                &self,
                spec: &$crate::traits::conv::ConvSpec<$float>,
            ) -> $crate::error::Result<Self::Conv> {
                $crate::CreateConv::<$float>::create_conv(
                    &$crate::Generic(self.dft.clone(), self.vecops.clone()),
                    spec,
                )
            }
        }
    };

    (@fir $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreateFir<$float> for $backend {
            type Fir =
                <$crate::Generic<$dft, $vecops> as $crate::CreateFir<$float>>::Fir;

            fn create_fir(
                &self,
                spec: &$crate::traits::fir::FirSpec<$float>,
            ) -> $crate::error::Result<Self::Fir> {
                $crate::CreateFir::<$float>::create_fir(
                    &$crate::Generic(self.dft.clone(), self.vecops.clone()),
                    spec,
                )
            }
        }
    };

    (@resampler $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreateResampler<$float> for $backend {
            type Resampler =
                <$crate::Generic<$dft, $vecops> as $crate::CreateResampler<$float>>::Resampler;

            fn create_resampler(
                &self,
                spec: &$crate::design::resample::ResampleSpec<$float>,
            ) -> $crate::error::Result<Self::Resampler> {
                $crate::CreateResampler::<$float>::create_resampler(
                    &$crate::Generic(self.dft.clone(), self.vecops.clone()),
                    spec,
                )
            }
        }
    };

    (@cqt $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreateCqt<$float> for $backend {
            type Cqt =
                <$crate::Generic<$dft, $vecops> as $crate::CreateCqt<$float>>::Cqt;

            fn create_cqt(
                &self,
                spec: &$crate::design::cqt::CqtSpec<$float>,
            ) -> $crate::error::Result<Self::Cqt> {
                $crate::CreateCqt::<$float>::create_cqt(
                    &$crate::Generic(self.dft.clone(), self.vecops.clone()),
                    spec,
                )
            }
        }
    };

    (@dft $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreateDft<$float> for $backend {
            type Dft =
                <$crate::Generic<$dft, $vecops> as $crate::CreateDft<$float>>::Dft;

            fn create_dft(
                &self,
                spec: &$crate::traits::dft::DftSpec<$float>,
            ) -> $crate::error::Result<Self::Dft> {
                $crate::CreateDft::<$float>::create_dft(
                    &$crate::Generic(self.dft.clone(), self.vecops.clone()),
                    spec,
                )
            }
        }
    };

    (@iir $backend:ty, $dft:ty, $vecops:ty, $float:ty) => {
        impl $crate::CreateIir<$float> for $backend {
            type Iir =
                <$crate::Generic<$dft, $vecops> as $crate::CreateIir<$float>>::Iir;

            fn create_iir(
                &self,
                spec: &$crate::traits::iir::IirSpec<$float>,
            ) -> $crate::error::Result<Self::Iir> {
                $crate::CreateIir::<$float>::create_iir(
                    &$crate::Generic(self.dft.clone(), self.vecops.clone()),
                    spec,
                )
            }
        }
    };
}
