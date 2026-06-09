// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Proc macros for `OmniDSP` backend generation.
//!
//! Provides [`impl_generic_backend!`], which generates `CreatePlan<S>`
//! implementations for all known spec types using generic composition
//! from `omnidsp-core` modules.

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Token, Type, parse_macro_input};

/// Known module names accepted by the `skip` list.
const KNOWN_MODULES: &[&str] = &[
    "conv",
    "fir",
    "resampler",
    "cqt",
    "dft",
    "iir",
    "hilbert",
    "dct",
    "xcorr",
];

/// Return `f32` and `f64` idents for codegen iteration.
fn float_idents() -> [Ident; 2] {
    [
        Ident::new("f32", Span::call_site()),
        Ident::new("f64", Span::call_site()),
    ]
}

// ─── Input parsing ─────────────────────────────────────────────────────

struct BackendInput {
    backend: Type,
    dft: Type,
    dftr2c: Type,
    dftc2r: Type,
    vecops: Type,
    skip: Vec<String>,
}

impl Parse for BackendInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut backend: Option<Type> = None;
        let mut dft: Option<Type> = None;
        let mut dftr2c: Option<Type> = None;
        let mut dftc2r: Option<Type> = None;
        let mut vecops: Option<Type> = None;
        let mut skip: Vec<String> = Vec::new();

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "backend" => {
                    backend = Some(input.parse()?);
                }
                "dft" => {
                    dft = Some(input.parse()?);
                }
                "dftr2c" => {
                    dftr2c = Some(input.parse()?);
                }
                "dftc2r" => {
                    dftc2r = Some(input.parse()?);
                }
                "vecops" => {
                    vecops = Some(input.parse()?);
                }
                "skip" => {
                    let content;
                    syn::bracketed!(content in input);
                    let names = Punctuated::<syn::Ident, Token![,]>::parse_terminated(&content)?;
                    for name in &names {
                        let s = name.to_string();
                        if !KNOWN_MODULES.contains(&s.as_str()) {
                            return Err(syn::Error::new(
                                name.span(),
                                format!(
                                    "unknown module `{s}`, expected one of: {}",
                                    KNOWN_MODULES.join(", "),
                                ),
                            ));
                        }
                    }
                    skip = names.iter().map(ToString::to_string).collect();
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown field `{other}`, expected `backend`, `dft`, `dftr2c`, \
                             `dftc2r`, `vecops`, or `skip`"
                        ),
                    ));
                }
            }

            // Consume optional trailing comma
            let _ = input.parse::<Token![,]>();
        }

        let backend = backend.ok_or_else(|| input.error("missing required field `backend`"))?;
        let dft = dft.ok_or_else(|| input.error("missing required field `dft`"))?;
        let dftr2c = dftr2c.ok_or_else(|| input.error("missing required field `dftr2c`"))?;
        let dftc2r = dftc2r.ok_or_else(|| input.error("missing required field `dftc2r`"))?;
        let vecops = vecops.ok_or_else(|| input.error("missing required field `vecops`"))?;

        Ok(Self {
            backend,
            dft,
            dftr2c,
            dftc2r,
            vecops,
            skip,
        })
    }
}

// ─── Module registry ───────────────────────────────────────────────────

type ModuleGenerator = fn(&BackendInput) -> TokenStream2;

/// Generate all `CreatePlan<S>` impls for a backend, skipping modules in
/// the `skip` list.
fn generate_impls(input: &BackendInput) -> TokenStream2 {
    let mut tokens = TokenStream2::new();

    let modules: &[(&str, ModuleGenerator)] = &[
        ("conv", gen_conv),
        ("fir", gen_fir),
        ("resampler", gen_resampler),
        ("cqt", gen_cqt),
        ("dft", gen_dft),
        ("iir", gen_iir),
        ("hilbert", gen_hilbert),
        ("dct", gen_dct),
        ("xcorr", gen_xcorr),
    ];

    for &(name, gen_fn) in modules {
        if !input.skip.iter().any(|s| s == name) {
            tokens.extend(gen_fn(input));
        }
    }

    tokens
}

// ─── Per-module generators ─────────────────────────────────────────────

fn gen_conv(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let dftr2c = &input.dftr2c;
    let dftc2r = &input.dftc2r;
    let vecops = &input.vecops;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::conv::ConvSpec<#float>>
                for #backend
            {
                // The composite plan type (which embeds the internal Hermitian
                // shaping) is named via the module's own `Conv::Plan` so this
                // impl stays decoupled from the module's internal composition.
                type Plan = <
                    ::omnidsp_core::modules::conv::OmniConv<#dftr2c, #dftc2r, #vecops>
                    as ::omnidsp_core::traits::conv::Conv<#float>
                >::Plan;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::conv::ConvSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::conv::OmniConv::new(
                        self.dftr2c.clone(),
                        self.dftc2r.clone(),
                        self.vecops.clone(),
                    );
                    ::omnidsp_core::traits::conv::Conv::<#float>::create_plan(&factory, spec)
                }
            }
        });
    }

    tokens
}

fn gen_fir(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let dftr2c = &input.dftr2c;
    let dftc2r = &input.dftc2r;
    let vecops = &input.vecops;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::fir::FirSpec<#float>>
                for #backend
            {
                type Plan = <
                    ::omnidsp_core::modules::fir::OmniFir<#dftr2c, #dftc2r, #vecops>
                    as ::omnidsp_core::traits::fir::Fir<#float>
                >::Plan;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::fir::FirSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::fir::OmniFir::new(
                        self.dftr2c.clone(),
                        self.dftc2r.clone(),
                        self.vecops.clone(),
                    );
                    ::omnidsp_core::traits::fir::Fir::<#float>::create_plan(&factory, spec)
                }
            }
        });
    }

    tokens
}

fn gen_resampler(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let vecops = &input.vecops;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<
                ::omnidsp_core::design::resample::ResampleSpec<#float>,
            > for #backend
            {
                type Plan = ::omnidsp_core::modules::resample::OmniResamplePlan<#float, #vecops>;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::design::resample::ResampleSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::resample::OmniResample::new(
                        self.vecops.clone(),
                    );
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

fn gen_cqt(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let dft = &input.dft;
    let vecops = &input.vecops;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::design::cqt::CqtSpec<#float>>
                for #backend
            {
                type Plan = ::omnidsp_core::modules::cqt::OmniCqtPlan<
                    #float,
                    <#dft as ::omnidsp_core::traits::dft::DftC2c<#float>>::Plan,
                    #vecops,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::design::cqt::CqtSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::cqt::OmniCqt::new(
                        self.dft.clone(),
                        self.vecops.clone(),
                    );
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

fn gen_dft(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let dft = &input.dft;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::dft::DftC2cSpec<#float>>
                for #backend
            {
                type Plan = <#dft as ::omnidsp_core::traits::dft::DftC2c<#float>>::Plan;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::dft::DftC2cSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    ::omnidsp_core::traits::dft::DftC2c::<#float>::create_plan(&self.dft, spec)
                }
            }
        });
    }

    tokens
}

fn gen_iir(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::iir::IirSpec<#float>>
                for #backend
            {
                type Plan = ::omnidsp_core::scalar::ScalarIirPlan<#float>;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::iir::IirSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    ::omnidsp_core::traits::iir::Iir::<#float>::create_plan(
                        &::omnidsp_core::scalar::ScalarIir,
                        spec,
                    )
                }
            }
        });
    }

    tokens
}

fn gen_hilbert(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let dft = &input.dft;
    let vecops = &input.vecops;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<
                ::omnidsp_core::modules::hilbert::HilbertSpec<#float>,
            > for #backend
            {
                type Plan = ::omnidsp_core::modules::hilbert::OmniHilbertPlan<
                    #float,
                    <#dft as ::omnidsp_core::traits::dft::DftC2c<#float>>::Plan,
                    #vecops,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::modules::hilbert::HilbertSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::hilbert::OmniHilbert::new(
                        self.dft.clone(),
                        self.vecops.clone(),
                    );
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

fn gen_dct(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let dft = &input.dft;
    let vecops = &input.vecops;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::dct::DctSpec<#float>>
                for #backend
            {
                type Plan = ::omnidsp_core::modules::dct::OmniDctPlan<
                    #float,
                    <#dft as ::omnidsp_core::traits::dft::DftC2c<#float>>::Plan,
                    #vecops,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::dct::DctSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::dct::OmniDct::new(
                        self.dft.clone(),
                        self.vecops.clone(),
                    );
                    ::omnidsp_core::traits::dct::Dct::<#float>::create_plan(&factory, spec)
                }
            }
        });
    }

    tokens
}

fn gen_xcorr(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let dft = &input.dft;
    let vecops = &input.vecops;
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<
                ::omnidsp_core::modules::xcorr::CrossCorrSpec<#float>,
            > for #backend
            {
                type Plan = ::omnidsp_core::modules::xcorr::OmniCrossCorrPlan<
                    #float,
                    <#dft as ::omnidsp_core::traits::dft::DftC2c<#float>>::Plan,
                    #vecops,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::modules::xcorr::CrossCorrSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::xcorr::OmniCrossCorr::new(
                        self.dft.clone(),
                        self.vecops.clone(),
                    );
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

// ─── Proc macro entry point ────────────────────────────────────────────

/// Generate `CreatePlan<S>` implementations for all known spec types
/// using generic composition.
///
/// The backend struct must have `dft`, `dftr2c`, `dftc2r`, and `vecops` fields
/// of the specified types.  The macro delegates to `omnidsp-core` modules
/// internally, producing composition-based plans.
///
/// Use `skip: [...]` to exclude modules that the backend overrides with
/// native implementations.  The backend can then hand-write
/// `CreatePlan<S>` for those specs without conflict.
///
/// # Module names for `skip`
///
/// `conv`, `fir`, `resampler`, `cqt`, `dft`, `iir`, `hilbert`, `dct`, `xcorr`
///
/// # Examples
///
/// ```ignore
/// // All modules (default)
/// omnidsp_macros::impl_generic_backend! {
///     backend: RustBackend,
///     dft: RustDftC2c,
///     dftr2c: RustDftR2c,
///     dftc2r: RustDftC2r,
///     vecops: ScalarVecOps,
/// }
///
/// // Skip modules with native overrides
/// omnidsp_macros::impl_generic_backend! {
///     backend: IppBackend,
///     dft: IppDftC2c,
///     dftr2c: IppDftR2c,
///     dftc2r: IppDftC2r,
///     vecops: IppVecOps,
///     skip: [conv, fir],
/// }
/// ```
#[proc_macro]
pub fn impl_generic_backend(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as BackendInput);
    generate_impls(&input).into()
}
