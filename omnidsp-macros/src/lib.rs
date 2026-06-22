// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Proc macros for `OmniDSP` backend generation.
//!
//! Provides [`impl_generic_backend!`], which generates the per-module
//! `CreatePlan<S>` implementations (via generic composition from `omnidsp-core`
//! modules) plus the `RawDft` accessor that
//! lets core's blanket impls expose the backend's shaped direct r2c/c2r and
//! bare c2c plans.
//!
//! The macro params mirror the **primitive** taxonomy (`dftc2c` required;
//! `dftr2c`/`dftc2r`/`vecops`/`iir` optional with realfft/scalar defaults), and
//! `skip` lists **modules** only.

#![allow(
    clippy::similar_names,
    reason = "DFT codegen uses systematically similar short names (c2c / r2c / c2r)"
)]

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Token, Type, parse_macro_input};

/// Module names accepted by the `skip` list.
///
/// `skip` is **modules-only**: the primitives (`dftc2c`/`dftr2c`/`dftc2r`/
/// `vecops`/`iir`) are configured by their params, never skipped.  A primitive
/// name here is a compile error.
const KNOWN_MODULES: &[&str] = &["conv", "fir", "resampler", "cqt", "hilbert", "dct", "xcorr"];

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
    /// Required: the complex-to-complex DFT base.
    dftc2c: Type,
    /// Optional: native r2c; defaults to the realfft floor `RustDftR2c`.
    dftr2c: Option<Type>,
    /// Optional: native c2r; defaults to the realfft floor `RustDftC2r`.
    dftc2r: Option<Type>,
    /// Optional: native `VecOps`; defaults to `ScalarVecOps`.
    vecops: Option<Type>,
    /// Optional: native IIR; defaults to `ScalarIir`.
    iir: Option<Type>,
    skip: Vec<String>,
}

impl Parse for BackendInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut backend: Option<Type> = None;
        let mut dftc2c: Option<Type> = None;
        let mut dftr2c: Option<Type> = None;
        let mut dftc2r: Option<Type> = None;
        let mut vecops: Option<Type> = None;
        let mut iir: Option<Type> = None;
        let mut skip: Vec<String> = Vec::new();

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "backend" => backend = Some(input.parse()?),
                "dftc2c" => dftc2c = Some(input.parse()?),
                "dftr2c" => dftr2c = Some(input.parse()?),
                "dftc2r" => dftc2r = Some(input.parse()?),
                "vecops" => vecops = Some(input.parse()?),
                "iir" => iir = Some(input.parse()?),
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
                                    "`{s}` is not a skippable module: `skip` lists modules \
                                     only (primitives are configured by the `dftc2c`/`dftr2c`/\
                                     `dftc2r`/`vecops`/`iir` params, not skipped). Expected one \
                                     of: {}",
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
                            "unknown field `{other}`, expected `backend`, `dftc2c`, `dftr2c`, \
                             `dftc2r`, `vecops`, `iir`, or `skip`"
                        ),
                    ));
                }
            }

            // Consume optional trailing comma
            let _ = input.parse::<Token![,]>();
        }

        let backend = backend.ok_or_else(|| input.error("missing required field `backend`"))?;
        let dftc2c = dftc2c.ok_or_else(|| {
            input.error(
                "missing required field `dftc2c` (the complex-to-complex DFT base; a backend \
                 with no native FFT writes `dftc2c: RustDftC2c`)",
            )
        })?;

        Ok(Self {
            backend,
            dftc2c,
            dftr2c,
            dftc2r,
            vecops,
            iir,
            skip,
        })
    }
}

impl BackendInput {
    /// Resolve an optional primitive slot to `(type, value)` token pairs.
    ///
    /// Provided → read `self.<field>.clone()`; omitted → materialize the
    /// `default` floor inline (a unit struct usable as both type and value),
    /// so an omitting backend needs no backing struct field.
    fn optional_handle(
        field: &str,
        ty: Option<&Type>,
        default: &TokenStream2,
    ) -> (TokenStream2, TokenStream2) {
        ty.map_or_else(
            || (default.clone(), default.clone()),
            |t| {
                let ident = Ident::new(field, Span::call_site());
                (quote! { #t }, quote! { self.#ident.clone() })
            },
        )
    }

    /// The required c2c base: type and `self.dftc2c.clone()` value.
    fn c2c_handle(&self) -> (TokenStream2, TokenStream2) {
        let ty = &self.dftc2c;
        (quote! { #ty }, quote! { self.dftc2c.clone() })
    }

    fn r2c_handle(&self) -> (TokenStream2, TokenStream2) {
        Self::optional_handle(
            "dftr2c",
            self.dftr2c.as_ref(),
            &quote! { ::omnidsp_rustfft::RustDftR2c },
        )
    }

    fn c2r_handle(&self) -> (TokenStream2, TokenStream2) {
        Self::optional_handle(
            "dftc2r",
            self.dftc2r.as_ref(),
            &quote! { ::omnidsp_rustfft::RustDftC2r },
        )
    }

    fn vecops_handle(&self) -> (TokenStream2, TokenStream2) {
        Self::optional_handle(
            "vecops",
            self.vecops.as_ref(),
            &quote! { ::omnidsp_core::scalar::ScalarVecOps },
        )
    }

    fn iir_handle(&self) -> (TokenStream2, TokenStream2) {
        Self::optional_handle(
            "iir",
            self.iir.as_ref(),
            &quote! { ::omnidsp_core::scalar::ScalarIir },
        )
    }
}

// ─── Module registry ───────────────────────────────────────────────────

type ModuleGenerator = fn(&BackendInput) -> TokenStream2;

/// Generate the `RawDft` accessor plus every module `CreatePlan<S>` impl,
/// skipping modules in the `skip` list.
///
/// The direct DFT-primitive `CreatePlan<Dft*Spec>` impls are **not** emitted
/// here: core's blanket impls over `RawDft`
/// provide them, wrapping r2c/c2r in the Hermitian shaping.  `iir`
/// is always emitted (it is overridden via the `iir:` param, not via `skip`).
fn generate_impls(input: &BackendInput) -> TokenStream2 {
    let mut tokens = gen_raw_dft_accessor(input);

    let modules: &[(&str, ModuleGenerator)] = &[
        ("conv", gen_conv),
        ("fir", gen_fir),
        ("resampler", gen_resampler),
        ("cqt", gen_cqt),
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

// ─── Direct-primitive dispatch ──────────────────────────────────────────

/// Emit the `RawDft` accessor impls (one per
/// float), exposing the backend's **raw** c2c/r2c/c2r factories.  Core's
/// blanket `CreatePlan<Dft*Spec>` impls wrap these in the Hermitian shaping.
fn gen_raw_dft_accessor(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let (c2c_ty, c2c_val) = input.c2c_handle();
    let (r2c_ty, r2c_val) = input.r2c_handle();
    let (c2r_ty, c2r_val) = input.c2r_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::dispatch::RawDft<#float> for #backend {
                type C2c = #c2c_ty;
                type R2c = #r2c_ty;
                type C2r = #c2r_ty;

                fn raw_dftc2c(&self) -> Self::C2c {
                    #c2c_val
                }
                fn raw_dftr2c(&self) -> Self::R2c {
                    #r2c_val
                }
                fn raw_dftc2r(&self) -> Self::C2r {
                    #c2r_val
                }
            }
        });
    }

    tokens
}

// ─── Per-module generators ─────────────────────────────────────────────

fn gen_conv(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let (r2c_ty, r2c_val) = input.r2c_handle();
    let (c2r_ty, c2r_val) = input.c2r_handle();
    let (vecops_ty, vecops_val) = input.vecops_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::conv::ConvSpec<#float>>
                for #backend
            {
                type Plan = ::omnidsp_core::modules::conv::OmniConvPlan<
                    #float,
                    <#r2c_ty as ::omnidsp_core::traits::dft::DftR2c<#float>>::Plan,
                    ::omnidsp_core::hermitian::HermitianC2rPlan<
                        <#c2r_ty as ::omnidsp_core::traits::dft::DftC2r<#float>>::Plan,
                    >,
                    #vecops_ty,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::conv::ConvSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::conv::OmniConv::new(
                        #r2c_val,
                        #c2r_val,
                        #vecops_val,
                    );
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

fn gen_fir(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let (r2c_ty, r2c_val) = input.r2c_handle();
    let (c2r_ty, c2r_val) = input.c2r_handle();
    let (vecops_ty, vecops_val) = input.vecops_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::fir::FirSpec<#float>>
                for #backend
            {
                type Plan = ::omnidsp_core::modules::fir::OmniFirPlan<
                    #float,
                    <#r2c_ty as ::omnidsp_core::traits::dft::DftR2c<#float>>::Plan,
                    ::omnidsp_core::hermitian::HermitianC2rPlan<
                        <#c2r_ty as ::omnidsp_core::traits::dft::DftC2r<#float>>::Plan,
                    >,
                    #vecops_ty,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::fir::FirSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::fir::OmniFir::new(
                        #r2c_val,
                        #c2r_val,
                        #vecops_val,
                    );
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

fn gen_resampler(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let (vecops_ty, vecops_val) = input.vecops_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<
                ::omnidsp_core::design::resample::ResampleSpec<#float>,
            > for #backend
            {
                type Plan = ::omnidsp_core::modules::resample::OmniResamplePlan<#float, #vecops_ty>;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::design::resample::ResampleSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::resample::OmniResample::new(#vecops_val);
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

fn gen_cqt(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let (r2c_ty, r2c_val) = input.r2c_handle();
    let (vecops_ty, vecops_val) = input.vecops_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        // Multirate CQT capstone: octave-recursive
        // r2c analysis with an `OmniResample(1, 2)` decimator routed as a sub-plan.
        // The backend itself (`self`) is threaded as the `CreatePlan<ResampleSpec>`
        // factory, so a vendor that overrides resampling accelerates the CQT
        // decimation; an unoverriding backend gets `OmniResample` over its own
        // `VecOps`.  Requires `Backend: CreatePlan<ResampleSpec<#float>>` — true
        // unless the `resampler` module is skipped.
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::design::cqt::CqtSpec<#float>>
                for #backend
            {
                type Plan = ::omnidsp_core::modules::cqt::OmniCqtPlan<
                    #float,
                    <#r2c_ty as ::omnidsp_core::traits::dft::DftR2c<#float>>::Plan,
                    #vecops_ty,
                    <#backend as ::omnidsp_core::create::CreatePlan<
                        ::omnidsp_core::design::resample::ResampleSpec<#float>,
                    >>::Plan,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::design::cqt::CqtSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::cqt::OmniCqt::new(
                        #r2c_val,
                        #vecops_val,
                    );
                    // The backend is the routed resample sub-plan factory (option A);
                    // it is borrowed here and dropped — never stored in the plan.
                    factory.create_plan(spec, self)
                }
            }
        });

        // Streaming, newest-anchored CQT: the streaming analogue of
        // the batch impl above, gated together under the same `"cqt"` module key.
        // `OmniCqt::create_stream_plan` mirrors `create_plan`, building one
        // continuous decimator sub-plan per octave transition from the routed
        // `CreatePlan<ResampleSpec>` factory (`self`, option A) and dropping it —
        // the plan retains only the concrete decimators.  Same
        // `Backend: CreatePlan<ResampleSpec<#float>>` precondition as the batch
        // path (the macro emits the resampler impl unless skipped).
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<
                ::omnidsp_core::modules::cqt::CqtStreamSpec<#float>,
            > for #backend
            {
                type Plan = ::omnidsp_core::modules::cqt::OmniCqtStreamPlan<
                    #float,
                    <#r2c_ty as ::omnidsp_core::traits::dft::DftR2c<#float>>::Plan,
                    #vecops_ty,
                    <#backend as ::omnidsp_core::create::CreatePlan<
                        ::omnidsp_core::design::resample::ResampleSpec<#float>,
                    >>::Plan,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::modules::cqt::CqtStreamSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::cqt::OmniCqt::new(
                        #r2c_val,
                        #vecops_val,
                    );
                    // The backend is the routed resample sub-plan factory (option A);
                    // it is borrowed here and dropped — never stored in the plan.
                    factory.create_stream_plan(spec, self)
                }
            }
        });
    }

    tokens
}

fn gen_iir(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let (iir_ty, iir_val) = input.iir_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::iir::IirSpec<#float>>
                for #backend
            {
                type Plan = <#iir_ty as ::omnidsp_core::traits::iir::Iir<#float>>::Plan;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::iir::IirSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = #iir_val;
                    ::omnidsp_core::traits::iir::Iir::<#float>::create_plan(&factory, spec)
                }
            }
        });
    }

    tokens
}

fn gen_hilbert(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let (c2c_ty, c2c_val) = input.c2c_handle();
    let (r2c_ty, r2c_val) = input.r2c_handle();
    let (vecops_ty, vecops_val) = input.vecops_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<
                ::omnidsp_core::modules::hilbert::HilbertSpec<#float>,
            > for #backend
            {
                // Hilbert mixes a real forward (r2c) with a complex inverse
                // (c2c) — its analytic output is complex.
                type Plan = ::omnidsp_core::modules::hilbert::OmniHilbertPlan<
                    #float,
                    <#r2c_ty as ::omnidsp_core::traits::dft::DftR2c<#float>>::Plan,
                    <#c2c_ty as ::omnidsp_core::traits::dft::DftC2c<#float>>::Plan,
                    #vecops_ty,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::modules::hilbert::HilbertSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::hilbert::OmniHilbert::new(
                        #r2c_val,
                        #c2c_val,
                        #vecops_val,
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
    let (r2c_ty, r2c_val) = input.r2c_handle();
    let (c2r_ty, c2r_val) = input.c2r_handle();
    let (vecops_ty, vecops_val) = input.vecops_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::dct::DctSpec<#float>>
                for #backend
            {
                type Plan = ::omnidsp_core::modules::dct::OmniDctPlan<
                    #float,
                    <#r2c_ty as ::omnidsp_core::traits::dft::DftR2c<#float>>::Plan,
                    ::omnidsp_core::hermitian::HermitianC2rPlan<
                        <#c2r_ty as ::omnidsp_core::traits::dft::DftC2r<#float>>::Plan,
                    >,
                    #vecops_ty,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::traits::dct::DctSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::dct::OmniDct::new(
                        #r2c_val,
                        #c2r_val,
                        #vecops_val,
                    );
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

fn gen_xcorr(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;
    let (r2c_ty, r2c_val) = input.r2c_handle();
    let (c2r_ty, c2r_val) = input.c2r_handle();
    let (vecops_ty, vecops_val) = input.vecops_handle();
    let mut tokens = TokenStream2::new();

    for float in float_idents() {
        tokens.extend(quote! {
            impl ::omnidsp_core::create::CreatePlan<
                ::omnidsp_core::modules::xcorr::CrossCorrSpec<#float>,
            > for #backend
            {
                type Plan = ::omnidsp_core::modules::xcorr::OmniCrossCorrPlan<
                    #float,
                    <#r2c_ty as ::omnidsp_core::traits::dft::DftR2c<#float>>::Plan,
                    ::omnidsp_core::hermitian::HermitianC2rPlan<
                        <#c2r_ty as ::omnidsp_core::traits::dft::DftC2r<#float>>::Plan,
                    >,
                    #vecops_ty,
                >;

                fn create_plan(
                    &self,
                    spec: &::omnidsp_core::modules::xcorr::CrossCorrSpec<#float>,
                ) -> ::omnidsp_core::error::Result<Self::Plan> {
                    let factory = ::omnidsp_core::modules::xcorr::OmniCrossCorr::new(
                        #r2c_val,
                        #c2r_val,
                        #vecops_val,
                    );
                    factory.create_plan(spec)
                }
            }
        });
    }

    tokens
}

// ─── Proc macro entry point ────────────────────────────────────────────

/// Generate a backend's `CreatePlan<S>` surface from its primitive slots.
///
/// Params mirror the **primitive** taxonomy; `skip` lists **modules** only:
///
/// - `backend:` — the backend struct (required).
/// - `dftc2c:` — the complex-to-complex DFT base (**required** — the DFT has no
///   pure-math fallback; a backend with no native FFT writes `dftc2c:
///   RustDftC2c`).
/// - `dftr2c:` / `dftc2r:` — native real-DFT kernels (optional; default to the
///   realfft floor `RustDftR2c` / `RustDftC2r`).
/// - `vecops:` — native vector ops (optional; default `ScalarVecOps`).
/// - `iir:` — native IIR (optional; default `ScalarIir`).
/// - `skip: [..]` — modules the backend overrides with a hand-written
///   `CreatePlan<S>` (no conflict).  A **primitive** name here is a compile
///   error.
///
/// For each provided slot the macro reads `self.<field>`; for an omitted slot
/// it materializes the default floor inline, so an omitting backend needs no
/// backing field.  It emits the per-module impls plus a
/// `RawDft` accessor; core's blanket impls
/// then provide the shaped direct r2c/c2r and bare c2c plans — the
/// macro itself emits no `CreatePlan<Dft*Spec>`.
///
/// # Module names for `skip`
///
/// `conv`, `fir`, `resampler`, `cqt`, `hilbert`, `dct`, `xcorr`
///
/// # Examples
///
/// ```ignore
/// // Pure-Rust floor: every slot named explicitly.
/// omnidsp_macros::impl_generic_backend! {
///     backend: RustBackend,
///     dftc2c: RustDftC2c,
///     dftr2c: RustDftR2c,
///     dftc2r: RustDftC2r,
///     vecops: ScalarVecOps,
///     iir:    ScalarIir,
/// }
///
/// // Vendor with a native c2c + VecOps; r2c/c2r/iir take their defaults,
/// // and the conv/fir modules are overridden natively.
/// omnidsp_macros::impl_generic_backend! {
///     backend: IppBackend,
///     dftc2c: IppDftC2c,
///     vecops: IppVecOps,
///     skip:   [conv, fir],
/// }
/// ```
#[proc_macro]
pub fn impl_generic_backend(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as BackendInput);
    generate_impls(&input).into()
}
