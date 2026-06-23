// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Proc macros for `OmniDSP` backend generation.
//!
//! Provides [`impl_generic_backend!`], which generates the per-module
//! `CreatePlan<S>` / `CreateProc<S>` implementations (via generic composition
//! from `omnidsp-core` modules) plus the `RawDft` accessor and the shaped direct
//! DFT-primitive impls.
//!
//! Both dispatch traits are generic-associated-type factories: the spec is
//! non-generic and the precision `T` arrives at the `create_plan::<T>` /
//! `create_proc::<T>` method, gated by `Self: Backend<T>` — the closed
//! foundational contract (the real-DFT family `C2c`/`R2c`/`C2r` plus `VecOps`).
//! The **backend** is the `VecOps` provider — there is no `vecops:` slot; the
//! backend implements `VecOps<f32>`/`VecOps<f64>` directly (the floor's are empty
//! scalar-default impls) and the macro threads `Self` as each module's `V`,
//! constructing the module with `self.clone()` as its vector-ops handle.  A
//! vendor hand-writes accelerated `VecOps` impls and gets every module
//! accelerated transitively.
//!
//! The macro params mirror the held **DFT sub-factory** primitives (`dftc2c`
//! required; `dftr2c`/`dftc2r` optional with realfft defaults), and `skip` lists
//! **modules** only (`iir` among them — its plan is the concrete
//! `ScalarIirProcessor`, overridable via `skip: [iir]`).

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
/// `skip` is **modules-only**: the held DFT primitives (`dftc2c`/`dftr2c`/
/// `dftc2r`) are configured by their params, never skipped.  A primitive name
/// here is a compile error.  `iir` is a module (its plan is the concrete
/// `ScalarIirProcessor`), so a vendor with native IIR overrides via `skip: [iir]`
/// plus a hand-written `CreateProc<IirSpec>`.
const KNOWN_MODULES: &[&str] = &[
    "conv",
    "fir",
    "resampler",
    "cqt",
    "iir",
    "hilbert",
    "dct",
    "xcorr",
];

/// The composition dependency graph: which **composer** module routes which
/// other module as an internal sub-processor.
///
/// This is the single declarative source of truth for the routing tier of the
/// module model.  Each `(composer, sub_module)` pair means "the generated
/// `composer` impl routes a `sub_module` sub-processor, so it is bounded on the
/// backend being able to supply it".  The composer's emitted bound uses the
/// `RoutesSubProc` marker (not bare `CreateProc`) so a backend that keeps the
/// composer but drops the sub-module gets a worded diagnostic naming the
/// dependency, rather than an opaque trait-bound error inside the composer.
///
/// The graph grows here as composers are added; today the only routing module
/// is the multirate CQT, which decimates an octave at a time through a
/// half-band resampler.
const COMPOSITION_DEPS: &[(&str, &str)] = &[("cqt", "resampler")];

/// Whether `composer` routes `sub_module` per [`COMPOSITION_DEPS`].
fn composer_routes(composer: &str, sub_module: &str) -> bool {
    COMPOSITION_DEPS
        .iter()
        .any(|&(c, s)| c == composer && s == sub_module)
}

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
    skip: Vec<String>,
}

impl Parse for BackendInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut backend: Option<Type> = None;
        let mut dftc2c: Option<Type> = None;
        let mut dftr2c: Option<Type> = None;
        let mut dftc2r: Option<Type> = None;
        let mut skip: Vec<String> = Vec::new();

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "backend" => backend = Some(input.parse()?),
                "dftc2c" => dftc2c = Some(input.parse()?),
                "dftr2c" => dftr2c = Some(input.parse()?),
                "dftc2r" => dftc2r = Some(input.parse()?),
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
                                     only (the DFT primitives are configured by the `dftc2c`/\
                                     `dftr2c`/`dftc2r` params, not skipped). Expected one \
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
                             `dftc2r`, or `skip`"
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
            skip,
        })
    }
}

impl BackendInput {
    /// Resolve an optional held primitive slot to `(type, value)` token pairs.
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
}

// ─── Module registry ───────────────────────────────────────────────────

type ModuleGenerator = fn(&BackendInput) -> TokenStream2;

/// Generate the `RawDft` accessor, the shaped direct DFT-primitive impls, plus
/// every module `CreatePlan` / `CreateProc` impl, skipping modules in the `skip`
/// list.  `iir` is a module like any other (its plan is the concrete
/// `ScalarIirProcessor`); a vendor with native IIR overrides it via `skip: [iir]`.
fn generate_impls(input: &BackendInput) -> TokenStream2 {
    let mut tokens = gen_raw_dft_accessor(input);
    tokens.extend(gen_dft_primitives(input));

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

/// Emit the `RawDft` accessor impls (one per float), exposing the backend's
/// **raw** c2c/r2c/c2r factories.  The shaped `CreatePlan<Dft*Spec>` impls
/// ([`gen_dft_primitives`]) wrap these through the `crate::hermitian` decorators.
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

/// Emit the per-backend `CreatePlan<Dft*Spec>` impls (GAT form):
///
/// - c2c — bare (no Hermitian convention to enforce);
/// - r2c — Hermitian-cleaned output (bit-exactly-real DC/Nyquist);
/// - c2r — Hermitian-projected input (the round-trip drift guard).
///
/// These are per-backend rather than a core blanket because the GAT `Plan<T>`
/// names a type whose body needs `Self: RawDft<T>`, which a blanket bounded only
/// by the trait's `Self: VecOps<T>` cannot guarantee; with `Self` concrete here
/// the `RawDft<f32>`/`RawDft<f64>` capabilities (emitted above) are both in scope.
fn gen_dft_primitives(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    // One impl each (not per-float): the GAT `Plan<T>` keys the whole body on the
    // method's precision `T`, since every sub-factory implements `RawDft<T>` /
    // `DftC2c<T>` for both widths.  The `Self: RawDft<T>` bound the body needs is
    // supplied per concrete backend by the `RawDft` accessor impls above.
    quote! {
        impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::dft::DftC2cSpec>
            for #backend
        {
            type Plan<T> = <<Self as ::omnidsp_core::dispatch::RawDft<T>>::C2c
                as ::omnidsp_core::traits::dft::DftC2c<T>>::Plan
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_plan<T>(
                &self,
                spec: &::omnidsp_core::traits::dft::DftC2cSpec,
            ) -> ::omnidsp_core::error::Result<Self::Plan<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                ::omnidsp_core::traits::dft::DftC2c::<T>::create_plan(
                    &::omnidsp_core::dispatch::RawDft::<T>::raw_dftc2c(self),
                    spec,
                )
            }
        }

        impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::dft::DftR2cSpec>
            for #backend
        {
            type Plan<T> = ::omnidsp_core::hermitian::HermitianR2cPlan<
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::R2c
                    as ::omnidsp_core::traits::dft::DftR2c<T>>::Plan,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_plan<T>(
                &self,
                spec: &::omnidsp_core::traits::dft::DftR2cSpec,
            ) -> ::omnidsp_core::error::Result<Self::Plan<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                ::omnidsp_core::traits::dft::DftR2c::<T>::create_plan(
                    &::omnidsp_core::hermitian::HermitianR2c::new(
                        ::omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
                    ),
                    spec,
                )
            }
        }

        impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::dft::DftC2rSpec>
            for #backend
        {
            type Plan<T> = ::omnidsp_core::hermitian::HermitianC2rPlan<
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::C2r
                    as ::omnidsp_core::traits::dft::DftC2r<T>>::Plan,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_plan<T>(
                &self,
                spec: &::omnidsp_core::traits::dft::DftC2rSpec,
            ) -> ::omnidsp_core::error::Result<Self::Plan<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                ::omnidsp_core::traits::dft::DftC2r::<T>::create_plan(
                    &::omnidsp_core::hermitian::HermitianC2r::new(
                        ::omnidsp_core::dispatch::RawDft::<T>::raw_dftc2r(self),
                    ),
                    spec,
                )
            }
        }
    }
}

// ─── Per-module generators ─────────────────────────────────────────────

fn gen_conv(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    // One impl (not per-float): the GAT `Plan<T>` and the method body are
    // generic over the precision `T`, routing the forward r2c and inverse c2r
    // sub-plans through the `RawDft<T>` accessor (`<Self as RawDft<T>>::R2c` /
    // `::C2r`) — the `Self: Backend<T>` bound supplies `RawDft<T>` and `VecOps<T>`
    // together, so the sub-factory capabilities are entailed per concrete backend.
    quote! {
        impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::conv::ConvSpec>
            for #backend
        {
            type Plan<T> = ::omnidsp_core::modules::conv::OmniConvPlan<
                T,
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::R2c
                    as ::omnidsp_core::traits::dft::DftR2c<T>>::Plan,
                ::omnidsp_core::hermitian::HermitianC2rPlan<
                    <<Self as ::omnidsp_core::dispatch::RawDft<T>>::C2r
                        as ::omnidsp_core::traits::dft::DftC2r<T>>::Plan,
                >,
                Self,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_plan<T>(
                &self,
                spec: &::omnidsp_core::traits::conv::ConvSpec,
            ) -> ::omnidsp_core::error::Result<Self::Plan<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                let factory = ::omnidsp_core::modules::conv::OmniConv::new(
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftc2r(self),
                    ::core::clone::Clone::clone(self),
                );
                factory.create_plan::<T>(spec)
            }
        }
    }
}

fn gen_fir(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    quote! {
        impl ::omnidsp_core::create::CreateProc<::omnidsp_core::traits::fir::FirSpec>
            for #backend
        {
            type Proc<T> = ::omnidsp_core::modules::fir::OmniFirProcessor<
                T,
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::R2c
                    as ::omnidsp_core::traits::dft::DftR2c<T>>::Plan,
                ::omnidsp_core::hermitian::HermitianC2rPlan<
                    <<Self as ::omnidsp_core::dispatch::RawDft<T>>::C2r
                        as ::omnidsp_core::traits::dft::DftC2r<T>>::Plan,
                >,
                Self,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_proc<T>(
                &self,
                spec: &::omnidsp_core::traits::fir::FirSpec,
            ) -> ::omnidsp_core::error::Result<Self::Proc<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                let factory = ::omnidsp_core::modules::fir::OmniFir::new(
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftc2r(self),
                    ::core::clone::Clone::clone(self),
                );
                factory.create_proc::<T>(spec)
            }
        }
    }
}

fn gen_resampler(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    quote! {
        impl ::omnidsp_core::create::CreateProc<
            ::omnidsp_core::design::resample::ResampleSpec,
        > for #backend
        {
            type Proc<T> =
                ::omnidsp_core::modules::resample::OmniResampleProcessor<T>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_proc<T>(
                &self,
                spec: &::omnidsp_core::design::resample::ResampleSpec,
            ) -> ::omnidsp_core::error::Result<Self::Proc<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                // The polyphase resampler is a concrete scalar module (its hot
                // path is a per-output-sample dot), so it carries no `VecOps`
                // handle — the backend is not threaded in here.
                let factory = ::omnidsp_core::modules::resample::OmniResample::new();
                factory.create_proc::<T>(spec)
            }
        }
    }
}

fn gen_cqt(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    // Multirate CQT capstone: octave-recursive r2c analysis with an
    // `OmniResample(1, 2)` decimator routed as a sub-processor.  The backend
    // itself (`self`) is threaded as the module `VecOps` `V` (for the CQT's bulk
    // spectral multiply) and as the routed resample factory, so a vendor that
    // overrides resampling accelerates the CQT decimation; an unoverriding
    // backend gets the scalar `OmniResample` decimator (its per-sample polyphase
    // dot stays scalar by design — never a per-sample FFI crossing).  The routed
    // factory must be a `Backend<T>` (only a backend dispatches sub-processors);
    // that holds for `Self` and additionally requires the backend to supply a
    // resample sub-processor.
    //
    // That routing dependency — cqt routes resampler — is declared once in the
    // `COMPOSITION_DEPS` graph (asserted below).  The emitted bound names the
    // `RoutesSubProc<ResampleSpec>` marker rather than bare
    // `CreateProc<ResampleSpec>`: both are equivalent (the marker is a blanket
    // over `CreateProc`), but the marker carries the `on_unimplemented`
    // diagnostic, so a backend that keeps `cqt` while dropping `resampler` (and
    // not hand-writing a native one) gets a worded error naming the cqt→resample
    // dependency instead of an opaque trait-bound failure inside the CQT
    // machinery.  The associated processor type is still reached through the
    // `CreateProc` supertrait projection.  The forward r2c plan is reached
    // through the `RawDft<T>` accessor.  One impl each, generic over `T`.
    assert!(
        composer_routes("cqt", "resampler"),
        "COMPOSITION_DEPS must declare that `cqt` routes `resampler`: the CQT \
         dispatch routes a resample sub-processor, so the graph is the single \
         source of truth for that dependency"
    );

    quote! {
        impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::design::cqt::CqtSpec>
            for #backend
        {
            type Plan<T> = ::omnidsp_core::modules::cqt::OmniCqtPlan<
                T,
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::R2c
                    as ::omnidsp_core::traits::dft::DftR2c<T>>::Plan,
                Self,
                <Self as ::omnidsp_core::create::CreateProc<
                    ::omnidsp_core::design::resample::ResampleSpec,
                >>::Proc<T>,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>
                    + ::omnidsp_core::create::RoutesSubProc<
                        ::omnidsp_core::design::resample::ResampleSpec,
                    >;

            fn create_plan<T>(
                &self,
                spec: &::omnidsp_core::design::cqt::CqtSpec,
            ) -> ::omnidsp_core::error::Result<Self::Plan<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>
                    + ::omnidsp_core::create::RoutesSubProc<
                        ::omnidsp_core::design::resample::ResampleSpec,
                    >,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
                <Self as ::omnidsp_core::create::CreateProc<
                    ::omnidsp_core::design::resample::ResampleSpec,
                >>::Proc<T>: ::omnidsp_core::modules::resample::ResampleProcessor<T>,
            {
                let factory = ::omnidsp_core::modules::cqt::OmniCqt::new(
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
                    ::core::clone::Clone::clone(self),
                );
                // The backend is the routed resample sub-processor factory; it is
                // borrowed here and dropped — never stored in the plan.
                factory.create_plan::<T, Self>(spec, self)
            }
        }

        // Streaming, newest-anchored CQT: reached via `CreateProc<CqtSpec>` over
        // the same spec.  Mirrors the batch impl, building one continuous
        // decimator sub-processor per octave transition from the routed resample
        // factory (`self`) and dropping it.
        impl ::omnidsp_core::create::CreateProc<::omnidsp_core::design::cqt::CqtSpec>
            for #backend
        {
            type Proc<T> = ::omnidsp_core::modules::cqt::OmniCqtProcessor<
                T,
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::R2c
                    as ::omnidsp_core::traits::dft::DftR2c<T>>::Plan,
                Self,
                <Self as ::omnidsp_core::create::CreateProc<
                    ::omnidsp_core::design::resample::ResampleSpec,
                >>::Proc<T>,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>
                    + ::omnidsp_core::create::RoutesSubProc<
                        ::omnidsp_core::design::resample::ResampleSpec,
                    >;

            fn create_proc<T>(
                &self,
                spec: &::omnidsp_core::design::cqt::CqtSpec,
            ) -> ::omnidsp_core::error::Result<Self::Proc<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>
                    + ::omnidsp_core::create::RoutesSubProc<
                        ::omnidsp_core::design::resample::ResampleSpec,
                    >,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
                <Self as ::omnidsp_core::create::CreateProc<
                    ::omnidsp_core::design::resample::ResampleSpec,
                >>::Proc<T>: ::omnidsp_core::modules::resample::ResampleProcessor<T>,
            {
                let factory = ::omnidsp_core::modules::cqt::OmniCqt::new(
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
                    ::core::clone::Clone::clone(self),
                );
                factory.create_proc::<T, Self>(spec, self)
            }
        }
    }
}

fn gen_iir(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    // IIR is a **module**, not part of the foundational backend contract: its
    // recurrence is a sequential scalar loop that no DFT/VecOps primitive
    // accelerates, so the floor plan is always the concrete `ScalarIirProcessor`
    // built from the spec's f64 sections cast to `T` (via `ScalarIir`'s
    // `create_proc`).  The `Self: Backend<T>` bound is satisfied-but-unused here.
    // A vendor with native IIR overrides via `skip: [iir]` plus a hand-written
    // `CreateProc<IirSpec>`.
    quote! {
        impl ::omnidsp_core::create::CreateProc<::omnidsp_core::traits::iir::IirSpec>
            for #backend
        {
            type Proc<T> = ::omnidsp_core::scalar::ScalarIirProcessor<T>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_proc<T>(
                &self,
                spec: &::omnidsp_core::traits::iir::IirSpec,
            ) -> ::omnidsp_core::error::Result<Self::Proc<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                ::omnidsp_core::traits::iir::Iir::<T>::create_proc(
                    &::omnidsp_core::scalar::ScalarIir,
                    spec,
                )
            }
        }
    }
}

fn gen_hilbert(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    quote! {
        impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::modules::hilbert::HilbertSpec>
            for #backend
        {
            // Hilbert mixes a real forward (r2c) with a complex inverse
            // (c2c) — its analytic output is complex.
            type Plan<T> = ::omnidsp_core::modules::hilbert::OmniHilbertPlan<
                T,
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::R2c
                    as ::omnidsp_core::traits::dft::DftR2c<T>>::Plan,
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::C2c
                    as ::omnidsp_core::traits::dft::DftC2c<T>>::Plan,
                Self,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_plan<T>(
                &self,
                spec: &::omnidsp_core::modules::hilbert::HilbertSpec,
            ) -> ::omnidsp_core::error::Result<Self::Plan<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                let factory = ::omnidsp_core::modules::hilbert::OmniHilbert::new(
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftc2c(self),
                    ::core::clone::Clone::clone(self),
                );
                factory.create_plan::<T>(spec)
            }
        }
    }
}

fn gen_dct(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    quote! {
        impl ::omnidsp_core::create::CreatePlan<::omnidsp_core::traits::dct::DctSpec>
            for #backend
        {
            type Plan<T> = ::omnidsp_core::modules::dct::OmniDctPlan<
                T,
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::R2c
                    as ::omnidsp_core::traits::dft::DftR2c<T>>::Plan,
                ::omnidsp_core::hermitian::HermitianC2rPlan<
                    <<Self as ::omnidsp_core::dispatch::RawDft<T>>::C2r
                        as ::omnidsp_core::traits::dft::DftC2r<T>>::Plan,
                >,
                Self,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_plan<T>(
                &self,
                spec: &::omnidsp_core::traits::dct::DctSpec,
            ) -> ::omnidsp_core::error::Result<Self::Plan<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                let factory = ::omnidsp_core::modules::dct::OmniDct::new(
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftc2r(self),
                    ::core::clone::Clone::clone(self),
                );
                factory.create_plan::<T>(spec)
            }
        }
    }
}

fn gen_xcorr(input: &BackendInput) -> TokenStream2 {
    let backend = &input.backend;

    quote! {
        impl ::omnidsp_core::create::CreatePlan<
            ::omnidsp_core::modules::xcorr::CrossCorrSpec,
        > for #backend
        {
            type Plan<T> = ::omnidsp_core::modules::xcorr::OmniCrossCorrPlan<
                T,
                <<Self as ::omnidsp_core::dispatch::RawDft<T>>::R2c
                    as ::omnidsp_core::traits::dft::DftR2c<T>>::Plan,
                ::omnidsp_core::hermitian::HermitianC2rPlan<
                    <<Self as ::omnidsp_core::dispatch::RawDft<T>>::C2r
                        as ::omnidsp_core::traits::dft::DftC2r<T>>::Plan,
                >,
                Self,
            >
            where
                Self: ::omnidsp_core::dispatch::Backend<T>;

            fn create_plan<T>(
                &self,
                spec: &::omnidsp_core::modules::xcorr::CrossCorrSpec,
            ) -> ::omnidsp_core::error::Result<Self::Plan<T>>
            where
                Self: ::omnidsp_core::dispatch::Backend<T>,
                T: ::omnidsp_core::types::DspFloat
                    + ::core::ops::AddAssign
                    + ::core::ops::MulAssign,
            {
                let factory = ::omnidsp_core::modules::xcorr::OmniCrossCorr::new(
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftr2c(self),
                    ::omnidsp_core::dispatch::RawDft::<T>::raw_dftc2r(self),
                    ::core::clone::Clone::clone(self),
                );
                factory.create_plan::<T>(spec)
            }
        }
    }
}

// ─── Proc macro entry point ────────────────────────────────────────────

/// Generate a backend's `CreatePlan` / `CreateProc` surface from its DFT
/// primitive slots.
///
/// Params mirror the held **DFT sub-factory** primitives; `skip` lists
/// **modules** only:
///
/// - `backend:` — the backend struct (required).  It must implement
///   `VecOps<f32>` and `VecOps<f64>` (the floor's are empty scalar-default
///   impls); the macro threads it as each module's vector-ops handle.
/// - `dftc2c:` — the complex-to-complex DFT base (**required** — the DFT has no
///   pure-math fallback; a backend with no native FFT writes `dftc2c:
///   RustDftC2c`).
/// - `dftr2c:` / `dftc2r:` — native real-DFT kernels (optional; default to the
///   realfft floor `RustDftR2c` / `RustDftC2r`).
/// - `skip: [..]` — modules the backend overrides with a hand-written
///   `CreatePlan` / `CreateProc` (no conflict).  A **DFT primitive** name here is
///   a compile error.  IIR is a module: a vendor with native IIR writes
///   `skip: [iir]` plus its own `CreateProc<IirSpec>` (the default plan is the
///   concrete `ScalarIirProcessor`).
///
/// For each provided slot the macro reads `self.<field>`; for an omitted slot
/// it materializes the default floor inline.  It emits the per-module impls plus
/// a `RawDft` accessor and the shaped direct DFT-primitive impls.
///
/// # Module names for `skip`
///
/// `conv`, `fir`, `resampler`, `cqt`, `iir`, `hilbert`, `dct`, `xcorr`
///
/// # Examples
///
/// ```ignore
/// // Pure-Rust floor: every held DFT slot named explicitly; the backend supplies
/// // `VecOps` directly (empty scalar-default impls written next to it).  IIR is
/// // a module — its floor plan (`ScalarIirProcessor`) is emitted without a slot.
/// omnidsp_macros::impl_generic_backend! {
///     backend: RustBackend,
///     dftc2c: RustDftC2c,
///     dftr2c: RustDftR2c,
///     dftc2r: RustDftC2r,
/// }
///
/// // Vendor with a native c2c; r2c/c2r take their defaults, the conv/fir/iir
/// // modules are overridden natively, and the backend hand-writes accelerated
/// // `VecOps` impls.
/// omnidsp_macros::impl_generic_backend! {
///     backend: IppBackend,
///     dftc2c: IppDftC2c,
///     skip:   [conv, fir, iir],
/// }
/// ```
#[proc_macro]
pub fn impl_generic_backend(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as BackendInput);
    generate_impls(&input).into()
}
