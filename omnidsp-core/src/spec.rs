// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Specification façade — the single import path for every operation spec
//! (ADR-006 §5), and the home of the library's public-surface conventions.
//!
//! This module re-exports every `*Spec` under one path so callers can `use
//! omnidsp_core::spec::*` to build any spec directly, without tracking which
//! submodule each lives in.  The definition sites stay put — this is a façade,
//! not a relocation.  The `design::*` helpers (FIR/IIR/resample/CQT) remain the
//! convenient way to compute coefficients; each returns one of these specs, so
//! every spec is also constructible directly via its `new`.
//!
//! # Surface conventions
//!
//! These rules are normative for the public API (surface-lock): new modules
//! follow them, and existing drift from them is a bug to fix, not a precedent.
//!
//! ## Specs are valid-by-construction
//!
//! Every `*Spec` has private fields and accessor methods.  `new` is **fallible
//! (`-> Result<Self>`) exactly when this layer has a cross-part invariant to
//! enforce** — e.g. `ResampleSpec::new` checks the prototype cutoff against the
//! polyphase factorization.  It is an **infallible `const fn new`** when the
//! spec is a pure composition of already-validated parts (`FirSpec::new` over a
//! `FirFilter`, `CqtStreamSpec::new` over a `CqtSpec`).  Fallibility tracks
//! invariants, not nesting; a `new` never returns `Result` solely to wrap a
//! numeric conversion that cannot fail for the supported float types.
//!
//! ## Numeric types
//!
//! - **Frequencies / Hz and fractional-sample quantities are `f64`** — on both
//!   specs and plan traits, regardless of the operation's generic `T`.  The
//!   design math computes in f64 (`design::window` et al.) and deep-bass
//!   resolution needs the precision; carrying `f64` keeps one rule and avoids a
//!   lossy round-trip through `T`.  A `T`-typed consumer downcasts once at its
//!   own edge (e.g. an `f32` FFI boundary), deliberately.
//! - **Counts and lengths are `usize`** (`num_bins`, `fft_length`,
//!   `max_output_*`, …).
//! - **Signal samples and filter coefficients are the generic `T`.**
//! - **Design scalars are `f64`** — Hz, normalized cutoffs, and **window shape
//!   parameters** (Kaiser β, Gaussian σ, Tukey taper) are `f64` on the public
//!   surface regardless of the operation's `T`; `T` is the signal/coefficient
//!   type only.  A value like [`Window`](crate::window::Window) that carries
//!   only design scalars is therefore **non-generic** — the precision is chosen
//!   at evaluation ([`Window::coefficients::<T>`](crate::window::Window::coefficients)),
//!   not pinned by the description (ADR-014).
//!
//! ## Value construction (ADR-013, ADR-014)
//!
//! A configured value is built one of two ways, and the choice is fixed up
//! front, not earned later:
//!
//! - **A small, closed set of kinds is a `#[non_exhaustive]` enum.** A
//!   [`Window`](crate::window::Window) is a plain enum: parameterless kinds are
//!   unit variants ([`Window::Hann`](crate::window::Window::Hann)), parameterized
//!   kinds carry their `f64` design scalar
//!   ([`Window::Kaiser`](crate::window::Window::Kaiser)).  `#[non_exhaustive]`
//!   lets new variants be added non-breakingly, the same way an *enum-argument*
//!   alternative (`fir::design(…, FirMethod)`) can grow a variant.  **Enum-grow
//!   is not breaking** (ADR-014).
//! - **Value vs derivation.** A value type is constructed directly (`Window::Hann`,
//!   `Window::Kaiser(β)`) and may *evaluate or transform itself*
//!   (`coefficients`), but never gains a `_for_`/`_from_` constructor that
//!   derives its parameter from a different requirement.  A derivation that
//!   outputs a heavy `create_plan` artifact (coefficients, kernels) is a
//!   `design::X::design(…)` function; a scalar reparameterization of a simple
//!   value — a Kaiser β solved from an attenuation target
//!   ([`window::kaiser::attenuation`](crate::window::kaiser::attenuation)) — is a
//!   free helper returning the scalar, which the caller wraps in the variant
//!   (`Window::Kaiser(…)`), never a false `design::` peer.  The test: does it
//!   output a `create_plan` input, or merely respell a value's parameter?
//! - **Alternatives as data.** Several parameterizations or algorithms producing
//!   the same output are expressed as **data, not a fan of methods** — an enum
//!   argument or variant (growable under `#[non_exhaustive]`), not a fan of
//!   methods.  Never `x_for_a` / `x_from_b`.
//!
//! ## Plan traits
//!
//! A plan trait is the backend-agnostic, `dyn`-usable surface.  It exposes:
//! 1. the **buffer-sizing facts a generic `process` caller needs** to allocate
//!    its in/out slices (`num_bins`, `fft_length`, `max_output_columns`, …) —
//!    these are part of the `process` contract, not optional inherent extras;
//! 2. **descriptive metadata**, kept parallel across a batch/streaming pair
//!    (`bin_frequencies`, …); and
//! 3. the dominant **reduction convenience** (`process_magnitude` for the CQT),
//!    so a `dyn` consumer is feature-complete.
//!
//! Each trait method is implemented by **delegating to the concrete plan's
//! inherent method** (inherent resolution takes precedence, so the delegation
//! is not recursive); a convenience **delegates to `process`, it never
//! re-implements it**.  Stateless `&self` transform plans are `Send + Sync`;
//! stateful streaming plans take `&mut self`, carry a `reset`, a `max_output_*`
//! sizing method and a count-returning `process`, and are **not** `Send + Sync`.
//!
//! ## Execution-mode variants (batch vs streaming)
//!
//! When two modes produce **differently-shaped plans** — the CQT batch `&self`
//! analyzer vs the streaming `&mut self` newest-anchored analyzer — they get
//! **distinct spec types** so dispatch stays compile-time via `CreatePlan<S>`
//! (ADR-006: the spec *type* selects the route; no runtime match).  The
//! streaming spec is a **newtype wrapping the batch spec** (`CqtStreamSpec` over
//! `CqtSpec`), reached by **spec conversion** (`CqtSpec::into_streaming`), never
//! a second designer — there is exactly one `design` entry per module, flavored
//! by an enum argument.  A mode that leaves the plan **shape unchanged** (same
//! plan, only a different output-count accounting) may instead remain a field
//! on one spec (`ResampleSpec`'s `ResampleMode`); that is the deliberate
//! exception, not a competing pattern.
//!
//! # Examples
//!
//! ```
//! use omnidsp_core::spec::{ConvSpec, DctSpec};
//! use omnidsp_core::traits::conv::ConvMethod;
//! use omnidsp_core::traits::dct::{DctNorm, DctType};
//!
//! let conv = ConvSpec::<f64>::new(1024, 64, ConvMethod::Auto).unwrap();
//! let dct = DctSpec::<f64>::new(512, DctType::II, DctNorm::Ortho).unwrap();
//! assert_eq!(conv.a_len(), 1024);
//! assert_eq!(dct.length(), 512);
//! ```

pub use crate::design::cqt::CqtSpec;
pub use crate::design::resample::ResampleSpec;
pub use crate::modules::hilbert::HilbertSpec;
pub use crate::modules::xcorr::{CrossCorrNorm, CrossCorrSpec};
pub use crate::traits::conv::ConvSpec;
pub use crate::traits::dct::DctSpec;
pub use crate::traits::dft::{DftC2cSpec, DftC2rSpec, DftR2cSpec};
pub use crate::traits::fir::FirSpec;
pub use crate::traits::iir::IirSpec;
// AutoCorrSpec / PsdSpec / DwtSpec join as they land (tickets 18/19/20).
