// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Specification façade — the single import path for every operation spec,
//! and the home of the library's public-surface conventions.
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
//! These rules are normative for the public API: new modules
//! follow them, and existing drift from them is a bug to fix, not a precedent.
//!
//! ## Specs are valid-by-construction
//!
//! Every `*Spec` has private fields and accessor methods.  `new` is **fallible
//! (`-> Result<Self>`) exactly when this layer has a cross-part invariant to
//! enforce** — e.g. `ResampleSpec::new` checks the prototype cutoff against the
//! polyphase factorization.  It is an **infallible `const fn new`** when the
//! spec is a pure composition of already-validated parts (`FirSpec::new` over a
//! `FirFilter`).  Fallibility tracks
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
//!   not pinned by the description.
//!
//! ## Value construction
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
//!   is not breaking.**
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
//! ## Plan and Processor — the two execution kinds
//!
//! The execution layer has exactly two kinds, split by whether a call depends on
//! the calls before it:
//!
//! - A **Plan** is stateless.  `execute(&self, in, out)` maps one complete input
//!   to one complete output; per-call scratch is not state, so a Plan is
//!   `Send + Sync`, shareable, parallelizable, and changed only by rebuilding.
//!   The DFT family, DCT, convolution, Hilbert, cross-correlation, and the batch
//!   CQT are Plans — reached via `create_plan`.
//! - A **Processor** is stateful.  `process(&mut self, in, out) -> usize` feeds
//!   any amount and returns how many samples it produced; `finish(&mut self,
//!   out) -> usize` signals end-of-stream and flushes the tail; `reset` clears
//!   the state without rebuilding.  A Processor takes `&mut self` and is **not**
//!   `Send + Sync`.  Resampling, FIR, IIR, and the streaming CQT are Processors —
//!   reached via `create_proc`.
//!
//! **Batch vs streaming is how you *drive* a Processor, not a separate mode or
//! spec.**  Batch is `process(everything)` then `finish`; streaming is
//! `process(chunk)` repeated with `finish` at the true end (or never).  The mode
//! lives in the driving, so no spec carries an execution flag.
//!
//! A module exposes **both** a Plan and a Processor only when its batch and
//! stream forms need **different state** — the CQT's stateless, embarrassingly
//! parallel `&self` batch analyzer vs its stateful newest-anchored `&mut self`
//! stream analyzer.  Then it is **two construction verbs over one spec**
//! (`create_plan` / `create_proc` from a single `CqtSpec`); the verb selects the
//! type, so there is no second spec and exactly one `design` entry per module.
//! When the two forms share state (resampling, FIR, IIR) there is a single
//! Processor, driven either way.
//!
//! ## Reconfigure — the live-retuning capability
//!
//! A Processor whose parameter changes only coefficient *values*, not its
//! buffer / state *layout*, can be retuned live without losing stream state.
//! Such Processors additionally implement
//! [`Reconfigure<P>`](crate::traits::reconfigure::Reconfigure): `reconfigure(&p)`
//! re-derives the coefficients from `p` in place, preserving the delay line /
//! per-section state / decimator history, and errors with
//! [`StructuralMismatch`](crate::error::Error::StructuralMismatch) if `p` would
//! change the layout (so the caller rebuilds instead).  The reconfigurable
//! Processors are **FIR** (`Reconfigure<FirFilter>` — keeps the delay line; a
//! changed tap count is structural), **IIR** (`Reconfigure<[BiquadSection]>` —
//! keeps the per-section state; a changed section count is structural), and the
//! **streaming CQT** (`Reconfigure<Window>` — keeps the decimators / rings /
//! hop phase and re-materializes every bin's kernel; the window is orthogonal to
//! the kernel sizes, so it is never structural).  It is **not** on resampling
//! (the knob worth retuning, the rate, is structural to the L/M polyphase
//! factorisation — a live rate change is a different algorithm) and **not** on
//! any Plan (no state, so changing a parameter is just a free rebuild).
//!
//! `Reconfigure` is a *capability*, not a third execution kind: a reconfigurable
//! Processor is the same kind as any Processor, differing by one method.  The
//! hard contract is **behavioural** — after `reconfigure(&p)`, processing uses
//! `p`; doing it glitch-free and zero-alloc (the in-place state-preserving
//! update the omni Processors perform) is a quality-of-implementation property,
//! like throughput, never a guaranteed part of the contract.
//!
//! ## The realization edge — recipe vs. coefficient artifact
//!
//! A spec carries a *description*, and *where* its numbers turn into materialized
//! `T`-typed buffers is the **realization edge** (the create verb).  Two kinds of
//! spec sit on opposite sides of that edge:
//!
//! - A **description** spec carries a **recipe**, never materialized `Vec<T>`
//!   artifacts: a window *kind* plus lengths, per-bin frequencies and sizes, FFT
//!   lengths.  The heavy `T`-typed buffers (a CQT's per-bin frequency-domain
//!   kernels) are built at `create_*`, the window evaluated per bin at its own
//!   length and accumulated in f64, cast to `T` once.  `CqtSpec` is the example —
//!   it holds a [`Window`](crate::window::Window) recipe and per-bin
//!   `{frequency, kernel_len}`, not coefficients, which is exactly what lets the
//!   streaming processor re-materialize its kernels for a new window on a live
//!   reconfigure.
//! - The one exception is the **coefficient-bearing artifact** spec — FIR
//!   (`FirFilter`), IIR (biquad sections), resample (prototype filter).  Here the
//!   coefficients *are* the bring-your-own-able artifact, so the spec carries
//!   them directly; they are **f64-canonical** (the design precision) and cast to
//!   `T` at the realization edge.  That single cast is the rule: design and store
//!   in f64, cast once at `create_*`, never round-trip a `T` artifact back
//!   through f64.
//!
//! ## Module tiers — what composes what, and what a vendor owes
//!
//! Operations sit in tiers by **what they compose, and whether that composition
//! is efficient** — a thing lives at the lowest tier it can be expressed at
//! without losing to a tuned native implementation.  The cheap litmus is **where
//! the primitive call sits relative to the hot loop**: outside it (once per
//! buffer) → the operation composes; inside it (per sample / per tap) → the
//! operation is a *leaf*, because a short call across a vendor's FFI boundary
//! every sample is dominated by the crossing, not the work.
//!
//! - **Vendor primitives** — the real-DFT family and
//!   [`VecOps`](crate::traits::vecops::VecOps): the closed, mandatory
//!   [`Backend<T>`](crate::dispatch::Backend) contract.  A vendor must provide
//!   them; everything else is built over them.
//! - **Omni primitives** — IIR, resampling.  Leaf operations (a per-sample
//!   recurrence; a per-output-sample polyphase dot) that *cannot* compose the
//!   vendor primitives efficiently, so their realization is **concrete scalar and
//!   takes no backend**.  A vendor should override them with a native kernel
//!   (skip the omni impl and hand-write one); the floor stays correct scalar,
//!   never a per-sample FFI penalty.
//! - **Modules** — convolution, FIR, DCT, Hilbert, cross-correlation, window,
//!   CQT.  Realized over the backend's primitives (a bulk transform plus a
//!   whole-buffer vector pass), so a tuned backend accelerates the
//!   frequency-domain path for free.  Convolution and FIR **straddle**: their
//!   scaling form (FFT / overlap-save) composes and accelerates, but for a small
//!   kernel — the `Auto` default at short lengths — they drop to a correct
//!   in-core scalar fast-path, because pushing a tiny filter through an FFT is
//!   wasteful.  That fast-path neither accelerates nor regresses on a vendor
//!   backend; it has no per-sample FFI crossing to penalize.  So the honest
//!   pitch is *frequency-domain modules accelerate for free; time-domain leaves
//!   (IIR, resampling) and small-kernel direct paths are native-or-correct-scalar,
//!   never a regression* — not "every module everywhere."
//! - **Composers** — operations that route a *module* sub-plan.  None today; a
//!   module that merely routes a *leaf* (the multirate CQT routes the resampler)
//!   is still a module.
//!
//! Two of these boundaries are **type-enforced**, not advisory.  A module's
//! dependency on the primitives is its `Self: Backend<T>` bound.  A routing
//! operation's dependency on a sub-processor is a bound the compiler checks: a
//! backend that drops a routed sub-module without supplying a native replacement
//! fails to compile, with a message naming the missing dependency.  The tier
//! *names* and the "a vendor should override this one" guidance are vocabulary;
//! the dependency bounds are contract.
//!
//! ## The trait is the complete contract
//!
//! A Plan or Processor trait is the backend-agnostic, `dyn`-usable surface, and
//! it is the **complete** behavioral contract: whatever the trait exposes, every
//! backend provides; whatever it omits, nobody relies on.  Backends differ in
//! *performance*, never in *contract* (an optional or floor-only method would be
//! a leaky abstraction that breaks on a backend swap).  So the surface exposes:
//! 1. the **buffer-sizing facts a generic caller needs** to allocate its in/out
//!    slices (`num_bins`, `fft_length`, `max_output_columns`, …) — part of the
//!    contract, not optional inherent extras;
//! 2. **descriptive metadata**, kept parallel across a Plan/Processor pair
//!    (`bin_frequencies`, …); and
//! 3. the dominant **reduction convenience** (`process_magnitude` for the CQT),
//!    so a `dyn` consumer is feature-complete.
//!
//! Each trait method **delegates to the concrete type's inherent method**
//! (inherent resolution takes precedence, so the delegation is not recursive),
//! and a convenience **delegates to the steady action, never re-implementing it**.
//!
//! # Examples
//!
//! ```
//! use omnidsp_core::spec::{ConvSpec, DctSpec};
//! use omnidsp_core::traits::conv::ConvMethod;
//! use omnidsp_core::traits::dct::{DctNorm, DctType};
//!
//! let conv = ConvSpec::new(1024, 64, ConvMethod::Auto).unwrap();
//! let dct = DctSpec::new(512, DctType::II, DctNorm::Ortho).unwrap();
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
// AutoCorrSpec / PsdSpec / DwtSpec join as they land.
