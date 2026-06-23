// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `OmniDSP` struct and `RustBackend` — top-level DSP engine with pluggable backends.

use omnidsp_core::design::cqt::CqtSpec;
use omnidsp_core::design::resample::ResampleSpec;
use omnidsp_core::error::Result;
use omnidsp_core::modules::hilbert::HilbertSpec;
use omnidsp_core::modules::xcorr::CrossCorrSpec;
use omnidsp_core::traits::conv::ConvSpec;
use omnidsp_core::traits::dct::DctSpec;
use omnidsp_core::traits::dft::{DftC2cSpec, DftC2rSpec, DftR2cSpec};
use omnidsp_core::traits::fir::FirSpec;
use omnidsp_core::traits::iir::IirSpec;
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_rustfft::{RustDftC2c, RustDftC2r, RustDftR2c};

use omnidsp_core::create::{CreatePlan, CreateProc};

// ─── RustBackend ───────────────────────────────────────────────────────

/// Pure Rust fallback backend.
///
/// Combines [`RustDftC2c`] (wrapping `RustFFT`) and [`RustDftR2c`] /
/// [`RustDftC2r`] (wrapping `realfft`) — the real-DFT family that, with the
/// vector-ops primitive, forms the foundational [`Backend`](omnidsp_core::dispatch::Backend)
/// contract.  It is also the floor [`VecOps`] provider: the empty
/// `impl VecOps<f32>`/`<f64>` below inherit the scalar (LLVM auto-vectorized)
/// defaults from the trait — the backend *is* the vector-ops layer the omni
/// modules route their glue through.  IIR is a module over this floor (its plan
/// is the scalar DF2T biquad cascade `ScalarIirProcessor`), not a held slot.
/// Builds on every platform with no external dependencies.
#[derive(Debug, Clone, Copy)]
pub struct RustBackend {
    /// Complex-to-complex DFT factory (`RustFFT` wrapper).
    pub(crate) dftc2c: RustDftC2c,
    /// Real-to-complex DFT factory (`realfft` wrapper).
    pub(crate) dftr2c: RustDftR2c,
    /// Complex-to-real DFT factory (`realfft` wrapper).
    pub(crate) dftc2r: RustDftC2r,
}

impl RustBackend {
    /// Create a new Rust fallback backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dftc2c: RustDftC2c,
            dftr2c: RustDftR2c,
            dftc2r: RustDftC2r,
        }
    }
}

impl Default for RustBackend {
    fn default() -> Self {
        Self::new()
    }
}

// The floor backend is its own `VecOps` provider.  The trait's methods are all
// scalar (LLVM auto-vectorized) defaults, so these empty impls *are* the scalar
// vector-ops layer; a vendor backend hand-writes accelerated impls instead.
impl VecOps<f32> for RustBackend {}
impl VecOps<f64> for RustBackend {}

omnidsp_macros::impl_generic_backend! {
    backend: RustBackend,
    dftc2c: RustDftC2c,
    dftr2c: RustDftR2c,
    dftc2r: RustDftC2r,
}

// ─── OmniDSP struct ─────────────────────────────────────────────────

/// Top-level DSP engine with a pluggable backend.
///
/// `OmniDSP<B>` wraps a backend `B` and provides universal
/// [`create_plan`](Self::create_plan) / [`create_proc`](Self::create_proc)
/// methods plus convenience wrappers for each module.  The spec type drives
/// compile-time dispatch; the precision `T` is chosen at the call.
///
/// # Examples
///
/// ```
/// use omnidsp::OmniDSP;
/// use omnidsp::traits::fir::{FirFilter, FirMeta, FirSpec, FirStrategy};
///
/// let dsp = OmniDSP::rust();
/// let filter = FirFilter::new(vec![1.0 / 3.0; 3], FirMeta::unknown()).unwrap();
/// let spec = FirSpec::new(filter, FirStrategy::Direct);
/// let proc = dsp.create_proc::<f32, _>(&spec).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OmniDSP<B> {
    backend: B,
}

impl<B> OmniDSP<B> {
    /// Create an engine wrapping the given backend.
    #[must_use]
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    /// The backend, as the [`VecOps`] provider — the seam for direct vector math.
    ///
    /// The omni modules route their frequency-domain glue through the backend's
    /// `VecOps` transitively (choosing a tuned backend accelerates the
    /// bulk-transform modules with no extra API; the time-domain leaves — IIR,
    /// resampling — are native-or-scalar, not VecOps-routed).  For *custom*
    /// accelerated glue, reach the same tuned vector math directly:
    /// `dsp.vecops().dot(&a, &b)`.
    #[must_use]
    pub const fn vecops(&self) -> &B {
        &self.backend
    }

    /// Create a stateless **Plan** from the given specification at precision `T`.
    ///
    /// The spec type `S` drives dispatch — the backend must implement
    /// `CreatePlan<S>` for the spec type and be a `Backend<T>` (the real-DFT
    /// family plus vector ops) for the precision.  The turbofish names the
    /// precision: `dsp.create_plan::<f32>(&spec)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn create_plan<T, S>(&self, spec: &S) -> Result<<B as CreatePlan<S>>::Plan<T>>
    where
        B: CreatePlan<S> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.backend.create_plan::<T>(spec)
    }

    /// Create a stateful **Processor** from the given specification at precision
    /// `T`.
    ///
    /// The stateful peer of [`create_plan`](Self::create_plan): the backend must
    /// implement `CreateProc<S>` and be a `Backend<T>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or processor creation fails.
    pub fn create_proc<T, S>(&self, spec: &S) -> Result<<B as CreateProc<S>>::Proc<T>>
    where
        B: CreateProc<S> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.backend.create_proc::<T>(spec)
    }

    /// Create a complex-to-complex DFT plan
    /// ([`DftC2c`](omnidsp_core::traits::dft::DftC2c)).
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn dft_c2c<T>(&self, spec: &DftC2cSpec) -> Result<<B as CreatePlan<DftC2cSpec>>::Plan<T>>
    where
        B: CreatePlan<DftC2cSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_plan::<T, _>(spec)
    }

    /// Create a real-to-complex DFT plan
    /// ([`DftR2c`](omnidsp_core::traits::dft::DftR2c)): `length` real samples →
    /// a `length / 2 + 1` complex half-spectrum.  Hermitian-shaped output.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn dft_r2c<T>(&self, spec: &DftR2cSpec) -> Result<<B as CreatePlan<DftR2cSpec>>::Plan<T>>
    where
        B: CreatePlan<DftR2cSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_plan::<T, _>(spec)
    }

    /// Create a complex-to-real DFT plan
    /// ([`DftC2r`](omnidsp_core::traits::dft::DftC2r)): a `length / 2 + 1`
    /// complex half-spectrum → `length` real samples.  Hermitian-shaped input.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn dft_c2r<T>(&self, spec: &DftC2rSpec) -> Result<<B as CreatePlan<DftC2rSpec>>::Plan<T>>
    where
        B: CreatePlan<DftC2rSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_plan::<T, _>(spec)
    }

    /// Create a FIR filter processor.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or processor creation fails.
    pub fn fir<T>(&self, spec: &FirSpec) -> Result<<B as CreateProc<FirSpec>>::Proc<T>>
    where
        B: CreateProc<FirSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_proc::<T, _>(spec)
    }

    /// Create an IIR filter processor.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or processor creation fails.
    pub fn iir<T>(&self, spec: &IirSpec) -> Result<<B as CreateProc<IirSpec>>::Proc<T>>
    where
        B: CreateProc<IirSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_proc::<T, _>(spec)
    }

    /// Create a convolution plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn conv<T>(&self, spec: &ConvSpec) -> Result<<B as CreatePlan<ConvSpec>>::Plan<T>>
    where
        B: CreatePlan<ConvSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_plan::<T, _>(spec)
    }

    /// Create a resampler processor.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or processor creation fails.
    pub fn resample<T>(
        &self,
        spec: &ResampleSpec,
    ) -> Result<<B as CreateProc<ResampleSpec>>::Proc<T>>
    where
        B: CreateProc<ResampleSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_proc::<T, _>(spec)
    }

    /// Create a batch CQT plan (stateless, parallel per-frame analyzer).
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn cqt<T>(&self, spec: &CqtSpec) -> Result<<B as CreatePlan<CqtSpec>>::Plan<T>>
    where
        B: CreatePlan<CqtSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_plan::<T, _>(spec)
    }

    /// Create a streaming, newest-anchored CQT processor over the same
    /// [`CqtSpec`] as [`cqt`](Self::cqt).
    ///
    /// A stateful `&mut self` analyzer whose per-bin latency collapses to the
    /// Gabor floor `Q/f` rather than the whole window length.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or processor creation fails.
    pub fn cqt_proc<T>(&self, spec: &CqtSpec) -> Result<<B as CreateProc<CqtSpec>>::Proc<T>>
    where
        B: CreateProc<CqtSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_proc::<T, _>(spec)
    }

    /// Create a Hilbert transform (analytic signal) plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn hilbert<T>(&self, spec: &HilbertSpec) -> Result<<B as CreatePlan<HilbertSpec>>::Plan<T>>
    where
        B: CreatePlan<HilbertSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_plan::<T, _>(spec)
    }

    /// Create a DCT plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn dct<T>(&self, spec: &DctSpec) -> Result<<B as CreatePlan<DctSpec>>::Plan<T>>
    where
        B: CreatePlan<DctSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_plan::<T, _>(spec)
    }

    /// Create a cross-correlation plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the spec is invalid or plan creation fails.
    pub fn xcorr<T>(
        &self,
        spec: &CrossCorrSpec,
    ) -> Result<<B as CreatePlan<CrossCorrSpec>>::Plan<T>>
    where
        B: CreatePlan<CrossCorrSpec> + omnidsp_core::dispatch::Backend<T>,
        T: omnidsp_core::types::DspFloat + core::ops::AddAssign + core::ops::MulAssign,
    {
        self.create_plan::<T, _>(spec)
    }
}

// ─── Named constructors ─────────────────────────────────────────────

impl OmniDSP<RustBackend> {
    /// Create an engine using the pure Rust fallback backend.
    #[must_use]
    pub const fn rust() -> Self {
        Self::new(RustBackend::new())
    }
}

#[cfg(feature = "onemkl")]
impl OmniDSP<crate::OneMklBackend> {
    /// Create an engine using the Intel oneMKL backend.
    ///
    /// Uses oneMKL DFTI FFTs with accelerated VM/BLAS vector ops; convolution
    /// and cross-correlation compose for free over the accelerated DFT.  Requires
    /// Intel oneMKL to be installed and linked (the `onemkl` Cargo feature).
    #[must_use]
    pub const fn onemkl() -> Self {
        Self::new(crate::OneMklBackend::new())
    }
}

impl crate::Auto {
    /// Create an engine using the best compiled-in backend.
    #[must_use]
    pub const fn auto() -> Self {
        Self::new(crate::Best::new())
    }
}

impl Default for crate::Auto {
    fn default() -> Self {
        Self::auto()
    }
}
