// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Cross-correlation module — FFT-based fast cross-correlation.
//!
//! [`OmniCrossCorr`] computes the full linear cross-correlation of two
//! real-valued signals using the frequency-domain method:
//! `xcorr(a, b) = c2r(r2c(a) · conj(r2c(b)))`.
//!
//! Generic over any [`DftR2c`] / [`DftC2r`] and [`VecOps`] implementation
//! (ADR-009 §6): `r2c(a)` and `conj(r2c(b))` are both Hermitian, so their
//! product is the half-spectrum of the (real) cross-correlation.  The inverse
//! c2r factory is Hermitian-shaped ([`HermitianC2r`]) so the DC/Nyquist
//! boundary is projected before the transform (ADR-010 §2/§5).  Internal
//! scratch buffers are behind a [`Mutex`] so that the plan satisfies
//! `Send + Sync` while taking `&self`.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign};
use std::sync::Mutex;

use num_complex::Complex;
use num_traits::Float;

use crate::error::{Error, Result};
use crate::hermitian::{HermitianC2r, HermitianC2rPlan};
use crate::traits::dft::{DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan, DftR2cSpec};
use crate::traits::vecops::VecOps;
use crate::types::DspFloat;

// ─── Spec ────────────────────────────────────────────────────────────────

/// Normalization applied to the cross-correlation output.
///
/// Reserved field (ADR-007 §4, surface-lock item 8): only
/// [`None`](Self::None) — today's raw behaviour — exists for now.  The
/// normalized variants land with the `AutoCorr` / `PSD` normalization work
/// (tickets 18/19).  Marked `#[non_exhaustive]` so adding variants later is
/// not a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CrossCorrNorm {
    /// Raw, unnormalized cross-correlation (the convolution-theorem result).
    #[default]
    None,
}

/// Cross-correlation operation specification.
///
/// Describes the input signal lengths.  The output length is always
/// `a_len + b_len - 1` (full linear cross-correlation).
///
/// The type parameter `T` ties the spec to a specific float type, making
/// specs fully self-describing for the dispatch layer's `CreatePlan<S>` trait.
/// Fields are private and the spec is valid-by-construction (ADR-006 §4).
///
/// # Examples
///
/// ```
/// use omnidsp_core::modules::xcorr::CrossCorrSpec;
///
/// let spec = CrossCorrSpec::<f64>::new(1024, 256).unwrap();
/// assert_eq!(spec.a_len(), 1024);
/// assert_eq!(spec.b_len(), 256);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CrossCorrSpec<T> {
    a_len: usize,
    b_len: usize,
    norm: CrossCorrNorm,
    _marker: PhantomData<T>,
}

impl<T> PartialEq for CrossCorrSpec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.a_len == other.a_len && self.b_len == other.b_len && self.norm == other.norm
    }
}

impl<T> Eq for CrossCorrSpec<T> {}

impl<T> CrossCorrSpec<T> {
    /// Create a new cross-correlation spec with the given input lengths.
    ///
    /// The [`norm`](CrossCorrSpec::norm) defaults to [`CrossCorrNorm::None`]
    /// (raw cross-correlation).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if either `a_len` or `b_len` is zero.
    pub fn new(a_len: usize, b_len: usize) -> Result<Self> {
        if a_len == 0 || b_len == 0 {
            return Err(Error::InvalidSpec(
                "cross-correlation input lengths must be non-zero".into(),
            ));
        }
        Ok(Self {
            a_len,
            b_len,
            norm: CrossCorrNorm::None,
            _marker: PhantomData,
        })
    }

    /// Length of the first input signal.
    #[must_use]
    pub const fn a_len(&self) -> usize {
        self.a_len
    }

    /// Length of the second input signal (reference/template).
    #[must_use]
    pub const fn b_len(&self) -> usize {
        self.b_len
    }

    /// Output normalization convention (reserved; currently always
    /// [`CrossCorrNorm::None`]).
    #[must_use]
    pub const fn norm(&self) -> CrossCorrNorm {
        self.norm
    }

    /// Output length: `a_len + b_len - 1`.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.a_len + self.b_len - 1
    }
}

// ─── Factory ─────────────────────────────────────────────────────────────

/// Generic cross-correlation factory backed by [`DftR2c`] / [`DftC2r`] and
/// [`VecOps`].
///
/// Creates [`OmniCrossCorrPlan`]s for specific input lengths.  The factory owns
/// the real-DFT factories (`r2c` forward, `c2r` inverse) and the `VecOps`
/// instance; plans own their sub-plans.  The c2r factory is Hermitian-shaped
/// internally (ADR-010 §5).
#[derive(Debug, Clone)]
pub struct OmniCrossCorr<R, C, V> {
    r2c: R,
    c2r: C,
    vecops: V,
}

impl<R, C, V> OmniCrossCorr<R, C, V> {
    /// Create a new cross-correlation factory.
    #[must_use]
    pub const fn new(r2c: R, c2r: C, vecops: V) -> Self {
        Self { r2c, c2r, vecops }
    }
}

// ─── Plan ────────────────────────────────────────────────────────────────

/// Execution plan for a cross-correlation operation.
///
/// Created by [`OmniCrossCorr::create_plan`].  Immutable and thread-safe
/// (`Send + Sync`).
///
/// Output length is `a_len + b_len - 1` (full linear cross-correlation).
/// The output represents lag values from `-(b_len-1)` to `+(a_len-1)`.
///
/// **Memory:** allocates one real buffer of `fft_length` plus two complex
/// half-spectrum buffers of `fft_length / 2 + 1` (behind a [`Mutex`] for
/// thread safety).
///
/// `RP` is the forward [`DftR2cPlan`]; `CP` is the inverse [`DftC2rPlan`]
/// (in practice a [`HermitianC2rPlan`]).  Neither is ever named by users.
pub struct OmniCrossCorrPlan<T, RP, CP, V> {
    fwd: RP,
    inv: CP,
    vecops: V,
    scratch: Mutex<Scratch<T>>,
    a_len: usize,
    b_len: usize,
    output_len: usize,
}

impl<T, RP, CP, V> fmt::Debug for OmniCrossCorrPlan<T, RP, CP, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniCrossCorrPlan")
            .field("a_len", &self.a_len)
            .field("b_len", &self.b_len)
            .field("output_len", &self.output_len)
            .finish_non_exhaustive()
    }
}

/// Scratch buffers for the FFT cross-correlation pipeline.
struct Scratch<T> {
    /// Real pad buffer (`fft_len`): holds each zero-padded input before the
    /// forward r2c consumes it, then receives the c2r real output.
    real: Vec<T>,
    /// Half-spectrum of the first input → holds the frequency-domain product.
    half_a: Vec<Complex<T>>,
    /// Half-spectrum of the second input (conjugated in-place).
    half_b: Vec<Complex<T>>,
}

// ─── Plan implementation ─────────────────────────────────────────────────

impl<T, RP, CP, V> OmniCrossCorrPlan<T, RP, CP, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    RP: DftR2cPlan<T>,
    CP: DftC2rPlan<T>,
    V: VecOps<T>,
{
    /// Compute the cross-correlation of `a` and `b`.
    ///
    /// `a` must have length `a_len`, `b` must have length `b_len`, and
    /// `output` must have length `a_len + b_len - 1`.
    ///
    /// The output is the full linear cross-correlation:
    /// `output[k] = sum_n a[n] * b[n - k + (b_len - 1)]`
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer length does not match the expected size.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire FFT pipeline"
    )]
    pub fn process(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        if a.len() != self.a_len {
            return Err(Error::BufferMismatch {
                expected: self.a_len,
                actual: a.len(),
            });
        }
        if b.len() != self.b_len {
            return Err(Error::BufferMismatch {
                expected: self.b_len,
                actual: b.len(),
            });
        }
        if output.len() != self.output_len {
            return Err(Error::BufferMismatch {
                expected: self.output_len,
                actual: output.len(),
            });
        }

        let mut guard = self
            .scratch
            .lock()
            .map_err(|e| Error::Internal(format!("scratch buffer lock poisoned: {e}")))?;

        let Scratch {
            real,
            half_a,
            half_b,
        } = &mut *guard;

        // 1. Zero-pad a → real, then forward r2c → half_a (real is consumed).
        pad_real(a, real);
        self.fwd.process(real, half_a)?;

        // 2. Zero-pad b → real (reuse), then forward r2c → half_b.
        pad_real(b, real);
        self.fwd.process(real, half_b)?;

        // 3. Conjugate the half-spectrum of b in place: conj(r2c(b)).
        self.vecops.conj_inplace(half_b);

        // 4. Half-spectrum element-wise multiply: half_a = r2c(a) · conj(r2c(b)).
        self.vecops.cmul_inplace(half_a, half_b)?;

        // 5. Inverse c2r: half_a → real (HermitianC2r projects DC/Nyquist).
        self.inv.process(half_a, real)?;

        // 6. Extract real output in full cross-correlation order.
        //    The IFFT produces circular correlation in wrap-around order:
        //      indices 0..a_len → non-negative lags (lag 0 to lag a_len-1)
        //      indices fft_len-(b_len-1)..fft_len → negative lags
        //    Full output order: negative lags first, then non-negative.
        let neg_lags = self.b_len - 1;
        let fft_len = real.len();
        output[..neg_lags].copy_from_slice(&real[fft_len - neg_lags..fft_len]);
        output[neg_lags..].copy_from_slice(&real[..self.a_len]);

        Ok(())
    }

    /// Length of the first input signal.
    #[must_use]
    pub const fn a_len(&self) -> usize {
        self.a_len
    }

    /// Length of the second input signal.
    #[must_use]
    pub const fn b_len(&self) -> usize {
        self.b_len
    }

    /// Output length (`a_len + b_len - 1`).
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.output_len
    }
}

// ─── Plan trait ──────────────────────────────────────────────────────────

/// Execution object for a configured cross-correlation.
///
/// The named, `Send + Sync` plan trait for the cross-correlation module,
/// mirroring [`ConvPlan`](crate::traits::conv::ConvPlan) /
/// [`DctPlan`](crate::traits::dct::DctPlan) (ADR-006 §2 eager plan traits,
/// ADR-007 §6).  It lets `process` be called generically (e.g. by the shared
/// conformance suite) without naming the concrete plan.  Implemented by
/// [`OmniCrossCorrPlan`]; vendor overrides may implement it directly.
pub trait CrossCorrPlan<T>: Send + Sync {
    /// Compute the full linear cross-correlation of `a` and `b`, writing the
    /// result to `output`.
    ///
    /// `a` and `b` must have the lengths the plan was created for; `output`
    /// must have length `a_len + b_len - 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer length does not match the expected size.
    fn process(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()>;
}

impl<T, RP, CP, V> CrossCorrPlan<T> for OmniCrossCorrPlan<T, RP, CP, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    RP: DftR2cPlan<T>,
    CP: DftC2rPlan<T>,
    V: VecOps<T>,
{
    fn process(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        // Delegate to the inherent `process`.  Inherent methods take precedence
        // over trait methods in resolution, so this is not recursive.
        self.process(a, b, output)
    }
}

// ─── Factory implementation ──────────────────────────────────────────────

/// The plan [`OmniCrossCorr::create_plan`] returns: the forward [`DftR2c`] plan
/// plus the Hermitian-shaped inverse [`DftC2r`] plan, over `VecOps` `V`.
type ShapedCrossCorrPlan<T, R, C, V> =
    OmniCrossCorrPlan<T, <R as DftR2c<T>>::Plan, HermitianC2rPlan<<C as DftC2r<T>>::Plan>, V>;

impl<R, C, V> OmniCrossCorr<R, C, V> {
    /// Create a cross-correlation plan from the given specification.
    ///
    /// The plan preallocates real-DFT sub-plans and scratch buffers so that
    /// execution is allocation-free.
    ///
    /// # Errors
    ///
    /// Returns an error if the FFT length overflows or DFT plan creation
    /// fails.  The length invariants are enforced by [`CrossCorrSpec::new`]
    /// (ADR-006 §4), so they are not re-checked here.
    pub fn create_plan<T>(&self, spec: &CrossCorrSpec<T>) -> Result<ShapedCrossCorrPlan<T, R, C, V>>
    where
        T: DspFloat + AddAssign + MulAssign,
        R: DftR2c<T>,
        C: DftC2r<T> + Clone,
        V: VecOps<T>,
    {
        let output_len = spec.output_len();
        let fft_len = output_len.checked_next_power_of_two().ok_or_else(|| {
            Error::InvalidSpec("cross-correlation FFT length overflow".to_owned())
        })?;
        let bins = fft_len / 2 + 1;

        // Cross-correlation theorem: xcorr(a,b) = c2r(r2c(a) · conj(r2c(b))).
        // DftNorm::Inverse gives c2r(r2c(x)) = x (1/N on inverse only).
        let fwd_spec = DftR2cSpec::new(fft_len, DftNorm::Inverse)?;
        let fwd = self.r2c.create_plan(&fwd_spec)?;
        // Hermitian-shaped inverse: the product of two Hermitian half-spectra
        // is Hermitian, but float drift leaves ~epsilon imaginary at DC/Nyquist
        // that the projection clears before the inverse (ADR-010 §2/§5).
        let inv_spec = DftC2rSpec::new(fft_len, DftNorm::Inverse)?;
        let inv = HermitianC2r::new(self.c2r.clone()).create_plan(&inv_spec)?;

        let zero = Complex::new(T::zero(), T::zero());
        let scratch = Scratch {
            real: vec![T::zero(); fft_len],
            half_a: vec![zero; bins],
            half_b: vec![zero; bins],
        };

        Ok(OmniCrossCorrPlan {
            fwd,
            inv,
            vecops: self.vecops.clone(),
            scratch: Mutex::new(scratch),
            a_len: spec.a_len(),
            b_len: spec.b_len(),
            output_len,
        })
    }
}

/// Zero-pad a real slice into a pre-allocated real buffer.
///
/// The first `input.len()` elements are copied; the remainder is zeroed.
fn pad_real<T: Float>(input: &[T], buf: &mut [T]) {
    let n = input.len();
    buf[..n].copy_from_slice(input);
    for x in &mut buf[n..] {
        *x = T::zero();
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;
    use crate::test_utils::{TestDftC2r, TestDftR2c, TestVecOps};

    fn make_factory() -> OmniCrossCorr<TestDftR2c, TestDftC2r, TestVecOps> {
        OmniCrossCorr::new(TestDftR2c, TestDftC2r, TestVecOps)
    }

    #[test]
    fn spec_output_len() {
        let spec = CrossCorrSpec::<f64>::new(5, 3).expect("valid xcorr spec");
        assert_eq!(
            spec.output_len(),
            7,
            "output_len should be a_len + b_len - 1"
        );
    }

    #[test]
    fn spec_equality() {
        let a = CrossCorrSpec::<f64>::new(10, 5).expect("valid xcorr spec");
        let b = CrossCorrSpec::<f64>::new(10, 5).expect("valid xcorr spec");
        let c = CrossCorrSpec::<f64>::new(10, 6).expect("valid xcorr spec");
        assert_eq!(a, b, "equal specs should compare equal");
        assert_ne!(a, c, "different specs should not compare equal");
    }

    #[test]
    fn zero_a_len_rejected() {
        assert!(
            CrossCorrSpec::<f64>::new(0, 5).is_err(),
            "zero a_len should be rejected by the spec constructor"
        );
    }

    #[test]
    fn zero_b_len_rejected() {
        assert!(
            CrossCorrSpec::<f64>::new(5, 0).is_err(),
            "zero b_len should be rejected by the spec constructor"
        );
    }

    #[test]
    fn buffer_mismatch_a() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(4, 3).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let a = [1.0; 5]; // wrong length
        let b = [1.0; 3];
        let mut out = vec![0.0; 6];
        let result = plan.process(&a, &b, &mut out);
        assert!(result.is_err(), "wrong a length should fail");
    }

    #[test]
    fn buffer_mismatch_b() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(4, 3).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let a = [1.0; 4];
        let b = [1.0; 2]; // wrong length
        let mut out = vec![0.0; 6];
        let result = plan.process(&a, &b, &mut out);
        assert!(result.is_err(), "wrong b length should fail");
    }

    #[test]
    fn buffer_mismatch_output() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(4, 3).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let a = [1.0; 4];
        let b = [1.0; 3];
        let mut out = vec![0.0; 5]; // wrong length (should be 6)
        let result = plan.process(&a, &b, &mut out);
        assert!(result.is_err(), "wrong output length should fail");
    }

    /// Cross-correlation of a signal with itself at zero lag should be the
    /// energy (sum of squares).
    #[test]
    fn autocorrelation_peak() {
        let factory = make_factory();
        let a = [1.0, 2.0, 3.0, 4.0];
        let spec = CrossCorrSpec::<f64>::new(a.len(), a.len()).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut output = vec![0.0; spec.output_len()];
        plan.process(&a, &a, &mut output)
            .expect("process should succeed");

        // Zero-lag is at index b_len - 1 = 3.
        let zero_lag = output[a.len() - 1];
        let energy: f64 = a.iter().map(|x| x * x).sum();
        assert!(
            (zero_lag - energy).abs() < 1e-10,
            "zero-lag autocorrelation should equal signal energy: got {zero_lag}, expected {energy}"
        );

        // Zero-lag should be the maximum.
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val <= zero_lag + 1e-10,
                "autocorrelation should peak at zero lag: output[{i}] = {val} > {zero_lag}"
            );
        }
    }

    /// Verify against the textbook definition:
    /// `xcorr(a, b)[k] = sum_n a[n] * b[n - k + (b_len - 1)]`
    #[test]
    fn known_cross_correlation() {
        let factory = make_factory();
        let a = [1.0, 2.0, 3.0];
        let b = [0.5, 1.0];
        let spec = CrossCorrSpec::<f64>::new(a.len(), b.len()).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut output = vec![0.0; spec.output_len()];
        plan.process(&a, &b, &mut output)
            .expect("process should succeed");

        // Manual computation of full cross-correlation:
        // output[0] = a[0]*b[1]                     = 1.0*1.0 = 1.0
        // output[1] = a[0]*b[0] + a[1]*b[1]         = 1.0*0.5 + 2.0*1.0 = 2.5
        // output[2] = a[1]*b[0] + a[2]*b[1]         = 2.0*0.5 + 3.0*1.0 = 4.0
        // output[3] = a[2]*b[0]                     = 3.0*0.5 = 1.5
        let expected = [1.0, 2.5, 4.0, 1.5];

        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "output[{i}] = {got}, expected {exp}"
            );
        }
    }

    /// Cross-correlation with a unit impulse at index 0 should yield `a`
    /// starting at offset `b_len - 1`.
    #[test]
    fn impulse_shift() {
        let factory = make_factory();
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 0.0];
        let spec = CrossCorrSpec::<f64>::new(a.len(), b.len()).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut output = vec![0.0; spec.output_len()];
        plan.process(&a, &b, &mut output)
            .expect("process should succeed");

        // xcorr(a, delta_0) → a appears starting at index b_len-1.
        let expected = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "output[{i}] = {got}, expected {exp}"
            );
        }
    }

    /// xcorr(a,b) reversed equals xcorr(b,a) (time-reversal property).
    #[test]
    fn time_reversal_property() {
        let factory = make_factory();
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];

        let spec_ab = CrossCorrSpec::<f64>::new(a.len(), b.len()).expect("valid xcorr spec");
        let plan_ab = factory
            .create_plan(&spec_ab)
            .expect("plan creation should succeed");
        let mut out_ab = vec![0.0; spec_ab.output_len()];
        plan_ab
            .process(&a, &b, &mut out_ab)
            .expect("process should succeed");

        let spec_ba = CrossCorrSpec::<f64>::new(b.len(), a.len()).expect("valid xcorr spec");
        let plan_ba = factory
            .create_plan(&spec_ba)
            .expect("plan creation should succeed");
        let mut out_ba = vec![0.0; spec_ba.output_len()];
        plan_ba
            .process(&b, &a, &mut out_ba)
            .expect("process should succeed");

        // xcorr(a,b) reversed should equal xcorr(b,a).
        let reversed: Vec<f64> = out_ab.iter().rev().copied().collect();
        for (i, (&got, &exp)) in reversed.iter().zip(out_ba.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "reversed xcorr(a,b)[{i}] = {got}, expected xcorr(b,a)[{i}] = {exp}"
            );
        }
    }

    /// Plan is reusable — multiple calls produce identical results.
    #[test]
    fn plan_reuse() {
        let factory = make_factory();
        let a = [1.0, -1.0, 2.0, -2.0];
        let b = [0.5, 1.5];
        let spec = CrossCorrSpec::<f64>::new(a.len(), b.len()).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut out1 = vec![0.0; spec.output_len()];
        let mut out2 = vec![0.0; spec.output_len()];

        plan.process(&a, &b, &mut out1)
            .expect("first process should succeed");
        plan.process(&a, &b, &mut out2)
            .expect("second process should succeed");

        for (i, (&v1, &v2)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-14,
                "results should be identical on reuse: output[{i}] = {v1} vs {v2}"
            );
        }
    }

    /// Accessor methods return correct values.
    #[test]
    fn accessor_methods() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(8, 4).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        assert_eq!(plan.a_len(), 8, "a_len accessor should match");
        assert_eq!(plan.b_len(), 4, "b_len accessor should match");
        assert_eq!(plan.output_len(), 11, "output_len accessor should match");
    }

    /// A generic consumer bound on `CrossCorrPlan` compiles and runs.
    #[test]
    fn implements_plan_trait() {
        fn check<P: CrossCorrPlan<f64>>(plan: &P, a: &[f64], b: &[f64], out: &mut [f64]) {
            plan.process(a, b, out)
                .expect("trait process should succeed");
        }

        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(4, 3).expect("valid xcorr spec");
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.5, 1.0, 1.5];
        let mut out = vec![0.0; spec.output_len()];
        check(&plan, &a, &b, &mut out);

        // The trait method must agree with the inherent `process`.
        let mut direct = vec![0.0; spec.output_len()];
        plan.process(&a, &b, &mut direct)
            .expect("inherent process should succeed");
        for (i, (t, d)) in out.iter().zip(direct.iter()).enumerate() {
            assert!(
                (t - d).abs() < 1e-15,
                "trait vs inherent mismatch at index {i}"
            );
        }
    }
}
