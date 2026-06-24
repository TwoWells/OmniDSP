// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Cross-correlation module — direct and FFT-based fast cross-correlation.
//!
//! [`OmniCrossCorr`] computes the full linear cross-correlation of two
//! real-valued signals.  It supports both a time-domain (direct) method and a
//! frequency-domain (FFT) method, selected via [`CorrMethod`].
//!
//! The FFT path uses the cross-correlation theorem:
//! `xcorr(a, b) = c2r(r2c(a) · conj(r2c(b)))`.  `r2c(a)` and `conj(r2c(b))` are
//! both Hermitian, so their product is the half-spectrum of the (real)
//! cross-correlation.  The inverse c2r factory is Hermitian-shaped
//! ([`HermitianC2r`]) so the DC/Nyquist boundary is projected before the
//! transform.  Internal scratch buffers are behind a [`Mutex`] so that the plan
//! satisfies `Send + Sync` while taking `&self`.
//!
//! The direct path evaluates the cross-correlation sum in the time domain,
//! producing output bit-for-bit consistent in convention with the FFT path:
//! full linear cross-correlation, length `a_len + b_len - 1`, with identical
//! lag-0 placement and ordering.  The [`recommend_corr_method`] function
//! provides an operation-count heuristic for the [`CorrMethod::Auto`] case.

use std::fmt;
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
/// Reserved field: only
/// [`None`](Self::None) — today's raw behaviour — exists for now.  The
/// normalized variants land with the `AutoCorr` / `PSD` normalization work.
/// Marked `#[non_exhaustive]` so adding variants later is
/// not a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CrossCorrNorm {
    /// Raw, unnormalized cross-correlation (the convolution-theorem result).
    #[default]
    None,
}

/// Cross-correlation implementation method.
///
/// Dedicated to cross-correlation rather than reusing the convolution method
/// enum: each operation owns its own primitive surface, so the cross-correlation
/// public API does not depend on the convolution one.  The two are decoupled by
/// default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrMethod {
    /// Backend decides direct vs. FFT.
    ///
    /// How and when `Auto` resolves is a backend detail: the pure-Rust floor
    /// resolves it at plan creation from operation counts
    /// (`recommend_corr_method`); a vendor backend may defer to its own internal
    /// auto-select mode.
    Auto,
    /// Frequency-domain (FFT-based) cross-correlation — `O(N log N)`.
    Fft,
    /// Time-domain (direct) cross-correlation — `O(a_len × b_len)`.
    Direct,
}

/// Cross-correlation operation specification.
///
/// Describes the input signal lengths and preferred implementation method.
/// The output length is always `a_len + b_len - 1` (full linear
/// cross-correlation).
///
/// The spec is non-generic; precision is chosen at `create_plan::<T>`.  Fields
/// are private and the spec is valid-by-construction.
///
/// # Examples
///
/// ```
/// use omnidsp_core::modules::xcorr::{CorrMethod, CrossCorrSpec};
///
/// // Let the backend pick direct vs. FFT
/// let spec = CrossCorrSpec::new(1024, 256, CorrMethod::Auto).unwrap();
/// assert_eq!(spec.a_len(), 1024);
/// assert_eq!(spec.b_len(), 256);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossCorrSpec {
    a_len: usize,
    b_len: usize,
    method: CorrMethod,
    norm: CrossCorrNorm,
}

impl CrossCorrSpec {
    /// Create a new cross-correlation spec with the given input lengths and
    /// method.
    ///
    /// The [`norm`](CrossCorrSpec::norm) defaults to [`CrossCorrNorm::None`]
    /// (raw cross-correlation).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if either `a_len` or `b_len` is zero.
    pub fn new(a_len: usize, b_len: usize, method: CorrMethod) -> Result<Self> {
        if a_len == 0 || b_len == 0 {
            return Err(Error::InvalidSpec(
                "cross-correlation input lengths must be non-zero".into(),
            ));
        }
        Ok(Self {
            a_len,
            b_len,
            method,
            norm: CrossCorrNorm::None,
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

    /// Preferred implementation method.  [`CorrMethod::Auto`] is resolved
    /// at plan creation time; the plan itself always uses a concrete method.
    #[must_use]
    pub const fn method(&self) -> CorrMethod {
        self.method
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

// ─── Public heuristic ──────────────────────────────────────────────────────

/// Recommend a cross-correlation method based on operation counts.
///
/// Compares the work of direct cross-correlation (`a_len * b_len` ops) against
/// FFT cross-correlation (`~3 * N * log2(N) + N` ops where `N` is the FFT
/// length).  Returns whichever is cheaper.
///
/// This is a generic heuristic that ignores constant factors (cache
/// effects, SIMD width, FFT implementation quality).  Vendor-specific
/// cross-correlation implementations may use a different decision boundary.
#[must_use]
pub fn recommend_corr_method(a_len: usize, b_len: usize) -> CorrMethod {
    if a_len == 0 || b_len == 0 {
        return CorrMethod::Direct;
    }

    let direct_ops = a_len as u64 * b_len as u64;

    let output_len = a_len + b_len - 1;
    let fft_len = output_len.next_power_of_two() as u64;
    // 3 FFTs (2 forward + 1 inverse) each ~N*log2(N), plus N for the
    // element-wise complex multiply.
    let log2_n = u64::BITS - fft_len.leading_zeros();
    let fft_ops = 3 * fft_len * u64::from(log2_n) + fft_len;

    if direct_ops <= fft_ops {
        CorrMethod::Direct
    } else {
        CorrMethod::Fft
    }
}

// ─── Factory ─────────────────────────────────────────────────────────────

/// Generic cross-correlation factory backed by [`DftR2c`] / [`DftC2r`] and
/// [`VecOps`].
///
/// Creates [`OmniCrossCorrPlan`]s for specific input lengths.  The factory owns
/// the real-DFT factories (`r2c` forward, `c2r` inverse) and the `VecOps`
/// instance; plans own their sub-plans.  The c2r factory is Hermitian-shaped
/// internally.
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
/// **Memory:** the direct path allocates nothing beyond the plan struct.  The
/// FFT path allocates one real buffer of `fft_length` plus two complex
/// half-spectrum buffers of `fft_length / 2 + 1` (behind a [`Mutex`] for
/// thread safety).
///
/// `RP` is the forward [`DftR2cPlan`]; `CP` is the inverse [`DftC2rPlan`]
/// (in practice a [`HermitianC2rPlan`]).  Neither is ever named by users.
pub struct OmniCrossCorrPlan<T, RP, CP, V> {
    inner: PlanInner<T, RP, CP, V>,
    a_len: usize,
    b_len: usize,
    output_len: usize,
}

impl<T, RP, CP, V> fmt::Debug for OmniCrossCorrPlan<T, RP, CP, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let method = match &self.inner {
            PlanInner::Direct => "Direct",
            PlanInner::Fft { .. } => "Fft",
        };
        f.debug_struct("OmniCrossCorrPlan")
            .field("method", &method)
            .field("a_len", &self.a_len)
            .field("b_len", &self.b_len)
            .field("output_len", &self.output_len)
            .finish_non_exhaustive()
    }
}

// ─── Plan internals ────────────────────────────────────────────────────────

enum PlanInner<T, RP, CP, V> {
    Direct,
    Fft { state: FftState<T, RP, CP, V> },
}

struct FftState<T, RP, CP, V> {
    /// Forward real-to-complex transform, reused for both inputs.
    fwd: RP,
    /// Inverse complex-to-real transform (Hermitian-shaped).
    inv: CP,
    vecops: V,
    scratch: Mutex<Scratch<T>>,
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
    T: DspFloat,
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
    pub fn execute(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
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

        match &self.inner {
            PlanInner::Direct => {
                Self::process_direct(a, b, output);
                Ok(())
            }
            PlanInner::Fft { state } => Self::process_fft(state, a, b, output),
        }
    }

    /// Direct (time-domain) cross-correlation: `O(a_len * b_len)`.
    ///
    /// Evaluates the full linear cross-correlation sum
    /// `output[k] = sum_n a[n] * b[n - k + (b_len - 1)]` for each output index
    /// `k`, matching the FFT path's convention exactly: lag 0 lands at index
    /// `b_len - 1`, and the ordering runs from the most negative lag to the
    /// most positive.
    fn process_direct(a: &[T], b: &[T], output: &mut [T]) {
        for o in output.iter_mut() {
            *o = T::zero();
        }
        // Accumulate by input pairing: a[i] * b[j] contributes to the output
        // index where the lag aligns them.  With output[k] indexing lag
        // k - (b_len - 1), the pairing of a[i] and b[j] (i.e. n = i,
        // n - k + (b_len - 1) = j) sits at k = i - j + (b_len - 1).
        let b_len = b.len();
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                output[i + (b_len - 1) - j] += ai * bj;
            }
        }
    }

    /// FFT-based (frequency-domain) cross-correlation over the real-DFT
    /// primitives.
    ///
    /// `xcorr(a,b) = c2r(r2c(a) · conj(r2c(b)))`: the product of the two
    /// Hermitian half-spectra is the half-spectrum of the (real)
    /// cross-correlation, so the element-wise multiply runs on the
    /// `fft_len / 2 + 1` bins directly.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "MutexGuard must live for the entire FFT pipeline"
    )]
    fn process_fft(
        state: &FftState<T, RP, CP, V>,
        a: &[T],
        b: &[T],
        output: &mut [T],
    ) -> Result<()> {
        let mut guard = state
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
        state.fwd.execute(real, half_a)?;

        // 2. Zero-pad b → real (reuse), then forward r2c → half_b.
        pad_real(b, real);
        state.fwd.execute(real, half_b)?;

        // 3. Conjugate the half-spectrum of b in place: conj(r2c(b)).
        state.vecops.conj_inplace(half_b);

        // 4. Half-spectrum element-wise multiply: half_a = r2c(a) · conj(r2c(b)).
        state.vecops.cmul_inplace(half_a, half_b)?;

        // 5. Inverse c2r: half_a → real (HermitianC2r projects DC/Nyquist).
        state.inv.execute(half_a, real)?;

        // 6. Extract real output in full cross-correlation order.
        //    The IFFT produces circular correlation in wrap-around order:
        //      indices 0..a_len → non-negative lags (lag 0 to lag a_len-1)
        //      indices fft_len-(b_len-1)..fft_len → negative lags
        //    Full output order: negative lags first, then non-negative.
        let neg_lags = b.len() - 1;
        let fft_len = real.len();
        output[..neg_lags].copy_from_slice(&real[fft_len - neg_lags..fft_len]);
        output[neg_lags..].copy_from_slice(&real[..a.len()]);

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
/// mirroring the eager plan traits [`ConvPlan`](crate::traits::conv::ConvPlan) /
/// [`DctPlan`](crate::traits::dct::DctPlan).  It lets `process` be called
/// generically (e.g. by the shared conformance suite) without naming the
/// concrete plan.  Implemented by
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
    fn execute(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()>;
}

impl<T, RP, CP, V> CrossCorrPlan<T> for OmniCrossCorrPlan<T, RP, CP, V>
where
    T: DspFloat,
    RP: DftR2cPlan<T>,
    CP: DftC2rPlan<T>,
    V: VecOps<T>,
{
    fn execute(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        // Delegate to the inherent `execute`.  Inherent methods take precedence
        // over trait methods in resolution, so this is not recursive.
        self.execute(a, b, output)
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
    /// execution is allocation-free.  [`CorrMethod::Auto`] is resolved here via
    /// [`recommend_corr_method`]; the plan always holds a concrete method.
    ///
    /// # Errors
    ///
    /// Returns an error if the FFT length overflows or DFT plan creation
    /// fails.  The length invariants are enforced by [`CrossCorrSpec::new`],
    /// so they are not re-checked here.
    pub fn create_plan<T>(&self, spec: &CrossCorrSpec) -> Result<ShapedCrossCorrPlan<T, R, C, V>>
    where
        T: DspFloat,
        R: DftR2c<T>,
        C: DftC2r<T> + Clone,
        V: VecOps<T>,
    {
        let output_len = spec.output_len();

        let method = match spec.method() {
            CorrMethod::Auto => recommend_corr_method(spec.a_len(), spec.b_len()),
            other => other,
        };

        let inner = match method {
            CorrMethod::Direct | CorrMethod::Auto => PlanInner::Direct,
            CorrMethod::Fft => {
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
                // that the projection clears before the inverse.
                let inv_spec = DftC2rSpec::new(fft_len, DftNorm::Inverse)?;
                let inv = HermitianC2r::new(self.c2r.clone()).create_plan(&inv_spec)?;

                let zero = Complex::new(T::zero(), T::zero());
                let scratch = Scratch {
                    real: vec![T::zero(); fft_len],
                    half_a: vec![zero; bins],
                    half_b: vec![zero; bins],
                };

                PlanInner::Fft {
                    state: FftState {
                        fwd,
                        inv,
                        vecops: self.vecops.clone(),
                        scratch: Mutex::new(scratch),
                    },
                }
            }
        };

        Ok(OmniCrossCorrPlan {
            inner,
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

    fn spec(a_len: usize, b_len: usize) -> CrossCorrSpec {
        CrossCorrSpec::new(a_len, b_len, CorrMethod::Fft).expect("valid xcorr spec")
    }

    fn spec_direct(a_len: usize, b_len: usize) -> CrossCorrSpec {
        CrossCorrSpec::new(a_len, b_len, CorrMethod::Direct).expect("valid xcorr spec")
    }

    fn spec_fft(a_len: usize, b_len: usize) -> CrossCorrSpec {
        CrossCorrSpec::new(a_len, b_len, CorrMethod::Fft).expect("valid xcorr spec")
    }

    #[test]
    fn spec_output_len() {
        let spec = CrossCorrSpec::new(5, 3, CorrMethod::Auto).expect("valid xcorr spec");
        assert_eq!(
            spec.output_len(),
            7,
            "output_len should be a_len + b_len - 1"
        );
    }

    #[test]
    fn spec_equality() {
        let a = CrossCorrSpec::new(10, 5, CorrMethod::Auto).expect("valid xcorr spec");
        let b = CrossCorrSpec::new(10, 5, CorrMethod::Auto).expect("valid xcorr spec");
        let c = CrossCorrSpec::new(10, 6, CorrMethod::Auto).expect("valid xcorr spec");
        assert_eq!(a, b, "equal specs should compare equal");
        assert_ne!(a, c, "different specs should not compare equal");
    }

    /// Specs that differ only in method are distinct.
    #[test]
    fn spec_method_inequality() {
        let direct = CrossCorrSpec::new(10, 5, CorrMethod::Direct).expect("valid xcorr spec");
        let fft = CrossCorrSpec::new(10, 5, CorrMethod::Fft).expect("valid xcorr spec");
        assert_ne!(
            direct, fft,
            "specs differing only in method should not compare equal"
        );
        assert_eq!(
            direct.method(),
            CorrMethod::Direct,
            "method accessor should return the configured method"
        );
        assert_eq!(
            fft.method(),
            CorrMethod::Fft,
            "method accessor should return the configured method"
        );
    }

    #[test]
    fn zero_a_len_rejected() {
        assert!(
            CrossCorrSpec::new(0, 5, CorrMethod::Auto).is_err(),
            "zero a_len should be rejected by the spec constructor"
        );
    }

    #[test]
    fn zero_b_len_rejected() {
        assert!(
            CrossCorrSpec::new(5, 0, CorrMethod::Auto).is_err(),
            "zero b_len should be rejected by the spec constructor"
        );
    }

    #[test]
    fn buffer_mismatch_a() {
        let factory = make_factory();
        let spec = spec(4, 3);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let a = [1.0; 5]; // wrong length
        let b = [1.0; 3];
        let mut out = vec![0.0; 6];
        let result = plan.execute(&a, &b, &mut out);
        assert!(result.is_err(), "wrong a length should fail");
    }

    #[test]
    fn buffer_mismatch_b() {
        let factory = make_factory();
        let spec = spec(4, 3);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let a = [1.0; 4];
        let b = [1.0; 2]; // wrong length
        let mut out = vec![0.0; 6];
        let result = plan.execute(&a, &b, &mut out);
        assert!(result.is_err(), "wrong b length should fail");
    }

    #[test]
    fn buffer_mismatch_output() {
        let factory = make_factory();
        let spec = spec(4, 3);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        let a = [1.0; 4];
        let b = [1.0; 3];
        let mut out = vec![0.0; 5]; // wrong length (should be 6)
        let result = plan.execute(&a, &b, &mut out);
        assert!(result.is_err(), "wrong output length should fail");
    }

    /// Cross-correlation of a signal with itself at zero lag should be the
    /// energy (sum of squares).
    #[test]
    fn autocorrelation_peak() {
        let factory = make_factory();
        let a = [1.0, 2.0, 3.0, 4.0];
        let spec = spec(a.len(), a.len());
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut output = vec![0.0; spec.output_len()];
        plan.execute(&a, &a, &mut output)
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
        let spec = spec(a.len(), b.len());
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut output = vec![0.0; spec.output_len()];
        plan.execute(&a, &b, &mut output)
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
        let spec = spec(a.len(), b.len());
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut output = vec![0.0; spec.output_len()];
        plan.execute(&a, &b, &mut output)
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

        let spec_ab = spec(a.len(), b.len());
        let plan_ab = factory
            .create_plan(&spec_ab)
            .expect("plan creation should succeed");
        let mut out_ab = vec![0.0; spec_ab.output_len()];
        plan_ab
            .execute(&a, &b, &mut out_ab)
            .expect("process should succeed");

        let spec_ba = spec(b.len(), a.len());
        let plan_ba = factory
            .create_plan(&spec_ba)
            .expect("plan creation should succeed");
        let mut out_ba = vec![0.0; spec_ba.output_len()];
        plan_ba
            .execute(&b, &a, &mut out_ba)
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
        let spec = spec(a.len(), b.len());
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let mut out1 = vec![0.0; spec.output_len()];
        let mut out2 = vec![0.0; spec.output_len()];

        plan.execute(&a, &b, &mut out1)
            .expect("first process should succeed");
        plan.execute(&a, &b, &mut out2)
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
        let spec = spec(8, 4);
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
            plan.execute(a, b, out)
                .expect("trait process should succeed");
        }

        let factory = make_factory();
        let spec = spec(4, 3);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");

        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.5, 1.0, 1.5];
        let mut out = vec![0.0; spec.output_len()];
        check(&plan, &a, &b, &mut out);

        // The trait method must agree with the inherent `process`.
        let mut direct = vec![0.0; spec.output_len()];
        plan.execute(&a, &b, &mut direct)
            .expect("inherent process should succeed");
        for (i, (t, d)) in out.iter().zip(direct.iter()).enumerate() {
            assert!(
                (t - d).abs() < 1e-15,
                "trait vs inherent mismatch at index {i}"
            );
        }
    }

    // ── Heuristic tests ────────────────────────────────────────────────

    #[test]
    fn recommend_small_prefers_direct() {
        // 4 * 4 = 16 ops direct, FFT would need N=8 → 3*8*3 + 8 = 80.
        assert_eq!(
            recommend_corr_method(4, 4),
            CorrMethod::Direct,
            "small inputs should prefer direct"
        );
    }

    #[test]
    fn recommend_large_prefers_fft() {
        // 1024 * 1024 = 1M ops direct, FFT N=2048 → 3*2048*11 + 2048 ≈ 69K.
        assert_eq!(
            recommend_corr_method(1024, 1024),
            CorrMethod::Fft,
            "large inputs should prefer FFT"
        );
    }

    #[test]
    fn recommend_asymmetric_short_template() {
        // 1 * 1024 = 1024 ops direct (short template → direct wins).
        assert_eq!(
            recommend_corr_method(1, 1024),
            CorrMethod::Direct,
            "short template against long signal should prefer direct"
        );
    }

    // ── Forced method tests ────────────────────────────────────────────

    /// The direct method reproduces the textbook cross-correlation.
    #[test]
    fn forced_direct_known() {
        let factory = make_factory();
        let a = [1.0, 2.0, 3.0];
        let b = [0.5, 1.0];
        let plan = factory
            .create_plan(&spec_direct(a.len(), b.len()))
            .expect("plan creation should succeed");

        let mut output = vec![0.0; a.len() + b.len() - 1];
        plan.execute(&a, &b, &mut output)
            .expect("process should succeed");

        // Same expectation as `known_cross_correlation`.
        let expected = [1.0, 2.5, 4.0, 1.5];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "direct output[{i}] = {got}, expected {exp}"
            );
        }
    }

    /// The Debug representation reflects the resolved method.
    #[test]
    fn debug_reports_method() {
        let factory = make_factory();
        let direct_plan = factory
            .create_plan::<f64>(&spec_direct(4, 3))
            .expect("direct plan should succeed");
        let fft_plan = factory
            .create_plan::<f64>(&spec_fft(4, 3))
            .expect("fft plan should succeed");
        assert!(
            format!("{direct_plan:?}").contains("Direct"),
            "direct plan Debug should report the Direct method"
        );
        assert!(
            format!("{fft_plan:?}").contains("Fft"),
            "fft plan Debug should report the Fft method"
        );
    }

    // ── Direct-vs-FFT oracle ───────────────────────────────────────────

    /// Generate a deterministic, sign-varying `f64` signal of length `n`.
    ///
    /// Two incommensurate sinusoids give an aperiodic, sign-varying sequence
    /// (no integer casts) — enough to exercise the cross-correlation with
    /// non-trivial data without pulling in an RNG dependency.  Values stay in
    /// roughly `[-1.5, 1.5]`.
    fn deterministic_signal(n: usize, phase: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        let mut t = phase;
        for _ in 0..n {
            out.push(0.5f64.mul_add((0.7f64.mul_add(t, phase)).cos(), (1.3 * t).sin()));
            t += 0.37;
        }
        out
    }

    /// The direct path must agree with the FFT path in convention: same
    /// length, same lag-0 placement, same ordering.  Checked over
    /// small/medium/large sizes.
    ///
    /// The test factories implement the real-DFT primitives only for `f64`, so
    /// this `f64` oracle pins the convention here; the `f32 + f64` oracle over
    /// the real backend lives in the integration suite, and the conformance
    /// suite pins both widths to the scipy reference across every method.
    #[test]
    fn direct_matches_fft_oracle() {
        let factory = make_factory();
        // (a_len, b_len) across small, medium, and large regimes, plus
        // asymmetric and unit-template edges.
        let sizes: &[(usize, usize)] = &[
            (1, 1),
            (3, 2),
            (4, 4),
            (8, 3),
            (16, 16),
            (33, 17),
            (64, 64),
            (100, 7),
            (128, 96),
            (256, 200),
        ];

        for &(a_len, b_len) in sizes {
            let a = deterministic_signal(a_len, 0.11);
            let b = deterministic_signal(b_len, 2.4);

            let plan_d = factory
                .create_plan::<f64>(&spec_direct(a_len, b_len))
                .expect("direct plan should succeed");
            let plan_f = factory
                .create_plan::<f64>(&spec_fft(a_len, b_len))
                .expect("fft plan should succeed");

            let out_len = a_len + b_len - 1;
            let mut out_d = vec![0.0_f64; out_len];
            let mut out_f = vec![0.0_f64; out_len];
            plan_d
                .execute(&a, &b, &mut out_d)
                .expect("direct execute should succeed");
            plan_f
                .execute(&a, &b, &mut out_f)
                .expect("fft execute should succeed");

            for (i, (&d, &f)) in out_d.iter().zip(out_f.iter()).enumerate() {
                assert!(
                    (d - f).abs() <= 1e-9 * f.abs().max(1.0),
                    "f64 direct vs fft mismatch at ({a_len},{b_len}) index {i}: \
                     direct {d}, fft {f}"
                );
            }
        }
    }
}
