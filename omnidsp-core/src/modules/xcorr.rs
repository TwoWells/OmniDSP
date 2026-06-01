// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Cross-correlation module — FFT-based fast cross-correlation.
//!
//! [`OmniCrossCorr`] computes the full linear cross-correlation of two
//! real-valued signals using the frequency-domain method:
//! `xcorr(a, b) = IFFT(FFT(a) · conj(FFT(b)))`.
//!
//! Generic over any [`Dft`] and [`VecOps`] implementation.  Internal scratch
//! buffers are behind a [`Mutex`] so that the plan satisfies `Send + Sync`
//! while taking `&self`.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign};
use std::sync::Mutex;

use num_complex::Complex;
use num_traits::Float;

use crate::error::{Error, Result};
use crate::traits::dft::{Dft, DftNorm, DftPlan, DftSpec};
use crate::traits::vecops::VecOps;
use crate::types::Direction;

// ─── Spec ────────────────────────────────────────────────────────────────

/// Cross-correlation operation specification.
///
/// Describes the input signal lengths.  The output length is always
/// `a_len + b_len - 1` (full linear cross-correlation).
///
/// The type parameter `T` ties the spec to a specific float type, making
/// specs fully self-describing for the dispatch layer's `CreatePlan<S>` trait.
///
/// # Examples
///
/// ```
/// use omnidsp_core::modules::xcorr::CrossCorrSpec;
///
/// let spec = CrossCorrSpec::<f64>::new(1024, 256);
/// assert_eq!(spec.a_len, 1024);
/// assert_eq!(spec.b_len, 256);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CrossCorrSpec<T> {
    /// Length of the first input signal.
    pub a_len: usize,
    /// Length of the second input signal (reference/template).
    pub b_len: usize,
    _marker: PhantomData<T>,
}

impl<T> PartialEq for CrossCorrSpec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.a_len == other.a_len && self.b_len == other.b_len
    }
}

impl<T> Eq for CrossCorrSpec<T> {}

impl<T> CrossCorrSpec<T> {
    /// Create a new cross-correlation spec with the given input lengths.
    #[must_use]
    pub const fn new(a_len: usize, b_len: usize) -> Self {
        Self {
            a_len,
            b_len,
            _marker: PhantomData,
        }
    }

    /// Output length: `a_len + b_len - 1`.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.a_len + self.b_len - 1
    }
}

// ─── Factory ─────────────────────────────────────────────────────────────

/// Generic cross-correlation factory backed by [`Dft`] and [`VecOps`].
///
/// Creates [`OmniCrossCorrPlan`]s for specific input lengths.  The factory
/// owns the DFT factory and `VecOps` instance; plans own their sub-plans.
#[derive(Debug, Clone)]
pub struct OmniCrossCorr<D, V> {
    dft: D,
    vecops: V,
}

impl<D, V> OmniCrossCorr<D, V> {
    /// Create a new cross-correlation factory.
    #[must_use]
    pub const fn new(dft: D, vecops: V) -> Self {
        Self { dft, vecops }
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
/// **Memory:** allocates 3 × `fft_length` complex values for scratch
/// buffers (behind a [`Mutex`] for thread safety).
pub struct OmniCrossCorrPlan<T, P, V> {
    fwd: P,
    inv: P,
    vecops: V,
    scratch: Mutex<Scratch<T>>,
    a_len: usize,
    b_len: usize,
    output_len: usize,
}

impl<T, P, V> fmt::Debug for OmniCrossCorrPlan<T, P, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OmniCrossCorrPlan")
            .field("a_len", &self.a_len)
            .field("b_len", &self.b_len)
            .field("output_len", &self.output_len)
            .finish_non_exhaustive()
    }
}

/// Scratch buffers for the FFT cross-correlation pipeline.
#[allow(
    clippy::struct_field_names,
    reason = "buf_ prefix clarifies these are reusable buffers"
)]
struct Scratch<T> {
    /// Padding / FFT input / IFFT output.
    buf_a: Vec<Complex<T>>,
    /// FFT output for first input → holds frequency-domain product.
    buf_b: Vec<Complex<T>>,
    /// FFT output for second input (conjugated in-place).
    buf_c: Vec<Complex<T>>,
}

/// Zero-pad a real slice into a pre-allocated complex buffer via `VecOps`.
///
/// First `real.len()` elements are converted via [`VecOps::real_to_complex`];
/// the remainder is zeroed.
fn pad_real_to_complex<T: Float + AddAssign + MulAssign, V: VecOps<T>>(
    vecops: &V,
    real: &[T],
    buf: &mut [Complex<T>],
) -> Result<()> {
    let n = real.len();
    vecops.real_to_complex(real, &mut buf[..n])?;
    for c in &mut buf[n..] {
        *c = Complex::new(T::zero(), T::zero());
    }
    Ok(())
}

// ─── Plan implementation ─────────────────────────────────────────────────

impl<T, P, V> OmniCrossCorrPlan<T, P, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    P: DftPlan<T>,
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
            buf_a,
            buf_b,
            buf_c,
        } = &mut *guard;

        // 1. Zero-pad a → buf_a, then forward FFT → buf_b.
        pad_real_to_complex(&self.vecops, a, buf_a)?;
        self.fwd.process(buf_a, buf_b)?;

        // 2. Zero-pad b → buf_a (reuse), then forward FFT → buf_c.
        pad_real_to_complex(&self.vecops, b, buf_a)?;
        self.fwd.process(buf_a, buf_c)?;

        // 3. Conjugate buf_c in-place: conj(FFT(b)).
        self.vecops.conj_inplace(buf_c);

        // 4. Complex element-wise multiply: buf_b = FFT(a) · conj(FFT(b)).
        self.vecops.cmul_inplace(buf_b, buf_c)?;

        // 5. Inverse FFT: buf_b → buf_a.
        self.inv.process(buf_b, buf_a)?;

        // 6. Extract real parts in full cross-correlation order.
        //    The IFFT produces circular correlation in wrap-around order:
        //      indices 0..a_len → non-negative lags (lag 0 to lag a_len-1)
        //      indices fft_len-(b_len-1)..fft_len → negative lags
        //    Full output order: negative lags first, then non-negative.
        let neg_lags = self.b_len - 1;
        let fft_len = buf_a.len();
        self.vecops
            .extract_real(&buf_a[fft_len - neg_lags..fft_len], &mut output[..neg_lags])?;
        self.vecops
            .extract_real(&buf_a[..self.a_len], &mut output[neg_lags..])?;

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

// ─── Factory implementation ──────────────────────────────────────────────

impl<D, V> OmniCrossCorr<D, V> {
    /// Create a cross-correlation plan from the given specification.
    ///
    /// The plan preallocates FFT sub-plans and scratch buffers so that
    /// execution is allocation-free.
    ///
    /// # Errors
    ///
    /// Returns an error if either length is zero, the FFT length overflows,
    /// or DFT plan creation fails.
    pub fn create_plan<T>(
        &self,
        spec: &CrossCorrSpec<T>,
    ) -> Result<OmniCrossCorrPlan<T, D::Plan, V>>
    where
        T: Float + AddAssign + MulAssign + Send + Sync,
        D: Dft<T>,
        V: VecOps<T> + Clone,
    {
        if spec.a_len == 0 {
            return Err(Error::InvalidSpec(
                "cross-correlation input a_len must be non-zero".to_owned(),
            ));
        }
        if spec.b_len == 0 {
            return Err(Error::InvalidSpec(
                "cross-correlation input b_len must be non-zero".to_owned(),
            ));
        }

        let output_len = spec.output_len();
        let fft_len = output_len.checked_next_power_of_two().ok_or_else(|| {
            Error::InvalidSpec("cross-correlation FFT length overflow".to_owned())
        })?;

        // Cross-correlation theorem: xcorr(a,b) = IFFT(FFT(a) · conj(FFT(b))).
        // DftNorm::Inverse gives IFFT(FFT(x)) = x (1/N on inverse only).
        let fwd_spec = DftSpec::new(fft_len, Direction::Forward, DftNorm::Inverse);
        let inv_spec = DftSpec::new(fft_len, Direction::Inverse, DftNorm::Inverse);
        let fwd = self.dft.create_plan(&fwd_spec)?;
        let inv = self.dft.create_plan(&inv_spec)?;

        let zero = Complex::new(T::zero(), T::zero());
        let scratch = Scratch {
            buf_a: vec![zero; fft_len],
            buf_b: vec![zero; fft_len],
            buf_c: vec![zero; fft_len],
        };

        Ok(OmniCrossCorrPlan {
            fwd,
            inv,
            vecops: self.vecops.clone(),
            scratch: Mutex::new(scratch),
            a_len: spec.a_len,
            b_len: spec.b_len,
            output_len,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::*;
    use crate::test_utils::{TestDft, TestVecOps};

    fn make_factory() -> OmniCrossCorr<TestDft, TestVecOps> {
        OmniCrossCorr::new(TestDft, TestVecOps)
    }

    #[test]
    fn spec_output_len() {
        let spec = CrossCorrSpec::<f64>::new(5, 3);
        assert_eq!(
            spec.output_len(),
            7,
            "output_len should be a_len + b_len - 1"
        );
    }

    #[test]
    fn spec_equality() {
        let a = CrossCorrSpec::<f64>::new(10, 5);
        let b = CrossCorrSpec::<f64>::new(10, 5);
        let c = CrossCorrSpec::<f64>::new(10, 6);
        assert_eq!(a, b, "equal specs should compare equal");
        assert_ne!(a, c, "different specs should not compare equal");
    }

    #[test]
    fn zero_a_len_rejected() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(0, 5);
        let result = factory.create_plan(&spec);
        assert!(result.is_err(), "zero a_len should be rejected");
    }

    #[test]
    fn zero_b_len_rejected() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(5, 0);
        let result = factory.create_plan(&spec);
        assert!(result.is_err(), "zero b_len should be rejected");
    }

    #[test]
    fn buffer_mismatch_a() {
        let factory = make_factory();
        let spec = CrossCorrSpec::<f64>::new(4, 3);
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
        let spec = CrossCorrSpec::<f64>::new(4, 3);
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
        let spec = CrossCorrSpec::<f64>::new(4, 3);
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
        let spec = CrossCorrSpec::<f64>::new(a.len(), a.len());
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
        let spec = CrossCorrSpec::<f64>::new(a.len(), b.len());
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
        let spec = CrossCorrSpec::<f64>::new(a.len(), b.len());
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

        let spec_ab = CrossCorrSpec::<f64>::new(a.len(), b.len());
        let plan_ab = factory
            .create_plan(&spec_ab)
            .expect("plan creation should succeed");
        let mut out_ab = vec![0.0; spec_ab.output_len()];
        plan_ab
            .process(&a, &b, &mut out_ab)
            .expect("process should succeed");

        let spec_ba = CrossCorrSpec::<f64>::new(b.len(), a.len());
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
        let spec = CrossCorrSpec::<f64>::new(a.len(), b.len());
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
        let spec = CrossCorrSpec::<f64>::new(8, 4);
        let plan = factory
            .create_plan(&spec)
            .expect("plan creation should succeed");
        assert_eq!(plan.a_len(), 8, "a_len accessor should match");
        assert_eq!(plan.b_len(), 4, "b_len accessor should match");
        assert_eq!(plan.output_len(), 11, "output_len accessor should match");
    }
}
