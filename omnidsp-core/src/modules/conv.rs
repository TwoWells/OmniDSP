// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Convolution module — direct and FFT-based fast convolution.
//!
//! [`OmniConv`] builds convolution plans via an inherent `create_plan`,
//! generic over any [`DftR2c`] / [`DftC2r`] and [`VecOps`] implementation.
//! Supports both time-domain (direct) and frequency-domain (FFT) convolution,
//! selected via [`ConvMethod`].
//!
//! The FFT path transforms the real inputs through the real-DFT primitives
//! ([`DftR2c`] forward, [`DftC2r`] inverse) rather than embedding them in a
//! complex transform, so it pays no ~2× complex-embedding tax.
//! The inverse c2r factory is wrapped in [`HermitianC2r`] so the DC/Nyquist
//! boundary is projected onto the nearest valid Hermitian spectrum before the
//! transform — the round-trip drift every `r2c → ⊙ → c2r`
//! chain accumulates cannot reach the kernel.
//!
//! The [`recommend_method`] function provides an operation-count heuristic
//! for the `Auto` case.
//!
//! Internal scratch buffers on the FFT path are behind a [`Mutex`] so that
//! the plan satisfies `Send + Sync` while taking `&self`.  The lock is
//! uncontended in the common single-threaded case.

use std::fmt;
use std::ops::{AddAssign, MulAssign};
use std::sync::Mutex;

use num_complex::Complex;
use num_traits::Float;

use crate::error::{Error, Result};
use crate::hermitian::{HermitianC2r, HermitianC2rPlan};
use crate::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
use crate::traits::dft::{DftC2r, DftC2rPlan, DftC2rSpec, DftNorm, DftR2c, DftR2cPlan, DftR2cSpec};
use crate::traits::vecops::VecOps;
use crate::types::DspFloat;

// ─── Public heuristic ─────────────────────────────────────────────────

/// Recommend a convolution method based on operation counts.
///
/// Compares the work of direct convolution (`a_len * b_len` ops) against
/// FFT convolution (`~3 * N * log2(N) + N` ops where `N` is the FFT
/// length).  Returns whichever is cheaper.
///
/// This is a generic heuristic that ignores constant factors (cache
/// effects, SIMD width, FFT implementation quality).  Vendor-specific
/// `Conv` implementations may use a different decision boundary.
#[must_use]
pub fn recommend_method(a_len: usize, b_len: usize) -> ConvMethod {
    if a_len == 0 || b_len == 0 {
        return ConvMethod::Direct;
    }

    let direct_ops = a_len as u64 * b_len as u64;

    let output_len = a_len + b_len - 1;
    let fft_len = output_len.next_power_of_two() as u64;
    // 3 FFTs (2 forward + 1 inverse) each ~N*log2(N), plus N for the
    // element-wise complex multiply.
    let log2_n = u64::BITS - fft_len.leading_zeros();
    let fft_ops = 3 * fft_len * u64::from(log2_n) + fft_len;

    if direct_ops <= fft_ops {
        ConvMethod::Direct
    } else {
        ConvMethod::Fft
    }
}

// ─── Public types ──────────────────────────────────────────────────────

/// Generic convolution factory backed by [`DftR2c`] / [`DftC2r`] and
/// [`VecOps`].
///
/// Creates [`OmniConvPlan`]s for specific input lengths.  The factory owns the
/// real-DFT factories (`r2c` forward, `c2r` inverse) and the `VecOps` instance;
/// plans own their sub-plans.  The c2r factory is Hermitian-shaped internally,
/// so any backend that reuses `OmniConv` inherits the projection.
#[derive(Debug, Clone)]
pub struct OmniConv<R, C, V> {
    r2c: R,
    c2r: C,
    vecops: V,
}

impl<R, C, V> OmniConv<R, C, V> {
    /// Create a new convolution factory.
    #[must_use]
    pub const fn new(r2c: R, c2r: C, vecops: V) -> Self {
        Self { r2c, c2r, vecops }
    }
}

/// Execution plan for a convolution operation.
///
/// Created by [`OmniConv::create_plan`].  Immutable and thread-safe
/// (`Send + Sync`).
///
/// Output length is `a_len + b_len - 1` (full convolution).
///
/// **Memory:** the direct path allocates nothing beyond the plan struct.
/// The FFT path allocates one real buffer of `fft_length` plus two complex
/// half-spectrum buffers of `fft_length / 2 + 1` (behind a [`Mutex`] for
/// thread safety) — roughly half the complex-DFT path's footprint.
///
/// `RP` is the forward [`DftR2cPlan`]; `CP` is the inverse [`DftC2rPlan`]
/// (in practice a [`HermitianC2rPlan`]).  Neither is ever named by users.
pub struct OmniConvPlan<T, RP, CP, V> {
    inner: PlanInner<T, RP, CP, V>,
    a_len: usize,
    b_len: usize,
    output_len: usize,
}

impl<T, RP, CP, V> fmt::Debug for OmniConvPlan<T, RP, CP, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let method = match &self.inner {
            PlanInner::Direct => "Direct",
            PlanInner::Fft { .. } => "Fft",
        };
        f.debug_struct("OmniConvPlan")
            .field("method", &method)
            .field("a_len", &self.a_len)
            .field("b_len", &self.b_len)
            .field("output_len", &self.output_len)
            .finish_non_exhaustive()
    }
}

// ─── Plan internals ───────────────────────────────────────────────────

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
    scratch: Mutex<FftScratch<T>>,
}

/// Scratch buffers for the FFT convolution pipeline.
struct FftScratch<T> {
    /// Real pad buffer (`fft_len`): holds each zero-padded input before the
    /// forward r2c consumes it, then receives the c2r real output.
    real: Vec<T>,
    /// Half-spectrum of the first input → holds the frequency-domain product.
    half_a: Vec<Complex<T>>,
    /// Half-spectrum of the second input.
    half_b: Vec<Complex<T>>,
}

// ─── Trait implementations ─────────────────────────────────────────────

impl<T, RP, CP, V> ConvPlan<T> for OmniConvPlan<T, RP, CP, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    RP: DftR2cPlan<T>,
    CP: DftC2rPlan<T>,
    V: VecOps<T>,
{
    fn execute(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
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
}

impl<T, RP, CP, V> OmniConvPlan<T, RP, CP, V>
where
    T: Float + AddAssign + MulAssign + Send + Sync,
    RP: DftR2cPlan<T>,
    CP: DftC2rPlan<T>,
    V: VecOps<T>,
{
    /// Direct (time-domain) convolution: `O(a_len * b_len)`.
    fn process_direct(a: &[T], b: &[T], output: &mut [T]) {
        for o in output.iter_mut() {
            *o = T::zero();
        }
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                output[i + j] += ai * bj;
            }
        }
    }

    /// FFT-based (frequency-domain) convolution over the real-DFT primitives.
    ///
    /// `conv(a,b) = c2r(r2c(a) ⊙ r2c(b))`: the product of two Hermitian
    /// half-spectra is the half-spectrum of the (real) convolution, so the
    /// element-wise multiply runs on the `fft_len / 2 + 1` bins directly.
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

        let FftScratch {
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

        // 3. Half-spectrum element-wise multiply: half_a *= half_b.
        state.vecops.cmul_inplace(half_a, half_b)?;

        // 4. Inverse c2r: half_a → real (HermitianC2r projects DC/Nyquist).
        state.inv.execute(half_a, real)?;

        // 5. The c2r output is already real — copy the first output_len samples.
        let out_len = output.len();
        output.copy_from_slice(&real[..out_len]);

        Ok(())
    }
}

impl<R, C, V> OmniConv<R, C, V> {
    /// Create a plan for a convolution described by `spec`.
    ///
    /// The plan preallocates any internal buffers so that execution is
    /// allocation-free.  The FFT path wraps the inverse c2r factory in
    /// [`HermitianC2r`] so the round-trip DC/Nyquist drift is projected away
    /// before the inverse transform.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpec`] if the FFT length overflows.  The input
    /// length invariants are enforced by [`ConvSpec::new`], so
    /// they are not re-checked here.
    #[allow(
        clippy::type_complexity,
        reason = "composite real-DFT plan type (r2c forward + Hermitian-shaped c2r \
                  inverse); the dispatch layer names it via `type Plan` aliases"
    )]
    pub fn create_plan<T>(
        &self,
        spec: &ConvSpec,
    ) -> Result<OmniConvPlan<T, <R as DftR2c<T>>::Plan, HermitianC2rPlan<<C as DftC2r<T>>::Plan>, V>>
    where
        T: DspFloat + AddAssign + MulAssign,
        R: DftR2c<T>,
        C: DftC2r<T> + Clone,
        V: VecOps<T>,
    {
        let output_len = spec.a_len() + spec.b_len() - 1;

        let method = match spec.method() {
            ConvMethod::Auto => recommend_method(spec.a_len(), spec.b_len()),
            other => other,
        };

        let inner = match method {
            ConvMethod::Direct | ConvMethod::Auto => PlanInner::Direct,
            ConvMethod::Fft => {
                let fft_len = output_len.checked_next_power_of_two().ok_or_else(|| {
                    Error::InvalidSpec("convolution FFT length overflow".to_owned())
                })?;
                let bins = fft_len / 2 + 1;

                // Convolution theorem: conv(a,b) = c2r(r2c(a) · r2c(b)).
                // This requires c2r(r2c(x)) = x, which DftNorm::Inverse provides
                // (1/N scaling on the inverse transform only).  The forward r2c
                // plan is reused for both inputs.
                let fwd_spec = DftR2cSpec::new(fft_len, DftNorm::Inverse)?;
                let fwd = self.r2c.create_plan(&fwd_spec)?;

                // The c2r factory is Hermitian-shaped: the
                // product of two Hermitian half-spectra is Hermitian, but float
                // drift leaves ~epsilon imaginary at DC/Nyquist that the
                // projection clears before the inverse transform.
                let inv_spec = DftC2rSpec::new(fft_len, DftNorm::Inverse)?;
                let inv = HermitianC2r::new(self.c2r.clone()).create_plan(&inv_spec)?;

                let zero = Complex::new(T::zero(), T::zero());
                let scratch = FftScratch {
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

        Ok(OmniConvPlan {
            inner,
            a_len: spec.a_len(),
            b_len: spec.b_len(),
            output_len,
        })
    }
}

/// Zero-pad a real slice into a pre-allocated real buffer.
///
/// The first `real.len()` elements are copied; the remainder is zeroed.
fn pad_real<T: Float>(input: &[T], buf: &mut [T]) {
    let n = input.len();
    buf[..n].copy_from_slice(input);
    for x in &mut buf[n..] {
        *x = T::zero();
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;
    use crate::test_utils::{TestDftC2r, TestDftR2c, TestVecOps};

    const EPSILON: f64 = 1e-8;

    fn spec(a_len: usize, b_len: usize) -> ConvSpec {
        ConvSpec::new(a_len, b_len, ConvMethod::Auto).expect("valid conv spec")
    }

    fn spec_direct(a_len: usize, b_len: usize) -> ConvSpec {
        ConvSpec::new(a_len, b_len, ConvMethod::Direct).expect("valid conv spec")
    }

    fn spec_fft(a_len: usize, b_len: usize) -> ConvSpec {
        ConvSpec::new(a_len, b_len, ConvMethod::Fft).expect("valid conv spec")
    }

    fn assert_approx_eq(actual: &[f64], expected: &[f64], eps: f64) {
        assert_eq!(actual.len(), expected.len(), "slice lengths differ");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < eps,
                "mismatch at index {i}: got {a}, expected {e}"
            );
        }
    }

    fn make_factory() -> OmniConv<TestDftR2c, TestDftC2r, TestVecOps> {
        OmniConv::new(TestDftR2c, TestDftC2r, TestVecOps)
    }

    // ── Heuristic tests ───────────────────────────────────────────────

    #[test]
    fn recommend_small_prefers_direct() {
        // 4 * 4 = 16 ops direct, FFT would need N=8 → 3*8*3 + 8 = 80
        assert_eq!(
            recommend_method(4, 4),
            ConvMethod::Direct,
            "small inputs should prefer direct"
        );
    }

    #[test]
    fn recommend_large_prefers_fft() {
        // 1024 * 1024 = 1M ops direct, FFT N=2048 → 3*2048*11 + 2048 ≈ 69K
        assert_eq!(
            recommend_method(1024, 1024),
            ConvMethod::Fft,
            "large inputs should prefer FFT"
        );
    }

    #[test]
    fn recommend_asymmetric_short_kernel() {
        // 1 * 1024 = 1024 ops direct (short kernel → direct wins)
        assert_eq!(
            recommend_method(1, 1024),
            ConvMethod::Direct,
            "short kernel against long signal should prefer direct"
        );
    }

    // ── Correctness tests (Auto) ──────────────────────────────────────

    /// `[1, 2, 3] * [4, 5] = [4, 13, 22, 15]`
    #[test]
    fn basic_convolution() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(3, 2))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 4];
        plan.execute(&[1.0, 2.0, 3.0], &[4.0, 5.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[4.0, 13.0, 22.0, 15.0], EPSILON);
    }

    /// Convolving with `[1]` is an identity operation.
    #[test]
    fn impulse_identity() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(1, 3))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 3];
        plan.execute(&[1.0], &[2.0, 3.0, 4.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[2.0, 3.0, 4.0], EPSILON);
    }

    /// `[1, 1, 1] * [1, 1, 1] = [1, 2, 3, 2, 1]`
    #[test]
    fn symmetric_inputs() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(3, 3))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 5];
        plan.execute(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[1.0, 2.0, 3.0, 2.0, 1.0], EPSILON);
    }

    /// `[3] * [7] = [21]`
    #[test]
    fn single_element() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(1, 1))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 1];
        plan.execute(&[3.0], &[7.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[21.0], EPSILON);
    }

    /// A delayed impulse shifts the signal.
    #[test]
    fn delayed_impulse() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(3, 3))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 5];
        plan.execute(&[0.0, 1.0, 0.0], &[2.0, 3.0, 4.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[0.0, 2.0, 3.0, 4.0, 0.0], EPSILON);
    }

    /// Convolution is commutative: `a * b == b * a`.
    #[test]
    fn commutative() {
        let factory = make_factory();
        let plan_ab = factory
            .create_plan(&spec(3, 2))
            .expect("plan ab should succeed");
        let plan_ba = factory
            .create_plan(&spec(2, 3))
            .expect("plan ba should succeed");

        let mut out_ab = [0.0_f64; 4];
        let mut out_ba = [0.0_f64; 4];

        plan_ab
            .execute(&[1.0, 2.0, 3.0], &[4.0, 5.0], &mut out_ab)
            .expect("ab should succeed");
        plan_ba
            .execute(&[4.0, 5.0], &[1.0, 2.0, 3.0], &mut out_ba)
            .expect("ba should succeed");

        assert_approx_eq(&out_ab, &out_ba, EPSILON);
    }

    /// A plan can be reused for multiple process calls.
    #[test]
    fn plan_reuse() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(2, 2))
            .expect("plan creation should succeed");

        let mut out1 = [0.0_f64; 3];
        let mut out2 = [0.0_f64; 3];

        plan.execute(&[1.0, 2.0], &[3.0, 4.0], &mut out1)
            .expect("first call should succeed");
        plan.execute(&[5.0, 6.0], &[7.0, 8.0], &mut out2)
            .expect("second call should succeed");

        // [1,2]*[3,4] = [3, 10, 8]
        assert_approx_eq(&out1, &[3.0, 10.0, 8.0], EPSILON);
        // [5,6]*[7,8] = [35, 82, 48]
        assert_approx_eq(&out2, &[35.0, 82.0, 48.0], EPSILON);
    }

    // ── Forced method tests ───────────────────────────────────────────

    /// Direct method gives correct result for larger inputs.
    #[test]
    fn forced_direct() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec_direct(3, 2))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 4];
        plan.execute(&[1.0, 2.0, 3.0], &[4.0, 5.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[4.0, 13.0, 22.0, 15.0], EPSILON);
    }

    /// FFT method gives correct result for small inputs.
    #[test]
    fn forced_fft() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec_fft(3, 2))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 4];
        plan.execute(&[1.0, 2.0, 3.0], &[4.0, 5.0], &mut output)
            .expect("process should succeed");
        assert_approx_eq(&output, &[4.0, 13.0, 22.0, 15.0], EPSILON);
    }

    /// Both methods agree on the same input.
    #[test]
    fn direct_and_fft_agree() {
        let factory = make_factory();
        let plan_d = factory
            .create_plan(&spec_direct(4, 3))
            .expect("direct plan should succeed");
        let plan_f = factory
            .create_plan(&spec_fft(4, 3))
            .expect("fft plan should succeed");

        let a = [1.0, -1.0, 2.0, 0.5];
        let b = [3.0, 0.0, -2.0];
        let mut out_d = [0.0_f64; 6];
        let mut out_f = [0.0_f64; 6];

        plan_d
            .execute(&a, &b, &mut out_d)
            .expect("direct should succeed");
        plan_f
            .execute(&a, &b, &mut out_f)
            .expect("fft should succeed");

        assert_approx_eq(&out_d, &out_f, EPSILON);
    }

    // ── Validation tests ───────────────────────────────────────────────

    #[test]
    fn zero_a_len_returns_error() {
        assert!(
            ConvSpec::new(0, 3, ConvMethod::Auto).is_err(),
            "zero a_len should be rejected by the spec constructor"
        );
    }

    #[test]
    fn zero_b_len_returns_error() {
        assert!(
            ConvSpec::new(3, 0, ConvMethod::Auto).is_err(),
            "zero b_len should be rejected by the spec constructor"
        );
    }

    #[test]
    fn wrong_a_length_returns_error() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(3, 2))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 4];
        assert!(
            plan.execute(&[1.0, 2.0], &[4.0, 5.0], &mut output).is_err(),
            "wrong a length should return error"
        );
    }

    #[test]
    fn wrong_b_length_returns_error() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(3, 2))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 4];
        assert!(
            plan.execute(&[1.0, 2.0, 3.0], &[4.0], &mut output).is_err(),
            "wrong b length should return error"
        );
    }

    #[test]
    fn wrong_output_length_returns_error() {
        let factory = make_factory();
        let plan = factory
            .create_plan(&spec(3, 2))
            .expect("plan creation should succeed");
        let mut output = [0.0_f64; 3];
        assert!(
            plan.execute(&[1.0, 2.0, 3.0], &[4.0, 5.0], &mut output)
                .is_err(),
            "wrong output length should return error"
        );
    }
}
