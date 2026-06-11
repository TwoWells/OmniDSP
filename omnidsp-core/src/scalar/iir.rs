// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! [`ScalarIir`] — scalar DF2T biquad cascade IIR filter.
//!
//! The inner loop is a sequential recurrence that cannot be vectorized, so this
//! implementation is scalar.  It is the baseline for all platforms.
//!
//! Plans are **mutable** — they hold biquad delay states that persist across
//! calls so successive `process` calls form a continuous stream.

use crate::error::{Error, Result};
use crate::traits::iir::{Iir, IirPlan, IirSpec};
use crate::types::{BiquadSection, DspFloat};

// ─── Public types ──────────────────────────────────────────────────────

/// Scalar IIR filter factory (DF2T biquad cascade).
///
/// Creates [`ScalarIirPlan`]s for biquad cascade specifications.  This is a
/// zero-sized type — all state lives in the plan.
#[derive(Debug, Clone, Copy)]
pub struct ScalarIir;

impl ScalarIir {
    /// Create a new IIR filter factory.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for ScalarIir {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution plan for a streaming IIR filter (biquad cascade).
///
/// Created by [`ScalarIir::create_plan`](Iir::create_plan).  Mutable — holds
/// biquad delay states that persist across calls.
pub struct ScalarIirPlan<T> {
    sections: Vec<BiquadSection<T>>,
    /// Delay state per section: `[s1, s2]` for DF2T.
    state: Vec<[T; 2]>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for ScalarIirPlan<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarIirPlan")
            .field("num_sections", &self.sections.len())
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

// ─── Trait implementations ────────────────────────────────────────────

impl<T: DspFloat> IirPlan<T> for ScalarIirPlan<T> {
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::BufferMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }

        // DF2T cascade: for each sample, run through all sections in order.
        for (x, out) in input.iter().zip(output.iter_mut()) {
            let mut y = *x;
            for (sec, st) in self.sections.iter().zip(self.state.iter_mut()) {
                let s1 = st[0];
                let s2 = st[1];
                let w = sec.b0 * y + s1;
                st[0] = sec.b1 * y - sec.a1 * w + s2;
                st[1] = sec.b2 * y - sec.a2 * w;
                y = w;
            }
            *out = y;
        }

        Ok(())
    }

    fn reset(&mut self) {
        for st in &mut self.state {
            *st = [T::zero(); 2];
        }
    }
}

impl<T: DspFloat> Iir<T> for ScalarIir {
    type Plan = ScalarIirPlan<T>;

    fn create_plan(&self, spec: &IirSpec<T>) -> Result<Self::Plan> {
        // The non-empty-sections invariant is enforced by `IirSpec::new`
        // (ADR-006 §4), so it is not re-checked here.
        let state = vec![[T::zero(); 2]; spec.sections().len()];

        Ok(ScalarIirPlan {
            sections: spec.sections().to_vec(),
            state,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;
    use crate::traits::iir::{Iir, IirPlan, IirSpec};

    const EPSILON: f64 = 1e-12;

    fn make_factory() -> ScalarIir {
        ScalarIir::new()
    }

    fn spec(sections: Vec<BiquadSection<f64>>) -> IirSpec<f64> {
        IirSpec::new(sections).expect("valid iir spec")
    }

    fn assert_approx_eq(actual: &[f64], expected: &[f64], eps: f64, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: slice lengths differ ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < eps,
                "{label}: mismatch at index {i}: got {a}, expected {e} (diff={})",
                (a - e).abs()
            );
        }
    }

    fn passthrough_section() -> BiquadSection<f64> {
        BiquadSection {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    fn gain_section(g: f64) -> BiquadSection<f64> {
        BiquadSection {
            b0: g,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    // ── Passthrough ──────────────────────────────────────────────────

    #[test]
    fn passthrough() {
        let factory = make_factory();
        let mut plan = Iir::<f64>::create_plan(&factory, &spec(vec![passthrough_section()]))
            .expect("plan creation");

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0; 5];
        plan.process(&input, &mut output).expect("process");

        assert_approx_eq(&output, &input, EPSILON, "passthrough");
    }

    // ── Gain ─────────────────────────────────────────────────────────

    #[test]
    fn gain() {
        let factory = make_factory();
        let mut plan = Iir::<f64>::create_plan(&factory, &spec(vec![gain_section(2.0)]))
            .expect("plan creation");

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0; 5];
        plan.process(&input, &mut output).expect("process");

        let expected: Vec<f64> = input.iter().map(|x| 2.0 * x).collect();
        assert_approx_eq(&output, &expected, EPSILON, "gain");
    }

    // ── Simple first-order ───────────────────────────────────────────

    #[test]
    fn first_order() {
        let factory = make_factory();
        // y[n] = 0.5*x[n] + 0.5*x[n-1] + 0.3*y[n-1]
        // DF2T: b0=0.5, b1=0.5, b2=0, a1=-0.3, a2=0
        let sec = BiquadSection {
            b0: 0.5,
            b1: 0.5,
            b2: 0.0,
            a1: -0.3,
            a2: 0.0,
        };
        let mut plan = Iir::<f64>::create_plan(&factory, &spec(vec![sec])).expect("plan creation");

        let input = [1.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = [0.0; 5];
        plan.process(&input, &mut output).expect("process");

        // Hand-computed DF2T:
        // n=0: y=0.5*1+0=0.5, s1=0.5*1-(-0.3)*0.5+0=0.65, s2=0
        // n=1: y=0.5*0+0.65=0.65, s1=0.5*0-(-0.3)*0.65+0=0.195, s2=0
        // n=2: y=0.5*0+0.195=0.195, s1=0-(-0.3)*0.195=0.0585, s2=0
        // n=3: y=0.5*0+0.0585=0.0585, s1=0-(-0.3)*0.0585=0.01755, s2=0
        // n=4: y=0.5*0+0.01755=0.01755, s1=(-0.3)*0.01755=0.005265, s2=0
        let expected = [0.5, 0.65, 0.195, 0.0585, 0.01755];
        assert_approx_eq(&output, &expected, EPSILON, "first-order impulse");
    }

    // ── Cascade of two sections ──────────────────────────────────────

    #[test]
    fn cascade_of_two_sections() {
        let factory = make_factory();
        let sec0 = BiquadSection {
            b0: 0.5,
            b1: 0.3,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let sec1 = BiquadSection {
            b0: 1.0,
            b1: -0.2,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };

        // Two-section cascade.
        let mut plan_cascade =
            Iir::<f64>::create_plan(&factory, &spec(vec![sec0, sec1])).expect("cascade plan");

        // Serial application: section 0 then section 1.
        let mut plan0 = Iir::<f64>::create_plan(&factory, &spec(vec![sec0])).expect("plan0");
        let mut plan1 = Iir::<f64>::create_plan(&factory, &spec(vec![sec1])).expect("plan1");

        let input = [1.0, 0.5, -0.3, 0.7, 0.0, 0.0, 0.0, 0.0];
        let mut out_cascade = [0.0; 8];
        let mut intermediate = [0.0; 8];
        let mut out_serial = [0.0; 8];

        plan_cascade
            .process(&input, &mut out_cascade)
            .expect("cascade");
        plan0
            .process(&input, &mut intermediate)
            .expect("serial stage 0");
        plan1
            .process(&intermediate, &mut out_serial)
            .expect("serial stage 1");

        assert_approx_eq(&out_cascade, &out_serial, EPSILON, "cascade vs serial");
    }

    // ── Streaming continuity ─────────────────────────────────────────

    #[test]
    fn streaming_continuity() {
        let factory = make_factory();
        let sec = BiquadSection {
            b0: 0.2,
            b1: 0.3,
            b2: 0.1,
            a1: -0.5,
            a2: -0.1,
        };

        // Reference: single-shot.
        let mut plan_ref = Iir::<f64>::create_plan(&factory, &spec(vec![sec])).expect("ref plan");
        let input: Vec<f64> = (0..20).map(|i| (f64::from(i) * 0.3).sin()).collect();
        let mut out_ref = vec![0.0; 20];
        plan_ref.process(&input, &mut out_ref).expect("ref process");

        // Streaming: two chunks.
        let mut plan_stream =
            Iir::<f64>::create_plan(&factory, &spec(vec![sec])).expect("stream plan");
        let split = 7;
        let mut out_stream = vec![0.0; 20];
        plan_stream
            .process(&input[..split], &mut out_stream[..split])
            .expect("chunk 1");
        plan_stream
            .process(&input[split..], &mut out_stream[split..])
            .expect("chunk 2");

        assert_approx_eq(&out_stream, &out_ref, EPSILON, "streaming continuity");
    }

    // ── Reset ────────────────────────────────────────────────────────

    #[test]
    fn reset() {
        let factory = make_factory();
        let sec = BiquadSection {
            b0: 0.2,
            b1: 0.3,
            b2: 0.1,
            a1: -0.5,
            a2: -0.1,
        };
        let mut plan = Iir::<f64>::create_plan(&factory, &spec(vec![sec])).expect("plan creation");

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output1 = [0.0; 5];
        let mut output2 = [0.0; 5];

        plan.process(&input, &mut output1).expect("first run");
        plan.reset();
        plan.process(&input, &mut output2).expect("second run");

        assert_approx_eq(&output2, &output1, EPSILON, "reset");
    }

    // ── Plan reuse ───────────────────────────────────────────────────

    #[test]
    fn plan_reuse() {
        let factory = make_factory();
        let mut plan = Iir::<f64>::create_plan(&factory, &spec(vec![passthrough_section()]))
            .expect("plan creation");

        let input1 = [1.0, 2.0, 3.0];
        let mut output1 = [0.0; 3];
        plan.process(&input1, &mut output1).expect("call 1");

        let input2 = [4.0, 5.0, 6.0];
        let mut output2 = [0.0; 3];
        plan.process(&input2, &mut output2).expect("call 2");

        assert_approx_eq(&output1, &[1.0, 2.0, 3.0], EPSILON, "plan reuse call 1");
        assert_approx_eq(&output2, &[4.0, 5.0, 6.0], EPSILON, "plan reuse call 2");
    }

    // ── DC convergence ───────────────────────────────────────────────

    #[test]
    fn lowpass_dc_converges() {
        use crate::design::iir::{FilterFamily, design};
        use crate::types::FilterType;

        let factory = make_factory();
        let iir_spec = design::<f64>(
            FilterFamily::Butterworth,
            FilterType::Lowpass,
            4,
            44100.0,
            1000.0,
            None,
        )
        .expect("design LP4");

        let mut plan = Iir::<f64>::create_plan(&factory, &iir_spec).expect("plan creation");

        // Feed DC=1.0 for a long time — output should converge to 1.0.
        let input = vec![1.0; 4096];
        let mut output = vec![0.0; 4096];
        plan.process(&input, &mut output).expect("process");

        // Check last samples are close to 1.0 (DC gain = 1).
        let tail = &output[3900..];
        for (i, &v) in tail.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "lowpass DC convergence: sample {}: got {v}, expected ~1.0",
                3900 + i
            );
        }
    }

    #[test]
    fn highpass_dc_decays() {
        use crate::design::iir::{FilterFamily, design};
        use crate::types::FilterType;

        let factory = make_factory();
        let iir_spec = design::<f64>(
            FilterFamily::Butterworth,
            FilterType::Highpass,
            4,
            44100.0,
            5000.0,
            None,
        )
        .expect("design HP4");

        let mut plan = Iir::<f64>::create_plan(&factory, &iir_spec).expect("plan creation");

        let input = vec![1.0; 4096];
        let mut output = vec![0.0; 4096];
        plan.process(&input, &mut output).expect("process");

        // DC through a highpass should decay to zero.
        let tail = &output[3900..];
        for (i, &v) in tail.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "highpass DC decay: sample {}: got {v}, expected ~0.0",
                3900 + i
            );
        }
    }

    // ── Validation ───────────────────────────────────────────────────

    #[test]
    fn empty_sections_returns_error() {
        assert!(
            IirSpec::<f64>::new(vec![]).is_err(),
            "empty sections should be rejected by the spec constructor"
        );
    }

    #[test]
    fn buffer_length_mismatch_returns_error() {
        let factory = make_factory();
        let mut plan = Iir::<f64>::create_plan(&factory, &spec(vec![passthrough_section()]))
            .expect("plan creation");

        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 2];
        assert!(
            plan.process(&input, &mut output).is_err(),
            "mismatched buffer lengths should return error"
        );
    }

    // ── Scipy sosfilt reference tests ────────────────────────────────
    //
    // Generated by scripts/gen_iir_sosfilt_reference.py.
    // Regenerate with: make gen-iir-sosfilt-reference

    #[allow(
        clippy::wildcard_imports,
        reason = "bulk golden-vector import in tests"
    )]
    use omnidsp_testdata::iir_sosfilt_scipy::*;

    /// Build `Vec<BiquadSection<f64>>` from scipy's `(b0, b1, b2, a1, a2)` tuples.
    fn scipy_to_sections(data: &[(f64, f64, f64, f64, f64)]) -> Vec<BiquadSection<f64>> {
        data.iter()
            .map(|&(b0, b1, b2, a1, a2)| BiquadSection { b0, b1, b2, a1, a2 })
            .collect()
    }

    fn test_scipy_sosfilt(
        sos: &[(f64, f64, f64, f64, f64)],
        input: &[f64],
        expected: &[f64],
        tol: f64,
        label: &str,
    ) {
        let factory = make_factory();
        let mut plan = Iir::<f64>::create_plan(&factory, &spec(scipy_to_sections(sos)))
            .expect("plan creation");

        let mut output = vec![0.0; input.len()];
        plan.process(input, &mut output).expect("process");

        assert_approx_eq(&output, expected, tol, label);
    }

    #[test]
    fn scipy_sosfilt_lowpass_order4() {
        test_scipy_sosfilt(
            SOSFILT_LP4_SOS,
            SOSFILT_INPUT,
            SOSFILT_LP4_OUTPUT,
            1e-10,
            "sosfilt LP4",
        );
    }

    #[test]
    fn scipy_sosfilt_highpass_order4() {
        test_scipy_sosfilt(
            SOSFILT_HP4_SOS,
            SOSFILT_INPUT,
            SOSFILT_HP4_OUTPUT,
            1e-10,
            "sosfilt HP4",
        );
    }

    #[test]
    fn scipy_sosfilt_bandpass_order4() {
        test_scipy_sosfilt(
            SOSFILT_BP4_SOS,
            SOSFILT_INPUT,
            SOSFILT_BP4_OUTPUT,
            1e-10,
            "sosfilt BP4",
        );
    }

    #[test]
    fn scipy_sosfilt_streaming() {
        // Process the long signal in chunks, compare against single-shot scipy.
        let factory = make_factory();
        let mut plan = Iir::<f64>::create_plan(&factory, &spec(scipy_to_sections(SOSFILT_LP4_SOS)))
            .expect("plan creation");

        let mut output = vec![0.0; SOSFILT_LONG_INPUT.len()];

        // Process in varied-size chunks to stress streaming.
        let chunks = [37, 100, 63, 200, 11, 256, 357];
        let mut pos = 0;
        for &chunk in &chunks {
            let end = (pos + chunk).min(output.len());
            if pos >= end {
                break;
            }
            plan.process(&SOSFILT_LONG_INPUT[pos..end], &mut output[pos..end])
                .expect("chunk process");
            pos = end;
        }

        assert_approx_eq(&output, SOSFILT_LONG_OUTPUT, 1e-10, "streaming sosfilt");
    }
}
