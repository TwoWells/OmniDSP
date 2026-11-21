// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Native Intel IPP biquad-cascade IIR override.
//!
//! [`IppBackend`] lists `iir` in its `skip` set and hand-writes
//! [`CreateProc<IirSpec>`](omnidsp_core::create::CreateProc) here, so the IIR
//! primitive runs IPP's `ippsIIR_BiQuad` cascade instead of the scalar floor.
//! IIR is a per-sample feedback recurrence that no DFT/`VecOps` primitive
//! accelerates, so this is a genuine native kernel, not composition over the
//! backend's DFT — exactly the case the four-tier model carves out as an
//! omni-primitive override.
//!
//! The processor is **stateful**: IPP advances the cascade's delay line inside
//! its state buffer on every [`process`](IirProcessor::process), so successive
//! calls stream continuously.  [`reset`](IirProcessor::reset) zeros that delay
//! line, and [`reconfigure`](Reconfigure::reconfigure) retunes the section
//! coefficients in place while preserving it.
//!
//! # Coefficient layout
//!
//! IPP expects a flat taps array of `{b0, b1, b2, a0, a1, a2}` per section, with
//! `a0` normalized to 1; [`BiquadSection`] carries the normalized
//! `{b0, b1, b2, a1, a2}` (its `a0 = 1` is implicit), so the wrapper inserts the
//! unit `a0`.  Both use scipy's `sosfilt` denominator sign convention
//! (`1 + a1·z⁻¹ + a2·z⁻²`), so `a1` / `a2` pass through unchanged.
//!
//! This module is entirely unsafe-free: every IPP call and the f32/f64 dispatch
//! live in [`crate::ffi`].

use omnidsp_core::create::CreateProc;
use omnidsp_core::dispatch::Backend;
use omnidsp_core::error::{Error, Result};
use omnidsp_core::traits::iir::{IirProcessor, IirSpec};
use omnidsp_core::traits::reconfigure::Reconfigure;
use omnidsp_core::types::{BiquadSection, DspFloat};

use crate::IppBackend;
use crate::ffi;

/// Cast an f64 design coefficient to the operation precision `T`.
fn cast<T: DspFloat>(v: f64) -> Result<T> {
    T::from_f64(v).ok_or_else(|| Error::backend(0, "cannot represent biquad coefficient in T"))
}

/// Build IPP's flat biquad taps (`{b0, b1, b2, a0, a1, a2}` per section, with the
/// implicit unit `a0`) from the spec's f64 sections, cast to `T`.
fn build_taps<T: DspFloat>(sections: &[BiquadSection<f64>]) -> Result<Vec<T>> {
    let mut taps = Vec::with_capacity(sections.len() * 6);
    for s in sections {
        taps.push(cast::<T>(s.b0)?);
        taps.push(cast::<T>(s.b1)?);
        taps.push(cast::<T>(s.b2)?);
        taps.push(cast::<T>(1.0)?); // a0, normalized to 1
        taps.push(cast::<T>(s.a1)?);
        taps.push(cast::<T>(s.a2)?);
    }
    Ok(taps)
}

/// Stateful streaming IIR processor backed by IPP's `ippsIIR_BiQuad` cascade.
///
/// Created by [`IppBackend`]'s [`CreateProc<IirSpec>`](CreateProc) impl.  Owns
/// the IPP state (delay line + coefficients) and the flat taps, the latter so
/// [`reset`](IirProcessor::reset) can re-establish the cascade with a zeroed
/// delay line.
pub struct IppIirProcessor<T> {
    inner: ffi::Iir,
    /// Flat IPP taps (`6 * num_sections`), retained so `reset` can re-init the
    /// cascade coefficients with a zeroed delay line.
    taps: Vec<T>,
}

impl<T> std::fmt::Debug for IppIirProcessor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IppIirProcessor")
            .field("num_sections", &self.inner.num_sections())
            .finish_non_exhaustive()
    }
}

impl<T: DspFloat> IppIirProcessor<T> {
    /// Build a processor for `spec` at precision `T`.
    fn new(spec: &IirSpec) -> Result<Self> {
        // The non-empty-sections invariant is enforced by `IirSpec::new`.
        let taps = build_taps::<T>(spec.sections())?;
        let inner = ffi::Iir::build::<T>(&taps, spec.sections().len())?;
        Ok(Self { inner, taps })
    }
}

impl<T: DspFloat> IirProcessor<T> for IppIirProcessor<T> {
    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if input.len() != output.len() {
            return Err(Error::BufferMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }
        self.inner.apply::<T>(input, output)?;
        Ok(input.len())
    }

    fn finish(&mut self, _output: &mut [T]) -> Result<usize> {
        // An IIR cascade has an infinite impulse response — no finite ring-down
        // tail to flush.  Clear the delay state and report nothing written.
        self.reset();
        Ok(0)
    }

    fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        self.reset();
        let n = self.process(input, &mut output[..input.len()])?;
        let tail = self.finish(&mut output[n..])?;
        Ok(n + tail)
    }

    fn reset(&mut self) {
        // Re-establish the cascade with the stored taps and a zeroed delay line.
        // This repeats the construction-time `Init` with identical taps and
        // section count, so it cannot newly fail; the trait's `reset` is
        // infallible, so the status is neither expected nor surfaced (mirroring
        // how the infallible `VecOps::scale` ignores its IPP status).
        let _ = self.inner.re_init::<T>(&self.taps, None);
    }
}

impl<T: DspFloat> Reconfigure<[BiquadSection<f64>]> for IppIirProcessor<T> {
    /// Retune the cascade's section coefficients in place from `sections`,
    /// preserving the running delay line so the stream continues uninterrupted.
    ///
    /// The number of sections is structural — it sizes the IPP state buffer — so
    /// a `sections` slice with a different count is rejected with
    /// [`Error::StructuralMismatch`]; the caller should rebuild the processor.
    /// When the count matches, the delay line is read out, the cascade is
    /// re-initialized with the new (cast) coefficients, and that same delay line
    /// is carried over.
    fn reconfigure(&mut self, sections: &[BiquadSection<f64>]) -> Result<()> {
        let num_sections = self.inner.num_sections();
        if sections.len() != num_sections {
            return Err(Error::StructuralMismatch(format!(
                "IIR section count changed ({} → {}); a different count resizes the per-section \
                 state — rebuild the processor",
                num_sections,
                sections.len(),
            )));
        }

        let taps = build_taps::<T>(sections)?;
        // Carry the running delay line over the coefficient swap: read it out,
        // then re-init the cascade with the new taps and that preserved state.
        let mut dly = vec![cast::<T>(0.0)?; 2 * num_sections];
        self.inner.get_dly_line::<T>(&mut dly)?;
        self.inner.re_init::<T>(&taps, Some(&dly))?;
        self.taps = taps;
        Ok(())
    }
}

impl CreateProc<IirSpec> for IppBackend {
    type Proc<T>
        = IppIirProcessor<T>
    where
        Self: Backend<T>;

    fn create_proc<T>(&self, spec: &IirSpec) -> Result<Self::Proc<T>>
    where
        Self: Backend<T>,
        T: DspFloat,
    {
        IppIirProcessor::<T>::new(spec)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests use expect for clarity")]
mod tests {
    use super::*;
    use omnidsp_core::scalar::ScalarIir;
    use omnidsp_core::traits::iir::Iir;

    /// A stable two-section cascade (the floor's own streaming-test sections).
    fn cascade() -> Vec<BiquadSection<f64>> {
        vec![
            BiquadSection {
                b0: 0.2,
                b1: 0.3,
                b2: 0.1,
                a1: -0.5,
                a2: -0.1,
            },
            BiquadSection {
                b0: 0.5,
                b1: 0.0,
                b2: 0.0,
                a1: 0.2,
                a2: 0.05,
            },
        ]
    }

    fn unity() -> BiquadSection<f64> {
        BiquadSection {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    fn gain(g: f64) -> BiquadSection<f64> {
        BiquadSection {
            b0: g,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    /// A modest test signal at precision `T` (`f64::from(i)` is lossless; the
    /// `from_f64` narrows it to the width under test with no `as` cast).
    fn make_signal<T: DspFloat>(n: i32) -> Vec<T> {
        (0..n)
            .map(|i| {
                T::from_f64((f64::from(i) * 0.05).sin()).expect("test sample representable in T")
            })
            .collect()
    }

    fn assert_approx_eq(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: slice lengths differ ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{label}: mismatch at index {i}: got {a}, expected {e} (diff={})",
                (a - e).abs()
            );
        }
    }

    fn to_f64<T: DspFloat>(data: &[T]) -> Vec<f64> {
        data.iter()
            .map(|x| x.to_f64().expect("value representable as f64"))
            .collect()
    }

    /// Run the same spec + input through IPP and the scalar floor from a fresh
    /// (zero) state and assert agreement; this is realization-independent for the
    /// output given the input, so it validates the taps/sign conversion directly.
    fn oracle<T: DspFloat>(spec: &IirSpec, input: &[T], tol: f64, label: &str)
    where
        IppBackend: Backend<T>,
    {
        let ipp = IppBackend::new();
        let mut ipp_proc = ipp.create_proc::<T>(spec).expect("ipp processor");

        let scalar = ScalarIir::new();
        let mut scalar_proc = Iir::<T>::create_proc(&scalar, spec).expect("scalar processor");

        let zero = T::from_f64(0.0).expect("zero representable");
        let mut a = vec![zero; input.len()];
        let mut b = vec![zero; input.len()];
        ipp_proc.process(input, &mut a).expect("ipp process");
        scalar_proc.process(input, &mut b).expect("scalar process");

        assert_approx_eq(&to_f64(&a), &to_f64(&b), tol, label);
    }

    #[test]
    fn passthrough_unity_section() {
        let ipp = IppBackend::new();
        let spec = IirSpec::new(vec![unity()]).expect("valid iir spec");
        let mut plan = ipp.create_proc::<f64>(&spec).expect("processor");

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0; 5];
        plan.process(&input, &mut output).expect("process");

        assert_approx_eq(&output, &input, 1e-12, "unity passthrough");
    }

    #[test]
    fn gain_section_scales() {
        let ipp = IppBackend::new();
        let spec = IirSpec::new(vec![gain(2.0)]).expect("valid iir spec");
        let mut plan = ipp.create_proc::<f64>(&spec).expect("processor");

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0; 5];
        plan.process(&input, &mut output).expect("process");

        let expected: Vec<f64> = input.iter().map(|x| 2.0 * x).collect();
        assert_approx_eq(&output, &expected, 1e-12, "gain x2");
    }

    #[test]
    fn oracle_vs_scalar_f64() {
        let spec = IirSpec::new(cascade()).expect("valid iir spec");
        let input = make_signal::<f64>(200);
        oracle::<f64>(&spec, &input, 1e-9, "oracle f64");
    }

    #[test]
    fn oracle_vs_scalar_f32() {
        let spec = IirSpec::new(cascade()).expect("valid iir spec");
        let input = make_signal::<f32>(200);
        oracle::<f32>(&spec, &input, 5e-3, "oracle f32");
    }

    #[test]
    fn streaming_continuity() {
        let ipp = IppBackend::new();
        let spec = IirSpec::new(cascade()).expect("valid iir spec");
        let input = make_signal::<f64>(50);

        let mut single = ipp.create_proc::<f64>(&spec).expect("processor");
        let mut out_single = vec![0.0; input.len()];
        single
            .process(&input, &mut out_single)
            .expect("single-shot");

        let mut chunked = ipp.create_proc::<f64>(&spec).expect("processor");
        let mut out_chunked = vec![0.0; input.len()];
        let split = 17;
        chunked
            .process(&input[..split], &mut out_chunked[..split])
            .expect("chunk 1");
        chunked
            .process(&input[split..], &mut out_chunked[split..])
            .expect("chunk 2");

        assert_approx_eq(&out_chunked, &out_single, 1e-12, "streaming continuity");
    }

    #[test]
    fn reset_zeros_delay_line() {
        let ipp = IppBackend::new();
        let spec = IirSpec::new(cascade()).expect("valid iir spec");
        let input = make_signal::<f64>(40);

        let mut plan = ipp.create_proc::<f64>(&spec).expect("processor");
        let mut out1 = vec![0.0; input.len()];
        let mut out2 = vec![0.0; input.len()];

        plan.process(&input, &mut out1).expect("first run");
        plan.reset();
        plan.process(&input, &mut out2).expect("second run");

        assert_approx_eq(&out2, &out1, 1e-12, "reset restarts the stream");
    }

    #[test]
    fn execute_is_a_fresh_one_shot() {
        let ipp = IppBackend::new();
        let spec = IirSpec::new(cascade()).expect("valid iir spec");
        let input = make_signal::<f64>(32);

        let mut plan = ipp.create_proc::<f64>(&spec).expect("processor");
        let mut out1 = vec![0.0; input.len()];
        let mut out2 = vec![0.0; input.len()];

        let n1 = plan.execute(&input, &mut out1).expect("execute 1");
        let n2 = plan.execute(&input, &mut out2).expect("execute 2");

        assert_eq!(n1, input.len(), "execute writes one sample per input");
        assert_eq!(n2, input.len(), "execute writes one sample per input");
        assert_approx_eq(&out2, &out1, 1e-12, "execute resets between calls");
    }

    #[test]
    fn reconfigure_same_coeffs_midstream_is_noop() {
        let ipp = IppBackend::new();
        let sections = cascade();
        let spec = IirSpec::new(sections.clone()).expect("valid iir spec");
        let input = make_signal::<f64>(60);
        let split = 25;

        // Reference: process the whole stream with no reconfigure.
        let mut plain = ipp.create_proc::<f64>(&spec).expect("processor");
        let mut out_ref = vec![0.0; input.len()];
        plain
            .process(&input[..split], &mut out_ref[..split])
            .expect("ref chunk 1");
        plain
            .process(&input[split..], &mut out_ref[split..])
            .expect("ref chunk 2");

        // Reconfigure to identical coefficients mid-stream — preserving the delay
        // line, so the resumed output must match the un-reconfigured reference.
        let mut retuned = ipp.create_proc::<f64>(&spec).expect("processor");
        let mut out_rc = vec![0.0; input.len()];
        retuned
            .process(&input[..split], &mut out_rc[..split])
            .expect("retuned chunk 1");
        retuned
            .reconfigure(sections.as_slice())
            .expect("reconfigure to same coefficients");
        retuned
            .process(&input[split..], &mut out_rc[split..])
            .expect("retuned chunk 2");

        assert_approx_eq(
            &out_rc,
            &out_ref,
            1e-12,
            "reconfigure to same coefficients preserves state",
        );
    }

    #[test]
    fn reconfigure_section_count_mismatch_errors() {
        let ipp = IppBackend::new();
        let spec = IirSpec::new(vec![unity()]).expect("valid iir spec");
        let mut plan = ipp.create_proc::<f64>(&spec).expect("processor");

        // Two sections where there was one → layout change → StructuralMismatch.
        let two = [unity(), gain(2.0)];
        let err = plan
            .reconfigure(two.as_slice())
            .expect_err("different section count must error");
        assert!(
            matches!(err, Error::StructuralMismatch(_)),
            "different section count must yield StructuralMismatch, got {err:?}"
        );
    }
}
