// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Conformance checks for the composed modules: convolution, FIR, IIR, DCT,
//! Hilbert, cross-correlation, resampling, and the multirate CQT.
//!
//! Each check exercises its spec's output-affecting fields (`ConvMethod`,
//! `FirStrategy`, `DctType` × `DctNorm`, `ResampleMode`, `CrossCorrNorm`) and
//! compares against the committed scipy/numpy golden vectors in
//! [`omnidsp_testdata`], at the documented per-width tolerances.

use num_complex::Complex;

use omnidsp_core::create::CreatePlan;
use omnidsp_core::design::cqt::{self, CqtSpec};
use omnidsp_core::design::resample::{ResampleMode, ResampleSpec};
use omnidsp_core::error::Result;
use omnidsp_core::modules::cqt::{CqtPlan, CqtStreamSpec, OmniCqt};
use omnidsp_core::modules::hilbert::{HilbertPlan, HilbertSpec};
use omnidsp_core::modules::resample::ResamplePlan;
use omnidsp_core::modules::xcorr::{CrossCorrPlan, CrossCorrSpec};
use omnidsp_core::scalar::ScalarVecOps;
use omnidsp_core::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
use omnidsp_core::traits::dct::{DctNorm, DctPlan, DctSpec, DctType};
use omnidsp_core::traits::dft::{DftNorm, DftR2c, DftR2cPlan, DftR2cSpec};
use omnidsp_core::traits::fir::{FirPlan, FirSpec, FirStrategy};
use omnidsp_core::traits::iir::{IirPlan, IirSpec};
use omnidsp_core::traits::vecops::VecOps;
use omnidsp_core::types::{BiquadSection, Window};

use omnidsp_testdata::{
    cqt_process_numpy as cqtp, dct_scipy as dct, fir_lfilter_scipy as firl, hilbert_scipy as hil,
    iir_sosfilt_scipy as sos, resample_poly_scipy as rp, xcorr_scipy as xc,
};

use crate::support::{ConformanceFloat, assert_complex, assert_magnitude, assert_real, to_vec};

/// Conformance for convolution ([`OmniConv`](omnidsp_core::modules::conv::OmniConv)).
///
/// Checks the impulse identity and the known result `[1,2,3] ∗ [1,1] = [1,3,5,3]`
/// across every [`ConvMethod`], plus a buffer-length error case.
///
/// # Panics
///
/// Panics if any method deviates from the analytic result beyond
/// [`ConformanceFloat::CONV_TOL`], or if an invalid call fails to error.
pub fn check_conv<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<ConvSpec<T>>,
    B::Plan: ConvPlan<T>,
{
    let a = to_vec::<T>(&[1.0, 2.0, 3.0]);
    let k = to_vec::<T>(&[1.0, 1.0]);

    for &method in &[ConvMethod::Auto, ConvMethod::Direct, ConvMethod::Fft] {
        let spec = ConvSpec::<T>::new(a.len(), k.len(), method).expect("valid conv spec");
        let plan = b.create_plan(&spec).expect("conv plan");
        let mut out = vec![T::zero(); a.len() + k.len() - 1];
        plan.process(&a, &k, &mut out).expect("conv process");
        assert_real(
            &out,
            &[1.0, 3.0, 5.0, 3.0],
            T::CONV_TOL,
            &format!("conv [1,2,3]*[1,1] {method:?}"),
        );

        // Impulse identity: x ∗ [1] = x.
        let one = to_vec::<T>(&[1.0]);
        let ispec = ConvSpec::<T>::new(a.len(), 1, method).expect("valid conv spec");
        let iplan = b.create_plan(&ispec).expect("conv plan");
        let mut iout = vec![T::zero(); a.len()];
        iplan.process(&a, &one, &mut iout).expect("conv process");
        assert_real(
            &iout,
            &[1.0, 2.0, 3.0],
            T::CONV_TOL,
            "conv impulse identity",
        );
    }

    let spec = ConvSpec::<T>::new(3, 2, ConvMethod::Direct).expect("valid conv spec");
    let plan = b.create_plan(&spec).expect("conv plan");
    let mut wrong = vec![T::zero(); 3];
    assert!(
        plan.process(&a, &k, &mut wrong).is_err(),
        "conv [{}]: output-length mismatch must error",
        T::WIDTH,
    );
}

/// Conformance for the FIR filter ([`OmniFir`](omnidsp_core::modules::fir::OmniFir)).
///
/// Checks the impulse response, plus `scipy.signal.lfilter` golden vectors for
/// both [`FirStrategy::Direct`] and [`FirStrategy::OverlapSave`].
///
/// # Panics
///
/// Panics if either strategy deviates from the scipy reference beyond
/// [`ConformanceFloat::FIR_TOL`], or if an invalid call fails to error.
pub fn check_fir<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<FirSpec<T>>,
    B::Plan: FirPlan<T>,
{
    // Impulse response equals the coefficients.
    let coeffs = [0.25, 0.5, 0.25];
    let spec = FirSpec::new(to_vec::<T>(&coeffs)).expect("valid fir spec");
    let mut plan = b.create_plan(&spec).expect("fir plan");
    let mut impulse = vec![T::zero(); 8];
    impulse[0] = T::one();
    let mut out = vec![T::zero(); 8];
    plan.process(&impulse, &mut out).expect("fir process");
    assert_real(&out[..3], &coeffs, T::FIR_TOL, "fir impulse response");

    let golden: &[(&str, &[f64], &[f64])] = &[
        (
            "LP30 Hamming",
            firl::LFILTER_LP30_HAMMING_TAPS,
            firl::LFILTER_LP30_HAMMING_OUTPUT,
        ),
        (
            "HP30 Hann",
            firl::LFILTER_HP30_HANN_TAPS,
            firl::LFILTER_HP30_HANN_OUTPUT,
        ),
    ];
    let input = to_vec::<T>(firl::LFILTER_INPUT);
    for &(label, taps, expected) in golden {
        for &strategy in &[FirStrategy::Direct, FirStrategy::OverlapSave] {
            let spec = FirSpec::new(to_vec::<T>(taps))
                .expect("valid fir spec")
                .with_strategy(strategy);
            let mut plan = b.create_plan(&spec).expect("fir plan");
            let mut out = vec![T::zero(); input.len()];
            plan.process(&input, &mut out).expect("fir process");
            assert_real(
                &out,
                expected,
                T::FIR_TOL,
                &format!("fir {label} ({strategy:?})"),
            );
        }
    }

    let mut plan = b
        .create_plan(&FirSpec::new(to_vec::<T>(&coeffs)).expect("valid fir spec"))
        .expect("fir plan");
    let small = vec![T::zero(); 4];
    let mut wrong = vec![T::zero(); 3];
    assert!(
        plan.process(&small, &mut wrong).is_err(),
        "fir [{}]: output-length mismatch must error",
        T::WIDTH,
    );
}

/// Build biquad sections from scipy's `(b0, b1, b2, a1, a2)` SOS tuples.
fn sections<T: ConformanceFloat>(data: &[(f64, f64, f64, f64, f64)]) -> Vec<BiquadSection<T>> {
    data.iter()
        .map(|&(b0, b1, b2, a1, a2)| BiquadSection {
            b0: T::lit(b0),
            b1: T::lit(b1),
            b2: T::lit(b2),
            a1: T::lit(a1),
            a2: T::lit(a2),
        })
        .collect()
}

/// Conformance for the IIR filter ([`Iir`](omnidsp_core::traits::iir::Iir)).
///
/// Checks the unity-passthrough biquad and `scipy.signal.sosfilt` golden vectors
/// (low/high/band-pass order-4 Butterworth).
///
/// # Panics
///
/// Panics if any cascade deviates from the scipy reference beyond
/// [`ConformanceFloat::IIR_TOL`], or if an invalid call fails to error.
pub fn check_iir<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<IirSpec<T>>,
    B::Plan: IirPlan<T>,
{
    // Unity passthrough: y[n] = x[n].
    let spec = IirSpec::new(vec![BiquadSection {
        b0: T::one(),
        b1: T::zero(),
        b2: T::zero(),
        a1: T::zero(),
        a2: T::zero(),
    }])
    .expect("valid iir spec");
    let mut plan = b.create_plan(&spec).expect("iir plan");
    let signal: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let mut out = vec![T::zero(); signal.len()];
    plan.process(&to_vec::<T>(&signal), &mut out)
        .expect("iir process");
    assert_real(&out, &signal, T::IIR_TOL, "iir unity passthrough");

    let golden: &[(&str, &[(f64, f64, f64, f64, f64)], &[f64])] = &[
        ("LP4", sos::SOSFILT_LP4_SOS, sos::SOSFILT_LP4_OUTPUT),
        ("HP4", sos::SOSFILT_HP4_SOS, sos::SOSFILT_HP4_OUTPUT),
        ("BP4", sos::SOSFILT_BP4_SOS, sos::SOSFILT_BP4_OUTPUT),
    ];
    let input = to_vec::<T>(sos::SOSFILT_INPUT);
    for &(label, sos_data, expected) in golden {
        let spec = IirSpec::new(sections::<T>(sos_data)).expect("valid iir spec");
        let mut plan = b.create_plan(&spec).expect("iir plan");
        let mut out = vec![T::zero(); input.len()];
        plan.process(&input, &mut out).expect("iir process");
        assert_real(&out, expected, T::IIR_TOL, &format!("iir sosfilt {label}"));
    }

    let mut plan = b.create_plan(&spec).expect("iir plan");
    let small = vec![T::zero(); 4];
    let mut wrong = vec![T::zero(); 3];
    assert!(
        plan.process(&small, &mut wrong).is_err(),
        "iir [{}]: output-length mismatch must error",
        T::WIDTH,
    );
}

/// Conformance for the DCT ([`OmniDct`](omnidsp_core::modules::dct::OmniDct)).
///
/// Compares against scipy golden vectors over the full [`DctType`] × [`DctNorm`]
/// cross-product and four representative input lengths (even, even, prime, large).
///
/// # Panics
///
/// Panics if any (type, norm, input) deviates from the scipy reference beyond
/// [`ConformanceFloat::DCT_TOL`], or if an invalid call fails to error.
#[allow(
    clippy::too_many_lines,
    reason = "the flat 16-row DctType × DctNorm × input golden table is the test matrix and reads clearest inline"
)]
pub fn check_dct<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<DctSpec<T>>,
    B::Plan: DctPlan<T>,
{
    let cases: &[(&str, &[f64], &[f64], DctType, DctNorm)] = &[
        (
            "RAMP8 II None",
            dct::DCT_INPUT_RAMP8,
            dct::DCT2_RAMP8,
            DctType::II,
            DctNorm::None,
        ),
        (
            "RAMP8 II Ortho",
            dct::DCT_INPUT_RAMP8,
            dct::DCT2_ORTHO_RAMP8,
            DctType::II,
            DctNorm::Ortho,
        ),
        (
            "RAMP8 III None",
            dct::DCT_INPUT_RAMP8,
            dct::DCT3_RAMP8,
            DctType::III,
            DctNorm::None,
        ),
        (
            "RAMP8 III Ortho",
            dct::DCT_INPUT_RAMP8,
            dct::DCT3_ORTHO_RAMP8,
            DctType::III,
            DctNorm::Ortho,
        ),
        (
            "SINE16 II None",
            dct::DCT_INPUT_SINE16,
            dct::DCT2_SINE16,
            DctType::II,
            DctNorm::None,
        ),
        (
            "SINE16 II Ortho",
            dct::DCT_INPUT_SINE16,
            dct::DCT2_ORTHO_SINE16,
            DctType::II,
            DctNorm::Ortho,
        ),
        (
            "SINE16 III None",
            dct::DCT_INPUT_SINE16,
            dct::DCT3_SINE16,
            DctType::III,
            DctNorm::None,
        ),
        (
            "SINE16 III Ortho",
            dct::DCT_INPUT_SINE16,
            dct::DCT3_ORTHO_SINE16,
            DctType::III,
            DctNorm::Ortho,
        ),
        (
            "COS17 II None",
            dct::DCT_INPUT_COS17,
            dct::DCT2_COS17,
            DctType::II,
            DctNorm::None,
        ),
        (
            "COS17 II Ortho",
            dct::DCT_INPUT_COS17,
            dct::DCT2_ORTHO_COS17,
            DctType::II,
            DctNorm::Ortho,
        ),
        (
            "COS17 III None",
            dct::DCT_INPUT_COS17,
            dct::DCT3_COS17,
            DctType::III,
            DctNorm::None,
        ),
        (
            "COS17 III Ortho",
            dct::DCT_INPUT_COS17,
            dct::DCT3_ORTHO_COS17,
            DctType::III,
            DctNorm::Ortho,
        ),
        (
            "RANDOM64 II None",
            dct::DCT_INPUT_RANDOM64,
            dct::DCT2_RANDOM64,
            DctType::II,
            DctNorm::None,
        ),
        (
            "RANDOM64 II Ortho",
            dct::DCT_INPUT_RANDOM64,
            dct::DCT2_ORTHO_RANDOM64,
            DctType::II,
            DctNorm::Ortho,
        ),
        (
            "RANDOM64 III None",
            dct::DCT_INPUT_RANDOM64,
            dct::DCT3_RANDOM64,
            DctType::III,
            DctNorm::None,
        ),
        (
            "RANDOM64 III Ortho",
            dct::DCT_INPUT_RANDOM64,
            dct::DCT3_ORTHO_RANDOM64,
            DctType::III,
            DctNorm::Ortho,
        ),
    ];

    for &(label, input, expected, dct_type, norm) in cases {
        let n = input.len();
        let spec = DctSpec::<T>::new(n, dct_type, norm).expect("valid dct spec");
        let plan = b.create_plan(&spec).expect("dct plan");
        let mut out = vec![T::zero(); n];
        plan.process(&to_vec::<T>(input), &mut out)
            .expect("dct process");
        assert_real(&out, expected, T::DCT_TOL, label);
    }

    let spec = DctSpec::<T>::new(4, DctType::II, DctNorm::None).expect("valid dct spec");
    let plan = b.create_plan(&spec).expect("dct plan");
    let input = to_vec::<T>(&[1.0, 2.0, 3.0, 4.0]);
    let mut wrong = vec![T::zero(); 3];
    assert!(
        plan.process(&input, &mut wrong).is_err(),
        "dct [{}]: output-length mismatch must error",
        T::WIDTH,
    );
}

/// Conformance for the Hilbert transform
/// ([`OmniHilbert`](omnidsp_core::modules::hilbert::OmniHilbert)).
///
/// Compares the analytic signal against hand-computed even-length references.
///
/// # Panics
///
/// Panics if the analytic signal deviates beyond
/// [`ConformanceFloat::HILBERT_TOL`], or if an invalid call fails to error.
pub fn check_hilbert<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<HilbertSpec<T>>,
    B::Plan: HilbertPlan<T>,
{
    let cases: &[(&str, &[f64], &[(f64, f64)])] = &[
        ("N4", hil::HAND_N4_INPUT, hil::HAND_N4_EXPECTED),
        (
            "N8 cos2",
            hil::HAND_N8_COS2_INPUT,
            hil::HAND_N8_COS2_EXPECTED,
        ),
    ];
    for &(label, input, expected) in cases {
        let spec = HilbertSpec::<T>::new(input.len()).expect("valid hilbert spec");
        let plan = b.create_plan(&spec).expect("hilbert plan");
        let mut out = vec![Complex::new(T::zero(), T::zero()); input.len()];
        plan.process(&to_vec::<T>(input), &mut out)
            .expect("hilbert process");
        assert_complex(&out, expected, T::HILBERT_TOL, &format!("hilbert {label}"));
    }

    let spec = HilbertSpec::<T>::new(4).expect("valid hilbert spec");
    let plan = b.create_plan(&spec).expect("hilbert plan");
    let input = to_vec::<T>(&[1.0, 2.0, 3.0, 4.0]);
    let mut wrong = vec![Complex::new(T::zero(), T::zero()); 3];
    assert!(
        plan.process(&input, &mut wrong).is_err(),
        "hilbert [{}]: output-length mismatch must error",
        T::WIDTH,
    );
}

/// Conformance for cross-correlation
/// ([`OmniCrossCorr`](omnidsp_core::modules::xcorr::OmniCrossCorr)).
///
/// Compares against `scipy.signal.correlate` golden vectors (asymmetric, equal,
/// and sinusoidal-delay cases).  Only [`CrossCorrNorm::None`] exists today
/// (the reserved field, ADR-007 §4), so the default-constructed spec covers it.
///
/// [`CrossCorrNorm::None`]: omnidsp_core::modules::xcorr::CrossCorrNorm::None
///
/// # Panics
///
/// Panics if any case deviates from the scipy reference beyond
/// [`ConformanceFloat::XCORR_TOL`], or if an invalid call fails to error.
pub fn check_xcorr<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<CrossCorrSpec<T>>,
    B::Plan: CrossCorrPlan<T>,
{
    let cases: &[(&str, &[f64], &[f64], &[f64])] = &[
        (
            "short",
            xc::XCORR_SHORT_A,
            xc::XCORR_SHORT_B,
            xc::XCORR_SHORT_RESULT,
        ),
        (
            "equal",
            xc::XCORR_EQUAL_A,
            xc::XCORR_EQUAL_B,
            xc::XCORR_EQUAL_RESULT,
        ),
        (
            "delay",
            xc::XCORR_DELAY_A,
            xc::XCORR_DELAY_B,
            xc::XCORR_DELAY_RESULT,
        ),
    ];
    for &(label, a_ref, b_ref, expected) in cases {
        let spec = CrossCorrSpec::<T>::new(a_ref.len(), b_ref.len()).expect("valid xcorr spec");
        let plan = b.create_plan(&spec).expect("xcorr plan");
        let mut out = vec![T::zero(); spec.output_len()];
        plan.process(&to_vec::<T>(a_ref), &to_vec::<T>(b_ref), &mut out)
            .expect("xcorr process");
        assert_real(&out, expected, T::XCORR_TOL, &format!("xcorr {label}"));
    }

    let spec = CrossCorrSpec::<T>::new(4, 2).expect("valid xcorr spec");
    let plan = b.create_plan(&spec).expect("xcorr plan");
    let a = to_vec::<T>(&[1.0, 2.0, 3.0, 4.0]);
    let bv = to_vec::<T>(&[1.0, 1.0]);
    let mut wrong = vec![T::zero(); 3];
    assert!(
        plan.process(&a, &bv, &mut wrong).is_err(),
        "xcorr [{}]: output-length mismatch must error",
        T::WIDTH,
    );
}

/// Conformance for the polyphase resampler
/// ([`OmniResample`](omnidsp_core::modules::resample::OmniResample)).
///
/// Compares against `scipy.signal.upfirdn` golden vectors for 2× up- and
/// down-sampling, exercises both [`ResampleMode`]s, and checks an
/// undersized-output error case.
///
/// # Panics
///
/// Panics if a resampled signal deviates from the scipy reference beyond
/// [`ConformanceFloat::RESAMPLE_TOL`], if too few samples are produced, or if an
/// invalid call fails to error.
pub fn check_resample<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<ResampleSpec<T>>,
    B::Plan: ResamplePlan<T>,
{
    let cases: &[(&str, usize, usize, &[f64], &[f64], &[f64])] = &[
        (
            "22→44 (2× up)",
            rp::RPOLY_22_44_UP,
            rp::RPOLY_22_44_DOWN,
            rp::RPOLY_22_44_PROTO,
            rp::RPOLY_22_44_INPUT,
            rp::RPOLY_22_44_OUTPUT,
        ),
        (
            "44→22 (2× down)",
            rp::RPOLY_44_22_UP,
            rp::RPOLY_44_22_DOWN,
            rp::RPOLY_44_22_PROTO,
            rp::RPOLY_44_22_INPUT,
            rp::RPOLY_44_22_OUTPUT,
        ),
    ];
    for &(label, up, down, proto, input, expected) in cases {
        let spec = ResampleSpec::<T>::new(up, down, to_vec::<T>(proto), ResampleMode::Streaming)
            .expect("valid resample spec");
        let mut plan = b.create_plan(&spec).expect("resample plan");
        let signal = to_vec::<T>(input);
        let mut out = vec![T::zero(); plan.max_output_len(signal.len())];
        let produced = plan.process(&signal, &mut out).expect("resample process");
        assert!(
            produced >= expected.len(),
            "resample {label} [{}]: produced {produced} < reference {}",
            T::WIDTH,
            expected.len(),
        );
        assert_real(
            &out[..expected.len()],
            expected,
            T::RESAMPLE_TOL,
            &format!("resample {label}"),
        );
    }

    // Both modes produce output; streaming 2× upsample yields 2× the count.
    let proto = to_vec::<T>(&[0.5, 1.0, 0.5]);
    let signal = to_vec::<T>(&(0..16).map(|i| i as f64).collect::<Vec<_>>());
    for &mode in &[ResampleMode::Streaming, ResampleMode::Batch] {
        let spec = ResampleSpec::<T>::new(2, 1, proto.clone(), mode).expect("valid resample spec");
        let mut plan = b.create_plan(&spec).expect("resample plan");
        let mut out = vec![T::zero(); plan.max_output_len(signal.len())];
        let produced = plan.process(&signal, &mut out).expect("resample process");
        assert!(
            produced > 0,
            "resample [{}] {mode:?}: must produce output",
            T::WIDTH,
        );
    }

    let spec =
        ResampleSpec::<T>::new(2, 1, proto, ResampleMode::Streaming).expect("valid resample spec");
    let mut plan = b.create_plan(&spec).expect("resample plan");
    let input = vec![T::zero(); 8];
    let mut tiny = vec![T::zero(); 1];
    assert!(
        plan.process(&input, &mut tiny).is_err(),
        "resample [{}]: undersized output must error",
        T::WIDTH,
    );
}

/// Conformance for the multirate CQT ([`OmniCqt`]) —
/// the surface-lock capstone (ADR-006 §2a).
///
/// Designs the numpy reference spec, then compares the multirate transform's bin
/// **magnitudes** against the committed numpy golden vectors (the half-spectrum
/// correlation drops negative-frequency leakage, so it matches in magnitude,
/// not exact phase), confirms the pure-tone peak bin, and checks an error case.
///
/// # Panics
///
/// Panics if any frame's magnitudes deviate beyond [`ConformanceFloat::CQT_TOL`],
/// if the peak bin is wrong, or if an invalid call fails to error.
pub fn check_cqt<B, T>(b: &B)
where
    T: ConformanceFloat,
    B: CreatePlan<CqtSpec<T>>,
    B::Plan: CqtPlan<T>,
{
    let spec = cqt::design::<T>(
        cqtp::CQT_PROC_SAMPLE_RATE,
        cqtp::CQT_PROC_MIN_FREQ,
        cqtp::CQT_PROC_MAX_FREQ,
        cqtp::CQT_PROC_BINS_PER_OCTAVE,
        &Window::Hann,
    )
    .expect("valid cqt design");
    assert_eq!(
        spec.fft_length(),
        cqtp::CQT_PROC_FFT_LENGTH,
        "cqt fft length"
    );
    assert_eq!(spec.num_bins(), cqtp::CQT_PROC_NUM_BINS, "cqt bin count");

    let plan = b.create_plan(&spec).expect("cqt plan");
    let zero = Complex::new(T::zero(), T::zero());

    let cases: &[(&str, &[f64], &[(f64, f64)])] = &[
        (
            "all-tones",
            cqtp::CQT_PROC_ALL_TONES_INPUT,
            cqtp::CQT_PROC_ALL_TONES_OUTPUT,
        ),
        (
            "pure-tone",
            cqtp::CQT_PROC_TONE_INPUT,
            cqtp::CQT_PROC_TONE_OUTPUT,
        ),
        (
            "two-tone",
            cqtp::CQT_PROC_TWO_TONE_INPUT,
            cqtp::CQT_PROC_TWO_TONE_OUTPUT,
        ),
    ];
    for &(label, input, expected) in cases {
        let mut out = vec![zero; spec.num_bins()];
        plan.process(&to_vec::<T>(input), &mut out)
            .expect("cqt process");
        assert_magnitude(&out, expected, T::CQT_TOL, &format!("cqt {label}"));
    }

    // The pure tone peaks in its designed bin.
    let mut out = vec![zero; spec.num_bins()];
    plan.process(&to_vec::<T>(cqtp::CQT_PROC_TONE_INPUT), &mut out)
        .expect("cqt process");
    let peak = out
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).expect("no NaN"))
        .expect("at least one bin")
        .0;
    assert_eq!(peak, cqtp::CQT_PROC_TONE_BIN, "cqt peak bin");

    let bad = vec![T::zero(); spec.fft_length() - 1];
    let mut out = vec![zero; spec.num_bins()];
    assert!(
        plan.process(&bad, &mut out).is_err(),
        "cqt [{}]: input-length mismatch must error",
        T::WIDTH,
    );
}

/// Adapts a backend's `CreatePlan<DftR2cSpec>` into a [`DftR2c`] factory.
///
/// The streaming CQT factory [`OmniCqt`] is generic over a [`DftR2c`] factory,
/// not over the dispatch trait `CreatePlan<DftR2cSpec>` — and 22a deliberately
/// does **not** wire `CreatePlan<CqtStreamSpec>` (that is 22c).  Their
/// `create_plan` signatures are identical, so this borrowing newtype lets
/// [`check_cqt_stream`] build the streaming plan over the backend under test's
/// own real-DFT primitive without a dispatch route.
struct R2cFactory<'b, B>(&'b B);

impl<B, T> DftR2c<T> for R2cFactory<'_, B>
where
    B: CreatePlan<DftR2cSpec<T>>,
    B::Plan: omnidsp_core::traits::dft::DftR2cPlan<T>,
{
    type Plan = B::Plan;

    fn create_plan(&self, spec: &DftR2cSpec<T>) -> Result<Self::Plan> {
        self.0.create_plan(spec)
    }
}

/// An **independent** newest-anchored single-FFT CQT reference for
/// [`check_cqt_stream`].
///
/// A decimation-free single-FFT CQT with **end-placed** kernels over the newest
/// `fft_length` samples — the same anchoring the streaming plan produces,
/// computed without the streaming plan, its rings, or its decimators (mirroring
/// how the batch 05L path is validated against the single-FFT oracle, and how
/// `OmniCqt`'s own conformance pins to a separate reference).  Each bin's kernel
/// `(1/N_k)·w[n]·exp(+j·2π·f_k·n/sr)` sits at frame indices
/// `fft_length−N_k … fft_length−1`, so the bin analyses the newest `N_k` samples
/// and its analysis ends at the window's last sample ("now").
struct NewestRef<T, P> {
    kernels: Vec<Vec<Complex<T>>>,
    plan: P,
    fft_length: usize,
}

impl<T, P> NewestRef<T, P>
where
    T: ConformanceFloat + std::ops::MulAssign,
    ScalarVecOps: VecOps<T>,
    P: DftR2cPlan<T>,
{
    /// Build the reference from a [`CqtSpec`] using `b`'s real-DFT primitive.
    #[allow(
        clippy::cast_precision_loss,
        reason = "kernel and FFT lengths are small enough for f64"
    )]
    fn new<B>(b: &B, spec: &CqtSpec<T>) -> Self
    where
        B: CreatePlan<DftR2cSpec<T>, Plan = P>,
    {
        let fft_length = spec.fft_length();
        let sr = spec.sample_rate();
        let half_len = fft_length / 2 + 1;
        let r2c_spec = DftR2cSpec::new(fft_length, DftNorm::None).expect("valid reference spec");
        let plan = b.create_plan(&r2c_spec).expect("reference r2c plan");

        let inv_n = 1.0 / fft_length as f64;
        let tau = std::f64::consts::TAU;
        let mut kernels = Vec::with_capacity(spec.num_bins());
        for bin in spec.bins() {
            let nk = bin.window.len();
            let kernel_start = fft_length - nk; // end placement
            let inv_nk = 1.0 / nk as f64;
            let mut k = vec![Complex::new(T::zero(), T::zero()); half_len];
            for (m, slot) in k.iter_mut().enumerate() {
                let mut acc_re = 0.0_f64;
                let mut acc_im = 0.0_f64;
                for (n, wn) in bin.window.iter().enumerate() {
                    let p = (kernel_start + n) as f64;
                    let angle =
                        tau * (bin.frequency * n as f64 / sr - (m as f64 * p) / fft_length as f64);
                    let amp = wn.to_f64().unwrap_or(0.0) * inv_nk;
                    acc_re += amp * angle.cos();
                    acc_im += amp * angle.sin();
                }
                // Stored kernel = conj(DFT)/fft_length, matching the plan, then
                // narrowed to the width under test.
                *slot = Complex::new(T::lit(acc_re * inv_n), T::lit(-acc_im * inv_n));
            }
            kernels.push(k);
        }

        Self {
            kernels,
            plan,
            fft_length,
        }
    }

    /// The newest-anchored CQT column of `window` (length `fft_length`).
    fn column(&self, window: &[T]) -> Vec<Complex<T>> {
        let half_len = self.fft_length / 2 + 1;
        let mut seg = window.to_vec();
        let mut half = vec![Complex::new(T::zero(), T::zero()); half_len];
        self.plan
            .process(&mut seg, &mut half)
            .expect("reference r2c process");
        let ops = ScalarVecOps;
        self.kernels
            .iter()
            .map(|k| ops.cdot(&half, k).expect("reference cdot"))
            .collect()
    }
}

/// Conformance for the streaming, newest-anchored CQT
/// ([`OmniCqtStreamPlan`](omnidsp_core::modules::cqt::OmniCqtStreamPlan), ticket 22).
///
/// Drives the stateful `&mut self` plan the way [`check_resample`] drives the
/// resampler — variable-count `process`, `max_output_columns` sizing, `reset` —
/// and verifies the **newest-anchored equivalence**: each settled column equals
/// an **independent** newest-anchored reference (`NewestRef`: a decimation-free
/// single-FFT CQT with end-placed kernels over the newest `fft_length` samples,
/// computed without the streaming plan, mirroring how 05L is validated against
/// the single-FFT oracle).
///
/// - **Top octave** (no decimation): identical frame end and kernel length, so
///   the streaming complex value matches the reference within
///   [`ConformanceFloat::CQT_TOL`].
/// - **Deeper octaves**: the multirate path runs the signal through 1–2
///   half-band ×2 decimation stages whose response is not perfectly flat, so it
///   differs from the decimation-free reference by a per-octave residual — the
///   same multirate-vs-single-FFT gap 05L tolerates.  The **top octave** (no
///   decimation) is held to `CQT_TOL` for magnitude *and* complex; the **deeper
///   octaves** are bounded by documented per-octave residual tolerances
///   (`deep_mag_tol` ≈ 7.5e-3, `deep_complex_tol` ≈ 6e-2 measured) that reflect
///   the decimator's amplitude/phase — a real residual, not a loosening of the
///   math (the undecimated top octave still matching to ~3e-5 proves the math).
///
/// A **stationary cross-check** rides alongside it: on a single steady tone the
/// streaming magnitude matches the oldest-anchored 05L batch (anchoring preserves
/// magnitude for signals stationary across the frame), and the two differ by a
/// pure per-bin phase — the only thing the oldest batch is still good for here.
///
/// The plans are built from the same [`OmniCqt`] over the backend's real-DFT
/// primitive (via `R2cFactory`) and [`ScalarVecOps`], routing the octave
/// decimator through `b`.  This is the **decisive** guard: the streaming plan
/// runs **continuous** per-octave decimators (persistent state, never reset per
/// frame) and analyses each frame from per-octave rings, whereas the reference
/// is decimation-free; they agree because the streaming path compensates each
/// octave's decimator group delay and snaps "now" to the deepest octave's grid.
///
/// **Settled**: a column is verified only once `now ≥ 2·fft_length` — past the
/// continuous decimators' startup *and* a full frame of real decimated samples
/// in every octave's ring.  "now" is the hop instant **snapped down** to the
/// deepest octave's stride `2^(O−1)` (matching the plan), so the reference window
/// is `signal[now − fft .. now]` at that snapped instant.
///
/// # Panics
///
/// Panics if `max_output_columns` fails to bound the emitted count, if `process`
/// writes the wrong count or wrong values, if a settled column deviates from the
/// independent newest-anchored reference beyond the documented tolerances, if the
/// stationary cross-check fails, or if `reset` fails to restore the initial
/// state.
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive harness: trait mechanics, newest-anchored equivalence, stationary cross-check, and reset"
)]
pub fn check_cqt_stream<B, T>(b: &B)
where
    T: ConformanceFloat + std::ops::MulAssign,
    ScalarVecOps: omnidsp_core::traits::vecops::VecOps<T>,
    B: CreatePlan<DftR2cSpec<T>> + CreatePlan<ResampleSpec<T>>,
    <B as CreatePlan<DftR2cSpec<T>>>::Plan: omnidsp_core::traits::dft::DftR2cPlan<T>,
    <B as CreatePlan<ResampleSpec<T>>>::Plan: ResamplePlan<T>,
{
    let spec =
        cqt::design::<T>(16000.0, 125.0, 1000.0, 12, &Window::Hann).expect("valid cqt design");
    let factory = OmniCqt::new(R2cFactory(b), ScalarVecOps);

    let stream_spec = CqtStreamSpec::new(spec.clone());
    let mut plan = factory
        .create_stream_plan(&stream_spec, b)
        .expect("cqt stream plan");
    // The independent newest-anchored reference (decimation-free, end-placed
    // kernels) and the oldest-anchored 05L batch (the stationary cross-check).
    let reference = NewestRef::new(b, &spec);
    let batch = factory
        .create_plan(&spec, b)
        .expect("cqt batch cross-check");

    let nb = plan.num_bins();
    assert_eq!(nb, spec.num_bins(), "cqt stream [{}]: bin count", T::WIDTH);
    let fft = batch.fft_length();
    let hop = plan.hop_length();
    let sr = spec.sample_rate();
    let freqs: Vec<f64> = plan.bin_frequencies().to_vec();
    let zero = Complex::new(T::zero(), T::zero());
    // The plan quantises "now" to the deepest octave's sample stride 2^(O−1);
    // the reference window must be taken at the same snapped instant.
    let num_octaves = batch.num_octaves();
    let align = 1usize << (num_octaves - 1);
    // The top octave (no decimation) is the high-frequency tail of the bins.
    let top_octave_lo = nb - 12.min(nb);
    // Deep-octave residuals against the **decimation-free** reference.  The
    // multirate streaming path runs deeper-octave bins through 1–2 half-band ×2
    // decimation stages whose amplitude/phase response is not perfectly flat, so
    // it differs from a single-FFT computed on the full-rate signal by a
    // per-octave residual — exactly the multirate-vs-single-FFT gap the 05L
    // capstone tolerates (it pins multirate to its single-FFT oracle only on the
    // no-decimation spec, and to librosa at `CQT_TOL` otherwise).  Measured
    // worst-case here (16 kHz, 125–1000 Hz, 12 b/oct): magnitude ≈ 7.5e-3,
    // complex ≈ 6e-2 (relative).  These bounds cover them with margin; they are
    // the real decimator residual, not a loosening — the top octave (no
    // decimation) is still held to the tight `CQT_TOL` for both magnitude and
    // complex.
    let deep_mag_tol = T::lit(1.5e-2);
    let deep_complex_tol = T::lit(1e-1);

    // ── Trait mechanics: max_output_columns bounds the emitted count. ──
    for &len in &[0usize, 1, hop, hop + 1, 2 * hop + 3] {
        let bound = plan.max_output_columns(len);
        let mut out = vec![zero; bound * nb];
        let input = vec![T::zero(); len];
        let produced = plan.process(&input, &mut out).expect("cqt stream process");
        assert!(
            produced <= bound,
            "cqt stream [{}]: produced {produced} > bound {bound} for len {len}",
            T::WIDTH,
        );
    }
    plan.reset();

    // ── Undersized output must error. ──
    {
        let input = vec![T::zero(); hop];
        let mut tiny = vec![zero; nb - 1];
        assert!(
            plan.process(&input, &mut tiny).is_err(),
            "cqt stream [{}]: undersized output must error",
            T::WIDTH,
        );
        plan.reset();
    }

    // ── A broadband signal long enough to settle several columns past the
    // warm-up (continuous-decimator startup + a full frame). ──
    let total = 3 * fft + 6 * hop;
    let signal: Vec<T> = (0..total)
        .map(|i| {
            let s: f64 = freqs
                .iter()
                .map(|&f| (std::f64::consts::TAU * f * i as f64 / sr).sin())
                .sum();
            T::lit(s)
        })
        .collect();

    let mut out = vec![zero; plan.max_output_columns(total) * nb];
    let columns = plan.process(&signal, &mut out).expect("cqt stream process");
    assert!(
        columns >= 1,
        "cqt stream [{}]: expected ≥1 column, got {columns}",
        T::WIDTH,
    );

    // Each settled column equals the independent newest-anchored reference of
    // its window — magnitude in every octave; complex tight for the top octave,
    // bounded by the decimation residual for deeper octaves.
    let mut settled = 0usize;
    for c in 0..columns {
        // Snap the hop instant to the deepest octave's grid (as the plan does),
        // then require a full settled frame past warm-up.
        let raw = (c + 1) * hop;
        let now = raw - (raw % align);
        if now < 2 * fft {
            continue; // warm-up: decimator startup + ring fill not yet flushed
        }
        let window = &signal[now - fft..now];
        let ref_col = reference.column(window);

        let col = &out[c * nb..(c + 1) * nb];
        for (k, (s, o)) in col.iter().zip(&ref_col).enumerate() {
            // Top octave (no decimation): magnitude *and* complex are tight.
            // Deeper octaves: magnitude close, complex bounded by the decimator
            // residual (both are newest-anchored, the residual is the multirate
            // decimation, not anchoring).
            let (mag_tol, cplx_tol) = if k >= top_octave_lo {
                (T::CQT_TOL, T::CQT_TOL)
            } else {
                (deep_mag_tol, deep_complex_tol)
            };
            assert!(
                (s.norm() - o.norm()).abs() <= mag_tol * (T::one() + o.norm()),
                "cqt stream [{}]: col {c} bin {k}: |stream| {:?} vs |reference| {:?}",
                T::WIDTH,
                s.norm(),
                o.norm(),
            );
            assert!(
                (*s - *o).norm() <= cplx_tol * (T::one() + o.norm()),
                "cqt stream [{}]: col {c} bin {k}: stream {:?} vs newest reference {:?}",
                T::WIDTH,
                s,
                o,
            );
        }
        settled += 1;
    }
    assert!(
        settled >= 1,
        "cqt stream [{}]: expected ≥1 settled column to verify",
        T::WIDTH,
    );

    // ── Stationary cross-check against the oldest-anchored 05L batch. ──
    // On a single steady tone (stationary across the frame) the streaming
    // magnitude matches the batch in every octave, and for the undecimated top
    // octave the two differ by a pure per-bin phase (|stream/batch| ≈ 1).
    let target = nb - 6;
    let f_tone = freqs[target];
    let tone: Vec<T> = (0..total)
        .map(|i| T::lit((std::f64::consts::TAU * f_tone * i as f64 / sr).sin()))
        .collect();
    plan.reset();
    let mut tone_out = vec![zero; plan.max_output_columns(total) * nb];
    let tone_cols = plan
        .process(&tone, &mut tone_out)
        .expect("stationary process");
    let mut stationary_checked = 0usize;
    for c in 0..tone_cols {
        let raw = (c + 1) * hop;
        let now = raw - (raw % align);
        if now < 2 * fft {
            continue;
        }
        let window = &tone[now - fft..now];
        let mut batch_col = vec![zero; nb];
        batch
            .process(window, &mut batch_col)
            .expect("batch cross-check");
        let col = &tone_out[c * nb..(c + 1) * nb];
        for (k, (s, bcol)) in col.iter().zip(&batch_col).enumerate() {
            assert!(
                (s.norm() - bcol.norm()).abs() <= T::CQT_TOL * (T::one() + bcol.norm()),
                "cqt stream [{}]: stationary col {c} bin {k}: |stream| {:?} vs |batch| {:?}",
                T::WIDTH,
                s.norm(),
                bcol.norm(),
            );
        }
        // Pure per-bin phase on the tone bin (top octave): |stream/batch| ≈ 1.
        let ratio = col[target] / batch_col[target];
        assert!(
            (ratio.norm() - T::one()).abs() <= T::CQT_TOL,
            "cqt stream [{}]: stationary tone bin {target} col {c}: |stream/batch| {:?} should be 1",
            T::WIDTH,
            ratio.norm(),
        );
        stationary_checked += 1;
    }
    assert!(
        stationary_checked >= 1,
        "cqt stream [{}]: expected ≥1 settled stationary column",
        T::WIDTH,
    );
    plan.reset();

    // ── reset returns to the initial state: a fresh feed reproduces the run. ──
    let mut fresh = factory
        .create_stream_plan(&stream_spec, b)
        .expect("cqt stream plan");
    plan.reset();
    let mut out_reset = vec![zero; plan.max_output_columns(total) * nb];
    let mut out_fresh = vec![zero; fresh.max_output_columns(total) * nb];
    let cols_reset = plan.process(&signal, &mut out_reset).expect("post-reset");
    let cols_fresh = fresh.process(&signal, &mut out_fresh).expect("fresh");
    assert_eq!(
        cols_reset,
        cols_fresh,
        "cqt stream [{}]: reset restores the column cadence",
        T::WIDTH,
    );
    for (k, (a, e)) in out_reset[..cols_reset * nb]
        .iter()
        .zip(&out_fresh)
        .enumerate()
    {
        assert!(
            (*a - *e).norm() <= T::CQT_TOL * (T::one() + e.norm()),
            "cqt stream [{}]: index {k}: post-reset {:?} vs fresh {:?}",
            T::WIDTH,
            a,
            e,
        );
    }
}
