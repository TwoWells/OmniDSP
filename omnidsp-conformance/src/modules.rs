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
use omnidsp_core::modules::cqt::CqtPlan;
use omnidsp_core::modules::hilbert::{HilbertPlan, HilbertSpec};
use omnidsp_core::modules::resample::ResamplePlan;
use omnidsp_core::modules::xcorr::{CrossCorrPlan, CrossCorrSpec};
use omnidsp_core::traits::conv::{ConvMethod, ConvPlan, ConvSpec};
use omnidsp_core::traits::dct::{DctNorm, DctPlan, DctSpec, DctType};
use omnidsp_core::traits::fir::{FirPlan, FirSpec, FirStrategy};
use omnidsp_core::traits::iir::{IirPlan, IirSpec};
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

/// Conformance for the multirate CQT ([`OmniCqt`](omnidsp_core::modules::cqt::OmniCqt)) —
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
