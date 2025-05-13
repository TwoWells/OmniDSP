/**
 * @file omnidsp.tpp
 * @brief Template method definitions for the OmniDSP class.
 * This file is intended to be included at the end of omnidsp.hpp.
 */

#ifndef OMNIDSP_TPP
#define OMNIDSP_TPP

#include "omnidsp.hpp"

#include "OmniDSP/params/fft.hpp"
#include "OmniDSP/params/convolution.hpp"
#include "OmniDSP/params/fir_filter.hpp"
#include "OmniDSP/params/iir_filter.hpp"
#include "OmniDSP/params/resample.hpp"
#include "OmniDSP/params/cqt.hpp"

#include "OmniDSP/utils.hpp"

namespace OmniDSP {

// --- One-off DSP Operations (Template Definitions) ---
// (convolve, fft, generate_window definitions are from previous version)
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<Utils::GetRealType<T>>> OmniDSP::convolve(
    const std::vector<Utils::GetRealType<T>>& input,
    const std::vector<Utils::GetRealType<T>>& kernel,
    ConvolutionType type,
    ConvolutionMethod method) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    if constexpr (std::is_same_v<Utils::GetRealType<T>, F32>) {
        return pimpl_->convolve_f32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<Utils::GetRealType<T>, F64>) {
        return pimpl_->convolve_f64(input, kernel, type, method);
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>> OmniDSP::convolve(
    const std::vector<Utils::GetComplexType<T>>& input,
    const std::vector<Utils::GetComplexType<T>>& kernel,
    ConvolutionType type,
    ConvolutionMethod method) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    if constexpr (std::is_same_v<Utils::GetComplexType<T>, C32>) {
        return pimpl_->convolve_c32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<Utils::GetComplexType<T>, C64>) {
        return pimpl_->convolve_c64(input, kernel, type, method);
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<Utils::GetRealType<T>>> OmniDSP::correlate(
    const std::vector<Utils::GetRealType<T>>& input,
    const std::vector<Utils::GetRealType<T>>& kernel,
    ConvolutionType type,
    ConvolutionMethod method) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    if constexpr (std::is_same_v<Utils::GetRealType<T>, F32>) {
        return pimpl_->correlate_f32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<Utils::GetRealType<T>, F64>) {
        return pimpl_->correlate_f64(input, kernel, type, method);
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>> OmniDSP::correlate(
    const std::vector<Utils::GetComplexType<T>>& input,
    const std::vector<Utils::GetComplexType<T>>& kernel,
    ConvolutionType type,
    ConvolutionMethod method) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    if constexpr (std::is_same_v<Utils::GetComplexType<T>, C32>) {
        return pimpl_->correlate_c32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<Utils::GetComplexType<T>, C64>) {
        return pimpl_->correlate_c64(input, kernel, type, method);
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
}


template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>> OmniDSP::fft(
    const std::vector<Utils::GetComplexType<T>>& input) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    if constexpr (std::is_same_v<Utils::GetComplexType<T>, C32>) {
        return pimpl_->fft_c32(input);
    } else if constexpr (std::is_same_v<Utils::GetComplexType<T>, C64>) {
        return pimpl_->fft_c64(input);
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
}

// Definitions for ifft, rfft, irfft
template <typename T> // T is C32 or C64
[[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>> OmniDSP::ifft(
    const std::vector<Utils::GetComplexType<T>>& input) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    if constexpr (std::is_same_v<Utils::GetComplexType<T>, C32>) {
        return pimpl_->ifft_c32(input);
    } else if constexpr (std::is_same_v<Utils::GetComplexType<T>, C64>) {
        return pimpl_->ifft_c64(input);
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T> // T is F32 or F64
[[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>> OmniDSP::rfft(
    const std::vector<Utils::GetRealType<T>>& input) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    if constexpr (std::is_same_v<Utils::GetRealType<T>, F32>) {
        return pimpl_->rfft_f32(input);
    } else if constexpr (std::is_same_v<Utils::GetRealType<T>, F64>) {
        return pimpl_->rfft_f64(input);
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T> // T is F32 or F64 (real type for output)
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::irfft(
    const std::vector<Utils::GetComplexType<T>>& input,
    size_t output_length) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    if constexpr (std::is_same_v<T, F32>) { // T is the real output type
        // Ensure input is C32 if output is F32
        static_assert(std::is_same_v<Utils::GetComplexType<T>, C32>, "Input for irfft<F32> must be C32 vector.");
        return pimpl_->irfft_c32(input, output_length);
    } else if constexpr (std::is_same_v<T, F64>) { // T is the real output type
        // Ensure input is C64 if output is F64
        static_assert(std::is_same_v<Utils::GetComplexType<T>, C64>, "Input for irfft<F64> must be C64 vector.");
        return pimpl_->irfft_c64(input, output_length);
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
}


template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::generate_window(
    const WindowSetup& setup) const {
    if (!pimpl_) return std::unexpected(Status::NotInitialized);
    std::vector<T> output_vec(setup.length);
    if (setup.length == 0) return output_vec;

    Status status = Status::Failure;
    if constexpr (std::is_same_v<T, F32>) {
        switch (setup.type) {
            case WindowType::Bartlett:    status = pimpl_->bartlett_window_f32(setup.length, output_vec); break;
            case WindowType::Blackman:    status = pimpl_->blackman_window_f32(setup.length, output_vec); break;
            case WindowType::Flattop:     status = pimpl_->flattop_window_f32(setup.length, output_vec); break;
            case WindowType::Gaussian:
                status = pimpl_->gaussian_window_f32(setup.length, setup.params.has_value() && setup.params->count("sigma") ? setup.params->at("sigma") : 0.25, output_vec);
                break;
            case WindowType::Hamming:     status = pimpl_->hamming_window_f32(setup.length, output_vec); break;
            case WindowType::Hann:        status = pimpl_->hann_window_f32(setup.length, output_vec); break;
            case WindowType::Kaiser:
                status = pimpl_->kaiser_window_f32(setup.length, setup.params.has_value() && setup.params->count("beta") ? setup.params->at("beta") : 0.0, output_vec);
                break;
            case WindowType::Rectangular: status = pimpl_->rectangular_window_f32(setup.length, output_vec); break;
            case WindowType::Triangular:  status = pimpl_->triangular_window_f32(setup.length, output_vec); break;
            default: return std::unexpected(Status::InvalidArgument);
        }
    } else if constexpr (std::is_same_v<T, F64>) {
        switch (setup.type) {
            case WindowType::Bartlett:    status = pimpl_->bartlett_window_f64(setup.length, output_vec); break;
            case WindowType::Blackman:    status = pimpl_->blackman_window_f64(setup.length, output_vec); break;
            case WindowType::Flattop:     status = pimpl_->flattop_window_f64(setup.length, output_vec); break;
            case WindowType::Gaussian:
                status = pimpl_->gaussian_window_f64(setup.length, setup.params.has_value() && setup.params->count("sigma") ? setup.params->at("sigma") : 0.25, output_vec);
                break;
            case WindowType::Hamming:     status = pimpl_->hamming_window_f64(setup.length, output_vec); break;
            case WindowType::Hann:        status = pimpl_->hann_window_f64(setup.length, output_vec); break;
            case WindowType::Kaiser:
                status = pimpl_->kaiser_window_f64(setup.length, setup.params.has_value() && setup.params->count("beta") ? setup.params->at("beta") : 0.0, output_vec);
                break;
            case WindowType::Rectangular: status = pimpl_->rectangular_window_f64(setup.length, output_vec); break;
            case WindowType::Triangular:  status = pimpl_->triangular_window_f64(setup.length, output_vec); break;
            default: return std::unexpected(Status::InvalidArgument);
        }
    } else {
        return std::unexpected(Status::UnsupportedFeature);
    }
    if(status != Status::Success) return std::unexpected(status);
    return output_vec;
}

// Deprecated window functions
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::bartlett_window(size_t length) const {
    return generate_window<T>(WindowSetup(WindowType::Bartlett, static_cast<int>(length)));
}
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::blackman_window(size_t length) const {
    return generate_window<T>(WindowSetup(WindowType::Blackman, static_cast<int>(length)));
}
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::flattop_window(size_t length) const {
    return generate_window<T>(WindowSetup(WindowType::Flattop, static_cast<int>(length)));
}
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::gaussian_window(size_t length, double stddev) const {
    return generate_window<T>(WindowSetup(WindowType::Gaussian, static_cast<int>(length), WindowParams{{"sigma", stddev}}));
}
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::hamming_window(size_t length) const {
    return generate_window<T>(WindowSetup(WindowType::Hamming, static_cast<int>(length)));
}
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::hann_window(size_t length) const {
    return generate_window<T>(WindowSetup(WindowType::Hann, static_cast<int>(length)));
}
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::kaiser_window(size_t length, double beta) const {
    return generate_window<T>(WindowSetup(WindowType::Kaiser, static_cast<int>(length), WindowParams{{"beta", beta}}));
}
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::rectangular_window(size_t length) const {
    return generate_window<T>(WindowSetup(WindowType::Rectangular, static_cast<int>(length)));
}
template <typename T>
[[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::triangular_window(size_t length) const {
    return generate_window<T>(WindowSetup(WindowType::Triangular, static_cast<int>(length)));
}


// --- create_plan / create_processor template definitions ---
// (Definitions for create_plan, create_processor, design_fir_filter, design_iir_filter
//  are from the previous artifact omnidsp_hpp_builder, which was based on omnidsp_hpp_create_overload)
// (Copied here for completeness of omnidsp.tpp)

// FFTPlan (stateless) from FFTParams
template <typename T_Complex>
[[nodiscard]] inline OmniExpected<std::unique_ptr<FFTPlan<T_Complex>>> OmniDSP::create_plan(
    const FFTParams& params) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>> pimpl_expected;
  if constexpr (std::is_same_v<T_Complex, C32>) {
      pimpl_expected = pimpl_->create_fft_plan_impl_c32(params.length_);
  } else if constexpr (std::is_same_v<T_Complex, C64>) {
      pimpl_expected = pimpl_->create_fft_plan_impl_c64(params.length_);
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
  if (!pimpl_expected) return std::unexpected(pimpl_expected.error());
  auto plan = FFTPlan<T_Complex>::create_from_impl(std::move(pimpl_expected.value()));
  if (!plan) return std::unexpected(Status::Failure);
  return plan;
}

// RFFTPlan (stateless) from RFFTParams
template <typename T_Real>
[[nodiscard]] inline OmniExpected<std::unique_ptr<RFFTPlan<T_Real>>> OmniDSP::create_plan(
    const RFFTParams& params) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>> pimpl_expected;
  if constexpr (std::is_same_v<T_Real, F32>) {
      pimpl_expected = pimpl_->create_rfft_plan_impl_f32(params.length_);
  } else if constexpr (std::is_same_v<T_Real, F64>) {
      pimpl_expected = pimpl_->create_rfft_plan_impl_f64(params.length_);
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
  if (!pimpl_expected) return std::unexpected(pimpl_expected.error());
  auto plan = RFFTPlan<T_Real>::create_from_impl(std::move(pimpl_expected.value()));
  if (!plan) return std::unexpected(Status::Failure);
  return plan;
}

// ConvolutionPlan (stateless) from ConvolutionParams + kernel_coeffs
template <typename T>
[[nodiscard]] inline OmniExpected<std::unique_ptr<ConvolutionPlan<T>>> OmniDSP::create_plan(
    const ConvolutionParams& params, const std::vector<T>& kernel_coeffs) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<T>>> pimpl_expected;
  if constexpr (std::is_same_v<T, F32>) {
      pimpl_expected = pimpl_->create_convolution_plan_impl_f32(kernel_coeffs, params.type_, params.method_hint_);
  } else if constexpr (std::is_same_v<T, F64>) {
      pimpl_expected = pimpl_->create_convolution_plan_impl_f64(kernel_coeffs, params.type_, params.method_hint_);
  } else if constexpr (std::is_same_v<T, C32>) {
      pimpl_expected = pimpl_->create_convolution_plan_impl_c32(kernel_coeffs, params.type_, params.method_hint_);
  } else if constexpr (std::is_same_v<T, C64>) {
      pimpl_expected = pimpl_->create_convolution_plan_impl_c64(kernel_coeffs, params.type_, params.method_hint_);
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
  if (!pimpl_expected) return std::unexpected(pimpl_expected.error());
  auto plan = ConvolutionPlan<T>::create_from_impl(std::move(pimpl_expected.value()));
  if (!plan) return std::unexpected(Status::Failure);
  return plan;
}

template <typename T>
OmniExpected<std::unique_ptr<CorrelationPlan<T>>> OmniDSP::create_plan(
    const CorrelationParams& params,
    const std::vector<T>& kernel_coeffs) const {
  if (!pimpl_) {
    return std::unexpected(Status::InvalidState); // Or some other appropriate error
  }

  // NOTE: This implementation assumes that the pimpl_->create_correlation_plan_impl
  // will be updated or is capable of using the information from params (especially max_input_length_)
  // and kernel_coeffs to correctly size internal FFTs if needed.
  // The current Abstract::Backend::create_correlation_plan_impl signature might need
  // to change to accept CorrelationParams or max_input_length.
  // This is tracked in a separate TODO:
  // "[ ] Add/Update create_correlation_plan_impl in Abstract::Backend to accept CorrelationParams (or max_input_length) and kernel."

  OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<T>>> plan_impl_expected(std::unexpected(Status::NotImplemented));

  if constexpr (std::is_same_v<T, F32>) {
    plan_impl_expected = pimpl_->create_correlation_plan_impl_f32(kernel_coeffs, params.type_, params.method_hint_);
  } else if constexpr (std::is_same_v<T, F64>) {
    plan_impl_expected = pimpl_->create_correlation_plan_impl_f64(kernel_coeffs, params.type_, params.method_hint_);
  } else if constexpr (std::is_same_v<T, C32>) {
    plan_impl_expected = pimpl_->create_correlation_plan_impl_c32(kernel_coeffs, params.type_, params.method_hint_);
  } else if constexpr (std::is_same_v<T, C64>) {
    plan_impl_expected = pimpl_->create_correlation_plan_impl_c64(kernel_coeffs, params.type_, params.method_hint_);
  } else {
    static_assert(always_false<T>::value, "Unsupported type for CorrelationPlan");
    return std::unexpected(Status::UnsupportedType);
  }

  if (!plan_impl_expected) {
    return std::unexpected(plan_impl_expected.error());
  }
  // Assuming CorrelationPlan<T>::create_from_impl exists similar to ConvolutionPlan
  return CorrelationPlan<T>::create_from_impl(std::move(plan_impl_expected.value()));
}

// FIRFilterProcessor (stateful) from FIRFilterParams
template <typename T>
[[nodiscard]] inline OmniExpected<std::unique_ptr<FIRFilterProcessor<T>>> /* TODO: FIRFilterProcessor */ OmniDSP::create_processor(
    const FIRFilterParams& params) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  auto coeffs_expected = design_fir_filter<T>(params);
  if (!coeffs_expected) {
      return std::unexpected(coeffs_expected.error());
  }
  return create_processor<T>(coeffs_expected.value());
}

// FIRFilterProcessor (stateful) from FIRCoefs
template <typename T>
[[nodiscard]] inline OmniExpected<std::unique_ptr<FIRFilterProcessor<T>>> /* TODO: FIRFilterProcessor */ OmniDSP::create_processor(
    const FIRCoefs<T>& coeffs) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  OmniExpected<std::unique_ptr<Abstract::FIRFilterProcessorImpl<T>>> /* TODO: Abstract::FIRFilterProcessorImpl */ pimpl_expected;
  if constexpr (std::is_same_v<T, F32>) {
      pimpl_expected = pimpl_->create_fir_filter_plan_impl_f32(coeffs); // TODO: create_fir_filter_processor_impl_f32
  } else if constexpr (std::is_same_v<T, F64>) {
      pimpl_expected = pimpl_->create_fir_filter_plan_impl_f64(coeffs); // TODO: create_fir_filter_processor_impl_f64
  } else if constexpr (std::is_same_v<T, C32>) {
      pimpl_expected = pimpl_->create_fir_filter_plan_impl_c32(coeffs); // TODO: create_fir_filter_processor_impl_c32
  } else if constexpr (std::is_same_v<T, C64>) {
      pimpl_expected = pimpl_->create_fir_filter_plan_impl_c64(coeffs); // TODO: create_fir_filter_processor_impl_c64
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
  if (!pimpl_expected) return std::unexpected(pimpl_expected.error());
  auto processor = FIRFilterProcessor<T>::create_from_impl(std::move(pimpl_expected.value())); // TODO: FIRFilterProcessor<T>
  if (!processor) return std::unexpected(Status::Failure);
  return processor;
}


// IIRFilterProcessor (stateful) from IIRFilterParams
template <typename T_Real>
[[nodiscard]] inline OmniExpected<std::unique_ptr<IIRFilterProcessor<T_Real>>> /* TODO: IIRFilterProcessor */ OmniDSP::create_processor(
    const IIRFilterParams& params) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  auto sos_coeffs_expected = design_iir_filter<T_Real>(params);
  if (!sos_coeffs_expected) {
      return std::unexpected(sos_coeffs_expected.error());
  }
  return create_processor<T_Real>(sos_coeffs_expected.value());
}

// IIRFilterProcessor (stateful) from SOS coefficients
template <typename T_Real>
[[nodiscard]] inline OmniExpected<std::unique_ptr<IIRFilterProcessor<T_Real>>> /* TODO: IIRFilterProcessor */ OmniDSP::create_processor(
    const std::vector<IIRFilterCoef>& sos_coeffs) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  OmniExpected<std::unique_ptr<Abstract::IIRFilterProcessorImpl<T_Real>>> /* TODO: Abstract::IIRFilterProcessorImpl */ pimpl_expected;
  if constexpr (std::is_same_v<T_Real, F32>) {
      pimpl_expected = pimpl_->create_iir_filter_plan_impl_f32(sos_coeffs); // TODO: create_iir_filter_processor_impl_f32
  } else if constexpr (std::is_same_v<T_Real, F64>) {
      pimpl_expected = pimpl_->create_iir_filter_plan_impl_f64(sos_coeffs); // TODO: create_iir_filter_processor_impl_f64
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
  if (!pimpl_expected) return std::unexpected(pimpl_expected.error());
  auto processor = IIRFilterProcessor<T_Real>::create_from_impl(std::move(pimpl_expected.value())); // TODO: IIRFilterProcessor<T_Real>
  if (!processor) return std::unexpected(Status::Failure);
  return processor;
}

// ResampleProcessor (stateful) from ResampleParams
template <typename T_Real>
[[nodiscard]] inline OmniExpected<std::unique_ptr<ResampleProcessor<T_Real>>> /* TODO: ResampleProcessor */ OmniDSP::create_processor(
    const ResampleParams& params) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  auto spec_expected = Utils::create_spec(params);
  if (!spec_expected) {
      return std::unexpected(spec_expected.error());
  }
  return create_processor<T_Real>(spec_expected.value());
}

// ResampleProcessor (stateful) from Design::Resample
template <typename T_Real>
[[nodiscard]] inline OmniExpected<std::unique_ptr<ResampleProcessor<T_Real>>> /* TODO: ResampleProcessor */ OmniDSP::create_processor(
    const Design::Resample& spec) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  OmniExpected<std::unique_ptr<Abstract::ResampleProcessorImpl<T_Real>>> /* TODO: Abstract::ResampleProcessorImpl */ pimpl_expected;
  if constexpr (std::is_same_v<T_Real, F32>) {
      pimpl_expected = pimpl_->create_resample_plan_impl_f32(spec); // TODO: create_resample_processor_impl_f32
  } else if constexpr (std::is_same_v<T_Real, F64>) {
      pimpl_expected = pimpl_->create_resample_plan_impl_f64(spec); // TODO: create_resample_processor_impl_f64
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
  if (!pimpl_expected) return std::unexpected(pimpl_expected.error());
  auto processor = ResampleProcessor<T_Real>::create_from_impl(std::move(pimpl_expected.value())); // TODO: ResampleProcessor<T_Real>
  if (!processor) return std::unexpected(Status::Failure);
  return processor;
}


// CQTProcessor (stateful) from CQTParams
template <typename T_Real>
[[nodiscard]] inline OmniExpected<std::unique_ptr<CQTProcessor<T_Real>>> /* TODO: CQTProcessor */ OmniDSP::create_processor(
    const CQTParams& params) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  auto spec_expected = Utils::create_spec(params);
  if (!spec_expected) {
      return std::unexpected(spec_expected.error());
  }
  return create_processor<T_Real>(spec_expected.value());
}

// CQTProcessor (stateful) from Design::CQT
template <typename T_Real>
[[nodiscard]] inline OmniExpected<std::unique_ptr<CQTProcessor<T_Real>>> /* TODO: CQTProcessor */ OmniDSP::create_processor(
    const Design::CQT& spec) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<T_Real>>> /* TODO: Abstract::CQTProcessorImpl */ pimpl_expected;
  if constexpr (std::is_same_v<T_Real, F32>) {
      pimpl_expected = pimpl_->create_cqt_plan_impl_f32(spec); // TODO: create_cqt_processor_impl_f32
  } else if constexpr (std::is_same_v<T_Real, F64>) {
      pimpl_expected = pimpl_->create_cqt_plan_impl_f64(spec); // TODO: create_cqt_processor_impl_f64
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
  if (!pimpl_expected) return std::unexpected(pimpl_expected.error());
  auto processor = CQTProcessor<T_Real>::create_from_impl(std::move(pimpl_expected.value())); // TODO: CQTProcessor<T_Real>
  if (!processor) return std::unexpected(Status::Failure);
  return processor;
}


// --- Filter Design Method Definitions ---
template <typename T>
[[nodiscard]] inline OmniExpected<FIRCoefs<T>> OmniDSP::design_fir_filter(
    const FIRFilterParams& params) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  auto spec_expected = Utils::create_spec(params);
  if (!spec_expected) {
      return std::unexpected(spec_expected.error());
  }
  if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->design_fir_filter_f32(spec_expected.value());
  } else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->design_fir_filter_f64(spec_expected.value());
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
}

template <typename T_Real>
[[nodiscard]] inline OmniExpected<std::vector<IIRFilterCoef>> OmniDSP::design_iir_filter(
    const IIRFilterParams& params) const {
  if (!pimpl_) return std::unexpected(Status::NotInitialized);
  auto spec_expected = Utils::create_spec(params);
  if (!spec_expected) {
      return std::unexpected(spec_expected.error());
  }
  if constexpr (std::is_same_v<T_Real, F32>) {
      return pimpl_->design_iir_filter_f32(spec_expected.value());
  } else if constexpr (std::is_same_v<T_Real, F64>) {
      return pimpl_->design_iir_filter_f64(spec_expected.value());
  } else {
      return std::unexpected(Status::UnsupportedFeature);
  }
}

} // namespace OmniDSP

#endif // OMNIDSP_TPP
