/**
 * @file omnidsp.tpp
 * @brief Template implementations for the OmniDSP class methods.
 * @details This file is typically included at the end of omnidsp.hpp.
 */

#ifndef OMNIDSP_TPP
#define OMNIDSP_TPP

#include "OmniDSP/omnidsp.hpp" // Primary header
#include "OmniDSP/utils.hpp"   // For Utils::create_spec

// Include public Plan and Processor headers
#include "OmniDSP/fft.hpp"
#include "OmniDSP/convolution.hpp"
#include "OmniDSP/cqt.hpp"
#include "OmniDSP/resample.hpp"
#include "OmniDSP/fir_filter.hpp"
#include "OmniDSP/iir_filter.hpp"

// Include Params headers for use in create_processor/create_plan
#include "OmniDSP/params/fft.hpp"
#include "OmniDSP/params/convolution.hpp"
#include "OmniDSP/params/cqt.hpp"
#include "OmniDSP/params/resample.hpp"
#include "OmniDSP/params/fir_filter.hpp"
#include "OmniDSP/params/iir_filter.hpp"

// Include Coefs/Design headers for use in create_processor/create_plan
#include "OmniDSP/coefs/fir_filter.hpp"
#include "OmniDSP/coefs/iir_filter.hpp"
#include "OmniDSP/design/cqt.hpp"
#include "OmniDSP/design/resample.hpp"
#include "OmniDSP/design/fir_filter.hpp" // For Design::FIRFilter
#include "OmniDSP/design/iir_filter.hpp" // For Design::IIRFilter


// #include <spdlog/spdlog.h> // For logging (optional, can be commented out)
#include <vector>
#include <memory>
#include <type_traits> // For std::is_same_v

namespace OmniDSP {

//--------------------------------------------------------------------------
// OmniDSP One-Off DSP Operations (Template Implementations)
//--------------------------------------------------------------------------

template <typename T>
OmniExpected<std::vector<typename Utils::GetRealType<T>>> OmniDSP::convolve(
    const std::vector<typename Utils::GetRealType<T>>& input,
    const std::vector<typename Utils::GetRealType<T>>& kernel,
    ConvolutionType type,
    ConvolutionMethod method) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::convolve (real): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if constexpr (std::is_same_v<typename Utils::GetRealType<T>, F32>) {
        return pimpl_->convolve_f32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<typename Utils::GetRealType<T>, F64>) {
        return pimpl_->convolve_f64(input, kernel, type, method);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::convolve (real): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported real type for convolve");
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<typename Utils::GetComplexType<T>>> OmniDSP::convolve(
    const std::vector<typename Utils::GetComplexType<T>>& input,
    const std::vector<typename Utils::GetComplexType<T>>& kernel,
    ConvolutionType type,
    ConvolutionMethod method) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::convolve (complex): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if constexpr (std::is_same_v<typename Utils::GetComplexType<T>, C32>) {
        return pimpl_->convolve_c32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<typename Utils::GetComplexType<T>, C64>) {
        return pimpl_->convolve_c64(input, kernel, type, method);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::convolve (complex): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported complex type for convolve");
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<typename Utils::GetRealType<T>>> OmniDSP::correlate(
    const std::vector<typename Utils::GetRealType<T>>& input,
    const std::vector<typename Utils::GetRealType<T>>& kernel,
    ConvolutionType type,
    ConvolutionMethod method) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::correlate (real): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if constexpr (std::is_same_v<typename Utils::GetRealType<T>, F32>) {
        return pimpl_->correlate_f32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<typename Utils::GetRealType<T>, F64>) {
        return pimpl_->correlate_f64(input, kernel, type, method);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::correlate (real): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported real type for correlate");
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<typename Utils::GetComplexType<T>>> OmniDSP::correlate(
    const std::vector<typename Utils::GetComplexType<T>>& input,
    const std::vector<typename Utils::GetComplexType<T>>& kernel,
    ConvolutionType type,
    ConvolutionMethod method) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::correlate (complex): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if constexpr (std::is_same_v<typename Utils::GetComplexType<T>, C32>) {
        return pimpl_->correlate_c32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<typename Utils::GetComplexType<T>, C64>) {
        return pimpl_->correlate_c64(input, kernel, type, method);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::correlate (complex): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported complex type for correlate");
        return std::unexpected(Status::UnsupportedFeature);
    }
}


template <typename T>
OmniExpected<std::vector<typename Utils::GetComplexType<T>>> OmniDSP::fft(
    const std::vector<typename Utils::GetComplexType<T>> &signal) const {
    static_assert(Utils::IsComplex_v<T>, "FFT input must be complex.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::fft: pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if constexpr (std::is_same_v<T, C32>) {
        return pimpl_->fft_c32(signal);
    } else if constexpr (std::is_same_v<T, C64>) {
        return pimpl_->fft_c64(signal);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::fft: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported complex type for fft");
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<typename Utils::GetComplexType<T>>> OmniDSP::ifft(
    const std::vector<typename Utils::GetComplexType<T>> &spectrum) const {
    static_assert(Utils::IsComplex_v<T>, "IFFT input must be complex.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::ifft: pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if constexpr (std::is_same_v<T, C32>) {
        return pimpl_->ifft_c32(spectrum);
    } else if constexpr (std::is_same_v<T, C64>) {
        return pimpl_->ifft_c64(spectrum);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::ifft: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported complex type for ifft");
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<typename Utils::GetComplexType<T>>> OmniDSP::rfft(
    const std::vector<typename Utils::GetRealType<T>> &signal) const {
    static_assert(!Utils::IsComplex_v<T>, "RFFT input must be real.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::rfft: pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if constexpr (std::is_same_v<T, F32>) {
        return pimpl_->rfft_f32(signal);
    } else if constexpr (std::is_same_v<T, F64>) {
        return pimpl_->rfft_f64(signal);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::rfft: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported real type for rfft");
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T_Real>
OmniExpected<std::vector<T_Real>> OmniDSP::irfft(
    const std::vector<typename Utils::GetComplexType<T_Real>>& input, size_t output_length) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "IRFFT output must be real.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::irfft: pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if constexpr (std::is_same_v<T_Real, F32>) {
        return pimpl_->irfft_c32(input, output_length);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        return pimpl_->irfft_c64(input, output_length);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::irfft: Unsupported data type for output.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for irfft output");
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<T>> OmniDSP::generate_window(const WindowSetup& setup) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::generate_window: pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    if (setup.length < 0) {
        // spdlog::get("OmniDSP")->error("OmniDSP::generate_window: Window length cannot be negative.");
        return std::unexpected(Status::InvalidArgument);
    }
    std::vector<T> coeffs(static_cast<size_t>(setup.length));
    Status status = Status::Failure;

    if constexpr (std::is_same_v<T, F32>) {
        switch (setup.type) {
            case WindowType::Bartlett:    status = pimpl_->bartlett_window_f32(coeffs.size(), coeffs); break;
            case WindowType::Blackman:    status = pimpl_->blackman_window_f32(coeffs.size(), coeffs); break;
            case WindowType::Flattop:     status = pimpl_->flattop_window_f32(coeffs.size(), coeffs); break;
            case WindowType::Gaussian:
                status = pimpl_->gaussian_window_f32(coeffs.size(), setup.params.has_value() && setup.params.value().count("sigma") ? setup.params.value().at("sigma") : 0.25, coeffs);
                break;
            case WindowType::Hamming:     status = pimpl_->hamming_window_f32(coeffs.size(), coeffs); break;
            case WindowType::Hann:        status = pimpl_->hann_window_f32(coeffs.size(), coeffs); break;
            case WindowType::Kaiser:
                status = pimpl_->kaiser_window_f32(coeffs.size(), setup.params.has_value() && setup.params.value().count("beta") ? setup.params.value().at("beta") : 0.0, coeffs);
                break;
            case WindowType::Rectangular: status = pimpl_->rectangular_window_f32(coeffs.size(), coeffs); break;
            case WindowType::Triangular:  status = pimpl_->triangular_window_f32(coeffs.size(), coeffs); break;
            default:                      status = Status::UnsupportedFeature; break;
        }
    } else if constexpr (std::is_same_v<T, F64>) {
         switch (setup.type) {
            case WindowType::Bartlett:    status = pimpl_->bartlett_window_f64(coeffs.size(), coeffs); break;
            case WindowType::Blackman:    status = pimpl_->blackman_window_f64(coeffs.size(), coeffs); break;
            case WindowType::Flattop:     status = pimpl_->flattop_window_f64(coeffs.size(), coeffs); break;
            case WindowType::Gaussian:
                status = pimpl_->gaussian_window_f64(coeffs.size(), setup.params.has_value() && setup.params.value().count("sigma") ? setup.params.value().at("sigma") : 0.25, coeffs);
                break;
            case WindowType::Hamming:     status = pimpl_->hamming_window_f64(coeffs.size(), coeffs); break;
            case WindowType::Hann:        status = pimpl_->hann_window_f64(coeffs.size(), coeffs); break;
            case WindowType::Kaiser:
                status = pimpl_->kaiser_window_f64(coeffs.size(), setup.params.has_value() && setup.params.value().count("beta") ? setup.params.value().at("beta") : 0.0, coeffs);
                break;
            case WindowType::Rectangular: status = pimpl_->rectangular_window_f64(coeffs.size(), coeffs); break;
            case WindowType::Triangular:  status = pimpl_->triangular_window_f64(coeffs.size(), coeffs); break;
            default:                      status = Status::UnsupportedFeature; break;
        }
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::generate_window: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported type for generate_window");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (status != Status::Success) {
        // spdlog::get("OmniDSP")->error("OmniDSP::generate_window: Backend failed to generate window. Status: {}", static_cast<int>(status));
        return std::unexpected(status);
    }
    return coeffs;
}

// Deprecated window functions
template <typename T> OmniExpected<std::vector<T>> OmniDSP::bartlett_window(size_t length) const { return generate_window<T>(WindowSetup(WindowType::Bartlett, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::blackman_window(size_t length) const { return generate_window<T>(WindowSetup(WindowType::Blackman, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::flattop_window(size_t length) const { return generate_window<T>(WindowSetup(WindowType::Flattop, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::gaussian_window(size_t length, double stddev) const { return generate_window<T>(WindowSetup(WindowType::Gaussian, static_cast<int>(length), WindowParams{{"sigma", stddev}})); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::hamming_window(size_t length) const { return generate_window<T>(WindowSetup(WindowType::Hamming, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::hann_window(size_t length) const { return generate_window<T>(WindowSetup(WindowType::Hann, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::kaiser_window(size_t length, double beta) const { return generate_window<T>(WindowSetup(WindowType::Kaiser, static_cast<int>(length), WindowParams{{"beta", beta}})); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::rectangular_window(size_t length) const { return generate_window<T>(WindowSetup(WindowType::Rectangular, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::triangular_window(size_t length) const { return generate_window<T>(WindowSetup(WindowType::Triangular, static_cast<int>(length))); }


//--------------------------------------------------------------------------
// OmniDSP Plan Factory Methods (Template Implementations)
//--------------------------------------------------------------------------

template <typename T_Complex>
OmniExpected<std::unique_ptr<FFTPlan<T_Complex>>>
OmniDSP::create_plan(const FFTParams& params) const {
    static_assert(Utils::IsComplex_v<T_Complex>, "FFTPlan requires a complex type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (FFTParams): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }

    OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>> plan_impl_expected;
    if constexpr (std::is_same_v<T_Complex, C32>) {
        plan_impl_expected = pimpl_->create_fft_plan_impl_c32(params.length_);
    } else if constexpr (std::is_same_v<T_Complex, C64>) {
        plan_impl_expected = pimpl_->create_fft_plan_impl_c64(params.length_);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (FFTParams): Unsupported complex type.");
        static_assert(always_false<T_Complex>::value, "Unsupported complex type for FFTPlan");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!plan_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (FFTParams): Failed to create FFTPlanImpl. Status: {}", static_cast<int>(plan_impl_expected.error()));
        return std::unexpected(plan_impl_expected.error());
    }

    auto plan = FFTPlan<T_Complex>::create_from_impl(std::move(plan_impl_expected.value()));
    if (!plan) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (FFTParams): Failed to wrap FFTPlanImpl.");
        return std::unexpected(Status::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created FFTPlan from FFTParams.");
    return plan;
}

template <typename T_Real>
OmniExpected<std::unique_ptr<RFFTPlan<T_Real>>>
OmniDSP::create_plan(const RFFTParams& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "RFFTPlan requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (RFFTParams): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }

    OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>> plan_impl_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        plan_impl_expected = pimpl_->create_rfft_plan_impl_f32(params.length_);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        plan_impl_expected = pimpl_->create_rfft_plan_impl_f64(params.length_);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (RFFTParams): Unsupported real type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for RFFTPlan");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!plan_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (RFFTParams): Failed to create RFFTPlanImpl. Status: {}", static_cast<int>(plan_impl_expected.error()));
        return std::unexpected(plan_impl_expected.error());
    }

    auto plan = RFFTPlan<T_Real>::create_from_impl(std::move(plan_impl_expected.value()));
    if (!plan) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (RFFTParams): Failed to wrap RFFTPlanImpl.");
        return std::unexpected(Status::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created RFFTPlan from RFFTParams.");
    return plan;
}

template <typename T>
OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
OmniDSP::create_plan(const ConvolutionParams& params, const std::vector<T>& kernel_coeffs) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (ConvolutionParams): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<T>>> plan_impl_expected;
    if constexpr (std::is_same_v<T, F32>) {
        plan_impl_expected = pimpl_->create_convolution_plan_impl_f32(kernel_coeffs, params.type_, params.method_hint_);
    } else if constexpr (std::is_same_v<T, F64>) {
        plan_impl_expected = pimpl_->create_convolution_plan_impl_f64(kernel_coeffs, params.type_, params.method_hint_);
    } else if constexpr (std::is_same_v<T, C32>) {
        plan_impl_expected = pimpl_->create_convolution_plan_impl_c32(kernel_coeffs, params.type_, params.method_hint_);
    } else if constexpr (std::is_same_v<T, C64>) {
        plan_impl_expected = pimpl_->create_convolution_plan_impl_c64(kernel_coeffs, params.type_, params.method_hint_);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (ConvolutionParams): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported type for ConvolutionPlan");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!plan_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (ConvolutionParams): Failed to create ConvolutionPlanImpl. Status: {}", static_cast<int>(plan_impl_expected.error()));
        return std::unexpected(plan_impl_expected.error());
    }

    auto plan = ConvolutionPlan<T>::create_from_impl(std::move(plan_impl_expected.value()));
     if (!plan) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (ConvolutionParams): Failed to wrap ConvolutionPlanImpl.");
        return std::unexpected(Status::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created ConvolutionPlan from ConvolutionParams.");
    return plan;
}


template <typename T>
OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
OmniDSP::create_plan(const CorrelationParams& params, const std::vector<T>& kernel_coeffs) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (CorrelationParams): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<T>>> plan_impl_expected;
     if constexpr (std::is_same_v<T, F32>) {
        plan_impl_expected = pimpl_->create_correlation_plan_impl_f32(kernel_coeffs, params.type_, params.method_hint_);
    } else if constexpr (std::is_same_v<T, F64>) {
        plan_impl_expected = pimpl_->create_correlation_plan_impl_f64(kernel_coeffs, params.type_, params.method_hint_);
    } else if constexpr (std::is_same_v<T, C32>) {
        plan_impl_expected = pimpl_->create_correlation_plan_impl_c32(kernel_coeffs, params.type_, params.method_hint_);
    } else if constexpr (std::is_same_v<T, C64>) {
        plan_impl_expected = pimpl_->create_correlation_plan_impl_c64(kernel_coeffs, params.type_, params.method_hint_);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (CorrelationParams): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported type for CorrelationPlan");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!plan_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (CorrelationParams): Failed to create CorrelationPlanImpl. Status: {}", static_cast<int>(plan_impl_expected.error()));
        return std::unexpected(plan_impl_expected.error());
    }

    auto plan = CorrelationPlan<T>::create_from_impl(std::move(plan_impl_expected.value()));
    if (!plan) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (CorrelationParams): Failed to wrap CorrelationPlanImpl.");
        return std::unexpected(Status::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created CorrelationPlan from CorrelationParams.");
    return plan;
}


//--------------------------------------------------------------------------
// OmniDSP Processor Factory Methods (Template Implementations)
//--------------------------------------------------------------------------

// --- FIRFilterProcessor ---
template <typename T>
OmniExpected<std::unique_ptr<FIRFilterProcessor<T>>>
OmniDSP::create_processor(const Params::FIRFilter& params) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::FIRFilter): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }

    auto spec_expected = Utils::create_spec(params);
    if (!spec_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::FIRFilter): Failed to create spec from params. Status: {}", static_cast<int>(spec_expected.error()));
        return std::unexpected(spec_expected.error());
    }
    const auto& spec = spec_expected.value();

    OmniExpected<FIRCoefs<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, F32>) {
        coeffs_expected = pimpl_->design_fir_filter_f32(spec);
    } else if constexpr (std::is_same_v<T, F64>) {
        coeffs_expected = pimpl_->design_fir_filter_f64(spec);
    } else if constexpr (std::is_same_v<T, C32>) {
         coeffs_expected = pimpl_->design_fir_filter_c32(spec);
    } else if constexpr (std::is_same_v<T, C64>) {
         coeffs_expected = pimpl_->design_fir_filter_c64(spec);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::FIRFilter): Unsupported data type for design.");
        static_assert(always_false<T>::value, "Unsupported type for FIRFilter design");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!coeffs_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::FIRFilter): Failed to design FIR filter. Status: {}", static_cast<int>(coeffs_expected.error()));
        return std::unexpected(coeffs_expected.error());
    }

    // Delegate to the Coefs overload
    return create_processor(coeffs_expected.value());
}

template <typename T>
OmniExpected<std::unique_ptr<FIRFilterProcessor<T>>>
OmniDSP::create_processor(const FIRCoefs<T>& coeffs) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (FIRCoefs): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }

    OmniExpected<std::unique_ptr<Abstract::FIRFilterProcessorImpl<T>>> processor_impl_expected;
    if constexpr (std::is_same_v<T, F32>) {
        processor_impl_expected = pimpl_->create_fir_filter_processor_impl_f32(coeffs);
    } else if constexpr (std::is_same_v<T, F64>) {
        processor_impl_expected = pimpl_->create_fir_filter_processor_impl_f64(coeffs);
    } else if constexpr (std::is_same_v<T, C32>) {
        processor_impl_expected = pimpl_->create_fir_filter_processor_impl_c32(coeffs);
    } else if constexpr (std::is_same_v<T, C64>) {
        processor_impl_expected = pimpl_->create_fir_filter_processor_impl_c64(coeffs);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (FIRCoefs): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported type for FIRFilterProcessor");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!processor_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (FIRCoefs): Failed to create FIRFilterProcessorImpl. Status: {}", static_cast<int>(processor_impl_expected.error()));
        return std::unexpected(processor_impl_expected.error());
    }

    auto processor = FIRFilterProcessor<T>::create_from_impl(std::move(processor_impl_expected.value()));
    if (!processor) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (FIRCoefs): Failed to wrap FIRFilterProcessorImpl.");
        return std::unexpected(Status::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created FIRFilterProcessor from FIRCoefs.");
    return processor;
}

// --- IIRFilterProcessor ---
template <typename T_Real>
OmniExpected<std::unique_ptr<IIRFilterProcessor<T_Real>>>
OmniDSP::create_processor(const IIRFilterParams& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "IIRFilterProcessor requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (IIRFilterParams): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }

    auto spec_expected = Utils::create_spec(params);
    if (!spec_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (IIRFilterParams): Failed to create spec from params. Status: {}", static_cast<int>(spec_expected.error()));
        return std::unexpected(spec_expected.error());
    }
    const auto& spec = spec_expected.value();

    OmniExpected<std::vector<IIRFilterCoef>> sos_coeffs_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        sos_coeffs_expected = pimpl_->design_iir_filter_f32(spec);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        sos_coeffs_expected = pimpl_->design_iir_filter_f64(spec);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (IIRFilterParams): Unsupported data type for design.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for IIRFilter design");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!sos_coeffs_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (IIRFilterParams): Failed to design IIR filter. Status: {}", static_cast<int>(sos_coeffs_expected.error()));
        return std::unexpected(sos_coeffs_expected.error());
    }

    // CORRECTED: Explicitly provide T_Real to the Coefs overload
    return create_processor<T_Real>(sos_coeffs_expected.value());
}

template <typename T_Real>
OmniExpected<std::unique_ptr<IIRFilterProcessor<T_Real>>>
OmniDSP::create_processor(const std::vector<IIRFilterCoef>& sos_coeffs) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "IIRFilterProcessor requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (IIRFilterCoef): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }

    OmniExpected<std::unique_ptr<Abstract::IIRFilterProcessorImpl<T_Real>>> processor_impl_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        processor_impl_expected = pimpl_->create_iir_filter_processor_impl_f32(sos_coeffs);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        processor_impl_expected = pimpl_->create_iir_filter_processor_impl_f64(sos_coeffs);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (IIRFilterCoef): Unsupported data type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for IIRFilterProcessor");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!processor_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (IIRFilterCoef): Failed to create IIRFilterProcessorImpl. Status: {}", static_cast<int>(processor_impl_expected.error()));
        return std::unexpected(processor_impl_expected.error());
    }

    auto processor = IIRFilterProcessor<T_Real>::create_from_impl(std::move(processor_impl_expected.value()));
    if (!processor) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (IIRFilterCoef): Failed to wrap IIRFilterProcessorImpl.");
        return std::unexpected(Status::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created IIRFilterProcessor from IIRFilterCoef.");
    return processor;
}

// --- ResampleProcessor ---
template <typename T_Real>
OmniExpected<std::unique_ptr<ResampleProcessor<T_Real>>>
OmniDSP::create_processor(const ResampleParams& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "ResampleProcessor requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (ResampleParams): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    auto spec_expected = Utils::create_spec(params);
    if (!spec_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (ResampleParams): Failed to create spec from params. Status: {}", static_cast<int>(spec_expected.error()));
        return std::unexpected(spec_expected.error());
    }
    return create_processor<T_Real>(spec_expected.value()); // Delegate
}

template <typename T_Real>
OmniExpected<std::unique_ptr<ResampleProcessor<T_Real>>>
OmniDSP::create_processor(const Design::Resample& spec) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "ResampleProcessor requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::Resample): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    OmniExpected<std::unique_ptr<Abstract::ResampleProcessorImpl<T_Real>>> processor_impl_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        processor_impl_expected = pimpl_->create_resample_processor_impl_f32(spec);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        processor_impl_expected = pimpl_->create_resample_processor_impl_f64(spec);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::Resample): Unsupported data type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for ResampleProcessor");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!processor_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::Resample): Failed to create ResampleProcessorImpl. Status: {}", static_cast<int>(processor_impl_expected.error()));
        return std::unexpected(processor_impl_expected.error());
    }
    auto processor = ResampleProcessor<T_Real>::create_from_impl(std::move(processor_impl_expected.value()));
    if (!processor) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::Resample): Failed to wrap ResampleProcessorImpl.");
        return std::unexpected(Status::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created ResampleProcessor from Design::Resample.");
    return processor;
}

// --- CQTProcessor ---
template <typename T_Real>
OmniExpected<std::unique_ptr<CQTProcessor<T_Real>>>
OmniDSP::create_processor(const CQTParams& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "CQTProcessor requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (CQTParams): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    auto spec_expected = Utils::create_spec(params);
    if (!spec_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (CQTParams): Failed to create spec from params. Status: {}", static_cast<int>(spec_expected.error()));
        return std::unexpected(spec_expected.error());
    }
    return create_processor<T_Real>(spec_expected.value()); // Delegate
}

template <typename T_Real>
OmniExpected<std::unique_ptr<CQTProcessor<T_Real>>>
OmniDSP::create_processor(const Design::CQT& spec) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "CQTProcessor requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::CQT): pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<T_Real>>> processor_impl_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        processor_impl_expected = pimpl_->create_cqt_processor_impl_f32(spec);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        processor_impl_expected = pimpl_->create_cqt_processor_impl_f64(spec);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::CQT): Unsupported data type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for CQTProcessor");
        return std::unexpected(Status::UnsupportedFeature);
    }

    if (!processor_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::CQT): Failed to create CQTProcessorImpl. Status: {}", static_cast<int>(processor_impl_expected.error()));
        return std::unexpected(processor_impl_expected.error());
    }
    auto processor = CQTProcessor<T_Real>::create_from_impl(std::move(processor_impl_expected.value()));
    if (!processor) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::CQT): Failed to wrap CQTProcessorImpl.");
        return std::unexpected(Status::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created CQTProcessor from Design::CQT.");
    return processor;
}


//--------------------------------------------------------------------------
// OmniDSP Filter Design Methods (Template Implementations)
//--------------------------------------------------------------------------
template <typename T>
OmniExpected<FIRCoefs<T>> OmniDSP::design_fir_filter(const Params::FIRFilter& params) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_fir_filter: pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    auto spec_expected = Utils::create_spec(params);
    if (!spec_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_fir_filter: Failed to create spec from params. Status: {}", static_cast<int>(spec_expected.error()));
        return std::unexpected(spec_expected.error());
    }
    const auto& spec = spec_expected.value();

    if constexpr (std::is_same_v<T, F32>) {
        return pimpl_->design_fir_filter_f32(spec);
    } else if constexpr (std::is_same_v<T, F64>) {
        return pimpl_->design_fir_filter_f64(spec);
    } else if constexpr (std::is_same_v<T, C32>) {
         return pimpl_->design_fir_filter_c32(spec);
    } else if constexpr (std::is_same_v<T, C64>) {
         return pimpl_->design_fir_filter_c64(spec);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_fir_filter: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported type for design_fir_filter");
        return std::unexpected(Status::UnsupportedFeature);
    }
}

template <typename T_Real>
OmniExpected<std::vector<IIRFilterCoef>> OmniDSP::design_iir_filter(const IIRFilterParams& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "design_iir_filter requires a real type for T_Real.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_iir_filter: pimpl_ is null.");
        return std::unexpected(Status::NotInitialized);
    }
    auto spec_expected = Utils::create_spec(params);
    if (!spec_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_iir_filter: Failed to create spec from params. Status: {}", static_cast<int>(spec_expected.error()));
        return std::unexpected(spec_expected.error());
    }
    const auto& spec = spec_expected.value();

    if constexpr (std::is_same_v<T_Real, F32>) {
        return pimpl_->design_iir_filter_f32(spec);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        return pimpl_->design_iir_filter_f64(spec);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_iir_filter: Unsupported data type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for design_iir_filter");
        return std::unexpected(Status::UnsupportedFeature);
    }
}


} // namespace OmniDSP

#endif // OMNIDSP_TPP
