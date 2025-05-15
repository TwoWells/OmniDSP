/**
 * @file omnidsp.tpp
 * @brief Template implementations for the OmniDSP class methods.
 * @details This file is typically included at the end of omnidsp.hpp.
 */

#ifndef OMNIDSP_TPP
#define OMNIDSP_TPP

#include "OmniDSP/omnidsp.hpp" // Primary header
#include "OmniDSP/design.hpp"

// Include public Plan and Processor headers
#include "OmniDSP/plan/fft.hpp"
#include "OmniDSP/plan/convolution.hpp"
#include "OmniDSP/processor/cqt.hpp"
#include "OmniDSP/processor/fir_filter.hpp"
#include "OmniDSP/processor/iir_filter.hpp"
#include "OmniDSP/processor/resample.hpp"

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
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if constexpr (std::is_same_v<typename Utils::GetRealType<T>, F32>) {
        return pimpl_->convolve_f32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<typename Utils::GetRealType<T>, F64>) {
        return pimpl_->convolve_f64(input, kernel, type, method);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::convolve (real): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported real type for convolve");
        return std::unexpected(OmniStatus::UnsupportedFeature);
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
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if constexpr (std::is_same_v<typename Utils::GetComplexType<T>, C32>) {
        return pimpl_->convolve_c32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<typename Utils::GetComplexType<T>, C64>) {
        return pimpl_->convolve_c64(input, kernel, type, method);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::convolve (complex): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported complex type for convolve");
        return std::unexpected(OmniStatus::UnsupportedFeature);
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
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if constexpr (std::is_same_v<typename Utils::GetRealType<T>, F32>) {
        return pimpl_->correlate_f32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<typename Utils::GetRealType<T>, F64>) {
        return pimpl_->correlate_f64(input, kernel, type, method);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::correlate (real): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported real type for correlate");
        return std::unexpected(OmniStatus::UnsupportedFeature);
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
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if constexpr (std::is_same_v<typename Utils::GetComplexType<T>, C32>) {
        return pimpl_->correlate_c32(input, kernel, type, method);
    } else if constexpr (std::is_same_v<typename Utils::GetComplexType<T>, C64>) {
        return pimpl_->correlate_c64(input, kernel, type, method);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::correlate (complex): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported complex type for correlate");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }
}


template <typename T>
OmniExpected<std::vector<typename Utils::GetComplexType<T>>> OmniDSP::fft(
    const std::vector<typename Utils::GetComplexType<T>> &signal) const {
    static_assert(Utils::IsComplex_v<T>, "FFT input must be complex.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::fft: pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if constexpr (std::is_same_v<T, C32>) {
        return pimpl_->fft_c32(signal);
    } else if constexpr (std::is_same_v<T, C64>) {
        return pimpl_->fft_c64(signal);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::fft: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported complex type for fft");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<typename Utils::GetComplexType<T>>> OmniDSP::ifft(
    const std::vector<typename Utils::GetComplexType<T>> &spectrum) const {
    static_assert(Utils::IsComplex_v<T>, "IFFT input must be complex.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::ifft: pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if constexpr (std::is_same_v<T, C32>) {
        return pimpl_->ifft_c32(spectrum);
    } else if constexpr (std::is_same_v<T, C64>) {
        return pimpl_->ifft_c64(spectrum);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::ifft: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported complex type for ifft");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<typename Utils::GetComplexType<T>>> OmniDSP::rfft(
    const std::vector<typename Utils::GetRealType<T>> &signal) const {
    static_assert(!Utils::IsComplex_v<T>, "RFFT input must be real.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::rfft: pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if constexpr (std::is_same_v<T, F32>) {
        return pimpl_->rfft_f32(signal);
    } else if constexpr (std::is_same_v<T, F64>) {
        return pimpl_->rfft_f64(signal);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::rfft: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported real type for rfft");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }
}

template <typename T_Real>
OmniExpected<std::vector<T_Real>> OmniDSP::irfft(
    const std::vector<typename Utils::GetComplexType<T_Real>>& input, size_t output_length) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "IRFFT output must be real.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::irfft: pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if constexpr (std::is_same_v<T_Real, F32>) {
        return pimpl_->irfft_c32(input, output_length);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        return pimpl_->irfft_c64(input, output_length);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::irfft: Unsupported data type for output.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for irfft output");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }
}

template <typename T>
OmniExpected<std::vector<T>> OmniDSP::generate_window(const WindowSetup& setup) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::generate_window: pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    if (setup.length < 0) {
        // spdlog::get("OmniDSP")->error("OmniDSP::generate_window: Window length cannot be negative.");
        return std::unexpected(OmniStatus::InvalidArgument);
    }
    std::vector<T> coeffs(static_cast<size_t>(setup.length));
    OmniStatus status = OmniStatus::Failure;

    if constexpr (std::is_same_v<T, F32>) {
        switch (setup.type) {
            case Type::Window::Bartlett:    status = pimpl_->bartlett_window_f32(coeffs.size(), coeffs); break;
            case Type::Window::Blackman:    status = pimpl_->blackman_window_f32(coeffs.size(), coeffs); break;
            case Type::Window::Flattop:     status = pimpl_->flattop_window_f32(coeffs.size(), coeffs); break;
            case Type::Window::Gaussian:
                status = pimpl_->gaussian_window_f32(coeffs.size(), setup.params.has_value() && setup.params.value().count("sigma") ? setup.params.value().at("sigma") : 0.25, coeffs);
                break;
            case Type::Window::Hamming:     status = pimpl_->hamming_window_f32(coeffs.size(), coeffs); break;
            case Type::Window::Hann:        status = pimpl_->hann_window_f32(coeffs.size(), coeffs); break;
            case Type::Window::Kaiser:
                status = pimpl_->kaiser_window_f32(coeffs.size(), setup.params.has_value() && setup.params.value().count("beta") ? setup.params.value().at("beta") : 0.0, coeffs);
                break;
            case Type::Window::Rectangular: status = pimpl_->rectangular_window_f32(coeffs.size(), coeffs); break;
            case Type::Window::Triangular:  status = pimpl_->triangular_window_f32(coeffs.size(), coeffs); break;
            default:                      status = OmniStatus::UnsupportedFeature; break;
        }
    } else if constexpr (std::is_same_v<T, F64>) {
         switch (setup.type) {
            case Type::Window::Bartlett:    status = pimpl_->bartlett_window_f64(coeffs.size(), coeffs); break;
            case Type::Window::Blackman:    status = pimpl_->blackman_window_f64(coeffs.size(), coeffs); break;
            case Type::Window::Flattop:     status = pimpl_->flattop_window_f64(coeffs.size(), coeffs); break;
            case Type::Window::Gaussian:
                status = pimpl_->gaussian_window_f64(coeffs.size(), setup.params.has_value() && setup.params.value().count("sigma") ? setup.params.value().at("sigma") : 0.25, coeffs);
                break;
            case Type::Window::Hamming:     status = pimpl_->hamming_window_f64(coeffs.size(), coeffs); break;
            case Type::Window::Hann:        status = pimpl_->hann_window_f64(coeffs.size(), coeffs); break;
            case Type::Window::Kaiser:
                status = pimpl_->kaiser_window_f64(coeffs.size(), setup.params.has_value() && setup.params.value().count("beta") ? setup.params.value().at("beta") : 0.0, coeffs);
                break;
            case Type::Window::Rectangular: status = pimpl_->rectangular_window_f64(coeffs.size(), coeffs); break;
            case Type::Window::Triangular:  status = pimpl_->triangular_window_f64(coeffs.size(), coeffs); break;
            default:                      status = OmniStatus::UnsupportedFeature; break;
        }
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::generate_window: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported type for generate_window");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (status != OmniStatus::Success) {
        // spdlog::get("OmniDSP")->error("OmniDSP::generate_window: Backend failed to generate window. OmniStatus: {}", static_cast<int>(status));
        return std::unexpected(status);
    }
    return coeffs;
}

// Deprecated window functions
template <typename T> OmniExpected<std::vector<T>> OmniDSP::bartlett_window(size_t length) const { return generate_window<T>(WindowSetup(Type::Window::Bartlett, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::blackman_window(size_t length) const { return generate_window<T>(WindowSetup(Type::Window::Blackman, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::flattop_window(size_t length) const { return generate_window<T>(WindowSetup(Type::Window::Flattop, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::gaussian_window(size_t length, double stddev) const { return generate_window<T>(WindowSetup(Type::Window::Gaussian, static_cast<int>(length), WindowParams{{"sigma", stddev}})); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::hamming_window(size_t length) const { return generate_window<T>(WindowSetup(Type::Window::Hamming, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::hann_window(size_t length) const { return generate_window<T>(WindowSetup(Type::Window::Hann, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::kaiser_window(size_t length, double beta) const { return generate_window<T>(WindowSetup(Type::Window::Kaiser, static_cast<int>(length), WindowParams{{"beta", beta}})); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::rectangular_window(size_t length) const { return generate_window<T>(WindowSetup(Type::Window::Rectangular, static_cast<int>(length))); }
template <typename T> OmniExpected<std::vector<T>> OmniDSP::triangular_window(size_t length) const { return generate_window<T>(WindowSetup(Type::Window::Triangular, static_cast<int>(length))); }


//--------------------------------------------------------------------------
// OmniDSP Plan Factory Methods (Template Implementations)
//--------------------------------------------------------------------------

template <typename T_Complex>
OmniExpected<std::unique_ptr<Plan::FFT<T_Complex>>>
OmniDSP::create_plan(const Params::FFT& params) const {
    static_assert(Utils::IsComplex_v<T_Complex>, "Plan::FFT requires a complex type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Params::FFT): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }

    OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>> plan_impl_expected;
    if constexpr (std::is_same_v<T_Complex, C32>) {
        plan_impl_expected = pimpl_->create_fft_plan_impl_c32(params.length_);
    } else if constexpr (std::is_same_v<T_Complex, C64>) {
        plan_impl_expected = pimpl_->create_fft_plan_impl_c64(params.length_);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Params::FFT): Unsupported complex type.");
        static_assert(always_false<T_Complex>::value, "Unsupported complex type for Plan::FFT");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!plan_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Params::FFT): Failed to create FFTPlanImpl. OmniStatus: {}", static_cast<int>(plan_impl_expected.error()));
        return std::unexpected(plan_impl_expected.error());
    }

    auto plan = Plan::FFT<T_Complex>::create_from_impl(std::move(plan_impl_expected.value()));
    if (!plan) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Params::FFT): Failed to wrap FFTPlanImpl.");
        return std::unexpected(OmniStatus::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created Plan::FFT from Params::FFT.");
    return plan;
}

template <typename T_Real>
OmniExpected<std::unique_ptr<Plan::RFFT<T_Real>>>
OmniDSP::create_plan(const Params::RFFT& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "Plan::RFFT requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Params::RFFT): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }

    OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>> plan_impl_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        plan_impl_expected = pimpl_->create_rfft_plan_impl_f32(params.length_);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        plan_impl_expected = pimpl_->create_rfft_plan_impl_f64(params.length_);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Params::RFFT): Unsupported real type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for Plan::RFFT");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!plan_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Params::RFFT): Failed to create RFFTPlanImpl. OmniStatus: {}", static_cast<int>(plan_impl_expected.error()));
        return std::unexpected(plan_impl_expected.error());
    }

    auto plan = Plan::RFFT<T_Real>::create_from_impl(std::move(plan_impl_expected.value()));
    if (!plan) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Params::RFFT): Failed to wrap RFFTPlanImpl.");
        return std::unexpected(OmniStatus::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created Plan::RFFT from Params::RFFT.");
    return plan;
}

// Plan::Convolution Factory - Updated
template <typename T> // T is F32, F64, C32, or C64
[[nodiscard]] OmniExpected<std::unique_ptr<Plan::Convolution<T>>> OmniDSP::create_plan(
    const Params::Convolution& params,
    const std::vector<T>& kernel_coeffs) const
{
    if (!pimpl_) return std::unexpected(OmniStatus::NotInitialized);
    OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<T>>> pimpl_expected;
    std::span<const T> kernel_coeffs_span(kernel_coeffs);

    if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected = pimpl_->create_convolution_plan_impl_f32(params, kernel_coeffs_span);
    } else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected = pimpl_->create_convolution_plan_impl_f64(params, kernel_coeffs_span);
    } else if constexpr (std::is_same_v<T, C32>) {
        pimpl_expected = pimpl_->create_convolution_plan_impl_c32(params, kernel_coeffs_span);
    } else if constexpr (std::is_same_v<T, C64>) {
        pimpl_expected = pimpl_->create_convolution_plan_impl_c64(params, kernel_coeffs_span);
    } else {
        static_assert(always_false<T>::value, "Unsupported type for Plan::Convolution.");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
    }
    auto plan = Plan::Convolution<T>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
        spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Plan::Convolution): Failed to create public plan from impl.");
        return std::unexpected(OmniStatus::Failure);
    }
    return plan;
}

// Plan::Correlation Factory - Updated
template <typename T> // T is F32, F64, C32, or C64
[[nodiscard]] OmniExpected<std::unique_ptr<Plan::Correlation<T>>> OmniDSP::create_plan(
    const Params::Correlation& params,
    const std::vector<T>& template_coeffs) const // Renamed kernel_coeffs to template_coeffs for clarity
{
    if (!pimpl_) return std::unexpected(OmniStatus::NotInitialized);
    OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<T>>> pimpl_expected;
    std::span<const T> template_coeffs_span(template_coeffs);

    if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected = pimpl_->create_correlation_plan_impl_f32(params, template_coeffs_span);
    } else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected = pimpl_->create_correlation_plan_impl_f64(params, template_coeffs_span);
    } else if constexpr (std::is_same_v<T, C32>) {
        pimpl_expected = pimpl_->create_correlation_plan_impl_c32(params, template_coeffs_span);
    } else if constexpr (std::is_same_v<T, C64>) {
        pimpl_expected = pimpl_->create_correlation_plan_impl_c64(params, template_coeffs_span);
    } else {
        static_assert(always_false<T>::value, "Unsupported type for Plan::Correlation.");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
    }
    auto plan = Plan::Correlation<T>::create_from_impl(std::move(pimpl_expected.value()));
     if (!plan) {
        spdlog::get("OmniDSP")->error("OmniDSP::create_plan (Plan::Correlation): Failed to create public plan from impl.");
        return std::unexpected(OmniStatus::Failure);
    }
    return plan;
}




//--------------------------------------------------------------------------
// OmniDSP Processor Factory Methods (Template Implementations)
//--------------------------------------------------------------------------

// --- FIRFilterProcessor ---
template <typename T>
OmniExpected<std::unique_ptr<Processor::FIRFilter<T>>>
OmniDSP::create_processor(const Params::FIRFilter& params) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::FIRFilter): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }

    auto design_expected = Design::create(params);
    if (!design_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::FIRFilter): Failed to create design from params. OmniStatus: {}", static_cast<int>(design_expected.error()));
        return std::unexpected(design_expected.error());
    }
    const auto& design = design_expected.value();

    OmniExpected<Coefs::FIRFilter<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, F32>) {
        coeffs_expected = pimpl_->design_fir_filter_f32(design);
    } else if constexpr (std::is_same_v<T, F64>) {
        coeffs_expected = pimpl_->design_fir_filter_f64(design);
    } else if constexpr (std::is_same_v<T, C32>) {
         coeffs_expected = pimpl_->design_fir_filter_c32(design);
    } else if constexpr (std::is_same_v<T, C64>) {
         coeffs_expected = pimpl_->design_fir_filter_c64(design);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::FIRFilter): Unsupported data type for design.");
        static_assert(always_false<T>::value, "Unsupported type for FIRFilter design");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!coeffs_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::FIRFilter): Failed to design FIR filter. OmniStatus: {}", static_cast<int>(coeffs_expected.error()));
        return std::unexpected(coeffs_expected.error());
    }

    // Delegate to the Coefs overload
    return create_processor(coeffs_expected.value());
}

template <typename T>
OmniExpected<std::unique_ptr<Processor::FIRFilter<T>>>
OmniDSP::create_processor(const Coefs::FIRFilter<T>& coeffs) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Coefs::FIRFilter): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
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
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Coefs::FIRFilter): Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported type for Processor::FIRFilter");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!processor_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Coefs::FIRFilter): Failed to create FIRFilterProcessorImpl. OmniStatus: {}", static_cast<int>(processor_impl_expected.error()));
        return std::unexpected(processor_impl_expected.error());
    }

    auto processor = Processor::FIRFilter<T>::create_from_impl(std::move(processor_impl_expected.value()));
    if (!processor) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Coefs::FIRFilter): Failed to wrap FIRFilterProcessorImpl.");
        return std::unexpected(OmniStatus::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created Processor::FIRFilter from Coefs::FIRFilter.");
    return processor;
}

// --- Processor::IIRFilter ---
template <typename T_Real>
OmniExpected<std::unique_ptr<Processor::IIRFilter<T_Real>>>
OmniDSP::create_processor(const Params::IIRFilter& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "Processor::IIRFilter requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::IIRFilter): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }

    auto design_expected = Design::create(params);
    if (!design_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::IIRFilter): Failed to create design from params. OmniStatus: {}", static_cast<int>(design_expected.error()));
        return std::unexpected(design_expected.error());
    }
    const auto& design = design_expected.value();

    OmniExpected<Coefs::IIRFilterSOS> sos_coeffs_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        sos_coeffs_expected = pimpl_->design_iir_filter_f32(design);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        sos_coeffs_expected = pimpl_->design_iir_filter_f64(design);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::IIRFilter): Unsupported data type for design.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for IIRFilter design");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!sos_coeffs_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::IIRFilter): Failed to design IIR filter. OmniStatus: {}", static_cast<int>(sos_coeffs_expected.error()));
        return std::unexpected(sos_coeffs_expected.error());
    }

    // CORRECTED: Explicitly provide T_Real to the Coefs overload
    return create_processor<T_Real>(sos_coeffs_expected.value());
}

template <typename T_Real>
OmniExpected<std::unique_ptr<Processor::IIRFilter<T_Real>>>
OmniDSP::create_processor(const Coefs::IIRFilterSOS& sos_coeffs) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "Processor::IIRFilter requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Coefs::SOS): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }

    OmniExpected<std::unique_ptr<Abstract::IIRFilterProcessorImpl<T_Real>>> processor_impl_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        processor_impl_expected = pimpl_->create_iir_filter_processor_impl_f32(sos_coeffs);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        processor_impl_expected = pimpl_->create_iir_filter_processor_impl_f64(sos_coeffs);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Coefs::SOS): Unsupported data type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for Processor::IIRFilter");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!processor_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Coefs::SOS): Failed to create IIRFilterProcessorImpl. OmniStatus: {}", static_cast<int>(processor_impl_expected.error()));
        return std::unexpected(processor_impl_expected.error());
    }

    auto processor = Processor::IIRFilter<T_Real>::create_from_impl(std::move(processor_impl_expected.value()));
    if (!processor) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Coefs::SOS): Failed to wrap IIRFilterProcessorImpl.");
        return std::unexpected(OmniStatus::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created Processor::IIRFilter from Coefs::SOS.");
    return processor;
}

// --- Processor::Resample ---
template <typename T_Real>
OmniExpected<std::unique_ptr<Processor::Resample<T_Real>>>
OmniDSP::create_processor(const Params::Resample& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "Processor::Resample requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::Resample): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    auto design_expected = Design::create(params);
    if (!design_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::Resample): Failed to create design from params. OmniStatus: {}", static_cast<int>(design_expected.error()));
        return std::unexpected(design_expected.error());
    }
    return create_processor<T_Real>(design_expected.value()); // Delegate
}

template <typename T_Real>
OmniExpected<std::unique_ptr<Processor::Resample<T_Real>>>
OmniDSP::create_processor(const Design::Resample& design) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "Processor::Resample requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::Resample): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    OmniExpected<std::unique_ptr<Abstract::ResampleProcessorImpl<T_Real>>> processor_impl_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        processor_impl_expected = pimpl_->create_resample_processor_impl_f32(design);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        processor_impl_expected = pimpl_->create_resample_processor_impl_f64(design);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::Resample): Unsupported data type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for Processor::Resample");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!processor_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::Resample): Failed to create ResampleProcessorImpl. OmniStatus: {}", static_cast<int>(processor_impl_expected.error()));
        return std::unexpected(processor_impl_expected.error());
    }
    auto processor = Processor::Resample<T_Real>::create_from_impl(std::move(processor_impl_expected.value()));
    if (!processor) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::Resample): Failed to wrap ResampleProcessorImpl.");
        return std::unexpected(OmniStatus::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created Processor::Resample from Design::Resample.");
    return processor;
}

// --- Processor::CQT ---
template <typename T_Real>
OmniExpected<std::unique_ptr<Processor::CQT<T_Real>>>
OmniDSP::create_processor(const Params::CQT& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "Processor::CQT requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::CQT): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    auto design_expected = Design::create(params);
    if (!design_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Params::CQT): Failed to create design from params. OmniStatus: {}", static_cast<int>(design_expected.error()));
        return std::unexpected(design_expected.error());
    }
    return create_processor<T_Real>(design_expected.value()); // Delegate
}

template <typename T_Real>
OmniExpected<std::unique_ptr<Processor::CQT<T_Real>>>
OmniDSP::create_processor(const Design::CQT& design) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "Processor::CQT requires a real type.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::CQT): pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<T_Real>>> processor_impl_expected;
    if constexpr (std::is_same_v<T_Real, F32>) {
        processor_impl_expected = pimpl_->create_cqt_processor_impl_f32(design);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        processor_impl_expected = pimpl_->create_cqt_processor_impl_f64(design);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::CQT): Unsupported data type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for Processor::CQT");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!processor_impl_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::CQT): Failed to create CQTProcessorImpl. OmniStatus: {}", static_cast<int>(processor_impl_expected.error()));
        return std::unexpected(processor_impl_expected.error());
    }
    auto processor = Processor::CQT<T_Real>::create_from_impl(std::move(processor_impl_expected.value()));
    if (!processor) {
        // spdlog::get("OmniDSP")->error("OmniDSP::create_processor (Design::CQT): Failed to wrap CQTProcessorImpl.");
        return std::unexpected(OmniStatus::Failure);
    }
    // spdlog::get("OmniDSP")->debug("Successfully created Processor::CQT from Design::CQT.");
    return processor;
}


//--------------------------------------------------------------------------
// OmniDSP Filter Design Methods (Template Implementations)
//--------------------------------------------------------------------------
template <typename T>
OmniExpected<Coefs::FIRFilter<T>> OmniDSP::design_fir_filter(const Params::FIRFilter& params) const {
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_fir_filter: pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    auto design_expected = Design::create(params);
    if (!design_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_fir_filter: Failed to create design from params. OmniStatus: {}", static_cast<int>(design_expected.error()));
        return std::unexpected(design_expected.error());
    }
    const auto& design = design_expected.value();

    if constexpr (std::is_same_v<T, F32>) {
        return pimpl_->design_fir_filter_f32(design);
    } else if constexpr (std::is_same_v<T, F64>) {
        return pimpl_->design_fir_filter_f64(design);
    } else if constexpr (std::is_same_v<T, C32>) {
         return pimpl_->design_fir_filter_c32(design);
    } else if constexpr (std::is_same_v<T, C64>) {
         return pimpl_->design_fir_filter_c64(design);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_fir_filter: Unsupported data type.");
        static_assert(always_false<T>::value, "Unsupported type for design_fir_filter");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }
}

template <typename T_Real>
OmniExpected<Coefs::IIRFilterSOS> OmniDSP::design_iir_filter(const Params::IIRFilter& params) const {
    static_assert(!Utils::IsComplex_v<T_Real>, "design_iir_filter requires a real type for T_Real.");
    if (!pimpl_) {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_iir_filter: pimpl_ is null.");
        return std::unexpected(OmniStatus::NotInitialized);
    }
    auto design_expected = Design::create(params);
    if (!design_expected) {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_iir_filter: Failed to create design from params. OmniStatus: {}", static_cast<int>(design_expected.error()));
        return std::unexpected(design_expected.error());
    }
    const auto& design = design_expected.value();

    if constexpr (std::is_same_v<T_Real, F32>) {
        return pimpl_->design_iir_filter_f32(design);
    } else if constexpr (std::is_same_v<T_Real, F64>) {
        return pimpl_->design_iir_filter_f64(design);
    } else {
        // spdlog::get("OmniDSP")->error("OmniDSP::design_iir_filter: Unsupported data type.");
        static_assert(always_false<T_Real>::value, "Unsupported real type for design_iir_filter");
        return std::unexpected(OmniStatus::UnsupportedFeature);
    }
}

} // namespace OmniDSP

#endif // OMNIDSP_TPP
