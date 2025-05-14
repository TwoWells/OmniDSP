/**
 * @file backend.cpp (dispatcher)
 * @brief Implements the Dispatcher::Backend class methods.
 */

#include "dispatcher/backend.hpp"  // Corresponding header

#include <stdexcept>  // For std::invalid_argument

#include "spdlog/spdlog.h"  // For logging

// Include headers for all Design/Coef types used in method signatures if not
// transitively included
#include "OmniDSP/coefs/fir_filter.hpp"
#include "OmniDSP/coefs/iir_filter.hpp"
#include "OmniDSP/design/cqt.hpp"
#include "OmniDSP/design/fir_filter.hpp"
#include "OmniDSP/design/iir_filter.hpp"
#include "OmniDSP/design/resample.hpp"

namespace OmniDSP::Dispatcher {

  // --- Constructor ---
  Backend::Backend(
      std::unique_ptr<Abstract::Backend> primary_backend,
      std::map<OperationCategory, std::shared_ptr<Abstract::Backend>>
          backend_overrides)
      : primary_backend_(std::move(primary_backend)),
        overrides_(std::move(backend_overrides))
  {
    if (!primary_backend_) {
      // This should be logged by the calling factory (e.g. OmniDSP::create)
      // too. Throwing here ensures the DispatcherBackend is not in an invalid
      // state.
      auto logger = spdlog::get("OmniDSP");  // Attempt to get logger
      if (logger) {
        logger->critical(
            "DispatcherBackend: primary_backend cannot be null during "
            "construction.");
      }
      throw std::invalid_argument(
          "DispatcherBackend: primary_backend cannot be null.");
    }

    auto logger = spdlog::get("OmniDSP");
    if (logger) {  // Check if logger was successfully retrieved
      logger->info(
          "DispatcherBackend created. Primary backend: {}",
          get_backend_name(primary_backend_->get_backend()));
      if (!overrides_.empty()) {
        logger->info("  With the following overrides:");
        for (const auto& pair : overrides_) {
          if (pair.second) {  // Ensure the shared_ptr is not null
            logger->info(
                "    Category {} -> Backend {}",
                static_cast<int>(pair.first),  // Example: log enum as int
                get_backend_name(pair.second->get_backend()));
          }
          else {
            logger->warn(
                "    Category {} has a null override backend pointer.",
                static_cast<int>(pair.first));
          }
        }
      }
      else {
        logger->info("  No specific backend overrides configured.");
      }
    }
  }

  // --- Core Backend Identification ---
  BackendType Backend::get_backend() const
  {
    // The "type" of the dispatcher could be considered its own, or the
    // primary's. Returning primary's type might be more informative for users
    // unaware of the dispatch. Or, a new BackendType::Dispatcher could be added
    // if that's meaningful. Based on Design Philosophy, it seems to act as a
    // proxy, so primary's type is reasonable.
    if (!primary_backend_) {
      // This state should not be reachable if constructor validation is
      // effective.
      auto logger = spdlog::get("OmniDSP");
      if (logger)
        logger->error(
            "DispatcherBackend::get_backend() called on an instance with a "
            "null primary_backend_.");
      // Fallback or throw an error, as this indicates a severely broken state.
      // For now, let's assume primary_backend_ is always valid after
      // construction. If it could become null later (which it shouldn't with
      // unique_ptr ownership), more robust error handling would be needed.
      return BackendType::Default;  // Or throw
    }
    return primary_backend_->get_backend();
  }

  // --- Helper to select backend ---
  Abstract::Backend* Backend::select_backend(OperationCategory category) const
  {
    auto it = overrides_.find(category);
    if (it != overrides_.end()
        && it->second) {  // Check if override exists and is not null
      return it->second.get();
    }
    // Fallback to GenericFallback if specific category not found, then to
    // primary This check avoids potential infinite recursion if GenericFallback
    // itself is queried
    if (category != OperationCategory::GenericFallback) {
      auto generic_it = overrides_.find(OperationCategory::GenericFallback);
      if (generic_it != overrides_.end() && generic_it->second) {
        return generic_it->second.get();
      }
    }
    // primary_backend_ is guaranteed to be non-null by the constructor.
    return primary_backend_.get();
  }

  // --- One-off DSP Operations ---
  [[nodiscard]] OmniExpected<F32Vec> Backend::convolve_f32(
      const F32Vec& input,
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Convolution)
        ->convolve_f32(input, kernel, type, method);
  }
  [[nodiscard]] OmniExpected<F64Vec> Backend::convolve_f64(
      const F64Vec& input,
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Convolution)
        ->convolve_f64(input, kernel, type, method);
  }
  [[nodiscard]] OmniExpected<C32Vec> Backend::convolve_c32(
      const C32Vec& input,
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Convolution)
        ->convolve_c32(input, kernel, type, method);
  }
  [[nodiscard]] OmniExpected<C64Vec> Backend::convolve_c64(
      const C64Vec& input,
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Convolution)
        ->convolve_c64(input, kernel, type, method);
  }

  [[nodiscard]] OmniExpected<F32Vec> Backend::correlate_f32(
      const F32Vec& input,
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Correlation)
        ->correlate_f32(input, kernel, type, method);
  }
  [[nodiscard]] OmniExpected<F64Vec> Backend::correlate_f64(
      const F64Vec& input,
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Correlation)
        ->correlate_f64(input, kernel, type, method);
  }
  [[nodiscard]] OmniExpected<C32Vec> Backend::correlate_c32(
      const C32Vec& input,
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Correlation)
        ->correlate_c32(input, kernel, type, method);
  }
  [[nodiscard]] OmniExpected<C64Vec> Backend::correlate_c64(
      const C64Vec& input,
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Correlation)
        ->correlate_c64(input, kernel, type, method);
  }

  [[nodiscard]] OmniExpected<C32Vec> Backend::fft_c32(const C32Vec& input) const
  {
    return select_backend(OperationCategory::FFT)->fft_c32(input);
  }
  [[nodiscard]] OmniExpected<C64Vec> Backend::fft_c64(const C64Vec& input) const
  {
    return select_backend(OperationCategory::FFT)->fft_c64(input);
  }
  [[nodiscard]] OmniExpected<C32Vec> Backend::ifft_c32(
      const C32Vec& input) const
  {
    return select_backend(OperationCategory::FFT)->ifft_c32(input);
  }
  [[nodiscard]] OmniExpected<C64Vec> Backend::ifft_c64(
      const C64Vec& input) const
  {
    return select_backend(OperationCategory::FFT)->ifft_c64(input);
  }
  [[nodiscard]] OmniExpected<C32Vec> Backend::rfft_f32(
      const F32Vec& input) const
  {
    return select_backend(OperationCategory::RFFT)->rfft_f32(input);
  }
  [[nodiscard]] OmniExpected<C64Vec> Backend::rfft_f64(
      const F64Vec& input) const
  {
    return select_backend(OperationCategory::RFFT)->rfft_f64(input);
  }
  [[nodiscard]] OmniExpected<F32Vec> Backend::irfft_c32(
      const C32Vec& input, size_t output_length) const
  {
    return select_backend(OperationCategory::RFFT)
        ->irfft_c32(input, output_length);
  }
  [[nodiscard]] OmniExpected<F64Vec> Backend::irfft_c64(
      const C64Vec& input, size_t output_length) const
  {
    return select_backend(OperationCategory::RFFT)
        ->irfft_c64(input, output_length);
  }

  // --- Specific Window Generation Methods ---
  [[nodiscard]] Status Backend::bartlett_window_f32(
      size_t length, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->bartlett_window_f32(length, output);
  }
  [[nodiscard]] Status Backend::bartlett_window_f64(
      size_t length, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->bartlett_window_f64(length, output);
  }
  [[nodiscard]] Status Backend::blackman_window_f32(
      size_t length, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->blackman_window_f32(length, output);
  }
  [[nodiscard]] Status Backend::blackman_window_f64(
      size_t length, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->blackman_window_f64(length, output);
  }
  [[nodiscard]] Status Backend::flattop_window_f32(
      size_t length, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->flattop_window_f32(length, output);
  }
  [[nodiscard]] Status Backend::flattop_window_f64(
      size_t length, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->flattop_window_f64(length, output);
  }
  [[nodiscard]] Status Backend::gaussian_window_f32(
      size_t length, double stddev, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->gaussian_window_f32(length, stddev, output);
  }
  [[nodiscard]] Status Backend::gaussian_window_f64(
      size_t length, double stddev, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->gaussian_window_f64(length, stddev, output);
  }
  [[nodiscard]] Status Backend::hamming_window_f32(
      size_t length, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->hamming_window_f32(length, output);
  }
  [[nodiscard]] Status Backend::hamming_window_f64(
      size_t length, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->hamming_window_f64(length, output);
  }
  [[nodiscard]] Status Backend::hann_window_f32(
      size_t length, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->hann_window_f32(length, output);
  }
  [[nodiscard]] Status Backend::hann_window_f64(
      size_t length, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->hann_window_f64(length, output);
  }
  [[nodiscard]] Status Backend::kaiser_window_f32(
      size_t length, double beta, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->kaiser_window_f32(length, beta, output);
  }
  [[nodiscard]] Status Backend::kaiser_window_f64(
      size_t length, double beta, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->kaiser_window_f64(length, beta, output);
  }
  [[nodiscard]] Status Backend::rectangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->rectangular_window_f32(length, output);
  }
  [[nodiscard]] Status Backend::rectangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->rectangular_window_f64(length, output);
  }
  [[nodiscard]] Status Backend::triangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->triangular_window_f32(length, output);
  }
  [[nodiscard]] Status Backend::triangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    return select_backend(OperationCategory::Windowing)
        ->triangular_window_f64(length, output);
  }

  // --- Plan Impl / Processor Impl Factory Methods ---
  // FFT (Stateless - Plan)
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
  Backend::create_fft_plan_impl_c32(size_t length) const
  {
    return select_backend(OperationCategory::FFT)
        ->create_fft_plan_impl_c32(length);
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
  Backend::create_fft_plan_impl_c64(size_t length) const
  {
    return select_backend(OperationCategory::FFT)
        ->create_fft_plan_impl_c64(length);
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
  Backend::create_rfft_plan_impl_f32(size_t length) const
  {
    return select_backend(OperationCategory::RFFT)
        ->create_rfft_plan_impl_f32(length);
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
  Backend::create_rfft_plan_impl_f64(size_t length) const
  {
    return select_backend(OperationCategory::RFFT)
        ->create_rfft_plan_impl_f64(length);
  }

  // CQT (Currently PlanImpl in Abstract::Backend)
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F32>>>
  Backend::create_cqt_processor_impl_f32(const Design::CQT& spec) const
  {
    return select_backend(OperationCategory::CQT)
        ->create_cqt_processor_impl_f32(spec);
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F64>>>
  Backend::create_cqt_processor_impl_f64(const Design::CQT& spec) const
  {
    return select_backend(OperationCategory::CQT)
        ->create_cqt_processor_impl_f64(spec);
  }

  // Resample (Currently PlanImpl in Abstract::Backend)
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>
  Backend::create_resample_processor_impl_f32(
      const Design::Resample& spec) const
  {
    return select_backend(OperationCategory::Resample)
        ->create_resample_processor_impl_f32(spec);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>
  Backend::create_resample_processor_impl_f64(
      const Design::Resample& spec) const
  {
    return select_backend(OperationCategory::Resample)
        ->create_resample_processor_impl_f64(spec);
  }

  // Convolution (Stateless - Plan)
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>
  Backend::create_convolution_plan_impl_f32(
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Convolution)
        ->create_convolution_plan_impl_f32(kernel, type, method);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>
  Backend::create_convolution_plan_impl_f64(
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Convolution)
        ->create_convolution_plan_impl_f64(kernel, type, method);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>
  Backend::create_convolution_plan_impl_c32(
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Convolution)
        ->create_convolution_plan_impl_c32(kernel, type, method);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>
  Backend::create_convolution_plan_impl_c64(
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Convolution)
        ->create_convolution_plan_impl_c64(kernel, type, method);
  }

  // Correlation (Stateless - Plan)
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>
  Backend::create_correlation_plan_impl_f32(
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Correlation)
        ->create_correlation_plan_impl_f32(kernel, type, method);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>
  Backend::create_correlation_plan_impl_f64(
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Correlation)
        ->create_correlation_plan_impl_f64(kernel, type, method);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>
  Backend::create_correlation_plan_impl_c32(
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Correlation)
        ->create_correlation_plan_impl_c32(kernel, type, method);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>
  Backend::create_correlation_plan_impl_c64(
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    return select_backend(OperationCategory::Correlation)
        ->create_correlation_plan_impl_c64(kernel, type, method);
  }

  // FIR Filter (Currently PlanImpl in Abstract::Backend)
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>
  Backend::create_fir_filter_processor_impl_f32(
      const F32Vec& coefficients) const
  {
    return select_backend(OperationCategory::FIRFilter)
        ->create_fir_filter_processor_impl_f32(coefficients);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>
  Backend::create_fir_filter_processor_impl_f64(
      const F64Vec& coefficients) const
  {
    return select_backend(OperationCategory::FIRFilter)
        ->create_fir_filter_processor_impl_f64(coefficients);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>
  Backend::create_fir_filter_processor_impl_c32(
      const C32Vec& coefficients) const
  {
    return select_backend(OperationCategory::FIRFilter)
        ->create_fir_filter_processor_impl_c32(coefficients);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>
  Backend::create_fir_filter_processor_impl_c64(
      const C64Vec& coefficients) const
  {
    return select_backend(OperationCategory::FIRFilter)
        ->create_fir_filter_processor_impl_c64(coefficients);
  }

  // IIR Filter (Currently PlanImpl in Abstract::Backend)
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>
  Backend::create_iir_filter_processor_impl_f32(
      const Coefs::IIRFilterSOS& sos_coefficients) const
  {
    return select_backend(OperationCategory::IIRFilter)
        ->create_iir_filter_processor_impl_f32(sos_coefficients);
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>
  Backend::create_iir_filter_processor_impl_f64(
      const Coefs::IIRFilterSOS& sos_coefficients) const
  {
    return select_backend(OperationCategory::IIRFilter)
        ->create_iir_filter_processor_impl_f64(sos_coefficients);
  }

  // --- Filter Design ---
  [[nodiscard]] OmniExpected<Coefs::FIRFilter<F32>>
  Backend::design_fir_filter_f32(const Design::FIRFilter& spec) const
  {
    return select_backend(
               OperationCategory::FIRFilter)  // Or a more general Design
                                              // category
        ->design_fir_filter_f32(spec);
  }
  [[nodiscard]] OmniExpected<Coefs::FIRFilter<F64>>
  Backend::design_fir_filter_f64(const Design::FIRFilter& spec) const
  {
    return select_backend(
               OperationCategory::FIRFilter)  // Or a more general Design
                                              // category
        ->design_fir_filter_f64(spec);
  }

  // Implementations for complex FIR filter design
  [[nodiscard]] OmniExpected<Coefs::FIRFilter<C32>>
  Backend::design_fir_filter_c32(const Design::FIRFilter& design) const
  {
    // Assuming FIRFilter category is appropriate for design of complex FIR
    // filters as well. If there's a more specific category (e.g.,
    // ComplexFIRFilterDesign), use that.
    return select_backend(OperationCategory::FIRFilter)
        ->design_fir_filter_c32(design);
  }

  [[nodiscard]] OmniExpected<Coefs::FIRFilter<C64>>
  Backend::design_fir_filter_c64(const Design::FIRFilter& design) const
  {
    // Assuming FIRFilter category is appropriate for design of complex FIR
    // filters as well.
    return select_backend(OperationCategory::FIRFilter)
        ->design_fir_filter_c64(design);
  }

  [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS>
  Backend::design_iir_filter_f32(const Design::IIRFilter& spec) const
  {
    return select_backend(
               OperationCategory::IIRFilter)  // Or a more general Design
                                              // category
        ->design_iir_filter_f32(spec);
  }
  [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS>
  Backend::design_iir_filter_f64(const Design::IIRFilter& spec) const
  {
    return select_backend(
               OperationCategory::IIRFilter)  // Or a more general Design
                                              // category
        ->design_iir_filter_f64(spec);
  }

}  // namespace OmniDSP::Dispatcher
