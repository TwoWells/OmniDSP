/**
 * @file omnidsp.hpp
 * @brief Main public header file for the OmniDSP library, defining the main
 * OmniDSP class interface.
 */

#ifndef OMNIDSP_HPP
#define OMNIDSP_HPP

#include <complex>
#include <cstddef>
#include <expected>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"
#include "convolution.hpp"
#include "core_types.hpp"
#include "cqt.hpp"
#include "fft.hpp"
#include "filter.hpp"
#include "resample.hpp"
#include "window.hpp"

// Include the backend interface definition - needed for pimpl_ type
#include "../src/omnidsp/interface/backend.hpp"  // Adjust path as needed

namespace OmniDSP {

  // Helper for static_assert in constexpr if
  template <typename T>
  struct always_false : std::false_type {};

  /**
   * @brief Main class providing access to OmniDSP functionalities.
   */
  class OMNIDSP_EXPORT OmniDSP {
   public:
    [[nodiscard]] static OmniExpected<OmniDSP> create(
        Backend backend = Backend::Default);

    ~OmniDSP();
    OmniDSP(OmniDSP&& other) noexcept;
    OmniDSP& operator=(OmniDSP&& other) noexcept;

    OmniDSP(const OmniDSP&) = delete;
    OmniDSP& operator=(const OmniDSP&) = delete;

    Backend get_backend() const;

    //-------------------------------------------------------------------------
    /** @defgroup DspOps DSP Operations (Member Methods)
     * @{ */
    //-------------------------------------------------------------------------

    /** @name Convolution and Correlation */
    ///@{
    // *** Default argument ONLY in the declaration within the class ***
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<Utils::GetRealType<T>>> convolve(
        const std::vector<Utils::GetRealType<T>>& input,
        const std::vector<Utils::GetRealType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> convolve(
        const std::vector<Utils::GetComplexType<T>>& input,
        const std::vector<Utils::GetComplexType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<Utils::GetRealType<T>>> correlate(
        const std::vector<Utils::GetRealType<T>>& input,
        const std::vector<Utils::GetRealType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> correlate(
        const std::vector<Utils::GetComplexType<T>>& input,
        const std::vector<Utils::GetComplexType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    ///@}

    /** @name Fourier Transforms (One-Off) */
    ///@{
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> fft(
        const std::vector<Utils::GetComplexType<T>>& input) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> ifft(
        const std::vector<Utils::GetComplexType<T>>& input) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> rfft(
        const std::vector<Utils::GetRealType<T>>& input) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> irfft(  // T is REAL type
        const std::vector<Utils::GetComplexType<T>>& input,
        size_t output_length) const;
    ///@}

    /** @name Window Coefficient Generation */
    ///@{
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> bartlett_window(
        size_t length) const;
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> blackman_window(
        size_t length) const;
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> flattop_window(
        size_t length) const;
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> gaussian_window(
        size_t length, double stddev) const;
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> hamming_window(
        size_t length) const;
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> hann_window(size_t length) const;
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> kaiser_window(
        size_t length, double beta) const;
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> rectangular_window(
        size_t length) const;
    template <typename T>  // T is REAL type
    [[nodiscard]] OmniExpected<std::vector<T>> triangular_window(
        size_t length) const;
    ///@}
    /** @} */  // End of DspOps group

    //-------------------------------------------------------------------------
    /** @defgroup PlanFactories Plan Factory Methods
     * @brief Methods to create optimized Plan objects for repeated operations.
     * @{ */
    //-------------------------------------------------------------------------
    // *** Default argument ONLY in the declaration within the class ***
    template <typename T>  // T is complex type
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>> create_fft_plan(
        size_t length) const;
    template <typename T>  // T is real type
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>> create_rfft_plan(
        size_t length) const;
    template <typename T>  // T is real type
    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>> create_cqt_plan(
        Utils::GetRealType<T> sample_rate,
        Utils::GetRealType<T> min_freq,
        Utils::GetRealType<T> max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec = WindowSpec()) const;
    template <typename T>  // T is real type
    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
    create_resample_plan(const ResampleSpec& spec) const;
    template <typename T>  // T can be REAL or COMPLEX
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
    create_convolution_plan(
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>  // T can be REAL or COMPLEX
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
    create_correlation_plan(
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>  // T can be REAL or COMPLEX
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
    create_fir_filter_plan(const std::vector<T>& coefficients) const;
    template <typename T>  // T is typically real
    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
    create_iir_filter_plan(
        const std::vector<IIRFilterCoef>& sos_coefficients) const;
    /** @} */  // End of PlanFactories group

    //-------------------------------------------------------------------------
    /** @defgroup FilterDesign Filter Design Methods
     * @brief Methods to design standard digital filters.
     * @{ */
    //-------------------------------------------------------------------------
    template <typename T>  // T is real type
    [[nodiscard]] OmniExpected<std::vector<T>> design_fir_filter(
        const FIRFilterSpec& spec) const;
    template <typename T>  // T is real type
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>> design_iir_filter(
        const IIRFilterSpec& spec) const;
    /** @} */  // End of FilterDesign group

   private:
    OmniDSP(std::unique_ptr<Abstract::AbstractBackend> impl);
    std::unique_ptr<Abstract::AbstractBackend> pimpl_;
  };

  //--------------------------------------------------------------------------
  // Template Method Definitions (Defined in Header)
  //--------------------------------------------------------------------------

  // --- DSP Operations ---
  // *** Default argument REMOVED from definition ***
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetRealType<T>>>
  OmniDSP::convolve(
      const std::vector<Utils::GetRealType<T>>& input,
      const std::vector<Utils::GetRealType<T>>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->convolve_f32(input, kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->convolve_f64(input, kernel, type, method);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type");
    }
  }

  // *** Default argument REMOVED from definition ***
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::convolve(
      const std::vector<Utils::GetComplexType<T>>& input,
      const std::vector<Utils::GetComplexType<T>>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->convolve_c32(input, kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->convolve_c64(input, kernel, type, method);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported complex type");
    }
  }

  // *** Default argument REMOVED from definition ***
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetRealType<T>>>
  OmniDSP::correlate(
      const std::vector<Utils::GetRealType<T>>& input,
      const std::vector<Utils::GetRealType<T>>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->correlate_f32(input, kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->correlate_f64(input, kernel, type, method);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type");
    }
  }

  // *** Default argument REMOVED from definition ***
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::correlate(
      const std::vector<Utils::GetComplexType<T>>& input,
      const std::vector<Utils::GetComplexType<T>>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->correlate_c32(input, kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->correlate_c64(input, kernel, type, method);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported complex type");
    }
  }

  // --- One-off FFTs ---
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::fft(const std::vector<Utils::GetComplexType<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->fft_c32(input);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->fft_c64(input);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported complex type");
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::ifft(const std::vector<Utils::GetComplexType<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->ifft_c32(input);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->ifft_c64(input);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported complex type");
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::rfft(const std::vector<Utils::GetRealType<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->rfft_f32(input);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->rfft_f64(input);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type");
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::irfft(
      const std::vector<Utils::GetComplexType<T>>& input,
      size_t output_length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->irfft_c32(input, output_length);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->irfft_c64(input, output_length);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type");
    }
  }

  // --- Window Generation ---
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::bartlett_window(
      size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->bartlett_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->bartlett_window_f64(length, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::blackman_window(
      size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->blackman_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->blackman_window_f64(length, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::flattop_window(
      size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->flattop_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->flattop_window_f64(length, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::gaussian_window(
      size_t length, double stddev) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->gaussian_window_f32(length, stddev, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->gaussian_window_f64(length, stddev, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::hamming_window(
      size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->hamming_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->hamming_window_f64(length, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::hann_window(
      size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->hann_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->hann_window_f64(length, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::kaiser_window(
      size_t length, double beta) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->kaiser_window_f32(length, beta, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->kaiser_window_f64(length, beta, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::rectangular_window(
      size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->rectangular_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->rectangular_window_f64(length, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::triangular_window(
      size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    std::vector<T> output(length);
    Status status;
    if constexpr (std::is_same_v<T, F32>) {
      status = pimpl_->triangular_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      status = pimpl_->triangular_window_f64(length, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
    if (status != Status::Success) return std::unexpected(status);
    return output;
  }

  // --- Plan Factories ---
  // *** Default argument REMOVED from definition ***
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::unique_ptr<FFTPlan<T>>>
  OmniDSP::create_fft_plan(size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->create_fft_plan_c32(length);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->create_fft_plan_c64(length);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported complex type");
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::unique_ptr<RFFTPlan<T>>>
  OmniDSP::create_rfft_plan(size_t length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_rfft_plan_f32(length);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_rfft_plan_f64(length);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type");
    }
  }

  // *** Default argument REMOVED from definition ***
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::unique_ptr<CQTPlan<T>>>
  OmniDSP::create_cqt_plan(
      Utils::GetRealType<T> sample_rate,
      Utils::GetRealType<T> min_freq,
      Utils::GetRealType<T> max_freq,
      int bins_per_octave,
      const WindowSpec& window_spec) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_cqt_plan_f32(
          sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_cqt_plan_f64(
          sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type");
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::unique_ptr<ResamplePlan<T>>>
  OmniDSP::create_resample_plan(const ResampleSpec& spec) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_resample_plan_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_resample_plan_f64(spec);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type");
    }
  }

  // *** Default argument REMOVED from definition ***
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
  OmniDSP::create_convolution_plan(
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_convolution_plan_f32(kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_convolution_plan_f64(kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->create_convolution_plan_c32(kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->create_convolution_plan_c64(kernel, type, method);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
  }

  // *** Default argument REMOVED from definition ***
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
  OmniDSP::create_correlation_plan(
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_correlation_plan_f32(kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_correlation_plan_f64(kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->create_correlation_plan_c32(kernel, type, method);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->create_correlation_plan_c64(kernel, type, method);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
  OmniDSP::create_fir_filter_plan(const std::vector<T>& coefficients) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_fir_filter_plan_f32(coefficients);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_fir_filter_plan_f64(coefficients);
    }
    else if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->create_fir_filter_plan_c32(coefficients);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->create_fir_filter_plan_c64(coefficients);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
  OmniDSP::create_iir_filter_plan(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_iir_filter_plan_f32(sos_coefficients);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_iir_filter_plan_f64(sos_coefficients);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
  }

  // --- Filter Design Methods ---
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::design_fir_filter(
      const FIRFilterSpec& spec) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->design_fir_filter_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->design_fir_filter_f64(spec);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<IIRFilterCoef>>
  OmniDSP::design_iir_filter(const IIRFilterSpec& spec) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->design_iir_filter_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->design_iir_filter_f64(spec);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type");
    }
  }

}  // namespace OmniDSP

#endif  // OMNIDSP_HPP
