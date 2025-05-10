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
#include "convolution.hpp"  // Includes ConvolutionPlan
#include "core_types.hpp"
#include "cqt.hpp"       // Includes CQTPlan
#include "fft.hpp"       // Includes FFTPlan, RFFTPlan (now with static create)
#include "filter.hpp"    // Includes FIRFilterPlan, IIRFilterPlan
#include "resample.hpp"  // Includes ResamplePlan
#include "window.hpp"    // Includes WindowSetup

// Include the backend interface definition - needed for pimpl_ type
#include "../src/omnidsp/interface/backend.hpp"  // Defines Abstract::Backend

namespace OmniDSP {

  // Helper for static_assert in constexpr if (if still needed elsewhere, or
  // remove if not)
  template <typename T>
  struct always_false : std::false_type {};

  /**
   * @brief Main class providing access to OmniDSP functionalities.
   */
  class OMNIDSP_EXPORT OmniDSP {
   public:
    [[nodiscard]] static OmniExpected<OmniDSP> create(
        BackendType backend = BackendType::Default);

    ~OmniDSP();
    OmniDSP(OmniDSP&& other) noexcept;
    OmniDSP& operator=(OmniDSP&& other) noexcept;

    OmniDSP(const OmniDSP&) = delete;
    OmniDSP& operator=(const OmniDSP&) = delete;

    BackendType get_backend() const;

    //-------------------------------------------------------------------------
    /** @defgroup DspOps DSP Operations (Member Methods)
     * @{ */
    //-------------------------------------------------------------------------

    /** @name Convolution and Correlation */
    ///@{
    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetRealType<T>>> convolve(
        const std::vector<Utils::GetRealType<T>>& input,
        const std::vector<Utils::GetRealType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>  // T is C32 or C64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> convolve(
        const std::vector<Utils::GetComplexType<T>>& input,
        const std::vector<Utils::GetComplexType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetRealType<T>>> correlate(
        const std::vector<Utils::GetRealType<T>>& input,
        const std::vector<Utils::GetRealType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>  // T is C32 or C64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> correlate(
        const std::vector<Utils::GetComplexType<T>>& input,
        const std::vector<Utils::GetComplexType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    ///@}

    /** @name Fourier Transforms (One-Off) */
    ///@{
    template <typename T>  // T is C32 or C64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> fft(
        const std::vector<Utils::GetComplexType<T>>& input) const;
    template <typename T>  // T is C32 or C64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> ifft(
        const std::vector<Utils::GetComplexType<T>>& input) const;
    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> rfft(
        const std::vector<Utils::GetRealType<T>>& input) const;
    template <typename T>  // T is F32 or F64 (real type)
    [[nodiscard]] OmniExpected<std::vector<T>> irfft(
        const std::vector<Utils::GetComplexType<T>>& input,
        size_t output_length) const;
    ///@}

    /** @name Window Coefficient Generation */
    ///@{
    template <typename T>  // T is F32 or F64 (real type)
    [[nodiscard]] OmniExpected<std::vector<T>> generate_window(
        const WindowSetup& setup) const;
    ///@}

    // Deprecated specific window functions are now implemented in terms of the
    // above.
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> bartlett_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> blackman_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> flattop_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> gaussian_window(
        size_t length, double stddev) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> hamming_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> hann_window(size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> kaiser_window(
        size_t length, double beta) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> rectangular_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> triangular_window(
        size_t length) const;

    /** @} */  // End of DspOps group

    //-------------------------------------------------------------------------
    /** @defgroup PlanFactories Plan Factory Methods
     * @brief Methods to create optimized Plan objects for repeated operations.
     * @{ */
    //-------------------------------------------------------------------------
    template <typename T>  // T is C32 or C64
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>> create_fft_plan(
        size_t length) const;

    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>> create_rfft_plan(
        size_t length) const;

    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>> create_cqt_plan(
        Utils::GetRealType<T> sample_rate,
        Utils::GetRealType<T> min_freq,
        Utils::GetRealType<T> max_freq,
        int bins_per_octave,
        const WindowSetup& window_setup
        = WindowSetup(WindowType::Hann, 0)) const;  // Default WindowSetup

    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
    create_resample_plan(const ResampleSpec& spec) const;

    template <typename T>  // T can be F32, F64, C32, C64
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
    create_convolution_plan(
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;

    template <typename T>  // T can be F32, F64, C32, C64
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
    create_correlation_plan(
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;

    template <typename T>  // T can be F32, F64, C32, C64
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
    create_fir_filter_plan(const std::vector<T>& coefficients) const;

    template <typename T>  // T is typically F32 or F64 (real)
    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
    create_iir_filter_plan(
        const std::vector<IIRFilterCoef>& sos_coefficients) const;
    /** @} */  // End of PlanFactories group

    //-------------------------------------------------------------------------
    /** @defgroup FilterDesign Filter Design Methods
     * @brief Methods to design standard digital filters.
     * @{ */
    //-------------------------------------------------------------------------
    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<FIRCoefs<T>>
    design_fir_filter(  // Changed return type
        const FIRFilterSpec& spec) const;

    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>> design_iir_filter(
        const IIRFilterSpec& spec) const;
    /** @} */  // End of FilterDesign group

   private:
    OmniDSP(std::unique_ptr<Abstract::Backend> impl);
    std::unique_ptr<Abstract::Backend> pimpl_;
  };

  //--------------------------------------------------------------------------
  // Template Method Definitions (Defined in Header)
  //--------------------------------------------------------------------------

  // --- DSP Operations (One-Off Implementations) ---
  // (Convolution, Correlation, FFT, Windowing implementations remain largely
  // the same,
  //  delegating to pimpl_ which calls the appropriate backend methods)
  //  The windowing methods are updated as per previous discussions.

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetRealType<T>>>
  OmniDSP::convolve(
      const std::vector<Utils::GetRealType<T>>& input,
      const std::vector<Utils::GetRealType<T>>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance (convolve real)");
    if constexpr (std::is_same_v<Utils::GetRealType<T>, F32>) {
      return pimpl_->convolve_f32(input, kernel, type, method);
    }
    else if constexpr (std::is_same_v<Utils::GetRealType<T>, F64>) {
      return pimpl_->convolve_f64(input, kernel, type, method);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported real type for convolve");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::convolve(
      const std::vector<Utils::GetComplexType<T>>& input,
      const std::vector<Utils::GetComplexType<T>>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance (convolve complex)");
    if constexpr (std::is_same_v<Utils::GetComplexType<T>, C32>) {
      return pimpl_->convolve_c32(input, kernel, type, method);
    }
    else if constexpr (std::is_same_v<Utils::GetComplexType<T>, C64>) {
      return pimpl_->convolve_c64(input, kernel, type, method);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported complex type for convolve");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetRealType<T>>>
  OmniDSP::correlate(
      const std::vector<Utils::GetRealType<T>>& input,
      const std::vector<Utils::GetRealType<T>>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance (correlate real)");
    if constexpr (std::is_same_v<Utils::GetRealType<T>, F32>) {
      return pimpl_->correlate_f32(input, kernel, type, method);
    }
    else if constexpr (std::is_same_v<Utils::GetRealType<T>, F64>) {
      return pimpl_->correlate_f64(input, kernel, type, method);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported real type for correlate");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::correlate(
      const std::vector<Utils::GetComplexType<T>>& input,
      const std::vector<Utils::GetComplexType<T>>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance (correlate complex)");
    if constexpr (std::is_same_v<Utils::GetComplexType<T>, C32>) {
      return pimpl_->correlate_c32(input, kernel, type, method);
    }
    else if constexpr (std::is_same_v<Utils::GetComplexType<T>, C64>) {
      return pimpl_->correlate_c64(input, kernel, type, method);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported complex type for correlate");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::fft(const std::vector<Utils::GetComplexType<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance (fft)");
    if constexpr (std::is_same_v<Utils::GetComplexType<T>, C32>) {
      return pimpl_->fft_c32(input);
    }
    else if constexpr (std::is_same_v<Utils::GetComplexType<T>, C64>) {
      return pimpl_->fft_c64(input);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported complex type for fft");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::ifft(const std::vector<Utils::GetComplexType<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance (ifft)");
    if constexpr (std::is_same_v<Utils::GetComplexType<T>, C32>) {
      return pimpl_->ifft_c32(input);
    }
    else if constexpr (std::is_same_v<Utils::GetComplexType<T>, C64>) {
      return pimpl_->ifft_c64(input);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported complex type for ifft");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<Utils::GetComplexType<T>>>
  OmniDSP::rfft(const std::vector<Utils::GetRealType<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance (rfft)");
    if constexpr (std::is_same_v<Utils::GetRealType<T>, F32>) {
      return pimpl_->rfft_f32(input);
    }
    else if constexpr (std::is_same_v<Utils::GetRealType<T>, F64>) {
      return pimpl_->rfft_f64(input);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type for rfft");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>  // T is F32 or F64 (real type)
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::irfft(
      const std::vector<Utils::GetComplexType<T>>& input,
      size_t output_length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance (irfft)");
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->irfft_c32(input, output_length);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->irfft_c64(input, output_length);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type for irfft");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  // --- Unified Window Generation ---
  template <typename T>  // T is F32 or F64
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::generate_window(
      const WindowSetup& setup) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::generate_window called on
      // invalid instance.");
      throw std::runtime_error("Invalid OmniDSP instance (generate_window)");
    }

    // Validate setup length before allocating vector to prevent large
    // allocations on bad input
    if (setup.length < 0) {
      // spdlog::get("OmniDSP")->warn("OmniDSP::generate_window: Invalid
      // setup.length ({})", setup.length);
      return std::unexpected(Status::InvalidArgument);
    }
    std::vector<T> output(static_cast<size_t>(setup.length));
    if (setup.length == 0) {
      return output;  // Return empty vector for zero length
    }

    Status status = Status::Failure;
    // Dispatch to the correct specific virtual window function on the backend
    switch (setup.type) {
      case WindowType::Bartlett:
        if constexpr (std::is_same_v<T, F32>)
          status = pimpl_->bartlett_window_f32(setup.length, output);
        else if constexpr (std::is_same_v<T, F64>)
          status = pimpl_->bartlett_window_f64(setup.length, output);
        break;
      case WindowType::Blackman:
        if constexpr (std::is_same_v<T, F32>)
          status = pimpl_->blackman_window_f32(setup.length, output);
        else if constexpr (std::is_same_v<T, F64>)
          status = pimpl_->blackman_window_f64(setup.length, output);
        break;
      case WindowType::Flattop:
        if constexpr (std::is_same_v<T, F32>)
          status = pimpl_->flattop_window_f32(setup.length, output);
        else if constexpr (std::is_same_v<T, F64>)
          status = pimpl_->flattop_window_f64(setup.length, output);
        break;
      case WindowType::Gaussian:
        if (setup.params && setup.params->count("sigma")) {
          double sigma = setup.params->at("sigma");
          if constexpr (std::is_same_v<T, F32>)
            status = pimpl_->gaussian_window_f32(setup.length, sigma, output);
          else if constexpr (std::is_same_v<T, F64>)
            status = pimpl_->gaussian_window_f64(setup.length, sigma, output);
        }
        else {
          status = Status::InvalidArgument; /* Missing sigma */
        }
        break;
      case WindowType::Hamming:
        if constexpr (std::is_same_v<T, F32>)
          status = pimpl_->hamming_window_f32(setup.length, output);
        else if constexpr (std::is_same_v<T, F64>)
          status = pimpl_->hamming_window_f64(setup.length, output);
        break;
      case WindowType::Hann:
        if constexpr (std::is_same_v<T, F32>)
          status = pimpl_->hann_window_f32(setup.length, output);
        else if constexpr (std::is_same_v<T, F64>)
          status = pimpl_->hann_window_f64(setup.length, output);
        break;
      case WindowType::Kaiser:
        if (setup.params && setup.params->count("beta")) {
          double beta = setup.params->at("beta");
          if constexpr (std::is_same_v<T, F32>)
            status = pimpl_->kaiser_window_f32(setup.length, beta, output);
          else if constexpr (std::is_same_v<T, F64>)
            status = pimpl_->kaiser_window_f64(setup.length, beta, output);
        }
        else {
          status = Status::InvalidArgument; /* Missing beta */
        }
        break;
      case WindowType::Rectangular:
        if constexpr (std::is_same_v<T, F32>)
          status = pimpl_->rectangular_window_f32(setup.length, output);
        else if constexpr (std::is_same_v<T, F64>)
          status = pimpl_->rectangular_window_f64(setup.length, output);
        break;
      case WindowType::Triangular:
        if constexpr (std::is_same_v<T, F32>)
          status = pimpl_->triangular_window_f32(setup.length, output);
        else if constexpr (std::is_same_v<T, F64>)
          status = pimpl_->triangular_window_f64(setup.length, output);
        break;
      default:
        // spdlog::get("OmniDSP")->warn("OmniDSP::generate_window: Unknown
        // window type in WindowSetup: {}", static_cast<int>(setup.type));
        status = Status::InvalidArgument;
        break;
    }

    if (status != Status::Success) return std::unexpected(status);
    return output;
  }

  // Deprecated specific window functions - implementations now call the unified
  // generate_window
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::bartlett_window(
      size_t length) const
  {
    try {
      return generate_window<T>(
          WindowSetup(WindowType::Bartlett, static_cast<int>(length)));
    }
    catch (const std::invalid_argument& e) { /* spdlog error */
      return std::unexpected(Status::InvalidArgument);
    }
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::blackman_window(
      size_t length) const
  {
    try {
      return generate_window<T>(
          WindowSetup(WindowType::Blackman, static_cast<int>(length)));
    }
    catch (const std::invalid_argument& e) {
      return std::unexpected(Status::InvalidArgument);
    }
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::flattop_window(
      size_t length) const
  {
    try {
      return generate_window<T>(
          WindowSetup(WindowType::Flattop, static_cast<int>(length)));
    }
    catch (const std::invalid_argument& e) {
      return std::unexpected(Status::InvalidArgument);
    }
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::gaussian_window(
      size_t length, double stddev) const
  {
    try {
      return generate_window<T>(WindowSetup(
          WindowType::Gaussian,
          static_cast<int>(length),
          WindowParams{{"sigma", stddev}}));
    }
    catch (const std::invalid_argument& e) {
      return std::unexpected(Status::InvalidArgument);
    }
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::hamming_window(
      size_t length) const
  {
    try {
      return generate_window<T>(
          WindowSetup(WindowType::Hamming, static_cast<int>(length)));
    }
    catch (const std::invalid_argument& e) {
      return std::unexpected(Status::InvalidArgument);
    }
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::hann_window(
      size_t length) const
  {
    try {
      return generate_window<T>(
          WindowSetup(WindowType::Hann, static_cast<int>(length)));
    }
    catch (const std::invalid_argument& e) {
      return std::unexpected(Status::InvalidArgument);
    }
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::kaiser_window(
      size_t length, double beta) const
  {
    try {
      return generate_window<T>(WindowSetup(
          WindowType::Kaiser,
          static_cast<int>(length),
          WindowParams{{"beta", beta}}));
    }
    catch (const std::invalid_argument& e) {
      return std::unexpected(Status::InvalidArgument);
    }
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::rectangular_window(
      size_t length) const
  {
    try {
      return generate_window<T>(
          WindowSetup(WindowType::Rectangular, static_cast<int>(length)));
    }
    catch (const std::invalid_argument& e) {
      return std::unexpected(Status::InvalidArgument);
    }
  }
  template <typename T>
  [[nodiscard]] inline OmniExpected<std::vector<T>> OmniDSP::triangular_window(
      size_t length) const
  {
    try {
      return generate_window<T>(
          WindowSetup(WindowType::Triangular, static_cast<int>(length)));
    }
    catch (const std::invalid_argument& e) {
      return std::unexpected(Status::InvalidArgument);
    }
  }

  // --- Plan Factories ---
  template <typename T>  // T is C32 or C64
  [[nodiscard]] inline OmniExpected<std::unique_ptr<FFTPlan<T>>>
  OmniDSP::create_fft_plan(size_t length) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::create_fft_plan called on
      // invalid instance.");
      throw std::runtime_error("Invalid OmniDSP instance (create_fft_plan)");
    }
    return FFTPlan<T>::create(*pimpl_, length);
  }

  template <typename T>  // T is F32 or F64
  [[nodiscard]] inline OmniExpected<std::unique_ptr<RFFTPlan<T>>>
  OmniDSP::create_rfft_plan(size_t length) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::create_rfft_plan called on
      // invalid instance.");
      throw std::runtime_error("Invalid OmniDSP instance (create_rfft_plan)");
    }
    return RFFTPlan<T>::create(*pimpl_, length);
  }

  template <typename T>  // T is F32 or F64
  [[nodiscard]] inline OmniExpected<std::unique_ptr<CQTPlan<T>>>
  OmniDSP::create_cqt_plan(
      Utils::GetRealType<T> sample_rate,
      Utils::GetRealType<T> min_freq,
      Utils::GetRealType<T> max_freq,
      int bins_per_octave,
      const WindowSetup& window_setup) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::create_cqt_plan called on
      // invalid instance.");
      throw std::runtime_error("Invalid OmniDSP instance (create_cqt_plan)");
    }
    // Assuming CQTPlan<T>::create is updated similarly
    return CQTPlan<T>::create(
        *pimpl_,
        sample_rate,
        min_freq,
        max_freq,
        bins_per_octave,
        window_setup);
  }

  template <typename T>  // T is F32 or F64
  [[nodiscard]] inline OmniExpected<std::unique_ptr<ResamplePlan<T>>>
  OmniDSP::create_resample_plan(const ResampleSpec& spec) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::create_resample_plan called on
      // invalid instance.");
      throw std::runtime_error(
          "Invalid OmniDSP instance (create_resample_plan)");
    }
    // Assuming ResamplePlan<T>::create is updated similarly
    return ResamplePlan<T>::create(*pimpl_, spec);
  }

  template <typename T>  // T can be F32, F64, C32, C64
  [[nodiscard]] inline OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
  OmniDSP::create_convolution_plan(
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::create_convolution_plan called
      // on invalid instance.");
      throw std::runtime_error(
          "Invalid OmniDSP instance (create_convolution_plan)");
    }
    // Assuming ConvolutionPlan<T>::create is updated similarly
    return ConvolutionPlan<T>::create(*pimpl_, kernel, type, method);
  }

  template <typename T>  // T can be F32, F64, C32, C64
  [[nodiscard]] inline OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
  OmniDSP::create_correlation_plan(
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::create_correlation_plan called
      // on invalid instance.");
      throw std::runtime_error(
          "Invalid OmniDSP instance (create_correlation_plan)");
    }
    // Assuming CorrelationPlan<T>::create is updated similarly
    return CorrelationPlan<T>::create(*pimpl_, kernel, type, method);
  }

  template <typename T>  // T can be F32, F64, C32, C64
  [[nodiscard]] inline OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
  OmniDSP::create_fir_filter_plan(const std::vector<T>& coefficients) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::create_fir_filter_plan called
      // on invalid instance.");
      throw std::runtime_error(
          "Invalid OmniDSP instance (create_fir_filter_plan)");
    }
    // Assuming FIRFilterPlan<T>::create is updated similarly
    return FIRFilterPlan<T>::create(*pimpl_, coefficients);
  }

  template <typename T>  // T is typically F32 or F64
  [[nodiscard]] inline OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
  OmniDSP::create_iir_filter_plan(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::create_iir_filter_plan called
      // on invalid instance.");
      throw std::runtime_error(
          "Invalid OmniDSP instance (create_iir_filter_plan)");
    }
    // Assuming IIRFilterPlan<T>::create is updated similarly
    return IIRFilterPlan<T>::create(*pimpl_, sos_coefficients);
  }

  // --- Filter Design Methods ---
  template <typename T>  // T is F32 or F64
  [[nodiscard]] inline OmniExpected<FIRCoefs<T>> OmniDSP::design_fir_filter(
      const FIRFilterSpec& spec) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::design_fir_filter called on
      // invalid instance.");
      throw std::runtime_error("Invalid OmniDSP instance (design_fir_filter)");
    }
    // Ensure FIRFilterSpec uses WindowSetup
    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->design_fir_filter_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->design_fir_filter_f64(spec);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for design_fir_filter");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>  // T is F32 or F64
  [[nodiscard]] inline OmniExpected<std::vector<IIRFilterCoef>>
  OmniDSP::design_iir_filter(const IIRFilterSpec& spec) const
  {
    if (!pimpl_) {
      // spdlog::get("OmniDSP")->error("OmniDSP::design_iir_filter called on
      // invalid instance.");
      throw std::runtime_error("Invalid OmniDSP instance (design_iir_filter)");
    }
    if constexpr (std::is_same_v<T, F32>) {  // Type T might be redundant here
                                             // if IIRFilterCoef is always
                                             // double
      return pimpl_->design_iir_filter_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->design_iir_filter_f64(spec);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for design_iir_filter");
      return std::unexpected(Status::UnsupportedFeature);
    }
  }

}  // namespace OmniDSP

#endif  // OMNIDSP_HPP
