/**
 * @file omnidsp.cpp
 * @brief Implements the OmniDSP class methods, handling backend selection and
 * forwarding calls to the specific backend implementation via the pimpl pattern
 * and compile-time dispatch.
 */

#include "OmniDSP/omnidsp.h"  // Corresponding header

// Include the backend interface definition from the new location
#include "interface/backend.h"  // Defines AbstractBackend

// Include concrete backend implementation headers for factory function from new
// locations
#include "default/backend.h"     // Defines DefaultBackend
#ifdef OMNIDSP_USE_ACCELERATE    // Use actual CMake flag name
#include "accelerate/backend.h"  // Defines AccelerateBackend
#endif
#ifdef OMNIDSP_USE_ONEMKL    // Use actual CMake flag name
#include "onemkl/backend.h"  // Defines OneMKLBackend
#endif

#include <iostream>     // For debug/error messages
#include <memory>       // For std::unique_ptr, std::make_unique
#include <stdexcept>    // For std::runtime_error
#include <type_traits>  // For std::is_same_v
#include <utility>      // For std::move
#include <vector>

namespace OmniDSP {

  // Helper for static_assert in constexpr if
  template <typename T>
  struct always_false : std::false_type {};

  //--------------------------------------------------------------------------
  // OmniDSP Method Definitions
  //--------------------------------------------------------------------------

  // --- Factory ---

  [[nodiscard]] /* static */ OmniExpected<OmniDSP>
  OmniDSP::create(  // Static method needs scope
      Backend backend)
  {
    // Use the new base class name for the pointer type
    std::unique_ptr<backend::AbstractBackend> pimpl = nullptr;

    try {
      switch (backend) {
        case Backend::Accelerate:
#ifdef OMNIDSP_USE_ACCELERATE  // Use actual CMake flag name
          // Instantiate the renamed AccelerateBackend
          pimpl = std::make_unique<backend::AccelerateBackend>();
#else
          std::cerr << "Warning: Accelerate backend requested but not enabled "
                       "during build (OMNIDSP_USE_ACCELERATE=OFF)."
                    << std::endl;
          return std::unexpected(Status::UnsupportedFeature);
#endif
          break;

        case Backend::OneMKL:
#ifdef OMNIDSP_USE_ONEMKL  // Use actual CMake flag name
          // Instantiate the renamed OneMKLBackend
          pimpl = std::make_unique<backend::OneMKLBackend>();
#else
          std::cerr
              << "Warning: oneMKL backend requested but not enabled during "
                 "build (OMNIDSP_USE_ONEMKL=OFF)."
              << std::endl;
          return std::unexpected(Status::UnsupportedFeature);
#endif
          break;

        case Backend::Default:
        default:  // Fallback to Default
          // Instantiate the renamed DefaultBackend
          pimpl = std::make_unique<backend::DefaultBackend>();
          break;
      }

      if (!pimpl) {
        // Should ideally not happen if make_unique succeeds or throws
        return std::unexpected(Status::BackendError);
      }

      // Use private constructor which now takes AbstractBackend
      OmniDSP dsp_instance(std::move(pimpl));
      return dsp_instance;  // Return the created instance (implicitly moves)
    }
    catch (const std::bad_alloc &e) {
      std::cerr << "Error: Memory allocation failed during backend creation: "
                << e.what() << std::endl;
      return std::unexpected(Status::AllocationError);
    }
    catch (const std::exception &e) {
      std::cerr << "Error: Exception during backend initialization: "
                << e.what() << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      std::cerr << "Error: Unknown exception during backend creation."
                << std::endl;
      return std::unexpected(Status::Failure);
    }
  }

  // --- Constructor / Destructor / Move Operations ---

  // Private Constructor: Takes ownership of the implementation pointer
  // Update parameter type to AbstractBackend
  OmniDSP::OmniDSP(std::unique_ptr<backend::AbstractBackend> impl)
      : pimpl_(std::move(impl))
  {}

  // Destructor: Needs scope. Definition requires AbstractBackend to be
  // complete.
  OmniDSP::~OmniDSP() = default;  // Default is fine with unique_ptr

  // Move Constructor: Needs scope
  OmniDSP::OmniDSP(OmniDSP &&other) noexcept = default;

  // Move Assignment Operator: Needs scope
  OmniDSP &OmniDSP::operator=(OmniDSP &&other) noexcept = default;

  // --- Public Member Functions (Forwarding to Pimpl via Dispatch) ---
  // The rest of the dispatch logic remains the same, as it calls virtual
  // functions on the pimpl_ pointer, which now points to AbstractBackend.

  Backend OmniDSP::get_backend() const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in get_backend.");
    return pimpl_->get_backend();
  }

  // --- DSP Operations ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::convolve(
      const std::vector<RealT<T>> &input,
      const std::vector<RealT<T>> &kernel,
      ConvolutionType mode) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in convolve.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->convolve_f32(input, kernel, mode);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->convolve_f64(input, kernel, mode);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported real type for convolve");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OmniDSP::convolve(
      const std::vector<ComplexT<T>> &input,
      const std::vector<ComplexT<T>> &kernel,
      ConvolutionType mode) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in convolve.");

    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->convolve_c32(input, kernel, mode);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->convolve_c64(input, kernel, mode);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported complex type for convolve");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::correlate(
      const std::vector<RealT<T>> &input,
      const std::vector<RealT<T>> &kernel,
      ConvolutionType mode) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in correlate.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->correlate_f32(input, kernel, mode);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->correlate_f64(input, kernel, mode);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported real type for correlate");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OmniDSP::correlate(
      const std::vector<ComplexT<T>> &input,
      const std::vector<ComplexT<T>> &kernel,
      ConvolutionType mode) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in correlate.");

    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->correlate_c32(input, kernel, mode);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->correlate_c64(input, kernel, mode);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported complex type for correlate");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  // --- One-off FFTs ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OmniDSP::fft(
      const std::vector<ComplexT<T>> &input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance in fft.");

    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->fft_c32(input);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->fft_c64(input);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported complex type for fft");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OmniDSP::ifft(
      const std::vector<ComplexT<T>> &input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance in ifft.");

    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->ifft_c32(input);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->ifft_c64(input);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported complex type for ifft");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OmniDSP::rfft(
      const std::vector<RealT<T>> &input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance in rfft.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->rfft_f32(input);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->rfft_f64(input);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported real type for rfft");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::irfft(
      const std::vector<ComplexT<T>> &input, size_t output_length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance in irfft.");

    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->irfft_c32(input, output_length);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->irfft_c64(input, output_length);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported complex type for irfft");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  // --- Window Generation ---
  template <typename T>
  [[nodiscard]] Status OmniDSP::bartlett_window(
      size_t length, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in bartlett_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->bartlett_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->bartlett_window_f64(length, output);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for bartlett_window");
      // return Status::UnsupportedFeature;
    }
  }

  template <typename T>
  [[nodiscard]] Status OmniDSP::blackman_window(
      size_t length, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in blackman_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->blackman_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->blackman_window_f64(length, output);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for blackman_window");
      // return Status::UnsupportedFeature;
    }
  }

  template <typename T>
  [[nodiscard]] Status OmniDSP::flattop_window(
      size_t length, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in flattop_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->flattop_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->flattop_window_f64(length, output);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for flattop_window");
      // return Status::UnsupportedFeature;
    }
  }

  template <typename T>
  [[nodiscard]] Status OmniDSP::gaussian_window(
      size_t length, double stddev, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in gaussian_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->gaussian_window_f32(length, stddev, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->gaussian_window_f64(length, stddev, output);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for gaussian_window");
      // return Status::UnsupportedFeature;
    }
  }

  template <typename T>
  [[nodiscard]] Status OmniDSP::hamming_window(
      size_t length, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in hamming_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->hamming_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->hamming_window_f64(length, output);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for hamming_window");
      // return Status::UnsupportedFeature;
    }
  }

  template <typename T>
  [[nodiscard]] Status OmniDSP::hann_window(
      size_t length, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in hann_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->hann_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->hann_window_f64(length, output);
    }
    else {
      static_assert(always_false<T>::value, "Unsupported type for hann_window");
      // return Status::UnsupportedFeature;
    }
  }

  template <typename T>
  [[nodiscard]] Status OmniDSP::kaiser_window(
      size_t length, double beta, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in kaiser_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->kaiser_window_f32(length, beta, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->kaiser_window_f64(length, beta, output);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for kaiser_window");
      // return Status::UnsupportedFeature;
    }
  }

  template <typename T>
  [[nodiscard]] Status OmniDSP::rectangular_window(
      size_t length, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in rectangular_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->rectangular_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->rectangular_window_f64(length, output);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for rectangular_window");
      // return Status::UnsupportedFeature;
    }
  }

  template <typename T>
  [[nodiscard]] Status OmniDSP::triangular_window(
      size_t length, std::span<RealT<T>> output) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in triangular_window.");
    if (output.size() < length) return Status::SizeMismatch;

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->triangular_window_f32(length, output);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->triangular_window_f64(length, output);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for triangular_window");
      // return Status::UnsupportedFeature;
    }
  }

  // --- Plan Factories ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>>
  OmniDSP::create_fft_plan(size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in create_fft_plan.");

    if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->create_fft_plan_c32(length);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->create_fft_plan_c64(length);
    }
    else {
      static_assert(
          always_false<T>::value,
          "Unsupported complex type for create_fft_plan");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>>
  OmniDSP::create_rfft_plan(size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in create_rfft_plan.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_rfft_plan_f32(length);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_rfft_plan_f64(length);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported real type for create_rfft_plan");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>>
  OmniDSP::create_cqt_plan(
      RealT<T> sample_rate,
      RealT<T> min_freq,
      RealT<T> max_freq,
      int bins_per_octave,
      const WindowSpec<T> &window_spec) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in create_cqt_plan.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_cqt_plan_f32(
          sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_cqt_plan_f64(
          sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported real type for create_cqt_plan");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
  OmniDSP::create_resample_plan(const ResampleSpec &spec) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in create_resample_plan.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_resample_plan_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_resample_plan_f64(spec);
    }
    else {
      static_assert(
          always_false<T>::value,
          "Unsupported real type for create_resample_plan");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
  OmniDSP::create_convolution_plan(
      const std::vector<T> &kernel, ConvolutionType mode) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in create_convolution_plan.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_convolution_plan_f32(kernel, mode);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_convolution_plan_f64(kernel, mode);
    }
    else if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->create_convolution_plan_c32(kernel, mode);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->create_convolution_plan_c64(kernel, mode);
    }
    else {
      static_assert(
          always_false<T>::value,
          "Unsupported type for create_convolution_plan");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
  OmniDSP::create_correlation_plan(
      const std::vector<T> &kernel, ConvolutionType mode) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in create_correlation_plan.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_correlation_plan_f32(kernel, mode);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_correlation_plan_f64(kernel, mode);
    }
    else if constexpr (std::is_same_v<T, C32>) {
      return pimpl_->create_correlation_plan_c32(kernel, mode);
    }
    else if constexpr (std::is_same_v<T, C64>) {
      return pimpl_->create_correlation_plan_c64(kernel, mode);
    }
    else {
      static_assert(
          always_false<T>::value,
          "Unsupported type for create_correlation_plan");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
  OmniDSP::create_fir_filter_plan(const std::vector<T> &coefficients) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in create_fir_filter_plan.");

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
      static_assert(
          always_false<T>::value,
          "Unsupported type for create_fir_filter_plan");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>  // T is typically real (F32, F64)
  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
  OmniDSP::create_iir_filter_plan(
      const std::vector<SecondOrderSection<T>> &sos_coefficients) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in create_iir_filter_plan.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->create_iir_filter_plan_f32(sos_coefficients);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->create_iir_filter_plan_f64(sos_coefficients);
    }
    else {
      static_assert(
          always_false<T>::value,
          "Unsupported type for create_iir_filter_plan");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  // --- Filter Design Methods ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::design_fir_filter(
      const FIRFilterSpec<T> &spec) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in design_fir_filter.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->design_fir_filter_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->design_fir_filter_f64(spec);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for design_fir_filter");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<SecondOrderSection<T>>>
  OmniDSP::design_iir_filter(const IIRFilterSpec<T> &spec) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in design_iir_filter.");

    if constexpr (std::is_same_v<T, F32>) {
      return pimpl_->design_iir_filter_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      return pimpl_->design_iir_filter_f64(spec);
    }
    else {
      static_assert(
          always_false<T>::value, "Unsupported type for design_iir_filter");
      // return std::unexpected(Status::UnsupportedFeature);
    }
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate templates for common types (float, double) to ensure code
  // generation for the public OmniDSP class methods.
  // These STILL require the OmniDSP:: scope qualifier.

  // DSP Operations
  template OmniExpected<F32Vec> OmniDSP::convolve<F32>(
      const F32Vec &, const F32Vec &, ConvolutionType) const;
  template OmniExpected<F64Vec> OmniDSP::convolve<F64>(
      const F64Vec &, const F64Vec &, ConvolutionType) const;
  template OmniExpected<C32Vec> OmniDSP::convolve<C32>(
      const C32Vec &, const C32Vec &, ConvolutionType) const;
  template OmniExpected<C64Vec> OmniDSP::convolve<C64>(
      const C64Vec &, const C64Vec &, ConvolutionType) const;

  template OmniExpected<F32Vec> OmniDSP::correlate<F32>(
      const F32Vec &, const F32Vec &, ConvolutionType) const;
  template OmniExpected<F64Vec> OmniDSP::correlate<F64>(
      const F64Vec &, const F64Vec &, ConvolutionType) const;
  template OmniExpected<C32Vec> OmniDSP::correlate<C32>(
      const C32Vec &, const C32Vec &, ConvolutionType) const;
  template OmniExpected<C64Vec> OmniDSP::correlate<C64>(
      const C64Vec &, const C64Vec &, ConvolutionType) const;

  // One-off FFTs
  template OmniExpected<C32Vec> OmniDSP::fft<C32>(const C32Vec &) const;
  template OmniExpected<C64Vec> OmniDSP::fft<C64>(const C64Vec &) const;
  template OmniExpected<C32Vec> OmniDSP::ifft<C32>(const C32Vec &) const;
  template OmniExpected<C64Vec> OmniDSP::ifft<C64>(const C64Vec &) const;
  template OmniExpected<C32Vec> OmniDSP::rfft<F32>(const F32Vec &) const;
  template OmniExpected<C64Vec> OmniDSP::rfft<F64>(const F64Vec &) const;
  template OmniExpected<F32Vec> OmniDSP::irfft<C32>(
      const C32Vec &, size_t) const;
  template OmniExpected<F64Vec> OmniDSP::irfft<C64>(
      const C64Vec &, size_t) const;

  // Window Generation
  template Status OmniDSP::bartlett_window<F32>(size_t, std::span<F32>) const;
  template Status OmniDSP::bartlett_window<F64>(size_t, std::span<F64>) const;
  template Status OmniDSP::blackman_window<F32>(size_t, std::span<F32>) const;
  template Status OmniDSP::blackman_window<F64>(size_t, std::span<F64>) const;
  template Status OmniDSP::flattop_window<F32>(size_t, std::span<F32>) const;
  template Status OmniDSP::flattop_window<F64>(size_t, std::span<F64>) const;
  template Status OmniDSP::gaussian_window<F32>(
      size_t, double, std::span<F32>) const;
  template Status OmniDSP::gaussian_window<F64>(
      size_t, double, std::span<F64>) const;
  template Status OmniDSP::hamming_window<F32>(size_t, std::span<F32>) const;
  template Status OmniDSP::hamming_window<F64>(size_t, std::span<F64>) const;
  template Status OmniDSP::hann_window<F32>(size_t, std::span<F32>) const;
  template Status OmniDSP::hann_window<F64>(size_t, std::span<F64>) const;
  template Status OmniDSP::kaiser_window<F32>(
      size_t, double, std::span<F32>) const;
  template Status OmniDSP::kaiser_window<F64>(
      size_t, double, std::span<F64>) const;
  template Status OmniDSP::rectangular_window<F32>(
      size_t, std::span<F32>) const;
  template Status OmniDSP::rectangular_window<F64>(
      size_t, std::span<F64>) const;
  template Status OmniDSP::triangular_window<F32>(size_t, std::span<F32>) const;
  template Status OmniDSP::triangular_window<F64>(size_t, std::span<F64>) const;

  // Plan Factories
  template OmniExpected<std::unique_ptr<FFTPlan<C32>>>
      OmniDSP::create_fft_plan<C32>(size_t) const;
  template OmniExpected<std::unique_ptr<FFTPlan<C64>>>
      OmniDSP::create_fft_plan<C64>(size_t) const;

  template OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
      OmniDSP::create_rfft_plan<F32>(size_t) const;
  template OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
      OmniDSP::create_rfft_plan<F64>(size_t) const;

  template OmniExpected<std::unique_ptr<CQTPlan<F32>>> OmniDSP::create_cqt_plan<
      F32>(F32, F32, F32, int, const WindowSpec<F32> &) const;
  template OmniExpected<std::unique_ptr<CQTPlan<F64>>> OmniDSP::create_cqt_plan<
      F64>(F64, F64, F64, int, const WindowSpec<F64> &) const;

  template OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
  OmniDSP::create_resample_plan<F32>(const ResampleSpec &) const;
  template OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
  OmniDSP::create_resample_plan<F64>(const ResampleSpec &) const;

  template OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
  OmniDSP::create_convolution_plan<F32>(const F32Vec &, ConvolutionType) const;
  template OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
  OmniDSP::create_convolution_plan<F64>(const F64Vec &, ConvolutionType) const;
  template OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
  OmniDSP::create_convolution_plan<C32>(const C32Vec &, ConvolutionType) const;
  template OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
  OmniDSP::create_convolution_plan<C64>(const C64Vec &, ConvolutionType) const;

  template OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
  OmniDSP::create_correlation_plan<F32>(const F32Vec &, ConvolutionType) const;
  template OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
  OmniDSP::create_correlation_plan<F64>(const F64Vec &, ConvolutionType) const;
  template OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
  OmniDSP::create_correlation_plan<C32>(const C32Vec &, ConvolutionType) const;
  template OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
  OmniDSP::create_correlation_plan<C64>(const C64Vec &, ConvolutionType) const;

  template OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
  OmniDSP::create_fir_filter_plan<F32>(const F32Vec &) const;
  template OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
  OmniDSP::create_fir_filter_plan<F64>(const F64Vec &) const;
  template OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
  OmniDSP::create_fir_filter_plan<C32>(const C32Vec &) const;
  template OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
  OmniDSP::create_fir_filter_plan<C64>(const C64Vec &) const;

  template OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
  OmniDSP::create_iir_filter_plan<F32>(
      const std::vector<SecondOrderSection<F32>> &) const;
  template OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
  OmniDSP::create_iir_filter_plan<F64>(
      const std::vector<SecondOrderSection<F64>> &) const;

  // Filter Design Methods
  template OmniExpected<F32Vec> OmniDSP::design_fir_filter<F32>(
      const FIRFilterSpec<F32> &) const;
  template OmniExpected<F64Vec> OmniDSP::design_fir_filter<F64>(
      const FIRFilterSpec<F64> &) const;

  template OmniExpected<std::vector<SecondOrderSection<F32>>>
  OmniDSP::design_iir_filter<F32>(const IIRFilterSpec<F32> &) const;
  template OmniExpected<std::vector<SecondOrderSection<F64>>>
  OmniDSP::design_iir_filter<F64>(const IIRFilterSpec<F64> &) const;

}  // namespace OmniDSP
