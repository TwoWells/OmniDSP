/**
 * @file omnidsp.h
 * @brief Main public header file for the OmniDSP library, defining the main
 * OmniDSP class interface.
 */

#ifndef OMNIDSP_H
#define OMNIDSP_H

#include <complex>      // For std::complex parameter and return types
#include <cstddef>      // For size_t
#include <memory>       // For std::unique_ptr (Pimpl and Plan factories)
#include <string_view>  // Potentially for future parameters
#include <vector>       // For std::vector parameter and return types

// Include core types first, as they are used throughout the interface.
#include "core_types.h"

// Include Plan interface definitions (assuming these headers define the
// abstract base classes/interfaces)
#include "convolution.h"  // Defines ConvolutionPlan, CorrelationPlan interfaces
#include "cqt.h"          // Defines CQTPlan interface
#include "fft.h"          // Defines FFTPlan, RFFTPlan interfaces
#include "resample.h"     // Defines ResamplePlan interface
// #include "filter.h" // Include when filter module is ready

namespace OmniDSP {

// Forward declaration for the implementation class (Pimpl idiom)
namespace backend {
class OmniDSPImpl;
}  // namespace backend

/**
 * @brief Main class providing access to OmniDSP functionalities.
 * @details This class acts as the primary entry point for using the library.
 * It manages the selected backend and provides methods for performing common
 * DSP operations directly or creating stateful Plan objects for optimized
 * repeated operations (like FFTs, CQT, resampling). All operations invoked via
 * an instance of this class will use the backend configured for that specific
 * instance.
 *
 * Use the static OmniDSP::create() factory function to instantiate this class.
 * This class is non-copyable but movable.
 *
 * **Design Rationale:**
 *
 * The decision to center the API around an `OmniDSP` instance (which manages
 * the backend and acts as a factory) rather than using standalone functions was
 * made to:
 *
 * - **Increase Clarity:** Explicitly creating an `OmniDSP` object makes the
 * user aware they are interacting with a specific backend instance and its
 * configuration. Standalone functions can obscure this dependency.
 * - **Ensure Consistency:** Centralizing backend setup within the `OmniDSP`
 * constructor ensures that all operations (member methods and created Plans)
 * execute within a consistent backend context (e.g., regarding threading
 * settings).
 * - **Improve Control:** Users gain clearer control over the backend
 * configuration by managing the `OmniDSP` instance. This could potentially
 * allow for multiple instances with different configurations in the future.
 * - **Enhance Flexibility:** This architecture provides a natural point for
 * future extensions, such as runtime backend switching, without requiring major
 * changes to the user-facing API of the Plans or member methods.
 *
 * While this approach requires users to instantiate an `OmniDSP` object first,
 * the benefits in terms of transparency, consistency, control, and future
 * maintainability were deemed to outweigh the simplicity of standalone
 * functions.
 */
class OmniDSP {
 public:
  /**
   * @brief Factory function to create an OmniDSP instance.
   * @details Attempts to initialize the library with the specified backend.
   * @param backend The desired computation backend. Defaults to Backend::Stub.
   * @return An OmniExpected<OmniDSP> containing the created instance on
   * success, or a Status error code on failure (e.g.,
   * Status::UnsupportedFeature, Status::BackendError).
   */
  [[nodiscard]] static OmniExpected<OmniDSP> create(
      Backend backend = Backend::Stub);

  /**
   * @brief Destructor. Cleans up implementation resources.
   * @details Defined in the source file to handle the Pimpl destruction.
   */
  ~OmniDSP();

  /**
   * @brief Move constructor.
   * @details Defined in the source file.
   */
  OmniDSP(OmniDSP&& other) noexcept;

  /**
   * @brief Move assignment operator.
   * @details Defined in the source file.
   */
  OmniDSP& operator=(OmniDSP&& other) noexcept;

  // --- Delete Copy Semantics ---
  /** @brief Deleted copy constructor. OmniDSP instances are non-copyable. */
  OmniDSP(const OmniDSP&) = delete;
  /** @brief Deleted copy assignment operator. OmniDSP instances are
   * non-copyable. */
  OmniDSP& operator=(const OmniDSP&) = delete;

  /**
   * @brief Gets the backend currently used by this OmniDSP instance.
   * @return The active Backend enum value.
   */
  Backend get_backend() const;

  //-------------------------------------------------------------------------
  /** @defgroup DspOps DSP Operations (Member Methods)
   * @brief Functions for common DSP operations.
   * @details These member functions are called on an OmniDSP instance and
   * utilize the backend configured for that specific instance. For repeated
   * operations with the same parameters (e.g., FFT size, convolution kernel),
   * using the corresponding Plan object (created via factory methods) is
   * generally more efficient.
   * @{
   */
  //-------------------------------------------------------------------------

  /** @name Convolution and Correlation
   * @{
   */

  /**
   * @brief Performs 1D convolution of two real signals using the instance's
   * backend.
   * @tparam T The floating-point type (e.g., float, double).
   * @param input The first input signal.
   * @param kernel The second input signal (kernel).
   * @param mode The convolution mode (Full, Same, Valid).
   * @return An OmniExpected containing the convolution result vector on
   * success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> convolve(
      const std::vector<RealT<T>>& input, const std::vector<RealT<T>>& kernel,
      ConvolutionMode mode) const;  // Renamed from convolve1d

  /**
   * @brief Performs 1D convolution of two complex signals using the instance's
   * backend.
   * @tparam T The floating-point type for complex numbers (e.g., float,
   * double).
   * @param input The first input signal.
   * @param kernel The second input signal (kernel).
   * @param mode The convolution mode (Full, Same, Valid).
   * @return An OmniExpected containing the convolution result vector on
   * success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> convolve(
      const std::vector<ComplexT<T>>& input,
      const std::vector<ComplexT<T>>& kernel,
      ConvolutionMode mode) const;  // Renamed from convolve1d

  /**
   * @brief Performs 1D cross-correlation of two real signals using the
   * instance's backend.
   * @tparam T The floating-point type (e.g., float, double).
   * @param input The first input signal.
   * @param kernel The second input signal (template).
   * @param mode The correlation mode (Full, Same, Valid - reuses
   * ConvolutionMode).
   * @return An OmniExpected containing the correlation result vector on
   * success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> correlate(
      const std::vector<RealT<T>>& input, const std::vector<RealT<T>>& kernel,
      ConvolutionMode mode) const;  // Renamed from correlate1d

  /**
   * @brief Performs 1D cross-correlation of two complex signals using the
   * instance's backend.
   * @details Computes sum(input[n+k] * conj(kernel[k])).
   * @tparam T The floating-point type for complex numbers (e.g., float,
   * double).
   * @param input The first input signal.
   * @param kernel The second input signal (template).
   * @param mode The correlation mode (Full, Same, Valid - reuses
   * ConvolutionMode).
   * @return An OmniExpected containing the correlation result vector on
   * success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> correlate(
      const std::vector<ComplexT<T>>& input,
      const std::vector<ComplexT<T>>& kernel,
      ConvolutionMode mode) const;  // Renamed from correlate1d

  /** @} */  // End of Convolution and Correlation group

  /** @name Fourier Transforms (One-Off)
   * @brief Convenience functions for single FFT calculations. For repeated
   * transforms of the same size, using an FFTPlan or RFFTPlan is more
   * efficient.
   * @{
   */

  /**
   * @brief Computes the forward complex-to-complex FFT of an input signal.
   * @tparam T The floating-point type for complex numbers (e.g., float,
   * double).
   * @param input The complex input signal vector.
   * @return An OmniExpected containing the complex FFT result vector on
   * success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> fft(
      const std::vector<ComplexT<T>>& input) const;

  /**
   * @brief Computes the inverse complex-to-complex FFT (IFFT) of an input
   * signal.
   * @details The output is typically unnormalized; multiply by (1.0 / N) if
   * needed.
   * @tparam T The floating-point type for complex numbers (e.g., float,
   * double).
   * @param input The complex input signal vector (frequency domain).
   * @return An OmniExpected containing the complex IFFT result vector (time
   * domain) on success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> ifft(
      const std::vector<ComplexT<T>>& input) const;

  /**
   * @brief Computes the forward real-to-complex FFT (RFFT) of an input signal.
   * @details Returns the non-redundant complex coefficients (size N/2 + 1).
   * @tparam T The floating-point type for real numbers (e.g., float, double).
   * @param input The real input signal vector of length N.
   * @return An OmniExpected containing the complex RFFT result vector (size N/2
   * + 1) on success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> rfft(
      const std::vector<RealT<T>>& input) const;

  /**
   * @brief Computes the inverse complex-to-real FFT (IRFFT).
   * @details Takes the non-redundant complex coefficients (size N/2 + 1) and
   * returns the real signal of length N. The output is typically unnormalized;
   * multiply by (1.0 / N) if needed.
   * @tparam T The floating-point type for real numbers (e.g., float, double).
   * @param input The complex input signal vector (frequency domain, size N/2 +
   * 1).
   * @param output_length The desired length (N) of the output real signal. This
   * must correspond to the original signal length before rfft.
   * @return An OmniExpected containing the real IRFFT result vector (time
   * domain, size N) on success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> irfft(
      const std::vector<ComplexT<T>>& input, size_t output_length) const;

  /** @} */  // End of Fourier Transforms group

  /** @name Window Coefficient Generation
   * @details While window generation is often simple, these are provided as
   * const member functions for API consistency. They typically do not rely
   * heavily on backend acceleration but are executed within the instance's
   * context.
   * @{
   */

  // ... (window methods remain unchanged) ...
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> bartlett_window(
      size_t length) const;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> blackman_window(
      size_t length) const;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> flattop_window(
      size_t length) const;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> gaussian_window(
      size_t length, RealT<T> stddev) const;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> hamming_window(
      size_t length) const;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> hann_window(
      size_t length) const;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> kaiser_window(
      size_t length, RealT<T> beta) const;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> rectangular_window(
      size_t length) const;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> triangular_window(
      size_t length) const;

  /** @} */  // End of Window Coefficient Generation group
  /** @} */  // End of DspOps group

  //-------------------------------------------------------------------------
  // Plan Factory Methods (Member Methods - Depend on Backend)
  //-------------------------------------------------------------------------

  /**
   * @brief Creates a plan for performing complex-to-complex Fast Fourier
   * Transforms (FFTs).
   * @details Uses the instance's backend for potential acceleration. Provides
   * optimal performance for repeated transforms of the same size.
   * @tparam T The floating-point type for complex numbers (e.g., float,
   * double).
   * @param length The length of the FFT. Backend limitations may apply (e.g.,
   * power-of-two).
   * @return An OmniExpected containing a unique_ptr to the FFTPlan interface on
   * success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>> create_fft_plan(
      size_t length) const;

  /**
   * @brief Creates a plan for performing real-to-complex Fast Fourier
   * Transforms (RFFTs).
   * @details Uses the instance's backend for potential acceleration. Provides
   * optimal performance for repeated transforms of the same size.
   * @tparam T The floating-point type for real numbers (e.g., float, double).
   * @param length The length of the real input signal for the FFT. Backend
   * limitations may apply.
   * @return An OmniExpected containing a unique_ptr to the RFFTPlan interface
   * on success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>> create_rfft_plan(
      size_t length) const;

  /**
   * @brief Creates a plan for performing Constant-Q Transforms (CQTs).
   * @details Uses the instance's backend for potential acceleration.
   * @tparam T The floating-point type (e.g., float, double).
   * @param sample_rate The sample rate of the input signal in Hz.
   * @param min_freq The minimum frequency for the CQT analysis in Hz.
   * @param max_freq The maximum frequency for the CQT analysis in Hz.
   * @param bins_per_octave The number of frequency bins per octave.
   * @return An OmniExpected containing a unique_ptr to the CQTPlan interface on
   * success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>> create_cqt_plan(
      RealT<T> sample_rate, RealT<T> min_freq, RealT<T> max_freq,
      int bins_per_octave) const;

  /**
   * @brief Creates a plan for resampling signals.
   * @details Uses the instance's backend for potential acceleration.
   * @tparam T The floating-point type (e.g., float, double). Typically
   * RealT<T>.
   * @param input_rate The sample rate of the input signal.
   * @param output_rate The desired sample rate of the output signal.
   * @param max_input_size An estimate of the maximum input size for
   * pre-allocation/optimization.
   * @return An OmniExpected containing a unique_ptr to the ResamplePlan
   * interface on success, or a Status error code.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
  create_resample_plan(double input_rate, double output_rate,
                       size_t max_input_size) const;

  // TODO: Add declarations for create_convolution_plan,
  // create_correlation_plan, create_iir_filter_plan

 private:
  /**
   * @brief Private constructor used by the static create factory function.
   */
  OmniDSP(
      std::unique_ptr<backend::OmniDSPImpl> impl);  // Definition now in .cpp

  /**
   * @brief Pointer to the implementation object (Pimpl idiom).
   */
  std::unique_ptr<backend::OmniDSPImpl> pimpl_;
};

// Template method definitions for member functions are typically defined in the
// header file itself (if simple and not dependent on Impl) or in a separate
// "-inl.h" header file included at the end of this header, or explicitly
// instantiated in a .cpp file. Since these depend on pimpl_, their definitions
// MUST be in omnidsp.cpp.

}  // namespace OmniDSP

#endif  // OMNIDSP_H
