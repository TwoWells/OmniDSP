/**
 * @file filter.h
 * @brief Defines the public API for FIR and IIR filter design and execution
 * Plans.
 * @details Provides interfaces for creating and using filter plans, which
 * encapsulate filter coefficients and state for efficient processing. Filter
 * design parameters are specified using spec structs, and plans are created via
 * factory methods on the main OmniDSP class.
 */

#ifndef OMNIDSP_FILTER_H
#define OMNIDSP_FILTER_H

#include <complex>
#include <cstddef>
#include <memory>    // For std::unique_ptr
#include <optional>  // For optional parameters in specs
#include <span>      // For input/output views (requires C++20)
#include <vector>

#include "core_types.h"  // Core types like RealT, ComplexT, Status, OmniExpected
#include "window.h"      // Include WindowSpec for FIR filter design

// Include the generated export header for DLL support
#include "OmniDSP/omnidsp_export.h"

namespace OmniDSP {

// Forward declare the main OmniDSP class (needed for friend declaration)
class OmniDSP;

// Forward declarations for implementation classes (Pimpl idiom)
namespace backend {
template <typename T>
class FIRFilterPlanImpl;
template <typename T>
class IIRFilterPlanImpl;
}  // namespace backend

/**
 * @brief Enumeration of standard filter types.
 */
enum class FilterType {
  Lowpass,
  Highpass,
  Bandpass,
  Bandstop
  // Could add others like Differentiator, Hilbert later
};

/**
 * @brief Specification for designing an FIR filter.
 * @details Used as input to OmniDSP::design_fir_filter or
 * OmniDSP::create_fir_filter_plan. Currently supports window-based design.
 * @tparam T The floating-point type (e.g., float, double).
 */
template <typename T>
struct FIRFilterSpec {
  FilterType type = FilterType::Lowpass;  ///< The desired filter type.
  size_t order = 0;  ///< Filter order (number of taps - 1). Must be > 0.
  RealT<T> sample_rate =
      0.0;  ///< Sample rate of the signal to be filtered (Hz). Must be > 0.
  RealT<T> cutoff1 = 0.0;  ///< Primary cutoff frequency (Hz). For Low/High
                           ///< pass. Must be > 0 and < sample_rate/2.
  std::optional<RealT<T>> cutoff2 =
      std::nullopt;  ///< Secondary cutoff frequency (Hz). For
                     ///< Bandpass/Bandstop. Must be > 0 and < sample_rate/2.
  WindowSpec<T> window;  ///< Specification for the window function (e.g., Hann,
                         ///< Kaiser with beta). Defaults to Hann.

  // Add validation method?
  bool validate() const {
    if (order == 0 || sample_rate <= 0.0 || cutoff1 <= 0.0 ||
        cutoff1 >= sample_rate / 2.0) {
      return false;
    }
    if (type == FilterType::Bandpass || type == FilterType::Bandstop) {
      if (!cutoff2.has_value() || cutoff2.value() <= 0.0 ||
          cutoff2.value() >= sample_rate / 2.0 || cutoff2.value() <= cutoff1) {
        // Ensure cutoff2 exists, is valid, and > cutoff1 for band filters
        return false;
      }
    } else {
      if (cutoff2.has_value()) {
        // Cutoff2 should not be set for low/high pass
        return false;
      }
    }
    // Add validation for window parameters if needed (e.g., Kaiser beta)
    return true;
  }
};

/**
 * @brief Represents coefficients for a single second-order section (SOS) of an
 * IIR filter.
 * @details IIR filters are typically implemented as a cascade of SOS for better
 * numerical stability. Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1
 * + a1*z^-1 + a2*z^-2) Note the convention: a0 is implicitly 1.
 * @tparam T The floating-point type (e.g., float, double).
 */
template <typename T>
struct SecondOrderSection {
  RealT<T> b0 = 1.0;  ///< Numerator coefficient b0.
  RealT<T> b1 = 0.0;  ///< Numerator coefficient b1.
  RealT<T> b2 = 0.0;  ///< Numerator coefficient b2.
  RealT<T> a1 = 0.0;  ///< Denominator coefficient a1 (a0 is implicitly 1).
  RealT<T> a2 = 0.0;  ///< Denominator coefficient a2.
};

/**
 * @brief Specification for designing an IIR filter.
 * @details Used as input to OmniDSP::design_iir_filter or
 * OmniDSP::create_iir_filter_plan. Specifies parameters for common IIR filter
 * types like Butterworth or Chebyshev. The design result is typically
 * represented as a series of Second-Order Sections (SOS).
 * @tparam T The floating-point type (e.g., float, double).
 */
template <typename T>
struct IIRFilterSpec {
  // TODO: Define IIR filter design parameters. Examples:
  // enum class IIRDesignType { Butterworth, ChebyshevI, ChebyshevII, Elliptic
  // }; IIRDesignType design = IIRDesignType::Butterworth;
  FilterType type = FilterType::Lowpass;  ///< The desired filter type (Lowpass,
                                          ///< Highpass, etc.).
  size_t order = 0;                       ///< Filter order. Must be > 0.
  RealT<T> sample_rate =
      0.0;  ///< Sample rate of the signal to be filtered (Hz). Must be > 0.
  RealT<T> cutoff1 =
      0.0;  ///< Primary cutoff frequency (Hz). Must be > 0 and < sample_rate/2.
  std::optional<RealT<T>> cutoff2 =
      std::nullopt;  ///< Secondary cutoff frequency (Hz) for Bandpass/Bandstop.
                     ///< Must be > 0 and < sample_rate/2.
  // std::optional<RealT<T>> passband_ripple_db = std::nullopt; // For Chebyshev
  // I, Elliptic std::optional<RealT<T>> stopband_attenuation_db = std::nullopt;
  // // For Chebyshev II, Elliptic

  // Add validation method?
  bool validate() const {
    if (order == 0 || sample_rate <= 0.0 || cutoff1 <= 0.0 ||
        cutoff1 >= sample_rate / 2.0) {
      return false;
    }
    if (type == FilterType::Bandpass || type == FilterType::Bandstop) {
      if (!cutoff2.has_value() || cutoff2.value() <= 0.0 ||
          cutoff2.value() >= sample_rate / 2.0 || cutoff2.value() <= cutoff1) {
        return false;
      }
    } else {
      if (cutoff2.has_value()) return false;
    }
    // Add checks for ripple/attenuation if those parameters are added
    return true;
  }
};

/**
 * @brief Plan object for executing Finite Impulse Response (FIR) filters.
 * @details Encapsulates the filter coefficients and potentially state (for
 * overlap-add/save) for efficient execution. Created via
 * OmniDSP::create_fir_filter_plan. Uses the Pimpl idiom to hide
 * backend-specific implementation details. This class is non-copyable but
 * movable.
 * @tparam T The data type for filtering (e.g., float, double,
 * std::complex<float>).
 */
template <typename T>
class OMNIDSP_EXPORT FIRFilterPlan {
  friend class OmniDSP;  // Allow OmniDSP factory methods to call private
                         // constructor

 public:
  /** @brief Destructor. */
  ~FIRFilterPlan();
  /** @brief Move constructor. */
  FIRFilterPlan(FIRFilterPlan&& other) noexcept;
  /** @brief Move assignment operator. */
  FIRFilterPlan& operator=(FIRFilterPlan&& other) noexcept;
  /** @brief Deleted copy constructor. */
  FIRFilterPlan(const FIRFilterPlan&) = delete;
  /** @brief Deleted copy assignment operator. */
  FIRFilterPlan& operator=(const FIRFilterPlan&) = delete;

  /**
   * @brief Applies the FIR filter to an input signal.
   * @details Filtering may be implemented using direct convolution, FFT
   * convolution, or overlap-add/save methods depending on the backend and plan
   * configuration. This method maintains the filter state between calls for
   * continuous processing.
   * @param input A span representing the input signal segment.
   * @param output A span representing the output buffer. Must be large enough
   * to hold the corresponding output segment (typically same size as input).
   * @return Status::Success on success, or an error code on failure.
   * @note The output span must be pre-allocated.
   */
  [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);

  /**
   * @brief Resets the internal state of the filter.
   * @details Call this if processing discontinuous segments of data.
   * @return Status::Success on success.
   */
  Status reset();

  /**
   * @brief Gets the order of the FIR filter.
   * @return The filter order (number of taps - 1).
   */
  size_t get_order() const;

  /**
   * @brief Gets the number of taps in the FIR filter.
   * @return The number of filter taps (coefficients).
   */
  size_t get_num_taps() const;

  // Potentially add: get_coefficients(), get_latency() ?

 private:
  /**
   * @brief Private constructor, called ONLY by OmniDSP factory methods.
   * @param pimpl A unique_ptr to the backend-specific implementation.
   */
  explicit FIRFilterPlan(std::unique_ptr<backend::FIRFilterPlanImpl<T>> pimpl);

  /** @brief Pointer to the implementation object (Pimpl idiom). */
  std::unique_ptr<backend::FIRFilterPlanImpl<T>> pimpl_;
};

/**
 * @brief Plan object for executing Infinite Impulse Response (IIR) filters.
 * @details Encapsulates the filter coefficients (typically as Second-Order
 * Sections, SOS) and state for efficient execution. Created via
 * OmniDSP::create_iir_filter_plan. Uses the Pimpl idiom to hide
 * backend-specific implementation details. This class is non-copyable but
 * movable.
 * @tparam T The data type for filtering (e.g., float, double). Note: IIR
 * filtering is typically only defined for real signals. Complex IIR filtering
 * is less common.
 */
template <typename T>
class OMNIDSP_EXPORT IIRFilterPlan {
  friend class OmniDSP;  // Allow OmniDSP factory methods to call private
                         // constructor

 public:
  /** @brief Destructor. */
  ~IIRFilterPlan();
  /** @brief Move constructor. */
  IIRFilterPlan(IIRFilterPlan&& other) noexcept;
  /** @brief Move assignment operator. */
  IIRFilterPlan& operator=(IIRFilterPlan&& other) noexcept;
  /** @brief Deleted copy constructor. */
  IIRFilterPlan(const IIRFilterPlan&) = delete;
  /** @brief Deleted copy assignment operator. */
  IIRFilterPlan& operator=(const IIRFilterPlan&) = delete;

  /**
   * @brief Applies the IIR filter (cascade of SOS) to an input signal.
   * @details This method maintains the filter state (delay elements for each
   * SOS) between calls for continuous processing.
   * @param input A span representing the input signal segment.
   * @param output A span representing the output buffer. Must be large enough
   * to hold the corresponding output segment (typically same size as input).
   * @return Status::Success on success, or an error code on failure.
   * @note The output span must be pre-allocated.
   */
  [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);

  /**
   * @brief Resets the internal state of the filter (delay elements).
   * @details Call this if processing discontinuous segments of data.
   * @return Status::Success on success.
   */
  Status reset();

  /**
   * @brief Gets the order of the IIR filter.
   * @return The filter order.
   */
  size_t get_order() const;

  /**
   * @brief Gets the number of second-order sections used in the filter
   * implementation.
   * @return The number of SOS.
   */
  size_t get_num_sections() const;

  // Potentially add: get_sos_coefficients(), get_latency() ?

 private:
  /**
   * @brief Private constructor, called ONLY by OmniDSP factory methods.
   * @param pimpl A unique_ptr to the backend-specific implementation.
   */
  explicit IIRFilterPlan(std::unique_ptr<backend::IIRFilterPlanImpl<T>> pimpl);

  /** @brief Pointer to the implementation object (Pimpl idiom). */
  std::unique_ptr<backend::IIRFilterPlanImpl<T>> pimpl_;
};

// --- Template Implementations (Definitions) ---
// Definitions for template class methods (constructors, destructors, move ops,
// execute, getters) MUST be provided in the corresponding .cpp file (e.g.,
// filter.cpp or potentially fir_filter.cpp / iir_filter.cpp).

}  // namespace OmniDSP

#endif  // OMNIDSP_FILTER_H
