/**
 * @file filter.hpp
 * @brief Defines the public API for FIR and IIR filter design and execution
 * Plans.
 */

#ifndef OMNIDSP_FILTER_H
#define OMNIDSP_FILTER_H

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"
#include "core_types.hpp"
#include "window.hpp"  // Include for non-templated WindowSpec

// Forward declare backend Impl classes
namespace OmniDSP::backend {
  template <typename T>
  class FIRFilterPlanImpl;
  template <typename T>
  class IIRFilterPlanImpl;
}  // namespace OmniDSP::backend

namespace OmniDSP {

  class OmniDSPImpl;  // Forward declare instead of including full omnidsp.hpp
                      // if possible

  /** @brief Enumeration of standard filter types. */
  enum class FilterType { Lowpass, Highpass, Bandpass, Bandstop };

  /** @brief Specification for designing an FIR filter. (Non-Templated) */
  struct OMNIDSP_EXPORT FIRFilterSpec {
    FilterType type = FilterType::Lowpass;
    size_t order = 0;          ///< Desired filter order (0 for automatic).
    double sample_rate = 1.0;  ///< Sample rate for cutoff freq interpretation
                               ///< (often normalized to 1.0). Use double.
    double cutoff1 = 0.0;      ///< Primary cutoff frequency (normalized 0.0 to
                               ///< 0.5). Use double.
    std::optional<double> cutoff2
        = std::nullopt;  ///< Secondary cutoff frequency for bandpass/bandstop.
                         ///< Use double.
    WindowSpec window;   ///< Use non-templated WindowSpec.
    // Optional parameters for specific design methods (e.g., Parks-McClellan)
    std::optional<double> transition_width
        = std::nullopt;  ///< Desired transition width (normalized). Use double.
    std::optional<double> stopband_attenuation_db
        = std::nullopt;  ///< Desired stopband attenuation in dB. Use double.

    /**
     * @brief Validates the FIR filter specification.
     * @return True if the spec is valid, false otherwise.
     */
    [[nodiscard]] bool validate() const
    {
      if (sample_rate <= 0.0) return false;
      // Allow cutoff1 to be exactly 0 for DC blocking (highpass) or similar?
      // Check design methods. For now, keep strict > 0 check.
      if (cutoff1 <= 0.0 || cutoff1 >= 0.5 * sample_rate)
        return false;  // Cutoff must be within (0, Nyquist)

      if (type == FilterType::Bandpass || type == FilterType::Bandstop) {
        if (!cutoff2.has_value()) return false;  // Need second cutoff
        if (cutoff2.value() <= 0.0 || cutoff2.value() >= 0.5 * sample_rate)
          return false;
        if (cutoff2.value() <= cutoff1)
          return false;  // Ensure cutoff2 > cutoff1
      }
      else {
        // For low/high pass, cutoff2 should not be set
        if (cutoff2.has_value()) return false;
      }
      // Validate window spec
      if (!window.validate()) return false;

      // Add validation for transition_width and stopband_attenuation_db if
      // needed
      if (transition_width.has_value() && transition_width.value() <= 0.0)
        return false;
      if (stopband_attenuation_db.has_value()
          && stopband_attenuation_db.value() <= 0.0)
        return false;

      return true;
    }
  };

  /**
   * @brief Represents coefficients for a single second-order section (SOS) of
   * an IIR filter. Part of the "Coef" category representing design results.
   */
  struct OMNIDSP_EXPORT
      IIRFilterCoef {  // *** RENAMED from SecondOrderSection ***
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;  // Numerator coefficients. Use double.
    double a0 = 1.0;
    double a1 = 0.0;
    double a2 = 0.0;  // Denominator coefficients (a0 usually normalized to 1).
                      // Use double.
  };

  /**
   * @brief Type alias for FIR filter coefficients.
   * Part of the "Coef" category representing design results.
   * @tparam T The coefficient data type (F32 or F64).
   */
  template <typename T>
  using FIRFilterCoef = std::vector<T>;  // Conceptually useful alias

  /** @brief Specification for designing an IIR filter. (Non-Templated) */
  struct OMNIDSP_EXPORT IIRFilterSpec {
    FilterType type = FilterType::Lowpass;
    size_t order = 0;  ///< Filter order (must be > 0).
    double sample_rate
        = 1.0;  ///< Sample rate for cutoff freq interpretation. Use double.
    double cutoff1 = 0.0;  ///< Primary cutoff frequency (normalized 0.0 to
                           ///< 0.5). Use double.
    std::optional<double> cutoff2
        = std::nullopt;  ///< Secondary cutoff frequency for bandpass/bandstop.
                         ///< Use double.
    // Optional parameters for specific design methods (e.g., ripple)
    std::optional<double> passband_ripple_db = std::nullopt;  ///< Use double.
    std::optional<double> stopband_attenuation_db
        = std::nullopt;  ///< Use double. For Chebyshev II / Elliptic

    /**
     * @brief Validates the IIR filter specification.
     * @return True if the spec is valid, false otherwise.
     */
    [[nodiscard]] bool validate() const
    {
      if (order == 0) return false;  // IIR order must be specified
      if (sample_rate <= 0.0) return false;
      if (cutoff1 <= 0.0 || cutoff1 >= 0.5 * sample_rate) return false;

      if (type == FilterType::Bandpass || type == FilterType::Bandstop) {
        if (!cutoff2.has_value()) return false;
        if (cutoff2.value() <= 0.0 || cutoff2.value() >= 0.5 * sample_rate)
          return false;
        if (cutoff2.value() <= cutoff1) return false;
      }
      else {
        if (cutoff2.has_value()) return false;
      }
      // Add validation for ripple/attenuation if needed
      if (passband_ripple_db.has_value() && passband_ripple_db.value() <= 0.0)
        return false;
      if (stopband_attenuation_db.has_value()
          && stopband_attenuation_db.value() <= 0.0)
        return false;

      return true;
    }
  };

  /** @brief Plan object for executing Finite Impulse Response (FIR) filters. */
  template <typename T>  // Keep Plan templated on data type T
  class OMNIDSP_EXPORT FIRFilterPlan {
    friend class OmniDSPImpl;  // Allow OmniDSPImpl to access private
                               // constructor

   public:
    ~FIRFilterPlan();
    FIRFilterPlan(FIRFilterPlan&& other) noexcept;
    FIRFilterPlan& operator=(FIRFilterPlan&& other) noexcept;
    FIRFilterPlan(const FIRFilterPlan&) = delete;
    FIRFilterPlan& operator=(const FIRFilterPlan&) = delete;

    [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);
    Status reset();
    size_t get_order() const;
    size_t get_num_taps() const;  // num_taps = order + 1

    /**
     * @brief Public static helper for factory methods to create instances.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public FIRFilterPlan. Returns
     * nullptr if pimpl is null.
     */
    static std::unique_ptr<FIRFilterPlan<T>> create_from_impl(
        std::unique_ptr<backend::FIRFilterPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      // Use private constructor via helper struct or make constructor
      // public/protected For simplicity here, assume constructor is accessible
      // or use a factory pattern
      return std::unique_ptr<FIRFilterPlan<T>>(
          new FIRFilterPlan<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called ONLY by create_from_impl or friend
     * OmniDSPImpl. */
    explicit FIRFilterPlan(
        std::unique_ptr<backend::FIRFilterPlanImpl<T>> pimpl);

    std::unique_ptr<backend::FIRFilterPlanImpl<T>> pimpl_;
  };

  /** @brief Plan object for executing Infinite Impulse Response (IIR) filters.
   */
  template <typename T>  // Keep Plan templated on data type T
  class OMNIDSP_EXPORT IIRFilterPlan {
    static_assert(
        !Detail::is_complex_v<T>,
        "IIRFilterPlan typically requires a real type (F32 or F64).");
    friend class OmniDSPImpl;  // Allow OmniDSPImpl to access private
                               // constructor

   public:
    ~IIRFilterPlan();
    IIRFilterPlan(IIRFilterPlan&& other) noexcept;
    IIRFilterPlan& operator=(IIRFilterPlan&& other) noexcept;
    IIRFilterPlan(const IIRFilterPlan&) = delete;
    IIRFilterPlan& operator=(const IIRFilterPlan&) = delete;

    [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);
    Status reset();
    size_t get_order() const;
    size_t get_num_sections() const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public IIRFilterPlan. Returns
     * nullptr if pimpl is null.
     */
    static std::unique_ptr<IIRFilterPlan<T>> create_from_impl(
        std::unique_ptr<backend::IIRFilterPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<IIRFilterPlan<T>>(
          new IIRFilterPlan<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called ONLY by create_from_impl or friend
     * OmniDSPImpl. */
    explicit IIRFilterPlan(
        std::unique_ptr<backend::IIRFilterPlanImpl<T>> pimpl);

    std::unique_ptr<backend::IIRFilterPlanImpl<T>> pimpl_;
  };

  // Definitions MUST be provided in a .cpp file

}  // namespace OmniDSP

#endif  // OMNIDSP_FILTER_H
