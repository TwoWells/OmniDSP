/**
 * @file filter.hpp
 * @brief Defines the public API for FIR and IIR filter design and execution
 * Plans.
 */

#ifndef OMNIDSP_FILTER_HPP  // Changed guard name
#define OMNIDSP_FILTER_HPP

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
namespace OmniDSP::Abstract {
  template <typename T>
  class FIRFilterPlanImpl;
  template <typename T>
  class IIRFilterPlanImpl;
}  // namespace OmniDSP::Abstract

namespace OmniDSP {

  class OmniDSPImpl;  // Forward declare instead of including full omnidsp.hpp
                      // if possible

  /** @brief Enumeration of standard filter types. */
  enum class FilterType { Lowpass, Highpass, Bandpass, Bandstop };

  /** @brief Specification for designing an FIR filter. (Non-Templated) */
  struct OMNIDSP_EXPORT FIRFilterSpec {
    FilterType type = FilterType::Lowpass;
    size_t order = 0;  ///< Desired filter order (0 for automatic estimation if
                       ///< supported by design method).
    double sample_rate = 1.0;  ///< Sample rate for cutoff freq interpretation
                               ///< (Hz or normalized). Use double.
    double cutoff1 = 0.0;  ///< Primary cutoff frequency (Hz or normalized 0.0
                           ///< to 0.5). Use double.
    std::optional<double> cutoff2
        = std::nullopt;  ///< Secondary cutoff frequency for bandpass/bandstop.
                         ///< Use double.
    WindowSpec window;   ///< Use non-templated WindowSpec. Defaults to
                         ///< Rectangular if default constructed.
    // Optional parameters for specific design methods (e.g., Parks-McClellan)
    std::optional<double> transition_width
        = std::nullopt;  ///< Desired transition width (normalized). Use double.
    std::optional<double> stopband_attenuation_db
        = std::nullopt;  ///< Desired stopband attenuation in dB. Use double.

    /** @brief Validates the FIR filter specification. */
    [[nodiscard]] bool validate() const
    {
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
      if (!window.validate()) return false;
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
   * an IIR filter.
   */
  struct OMNIDSP_EXPORT IIRFilterCoef {
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a0 = 1.0;
    double a1 = 0.0;
    double a2 = 0.0;
  };

  // *** ADDED Type Alias for FIR Coefficients ***
  /**
   * @brief Type alias for FIR filter coefficients (taps).
   * @tparam T The coefficient data type (F32 or F64). Complex FIR filters also
   * use this.
   */
  template <typename T>
  using FIRCoefs = std::vector<T>;

  /** @brief Specification for designing an IIR filter. (Non-Templated) */
  struct OMNIDSP_EXPORT IIRFilterSpec {
    FilterType type = FilterType::Lowpass;
    size_t order = 0;  ///< Filter order (must be > 0).
    double sample_rate = 1.0;
    double cutoff1 = 0.0;
    std::optional<double> cutoff2 = std::nullopt;
    std::optional<double> passband_ripple_db = std::nullopt;
    std::optional<double> stopband_attenuation_db = std::nullopt;

    /** @brief Validates the IIR filter specification. */
    [[nodiscard]] bool validate() const
    {
      if (order == 0) return false;
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
    // [[nodiscard]] std::span<const T> get_coefficients() const; // Optional
    // public getter

    static std::unique_ptr<FIRFilterPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::FIRFilterPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<FIRFilterPlan<T>>(
          new FIRFilterPlan<T>(std::move(pimpl)));
    }

   private:
    explicit FIRFilterPlan(
        std::unique_ptr<Abstract::FIRFilterPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::FIRFilterPlanImpl<T>> pimpl_;
  };

  /** @brief Plan object for executing Infinite Impulse Response (IIR) filters.
   */
  template <typename T>  // Keep Plan templated on data type T
  class OMNIDSP_EXPORT IIRFilterPlan {
    // *** UPDATED Namespace ***
    static_assert(
        !Utils::IsComplex_v<T>,
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

    static std::unique_ptr<IIRFilterPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::IIRFilterPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<IIRFilterPlan<T>>(
          new IIRFilterPlan<T>(std::move(pimpl)));
    }

   private:
    explicit IIRFilterPlan(
        std::unique_ptr<Abstract::IIRFilterPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::IIRFilterPlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_FILTER_HPP
