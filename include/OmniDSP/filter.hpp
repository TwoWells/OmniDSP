/**
 * @file filter.hpp
 * @brief Defines the public API for FIR and IIR filter design and execution
 * Plans, and their Specifications.
 */

#ifndef OMNIDSP_FILTER_HPP
#define OMNIDSP_FILTER_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected, F32, C32, Utils::IsComplex_v etc.
#include "OmniDSP/types/filter.hpp"  // For FilterType, FIRFilterDesignMethod, IIRFilterFormat
#include "OmniDSP/window.hpp"  // For WindowSetup

// Corrected include path for Abstract::Backend.
#include "interface/backend.hpp"  // Defines Abstract::Backend

// Forward declare backend Impl classes (already present)
namespace OmniDSP::Abstract {
  template <typename T>
  class FIRFilterPlanImpl;
  template <typename T>
  class IIRFilterPlanImpl;
}  // namespace OmniDSP::Abstract

namespace OmniDSP {

  /**
   * @brief Fully resolved specification for a Finite Impulse Response (FIR)
   * filter. This struct is typically the output of a utility function like
   * `OmniDSP::Utils::create_spec` and serves as direct input for creating an
   * `FIRFilterPlan`. It contains all necessary parameters, with filter order
   * and window length definitively set.
   */
  struct OMNIDSP_EXPORT FIRFilterSpec {
    FilterType type;  ///< Type of filter (Lowpass, Highpass, etc.).
    size_t order;  ///< Filter order (number of taps - 1). This is definitively
                   ///< set.
    double sample_rate;             ///< Sample rate of the signal in Hz.
    double cutoff1;                 ///< Primary cutoff frequency in Hz.
    std::optional<double> cutoff2;  ///< Optional secondary cutoff frequency in
                                    ///< Hz (for bandpass/bandstop).
    WindowSetup window_setup;  ///< Windowing function setup to be used. Its
                               ///< `length` member will be `order + 1`.
    // FIRFilterDesignMethod design_method; // This might be part of Params, but
    // Spec is the result. The backend uses the spec to generate coeffs, method
    // isn't needed by backend *after* spec is made. However, if different
    // backends might generate coeffs differently from the *same* spec, it could
    // be here. For now, assuming design_method guides Utils::create_spec.

    /**
     * @brief Constructor for FIRFilterSpec.
     * Primarily intended for internal use by `Utils::create_spec`.
     * Assumes parameters have been validated and resolved.
     */
    explicit FIRFilterSpec(
        FilterType p_type,
        size_t p_order,
        double p_sample_rate,
        double p_cutoff1,
        std::optional<double> p_cutoff2,
        WindowSetup p_window_setup  // WindowSetup's length should be consistent
                                    // with p_order
        )
        : type(p_type),
          order(p_order),
          sample_rate(p_sample_rate),
          cutoff1(p_cutoff1),
          cutoff2(std::move(p_cutoff2)),
          window_setup(std::move(p_window_setup))
    {
      // Basic internal consistency check: window_setup.length should match
      // order + 1
      if (static_cast<size_t>(this->window_setup.length) != (this->order + 1)) {
        // This indicates an internal error in how Utils::create_spec populated
        // this. Consider throwing or logging a critical error. For now, we
        // could adjust it, or rely on Utils::create_spec to set it correctly.
        // Let's assume Utils::create_spec sets it correctly.
        // If not, an assert could be useful here during debugging:
        // assert(static_cast<size_t>(this->window_setup.length) == (this->order
        // + 1));
      }
    }

    // Default constructor might be needed if it's a member of another class
    // that's default-constructed. However, a Spec should ideally always be
    // fully initialized. FIRFilterSpec() = default; // Or delete it if it
    // should always be fully specified.

    [[nodiscard]] size_t num_taps() const { return order + 1; }

    // A simple validation, mostly for internal consistency.
    // The main validation burden is on FIRFilterParams constructor and
    // Utils::create_spec.
    [[nodiscard]] bool validate_consistency() const
    {
      if (sample_rate <= 0.0) return false;
      if (cutoff1 <= 0.0 || cutoff1 >= 0.5 * sample_rate) return false;
      if (type == FilterType::Bandpass || type == FilterType::Bandstop) {
        if (!cutoff2.has_value() || cutoff2.value() <= 0.0
            || cutoff2.value() >= 0.5 * sample_rate
            || cutoff2.value() <= cutoff1) {
          return false;
        }
      }
      else {
        if (cutoff2.has_value()) return false;
      }
      if (static_cast<size_t>(window_setup.length) != (order + 1)
          && window_setup.length != 0) {
        // Allow length 0 if it means "to be set by order", but by this stage it
        // should be set. This check is more of an assertion that
        // Utils::create_spec did its job.
        return false;
      }
      return true;
    }
  };

  /** @brief Represents coefficients for a single second-order section (SOS) of
   * an IIR filter.
   */
  struct OMNIDSP_EXPORT IIRFilterCoef {
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a0 = 1.0;  // Typically normalized to 1 for SOS
    double a1 = 0.0;
    double a2 = 0.0;
  };

  /** @brief Specification for designing an IIR filter. */
  struct OMNIDSP_EXPORT IIRFilterSpec {
    FilterType type = FilterType::Lowpass;
    size_t order = 4;  // Effective order
    double sample_rate = 1.0;
    double cutoff1 = 0.1;
    std::optional<double> cutoff2 = std::nullopt;
    std::optional<double> passband_ripple_db
        = std::nullopt;  // e.g., for Chebyshev, Elliptic
    std::optional<double> stopband_attenuation_db
        = std::nullopt;  // e.g., for Chebyshev, Elliptic, Kaiserord (FIR)
    IIRFilterFormat output_format
        = IIRFilterFormat::SOS;  // Preferred output format for coefficients

    // Constructor for member initialization (used by Utils::create_spec)
    explicit IIRFilterSpec(
        FilterType p_type,
        size_t p_order,
        double p_sample_rate,
        double p_cutoff1,
        std::optional<double> p_cutoff2,
        std::optional<double> p_passband_ripple_db,
        std::optional<double> p_stopband_attenuation_db,
        IIRFilterFormat p_output_format = IIRFilterFormat::SOS)
        : type(p_type),
          order(p_order),
          sample_rate(p_sample_rate),
          cutoff1(p_cutoff1),
          cutoff2(std::move(p_cutoff2)),
          passband_ripple_db(std::move(p_passband_ripple_db)),
          stopband_attenuation_db(std::move(p_stopband_attenuation_db)),
          output_format(p_output_format)
    {}

    // Basic validation
    [[nodiscard]] bool validate_consistency() const
    {
      if (order == 0) return false;
      if (sample_rate <= 0.0) return false;
      if (cutoff1 <= 0.0 || cutoff1 >= 0.5 * sample_rate) return false;
      if (type == FilterType::Bandpass || type == FilterType::Bandstop) {
        if (!cutoff2.has_value() || cutoff2.value() <= 0.0
            || cutoff2.value() >= 0.5 * sample_rate
            || cutoff2.value() <= cutoff1) {
          return false;
        }
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

  template <typename T>
  using FIRCoefs = std::vector<T>;  // This alias is fine here or in core_types
                                    // if very general

  /** @brief Plan object for executing Finite Impulse Response (FIR) filters. */
  template <typename T>  // T can be F32, F64, C32, C64
  class OMNIDSP_EXPORT FIRFilterPlan {
   public:
    ~FIRFilterPlan();
    FIRFilterPlan(FIRFilterPlan&& other) noexcept;
    FIRFilterPlan& operator=(FIRFilterPlan&& other) noexcept;
    FIRFilterPlan(const FIRFilterPlan&) = delete;
    FIRFilterPlan& operator=(const FIRFilterPlan&) = delete;

    [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);
    Status reset();
    size_t get_order() const;
    size_t get_num_taps() const;

    [[nodiscard]] static OmniExpected<std::unique_ptr<FIRFilterPlan<T>>> create(
        const Abstract::Backend& backend, const FIRFilterSpec& spec);

    [[nodiscard]] static OmniExpected<std::unique_ptr<FIRFilterPlan<T>>> create(
        const Abstract::Backend& backend, const FIRCoefs<T>& coefficients);

    static std::unique_ptr<FIRFilterPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::FIRFilterPlanImpl<T>> pimpl);

   private:
    explicit FIRFilterPlan(
        std::unique_ptr<Abstract::FIRFilterPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::FIRFilterPlanImpl<T>> pimpl_;
  };

  /** @brief Plan object for executing Infinite Impulse Response (IIR) filters.
   */
  template <typename T>  // T is typically F32 or F64 (real)
  class OMNIDSP_EXPORT IIRFilterPlan {
    static_assert(
        !Utils::IsComplex_v<T>,
        "IIRFilterPlan typically requires a real type (F32 or F64).");

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

    [[nodiscard]] static OmniExpected<std::unique_ptr<IIRFilterPlan<T>>> create(
        const Abstract::Backend& backend, const IIRFilterSpec& spec);

    [[nodiscard]] static OmniExpected<std::unique_ptr<IIRFilterPlan<T>>> create(
        const Abstract::Backend& backend,
        const std::vector<IIRFilterCoef>& sos_coefficients);

    static std::unique_ptr<IIRFilterPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::IIRFilterPlanImpl<T>> pimpl);

   private:
    explicit IIRFilterPlan(
        std::unique_ptr<Abstract::IIRFilterPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::IIRFilterPlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_FILTER_HPP
