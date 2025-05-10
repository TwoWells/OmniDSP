/**
 * @file filter.hpp
 * @brief Defines the public API for FIR and IIR filter design and execution
 * Plans.
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

#include "OmniDSP/omnidsp_export.hpp"
#include "core_types.hpp"  // For Status, OmniExpected, F32, C32, Utils::IsComplex_v etc.
#include "window.hpp"  // For WindowSetup

// Corrected include path for Abstract::Backend.
// This assumes "src" is an include directory for the library's compilation,
// allowing "omnidsp/interface/backend.hpp" to be found.
#include "interface/backend.hpp"  // Defines Abstract::Backend

// Forward declare backend Impl classes (already present)
namespace OmniDSP::Abstract {
  template <typename T>
  class FIRFilterPlanImpl;
  template <typename T>
  class IIRFilterPlanImpl;
}  // namespace OmniDSP::Abstract

namespace OmniDSP {

  // class OmniDSPImpl; // Forward declaration if needed by friends

  /** @brief Enumeration of standard filter types. */
  enum class FilterType { Lowpass, Highpass, Bandpass, Bandstop };

  /** @brief Specification for designing an FIR filter. */
  struct OMNIDSP_EXPORT FIRFilterSpec {
    FilterType type = FilterType::Lowpass;
    size_t order = 64;
    double sample_rate = 1.0;
    double cutoff1 = 0.1;
    std::optional<double> cutoff2 = std::nullopt;
    WindowSetup window_setup{WindowType::Hamming, static_cast<int>(64 + 1)};

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
      if (window_setup.length <= 0
          || static_cast<size_t>(window_setup.length) != (order + 1)) {
        // Consider logging this specific validation failure if spdlog is
        // available spdlog::get("OmniDSP")->warn("FIRFilterSpec validation:
        // window_setup.length ({}) != order + 1 ({})", window_setup.length,
        // order + 1);
        return false;
      }
      // WindowSetup itself is validated on construction.
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

  template <typename T>
  using FIRCoefs = std::vector<T>;

  /** @brief Specification for designing an IIR filter. */
  struct OMNIDSP_EXPORT IIRFilterSpec {
    FilterType type = FilterType::Lowpass;
    size_t order = 4;
    double sample_rate = 1.0;
    double cutoff1 = 0.1;
    std::optional<double> cutoff2 = std::nullopt;
    std::optional<double> passband_ripple_db = std::nullopt;
    std::optional<double> stopband_attenuation_db = std::nullopt;

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
  template <typename T>  // T can be F32, F64, C32, C64
  class OMNIDSP_EXPORT FIRFilterPlan {
    // friend class OmniDSPImpl;
   public:
    ~FIRFilterPlan();                               // Definition in .cpp
    FIRFilterPlan(FIRFilterPlan&& other) noexcept;  // Definition in .cpp
    FIRFilterPlan& operator=(
        FIRFilterPlan&& other) noexcept;  // Definition in .cpp
    FIRFilterPlan(const FIRFilterPlan&) = delete;
    FIRFilterPlan& operator=(const FIRFilterPlan&) = delete;

    [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);
    Status reset();
    size_t get_order() const;
    size_t get_num_taps() const;

    /**
     * @brief Creates a FIRFilterPlan using the provided backend and
     * coefficients.
     * @param backend A reference to the backend implementation to use.
     * @param coefficients The FIR filter coefficients (taps).
     * @return An OmniExpected containing a unique_ptr to the FIRFilterPlan on
     * success, or a Status on failure.
     */
    [[nodiscard]] static OmniExpected<std::unique_ptr<FIRFilterPlan<T>>> create(
        const Abstract::Backend& backend, const std::vector<T>& coefficients)
    {
      OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<T>>>
          pimpl_expected;
      if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected = backend.create_fir_filter_plan_impl_f32(coefficients);
      }
      else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected = backend.create_fir_filter_plan_impl_f64(coefficients);
      }
      else if constexpr (std::is_same_v<T, C32>) {
        pimpl_expected = backend.create_fir_filter_plan_impl_c32(coefficients);
      }
      else if constexpr (std::is_same_v<T, C64>) {
        pimpl_expected = backend.create_fir_filter_plan_impl_c64(coefficients);
      }
      else {
        // This case should ideally be caught by a static_assert if T is
        // constrained. For now, returning UnsupportedFeature.
        // static_assert(always_false<T>::value, "Unsupported type for
        // FIRFilterPlan::create");
        return std::unexpected(Status::UnsupportedFeature);
      }

      if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
      }

      auto plan = FIRFilterPlan<T>::create_from_impl(
          std::move(pimpl_expected.value()));
      if (!plan) {
        // This could happen if 'new FIRFilterPlan<T>(...)' fails (e.g.
        // bad_alloc) or if pimpl_expected.value() was somehow null after a
        // success.
        return std::unexpected(Status::Failure);
      }
      return plan;
    }

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
  template <typename T>  // T is typically F32 or F64 (real)
  class OMNIDSP_EXPORT IIRFilterPlan {
    static_assert(
        !Utils::IsComplex_v<T>,
        "IIRFilterPlan typically requires a real type (F32 or F64).");
    // friend class OmniDSPImpl;
   public:
    ~IIRFilterPlan();                               // Definition in .cpp
    IIRFilterPlan(IIRFilterPlan&& other) noexcept;  // Definition in .cpp
    IIRFilterPlan& operator=(
        IIRFilterPlan&& other) noexcept;  // Definition in .cpp
    IIRFilterPlan(const IIRFilterPlan&) = delete;
    IIRFilterPlan& operator=(const IIRFilterPlan&) = delete;

    [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);
    Status reset();
    size_t get_order() const;
    size_t get_num_sections() const;

    /**
     * @brief Creates an IIRFilterPlan using the provided backend and SOS
     * coefficients.
     * @param backend A reference to the backend implementation to use.
     * @param sos_coefficients A vector of second-order section coefficients.
     * @return An OmniExpected containing a unique_ptr to the IIRFilterPlan on
     * success, or a Status on failure.
     */
    [[nodiscard]] static OmniExpected<std::unique_ptr<IIRFilterPlan<T>>> create(
        const Abstract::Backend& backend,
        const std::vector<IIRFilterCoef>& sos_coefficients)
    {
      OmniExpected<std::unique_ptr<Abstract::IIRFilterPlanImpl<T>>>
          pimpl_expected;
      if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected
            = backend.create_iir_filter_plan_impl_f32(sos_coefficients);
      }
      else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected
            = backend.create_iir_filter_plan_impl_f64(sos_coefficients);
      }
      else {
        // static_assert(always_false<T>::value, "Unsupported type for
        // IIRFilterPlan::create");
        return std::unexpected(Status::UnsupportedFeature);
      }

      if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
      }

      auto plan = IIRFilterPlan<T>::create_from_impl(
          std::move(pimpl_expected.value()));
      if (!plan) {
        return std::unexpected(Status::Failure);
      }
      return plan;
    }

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
