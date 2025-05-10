/**
 * @file resample.hpp
 * @brief Defines the interface for Resample Plan objects and related
 * specifications.
 */

#ifndef OMNIDSP_RESAMPLE_HPP
#define OMNIDSP_RESAMPLE_HPP

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For exception messages
#include <utility>    // For std::move
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"
#include "core_types.hpp"  // For Status, OmniExpected, Utils::IsComplex_v, F32, F64 etc.
#include "window.hpp"  // For WindowSetup

// Include Abstract::Backend for the static create method
#include "interface/backend.hpp"  // Defines Abstract::Backend

// spdlog for logging errors in ResampleSpec constructor (if not already
// included via core_types or window) #include "spdlog/spdlog.h"

// Forward declare backend Impl class (already present)
namespace OmniDSP::Abstract {
  template <typename T>
  class ResamplePlanImpl;
}  // namespace OmniDSP::Abstract

namespace OmniDSP {

  // class OmniDSPImpl; // Forward declaration if needed by friends

  /**
   * @brief Specification for creating a resampling plan.
   * Validated upon construction.
   */
  struct OMNIDSP_EXPORT ResampleSpec {
    double input_rate;
    double output_rate;
    int quality;
    WindowSetup window_setup;

    explicit ResampleSpec(
        double ir,
        double or_rate,
        int q,
        WindowSetup ws
        = WindowSetup{WindowType::Kaiser, 32, WindowParams{{"beta", 5.0}}})
        : input_rate(ir),
          output_rate(or_rate),
          quality(q),
          window_setup(std::move(ws))
    {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) {
        logger = spdlog::default_logger();
      }

      if (input_rate <= 0.0) {
        std::string msg = "ResampleSpec construction failed: input_rate ("
                          + std::to_string(input_rate) + ") must be positive.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (output_rate <= 0.0) {
        std::string msg = "ResampleSpec construction failed: output_rate ("
                          + std::to_string(output_rate) + ") must be positive.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (quality < 0 || quality > 15) {
        std::string msg = "ResampleSpec construction failed: quality ("
                          + std::to_string(quality)
                          + ") is out of expected range [0, 15].";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
    }
  };

  /**
   * @brief Plan object for executing signal resampling operations.
   * @tparam T The underlying REAL floating-point type (e.g., F32, F64).
   */
  template <typename T>
  class OMNIDSP_EXPORT ResamplePlan {
    static_assert(
        !Utils::IsComplex_v<T>,
        "ResamplePlan requires a real type (F32 or F64).");
    // friend class OmniDSPImpl;

   public:
    ~ResamplePlan();                              // Definition in .cpp
    ResamplePlan(ResamplePlan&& other) noexcept;  // Definition in .cpp
    ResamplePlan& operator=(
        ResamplePlan&& other) noexcept;  // Definition in .cpp
    ResamplePlan(const ResamplePlan&) = delete;
    ResamplePlan& operator=(const ResamplePlan&) = delete;

    [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);
    Status reset();
    double get_input_rate() const;
    double get_output_rate() const;
    size_t get_output_length(size_t input_length) const;

    /**
     * @brief Creates a ResamplePlan using the provided backend and
     * specification.
     * @param backend A reference to the backend implementation to use.
     * @param spec The ResampleSpec detailing the resampling parameters.
     * @return An OmniExpected containing a unique_ptr to the ResamplePlan on
     * success, or a Status on failure.
     */
    [[nodiscard]] static OmniExpected<std::unique_ptr<ResamplePlan<T>>> create(
        const Abstract::Backend& backend, const ResampleSpec& spec)
    {
      OmniExpected<std::unique_ptr<Abstract::ResamplePlanImpl<T>>>
          pimpl_expected;
      if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected = backend.create_resample_plan_impl_f32(spec);
      }
      else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected = backend.create_resample_plan_impl_f64(spec);
      }
      else {
        // static_assert(always_false<T>::value, "Unsupported real type for
        // ResamplePlan::create");
        return std::unexpected(Status::UnsupportedFeature);
      }

      if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
      }

      auto plan = ResamplePlan<T>::create_from_impl(
          std::move(pimpl_expected.value()));
      if (!plan) {
        return std::unexpected(Status::Failure);  // Or AllocationError
      }
      return plan;
    }

    static std::unique_ptr<ResamplePlan<T>> create_from_impl(
        std::unique_ptr<Abstract::ResamplePlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<ResamplePlan<T>>(
          new ResamplePlan<T>(std::move(pimpl)));
    }

   private:
    explicit ResamplePlan(std::unique_ptr<Abstract::ResamplePlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::ResamplePlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_RESAMPLE_HPP
