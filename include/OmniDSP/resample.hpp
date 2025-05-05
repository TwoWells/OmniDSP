/**
 * @file resample.hpp
 * @brief Defines the interface for Resample Plan objects and related
 * specifications.
 */

#ifndef OMNIDSP_RESAMPLE_H
#define OMNIDSP_RESAMPLE_H

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"
#include "core_types.hpp"
#include "window.hpp"  // Include for non-templated WindowSpec

// Forward declare backend Impl class
namespace OmniDSP::abstract {
  template <typename T>
  class ResamplePlanImpl;
}  // namespace OmniDSP::abstract

namespace OmniDSP {

  class OmniDSPImpl;  // Forward declare instead of including full omnidsp.hpp

  /**
   * @brief Specification for creating a resampling plan.
   */
  struct OMNIDSP_EXPORT ResampleSpec {
    double input_rate = 0.0;   ///< Input sample rate in Hz.
    double output_rate = 0.0;  ///< Desired output sample rate in Hz.
    int quality = 12;          ///< Quality setting (e.g., controls filter
                       ///< order/transition band). Higher is better quality but
                       ///< more costly. Interpretation depends on backend.
    WindowSpec window;  ///< *** UPDATED: Use non-templated WindowSpec (defaults
                        ///< to Rectangular). ***

    /**
     * @brief Validates the resampling specification.
     * @return True if the spec is valid, false otherwise.
     */
    [[nodiscard]] bool validate() const
    {
      if (input_rate <= 0.0 || output_rate <= 0.0) {
        return false;
      }
      if (quality < 0) {  // Example validation for quality
        return false;
      }
      // Validate the contained window spec
      if (!window.validate()) {
        return false;
      }
      return true;
    }
  };

  /**
   * @brief Plan object for executing signal resampling operations.
   * @tparam T The underlying REAL floating-point type (e.g., float, double).
   */
  template <typename T>  // T is REAL type here
  class OMNIDSP_EXPORT ResamplePlan {
    static_assert(
        !Utils::IsComplex_v<T>,
        "ResamplePlan requires a real type (F32 or F64).");
    friend class OmniDSPImpl;  // Allow OmniDSPImpl to access private
                               // constructor

   public:
    ~ResamplePlan();
    ResamplePlan(ResamplePlan&& other) noexcept;
    ResamplePlan& operator=(ResamplePlan&& other) noexcept;
    ResamplePlan(const ResamplePlan&) = delete;
    ResamplePlan& operator=(const ResamplePlan&) = delete;

    [[nodiscard]] Status execute(std::span<const T> input, std::span<T> output);
    Status reset();
    double get_input_rate() const;
    double get_output_rate() const;
    size_t get_output_length(size_t input_length) const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public ResamplePlan. Returns
     * nullptr if pimpl is null.
     */
    static std::unique_ptr<ResamplePlan<T>> create_from_impl(
        std::unique_ptr<abstract::ResamplePlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<ResamplePlan<T>>(
          new ResamplePlan<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called by create_from_impl or friend
     * OmniDSPImpl. */
    explicit ResamplePlan(std::unique_ptr<abstract::ResamplePlanImpl<T>> pimpl);

    std::unique_ptr<abstract::ResamplePlanImpl<T>> pimpl_;
  };

  // Definitions MUST be provided in a .cpp file

}  // namespace OmniDSP

#endif  // OMNIDSP_RESAMPLE_H
