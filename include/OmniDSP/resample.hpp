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

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT
#include "core_types.hpp"  // For Status, OmniExpected, Utils::IsComplex_v, F32, F64 etc.
#include "filter.hpp"  // For FIRFilterSpec (which ResampleSpec will embed)
#include "interface/backend.hpp"  // Defines Abstract::Backend and Abstract::ResamplePlanImpl
#include "window.hpp"  // For WindowSetup (used by ResampleParams, and indirectly by FIRFilterSpec)

// Forward declaration for ResampleParams, defined in
// "OmniDSP/params/resample.hpp" namespace OmniDSP { struct ResampleParams; }

namespace OmniDSP {

  /**
   * @brief Fully resolved specification for creating a resampling plan.
   * This struct is typically the output of `OmniDSP::Utils::create_spec(const
   * ResampleParams&)` and serves as direct input for creating a `ResamplePlan`.
   * It contains the definitive parameters for the resampling operation,
   * including the specification for the internal anti-aliasing/anti-imaging
   * filter.
   */
  struct OMNIDSP_EXPORT ResampleSpec {
    double input_rate;     ///< Input sample rate in Hz.
    double output_rate;    ///< Desired output sample rate in Hz.
    int quality;           ///< Quality setting used for the resampling process
                           ///< (influences filter design).
    size_t up_factor_L;    ///< Calculated upsampling factor.
    size_t down_factor_M;  ///< Calculated downsampling factor.
    FIRFilterSpec prototype_fir_spec;  ///< Specification for the internal
                                       ///< prototype FIR filter.

    explicit ResampleSpec(
        double p_input_rate,
        double p_output_rate,
        int p_quality,
        size_t p_up_factor_L,
        size_t p_down_factor_M,
        FIRFilterSpec p_prototype_fir_spec)
        : input_rate(p_input_rate),
          output_rate(p_output_rate),
          quality(p_quality),
          up_factor_L(p_up_factor_L),
          down_factor_M(p_down_factor_M),
          prototype_fir_spec(std::move(p_prototype_fir_spec))
    {
      if (input_rate <= 0.0 || output_rate <= 0.0) {
        throw std::logic_error(
            "ResampleSpec: Input/Output rates must be positive.");
      }
      if (up_factor_L == 0 || down_factor_M == 0) {
        throw std::logic_error(
            "ResampleSpec: Upsampling/Downsampling factors L/M cannot be "
            "zero.");
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

    // Declaration of the static create method
    [[nodiscard]] static OmniExpected<std::unique_ptr<ResamplePlan<T>>> create(
        const Abstract::Backend& backend, const ResampleSpec& spec);

    // Definition of create_from_impl remains in the header as it's simple and
    // commonly inlined
    static std::unique_ptr<ResamplePlan<T>> create_from_impl(
        std::unique_ptr<Abstract::ResamplePlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      // Use private constructor
      return std::unique_ptr<ResamplePlan<T>>(
          new ResamplePlan<T>(std::move(pimpl)));
    }

   private:
    // Private constructor, called by create_from_impl
    explicit ResamplePlan(std::unique_ptr<Abstract::ResamplePlanImpl<T>> pimpl)
        : pimpl_(std::move(pimpl))
    {
      if (!pimpl_) {
        throw std::runtime_error(
            "ResamplePlan constructed with null implementation pointer.");
      }
    }
    std::unique_ptr<Abstract::ResamplePlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_RESAMPLE_HPP
