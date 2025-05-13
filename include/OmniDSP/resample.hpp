/**
 * @file resample.hpp
 * @brief Defines the interface for Resample Processor objects and related
 * design specifications (Design::Resample).
 */

#ifndef OMNIDSP_RESAMPLE_HPP
#define OMNIDSP_RESAMPLE_HPP

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <string>
#include <utility>
#include <vector>

#include "OmniDSP/core_types.hpp"
#include "OmniDSP/omnidsp_export.hpp"
// Design::FIRFilter is part of Design::Resample
#include "OmniDSP/design/resample.hpp"  // Defines Design::Resample, which includes design/fir_filter.hpp

// interface/backend.hpp defines ::OmniDSP::Abstract::Backend and
// ::OmniDSP::Abstract::ResamplePlanImpl
#include "interface/backend.hpp"

namespace OmniDSP {

  template <typename T>
  class OMNIDSP_EXPORT ResamplePlan {  // Or ResampleProcessor
    static_assert(
        !Utils::IsComplex_v<T>,
        "ResamplePlan/Processor requires a real type (F32 or F64).");

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

    [[nodiscard]] static OmniExpected<std::unique_ptr<ResamplePlan<T>>> create(
        const ::OmniDSP::Abstract::Backend& backend,
        const Design::Resample& design);

    static std::unique_ptr<ResamplePlan<T>> create_from_impl(
        std::unique_ptr<::OmniDSP::Abstract::ResamplePlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      // Call the private constructor
      return std::unique_ptr<ResamplePlan<T>>(
          new ResamplePlan<T>(std::move(pimpl)));
    }

   private:
    // Define the private constructor inline in the header
    explicit ResamplePlan(
        std::unique_ptr<::OmniDSP::Abstract::ResamplePlanImpl<T>> pimpl)
        : pimpl_(std::move(pimpl))
    {
      if (!pimpl_) {
        // This check is good, though create_from_impl also checks.
        // If this constructor is only called by create_from_impl,
        // the check in create_from_impl might be sufficient.
        throw std::runtime_error(
            "ResamplePlan/Processor constructed with null implementation "
            "pointer.");
      }
    }
    std::unique_ptr<::OmniDSP::Abstract::ResamplePlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_RESAMPLE_HPP
