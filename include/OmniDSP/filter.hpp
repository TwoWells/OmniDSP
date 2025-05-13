/**
 * @file filter.hpp
 * @brief Defines filter design specifications (Design::FIRFilter,
 * Design::IIRFilter) and related types.
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

#include "OmniDSP/coefs/fir_filter.hpp"
#include "OmniDSP/coefs/iir_filter.hpp"
#include "OmniDSP/core_types.hpp"
#include "OmniDSP/design/fir_filter.hpp"
#include "OmniDSP/design/iir_filter.hpp"
#include "OmniDSP/omnidsp_export.hpp"
#include "OmniDSP/types/filter.hpp"
#include "OmniDSP/window.hpp"
// This include defines ::OmniDSP::Abstract::FIRFilterPlanImpl and
// ::OmniDSP::Abstract::IIRFilterPlanImpl
#include "interface/backend.hpp"

namespace OmniDSP {

  template <typename T>
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

    // Use fully qualified name for Abstract::FIRFilterPlanImpl
    static std::unique_ptr<FIRFilterPlan<T>> create_from_impl(
        std::unique_ptr<::OmniDSP::Abstract::FIRFilterPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<FIRFilterPlan<T>>(
          new FIRFilterPlan<T>(std::move(pimpl)));
    }

   private:
    // Use fully qualified name for Abstract::FIRFilterPlanImpl
    explicit FIRFilterPlan(
        std::unique_ptr<::OmniDSP::Abstract::FIRFilterPlanImpl<T>> pimpl);
    std::unique_ptr<::OmniDSP::Abstract::FIRFilterPlanImpl<T>> pimpl_;
  };

  template <typename T>
  class OMNIDSP_EXPORT IIRFilterPlan {
    static_assert(
        !Utils::IsComplex_v<T>,
        "IIRFilterPlan/Processor typically requires a real type (F32 or F64).");

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

    // Use fully qualified name for Abstract::IIRFilterPlanImpl
    static std::unique_ptr<IIRFilterPlan<T>> create_from_impl(
        std::unique_ptr<::OmniDSP::Abstract::IIRFilterPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<IIRFilterPlan<T>>(
          new IIRFilterPlan<T>(std::move(pimpl)));
    }

   private:
    // Use fully qualified name for Abstract::IIRFilterPlanImpl
    explicit IIRFilterPlan(
        std::unique_ptr<::OmniDSP::Abstract::IIRFilterPlanImpl<T>> pimpl);
    std::unique_ptr<::OmniDSP::Abstract::IIRFilterPlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_FILTER_HPP
