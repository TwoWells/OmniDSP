/**
 * @file convolution.h
 * @brief Defines the public API for Convolution and Correlation Plan objects.
 */

#ifndef OMNIDSP_CONVOLUTION_H
#define OMNIDSP_CONVOLUTION_H

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"
#include "core_types.hpp"  // Includes Status, OmniExpected, Detail::* etc.

// Forward declare backend Impl classes
namespace OmniDSP::backend {
  template <typename T>
  class ConvolutionPlanImpl;
  template <typename T>
  class CorrelationPlanImpl;
}  // namespace OmniDSP::backend

namespace OmniDSP {

  class OmniDSP;

  /** @brief Specifies the type (output size/boundary handling). */
  enum class ConvolutionType { Full, Same, Valid };

  /** @brief Specifies the underlying algorithm to use. */
  enum class ConvolutionMethod { Direct, FFT, Auto };

  /** @brief Gets the string name corresponding to a ConvolutionType. */
  inline std::string_view get_convolution_type_name(
      ConvolutionType type) noexcept
  { /* ... */
    switch (type) {
      case ConvolutionType::Full:
        return "Full";
      case ConvolutionType::Same:
        return "Same";
      case ConvolutionType::Valid:
        return "Valid";
      default:
        return "Unknown ConvolutionType";
    }
  }

  /** @brief Gets the string name corresponding to a ConvolutionMethod. */
  inline std::string_view get_convolution_method_name(
      ConvolutionMethod method) noexcept
  { /* ... */
    switch (method) {
      case ConvolutionMethod::Direct:
        return "Direct";
      case ConvolutionMethod::FFT:
        return "FFT";
      case ConvolutionMethod::Auto:
        return "Auto";
      default:
        return "Unknown ConvolutionMethod";
    }
  }

  /**
   * @brief Plan object for executing Convolution operations.
   * @tparam T The data type (e.g., float, double, std::complex<float>).
   */
  template <typename T>
  class OMNIDSP_EXPORT ConvolutionPlan {
    friend class OmniDSP;  // Keep friend for OmniDSP if needed
    // Removed friend declarations for AbstractBackend and DefaultBackend

   public:
    ~ConvolutionPlan();
    ConvolutionPlan(ConvolutionPlan&& other) noexcept;
    ConvolutionPlan& operator=(ConvolutionPlan&& other) noexcept;
    ConvolutionPlan(const ConvolutionPlan&) = delete;
    ConvolutionPlan& operator=(const ConvolutionPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;
    size_t get_kernel_length() const;
    ConvolutionType get_type() const;
    ConvolutionMethod get_method_hint() const;
    size_t get_output_length(size_t input_length) const;
    std::span<const T> get_kernel() const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public ConvolutionPlan. Returns
     * nullptr if pimpl is null.
     */
    static std::unique_ptr<ConvolutionPlan<T>> create_from_impl(
        std::unique_ptr<backend::ConvolutionPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<ConvolutionPlan<T>>(
          new ConvolutionPlan<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called by create_from_impl. */
    explicit ConvolutionPlan(
        std::unique_ptr<backend::ConvolutionPlanImpl<T>> pimpl);

    // Removed the private static helper and MakeUniqueEnabler struct

    std::unique_ptr<backend::ConvolutionPlanImpl<T>> pimpl_;
  };

  /**
   * @brief Plan object for executing Correlation operations.
   * @tparam T The data type (e.g., float, double, std::complex<float>).
   */
  template <typename T>
  class OMNIDSP_EXPORT CorrelationPlan {
    friend class OmniDSP;  // Keep friend for OmniDSP if needed
    // Removed friend declarations for AbstractBackend and DefaultBackend

   public:
    ~CorrelationPlan();
    CorrelationPlan(CorrelationPlan&& other) noexcept;
    CorrelationPlan& operator=(CorrelationPlan&& other) noexcept;
    CorrelationPlan(const CorrelationPlan&) = delete;
    CorrelationPlan& operator=(const CorrelationPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;
    size_t get_template_length() const;
    ConvolutionType get_type() const;
    ConvolutionMethod get_method_hint() const;
    size_t get_output_length(size_t input_length) const;
    std::span<const T> get_template() const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public CorrelationPlan. Returns
     * nullptr if pimpl is null.
     */
    static std::unique_ptr<CorrelationPlan<T>> create_from_impl(
        std::unique_ptr<backend::CorrelationPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<CorrelationPlan<T>>(
          new CorrelationPlan<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called by create_from_impl. */
    explicit CorrelationPlan(
        std::unique_ptr<backend::CorrelationPlanImpl<T>> pimpl);

    // Removed the private static helper and MakeUniqueEnabler struct

    std::unique_ptr<backend::CorrelationPlanImpl<T>> pimpl_;
  };

  // Definitions MUST be provided in the corresponding .cpp file

}  // namespace OmniDSP

#endif  // OMNIDSP_CONVOLUTION_H
