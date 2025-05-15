/**
 * @file convolution.hpp
 * @brief Defines the public API for Convolution and Correlation Plan objects
 * under the OmniDSP::Plan namespace.
 */

#ifndef OMNIDSP_PLAN_CONVOLUTION_HPP
#define OMNIDSP_PLAN_CONVOLUTION_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>  // For std::move
#include <vector>  // Required for std::vector in old signature, kept for safety

#include "OmniDSP/core_types.hpp"  // Includes Status, OmniExpected, F32, C32 etc.
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/types/convolution.hpp"  // For ConvolutionType, ConvolutionMethod (used by Params)

// Include Abstract::Backend for the static create method
#include "interface/backend.hpp"  // Defines Abstract::Backend

// Include Params types for Convolution and Correlation for the static create
// method
#include "OmniDSP/params/convolution.hpp"  // For Params::Convolution, Params::Correlation

// Forward declare Abstract Impl classes. These remain in the Abstract
// namespace.
namespace OmniDSP::Abstract {
  template <typename T>
  class ConvolutionPlanImpl;  // Implementation for Plan::Convolution
  template <typename T>
  class CorrelationPlanImpl;  // Implementation for Plan::Correlation
}  // namespace OmniDSP::Abstract

namespace OmniDSP::Plan {

  /**
   * @brief Plan object for executing Convolution operations.
   * @tparam T The data type (e.g., F32, F64, C32, C64).
   *
   * Formerly OmniDSP::ConvolutionPlan.
   */
  template <typename T>
  class Convolution {  // Renamed from ConvolutionPlan
   public:
    ~Convolution();                                        // Destructor
    Convolution(Convolution&& other) noexcept;             // Move constructor
    Convolution& operator=(Convolution&& other) noexcept;  // Move assignment
    Convolution(const Convolution&) = delete;  // Disable copy constructor
    Convolution& operator=(const Convolution&)
        = delete;  // Disable copy assignment

    /**
     * @brief Executes the convolution.
     * @param input The input data span.
     * @param output The output data span.
     * @return OmniStatus indicating success or failure.
     */
    [[nodiscard]] OmniStatus execute(
        std::span<const T> input, std::span<T> output) const;

    /** @return The length of the convolution kernel. */
    size_t get_kernel_length() const;

    /** @return The type of convolution (e.g., Valid, Same, Full). */
    ConvolutionType get_type() const;

    /** @return The method hint used for convolution (e.g., Auto, Direct, FFT).
     */
    ConvolutionMethod get_method_hint() const;

    /**
     * @brief Calculates the expected output length for a given input length.
     * @param input_length The length of the input signal.
     * @return The expected length of the output signal.
     */
    size_t get_output_length(size_t input_length) const;

    /** @return A span view of the kernel coefficients. */
    std::span<const T> get_kernel() const;

    /**
     * @brief Creates a Convolution plan using a specific backend and
     * parameters.
     * @param backend The backend to use for creating the plan.
     * @param params The parameters for the convolution operation.
     * @param kernel_coeffs The convolution kernel coefficients.
     * @return An OmniExpected containing a unique_ptr to the
     * OmniDSP::Plan::Convolution or an error OmniStatus.
     */
    [[nodiscard]] static OmniExpected<
        std::unique_ptr<Convolution<T>>>  // Updated return type
    create(
        const Abstract::Backend& backend,
        const Params::Convolution& params,
        std::span<const T> kernel_coeffs);

    /**
     * @brief (Internal) Creates a public Convolution plan from a private
     * implementation.
     * @param pimpl Pointer to the abstract implementation.
     * @return A unique_ptr to the OmniDSP::Plan::Convolution.
     */
    static std::unique_ptr<Convolution<T>>
    create_from_impl(  // Updated return type
        std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl);

   private:
    // Private constructor for use by create_from_impl
    explicit Convolution(
        std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::ConvolutionPlanImpl<T>>
        pimpl_;  // Pointer to implementation
  };

  /**
   * @brief Plan object for executing Correlation operations.
   * @tparam T The data type (e.g., F32, F64, C32, C64).
   *
   * Formerly OmniDSP::CorrelationPlan.
   */
  template <typename T>
  class Correlation {  // Renamed from CorrelationPlan
   public:
    ~Correlation();                                        // Destructor
    Correlation(Correlation&& other) noexcept;             // Move constructor
    Correlation& operator=(Correlation&& other) noexcept;  // Move assignment
    Correlation(const Correlation&) = delete;  // Disable copy constructor
    Correlation& operator=(const Correlation&)
        = delete;  // Disable copy assignment

    /**
     * @brief Executes the correlation.
     * @param input The input data span.
     * @param output The output data span.
     * @return OmniStatus indicating success or failure.
     */
    [[nodiscard]] OmniStatus execute(
        std::span<const T> input, std::span<T> output) const;

    /** @return The length of the correlation template. */
    size_t get_template_length() const;

    /** @return The type of correlation (e.g., Valid, Same, Full). */
    ConvolutionType get_type()
        const;  // Assuming Correlation uses ConvolutionType for mode

    /** @return The method hint used for correlation (e.g., Auto, Direct, FFT).
     */
    ConvolutionMethod get_method_hint()
        const;  // Assuming Correlation uses ConvolutionMethod

    /**
     * @brief Calculates the expected output length for a given input length.
     * @param input_length The length of the input signal.
     * @return The expected length of the output signal.
     */
    size_t get_output_length(size_t input_length) const;

    /** @return A span view of the template coefficients. */
    std::span<const T> get_template() const;

    /**
     * @brief Creates a Correlation plan using a specific backend and
     * parameters.
     * @param backend The backend to use for creating the plan.
     * @param params The parameters for the correlation operation.
     * @param template_coeffs The correlation template coefficients.
     * @return An OmniExpected containing a unique_ptr to the
     * OmniDSP::Plan::Correlation or an error OmniStatus.
     */
    [[nodiscard]] static OmniExpected<
        std::unique_ptr<Correlation<T>>>  // Updated return type
    create(
        const Abstract::Backend& backend,
        const Params::Correlation& params,
        std::span<const T> template_coeffs);

    /**
     * @brief (Internal) Creates a public Correlation plan from a private
     * implementation.
     * @param pimpl Pointer to the abstract implementation.
     * @return A unique_ptr to the OmniDSP::Plan::Correlation.
     */
    static std::unique_ptr<Correlation<T>>
    create_from_impl(  // Updated return type
        std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl);

   private:
    // Private constructor for use by create_from_impl
    explicit Correlation(
        std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::CorrelationPlanImpl<T>>
        pimpl_;  // Pointer to implementation
  };

}  // namespace OmniDSP::Plan

#endif  // OMNIDSP_PLAN_CONVOLUTION_HPP
