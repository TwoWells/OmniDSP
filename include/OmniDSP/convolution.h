/**
 * @file convolution.h
 * @brief Defines the public API for Convolution and Correlation Plan objects.
 */

#ifndef OMNIDSP_CONVOLUTION_H
#define OMNIDSP_CONVOLUTION_H

#include <complex>
#include <cstddef>
#include <memory>       // For std::unique_ptr
#include <span>         // For input/output views (requires C++20)
#include <string_view>  // For get_convolution_type_name
#include <vector>

#include "core_types.h"  // Core types like RealT, ComplexT, Status

// Include the generated export header for DLL support
#include "OmniDSP/omnidsp_export.h"

namespace OmniDSP {

  // Forward declare the main OmniDSP class (needed for friend declaration)
  class OmniDSP;

  // Forward declarations for implementation classes (Pimpl idiom)
  namespace backend {
    template <typename T> class ConvolutionPlanImpl;
    template <typename T> class CorrelationPlanImpl;
  }  // namespace backend

  /**
   * @brief Specifies the type (output size/boundary handling) for convolution
   * and correlation operations. <<< MOVED HERE
   * @details Determines the length and alignment of the output signal relative
   * to the input signals.
   */
  enum class ConvolutionType {
    Full,  ///< Output contains all computed values where the signals overlap at
           ///< all, size = N + M - 1.
    Same,  ///< Output size is the same as the first input N. The 'center' part
           ///< of the full result is returned, aligned with the first input.
    Valid  ///< Output includes only points where the signals fully overlap
           ///< without any zero-padding assumptions, size = max(0, N - M + 1).
  };

  /**
   * @brief Gets the string name corresponding to a ConvolutionType enum value.
   * <<< MOVED HERE
   * @param type The convolution type enum value.
   * @return A string_view representing the convolution type name.
   */
  inline std::string_view get_convolution_type_name(
      ConvolutionType type) noexcept
  {
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

  /**
   * @brief Plan object for executing Convolution operations.
   * @details Encapsulates the kernel and backend-specific context for efficient
   * execution using methods like direct summation or FFT-based convolution.
   * Created via OmniDSP::create_convolution_plan.
   * Uses the Pimpl idiom. Non-copyable but movable.
   * @tparam T The data type (e.g., float, double, std::complex<float>).
   */
  template <typename T> class OMNIDSP_EXPORT ConvolutionPlan {
    friend class OmniDSP;  // Allow OmniDSP factory methods to call private
                           // constructor

   public:
    /** @brief Destructor. */
    ~ConvolutionPlan();
    /** @brief Move constructor. */
    ConvolutionPlan(ConvolutionPlan&& other) noexcept;
    /** @brief Move assignment operator. */
    ConvolutionPlan& operator=(ConvolutionPlan&& other) noexcept;
    /** @brief Deleted copy constructor. */
    ConvolutionPlan(const ConvolutionPlan&) = delete;
    /** @brief Deleted copy assignment operator. */
    ConvolutionPlan& operator=(const ConvolutionPlan&) = delete;

    /**
     * @brief Executes the convolution operation.
     * @param input A span representing the input signal.
     * @param output A span representing the output buffer. Must be
     * pre-allocated with the size returned by get_output_length(input.size()).
     * @return Status::Success on success, or an error code on failure.
     */
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;

    /**
     * @brief Gets the length of the kernel used by this plan.
     * @return The kernel length.
     */
    size_t get_kernel_length() const;

    /**
     * @brief Gets the convolution type (boundary handling mode) configured for
     * this plan. <<< UPDATED
     * @return The ConvolutionType (Full, Same, Valid).
     */
    ConvolutionType get_type() const;  // <<< Renamed from get_mode

    /**
     * @brief Calculates the expected output length for a given input length
     * based on the kernel length and convolution type.
     * @param input_length Length of the input signal.
     * @return Expected output length.
     */
    size_t get_output_length(size_t input_length) const;

   private:
    /**
     * @brief Private constructor, called ONLY by OmniDSP factory methods.
     * @param pimpl A unique_ptr to the backend-specific implementation.
     */
    explicit ConvolutionPlan(
        std::unique_ptr<backend::ConvolutionPlanImpl<T>> pimpl);

    /** @brief Pointer to the implementation object (Pimpl idiom). */
    std::unique_ptr<backend::ConvolutionPlanImpl<T>> pimpl_;
  };

  /**
   * @brief Plan object for executing Correlation operations.
   * @details Encapsulates the template (kernel) and backend-specific context
   * for efficient execution. Created via OmniDSP::create_correlation_plan. Uses
   * the Pimpl idiom. Non-copyable but movable.
   * @tparam T The data type (e.g., float, double, std::complex<float>).
   */
  template <typename T> class OMNIDSP_EXPORT CorrelationPlan {
    friend class OmniDSP;  // Allow OmniDSP factory methods to call private
                           // constructor

   public:
    /** @brief Destructor. */
    ~CorrelationPlan();
    /** @brief Move constructor. */
    CorrelationPlan(CorrelationPlan&& other) noexcept;
    /** @brief Move assignment operator. */
    CorrelationPlan& operator=(CorrelationPlan&& other) noexcept;
    /** @brief Deleted copy constructor. */
    CorrelationPlan(const CorrelationPlan&) = delete;
    /** @brief Deleted copy assignment operator. */
    CorrelationPlan& operator=(const CorrelationPlan&) = delete;

    /**
     * @brief Executes the correlation operation.
     * @param input A span representing the input signal.
     * @param output A span representing the output buffer. Must be
     * pre-allocated with the size returned by get_output_length(input.size()).
     * @return Status::Success on success, or an error code on failure.
     */
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;

    /**
     * @brief Gets the length of the template (kernel) used by this plan.
     * @return The template length.
     */
    size_t get_template_length() const;

    /**
     * @brief Gets the correlation type (boundary handling mode) configured for
     * this plan. <<< UPDATED
     * @return The ConvolutionType (Full, Same, Valid).
     */
    ConvolutionType get_type() const;  // <<< Renamed from get_mode

    /**
     * @brief Calculates the expected output length for a given input length
     * based on the template length and correlation type.
     * @param input_length Length of the input signal.
     * @return Expected output length.
     */
    size_t get_output_length(size_t input_length) const;

   private:
    /**
     * @brief Private constructor, called ONLY by OmniDSP factory methods.
     * @param pimpl A unique_ptr to the backend-specific implementation.
     */
    explicit CorrelationPlan(
        std::unique_ptr<backend::CorrelationPlanImpl<T>> pimpl);

    /** @brief Pointer to the implementation object (Pimpl idiom). */
    std::unique_ptr<backend::CorrelationPlanImpl<T>> pimpl_;
  };

  // --- Template Implementations (Definitions) ---
  // Definitions for template class methods (constructors, destructors, move
  // ops, execute, getters) MUST be provided in the corresponding .cpp file
  // (convolution.cpp).

}  // namespace OmniDSP

#endif  // OMNIDSP_CONVOLUTION_H
