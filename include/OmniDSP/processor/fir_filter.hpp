/**
 * @file fir_filter.hpp // New conceptual path: processor/fir_filter.hpp
 * @brief Defines the public API for FIR (Finite Impulse Response) Filter
 * Processor objects under the OmniDSP::Processor namespace.
 */

#ifndef OMNIDSP_PROCESSOR_FIR_FILTER_HPP  // Updated include guard
#define OMNIDSP_PROCESSOR_FIR_FILTER_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>  // For std::optional if used in future API
#include <span>
#include <utility>  // For std::move
#include <vector>   // For std::vector if used in future API

#include "OmniDSP/coefs/fir_filter.hpp"  // For Coefs::FIRFilter (parameter to create method)
#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected, F32, C32, etc.
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT (used for explicit instantiations)
// For Abstract::Backend and Abstract::FIRFilterProcessorImpl:
#include "interface/backend.hpp"

namespace OmniDSP::Processor {
  /**
   * @brief Processor object for FIR filtering.
   * @tparam T_Data The data type (e.g., F32, F64, C32, C64).
   *
   * Formerly OmniDSP::FIRFilterProcessor.
   */
  template <
      typename T_Data>  // Template parameter T renamed to T_Data for clarity
  class FIRFilter {  // Renamed from FIRFilterProcessor. OMNIDSP_EXPORT removed
                     // from class template.
    // Basic type checking (can be expanded if necessary)
    static_assert(
        std::is_floating_point_v<typename Utils::GetRealType<T_Data>>
            || Utils::IsComplex_v<T_Data>,
        "Processor::FIRFilter requires a real or complex floating-point type.");

   public:
    ~FIRFilter();                           // Destructor
    FIRFilter(FIRFilter&& other) noexcept;  // Move constructor
    FIRFilter& operator=(
        FIRFilter&& other) noexcept;       // Move assignment operator
    FIRFilter(const FIRFilter&) = delete;  // Disable copy constructor
    FIRFilter& operator=(const FIRFilter&)
        = delete;  // Disable copy assignment operator

    /**
     * @brief Executes the FIR filter on the input data.
     * @param input A span representing the input signal.
     * @param output A span representing the output buffer. Must be large
     * enough.
     * @return OmniStatus::Success on success, or an error code.
     */
    [[nodiscard]] OmniStatus execute(
        std::span<const T_Data> input,
        std::span<T_Data> output);  // State is managed by pimpl_

    /**
     * @brief Resets the internal state of the filter (e.g., clears delay
     * lines).
     * @return OmniStatus::Success on success, or an error code.
     */
    OmniStatus reset();

    /**
     * @brief Gets the order of the filter.
     * @return The filter order (number of taps - 1).
     */
    size_t get_order() const;

    /**
     * @brief Gets the number of taps (coefficients) in the filter.
     * @return The number of filter taps.
     */
    size_t get_num_taps() const;

    /**
     * @brief Creates an FIR Filter Processor using the specified backend and
     * coefficients.
     * @param backend The abstract backend interface to use for creating the
     * implementation.
     * @param coefs The FIR filter coefficients.
     * @return An OmniExpected containing a unique_ptr to the
     * Processor::FIRFilter on success, or an error OmniStatus.
     */
    [[nodiscard]] static OmniExpected<
        std::unique_ptr<FIRFilter<T_Data>>>  // Updated return type
    create(
        const Abstract::Backend& backend,
        const Coefs::FIRFilter<T_Data>& coefs)  // Takes Coefs::FIRFilter
    {
      OmniExpected<std::unique_ptr<Abstract::FIRFilterProcessorImpl<T_Data>>>
          pimpl_expected;

      // Type dispatch to call the correct backend implementation method
      if constexpr (std::is_same_v<T_Data, F32>) {
        pimpl_expected = backend.create_fir_filter_processor_impl_f32(coefs);
      }
      else if constexpr (std::is_same_v<T_Data, F64>) {
        pimpl_expected = backend.create_fir_filter_processor_impl_f64(coefs);
      }
      else if constexpr (std::is_same_v<T_Data, C32>) {
        pimpl_expected = backend.create_fir_filter_processor_impl_c32(coefs);
      }
      else if constexpr (std::is_same_v<T_Data, C64>) {
        pimpl_expected = backend.create_fir_filter_processor_impl_c64(coefs);
      }
      else {
        // This case should ideally be caught by the class-level static_assert,
        // but as a fallback:
        return std::unexpected(OmniStatus::UnsupportedType);
      }

      if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
      }

      // Call the static create_from_impl method of this class
      return create_from_impl(std::move(pimpl_expected.value()));
    }

    /**
     * @brief (Internal) Creates a public FIRFilter Processor from a private
     * implementation.
     * @param pimpl A unique_ptr to the backend-specific abstract
     * FIRFilterProcessorImpl.
     * @return A unique_ptr to the Processor::FIRFilter. Returns nullptr if
     * pimpl is null.
     */
    static std::unique_ptr<FIRFilter<T_Data>>
    create_from_impl(  // Updated return type
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<T_Data>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      // Use private constructor for controlled creation
      return std::unique_ptr<FIRFilter<T_Data>>(
          new FIRFilter<T_Data>(std::move(pimpl)));
    }

   private:
    // Private constructor, called by create_from_impl.
    explicit FIRFilter(
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<T_Data>> pimpl);
    std::unique_ptr<Abstract::FIRFilterProcessorImpl<T_Data>>
        pimpl_;  // Pointer to implementation
  };

}  // namespace OmniDSP::Processor

#endif  // OMNIDSP_PROCESSOR_FIR_FILTER_HPP
