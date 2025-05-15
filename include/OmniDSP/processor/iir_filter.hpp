/**
 * @file iir_filter.hpp // New conceptual path: processor/iir_filter.hpp
 * @brief Defines the public API for IIR (Infinite Impulse Response) Filter
 * Processor objects under the OmniDSP::Processor namespace.
 */

#ifndef OMNIDSP_PROCESSOR_IIR_FILTER_HPP  // Updated include guard
#define OMNIDSP_PROCESSOR_IIR_FILTER_HPP

#include <complex>  // Included for completeness, though IIR is often real-only
#include <cstddef>
#include <memory>
#include <optional>  // For std::optional if used in future API
#include <span>
#include <utility>  // For std::move
#include <vector>   // For std::vector if used in future API

#include "OmniDSP/coefs/iir_filter.hpp"  // For Coefs::SOS (parameter to create method)
#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected, F32, F64, Utils::IsComplex_v etc.
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT (used for explicit instantiations)
// For Abstract::Backend and Abstract::IIRFilterProcessorImpl:
#include "interface/backend.hpp"

namespace OmniDSP {

  // New 'Processor' namespace
  namespace Processor {

    /**
     * @brief Processor object for IIR filtering.
     * @tparam T_Data The data type, typically real (e.g., F32, F64).
     *
     * Formerly OmniDSP::IIRFilterProcessor.
     */
    template <
        typename T_Data>  // Template parameter T renamed to T_Data for clarity
    class IIRFilter {     // Renamed from IIRFilterProcessor. OMNIDSP_EXPORT
                          // removed from class template.
      static_assert(
          !Utils::IsComplex_v<T_Data>,
          "Processor::IIRFilter currently supports real types (F32 or F64).");

     public:
      ~IIRFilter();                           // Destructor
      IIRFilter(IIRFilter&& other) noexcept;  // Move constructor
      IIRFilter& operator=(
          IIRFilter&& other) noexcept;       // Move assignment operator
      IIRFilter(const IIRFilter&) = delete;  // Disable copy constructor
      IIRFilter& operator=(const IIRFilter&)
          = delete;  // Disable copy assignment operator

      /**
       * @brief Executes the IIR filter on the input data.
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
       * @brief Gets the effective order of the filter.
       * @details For SOS filters, this is typically 2 * number of sections.
       * @return The filter order.
       */
      size_t get_order() const;

      /**
       * @brief Gets the number of second-order sections (SOS) in the filter.
       * @return The number of SOS.
       */
      size_t get_num_sections() const;

      /**
       * @brief Creates an IIR Filter Processor using the specified backend and
       * SOS coefficients.
       * @param backend The abstract backend interface to use for creating the
       * implementation.
       * @param sos_coefs The IIR filter coefficients in Second-Order Sections
       * format.
       * @return An OmniExpected containing a unique_ptr to the
       * Processor::IIRFilter on success, or an error OmniStatus.
       */
      [[nodiscard]] static OmniExpected<
          std::unique_ptr<IIRFilter<T_Data>>>  // Updated return type
      create(
          const Abstract::Backend& backend,
          const Coefs::IIRFilterSOS& sos_coefs)
      {
        OmniExpected<std::unique_ptr<Abstract::IIRFilterProcessorImpl<T_Data>>>
            pimpl_expected;

        // Type dispatch to call the correct backend implementation method
        if constexpr (std::is_same_v<T_Data, F32>) {
          pimpl_expected
              = backend.create_iir_filter_processor_impl_f32(sos_coefs);
        }
        else if constexpr (std::is_same_v<T_Data, F64>) {
          pimpl_expected
              = backend.create_iir_filter_processor_impl_f64(sos_coefs);
        }
        // Add other types like C32, C64 if IIR for complex is supported by
        // backends For now, sticking to the static_assert for real types.
        else {
          // This case should ideally be caught by the class-level
          // static_assert, but as a fallback:
          return std::unexpected(OmniStatus::UnsupportedType);
        }

        if (!pimpl_expected) {
          return std::unexpected(pimpl_expected.error());
        }

        // Call the static create_from_impl method of this class
        return create_from_impl(std::move(pimpl_expected.value()));
      }

      /**
       * @brief (Internal) Creates a public IIRFilter Processor from a private
       * implementation.
       * @param pimpl A unique_ptr to the backend-specific abstract
       * IIRFilterProcessorImpl.
       * @return A unique_ptr to the Processor::IIRFilter. Returns nullptr if
       * pimpl is null.
       */
      static std::unique_ptr<IIRFilter<T_Data>>
      create_from_impl(  // Updated return type
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<T_Data>> pimpl)
      {
        if (!pimpl) {
          return nullptr;
        }
        // Use private constructor for controlled creation
        return std::unique_ptr<IIRFilter<T_Data>>(
            new IIRFilter<T_Data>(std::move(pimpl)));
      }

     private:
      // Private constructor, called by create_from_impl.
      explicit IIRFilter(
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<T_Data>> pimpl);
      std::unique_ptr<Abstract::IIRFilterProcessorImpl<T_Data>>
          pimpl_;  // Pointer to implementation
    };

  }  // namespace Processor
}  // namespace OmniDSP

#endif  // OMNIDSP_PROCESSOR_IIR_FILTER_HPP
