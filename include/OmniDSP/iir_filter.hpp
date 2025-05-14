/**
 * @file iir_filter.hpp
 * @brief Defines the public API for IIR (Infinite Impulse Response) Filter
 * Plan/Processor objects.
 */

#ifndef OMNIDSP_IIR_FILTER_HPP
#define OMNIDSP_IIR_FILTER_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/coefs/iir_filter.hpp"  // Defines Coefs::SOS
#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected, F32, C32, Utils::IsComplex_v etc.
#include "OmniDSP/design/iir_filter.hpp"  // Defines Design::IIRFilter (used by params)
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/types/filter.hpp"  // For FilterType, IIRFilterFormat (if needed by IIR specific params or design)
// interface/backend.hpp is included by individual .cpp files that need the
// Abstract::IIRFilterProcessorImpl or by OmniDSP.hpp if IIRFilterProcessor is
// returned by OmniDSP class methods.
#include "interface/backend.hpp"  // For Abstract::IIRFilterPlanImpl

namespace OmniDSP {

  // Forward declare OmniDSP if it's a friend or used in static create methods
  // class OmniDSP;

  /**
   * @brief Plan/Processor object for IIR filtering.
   * @tparam T The data type, typically real (e.g., F32, F64).
   * * TODO: Rename to IIRFilterProcessor once stateful aspects are fully
   * implemented.
   */
  template <typename T>
  class OMNIDSP_EXPORT
      IIRFilterProcessor {  // TODO: Rename to IIRFilterProcessor
    static_assert(
        !Utils::IsComplex_v<T>,
        "IIRFilterProcessor/Processor typically requires a real type (F32 or "
        "F64).");

   public:
    ~IIRFilterProcessor();
    IIRFilterProcessor(IIRFilterProcessor&& other) noexcept;
    IIRFilterProcessor& operator=(IIRFilterProcessor&& other) noexcept;
    IIRFilterProcessor(const IIRFilterProcessor&) = delete;
    IIRFilterProcessor& operator=(const IIRFilterProcessor&) = delete;

    /**
     * @brief Executes the IIR filter on the input data.
     * @param input A span representing the input signal.
     * @param output A span representing the output buffer. Must be large
     * enough.
     * @return Status::Success on success, or an error code.
     */
    [[nodiscard]] Status execute(
        std::span<const T> input,
        std::span<T> output);  // State is managed by pimpl_

    /**
     * @brief Resets the internal state of the filter.
     * @return Status::Success on success, or an error code.
     */
    Status reset();

    /**
     * @brief Gets the order of the filter.
     * @return The filter order.
     */
    size_t get_order() const;

    /**
     * @brief Gets the number of second-order sections (SOS) in the filter.
     * @return The number of SOS.
     */
    size_t get_num_sections() const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @details This is the intended way for backends to construct the public
     * Plan/Processor handle.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public IIRFilterProcessor.
     * Returns nullptr if pimpl is null.
     */
    static std::unique_ptr<IIRFilterProcessor<T>> create_from_impl(
        std::unique_ptr<::OmniDSP::Abstract::IIRFilterProcessorImpl<T>>
            pimpl)  // TODO: Use Abstract::IIRFilterProcessorImpl
    {
      if (!pimpl) {
        return nullptr;
      }
      // Directly call the private constructor
      return std::unique_ptr<IIRFilterProcessor<T>>(
          new IIRFilterProcessor<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called by create_from_impl or OmniDSP class.
     */
    explicit IIRFilterProcessor(
        std::unique_ptr<::OmniDSP::Abstract::IIRFilterProcessorImpl<T>>
            pimpl);  // TODO: Use Abstract::IIRFilterProcessorImpl

    std::unique_ptr<::OmniDSP::Abstract::IIRFilterProcessorImpl<T>>
        pimpl_;  // TODO: Use Abstract::IIRFilterProcessorImpl
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_IIR_FILTER_HPP
