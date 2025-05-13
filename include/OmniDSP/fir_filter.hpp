/**
 * @file fir_filter.hpp
 * @brief Defines the public API for FIR (Finite Impulse Response) Filter
 * Plan/Processor objects.
 */

#ifndef OMNIDSP_FIR_FILTER_HPP
#define OMNIDSP_FIR_FILTER_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/coefs/fir_filter.hpp"  // Defines FIRCoefs
#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected, F32, C32 etc.
#include "OmniDSP/design/fir_filter.hpp"  // Defines Design::FIRFilter (used by params)
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/types/filter.hpp"  // For FilterType (if needed by FIR specific params or design)
// interface/backend.hpp is included by individual .cpp files that need the
// Abstract::FIRFilterProcessorImpl or by OmniDSP.hpp if FIRFilterProcessor is
// returned by OmniDSP class methods. For a clean header, only include what this
// specific header *defines* or *uses directly in its declarations*. If
// Abstract::FIRFilterProcessorImpl is only used as a template argument to
// create_from_impl, forward declaration might be possible, or the include can
// be kept if it's small/foundational.
#include "interface/backend.hpp"  // For Abstract::FIRFilterPlanImpl

namespace OmniDSP {

  // Forward declare OmniDSP if it's a friend or used in static create methods
  // class OmniDSP;

  /**
   * @brief Plan/Processor object for FIR filtering.
   * @tparam T The data type (e.g., F32, F64, C32, C64).
   * * TODO: Rename to FIRFilterProcessor once stateful aspects are fully
   * implemented.
   */
  template <typename T>
  class OMNIDSP_EXPORT
      FIRFilterProcessor {  // TODO: Rename to FIRFilterProcessor
   public:
    ~FIRFilterProcessor();
    FIRFilterProcessor(FIRFilterProcessor&& other) noexcept;
    FIRFilterProcessor& operator=(FIRFilterProcessor&& other) noexcept;
    FIRFilterProcessor(const FIRFilterProcessor&) = delete;
    FIRFilterProcessor& operator=(const FIRFilterProcessor&) = delete;

    /**
     * @brief Executes the FIR filter on the input data.
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
     * @brief Gets the number of taps (coefficients) in the filter.
     * @return The number of filter taps.
     */
    size_t get_num_taps() const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @details This is the intended way for backends to construct the public
     * Plan/Processor handle.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public FIRFilterProcessor.
     * Returns nullptr if pimpl is null.
     */
    static std::unique_ptr<FIRFilterProcessor<T>> create_from_impl(
        std::unique_ptr<::OmniDSP::Abstract::FIRFilterProcessorImpl<T>>
            pimpl)  // TODO: Use Abstract::FIRFilterProcessorImpl
    {
      if (!pimpl) {
        return nullptr;
      }
      // Directly call the private constructor
      return std::unique_ptr<FIRFilterProcessor<T>>(
          new FIRFilterProcessor<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called by create_from_impl or OmniDSP class.
     */
    explicit FIRFilterProcessor(
        std::unique_ptr<::OmniDSP::Abstract::FIRFilterProcessorImpl<T>>
            pimpl);  // TODO: Use Abstract::FIRFilterProcessorImpl

    std::unique_ptr<::OmniDSP::Abstract::FIRFilterProcessorImpl<T>>
        pimpl_;  // TODO: Use Abstract::FIRFilterProcessorImpl
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_FIR_FILTER_HPP
