/**
 * @file cqt.hpp // New conceptual path: processor/cqt.hpp
 * @brief Defines the interface for Constant-Q Transform (CQT) Processor objects
 * under the OmniDSP::Processor namespace.
 */

#ifndef OMNIDSP_PROCESSOR_CQT_HPP  // Updated include guard
#define OMNIDSP_PROCESSOR_CQT_HPP

#include <complex>
#include <cstddef>
#include <expected>  // Required for std::unexpected
#include <memory>
#include <span>
#include <string>   // Potentially for error messages or future use
#include <utility>  // For std::move
#include <vector>   // Potentially for internal use or future API

#include "OmniDSP/core_types.hpp"  // Defines OmniExpected, Status, F32, F64, Utils::IsComplex_v, Utils::GetComplexType
#include "OmniDSP/design/cqt.hpp"  // For Design::CQT (parameter to create method)
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT (used for explicit instantiations, not class template)
// For Abstract::Backend and Abstract::CQTProcessorImpl:
#include "interface/backend.hpp"

namespace OmniDSP::Processor {

  /**
   * @brief Processor object for Constant-Q Transform (CQT) operations.
   * @tparam T_Real The underlying REAL floating-point type (e.g., F32, F64).
   *
   * Formerly OmniDSP::CQTProcessor.
   */
  template <
      typename T_Real>  // Template parameter T renamed to T_Real for clarity
  class CQT {  // Renamed from CQTProcessor. OMNIDSP_EXPORT removed from class
               // template.
    static_assert(
        !Utils::IsComplex_v<T_Real>,
        "Processor::CQT requires a real type (F32 or F64).");
    using Complex = Utils::GetComplexType<T_Real>;

   public:
    ~CQT();                                // Destructor
    CQT(CQT&& other) noexcept;             // Move constructor
    CQT& operator=(CQT&& other) noexcept;  // Move assignment operator
    CQT(const CQT&) = delete;              // Disable copy constructor
    CQT& operator=(const CQT&) = delete;   // Disable copy assignment operator

    /**
     * @brief Executes the CQT.
     * @param input The input real data span.
     * @param output The output complex data span (CQT coefficients).
     * @return OmniStatus indicating success or failure.
     */
    [[nodiscard]] OmniStatus execute(
        std::span<const T_Real> input, std::span<Complex> output) const;

    /** @return The total number of CQT bins. */
    size_t get_num_bins() const;

    /**
     * @brief Calculates the number of output frames (time slices) for a given
     * input length.
     * @param input_length The length of the input signal.
     * @return The number of output CQT frames.
     */
    size_t get_num_output_frames(size_t input_length) const;

    /** @return The hop length (in samples) between CQT frames. */
    size_t get_hop_length() const;

    /**
     * @brief Creates a CQT Processor using the specified backend and design.
     * @param backend The abstract backend interface to use for creating the
     * implementation.
     * @param design The CQT design specification.
     * @return An OmniExpected containing a unique_ptr to the Processor::CQT on
     * success, or an error OmniStatus.
     */
    [[nodiscard]] static OmniExpected<
        std::unique_ptr<CQT<T_Real>>>  // Updated return type
    create(
        const Abstract::Backend& backend,  // Correctly using Abstract::Backend
        const Design::CQT& design)
    {
      OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<T_Real>>>
          pimpl_expected;

      if constexpr (std::is_same_v<T_Real, F32>) {
        pimpl_expected = backend.create_cqt_processor_impl_f32(design);
      }
      else if constexpr (std::is_same_v<T_Real, F64>) {
        pimpl_expected = backend.create_cqt_processor_impl_f64(design);
      }
      else {
        // This static_assert is somewhat redundant due to the class-level one,
        // but doesn't hurt for clarity within the create method.
        static_assert(
            std::is_same_v<T_Real, F32> || std::is_same_v<T_Real, F64>,
            "Unsupported type for Processor::CQT::create");
        return std::unexpected(OmniStatus::UnsupportedType);
      }

      if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
      }

      // Call the static create_from_impl method of this class
      return create_from_impl(std::move(pimpl_expected.value()));
    }

    /**
     * @brief (Internal) Creates a public CQT Processor from a private
     * implementation.
     * @param pimpl A unique_ptr to the backend-specific abstract
     * CQTProcessorImpl.
     * @return A unique_ptr to the Processor::CQT. Returns nullptr if pimpl is
     * null.
     */
    static std::unique_ptr<CQT<T_Real>>
    create_from_impl(  // Updated return type
        std::unique_ptr<Abstract::CQTProcessorImpl<T_Real>>
            pimpl)  // Correctly using Abstract::CQTProcessorImpl
    {
      if (!pimpl) {
        return nullptr;
      }
      // Use private constructor for controlled creation
      return std::unique_ptr<CQT<T_Real>>(new CQT<T_Real>(std::move(pimpl)));
    }

   private:
    // Private constructor, called by create_from_impl.
    explicit CQT(std::unique_ptr<Abstract::CQTProcessorImpl<T_Real>> pimpl);
    std::unique_ptr<Abstract::CQTProcessorImpl<T_Real>>
        pimpl_;  // Pointer to implementation
  };

}  // namespace OmniDSP::Processor

#endif  // OMNIDSP_PROCESSOR_CQT_HPP
