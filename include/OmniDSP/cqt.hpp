/**
 * @file cqt.hpp
 * @brief Defines the interface for Constant-Q Transform (CQT) Processor objects
 * and related design specifications (Design::CQT, Design::CQTOctave).
 */

#ifndef OMNIDSP_CQT_HPP
#define OMNIDSP_CQT_HPP

#include <complex>
#include <cstddef>
#include <expected>  // Required for std::unexpected
#include <memory>
#include <span>
#include <string>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/core_types.hpp"  // Defines OmniExpected, Status
#include "OmniDSP/design/cqt.hpp"
#include "OmniDSP/window.hpp"  // For WindowSetup
// This include defines ::OmniDSP::Abstract::Backend and
// ::OmniDSP::Abstract::CQTProcessorImpl
#include "interface/backend.hpp"  // Defines Abstract::Backend and Abstract::CQTPlanImpl

namespace OmniDSP {

  // Forward declaration of OmniDSP class if needed for friendship,
  // though static create methods usually don't need friendship with OmniDSP
  // itself class OmniDSP;

  template <typename T>  // T is REAL type (F32 or F64)
  class OMNIDSP_EXPORT
      CQTProcessor {  // TODO: Rename to CQTProcessor if stateful
    static_assert(
        !Utils::IsComplex_v<T>,
        "CQTProcessor/Processor requires a real type (F32 or F64).");
    using Complex = Utils::GetComplexType<T>;

   public:
    ~CQTProcessor();
    CQTProcessor(CQTProcessor&& other) noexcept;
    CQTProcessor& operator=(CQTProcessor&& other) noexcept;
    CQTProcessor(const CQTProcessor&) = delete;
    CQTProcessor& operator=(const CQTProcessor&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<Complex> output) const;
    size_t get_num_bins() const;
    size_t get_num_output_frames(size_t input_length) const;
    size_t get_hop_length() const;

    /**
     * @brief Creates a CQTProcessor (or CQTProcessor) using the specified
     * backend and design.
     * @param backend The abstract backend interface to use for creating the
     * implementation.
     * @param design The CQT design specification.
     * @return An OmniExpected containing a unique_ptr to the CQTProcessor on
     * success, or an error.
     */
    [[nodiscard]] static OmniExpected<std::unique_ptr<CQTProcessor<T>>> create(
        const ::OmniDSP::Abstract::Backend& backend, const Design::CQT& design)
    {
      // Initialize pimpl_expected with an unexpected value.
      // The type of pimpl_expected is OmniExpected<..., Status>, so ErrorType
      // is Status.
      OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<T>>>
          pimpl_expected
          = std::unexpected<OmniDSP::Status>(OmniDSP::Status::NotImplemented);

      if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected = backend.create_cqt_processor_impl_f32(design);
      }
      else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected = backend.create_cqt_processor_impl_f64(design);
      }
      else {
        // This static_assert will fire at compile time if T is not F32 or F64,
        // which is already ensured by the class-level static_assert.
        static_assert(
            std::is_same_v<T, F32> || std::is_same_v<T, F64>,
            "Unsupported type for CQTProcessor::create");
        // Return an unexpected value with the correct error type.
        return std::unexpected<OmniDSP::Status>(
            OmniDSP::Status::UnsupportedType);
      }

      if (!pimpl_expected) {
        // pimpl_expected.error() returns Status&, which is compatible.
        return std::unexpected<OmniDSP::Status>(pimpl_expected.error());
      }
      // pimpl_expected.value() returns unique_ptr&, std::move is correct.
      return create_from_impl(std::move(pimpl_expected.value()));
    }

    /**
     * @brief Creates a CQTProcessor from a backend-specific implementation
     * pointer.
     * @param pimpl A unique_ptr to the backend-specific CQTProcessorImpl.
     * @return A unique_ptr to the CQTProcessor. Returns nullptr if pimpl is
     * null.
     */
    static std::unique_ptr<CQTProcessor<T>> create_from_impl(
        std::unique_ptr<::OmniDSP::Abstract::CQTProcessorImpl<T>> pimpl)
    {
      if (!pimpl) {
        // Optionally, log an error here or throw, though returning nullptr
        // is also a valid strategy if the caller checks.
        return nullptr;
      }
      // Use new directly as CQTProcessor constructor is private.
      // The unique_ptr will manage the memory.
      return std::unique_ptr<CQTProcessor<T>>(
          new CQTProcessor<T>(std::move(pimpl)));
    }

   private:
    // Private constructor, called by create_from_impl.
    explicit CQTProcessor(
        std::unique_ptr<::OmniDSP::Abstract::CQTProcessorImpl<T>> pimpl);
    std::unique_ptr<::OmniDSP::Abstract::CQTProcessorImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_CQT_HPP
