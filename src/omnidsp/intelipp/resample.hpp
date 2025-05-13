/**
 * @file resample.hpp (IntelIPP)
 * @brief Declares the Intel IPP backend ResampleProcessorImpl class using IPP
 * FIRMR.
 */

#ifndef OMNIDSP_INTELIPP_RESAMPLE_HPP
#define OMNIDSP_INTELIPP_RESAMPLE_HPP

#include <ipps.h>  // IPP Signal Processing header

#include <OmniDSP/core_types.hpp>  // For Status, F32, F64, Utils::IsComplex_v
#include <OmniDSP/resample.hpp>    // For Design::Resample definition
#include <OmniDSP/window.hpp>      // For WindowSetup
#include <cstddef>                 // For size_t
#include <memory>                  // For std::unique_ptr
#include <span>
#include <vector>

// Forward declaration for Abstract::Backend if used as pointer/reference
namespace OmniDSP::Abstract {
  class Backend;
}

namespace OmniDSP::IntelIPP {

  /**
   * @brief Intel IPP implementation for resampling plans using IPP FIR-based
   * multi-rate resampling (ippsFIRMR).
   * @tparam T Real data type (F32 or F64).
   */
  template <typename T>
  class ResampleProcessorImpl final
      : public Abstract::ResampleProcessorImpl<T> {
    static_assert(
        !::OmniDSP::Utils::IsComplex_v<T>,
        "IntelIPP::ResampleProcessorImpl requires a real type.");

   public:
    explicit ResampleProcessorImpl(
        const Abstract::Backend* owner, const Design::Resample& spec);

    ~ResampleProcessorImpl() override;

    // --- Deleted Copy/Move ---
    ResampleProcessorImpl(const ResampleProcessorImpl&) = delete;
    ResampleProcessorImpl& operator=(const ResampleProcessorImpl&) = delete;
    ResampleProcessorImpl(ResampleProcessorImpl&&) = delete;
    ResampleProcessorImpl& operator=(ResampleProcessorImpl&&) = delete;

    // --- Interface Methods ---
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] Status reset() override;
    double get_input_rate() const override;
    double get_output_rate() const override;
    size_t get_output_length(size_t input_length) const override;

   private:
    // --- Configuration ---
    // Store the full Design::Resample to access L/M factors and other resolved
    // parameters.
    Design::Resample spec_;
    // Individual members for frequently accessed or IPP-specific parameters can
    // still be kept if it improves clarity or avoids frequent spec_.member
    // access, but spec_ holds the truth. For now, input_rate_, output_rate_,
    // quality_ are initialized from spec in the .cpp and can be accessed
    // directly or via spec_ as needed.
    double input_rate_;
    double output_rate_;
    int quality_;

    // --- IPP State ---
    Ipp8u* p_spec_mem_
        = nullptr;               // Raw memory for the IPP FIR Design structure
    Ipp8u* p_buffer_ = nullptr;  // Pointer to the IPP working buffer
    int spec_mem_size_ = 0;      // Size of the p_spec_mem_ buffer
    int buffer_size_ = 0;        // Size of the p_buffer_

    // Typed pointer to the initialized IPP FIR Design structure
    void* p_ipp_fir_spec_ = nullptr;
  };

}  // namespace OmniDSP::IntelIPP

#endif  // OMNIDSP_INTELIPP_RESAMPLE_HPP
