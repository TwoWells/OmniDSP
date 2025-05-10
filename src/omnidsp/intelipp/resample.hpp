/**
 * @file resample.hpp (IntelIPP)
 * @brief Declares the Intel IPP backend ResamplePlanImpl class using IPP FIRMR.
 */

#ifndef OMNIDSP_INTELIPP_RESAMPLE_HPP
#define OMNIDSP_INTELIPP_RESAMPLE_HPP

#include <ipps.h>  // IPP Signal Processing header

#include <OmniDSP/core_types.hpp>  // For Status, F32, F64, Utils::IsComplex_v
#include <OmniDSP/resample.hpp>  // For ResampleSpec definition (which now uses WindowSetup)
                                 // and Abstract::ResamplePlanImpl
#include <OmniDSP/window.hpp>  // For WindowSetup (though ResampleSpec includes it)
#include <cstddef>             // For size_t
#include <memory>              // For std::unique_ptr
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
  class ResamplePlanImpl final : public Abstract::ResamplePlanImpl<T> {
    static_assert(
        !::OmniDSP::Utils::IsComplex_v<T>,
        "IntelIPP::ResamplePlanImpl requires a real type.");

   public:
    explicit ResamplePlanImpl(
        const Abstract::Backend* owner, const ResampleSpec& spec);

    ~ResamplePlanImpl() override;

    // --- Deleted Copy/Move ---
    ResamplePlanImpl(const ResamplePlanImpl&) = delete;
    ResamplePlanImpl& operator=(const ResamplePlanImpl&) = delete;
    ResamplePlanImpl(ResamplePlanImpl&&) = delete;
    ResamplePlanImpl& operator=(ResamplePlanImpl&&) = delete;

    // --- Interface Methods ---
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] Status reset() override;
    double get_input_rate() const override;
    double get_output_rate() const override;
    size_t get_output_length(size_t input_length) const override;

   private:
    // --- Configuration ---
    double input_rate_;
    double output_rate_;
    int quality_;

    // --- IPP State ---
    Ipp8u* p_spec_mem_ = nullptr;  // Raw memory for the IPP FIR Spec structure
    Ipp8u* p_buffer_ = nullptr;    // Pointer to the IPP working buffer
    int spec_mem_size_ = 0;        // Size of the p_spec_mem_ buffer
    int buffer_size_ = 0;          // Size of the p_buffer_

    // Typed pointer to the initialized IPP FIR Spec structure
    // This will be IppsFIRSpec_32f* or IppsFIRSpec_64f*
    void* p_ipp_fir_spec_ = nullptr;
  };

}  // namespace OmniDSP::IntelIPP

#endif  // OMNIDSP_INTELIPP_RESAMPLE_HPP
