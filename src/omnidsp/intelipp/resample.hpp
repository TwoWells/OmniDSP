/**
 * @file resample.hpp (intelipp)
 * @brief Declares the Intel IPP backend ResamplePlanImpl class using IPP FIRMR.
 */

#ifndef OMNIDSP_INTELIPP_RESAMPLE_HPP
#define OMNIDSP_INTELIPP_RESAMPLE_HPP

#include <ipps.h>  // IPP Signal Processing header

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/resample.hpp>  // For ResampleSpec definition & Base PlanImpl interface
#include <cstddef>               // For size_t
#include <memory>                // For std::unique_ptr
#include <span>
#include <vector>

#include "backend.hpp"  // <-- ADDED: Include the header that defines IntelIPPBackend

// Forward declare the base class implementation if needed (usually defined in
// interface/backend.hpp or similar)
namespace OmniDSP::Abstract {
  template <typename T>
  class ResamplePlanImpl;
}

namespace OmniDSP::IntelIPP {

  // Forward declare the backend class used in the constructor
  class Backend;

  /**
   * @brief Intel IPP implementation for resampling plans using IPP FIR-based
   * multi-rate resampling (ippsFIRMR).
   * @tparam T Real data type (F32 or F64).
   */
  template <typename T>
  class IntelIPPResamplePlanImpl final
      : public Abstract::ResamplePlanImpl<T> {  // Assuming base is in abstract
                                                // namespace
    static_assert(
        !Utils::IsComplex_v<T>,
        "IntelIPPResamplePlanImpl requires a real type.");

   public:
    /**
     * @brief Constructor. Initializes IPP resampler state.
     * @param owner Pointer to the Backend instance creating this plan.
     * Used to access filter design capabilities inherited from DefaultBackend.
     * Must not be null.
     * @param spec The resampling specification.
     * @throws std::invalid_argument If spec is invalid or owner is null.
     * @throws OmniDSPException If IPP state allocation or initialization fails,
     * or if internal filter design fails.
     */
    explicit IntelIPPResamplePlanImpl(
        const Abstract::Backend* owner, const ResampleSpec& spec);

    /**
     * @brief Destructor. Frees the IPP resampler state.
     */
    ~IntelIPPResamplePlanImpl() override;

    // --- Deleted Copy/Move ---
    IntelIPPResamplePlanImpl(const IntelIPPResamplePlanImpl&) = delete;
    IntelIPPResamplePlanImpl& operator=(const IntelIPPResamplePlanImpl&)
        = delete;
    IntelIPPResamplePlanImpl(IntelIPPResamplePlanImpl&&) = delete;
    IntelIPPResamplePlanImpl& operator=(IntelIPPResamplePlanImpl&&) = delete;

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
    int quality_;  // Store quality factor used

    // --- IPP State ---
    Ipp8u* p_spec_
        = nullptr;  // Pointer to the IPP FIR MR Spec structure buffer
    Ipp8u* p_buffer_ = nullptr;  // Pointer to the IPP working buffer
    int spec_size_ = 0;          // Store size for freeing memory
    int buffer_size_ = 0;        // Store size for freeing memory
    // Store the typed spec pointer for use in execute/reset
    // (Type depends on T, handled internally)
    void* p_ipp_spec_typed_ = nullptr;
  };

}  // namespace OmniDSP::IntelIPP

#endif  // OMNIDSP_INTELIPP_RESAMPLE_HPP
