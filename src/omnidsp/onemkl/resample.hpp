/**
 * @file resample.hpp (onemkl)
 * @brief Declares the oneMKL backend ResamplePlanImpl class using IPP.
 */

#ifndef OMNIDSP_ONEMKL_RESAMPLE_HPP
#define OMNIDSP_ONEMKL_RESAMPLE_HPP

#include <ipps.h>  // IPP Signal Processing header

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/resample.hpp>  // For ResampleSpec definition
#include <cstddef>               // For size_t
#include <memory>                // For std::unique_ptr
#include <span>
#include <vector>

#include "../interface/backend.hpp"  // Base ResamplePlanImpl interface and AbstractBackend

namespace OmniDSP::backend {

  /**
   * @brief oneMKL implementation for resampling plans using IPP FIR-based
   * resampling.
   * @tparam T Real data type (F32 or F64).
   */
  template <typename T>  // T is real type here (F32, F64)
  class OneMKLResamplePlanImpl final : public ResamplePlanImpl<T> {
    static_assert(
        !Utils::is_complex_v<T>,
        "OneMKLResamplePlanImpl requires a real type.");

   public:
    /**
     * @brief Constructor. Initializes IPP resampler state.
     * @param owner Pointer to the backend instance creating this plan (needed
     * for filter design). Must not be null.
     * @param spec The resampling specification.
     * @throws std::invalid_argument If spec is invalid or owner is null.
     * @throws std::runtime_error If IPP state allocation or initialization
     * fails, or filter design fails.
     * @throws std::bad_alloc If memory allocation fails.
     */
    // *** UPDATED: Added owner parameter ***
    explicit OneMKLResamplePlanImpl(
        const AbstractBackend* owner, const ResampleSpec& spec);

    /**
     * @brief Destructor. Frees the IPP resampler state.
     */
    ~OneMKLResamplePlanImpl() override;

    // --- Deleted Copy/Move ---
    OneMKLResamplePlanImpl(const OneMKLResamplePlanImpl&) = delete;
    OneMKLResamplePlanImpl& operator=(const OneMKLResamplePlanImpl&) = delete;
    OneMKLResamplePlanImpl(OneMKLResamplePlanImpl&&) = delete;
    OneMKLResamplePlanImpl& operator=(OneMKLResamplePlanImpl&&) = delete;

    // --- Interface Methods ---
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] Status reset() override;
    double get_input_rate() const override;
    double get_output_rate() const override;
    size_t get_output_length(size_t input_length) const override;

   private:
    // --- Configuration ---
    const AbstractBackend*
        owner_backend_;  // *** ADDED: Non-owning pointer to owner ***
    double input_rate_;
    double output_rate_;
    int quality_;  // Store quality factor used

    // --- IPP State ---
    // Using void* and casting, as IPP state types are opaque
    Ipp8u* p_spec_
        = nullptr;  // Pointer to the IPP FIR MR Spec structure buffer
    Ipp8u* p_buffer_ = nullptr;  // Pointer to the IPP working buffer
    // Store sizes needed for freeing memory
    int spec_size_ = 0;
    int buffer_size_ = 0;

    // IPP delay line state (part of pSpec from IPPS)
    // We might need to manage this explicitly if reset needs more than
    // ippsFIRMRInit
  };

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_ONEMKL_RESAMPLE_HPP
