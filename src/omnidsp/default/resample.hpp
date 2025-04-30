/**
 * @file resample.hpp (default)
 * @brief Declares the concrete ResamplePlanImpl class for the Default backend.
 */

#ifndef OMNIDSP_DEFAULT_RESAMPLE_HPP
#define OMNIDSP_DEFAULT_RESAMPLE_HPP

#include <OmniDSP/core_types.hpp>  // Status, F32, F64, ResampleSpec
#include <OmniDSP/filter.hpp>  // May need FIR filter spec/design for polyphase implementation
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr
#include <span>     // For std::span
#include <vector>

#include "../interface/backend.hpp"  // Base ResamplePlanImpl interface

namespace OmniDSP::backend {

  // Forward declare FIR plan if used internally
  template <typename T>
  class FIRFilterPlanImpl;

  /**
   * @brief Default backend implementation for a Resampling Plan.
   * @details Uses a polyphase FIR filter approach for rational resampling.
   * Stateful.
   * @tparam T The REAL data type (F32 or F64).
   */
  template <typename T>  // T is real type (F32, F64)
  class DefaultResamplePlanImpl final : public ResamplePlanImpl<T> {
   public:
    /**
     * @brief Constructs a DefaultResamplePlanImpl.
     * @param owner_backend Pointer to the backend instance creating this plan
     * (needed for filter design).
     * @param spec The resampling specification (input rate, output rate,
     * quality).
     * @throws std::invalid_argument if spec contains invalid parameters (e.g.,
     * rates <= 0).
     * @throws std::runtime_error if filter design or internal setup fails.
     */
    DefaultResamplePlanImpl(
        const AbstractBackend* owner_backend, const ResampleSpec& spec);

    /**
     * @brief Destructor.
     */
    ~DefaultResamplePlanImpl() override;

    // --- Disable Copy/Move ---
    DefaultResamplePlanImpl(const DefaultResamplePlanImpl&) = delete;
    DefaultResamplePlanImpl& operator=(const DefaultResamplePlanImpl&) = delete;
    DefaultResamplePlanImpl(DefaultResamplePlanImpl&&) = delete;
    DefaultResamplePlanImpl& operator=(DefaultResamplePlanImpl&&) = delete;

    // --- Interface Methods Implementation ---

    /**
     * @brief Resamples a block of input samples.
     * @param input A span representing the input signal block.
     * @param output A span representing the output buffer. Its size must be at
     * least get_output_length(input.size()). The actual number of samples
     * written might vary slightly due to internal buffering and phase.
     * @return Status indicating success or failure. The number of samples
     * actually written to the output buffer might need to be returned or
     * handled. (Interface might need adjustment for this).
     */
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;  // Not const

    /**
     * @brief Resets the internal state (delay lines, phase accumulators) of the
     * resampler.
     * @return Status::Success.
     */
    [[nodiscard]] Status reset() override;  // Not const

    /**
     * @brief Gets the input sample rate configured for this plan.
     * @return The input sample rate in Hz.
     */
    double get_input_rate() const override;

    /**
     * @brief Gets the output sample rate configured for this plan.
     * @return The output sample rate in Hz.
     */
    double get_output_rate() const override;

    /**
     * @brief Calculates the *maximum* expected output length for a given input
     * length.
     * @details The actual output length per call might vary slightly.
     * @param input_length Length of the input signal block in samples.
     * @return Maximum expected output length for the block.
     */
    size_t get_output_length(size_t input_length) const override;

   private:
    // --- Configuration ---
    const AbstractBackend*
        owner_backend_;  // Non-owning pointer to backend for filter design
    ResampleSpec spec_;
    size_t interpolation_factor_;  // L
    size_t decimation_factor_;     // M
    size_t filter_length_;         // Length of the prototype FIR filter

    // --- Internal State ---
    std::vector<T> state_;  // Delay line for the polyphase filter structure
    size_t current_phase_;  // Current polyphase filter index (0 to L-1)
    // Add any other necessary state variables (e.g., fractional sample
    // position)

    // --- Precomputed Data / Internal Plans ---
    // Store the polyphase filter coefficients (L filters of length N/L)
    std::vector<std::vector<T>> polyphase_coeffs_;

    // Helper methods
    void design_filter();            // Designs the prototype FIR filter
    void build_polyphase_filters();  // Decomposes the prototype into polyphase
                                     // components
    size_t calculate_max_output(size_t input_len);  // Internal calculation
  };

  // --- Explicit Template Instantiations (Declaration) ---
  extern template class DefaultResamplePlanImpl<F32>;
  extern template class DefaultResamplePlanImpl<F64>;

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_DEFAULT_RESAMPLE_HPP
