/**
 * @file resample.hpp (Default)
 * @brief Declares the concrete ResampleProcessorImpl class for the Default
 * backend.
 */

#ifndef OMNIDSP_DEFAULT_RESAMPLE_HPP
#define OMNIDSP_DEFAULT_RESAMPLE_HPP

#include <OmniDSP/core_types.hpp>  // For Status, F32, F64
#include <OmniDSP/design/fir_filter.hpp>  // For Design::FIRFilter (used by design_filter indirectly)
#include <OmniDSP/design/resample.hpp>  // For Design::Resample
#include <OmniDSP/window.hpp>  // For WindowSetup (used by Design::Resample)
#include <cstddef>             // For size_t
#include <memory>              // For std::unique_ptr
#include <span>                // For std::span
#include <vector>

#include "../interface/backend.hpp"  // For Abstract::ResamplePlanImpl base class
// and Abstract::Backend (for owner_backend_ pointer)

namespace OmniDSP::Default {

  // Forward declare FIR plan if used internally (not directly by this class's
  // public interface) template <typename T> class FIRFilterProcessorImpl; //
  // Not strictly needed if only design_fir_filter_fXX is called on
  // owner_backend_

  /**
   * @brief Default backend implementation for a Resampling Plan.
   * @details Uses a polyphase FIR filter approach for rational resampling.
   * Stateful.
   * @tparam T The REAL data type (F32 or F64).
   */
  template <typename T>
  class ResampleProcessorImpl final
      : public Abstract::ResampleProcessorImpl<T> {
   public:
    /**
     * @brief Constructs a ResampleProcessorImpl.
     * @param owner_backend Pointer to the backend instance creating this plan
     * (needed for filter design).
     * @param spec The resampling specification (input rate, output rate,
     * quality, window_setup).
     * @throws std::invalid_argument if spec contains invalid parameters or
     * owner_backend is null.
     * @throws std::runtime_error if filter design or internal setup fails.
     */
    ResampleProcessorImpl(
        const Abstract::Backend* owner_backend, const Design::Resample& spec);

    /**
     * @brief Destructor.
     */
    ~ResampleProcessorImpl() override;

    // --- Disable Copy/Move ---
    ResampleProcessorImpl(const ResampleProcessorImpl&) = delete;
    ResampleProcessorImpl& operator=(const ResampleProcessorImpl&) = delete;
    ResampleProcessorImpl(ResampleProcessorImpl&&) = delete;
    ResampleProcessorImpl& operator=(ResampleProcessorImpl&&) = delete;

    // --- Interface Methods Implementation ---
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;

    [[nodiscard]] Status reset() override;

    double get_input_rate() const override;
    double get_output_rate() const override;
    size_t get_output_length(size_t input_length) const override;

   private:
    // --- Configuration ---
    const Abstract::Backend*
        owner_backend_;  // Non-owning pointer to backend for filter design
    Design::Resample
        spec_;  // Stores the user-provided resampling specification
    size_t interpolation_factor_;  // L
    size_t decimation_factor_;     // M

    // --- Filter Data ---
    std::vector<T> prototype_coeffs_;  // ADDED: Stores the designed prototype
                                       // FIR filter coefficients
    size_t
        filter_length_;  // Length of the prototype FIR filter (number of taps)
    std::vector<std::vector<T>> polyphase_coeffs_;  // Polyphase filter bank

    // --- Internal State ---
    std::vector<T> state_;  // Delay line for the polyphase filter structure
    size_t current_phase_;  // Current polyphase filter index (0 to L-1)

    // Helper methods
    void design_filter();  // Designs the prototype FIR filter and stores in
                           // prototype_coeffs_
    void build_polyphase_filters();  // Decomposes prototype_coeffs_ into
                                     // polyphase_coeffs_

    // calculate_max_output was an internal helper in .cpp, get_output_length is
    // the public interface. If calculate_max_output is still used internally by
    // get_output_length, it should be declared here. For now, assuming
    // get_output_length directly implements its logic or calls a static
    // utility.
  };

  // --- Explicit Template Instantiations (Declaration) ---
  // These should match the ones in the .cpp file.
  extern template class ResampleProcessorImpl<F32>;
  extern template class ResampleProcessorImpl<F64>;

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_RESAMPLE_HPP
