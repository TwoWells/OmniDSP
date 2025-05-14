/**
 * @file iir_filter.hpp (Default)
 * @brief Declares the concrete IIRFilterProcessorImpl class and IIR design
 * helpers for the Default backend.
 */

#ifndef OMNIDSP_DEFAULT_IIR_FILTER_HPP
#define OMNIDSP_DEFAULT_IIR_FILTER_HPP

#include <OmniDSP/core_types.hpp>  // Status, F32, F64, OmniExpected
#include <OmniDSP/iir_filter.hpp>  // For Design::IIRFilter, Coefs::SOS (public specs)
#include <complex>
#include <cstddef>  // For size_t
#include <span>     // For std::span
#include <vector>

#include "../interface/backend.hpp"  // Base PlanImpl interfaces (Abstract::IIRFilterPlanImpl)

namespace OmniDSP::Default {

  /**
   * @brief Default backend implementation for an IIR Filter Plan.
   * @details Uses transposed direct form II for applying second-order sections.
   * Stateful.
   * @tparam T The data type (typically F32 or F64).
   */
  template <typename T>
  class IIRFilterProcessorImpl final
      : public Abstract::IIRFilterProcessorImpl<T> {
   private:
    // Private struct to hold coefficients of the correct type T
    template <typename Type>
    struct InternalSOS {
      Type b0 = 1.0;
      Type b1 = 0.0;
      Type b2 = 0.0;
      Type a1 = 0.0;  // Assuming a0 is normalized to 1
      Type a2 = 0.0;
    };

   public:
    /**
     * @brief Constructs an IIRFilterProcessorImpl from second-order sections.
     * @param sos_coefficients A vector of Coefs::SOS representing the filter
     * sections.
     * @throws std::invalid_argument if sos_coefficients vector is empty.
     */
    explicit IIRFilterProcessorImpl(
        const Coefs::IIRFilterSOS& sos_coefficients);

    /**
     * @brief Destructor.
     */
    ~IIRFilterProcessorImpl() override;

    // --- Disable Copy/Move ---
    IIRFilterProcessorImpl(const IIRFilterProcessorImpl&) = delete;
    IIRFilterProcessorImpl& operator=(const IIRFilterProcessorImpl&) = delete;
    IIRFilterProcessorImpl(IIRFilterProcessorImpl&&) = delete;
    IIRFilterProcessorImpl& operator=(IIRFilterProcessorImpl&&) = delete;

    // --- Interface Methods Implementation ---
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] Status reset() override;
    size_t get_order() const override;
    size_t get_num_sections() const override;

   private:
    std::vector<InternalSOS<T>> internal_coeffs_;  // Store coeffs as type T
    std::vector<T> state_;  // State variables for transposed direct form II (2
                            // per section)
  };

  /**
   * @brief Designs IIR filter coefficients for the Default backend
   * (Placeholder).
   * @param spec The fully resolved Design::IIRFilter.
   * @return OmniExpected<Coefs::IIRFilterSOS> The designed SOS
   * coefficients or an error status.
   */
  [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS> generate_iir_filter_coeffs(
      const Design::IIRFilter& spec);

  // Explicit template instantiations (declarations for linking)
  extern template class IIRFilterProcessorImpl<F32>;
  extern template class IIRFilterProcessorImpl<F64>;

  // No instantiation for generate_iir_filter_coeffs as it's not templated on T

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_IIR_FILTER_HPP
