/**
 * @file fir_filter.hpp (Default)
 * @brief Declares the concrete FIRFilterProcessorImpl class and FIR design
 * helpers for the Default backend.
 */

#ifndef OMNIDSP_DEFAULT_FIR_FILTER_HPP
#define OMNIDSP_DEFAULT_FIR_FILTER_HPP

#include <OmniDSP/core_types.hpp>  // Status, F32, F64, C32, C64, OmniExpected, FIRCoefs, C32Vec, C64Vec
#include <OmniDSP/design/fir_filter.hpp>  // For Design::FIRFilter (public spec)
#include <complex>
#include <cstddef>  // For size_t
#include <span>     // For std::span
#include <vector>

#include "../interface/backend.hpp"  // Base PlanImpl interfaces (Abstract::FIRFilterPlanImpl)

namespace OmniDSP::Default {

  /**
   * @brief Default backend implementation for an FIR Filter Plan.
   * @details Uses direct form convolution. Stateful.
   * @tparam T The data type (F32, F64, C32, C64).
   */
  template <typename T>
  class FIRFilterProcessorImpl final
      : public Abstract::FIRFilterProcessorImpl<T> {
   public:
    /**
     * @brief Constructs a FIRFilterProcessorImpl.
     * @param coefficients The FIR filter coefficients (taps). Copied
     * internally.
     * @throws std::invalid_argument if coefficients vector is empty.
     */
    explicit FIRFilterProcessorImpl(const std::vector<T>& coefficients);

    /**
     * @brief Destructor.
     */
    ~FIRFilterProcessorImpl() override;

    // --- Disable Copy/Move ---
    FIRFilterProcessorImpl(const FIRFilterProcessorImpl&) = delete;
    FIRFilterProcessorImpl& operator=(const FIRFilterProcessorImpl&) = delete;
    FIRFilterProcessorImpl(FIRFilterProcessorImpl&&) = delete;
    FIRFilterProcessorImpl& operator=(FIRFilterProcessorImpl&&) = delete;

    // --- Interface Methods Implementation ---
    [[nodiscard]] OmniStatus execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] OmniStatus reset() override;
    size_t get_order() const override;
    size_t get_num_taps() const override;
    std::span<const T> get_coefficients()
        const;  // Specific to this impl, not in Abstract

   private:
    std::vector<T> coefficients_;  // Filter taps (b[n])
    std::vector<T> state_;         // Internal state (delay line)
  };

  /**
   * @brief Designs FIR filter coefficients using the windowed sinc method for
   * the Default backend.
   * @tparam T Data type for coefficients (F32, F64, std::complex<F32>,
   * std::complex<F64>).
   * @param spec The fully resolved Design::FIRFilter.
   * @return OmniExpected<Coefs::FIRFilter<T>> The designed coefficients or an
   * error status. If T is complex, Coefs::FIRFilter<T> should resolve to a
   * complex vector type (e.g. C32Vec).
   */
  template <typename T>
  [[nodiscard]] OmniExpected<Coefs::FIRFilter<T>> generate_fir_filter_coeffs(
      const Design::FIRFilter& spec);

  // Explicit template instantiations (declarations for linking)
  extern template class FIRFilterProcessorImpl<F32>;
  extern template class FIRFilterProcessorImpl<F64>;
  extern template class FIRFilterProcessorImpl<C32>;
  extern template class FIRFilterProcessorImpl<C64>;

  // For real types
  extern template OmniExpected<Coefs::FIRFilter<F32>>
  generate_fir_filter_coeffs<F32>(const Design::FIRFilter& spec);
  extern template OmniExpected<Coefs::FIRFilter<F64>>
  generate_fir_filter_coeffs<F64>(const Design::FIRFilter& spec);

  // For complex types (NEW - ensure Coefs::FIRFilter<std::complex<T>> matches
  // C<N>Vec or is compatible) Assuming Coefs::FIRFilter<std::complex<float>> is
  // compatible with C32Vec and Coefs::FIRFilter<std::complex<double>> is
  // compatible with C64Vec. The linker error indicates it's looking for a
  // function returning std::expected<std::vector<std::complex<...>>> So, we
  // assume Coefs::FIRFilter<std::complex<float>> is effectively
  // std::vector<std::complex<float>>
  extern template OmniExpected<Coefs::FIRFilter<C32>>
  generate_fir_filter_coeffs<C32>(const Design::FIRFilter& spec);
  extern template OmniExpected<Coefs::FIRFilter<C64>>
  generate_fir_filter_coeffs<C64>(const Design::FIRFilter& spec);

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_FIR_FILTER_HPP
