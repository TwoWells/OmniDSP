/**
 * @file fir_filter.hpp (Default)
 * @brief Declares the concrete FIRFilterPlanImpl class and FIR design helpers
 * for the Default backend.
 */

#ifndef OMNIDSP_DEFAULT_FIR_FILTER_HPP
#define OMNIDSP_DEFAULT_FIR_FILTER_HPP

#include <OmniDSP/core_types.hpp>  // Status, F32, F64, C32, C64, OmniExpected, FIRCoefs
#include <OmniDSP/filter.hpp>  // For Design::FIRFilter (public spec)
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
  class FIRFilterPlanImpl final : public Abstract::FIRFilterPlanImpl<T> {
   public:
    /**
     * @brief Constructs a FIRFilterPlanImpl.
     * @param coefficients The FIR filter coefficients (taps). Copied
     * internally.
     * @throws std::invalid_argument if coefficients vector is empty.
     */
    explicit FIRFilterPlanImpl(const std::vector<T>& coefficients);

    /**
     * @brief Destructor.
     */
    ~FIRFilterPlanImpl() override;

    // --- Disable Copy/Move ---
    FIRFilterPlanImpl(const FIRFilterPlanImpl&) = delete;
    FIRFilterPlanImpl& operator=(const FIRFilterPlanImpl&) = delete;
    FIRFilterPlanImpl(FIRFilterPlanImpl&&) = delete;
    FIRFilterPlanImpl& operator=(FIRFilterPlanImpl&&) = delete;

    // --- Interface Methods Implementation ---
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] Status reset() override;
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
   * @tparam T Data type for coefficients (F32 or F64).
   * @param spec The fully resolved Design::FIRFilter.
   * @return OmniExpected<FIRCoefs<T>> The designed coefficients or an error
   * status.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>> generate_fir_filter_coeffs(
      const Design::FIRFilter& spec);

  // Explicit template instantiations (declarations for linking)
  extern template class FIRFilterPlanImpl<F32>;
  extern template class FIRFilterPlanImpl<F64>;
  extern template class FIRFilterPlanImpl<C32>;
  extern template class FIRFilterPlanImpl<C64>;

  extern template OmniExpected<FIRCoefs<F32>> generate_fir_filter_coeffs<F32>(
      const Design::FIRFilter& spec);
  extern template OmniExpected<FIRCoefs<F64>> generate_fir_filter_coeffs<F64>(
      const Design::FIRFilter& spec);

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_FIR_FILTER_HPP
