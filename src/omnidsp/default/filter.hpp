/**
 * @file filter.hpp (Default)
 * @brief Declares the concrete FIRFilterPlanImpl and IIRFilterPlanImpl classes
 * for the Default backend.
 */

#ifndef OMNIDSP_DEFAULT_FILTER_HPP
#define OMNIDSP_DEFAULT_FILTER_HPP

#include <OmniDSP/core_types.hpp>  // Status, F32, F64, C32, C64
#include <OmniDSP/filter.hpp>  // Public filter interfaces and specs (needed for IIRFilterCoef)
#include <complex>
#include <cstddef>  // For size_t
#include <span>     // For std::span
#include <vector>

#include "../interface/backend.hpp"  // Base PlanImpl interfaces

namespace OmniDSP::Default {

  /**
   * @brief Default backend implementation for an FIR Filter Plan.
   * @details Uses direct form convolution. Stateful.
   * @tparam T The data type (F32, F64, C32, C64).
   */
  template <typename T>
  class DefaultFIRFilterPlanImpl final : public Abstract::FIRFilterPlanImpl<T> {
   public:
    /**
     * @brief Constructs a DefaultFIRFilterPlanImpl.
     * @param coefficients The FIR filter coefficients (taps). Copied
     * internally.
     * @throws std::invalid_argument if coefficients vector is empty.
     */
    explicit DefaultFIRFilterPlanImpl(const std::vector<T>& coefficients);

    /**
     * @brief Destructor.
     */
    ~DefaultFIRFilterPlanImpl() override;

    // --- Disable Copy/Move ---
    DefaultFIRFilterPlanImpl(const DefaultFIRFilterPlanImpl&) = delete;
    DefaultFIRFilterPlanImpl& operator=(const DefaultFIRFilterPlanImpl&)
        = delete;
    DefaultFIRFilterPlanImpl(DefaultFIRFilterPlanImpl&&) = delete;
    DefaultFIRFilterPlanImpl& operator=(DefaultFIRFilterPlanImpl&&) = delete;

    // --- Interface Methods Implementation ---

    /**
     * @brief Applies the FIR filter to a block of input samples.
     * @param input A span representing the input signal block.
     * @param output A span representing the output buffer. Must have the same
     * size as input.
     * @return Status indicating success or failure.
     */
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;  // Not const

    /**
     * @brief Resets the internal state (delay line) of the filter to zero.
     * @return Status::Success.
     */
    [[nodiscard]] Status reset() override;  // Not const

    /**
     * @brief Gets the order of the filter (number of taps - 1).
     * @return The filter order.
     */
    size_t get_order() const override;

    /**
     * @brief Gets the number of filter coefficients (taps).
     * @return The number of taps.
     */
    size_t get_num_taps() const override;

    /**
     * @brief Gets the filter coefficients.
     * @return A span viewing the filter coefficients.
     */
    // Note: get_coefficients() is not in the base interface currently,
    // but often useful for derived classes or debugging.
    std::span<const T> get_coefficients() const;

   private:
    std::vector<T> coefficients_;  // Filter taps (b[n])
    std::vector<T> state_;  // Internal state (delay line) z^-1, z^-2, ...
    // size_t num_taps_; // Redundant, use coefficients_.size()
    // size_t order_; // Redundant, use get_order()
  };

  /**
   * @brief Default backend implementation for an IIR Filter Plan.
   * @details Uses transposed direct form II for applying second-order sections.
   * Stateful.
   * @tparam T The data type (typically F32 or F64).
   */
  template <typename T>  // T is typically real (F32, F64)
  class DefaultIIRFilterPlanImpl final : public Abstract::IIRFilterPlanImpl<T> {
   private:
    // Private struct to hold coefficients of the correct type T
    template <typename Type>
    struct InternalSOS {
      Type b0 = 1.0;
      Type b1 = 0.0;
      Type b2 = 0.0;
      Type a1 = 0.0;
      Type a2 = 0.0;  // Assuming a0 is normalized to 1
    };

   public:
    /**
     * @brief Constructs a DefaultIIRFilterPlanImpl from second-order sections.
     * @param sos_coefficients A vector of IIRFilterCoef representing the filter
     * sections.
     * @throws std::invalid_argument if sos_coefficients vector is empty.
     * @throws std::runtime_error if sos_coefficients contain non-normalized a0.
     */
    // *** UPDATED: Constructor takes non-templated IIRFilterCoef ***
    explicit DefaultIIRFilterPlanImpl(
        const std::vector<IIRFilterCoef>& sos_coefficients);

    /**
     * @brief Destructor.
     */
    ~DefaultIIRFilterPlanImpl() override;

    // --- Disable Copy/Move ---
    DefaultIIRFilterPlanImpl(const DefaultIIRFilterPlanImpl&) = delete;
    DefaultIIRFilterPlanImpl& operator=(const DefaultIIRFilterPlanImpl&)
        = delete;
    DefaultIIRFilterPlanImpl(DefaultIIRFilterPlanImpl&&) = delete;
    DefaultIIRFilterPlanImpl& operator=(DefaultIIRFilterPlanImpl&&) = delete;

    // --- Interface Methods Implementation ---

    /**
     * @brief Applies the IIR filter to a block of input samples.
     * @param input A span representing the input signal block.
     * @param output A span representing the output buffer. Must have the same
     * size as input.
     * @return Status indicating success or failure.
     */
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;  // Not const

    /**
     * @brief Resets the internal state (delay lines) of the filter to zero.
     * @return Status::Success.
     */
    [[nodiscard]] Status reset() override;  // Not const

    /**
     * @brief Gets the overall order of the filter (typically 2 * num_sections).
     * @return The filter order.
     */
    size_t get_order() const override;

    /**
     * @brief Gets the number of second-order sections used.
     * @return The number of sections.
     */
    size_t get_num_sections() const override;

    /**
     * @brief Gets the second-order section coefficients. (REMOVED - internal
     * representation changed)
     */
    // const std::vector<IIRFilterCoef>& get_sections() const; // REMOVED

   private:
    // std::vector<IIRFilterCoef> sos_coeffs_; // *** REMOVED - Store internal
    // format ***
    std::vector<InternalSOS<T>>
        internal_coeffs_;  // *** ADDED - Store coeffs as type T ***

    // State variables for transposed direct form II (2 per section)
    std::vector<T> state_;  // Size = 2 * num_sections
    // size_t num_sections_; // Redundant, use internal_coeffs_.size()
    // size_t order_; // Redundant, use get_order()

    // Helper to apply a single SOS (can remain private if only used internally)
    // T apply_sos(size_t section_idx, T input_sample); // Declaration if needed
  };

  // --- Explicit Template Instantiations (Declaration) ---
  extern template class DefaultFIRFilterPlanImpl<F32>;
  extern template class DefaultFIRFilterPlanImpl<F64>;
  extern template class DefaultFIRFilterPlanImpl<C32>;
  extern template class DefaultFIRFilterPlanImpl<C64>;

  extern template class DefaultIIRFilterPlanImpl<F32>;
  extern template class DefaultIIRFilterPlanImpl<F64>;

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_FILTER_HPP
