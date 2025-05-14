/**
 * @file filter_design.hpp
 * @brief Utility functions for common filter design tasks, like designing
 * the prototype filter for resampling.
 */
#ifndef OMNIDSP_UTILS_FILTER_DESIGN_HPP
#define OMNIDSP_UTILS_FILTER_DESIGN_HPP

#include <OmniDSP/coefs/fir_filter.hpp>  // For Design::FIRFilter, FIRCoefs (Design::FIRFilter now uses WindowSetup)
#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, F64
#include <OmniDSP/design/fir_filter.hpp>  // For Design::FIRFilter, FIRCoefs (Design::FIRFilter now uses WindowSetup)
#include <OmniDSP/window.hpp>  // For WindowSetup
#include <cstddef>             // For size_t
#include <vector>

// Forward declare AbstractBackend
namespace OmniDSP::Abstract {
  class Backend;
}

namespace OmniDSP::Utils {

  /**
   * @brief Designs a prototype lowpass FIR filter suitable for resampling.
   *
   * Calculates filter parameters based on resampling factors and quality,
   * then calls the appropriate design function from the provided backend
   * instance. The window shape and its specific parameters (e.g., beta for
   * Kaiser) are taken from the input_window_setup. The length of the window
   * used for FIR design is determined internally based on the estimated filter
   * order.
   *
   * @tparam T The data type for filter coefficients (F32 or F64).
   * @param owner_backend Pointer to the backend instance to use for filter
   * design. Must not be null.
   * @param L The interpolation factor.
   * @param M The decimation factor.
   * @param quality A quality hint (e.g., 0-15) to determine filter
   * order/parameters.
   * @param input_window_setup The window setup (type and parameters like beta)
   * to use for the FIR design. The 'length' field in this input_window_setup is
   * ignored by this function when setting up the Design::FIRFilter's
   * window_setup.length; filter order (and thus window length for design) is
   * determined internally by this utility.
   * @return An OmniExpected containing the designed FIR coefficients
   * (FIRCoefs<T>) on success, or a Status code on failure.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<Coefs::FIRFilter<T>>
  design_resampling_prototype_filter(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSetup& input_window_setup);

  // Explicit template instantiation declarations (implementation in .cpp)
  extern template OmniExpected<Coefs::FIRFilter<F32>>
  design_resampling_prototype_filter<F32>(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSetup& input_window_setup);
  extern template OmniExpected<Coefs::FIRFilter<F64>>
  design_resampling_prototype_filter<F64>(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSetup& input_window_setup);

}  // namespace OmniDSP::Utils

#endif  // OMNIDSP_UTILS_FILTER_DESIGN_HPP
