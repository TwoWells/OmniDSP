/**
 * @file filter_design.hpp // Renamed
 * @brief Utility functions for common filter design tasks, like designing
 * the prototype filter for resampling.
 */
#ifndef OMNIDSP_UTILS_FILTER_DESIGN_HPP  // Renamed include guard
#define OMNIDSP_UTILS_FILTER_DESIGN_HPP  // Renamed include guard

#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, F64
#include <OmniDSP/filter.hpp>      // For FIRFilterSpec, FIRCoefs
#include <OmniDSP/resample.hpp>    // For ResampleSpec (to get quality/window)
#include <OmniDSP/window.hpp>      // For WindowSpec
#include <cstddef>                 // For size_t
#include <vector>

// Forward declare AbstractBackend
namespace OmniDSP::Abstract {
  class AbstractBackend;
}

namespace OmniDSP::Utils {

  /**
   * @brief Designs a prototype lowpass FIR filter suitable for resampling.
   *
   * Calculates filter parameters based on resampling factors and quality,
   * then calls the appropriate design function from the provided backend
   * instance.
   *
   * @tparam T The data type for filter coefficients (F32 or F64).
   * @param owner_backend Pointer to the backend instance to use for filter
   * design. Must not be null.
   * @param L The interpolation factor.
   * @param M The decimation factor.
   * @param quality A quality hint (e.g., 0-20) to determine filter parameters.
   * @param window_spec The window specification to use for the FIR design.
   * @return An OmniExpected containing the designed FIR coefficients
   * (FIRCoefs<T>) on success, or a Status code on failure.
   */
  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>> design_resampling_prototype_filter(
      const Abstract::AbstractBackend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSpec& window_spec);

  // Explicit template instantiation declarations (implementation in .cpp)
  extern template OmniExpected<FIRCoefs<F32>>
  design_resampling_prototype_filter<F32>(
      const Abstract::AbstractBackend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSpec& window_spec);
  extern template OmniExpected<FIRCoefs<F64>>
  design_resampling_prototype_filter<F64>(
      const Abstract::AbstractBackend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSpec& window_spec);

}  // namespace OmniDSP::Utils

#endif  // OMNIDSP_UTILS_FILTER_DESIGN_HPP // Renamed include guard
