/**
 * @file filter_design.cpp // Renamed
 * @brief Implements utility functions for common filter design tasks.
 */
#include "filter_design.hpp"  // Renamed include

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/filter.hpp>
#include <OmniDSP/window.hpp>
#include <algorithm>  // For std::max
#include <cmath>      // For std::max, std::pow etc. if needed by design logic
#include <stdexcept>  // For std::runtime_error
#include <string>     // For error messages
#include <vector>

#include "../interface/backend.hpp"  // For AbstractBackend definition

namespace OmniDSP::Utils {

  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>> design_resampling_prototype_filter(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSpec& window_spec)
  {
    if (!owner_backend) {
      // Cannot proceed without a backend to perform the design
      return std::unexpected(Status::InvalidArgument);
    }
    if (L == 0 || M == 0) {
      // Invalid resampling factors
      return std::unexpected(Status::InvalidArgument);
    }

    // 1. Determine filter specifications based on L, M, and quality
    double normalized_cutoff
        = 1.0 / (2.0 * static_cast<double>(std::max(L, M)));

    // Map quality to transition width / attenuation (Example values - adjust as
    // needed) These values could be refined or made configurable
    double transition_width_factor = 0.1;
    double stopband_attenuation_db = 60.0;
    if (quality <= 6) {
      transition_width_factor = 0.25;
      stopband_attenuation_db = 40.0;
    }
    else if (quality <= 12) {
      transition_width_factor = 0.1;
      stopband_attenuation_db = 60.0;
    }
    else {  // Higher quality
      transition_width_factor = 0.05;
      stopband_attenuation_db = 80.0;
    }
    double transition_width = transition_width_factor * normalized_cutoff;

    // 2. Create a non-templated FIRFilterSpec
    FIRFilterSpec filter_spec;
    filter_spec.type = FilterType::Lowpass;
    filter_spec.order = 0;  // Let design method determine order
    filter_spec.cutoff1 = normalized_cutoff;
    filter_spec.sample_rate = 1.0;  // Normalized design
    filter_spec.transition_width = transition_width;
    filter_spec.stopband_attenuation_db = stopband_attenuation_db;
    filter_spec.window = window_spec;  // Use the provided window spec

    // 3. Call the owner's filter design method via the AbstractBackend
    // interface
    OmniExpected<FIRCoefs<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, float>) {
      coeffs_expected = owner_backend->design_fir_filter_f32(filter_spec);
    }
    else if constexpr (std::is_same_v<T, double>) {
      coeffs_expected = owner_backend->design_fir_filter_f64(filter_spec);
    }
    else {
      // Should not happen due to template constraints, but defensive check
      return std::unexpected(Status::UnsupportedFeature);
    }

    if (!coeffs_expected) {
      // Propagate the error from the backend's design function
      return std::unexpected(coeffs_expected.error());
    }

    FIRCoefs<T> prototype_coeffs = std::move(coeffs_expected.value());

    if (prototype_coeffs.empty()) {
      // The design function succeeded but returned empty coefficients
      return std::unexpected(Status::Failure);
    }

    // 4. Scale coefficients by the upsampling factor L (important for
    // polyphase)
    T scale_factor = static_cast<T>(L);
    for (T& coeff : prototype_coeffs) {
      coeff *= scale_factor;
    }

    return prototype_coeffs;  // Return the scaled coefficients
  }

  // Explicit template instantiations
  template OmniExpected<FIRCoefs<F32>> design_resampling_prototype_filter<F32>(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSpec& window_spec);
  template OmniExpected<FIRCoefs<F64>> design_resampling_prototype_filter<F64>(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSpec& window_spec);

}  // namespace OmniDSP::Utils
