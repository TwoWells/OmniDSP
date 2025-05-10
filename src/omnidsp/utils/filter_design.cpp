/**
 * @file filter_design.cpp
 * @brief Implements utility functions for common filter design tasks.
 */
#include "filter_design.hpp"

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/filter.hpp>  // For FIRFilterSpec, FilterType
#include <OmniDSP/window.hpp>  // For WindowSetup, WindowType, WindowParams
#include <algorithm>           // For std::max
#include <cmath>               // For std::ceil, std::log10, std::abs
#include <stdexcept>           // For std::runtime_error
#include <string>              // For error messages
#include <vector>

#include "interface/backend.hpp"  // For AbstractBackend definition
#include "spdlog/spdlog.h"        // For logging

namespace OmniDSP::Utils {

  // Heuristic to estimate FIR filter order for resampling based on quality.
  size_t estimate_resampling_fir_order(
      size_t L,
      size_t M,
      int quality,
      double normalized_transition_width_factor)
  {
    double dF;
    if (quality <= 5) {
      dF = 0.2 * normalized_transition_width_factor;
    }
    else if (quality <= 10) {
      dF = 0.1 * normalized_transition_width_factor;
    }
    else {
      dF = 0.05 * normalized_transition_width_factor;
    }
    if (dF < 1e-6) dF = 1e-6;

    double desired_atten_dB = 40.0 + quality * 3.0;
    if (desired_atten_dB < 21) desired_atten_dB = 21;
    if (desired_atten_dB > 120) desired_atten_dB = 120;

    size_t order = static_cast<size_t>(
        std::ceil((desired_atten_dB - 7.95) / (2.285 * (2.0 * dF))));

    size_t min_order_factor = 2;
    if (quality > 10)
      min_order_factor = 4;
    else if (quality > 5)
      min_order_factor = 3;

    size_t min_order = min_order_factor * std::max(L, M);
    if (order < min_order) order = min_order;

    if (order < 16) order = 16;
    if (order > 2048) order = 2048;

    if (order % 2 != 0) {
      order++;
    }
    return order;
  }

  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>> design_resampling_prototype_filter(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSetup& input_window_setup)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner_backend) {
      logger->error(
          "design_resampling_prototype_filter: owner_backend is null.");
      return std::unexpected(Status::InvalidArgument);
    }
    if (L == 0 || M == 0) {
      logger->error(
          "design_resampling_prototype_filter: L ({}) or M ({}) is zero.",
          L,
          M);
      return std::unexpected(Status::InvalidArgument);
    }

    double normalized_cutoff
        = 1.0 / (2.0 * static_cast<double>(std::max(L, M)));
    double normalized_transition_width_factor_for_estimation
        = normalized_cutoff;
    size_t estimated_order = estimate_resampling_fir_order(
        L, M, quality, normalized_transition_width_factor_for_estimation);
    size_t num_taps = estimated_order + 1;

    logger->debug(
        "Resampling filter design: L={}, M={}, Q={}, NormCutoff={}, "
        "EstOrder={}, Taps={}",
        L,
        M,
        quality,
        normalized_cutoff,
        estimated_order,
        num_taps);

    // Create the FIRFilterSpec to pass to the backend's design function
    FIRFilterSpec prototype_fir_spec;
    prototype_fir_spec.type = FilterType::Lowpass;
    prototype_fir_spec.order = estimated_order;
    prototype_fir_spec.sample_rate = 1.0;  // Design is normalized
    prototype_fir_spec.cutoff1 = normalized_cutoff;
    // prototype_fir_spec.cutoff2 is std::nullopt by default, which is correct
    // for lowpass.

    // Correctly populate the window_setup member of FIRFilterSpec
    prototype_fir_spec.window_setup.type = input_window_setup.type;
    prototype_fir_spec.window_setup.params
        = input_window_setup.params;  // Copies the std::optional<WindowParams>
    prototype_fir_spec.window_setup.length = static_cast<int>(
        num_taps);  // CRITICAL: Set length based on estimated order

    // The members 'transition_width' and 'stopband_attenuation_db' are NOT part
    // of the current FIRFilterSpec struct. Do NOT attempt to set them.

    // Validate the constructed FIRFilterSpec before passing it to the backend.
    if (!prototype_fir_spec.validate()) {
      logger->error(
          "design_resampling_prototype_filter: Constructed internal "
          "FIRFilterSpec is invalid. Order={}, WindowLength={}",
          prototype_fir_spec.order,
          prototype_fir_spec.window_setup.length);
      return std::unexpected(Status::InvalidArgument);
    }

    OmniExpected<FIRCoefs<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, float>) {
      coeffs_expected
          = owner_backend->design_fir_filter_f32(prototype_fir_spec);
    }
    else if constexpr (std::is_same_v<T, double>) {
      coeffs_expected
          = owner_backend->design_fir_filter_f64(prototype_fir_spec);
    }
    else {
      logger->error(
          "design_resampling_prototype_filter: Unsupported data type.");
      return std::unexpected(Status::UnsupportedFeature);
    }

    if (!coeffs_expected) {
      logger->error(
          "design_resampling_prototype_filter: Backend FIR filter design "
          "failed with status: {}",
          static_cast<int>(coeffs_expected.error()));
      return std::unexpected(coeffs_expected.error());
    }

    FIRCoefs<T> prototype_coeffs = std::move(coeffs_expected.value());

    if (prototype_coeffs.empty()) {
      logger->error(
          "design_resampling_prototype_filter: Backend FIR filter design "
          "succeeded but returned empty coefficients.");
      return std::unexpected(Status::Failure);
    }

    T scale_factor = static_cast<T>(L);
    for (T& coeff : prototype_coeffs) {
      coeff *= scale_factor;
    }

    logger->debug(
        "Resampling filter designed successfully. Num taps: {}, First coeff "
        "(scaled): {}",
        prototype_coeffs.size(),
        prototype_coeffs.empty() ? 0.0 : prototype_coeffs[0]);

    return prototype_coeffs;
  }

  // Explicit template instantiations
  template OmniExpected<FIRCoefs<F32>> design_resampling_prototype_filter<F32>(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSetup& input_window_setup);
  template OmniExpected<FIRCoefs<F64>> design_resampling_prototype_filter<F64>(
      const Abstract::Backend* owner_backend,
      size_t L,
      size_t M,
      int quality,
      const WindowSetup& input_window_setup);

}  // namespace OmniDSP::Utils
