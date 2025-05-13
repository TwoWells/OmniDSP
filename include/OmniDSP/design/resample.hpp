/**
 * @file resample.hpp (design)
 * @brief Defines the Design::Resample structure for resampling design
 * specifications.
 */
#ifndef OMNIDSP_DESIGN_RESAMPLE_HPP
#define OMNIDSP_DESIGN_RESAMPLE_HPP

#include <cstddef>    // For size_t
#include <stdexcept>  // For std::logic_error
#include <utility>    // For std::move

#include "OmniDSP/core_types.hpp"         // For OMNIDSP_EXPORT
#include "OmniDSP/design/fir_filter.hpp"  // For Design::FIRFilter

namespace OmniDSP::Design {

  /**
   * @brief Fully resolved design specification for creating a resampling
   * processor.
   */
  struct OMNIDSP_EXPORT Resample {
    double input_rate;
    double output_rate;
    int quality;
    size_t up_factor_L;
    size_t down_factor_M;
    Design::FIRFilter prototype_fir_design;

    explicit Resample(
        double p_input_rate,
        double p_output_rate,
        int p_quality,
        size_t p_up_factor_L,
        size_t p_down_factor_M,
        Design::FIRFilter p_prototype_fir_design)
        : input_rate(p_input_rate),
          output_rate(p_output_rate),
          quality(p_quality),
          up_factor_L(p_up_factor_L),
          down_factor_M(p_down_factor_M),
          prototype_fir_design(std::move(p_prototype_fir_design))
    {
      if (input_rate <= 0.0 || output_rate <= 0.0) {
        throw std::logic_error(
            "Design::Resample: Input/Output rates must be positive.");
      }
      if (up_factor_L == 0 || down_factor_M == 0) {
        throw std::logic_error(
            "Design::Resample: Upsampling/Downsampling factors L/M cannot be "
            "zero.");
      }
    }
  };

}  // namespace OmniDSP::Design

#endif  // OMNIDSP_DESIGN_RESAMPLE_HPP
