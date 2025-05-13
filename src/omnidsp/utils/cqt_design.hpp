/**
 * @file cqt_design.hpp
 * @brief Declares internal utility functions for creating Design::CQT.
 * These are not part of the public OmniDSP::Utils API and are for use
 * within the cqt_design.cpp implementation or other internal utilities.
 */

#ifndef OMNIDSP_UTILS_INTERNAL_CQT_DESIGN_HPP
#define OMNIDSP_UTILS_INTERNAL_CQT_DESIGN_HPP

#include <cstddef>  // For size_t
#include <vector>   // If helpers operate on or return vectors

#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected (if helpers return these)

// Forward declare any types from public headers if needed by internal helpers,
// to avoid including full public headers here if possible.
// However, if helpers take CQTParams or Design::CQT, those headers would be
// needed. For now, this is a placeholder.

namespace OmniDSP {
  // Forward declare if needed and not wanting to include the full header.
  // struct CQTParams; // From "OmniDSP/params/cqt.hpp"
  // struct Design::CQT;   // From "OmniDSP/cqt.hpp"
}

namespace OmniDSP::Utils::Internal {

  // Example of a helper function declaration that might live here if factored
  // out:
  /*
  double calculate_cqt_q_factor(int bins_per_octave);

  std::vector<double> generate_cqt_bin_frequencies(
      double min_freq,
      double max_freq,
      double sample_rate,
      int bins_per_octave
  );

  size_t estimate_cqt_hop_length(
      double q_factor,
      double sample_rate,
      double lowest_bin_frequency
  );
  */

  // Currently, the main logic is within create_spec in cqt_design.cpp.
  // This header is ready for future internal helper declarations.

}  // namespace OmniDSP::Utils::Internal

#endif  // OMNIDSP_UTILS_INTERNAL_CQT_DESIGN_HPP
