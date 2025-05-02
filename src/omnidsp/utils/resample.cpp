/**
 * @file resample_utils.cpp
 * @brief Implements common utility functions for resampling.
 */
// *** UPDATED Include Path ***
#include "resample.hpp"  // Corresponding header

#include <OmniDSP/core_types.hpp>  // For Status enum
#include <cmath>    // For std::floor (needed for rounding in calculation)
#include <limits>   // For std::numeric_limits
#include <numeric>  // For std::gcd

namespace OmniDSP::Utils {

  // Definition moved from header
  Status calculate_resampling_factors(
      double in_rate, double out_rate, size_t& L, size_t& M)
  {
    if (in_rate <= 0.0 || out_rate <= 0.0) {
      return Status::InvalidArgument;
    }

    // Use large integer base for approximation to find rational L/M
    const long long factor_base = 16777216LL;  // 2^24

    // Check for potential overflow before multiplication
    if (out_rate > static_cast<double>(
            std::numeric_limits<long long>::max() / factor_base)) {
      return Status::Failure;  // Indicate potential overflow
    }

    // Calculate target ratio * base, round to nearest integer
    // Use std::floor for consistent rounding behavior across platforms compared
    // to simple cast + 0.5
    long long num_approx = static_cast<long long>(
        std::floor((out_rate / in_rate) * factor_base + 0.5));

    if (num_approx == 0 && out_rate > 0)
      num_approx = 1;  // Avoid zero upsample factor if output rate > 0
    if (num_approx < 0)
      return Status::Failure;  // Should not happen if rates > 0

    // Find greatest common divisor (GCD)
    long long common = std::gcd(num_approx, factor_base);

    // Check for potential issues before division
    if (common == 0) {
      return Status::Failure;  // Unexpected GCD result
    }
    if (common < 0) {
      common = -common;  // Ensure GCD is positive
    }

    // Check for division by zero before casting
    if (common == 0) {
      return Status::Failure;  // Avoid division by zero
    }

    // Perform division using checked common divisor
    long long l_long = num_approx / common;
    long long m_long = factor_base / common;

    // Check if results fit into size_t
    if (l_long > static_cast<long long>(std::numeric_limits<size_t>::max())
        || m_long
               > static_cast<long long>(std::numeric_limits<size_t>::max())) {
      return Status::Failure;  // Resulting factor overflows size_t
    }

    L = static_cast<size_t>(l_long);
    M = static_cast<size_t>(m_long);

    // Final check for zero factors
    if (L == 0 || M == 0) {
      if (out_rate > 0 && L == 0) L = 1;
      if (in_rate > 0 && M == 0) M = 1;
      if (L == 0 || M == 0) return Status::Failure;
    }

    return Status::Success;
  }

}  // namespace OmniDSP::Utils
