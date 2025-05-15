/**
 * @file fft.hpp
 * @brief Defines parameters for FFT and RFFT operation specification.
 */

#ifndef OMNIDSP_PARAMS_FFT_HPP
#define OMNIDSP_PARAMS_FFT_HPP

#include <cstddef>    // For size_t
#include <ostream>    // For std::ostream
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::string in validation messages

#include "OmniDSP/core_types.hpp"  // For logging (spdlog is included in .cpp)

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

// spdlog include is deferred to .cpp

namespace OmniDSP::Params {

  /**
   * @brief Parameters for specifying a complex-to-complex Fast Fourier
   * Transform (FFT).
   *
   * This structure is typically used to specify the configuration for creating
   * an FFTPlan. Construction of this object validates the provided parameters.
   * Fluent setters are available for modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT FFT {
    size_t length_;  ///< Length of the FFT (number of complex input/output
                     ///< points). Must be positive.

    /**
     * @brief Explicit constructor that validates parameters.
     * @param p_length The length of the FFT. Must be positive.
     * Specific backends might impose further restrictions (e.g., power of 2).
     * @throws std::invalid_argument if parameters are invalid.
     */
    explicit FFT(size_t p_length);

    // --- Fluent Setters ---

    /**
     * @brief Sets the FFT length.
     * @param val The new FFT length. Must be positive.
     * @return A reference to this Params::FFT object.
     * @throws std::invalid_argument if val is not positive.
     */
    FFT& length(size_t val);
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of Params::FFT.
   * @param os The output stream.
   * @param params The Params::FFT object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const FFT& params)
  {
    os << "Params::FFT(Length: " << params.length_ << ")";
    return os;
  }

  /**
   * @brief Parameters for specifying a real-to-complex Fast Fourier Transform
   * (RFFT).
   *
   * This structure is typically used to specify the configuration for creating
   * an RFFTPlan. The length refers to the number of real input points. The
   * complex output will have length N/2 + 1. Construction of this object
   * validates the provided parameters. Fluent setters are available for
   * modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT RFFT {
    size_t
        length_;  ///< Length of the RFFT (number of real input points). Must be
                  ///< positive. The complex output will have length N/2 + 1.

    /**
     * @brief Explicit constructor that validates parameters.
     * @param p_length The length of the RFFT (number of real input points).
     * Must be positive. Specific backends might impose further restrictions
     * (e.g., power of 2, minimum length).
     * @throws std::invalid_argument if parameters are invalid.
     */
    explicit RFFT(size_t p_length);

    // --- Fluent Setters ---

    /**
     * @brief Sets the RFFT length (number of real input points).
     * @param val The new RFFT length. Must be positive.
     * @return A reference to this Params::RFFT object.
     * @throws std::invalid_argument if val is not positive.
     */
    RFFT& length(size_t val);
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of Params::RFFT.
   * @param os The output stream.
   * @param params The Params::RFFT object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const RFFT& params)
  {
    os << "Params::RFFT(Length: " << params.length_ << ")";
    return os;
  }

}  // namespace OmniDSP::Params

// Specialization of fmt::formatter for OmniDSP::Params::FFT and
// OmniDSP::Params::RFFT
template <>
struct fmt::formatter<OmniDSP::Params::FFT> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Params::RFFT> : fmt::ostream_formatter {};

#endif  // OMNIDSP_PARAMS_FFT_HPP
