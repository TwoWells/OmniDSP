/**
 * @file transform.hpp
 * @brief Defines lightweight specification structs for various transform
 * functions.
 * @details These structs act as type tags and parameter holders for configuring
 * transform operations. They include validation for parameters and support for
 * easy logging via std::ostream and fmt::format (spdlog compatible).
 */
#ifndef OMNIDSP_SPECS_TRANSFORM_HPP
#define OMNIDSP_SPECS_TRANSFORM_HPP

#include <cstddef>  // For size_t
#include <ostream>  // For std::ostream
#include <string>   // For std::string (used by DWT)
#include <vector>   // For std::vector (used by DWT, STFT)

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/specs/window.hpp"  // For OmniDSP::Window specs (used in STFT)
// Note: This creates a dependency. Consider if STFT should
// hold a more generic window spec identifier or use a variant.

// fmt library for spdlog compatibility and custom formatting
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>  // For formatting ranges like vectors

namespace OmniDSP {
  namespace Transform {

    /**
     * @brief Specification for a Fast Fourier Transform (FFT).
     */
    struct OMNIDSP_EXPORT FFT {
      size_t length;  ///< Length of the FFT (number of complex input/output
                      ///< points).

      /**
       * @brief Constructor for FFT specification.
       * @param len The length of the FFT. Must be positive.
       * @throws std::invalid_argument if length is zero.
       */
      explicit FFT(size_t len);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const FFT& spec);

    /**
     * @brief Specification for a Real Fast Fourier Transform (RFFT).
     */
    struct OMNIDSP_EXPORT RFFT {
      size_t
          real_length;  ///< Length of the RFFT (number of real input points).

      /**
       * @brief Constructor for RFFT specification.
       * @param len The real length of the RFFT. Must be positive.
       * @throws std::invalid_argument if real_length is zero.
       */
      explicit RFFT(size_t len);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const RFFT& spec);

    /**
     * @brief Specification for a Discrete Fourier Transform (DFT).
     */
    struct OMNIDSP_EXPORT DFT {
      size_t length;  ///< Length of the DFT.

      /**
       * @brief Constructor for DFT specification.
       * @param len The length of the DFT. Must be positive.
       * @throws std::invalid_argument if length is zero.
       */
      explicit DFT(size_t len);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const DFT& spec);

    /**
     * @brief Specification for a Short-Time Fourier Transform (STFT).
     */
    struct OMNIDSP_EXPORT STFT {
      size_t frame_length;
      size_t hop_length;
      size_t fft_length;
      OmniDSP::Window::Hann
          window_spec;  // Example, consider making this more generic
      bool center_frames;

      /**
       * @brief Constructor for STFT specification.
       * @param frame_len Length of the analysis window. Must be positive.
       * @param hop_len Hop size. Must be positive.
       * @param fft_len FFT length. Must be positive and typically >= frame_len.
       * @param win_spec Window specification struct.
       * @param center Whether to center frames.
       * @throws std::invalid_argument for invalid parameters.
       */
      explicit STFT(
          size_t frame_len,
          size_t hop_len,
          size_t fft_len,
          OmniDSP::Window::Hann win_spec = OmniDSP::Window::Hann{},
          bool center = true);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const STFT& spec);

    /**
     * @brief Enum for DCT types.
     */
    enum class DCT_Type { DCT_I, DCT_II, DCT_III, DCT_IV };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const DCT_Type& type);

    /**
     * @brief Specification for a Discrete Cosine Transform (DCT).
     */
    struct OMNIDSP_EXPORT DCT {
      size_t length;
      DCT_Type type;

      explicit DCT(size_t len, DCT_Type dct_type = DCT_Type::DCT_II);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const DCT& spec);

    /**
     * @brief Enum for DST types.
     */
    enum class DST_Type { DST_I, DST_II, DST_III, DST_IV };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const DST_Type& type);

    /**
     * @brief Specification for a Discrete Sine Transform (DST).
     */
    struct OMNIDSP_EXPORT DST {
      size_t length;
      DST_Type type;

      explicit DST(size_t len, DST_Type dst_type = DST_Type::DST_II);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const DST& spec);

    /**
     * @brief Specification for a Discrete Wavelet Transform (DWT).
     */
    struct OMNIDSP_EXPORT DWT {
      std::string wavelet_name;
      int levels;
      std::string mode;

      explicit DWT(
          std::string name,
          int num_levels = 1,
          std::string ext_mode = "symmetric");
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const DWT& spec);

    /**
     * @brief Specification for a Discrete Hartley Transform (DHT).
     */
    struct OMNIDSP_EXPORT DHT {
      size_t length;

      explicit DHT(size_t len);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const DHT& spec);

  }  // namespace Transform
}  // namespace OmniDSP

// fmt::formatter specializations
template <>
struct fmt::formatter<OmniDSP::Transform::FFT> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::RFFT> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::DFT> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::STFT> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::DCT_Type> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::DCT> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::DST_Type> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::DST> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::DWT> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Transform::DHT> : fmt::ostream_formatter {};

#endif  // OMNIDSP_SPECS_TRANSFORM_HPP
