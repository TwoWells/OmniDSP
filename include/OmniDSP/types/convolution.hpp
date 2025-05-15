/**
 * @file convolution.hpp
 * @brief Defines common enumerations related to convolution and correlation.
 * @note This header is part of the types directory, separating type
 * definitions.
 */

#ifndef OMNIDSP_TYPES_CONVOLUTION_HPP
#define OMNIDSP_TYPES_CONVOLUTION_HPP

#include <ostream>  // For std::ostream
#include <string_view>  // For std::string_view (can be removed if not used elsewhere)

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT if enums need it

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP {

  /**
   * @brief Specifies the type (output size/boundary handling) for
   * convolution/correlation.
   */
  enum class OMNIDSP_EXPORT ConvolutionType {
    Full,  ///< The full discrete linear convolution of the inputs. Output size
           ///< is input_len + kernel_len - 1.
    Same,  ///< The output is the same size as the input, centered with respect
           ///< to the 'full' output.
    Valid  ///< The output consists only of those parts of the convolution that
           ///< are computed without zero-padding. Output size is max(input_len,
           ///< kernel_len) - min(input_len, kernel_len) + 1.
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * ConvolutionType.
   * @param os The output stream.
   * @param type The ConvolutionType enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, ConvolutionType type)
  {
    switch (type) {
      case ConvolutionType::Full:
        os << "Full";
        break;
      case ConvolutionType::Same:
        os << "Same";
        break;
      case ConvolutionType::Valid:
        os << "Valid";
        break;
      default:
        os << "Unknown ConvolutionType";
        break;
    }
    return os;
  }

  /**
   * @brief Specifies the underlying algorithm to use for
   * convolution/correlation.
   */
  enum class OMNIDSP_EXPORT ConvolutionMethod {
    Auto,    ///< Automatically choose the best method based on input sizes.
    Direct,  ///< Use direct, time-domain computation.
    FFT      ///< Use FFT-based computation in the frequency domain.
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * ConvolutionMethod.
   * @param os The output stream.
   * @param method The ConvolutionMethod enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, ConvolutionMethod method)
  {
    switch (method) {
      case ConvolutionMethod::Auto:
        os << "Auto";
        break;
      case ConvolutionMethod::Direct:
        os << "Direct";
        break;
      case ConvolutionMethod::FFT:
        os << "FFT";
        break;
      default:
        os << "Unknown ConvolutionMethod";
        break;
    }
    return os;
  }

  /**
   * @brief Specifies how to pad signals, typically before operations like
   * FFT-based convolution.
   */
  enum class OMNIDSP_EXPORT PaddingMode {
    Zeros,     ///< Pad with zeros. Also known as zero-padding. This is the most
               ///< common mode for FFT-based convolution.
    Constant,  ///< Pad with a constant value. The constant value would need to
               ///< be specified elsewhere.
    Reflect,   ///< Pad by reflecting the signal at the boundaries (e.g.,
               ///< abc|cba). The edge value is not repeated.
    Symmetric,  ///< Pad by symmetrically reflecting the signal at the
                ///< boundaries, including the edge value (e.g., abc|cba where
                ///< 'a' is the edge for cba). SciPy calls this 'reflect_type =
                ///< even'.
    Wrap  ///< Pad by wrapping the signal around (circular padding). Also known
          ///< as periodic padding.
    // Edge,    ///< Pad with the edge values of the signal. (SciPy calls this
    // 'edge') - Can be added if needed.
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of PaddingMode.
   * @param os The output stream.
   * @param mode The PaddingMode enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, PaddingMode mode)
  {
    switch (mode) {
      case PaddingMode::Zeros:
        os << "Zeros";
        break;
      case PaddingMode::Constant:
        os << "Constant";
        break;
      case PaddingMode::Reflect:
        os << "Reflect";
        break;
      case PaddingMode::Symmetric:
        os << "Symmetric";
        break;
      case PaddingMode::Wrap:
        os << "Wrap";
        break;
      default:
        os << "Unknown PaddingMode";
        break;
    }
    return os;
  }

  // Removed old get_..._name functions:
  // get_convolution_type_name(ConvolutionType type)
  // get_convolution_method_name(ConvolutionMethod method)
  // get_padding_mode_name(PaddingMode mode)

}  // namespace OmniDSP

// fmt::formatter specializations for direct logging with spdlog using {}
template <>
struct fmt::formatter<OmniDSP::ConvolutionType> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::ConvolutionMethod> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::PaddingMode> : fmt::ostream_formatter {};

#endif  // OMNIDSP_TYPES_CONVOLUTION_HPP
