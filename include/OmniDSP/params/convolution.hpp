/**
 * @file convolution.hpp
 * @brief Defines parameters for Convolution and Correlation operation
 * specification.
 */

#ifndef OMNIDSP_PARAMS_CONVOLUTION_HPP
#define OMNIDSP_PARAMS_CONVOLUTION_HPP

#include <cstddef>    // For size_t
#include <ostream>    // For std::ostream
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::string in validation messages

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/types/convolution.hpp"  // For ConvolutionType, ConvolutionMethod and their operator<<

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

// spdlog include is deferred to .cpp

namespace OmniDSP::Params {

  /**
   * @brief Parameters for specifying a Convolution operation.
   *
   * This structure is typically used to specify the configuration for creating
   * a ConvolutionSpec or directly a ConvolutionPlan, especially when
   * characteristics like maximum input length and kernel length need to be
   * known upfront for resource allocation (e.g., determining FFT sizes).
   *
   * Construction of this object validates the provided parameters.
   * Fluent setters are available for modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT Convolution {
    size_t max_input_length_;  ///< Maximum anticipated length of the input
                               ///< signal. Must be positive.
    size_t kernel_length_;     ///< Length of the convolution kernel. Must be
                               ///< positive.
    ConvolutionType type_;     ///< Type of convolution (Full, Same, Valid).
    ConvolutionMethod
        method_hint_;  ///< Hint for the convolution method (Auto, Direct, FFT).

    /**
     * @brief Explicit constructor that validates parameters.
     * @param p_max_input_length Maximum anticipated input signal length.
     * @param p_kernel_length Length of the convolution kernel.
     * @param p_type Type of convolution.
     * @param p_method_hint Hint for the convolution method.
     * @throws std::invalid_argument if parameters are invalid (e.g., zero
     * lengths).
     */
    explicit Convolution(
        size_t p_max_input_length,
        size_t p_kernel_length,
        ConvolutionType p_type = ConvolutionType::Full,
        ConvolutionMethod p_method_hint = ConvolutionMethod::Auto);

    // --- Fluent Setters ---

    /**
     * @brief Sets the maximum anticipated input length.
     * @param val The new maximum input length. Must be positive.
     * @return A reference to this Params::Convolution object.
     * @throws std::invalid_argument if val is not positive.
     */
    Convolution& max_input_length(size_t val);

    /**
     * @brief Sets the kernel length.
     * @param val The new kernel length. Must be positive.
     * @return A reference to this Params::Convolution object.
     * @throws std::invalid_argument if val is not positive.
     */
    Convolution& kernel_length(size_t val);

    /**
     * @brief Sets the convolution type.
     * @param val The new convolution type.
     * @return A reference to this Params::Convolution object.
     */
    Convolution& type(ConvolutionType val);

    /**
     * @brief Sets the convolution method hint.
     * @param val The new convolution method hint.
     * @return A reference to this Params::Convolution object.
     */
    Convolution& method_hint(ConvolutionMethod val);
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * Params::Convolution.
   * @param os The output stream.
   * @param params The Params::Convolution object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const Convolution& params)
  {
    os << "Params::Convolution(MaxInputLen: " << params.max_input_length_
       << ", KernelLen: " << params.kernel_length_
       << ", Type: " << params.type_  // Uses ConvolutionType::operator<<
       << ", MethodHint: "
       << params.method_hint_  // Uses ConvolutionMethod::operator<<
       << ")";
    return os;
  }

  /**
   * @brief Parameters for specifying a Correlation operation.
   *
   * Similar to Params::Convolution, this structure is used for configuring
   * correlation. Construction of this object validates the provided parameters.
   * Fluent setters are available for modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT Correlation {
    size_t max_input_length_;  ///< Maximum anticipated length of the input
                               ///< signal. Must be positive.
    size_t template_length_;   ///< Length of the correlation template. Must be
                               ///< positive.
    ConvolutionType type_;     ///< Type of correlation (Full, Same, Valid -
                               ///< analogous to convolution).
    ConvolutionMethod
        method_hint_;  ///< Hint for the correlation method (Auto, Direct, FFT).

    /**
     * @brief Explicit constructor that validates parameters.
     * @param p_max_input_length Maximum anticipated input signal length.
     * @param p_template_length Length of the correlation template.
     * @param p_type Type of correlation.
     * @param p_method_hint Hint for the correlation method.
     * @throws std::invalid_argument if parameters are invalid (e.g., zero
     * lengths).
     */
    explicit Correlation(
        size_t p_max_input_length,
        size_t p_template_length,
        ConvolutionType p_type = ConvolutionType::Full,
        ConvolutionMethod p_method_hint = ConvolutionMethod::Auto);

    // --- Fluent Setters ---

    /**
     * @brief Sets the maximum anticipated input length.
     * @param val The new maximum input length. Must be positive.
     * @return A reference to this Params::Correlation object.
     * @throws std::invalid_argument if val is not positive.
     */
    Correlation& max_input_length(size_t val);

    /**
     * @brief Sets the template length.
     * @param val The new template length. Must be positive.
     * @return A reference to this Params::Correlation object.
     * @throws std::invalid_argument if val is not positive.
     */
    Correlation& template_length(size_t val);

    /**
     * @brief Sets the correlation type.
     * @param val The new correlation type.
     * @return A reference to this Params::Correlation object.
     */
    Correlation& type(ConvolutionType val);

    /**
     * @brief Sets the correlation method hint.
     * @param val The new correlation method hint.
     * @return A reference to this Params::Correlation object.
     */
    Correlation& method_hint(ConvolutionMethod val);
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * Params::Correlation.
   * @param os The output stream.
   * @param params The Params::Correlation object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const Correlation& params)
  {
    os << "Params::Correlation(MaxInputLen: " << params.max_input_length_
       << ", TemplateLen: " << params.template_length_
       << ", Type: " << params.type_  // Uses ConvolutionType::operator<<
       << ", MethodHint: "
       << params.method_hint_  // Uses ConvolutionMethod::operator<<
       << ")";
    return os;
  }

}  // namespace OmniDSP::Params

// Specialization of fmt::formatter for OmniDSP::Params::Convolution and
// OmniDSP::Params::Correlation
template <>
struct fmt::formatter<OmniDSP::Params::Convolution> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Params::Correlation> : fmt::ostream_formatter {};

#endif  // OMNIDSP_PARAMS_CONVOLUTION_HPP
