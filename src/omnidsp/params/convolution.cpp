/**
 * @file convolution.cpp
 * @brief Implements the constructors and fluent setters for Params::Convolution
 * and Params::Correlation.
 */

#include "OmniDSP/params/convolution.hpp"  // Corresponding header (includes omnidsp_export.hpp via core_types.hpp)

#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog
#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation

// OMNIDSP_EXPORT should be defined by omnidsp_export.hpp included via
// convolution.hpp No need to include omnidsp_export.hpp directly again if
// convolution.hpp does it.

/**
 * @namespace OmniDSP::Params
 * @brief Contains structures for specifying parameters for various DSP
 * operations before full design.
 */
namespace OmniDSP::Params {

  // --- Params::Convolution Implementation ---

  /**
   * @brief Constructor for Convolution parameters.
   * @param p_max_input_length Maximum anticipated input signal length. Must be
   * positive.
   * @param p_kernel_length Length of the convolution kernel. Must be positive.
   * @param p_type Type of convolution (e.g., Full, Same, Valid).
   * @param p_method_hint Hint for the convolution method (e.g., Auto, Direct,
   * FFT).
   * @throws std::invalid_argument if p_max_input_length or p_kernel_length is
   * zero.
   */
  OMNIDSP_EXPORT Convolution::Convolution(  // Added OMNIDSP_EXPORT
      size_t p_max_input_length,
      size_t p_kernel_length,
      ConvolutionType p_type,
      ConvolutionMethod p_method_hint)
      : max_input_length_(p_max_input_length),
        kernel_length_(p_kernel_length),
        type_(p_type),
        method_hint_(p_method_hint)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();  // Fallback logger
    }
    std::string msg;

    if (max_input_length_ == 0) {
      msg = "Params::Convolution Constructor: max_input_length ("
            + std::to_string(max_input_length_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (kernel_length_ == 0) {
      msg = "Params::Convolution Constructor: kernel_length ("
            + std::to_string(kernel_length_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Convolution constructed: {}", *this);
    }
  }

  /**
   * @brief Sets the maximum anticipated input length for the convolution.
   * @param val The new maximum input length. Must be positive.
   * @return A reference to this Params::Convolution object for chaining.
   * @throws std::invalid_argument if val is zero.
   */
  Convolution& Convolution::max_input_length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "Params::Convolution::max_input_length: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    max_input_length_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Convolution::max_input_length updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the length of the convolution kernel.
   * @param val The new kernel length. Must be positive.
   * @return A reference to this Params::Convolution object for chaining.
   * @throws std::invalid_argument if val is zero.
   */
  Convolution& Convolution::kernel_length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "Params::Convolution::kernel_length: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    kernel_length_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Convolution::kernel_length updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the type of convolution (e.g., Full, Same, Valid).
   * @param val The new convolution type.
   * @return A reference to this Params::Convolution object for chaining.
   */
  Convolution& Convolution::type(ConvolutionType val)
  {
    type_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Convolution::type updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the hint for the convolution method (e.g., Auto, Direct, FFT).
   * @param val The new convolution method hint.
   * @return A reference to this Params::Convolution object for chaining.
   */
  Convolution& Convolution::method_hint(ConvolutionMethod val)
  {
    method_hint_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Convolution::method_hint updated: {}", *this);
    }
    return *this;
  }

  // --- Params::Correlation Implementation ---

  /**
   * @brief Constructor for Correlation parameters.
   * @param p_max_input_length Maximum anticipated input signal length. Must be
   * positive.
   * @param p_template_length Length of the correlation template. Must be
   * positive.
   * @param p_type Type of correlation (e.g., Full, Same, Valid).
   * @param p_method_hint Hint for the correlation method (e.g., Auto, Direct,
   * FFT).
   * @throws std::invalid_argument if p_max_input_length or p_template_length is
   * zero.
   */
  OMNIDSP_EXPORT Correlation::Correlation(  // Added OMNIDSP_EXPORT
      size_t p_max_input_length,
      size_t p_template_length,
      ConvolutionType p_type,
      ConvolutionMethod p_method_hint)
      : max_input_length_(p_max_input_length),
        template_length_(p_template_length),
        type_(p_type),
        method_hint_(p_method_hint)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    std::string msg;

    if (max_input_length_ == 0) {
      msg = "Params::Correlation Constructor: max_input_length ("
            + std::to_string(max_input_length_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (template_length_ == 0) {
      msg = "Params::Correlation Constructor: template_length ("
            + std::to_string(template_length_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }

    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Correlation constructed: {}", *this);
    }
  }

  /**
   * @brief Sets the maximum anticipated input length for the correlation.
   * @param val The new maximum input length. Must be positive.
   * @return A reference to this Params::Correlation object for chaining.
   * @throws std::invalid_argument if val is zero.
   */
  Correlation& Correlation::max_input_length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "Params::Correlation::max_input_length: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    max_input_length_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Correlation::max_input_length updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the length of the correlation template.
   * @param val The new template length. Must be positive.
   * @return A reference to this Params::Correlation object for chaining.
   * @throws std::invalid_argument if val is zero.
   */
  Correlation& Correlation::template_length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "Params::Correlation::template_length: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    template_length_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Correlation::template_length updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the type of correlation (e.g., Full, Same, Valid).
   * @param val The new correlation type.
   * @return A reference to this Params::Correlation object for chaining.
   */
  Correlation& Correlation::type(ConvolutionType val)
  {
    type_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Correlation::type updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the hint for the correlation method (e.g., Auto, Direct, FFT).
   * @param val The new correlation method hint.
   * @return A reference to this Params::Correlation object for chaining.
   */
  Correlation& Correlation::method_hint(ConvolutionMethod val)
  {
    method_hint_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Correlation::method_hint updated: {}", *this);
    }
    return *this;
  }

}  // namespace OmniDSP::Params
