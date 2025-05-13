/**
 * @file convolution.cpp
 * @brief Implements the constructors and fluent setters for ConvolutionParams
 * and CorrelationParams.
 */

#include "OmniDSP/params/convolution.hpp"  // Corresponding header

#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation

namespace OmniDSP {

  // --- ConvolutionParams Implementation ---

  ConvolutionParams::ConvolutionParams(
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
      msg = "ConvolutionParams Constructor: max_input_length ("
            + std::to_string(max_input_length_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (kernel_length_ == 0) {
      msg = "ConvolutionParams Constructor: kernel_length ("
            + std::to_string(kernel_length_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // ConvolutionType and ConvolutionMethod are enums, assume valid values are
    // passed. More complex validation (e.g., if FFT method requires lengths to
    // be powers of 2) would typically be handled during Spec creation or Plan
    // creation.
    if (logger) {
      logger->trace(
          "ConvolutionParams constructed: MaxInputLen={}, KernelLen={}, "
          "Type={}, MethodHint={}",
          max_input_length_,
          kernel_length_,
          static_cast<int>(type_),
          static_cast<int>(method_hint_));
    }
  }

  ConvolutionParams& ConvolutionParams::max_input_length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "ConvolutionParams::max_input_length: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    max_input_length_ = val;
    if (logger) logger->trace("ConvolutionParams::max_input_length to {}", val);
    return *this;
  }

  ConvolutionParams& ConvolutionParams::kernel_length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "ConvolutionParams::kernel_length: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    kernel_length_ = val;
    if (logger) logger->trace("ConvolutionParams::kernel_length to {}", val);
    return *this;
  }

  ConvolutionParams& ConvolutionParams::type(ConvolutionType val)
  {
    type_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace("ConvolutionParams::type to {}", static_cast<int>(val));
    return *this;
  }

  ConvolutionParams& ConvolutionParams::method_hint(ConvolutionMethod val)
  {
    method_hint_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace(
          "ConvolutionParams::method_hint to {}", static_cast<int>(val));
    return *this;
  }

  // --- CorrelationParams Implementation ---

  CorrelationParams::CorrelationParams(
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
      msg = "CorrelationParams Constructor: max_input_length ("
            + std::to_string(max_input_length_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (template_length_ == 0) {
      msg = "CorrelationParams Constructor: template_length ("
            + std::to_string(template_length_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }

    if (logger) {
      logger->trace(
          "CorrelationParams constructed: MaxInputLen={}, TemplateLen={}, "
          "Type={}, MethodHint={}",
          max_input_length_,
          template_length_,
          static_cast<int>(type_),
          static_cast<int>(method_hint_));
    }
  }

  CorrelationParams& CorrelationParams::max_input_length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "CorrelationParams::max_input_length: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    max_input_length_ = val;
    if (logger) logger->trace("CorrelationParams::max_input_length to {}", val);
    return *this;
  }

  CorrelationParams& CorrelationParams::template_length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "CorrelationParams::template_length: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    template_length_ = val;
    if (logger) logger->trace("CorrelationParams::template_length to {}", val);
    return *this;
  }

  CorrelationParams& CorrelationParams::type(ConvolutionType val)
  {
    type_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace("CorrelationParams::type to {}", static_cast<int>(val));
    return *this;
  }

  CorrelationParams& CorrelationParams::method_hint(ConvolutionMethod val)
  {
    method_hint_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace(
          "CorrelationParams::method_hint to {}", static_cast<int>(val));
    return *this;
  }

}  // namespace OmniDSP
