/**
 * @file resample.cpp
 * @brief Implements the ResampleProcessor class methods, forwarding calls to
 * backend implementations (Pimpl pattern).
 */

#include "OmniDSP/processor/resample.hpp"  // Corresponding header

// Core OmniDSP includes
#include "OmniDSP/core_types.hpp"

// Backend interface (for Abstract::ResampleProcessorImpl and Abstract::Backend)
#include "backend.hpp"

// Standard library includes
#include <memory>       // For std::unique_ptr
#include <span>         // For std::span
#include <stdexcept>    // For std::runtime_error
#include <type_traits>  // For std::is_same_v (used in header's create)
#include <utility>      // For std::move

// Logging
#include <spdlog/fmt/ostr.h>  // For custom types logging if needed
#include <spdlog/spdlog.h>

namespace OmniDSP::Processor {

  //--------------------------------------------------------------------------
  // Processor::Resample Method Definitions
  //--------------------------------------------------------------------------

  // NOTE: The static `Resample<T>::create` method definition has been removed
  // from this .cpp file. Its definition is now solely in the header
  // OmniDSP/include/OmniDSP/processor/resample.hpp, defined inline.

  /**
   * @brief Destructor for the Resample processor.
   * @tparam T The data type of the audio samples (e.g., float, double).
   */
  template <typename T>
  Resample<T>::~Resample()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      // Creating a string representation of the type T for logging.
      std::string type_name = "unknown_type";
      if constexpr (std::is_same_v<T, float>)
        type_name = "F32";
      else if constexpr (std::is_same_v<T, double>)
        type_name = "F64";
      logger->trace("Processor::Resample<{}> destructed.", type_name);
    }
    // Default destructor behavior is sufficient due to std::unique_ptr pimpl_
  }

  /**
   * @brief Move constructor for the Resample processor.
   * @tparam T The data type of the audio samples.
   * @param other The Resample object to move from.
   */
  template <typename T>
  Resample<T>::Resample(Resample&& other) noexcept
      : pimpl_(std::move(other.pimpl_))
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      std::string type_name = "unknown_type";
      if constexpr (std::is_same_v<T, float>)
        type_name = "F32";
      else if constexpr (std::is_same_v<T, double>)
        type_name = "F64";
      logger->trace("Processor::Resample<{}> move constructed.", type_name);
    }
  }

  /**
   * @brief Move assignment operator for the Resample processor.
   * @tparam T The data type of the audio samples.
   * @param other The Resample object to move assign from.
   * @return A reference to this Resample object.
   */
  template <typename T>
  Resample<T>& Resample<T>::operator=(Resample&& other) noexcept
  {
    auto logger = spdlog::get("OmniDSP");
    if (this != &other) {
      pimpl_ = std::move(other.pimpl_);
      if (logger && logger->should_log(spdlog::level::trace)) {
        std::string type_name = "unknown_type";
        if constexpr (std::is_same_v<T, float>)
          type_name = "F32";
        else if constexpr (std::is_same_v<T, double>)
          type_name = "F64";
        logger->trace("Processor::Resample<{}> move assigned.", type_name);
      }
    }
    return *this;
  }

  /**
   * @brief Executes the resampling operation on a block of audio data.
   * @tparam T The data type of the audio samples.
   * @param input A span representing the input signal.
   * @param output A span representing the output buffer. Its size must be
   * sufficient to hold the resampled output, as determined by
   * get_output_length().
   * @return OmniStatus::Success on successful resampling.
   * @return OmniStatus::InvalidOperation if the processor is not properly
   * initialized.
   * @return OmniStatus::SizeMismatch if the output span is too small
   * (potentially, depending on backend).
   * @return Other OmniStatus error codes if the underlying backend
   * implementation fails.
   */
  template <typename T>
  [[nodiscard]] OmniStatus Resample<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Processor::Resample::execute called. Input size: {}, Output span "
          "size: {}",
          input.size(),
          output.size());
    }

    if (!pimpl_) {
      if (logger) {
        logger->error(
            "Processor::Resample::execute called on an invalid (null pimpl_) "
            "instance.");
      }
      return OmniStatus::InvalidOperation;
    }

    // Basic size check before calling pimpl, though pimpl might do more
    // thorough checks.
    if (output.size() < get_output_length(input.size()) && !input.empty()) {
      if (logger) {
        logger->warn(
            "Processor::Resample::execute: Output span (size {}) might be too "
            "small for input (size {}), expected output length {}.",
            output.size(),
            input.size(),
            get_output_length(input.size()));
      }
      // Depending on desired strictness, could return SizeMismatch here.
      // For now, let pimpl handle it.
    }

    OmniStatus status = pimpl_->execute(input, output);

    if (logger && status != OmniStatus::Success
        && logger->should_log(spdlog::level::warn)) {
      logger->warn(
          "Processor::Resample::execute failed with status: {}",
          static_cast<int>(status));
    }
    return status;
  }

  /**
   * @brief Resets the internal state of the resampler.
   * This typically clears any internal filter delay lines or state variables.
   * @tparam T The data type of the audio samples.
   * @return OmniStatus::Success on successful reset.
   * @return OmniStatus::InvalidOperation if the processor is not properly
   * initialized.
   * @return Other OmniStatus error codes if the underlying backend
   * implementation fails to reset.
   */
  template <typename T>
  OmniStatus Resample<T>::reset()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::debug)) {
      logger->debug("Processor::Resample::reset called.");
    }

    if (!pimpl_) {
      if (logger) {
        logger->error(
            "Processor::Resample::reset called on an invalid (null pimpl_) "
            "instance.");
      }
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->reset();
  }

  /**
   * @brief Gets the input sample rate of the resampler.
   * @tparam T The data type of the audio samples.
   * @return The input sample rate in Hz.
   * @throws std::runtime_error if the processor instance is invalid (e.g.,
   * moved from).
   */
  template <typename T>
  double Resample<T>::get_input_rate() const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!pimpl_) {
      if (logger) {
        logger->error(
            "Processor::Resample::get_input_rate called on an invalid (null "
            "pimpl_) instance.");
      }
      // Throwing an exception for getters on an invalid object is a common
      // pattern.
      throw std::runtime_error(
          "Processor::Resample::get_input_rate called on an invalid instance.");
    }
    double rate = pimpl_->get_input_rate();
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Processor::Resample::get_input_rate returning: {}", rate);
    }
    return rate;
  }

  /**
   * @brief Gets the output sample rate of the resampler.
   * @tparam T The data type of the audio samples.
   * @return The output sample rate in Hz.
   * @throws std::runtime_error if the processor instance is invalid.
   */
  template <typename T>
  double Resample<T>::get_output_rate() const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!pimpl_) {
      if (logger) {
        logger->error(
            "Processor::Resample::get_output_rate called on an invalid (null "
            "pimpl_) instance.");
      }
      throw std::runtime_error(
          "Processor::Resample::get_output_rate called on an invalid "
          "instance.");
    }
    double rate = pimpl_->get_output_rate();
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Processor::Resample::get_output_rate returning: {}", rate);
    }
    return rate;
  }

  /**
   * @brief Calculates the expected number of output samples for a given number
   * of input samples.
   * @tparam T The data type of the audio samples.
   * @param input_length The number of input samples.
   * @return The expected number of output samples.
   * @throws std::runtime_error if the processor instance is invalid.
   */
  template <typename T>
  size_t Resample<T>::get_output_length(size_t input_length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!pimpl_) {
      if (logger) {
        logger->error(
            "Processor::Resample::get_output_length called on an invalid (null "
            "pimpl_) instance.");
      }
      throw std::runtime_error(
          "Processor::Resample::get_output_length called on an invalid "
          "instance.");
    }
    size_t length = pimpl_->get_output_length(input_length);
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Processor::Resample::get_output_length for input_length {} "
          "returning: {}",
          input_length,
          length);
    }
    return length;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // These instantiate the class template for F32 and F64.
  // The OMNIDSP_EXPORT macro handles visibility for shared libraries.
  template class OMNIDSP_EXPORT Resample<F32>;
  template class OMNIDSP_EXPORT Resample<F64>;

  // NOTE: Explicit instantiations for the static `Resample<T>::create` method
  // have been removed as its definition is now solely in the header.

}  // namespace OmniDSP::Processor
