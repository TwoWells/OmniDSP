/**
 * @file resample.cpp (IntelIPP)
 * @brief Implements the Intel IPP backend ResampleProcessorImpl class using
 * Intel IPP FIRMR.
 */

#include "resample.hpp"  // Include the corresponding header file

#include <ipps.h>             // For IPP functions
#include <spdlog/fmt/ostr.h>  // For logging custom types with operator<<

#include <OmniDSP/core_types.hpp>  // For Status, OmniException, F32, F64, Coefs::FIRFilter, OmniStatus::operator<<
#include <OmniDSP/design/fir_filter.hpp>  // For Design::FIRFilter (part of Design::Resample) and its operator<<
#include <OmniDSP/design/resample.hpp>  // For Design::Resample definition and its operator<<
#include <cmath>  // For std::abs (not directly used here, but common)
#include <iostream>  // For std::cerr (used by some logging, though spdlog is primary)
#include <memory>  // For std::unique_ptr (not directly used in this file's logic but often in context)
#include <numeric>    // For std::gcd (not needed here, L/M from spec)
#include <span>       // For std::span
#include <sstream>    // For std::ostringstream
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For std::string in exceptions
#include <utility>    // For std::move
#include <vector>     // For std::vector (used for pTaps)

#include "details.hpp"  // Include the intelipp utility header for IPP status checks and type helpers
#include "interface/backend.hpp"  // For Abstract::Backend
#include "spdlog/spdlog.h"

/**
 * @namespace OmniDSP::IntelIPP
 * @brief Contains the Intel IPP backend implementations for OmniDSP operations.
 */
namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // ResampleProcessorImpl Method Definitions
  //--------------------------------------------------------------------------

  /**
   * @brief Constructor for the Intel IPP ResampleProcessorImpl.
   * @details Initializes the IPP FIRMR (Finite Impulse Response Multi-Rate)
   * resampler. This involves obtaining prototype filter coefficients from the
   * owner backend, scaling them, and then initializing the IPP FIRMR
   * specification structure.
   * @param owner Pointer to the Abstract::Backend instance that owns this
   * processor. Used for designing the prototype FIR filter. Must not be null.
   * @param spec The fully resolved Design::Resample specification.
   * @throws OmniException if owner_backend is null, resampling factors are
   * invalid, prototype filter design fails, or if IPP initialization fails
   * (e.g., memory allocation, invalid parameters).
   */
  template <typename T>
  ResampleProcessorImpl<T>::ResampleProcessorImpl(
      const Abstract::Backend* owner, const Design::Resample& spec)
      : spec_(spec),
        input_rate_(spec.input_rate),
        output_rate_(spec.output_rate),
        quality_(spec.quality),
        p_spec_mem_(nullptr),  // Initialize pointers
        p_buffer_(nullptr),
        spec_mem_size_(0),
        buffer_size_(0),
        p_ipp_fir_spec_(nullptr)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Creating IntelIPP::ResampleProcessorImpl with Design: {}", spec_);
    }

    if (!owner) {
      std::string msg
          = "IntelIPP::ResampleProcessorImpl requires a non-null owner backend "
            "pointer.";
      if (logger) logger->error(msg);
      throw OmniException(msg, OmniStatus::InvalidArgument);
    }

    IppStatus ipp_status = ippStsNoErr;
    size_t L = spec_.up_factor_L;
    size_t M = spec_.down_factor_M;

    if (L == 0 || M == 0) {
      std::ostringstream msg_stream;
      msg_stream << "IntelIPP::ResampleProcessorImpl: Resampling factors L="
                 << L << " or M=" << M << " from spec are zero.";
      if (logger) logger->error(msg_stream.str());
      throw OmniException(msg_stream.str(), OmniStatus::InvalidArgument);
    }

    int upFactor = static_cast<int>(L);
    int downFactor = static_cast<int>(M);
    int upPhase = 0;
    int downPhase = 0;

    OmniExpected<Coefs::FIRFilter<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, float>) {
      coeffs_expected
          = owner->design_fir_filter_f32(spec_.prototype_fir_design);
    }
    else if constexpr (std::is_same_v<T, double>) {
      coeffs_expected
          = owner->design_fir_filter_f64(spec_.prototype_fir_design);
    }
    else {
      std::string msg
          = "IntelIPP::ResampleProcessorImpl: Unsupported data type for "
            "prototype filter design.";
      if (logger)
        logger->critical(
            msg);  // This should be caught by static_assert in header
      throw OmniException(msg, OmniStatus::UnsupportedFeature);
    }

    if (!coeffs_expected) {
      std::ostringstream msg_stream;
      msg_stream << "IntelIPP::ResampleProcessorImpl: Failed to design "
                    "internal prototype "
                 << "filter using owner backend. Status: "
                 << coeffs_expected.error();
      if (logger) logger->error(msg_stream.str());
      throw OmniException(msg_stream.str(), coeffs_expected.error());
    }

    Coefs::FIRFilter<T> pTaps_unscaled = std::move(coeffs_expected.value());
    if (pTaps_unscaled.empty()) {
      std::string msg
          = "IntelIPP::ResampleProcessorImpl: Internal prototype filter design "
            "resulted in empty coefficients.";
      if (logger) logger->error(msg);
      throw OmniException(msg, OmniStatus::Failure);
    }

    Coefs::FIRFilter<T> pTaps = pTaps_unscaled;
    T scale_factor = static_cast<T>(L);
    for (T& coeff : pTaps) {
      coeff *= scale_factor;
    }
    int tapsLen = static_cast<int>(pTaps.size());

    IppDataType ippDataType;
    if constexpr (std::is_same_v<T, float>) {
      ippDataType = ipp32f;
    }
    else if constexpr (std::is_same_v<T, double>) {
      ippDataType = ipp64f;
    }
    else {
      std::string msg
          = "IntelIPP::ResampleProcessorImpl: Unsupported data type T in "
            "constructor.";
      if (logger) logger->critical(msg);
      throw OmniException(msg, OmniStatus::UnsupportedFeature);
    }

    ipp_status = ippsFIRMRGetSize(
        tapsLen,
        upFactor,
        downFactor,
        ippDataType,
        &spec_mem_size_,
        &buffer_size_);
    OMNI_CHECK_IPP_STATUS_THROW(
        ipp_status, "IntelIPP Resample: ippsFIRMRGetSize failed");

    p_spec_mem_ = ippsMalloc_8u(spec_mem_size_);
    if (buffer_size_ > 0) {
      p_buffer_ = ippsMalloc_8u(buffer_size_);
    }
    else {
      p_buffer_ = nullptr;
    }

    if (!p_spec_mem_ || (buffer_size_ > 0 && !p_buffer_)) {
      ippsFree(p_spec_mem_);  // Safe to call on nullptr
      ippsFree(p_buffer_);    // Safe to call on nullptr
      p_spec_mem_ = nullptr;
      p_buffer_ = nullptr;
      std::string msg
          = "IntelIPP::ResampleProcessorImpl: Failed to allocate memory for "
            "IPP FIRMR spec or buffer.";
      if (logger) logger->error(msg);
      throw OmniException(msg, OmniStatus::AllocationError);
    }

    const Details::GetIPPType<T>* pTaps_ipp
        = reinterpret_cast<const Details::GetIPPType<T>*>(pTaps.data());

    if constexpr (std::is_same_v<T, float>) {
      p_ipp_fir_spec_ = reinterpret_cast<IppsFIRSpec_32f*>(p_spec_mem_);
      ipp_status = ippsFIRMRInit_32f(
          pTaps_ipp,
          tapsLen,
          upFactor,
          upPhase,
          downFactor,
          downPhase,
          static_cast<IppsFIRSpec_32f*>(p_ipp_fir_spec_));
    }
    else {  // double
      p_ipp_fir_spec_ = reinterpret_cast<IppsFIRSpec_64f*>(p_spec_mem_);
      ipp_status = ippsFIRMRInit_64f(
          pTaps_ipp,
          tapsLen,
          upFactor,
          upPhase,
          downFactor,
          downPhase,
          static_cast<IppsFIRSpec_64f*>(p_ipp_fir_spec_));
    }

    if (ipp_status != ippStsNoErr) {
      ippsFree(p_spec_mem_);
      ippsFree(p_buffer_);
      p_spec_mem_ = nullptr;
      p_buffer_ = nullptr;
      p_ipp_fir_spec_ = nullptr;
      OMNI_CHECK_IPP_STATUS_THROW(
          ipp_status, "IntelIPP Resample: ippsFIRMRInit failed");
    }

    if (logger && logger->should_log(spdlog::level::debug)) {
      logger->debug(
          "IntelIPP::ResampleProcessorImpl created: IR={}, OR={}, Q={}, "
          "Taps={}, L={}, M={}",
          input_rate_,
          output_rate_,
          quality_,
          tapsLen,
          L,
          M);
    }
  }

  /**
   * @brief Destructor for the Intel IPP ResampleProcessorImpl.
   * @details Frees the memory allocated for the IPP FIRMR specification and
   * work buffer.
   * @tparam T Data type of the samples.
   */
  template <typename T>
  ResampleProcessorImpl<T>::~ResampleProcessorImpl()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "IntelIPP::ResampleProcessorImpl<{}> destructed.", typeid(T).name());
    }
    ippsFree(p_spec_mem_);
    ippsFree(p_buffer_);
  }

  /**
   * @brief Executes the resampling process using IPP FIRMR.
   * @tparam T Data type of the samples.
   * @param input A span of constant input samples.
   * @param output A span for the resampled output samples. The size of this
   * span dictates how many output samples IPP will attempt to produce.
   * @return OmniStatus::Success if the operation is successful.
   * @return OmniStatus::NotInitialized if the IPP specification structure is
   * null.
   * @return OmniStatus::SizeMismatch if output span is empty when input is not
   * (potentially).
   * @return Other OmniStatus codes corresponding to IPP errors.
   */
  template <typename T>
  OmniStatus ResampleProcessorImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "IntelIPP::ResampleProcessorImpl::execute called. Input size: {}, "
          "Output span size: {}",
          input.size(),
          output.size());
    }

    if (!p_ipp_fir_spec_) {
      if (logger)
        logger->warn(
            "IntelIPP::ResampleProcessorImpl::execute: Not initialized "
            "(p_ipp_fir_spec_ is null).");
      return OmniStatus::NotInitialized;
    }

    size_t input_len = input.size();
    if (input_len == 0) {
      if (!output.empty()) {
        std::fill(output.begin(), output.end(), T{0});
      }
      if (logger && logger->should_log(spdlog::level::trace)) {
        logger->trace(
            "IntelIPP::ResampleProcessorImpl::execute: Empty input, output "
            "zeroed (if not empty).");
      }
      return OmniStatus::Success;
    }

    // IPP's ippsFIRMR function takes the number of *output* samples to
    // generate. The caller must ensure output.size() is appropriate.
    // get_output_length() provides an estimate, but IPP FIRMR might have
    // different internal buffering. For safety, we use output.size() as
    // numOutputSamples.
    int numOutputSamples = static_cast<int>(output.size());

    if (numOutputSamples <= 0 && input_len > 0) {
      if (logger)
        logger->warn(
            "IntelIPP::ResampleProcessorImpl::execute: numOutputSamples ({}) "
            "is zero or negative, but input exists ({} samples). Output span "
            "might be too small or empty.",
            numOutputSamples,
            input_len);
      if (numOutputSamples == 0)
        return OmniStatus::Success;  // No output to produce
      // If numOutputSamples < 0, it's an error with the span.
      return OmniStatus::SizeMismatch;
    }

    // Check if output span is sufficient for at least one sample if input is
    // present
    size_t min_expected_output = get_output_length(input_len);
    if (min_expected_output > 0 && output.empty()) {
      if (logger)
        logger->warn(
            "IntelIPP::ResampleProcessorImpl::execute: Output span is empty "
            "but expected {} output samples.",
            min_expected_output);
      return OmniStatus::SizeMismatch;
    }

    IppStatus ipp_status = ippStsNoErr;
    const Details::GetIPPType<T>* p_in_ipp
        = reinterpret_cast<const Details::GetIPPType<T>*>(input.data());
    Details::GetIPPType<T>* p_out_ipp
        = reinterpret_cast<Details::GetIPPType<T>*>(output.data());

    if constexpr (std::is_same_v<T, float>) {
      ipp_status = ippsFIRMR_32f(
          p_in_ipp,
          p_out_ipp,
          numOutputSamples,
          static_cast<IppsFIRSpec_32f*>(p_ipp_fir_spec_),
          nullptr,
          nullptr,
          p_buffer_);
    }
    else {  // double
      ipp_status = ippsFIRMR_64f(
          p_in_ipp,
          p_out_ipp,
          numOutputSamples,
          static_cast<IppsFIRSpec_64f*>(p_ipp_fir_spec_),
          nullptr,
          nullptr,
          p_buffer_);
    }

    OmniStatus os_status = Details::ipp_status_to_omnidsp_status(ipp_status);
    if (os_status != OmniStatus::Success && logger) {
      logger->warn(
          "IntelIPP::ResampleProcessorImpl::execute: ippsFIRMR failed with IPP "
          "status {}, OmniStatus {}. InputLen: {}, OutputLenProvided: {}",
          static_cast<int>(ipp_status),
          os_status,
          input_len,
          numOutputSamples);
    }
    return os_status;
  }

  /**
   * @brief Resets the internal state of the Intel IPP FIRMR resampler.
   * @details This typically involves zeroing out the internal delay lines.
   * For IPP FIRMR, this might mean zeroing the work buffer if it holds state,
   * or re-initializing the specification if that's how IPP manages state reset.
   * The current implementation zeros the work buffer.
   * @tparam T Data type of the samples.
   * @return OmniStatus::Success if the work buffer is zeroed (if applicable).
   * @return OmniStatus::NotInitialized if the processor was not properly
   * initialized.
   */
  template <typename T>
  OmniStatus ResampleProcessorImpl<T>::reset()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::debug)) {
      logger->debug("IntelIPP::ResampleProcessorImpl::reset() called.");
    }

    if (!p_ipp_fir_spec_) {  // Check if spec was initialized
      if (logger)
        logger->warn(
            "IntelIPP::ResampleProcessorImpl::reset: Not initialized "
            "(p_ipp_fir_spec_ is null).");
      return OmniStatus::NotInitialized;
    }

    // IPP FIRMR functions often manage their state internally within the spec
    // structure. Zeroing the external buffer might not be sufficient or correct
    // for all IPP FIR variants. For ippsFIRMR, the delay line is part of the
    // pSpec. A full reset would typically involve re-calling ippsFIRMRInit.
    // However, if the intention is just to clear any transient processing
    // buffer:
    if (p_buffer_ && buffer_size_ > 0) {
      IppStatus ipp_status = ippsZero_8u(p_buffer_, buffer_size_);
      if (ipp_status != ippStsNoErr) {
        if (logger)
          logger->warn(
              "IntelIPP::ResampleProcessorImpl::reset: Failed to zero work "
              "buffer, IPP status: {}. True state might persist in spec.",
              static_cast<int>(ipp_status));
        // Not returning error here as it's a best-effort on the work buffer.
      }
    }
    if (logger)
      logger->debug(
          "IntelIPP::ResampleProcessorImpl::reset() attempted to zero work "
          "buffer. For full state reset, re-initialization of IPP spec might "
          "be needed.");
    return OmniStatus::Success;  // Or NotImplemented if true reset isn't simple
  }

  /**
   * @brief Gets the input sample rate of the resampler.
   * @tparam T Data type of the samples.
   * @return The input sample rate in Hz, as stored in the design specification.
   */
  template <typename T>
  double ResampleProcessorImpl<T>::get_input_rate() const
  {
    return input_rate_;
  }

  /**
   * @brief Gets the output sample rate of the resampler.
   * @tparam T Data type of the samples.
   * @return The output sample rate in Hz, as stored in the design
   * specification.
   */
  template <typename T>
  double ResampleProcessorImpl<T>::get_output_rate() const
  {
    return output_rate_;
  }

  /**
   * @brief Calculates the expected number of output samples for a given number
   * of input samples.
   * @details This uses the rational resampling formula: output_length =
   * ceil(input_length * L / M). More precise calculation for block-based
   * processing might involve considering filter delay/state. IPP FIRMR's output
   * count is directly specified by the user in the execute call. This function
   * provides an estimate based on the rate change.
   * @tparam T Data type of the samples.
   * @param input_length The number of input samples.
   * @return The estimated number of output samples. Returns 0 if rates or
   * factors are invalid, or if input_length is 0.
   */
  template <typename T>
  size_t ResampleProcessorImpl<T>::get_output_length(size_t input_length) const
  {
    if (input_rate_ <= 0.0 || output_rate_ <= 0.0 || input_length == 0) {
      return 0;
    }
    size_t L = spec_.up_factor_L;
    size_t M = spec_.down_factor_M;

    if (M == 0) return 0;  // Avoid division by zero

    // Standard formula for output length in polyphase resampling for a block.
    // (input_length * L) / M can truncate.
    // A common way to ensure enough space is ceil(input_length * L / M).
    // Or, for integer arithmetic: (input_length * L + M - 1) / M
    return (input_length * L + M - 1) / M;
  }

  // Explicit template instantiations
  template class ResampleProcessorImpl<float>;
  template class ResampleProcessorImpl<double>;

}  // namespace OmniDSP::IntelIPP
