/**
 * @file resample.cpp (IntelIPP)
 * @brief Implements the Intel IPP backend ResampleProcessorImpl class using
 * Intel IPP FIRMR.
 */

#include "resample.hpp"  // Include the corresponding header file

#include <ipps.h>  // For IPP functions

#include <OmniDSP/core_types.hpp>  // For Status, OmniException, F32, F64, FIRCoefs
#include <OmniDSP/design/fir_filter.hpp>  // For Design::FIRFilter (part of Design::Resample)
#include <OmniDSP/resample.hpp>  // For Design::Resample definition

#include "details.hpp"  // Include the intelipp utility header for IPP status checks and type helpers
// Window.hpp is included via filter.hpp or resample.hpp as needed

#include <cmath>
#include <iostream>  // For logging via spdlog or std::cerr
#include <memory>
#include <numeric>  // For std::gcd (not needed here, L/M from spec)
#include <span>
#include <stdexcept>
#include <string>
#include <utility>  // For std::move
#include <vector>

#include "interface/backend.hpp"  // For Abstract::Backend
#include "spdlog/spdlog.h"
// "utils/filter_design.hpp" and "utils/resample.hpp" are no longer directly
// needed here for design/factor calculation as Design::Resample provides this
// information.

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // ResampleProcessorImpl Method Definitions
  //--------------------------------------------------------------------------

  template <typename T>
  ResampleProcessorImpl<T>::ResampleProcessorImpl(
      const Abstract::Backend* owner, const Design::Resample& spec)
      : spec_(spec),  // Initialize the spec_ member
        input_rate_(spec.input_rate),
        output_rate_(spec.output_rate),
        quality_(spec.quality)
  // p_spec_mem_, p_buffer_, spec_mem_size_, buffer_size_, p_ipp_fir_spec_ are
  // initialized below
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner) {
      std::string msg
          = "IntelIPP::ResampleProcessorImpl requires a non-null owner backend "
            "pointer.";
      logger->error(msg);
      throw OmniException(msg, Status::InvalidArgument);
    }
    // Design::Resample's constructor and Utils::create_spec should have
    // validated its contents, including the embedded prototype_fir_spec.

    IppStatus ipp_status = ippStsNoErr;
    size_t L = spec_.up_factor_L;    // Use spec_ member
    size_t M = spec_.down_factor_M;  // Use spec_ member

    if (L == 0 || M == 0) {
      std::string msg = "IntelIPP::ResampleProcessorImpl: Resampling factors L="
                        + std::to_string(L) + " or M=" + std::to_string(M)
                        + " from spec are zero.";
      logger->error(msg);
      throw OmniException(msg, Status::InvalidArgument);
    }

    int upFactor = static_cast<int>(L);
    int downFactor = static_cast<int>(M);
    int upPhase = 0;    // Default phase for IPP FIRMR
    int downPhase = 0;  // Default phase for IPP FIRMR

    // Get prototype filter coefficients using the owner backend and the spec's
    // prototype_fir_spec
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
            "prototype "
            "filter design.";
      logger->critical(msg);
      throw OmniException(msg, Status::UnsupportedFeature);
    }

    if (!coeffs_expected) {
      std::string msg
          = "IntelIPP::ResampleProcessorImpl: Failed to design internal "
            "prototype "
            "filter using owner backend. Status: "
            + std::string(get_status_string(coeffs_expected.error()));
      logger->error(msg);
      throw OmniException(msg, coeffs_expected.error());
    }

    Coefs::FIRFilter<T> pTaps_unscaled = std::move(coeffs_expected.value());
    if (pTaps_unscaled.empty()) {
      std::string msg
          = "IntelIPP::ResampleProcessorImpl: Internal prototype filter design "
            "resulted in empty coefficients.";
      logger->error(msg);
      throw OmniException(msg, Status::Failure);
    }

    // Scale taps by L (interpolation factor) for IPP FIRMR
    Coefs::FIRFilter<T> pTaps = pTaps_unscaled;  // Make a mutable copy
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
      // This path should be unreachable due to static_assert in header, but
      // defensive programming.
      std::string msg
          = "IntelIPP::ResampleProcessorImpl: Unsupported data type T in "
            "constructor.";
      logger->critical(msg);  // Should not happen
      throw OmniException(msg, Status::UnsupportedFeature);
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
      p_buffer_ = nullptr;  // Ensure p_buffer_ is null if buffer_size_ is 0
    }

    if (!p_spec_mem_ || (buffer_size_ > 0 && !p_buffer_)) {
      ippsFree(p_spec_mem_);
      ippsFree(p_buffer_);
      p_spec_mem_ = nullptr;
      p_buffer_ = nullptr;
      std::string msg
          = "IntelIPP::ResampleProcessorImpl: Failed to allocate memory for "
            "IPP "
            "FIRMR spec or buffer.";
      logger->error(msg);
      throw OmniException(msg, Status::AllocationError);
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

    logger->debug(
        "IntelIPP::ResampleProcessorImpl created: IR={}, OR={}, Q={}, Taps={}, "
        "L={}, M={}",
        input_rate_,
        output_rate_,
        quality_,
        tapsLen,
        L,
        M);
  }

  template <typename T>
  ResampleProcessorImpl<T>::~ResampleProcessorImpl()
  {
    ippsFree(p_spec_mem_);
    ippsFree(p_buffer_);
    // auto logger = spdlog::get("OmniDSP");
    // if (logger) logger->trace("IntelIPP::ResampleProcessorImpl destructed.");
  }

  template <typename T>
  Status ResampleProcessorImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!p_ipp_fir_spec_) {
      return Status::NotInitialized;
    }

    size_t input_len = input.size();
    if (input_len == 0) {
      if (!output.empty()) {  // Only fill if output span is not empty
        std::fill(output.begin(), output.end(), T{0});
      }
      return Status::Success;
    }

    int numOutputSamples = static_cast<int>(output.size());
    if (numOutputSamples <= 0 && input_len > 0) {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) {
        logger = spdlog::default_logger();
      }
      logger->warn(
          "IntelIPP::ResampleProcessorImpl::execute: numOutputSamples ({}) is "
          "zero "
          "or negative but input exists ({} samples). Output span might be too "
          "small.",
          numOutputSamples,
          input_len);
      if (numOutputSamples == 0) return Status::Success;
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
    return Details::ipp_status_to_omnidsp_status(ipp_status);
  }

  template <typename T>
  Status ResampleProcessorImpl<T>::reset()
  {
    if (!p_ipp_fir_spec_) {
      return Status::NotInitialized;
    }
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (p_buffer_ && buffer_size_ > 0) {
      IppStatus ipp_status = ippsZero_8u(p_buffer_, buffer_size_);
      if (ipp_status != ippStsNoErr) {
        logger->warn(
            "IntelIPP::ResampleProcessorImpl::reset: Failed to zero work "
            "buffer, "
            "IPP status: {}",
            static_cast<int>(ipp_status));
      }
    }
    logger->debug(
        "IntelIPP::ResampleProcessorImpl::reset() called. Work buffer zeroed "
        "(if "
        "exists). "
        "Note: True internal filter state reset for IPP FIRMR typically "
        "requires re-initialization of the spec.");
    return Status::Success;
  }

  template <typename T>
  double ResampleProcessorImpl<T>::get_input_rate() const
  {
    return input_rate_;
  }

  template <typename T>
  double ResampleProcessorImpl<T>::get_output_rate() const
  {
    return output_rate_;
  }

  template <typename T>
  size_t ResampleProcessorImpl<T>::get_output_length(size_t input_length) const
  {
    if (input_rate_ <= 0.0 || output_rate_ <= 0.0 || input_length == 0) {
      return 0;
    }
    // Use L and M from the stored spec_ member
    size_t L = spec_.up_factor_L;
    size_t M = spec_.down_factor_M;

    if (M == 0) return 0;

    return (input_length * L + M - 1) / M;
  }

  template class ResampleProcessorImpl<float>;
  template class ResampleProcessorImpl<double>;

}  // namespace OmniDSP::IntelIPP
