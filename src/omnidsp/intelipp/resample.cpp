/**
 * @file resample.cpp (intelipp)
 * @brief Implements the Intel IPP backend ResamplePlanImpl class using Intel
 * IPP FIRMR.
 * @details Designs the FIR filter coefficients using a common utility function
 * before initializing the IPP state using ippsFIRMRInit_*.
 */

#include "resample.hpp"  // Include the corresponding header file

#include "backend.hpp"  // Include the backend header for IntelIPPBackend definition
#include "utils.hpp"  // Include the intelipp utility header

// Include IPP signal processing header
#include <ipps.h>
// Include IPP core header for ippGetStatusString
#include <ippcore.h>

#include <cmath>     // For std::ceil, std::max, std::abs, std::pow
#include <iostream>  // For error messages (used by utils)
#include <memory>    // For std::unique_ptr
#include <numeric>   // For std::gcd (needed by helper)
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For std::string in error message
#include <vector>

// Include necessary OmniDSP headers
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/filter.hpp>  // For FIRFilterSpec, FilterType, FIRCoefs
#include <OmniDSP/resample.hpp>
#include <OmniDSP/window.hpp>  // For WindowSpec

// Include the common utility headers
#include "../utils/filter_design.hpp"  // Renamed include
#include "../utils/resample.hpp"       // For calculate_resampling_factors

// Changed namespace
namespace OmniDSP::intelipp {

  // using directive for OmniException
  using ::OmniDSP::OmniException;

  //--------------------------------------------------------------------------
  // IntelIPPResamplePlanImpl Method Definitions
  //--------------------------------------------------------------------------

  // Constructor Implementation
  template <typename T>
  IntelIPPResamplePlanImpl<T>::IntelIPPResamplePlanImpl(
      const Abstract::AbstractBackend* owner, const ResampleSpec& spec)
      : input_rate_(spec.input_rate),
        output_rate_(spec.output_rate),
        quality_(spec.quality)
  {
    // Check owner pointer
    if (!owner) {
      throw OmniException(
          "IntelIPPResamplePlanImpl requires a non-null owner backend pointer.",
          Status::InvalidArgument);
    }
    if (!spec.validate()) {
      throw OmniException(
          "Invalid ResampleSpec provided.", Status::InvalidArgument);
    }

    // --- Filter Design ---
    IppStatus status = ippStsErr;
    size_t L = 0, M = 0;
    // Use the common factor calculation utility
    Status factor_status
        = Utils::calculate_resampling_factors(input_rate_, output_rate_, L, M);
    if (factor_status != Status::Success || L == 0 || M == 0) {
      throw OmniException(
          "Failed to calculate valid integer resampling factors L="
              + std::to_string(L) + ", M=" + std::to_string(M) + " for IPP.",
          factor_status);
    }
    int upFactor = static_cast<int>(L);
    int downFactor = static_cast<int>(M);
    int upPhase = 0;    // Start phase for upsampler
    int downPhase = 0;  // Start phase for downsampler

    // 1. Design the prototype FIR filter using the common utility function
    //    which internally calls the owner's design method.
    OmniExpected<FIRCoefs<T>> coeffs_expected
        = Utils::design_resampling_prototype_filter<T>(
            owner, L, M, quality_, spec.window);

    if (!coeffs_expected) {
      throw OmniException(
          "IPP Resample: Failed to design internal filter using utility.",
          coeffs_expected.error());
    }
    FIRCoefs<T> pTaps
        = std::move(coeffs_expected.value());  // Already scaled by L in utility
    if (pTaps.empty()) {
      throw OmniException(
          "IPP Resample: Internal filter design resulted in empty "
          "coefficients.",
          Status::Failure);
    }
    int tapsLen = static_cast<int>(pTaps.size());
    // --- End Filter Design ---

    // Determine IPP data type
    IppDataType ippDataType;
    if constexpr (std::is_same_v<T, float>) {
      ippDataType = ipp32f;
    }
    else if constexpr (std::is_same_v<T, double>) {
      ippDataType = ipp64f;
    }
    else { /* static assert in header prevents this */
    }

    // 2. Get sizes for spec and buffer using the *actual* tapsLen
    status = ippsFIRMRGetSize(
        tapsLen, upFactor, downFactor, ippDataType, &spec_size_, &buffer_size_);
    // Use the macro from intelipp/utils.hpp
    OMNI_CHECK_IPP_STATUS_THROW(
        status, "IPP Resample: ippsFIRMRGetSize failed");

    // 3. Allocate memory
    p_spec_ = ippsMalloc_8u(spec_size_);
    p_buffer_ = (buffer_size_ > 0) ? ippsMalloc_8u(buffer_size_) : nullptr;
    if (!p_spec_ || (buffer_size_ > 0 && !p_buffer_)) {
      ippsFree(p_spec_);
      ippsFree(p_buffer_);
      p_spec_ = nullptr;
      p_buffer_ = nullptr;
      throw OmniException(
          "IPP Resample: Failed to allocate spec or buffer memory.",
          Status::AllocationError);
    }

    // 4. Initialize the spec structure using the designed taps
    // Cast taps pointer to the IPP type
    const utils::GetIPPType<T>* pTaps_ipp
        = reinterpret_cast<const utils::GetIPPType<T>*>(pTaps.data());

    if constexpr (std::is_same_v<T, float>) {
      IppsFIRSpec_32f* typed_spec = reinterpret_cast<IppsFIRSpec_32f*>(p_spec_);
      status = ippsFIRMRInit_32f(
          pTaps_ipp,
          tapsLen,
          upFactor,
          upPhase,
          downFactor,
          downPhase,
          typed_spec);
      p_ipp_spec_typed_ = typed_spec;  // Store typed pointer
    }
    else {  // double
      IppsFIRSpec_64f* typed_spec = reinterpret_cast<IppsFIRSpec_64f*>(p_spec_);
      status = ippsFIRMRInit_64f(
          pTaps_ipp,
          tapsLen,
          upFactor,
          upPhase,
          downFactor,
          downPhase,
          typed_spec);
      p_ipp_spec_typed_ = typed_spec;  // Store typed pointer
    }

    if (status != ippStsNoErr) {
      ippsFree(p_spec_);
      ippsFree(p_buffer_);
      p_spec_ = nullptr;
      p_buffer_ = nullptr;
      p_ipp_spec_typed_ = nullptr;
      // Use the macro from intelipp/utils.hpp
      OMNI_CHECK_IPP_STATUS_THROW(status, "IPP Resample: ippsFIRMRInit failed");
    }
  }

  // Destructor Implementation
  template <typename T>
  IntelIPPResamplePlanImpl<T>::~IntelIPPResamplePlanImpl()
  {
    ippsFree(p_spec_);    // Safe to call on nullptr
    ippsFree(p_buffer_);  // Safe to call on nullptr
  }

  // Execute Method Implementation
  template <typename T>
  Status IntelIPPResamplePlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!p_ipp_spec_typed_) {           // Check typed pointer
      return Status::InvalidOperation;  // Not initialized
    }

    size_t input_len = input.size();
    size_t output_len_required
        = get_output_length(input_len);  // Use estimation

    if (input_len == 0) {
      Status reset_status = reset();  // Attempt to reset state
      if (reset_status != Status::Success) {
        std::cerr << "Warning: Reset failed within execute when input_len is 0."
                  << std::endl;
        // Return the reset error status
        return reset_status;
      }
      // Zero the output buffer if input is empty
      std::fill(output.begin(), output.end(), T{0});
      return Status::Success;
    }

    IppStatus status = ippStsErr;
    // IPP FIRMR takes the number of *output* samples to generate.
    int numOutputSamples = static_cast<int>(output.size());
    if (numOutputSamples <= 0) {
      return Status::Success;  // Nothing to generate
    }

    // Cast pointers to IPP types
    const utils::GetIPPType<T>* p_in_ipp
        = reinterpret_cast<const utils::GetIPPType<T>*>(input.data());
    utils::GetIPPType<T>* p_out_ipp
        = reinterpret_cast<utils::GetIPPType<T>*>(output.data());

    // Call the correct IPP function based on type T
    if constexpr (std::is_same_v<T, float>) {
      status = ippsFIRMR_32f(
          p_in_ipp,          // pSrc
          p_out_ipp,         // pDst
          numOutputSamples,  // numIters (number of output samples to generate)
          static_cast<IppsFIRSpec_32f*>(
              p_ipp_spec_typed_),  // pSpec (use typed pointer)
          nullptr,                 // pDlySrc (not managing external delay)
          nullptr,                 // pDlyDst (not managing external delay)
          p_buffer_                // pBuf
      );
    }
    else {  // double
      status = ippsFIRMR_64f(
          p_in_ipp,          // pSrc
          p_out_ipp,         // pDst
          numOutputSamples,  // numIters
          static_cast<IppsFIRSpec_64f*>(
              p_ipp_spec_typed_),  // pSpec (use typed pointer)
          nullptr,                 // pDlySrc
          nullptr,                 // pDlyDst
          p_buffer_                // pBuf
      );
    }

    // Use the function from intelipp/utils.hpp
    return utils::ipp_status_to_omnidsp_status(status);
  }

  // Reset Method Implementation
  template <typename T>
  Status IntelIPPResamplePlanImpl<T>::reset()
  {
    if (!p_ipp_spec_typed_) {  // Check typed pointer
      return Status::InvalidOperation;
    }
    // IPP FIRMR state is contained within the spec structure.
    // There isn't a specific "reset" function. Zeroing the buffer
    // might not be sufficient. Re-initializing is the safest bet,
    // but requires storing all init parameters (taps, factors, phases).
    // For now, just zero the work buffer as a minimal attempt.
    if (p_buffer_ && buffer_size_ > 0) {
      IppStatus status = ippsZero_8u(p_buffer_, buffer_size_);
      return utils::ipp_status_to_omnidsp_status(status);
    }
    return Status::Success;  // No buffer to zero
  }

  // Getters Implementation
  template <typename T>
  double IntelIPPResamplePlanImpl<T>::get_input_rate() const
  {
    return input_rate_;
  }

  template <typename T>
  double IntelIPPResamplePlanImpl<T>::get_output_rate() const
  {
    return output_rate_;
  }

  template <typename T>
  size_t IntelIPPResamplePlanImpl<T>::get_output_length(
      size_t input_length) const
  {
    if (input_rate_ <= 0.0 || output_rate_ <= 0.0 || input_length == 0)
      return 0;
    double ratio = output_rate_ / input_rate_;
    // Using ceil(input_length * ratio) + some margin is a common estimation.
    size_t estimated_len = static_cast<size_t>(
        std::ceil(static_cast<double>(input_length) * ratio));
    // Add a margin related to potential filter delay/transients.
    // This is hard to get exactly right without knowing the filter length used
    // by Init. Adding 2 is a minimal margin, might need adjustment based on
    // testing.
    estimated_len += 2;
    return estimated_len;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class IntelIPPResamplePlanImpl<float>;
  template class IntelIPPResamplePlanImpl<double>;

}  // namespace OmniDSP::intelipp
