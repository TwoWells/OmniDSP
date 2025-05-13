/**
 * @file iir_filter.cpp (IntelIPP)
 * @brief Implements Intel IPP backend IIRFilterPlanImpl class using IPP.
 */

#include "iir_filter.hpp"  // Include the corresponding header file

#include <ippcore.h>  // For ippGetStatusString (included via ipp.h in iir_filter.hpp)
#include <ipps.h>  // IPP Signal Processing header (included via ipp.h in iir_filter.hpp)

#include <algorithm>    // For std::min (if needed)
#include <cmath>        // For std::abs
#include <cstring>      // For std::memcpy, std::memset (if needed)
#include <iostream>     // For error messages
#include <numeric>      // For std::fill (if needed)
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <string>       // For std::to_string
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "details.hpp"

// Include OmniDSP headers
#include <OmniDSP/coefs/iir_filter.hpp>  // Defines IIRFilterCoef
#include <OmniDSP/core_types.hpp>        // Defines OmniException, Status etc.

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // IIRFilterPlanImpl Method Definitions
  //--------------------------------------------------------------------------

  template <typename T>
  IIRFilterProcessorImpl<T>::IIRFilterProcessorImpl(
      const std::vector<IIRFilterCoef>& sos_coefficients)
      : num_sections_(sos_coefficients.size()),
        order_(sos_coefficients.empty() ? 0 : sos_coefficients.size() * 2),
        p_state_(nullptr),  // Init pointer
        state_mem_(),       // Init vector
        state_size_bytes_(0),
        taps_interleaved_()  // Init vector
  {
    if (num_sections_ == 0) {
      throw OmniException(
          "IPP IIR: SOS coefficients cannot be empty.",
          Status::InvalidArgument);
    }

    IppStatus status = ippStsErr;
    int num_biquads = static_cast<int>(num_sections_);

    // 1. Prepare taps in the correct type T and IPP layout (b0,b1,b2,a0,a1,a2)
    taps_interleaved_.resize(num_sections_ * 6);  // Use member variable
    for (size_t i = 0; i < num_sections_; ++i) {
      const auto& sos = sos_coefficients[i];
      if (std::abs(sos.a0 - 1.0) > 1e-9) {
        // IPP BiQuad functions assume a0=1. Pre-normalize if needed, or throw.
        throw OmniException(
            "IPP IIR requires SOS coefficients with a0=1.0 (normalize first). "
            "Section "
                + std::to_string(i),
            Status::InvalidArgument);
      }
      size_t base_idx = i * 6;
      taps_interleaved_[base_idx + 0] = static_cast<T>(sos.b0);
      taps_interleaved_[base_idx + 1] = static_cast<T>(sos.b1);
      taps_interleaved_[base_idx + 2] = static_cast<T>(sos.b2);
      taps_interleaved_[base_idx + 3]
          = static_cast<T>(1.0);  // a0 (must be 1.0)
      taps_interleaved_[base_idx + 4] = static_cast<T>(sos.a1);
      taps_interleaved_[base_idx + 5] = static_cast<T>(sos.a2);
    }

    // 2. Get state buffer size using internal wrapper
    status = internal::ippsIIRGetStateSize_BiQuad<T>(
        num_biquads, &state_size_bytes_);
    OMNI_CHECK_IPP_STATUS_THROW(
        status, "IPP IIR: ippsIIRGetStateSize_BiQuad failed");

    // 3. Allocate state memory
    try {
      state_mem_.resize(state_size_bytes_);
    }
    catch (const std::bad_alloc& e) {
      throw OmniException(
          "IPP IIR: Failed to allocate state memory.", Status::AllocationError);
    }
    // Get a typed pointer
    if (state_mem_.empty()
        && state_size_bytes_
               > 0) {  // Check if size > 0 but alloc failed/resulted in empty
      throw OmniException(
          "IPP IIR: State size is non-zero but memory allocation failed.",
          Status::AllocationError);
    }
    if (state_size_bytes_ == 0 && !state_mem_.empty()) {  // Check consistency
      state_mem_.clear();  // Ensure vector is empty if size is 0
    }
    // Only assign p_state_ if memory was actually allocated
    p_state_ = state_mem_.empty()
                   ? nullptr
                   : reinterpret_cast<Details::GetIPPIIRState<T>*>(
                         state_mem_.data());

    // 4. Initialize state using the dispatch helper
    // Pass correctly typed taps pointer
    const T* p_taps = taps_interleaved_.data();
    // Cast to the IPP type pointer
    const Details::GetIPPType<T>* p_taps_ipp
        = reinterpret_cast<const Details::GetIPPType<T>*>(p_taps);

    // Pass 5 arguments to the corrected wrapper
    status = internal::ippsIIRInit_BiQuad<T>(
        &p_state_,   // Address of the typed state pointer
        p_taps_ipp,  // Pass correctly typed & casted taps
        num_biquads,
        nullptr,  // Initialize delay line to zero
        nullptr   // Pass nullptr for the 5th pBuf argument (not needed for
                  // BiQuad Init)
    );

    // Check status after Init
    if (status != ippStsNoErr) {
      p_state_ = nullptr;  // Nullify pointer on failure
      state_mem_.clear();  // Clear memory
      OMNI_CHECK_IPP_STATUS_THROW(status, "IPP IIR: ippsIIRInit_BiQuad failed");
    }
    // Defensive check: Verify the pointer returned by IPP matches our buffer
    // start
    if (status == ippStsNoErr && !state_mem_.empty()
        && reinterpret_cast<Ipp8u*>(p_state_) != state_mem_.data()) {
      p_state_ = nullptr;
      state_mem_.clear();
      throw OmniException(
          "IPP IIR: Init status OK but state pointer mismatch.",
          Status::BackendError);
    }
    // If state_size_bytes_ was 0, p_state_ should be nullptr after init
    if (status == ippStsNoErr && state_size_bytes_ == 0
        && p_state_ != nullptr) {
      throw OmniException(
          "IPP IIR: State size is zero but state pointer is not null after "
          "init.",
          Status::BackendError);
    }
  }

  template <typename T>
  IIRFilterProcessorImpl<T>::~IIRFilterProcessorImpl()
  {
    // No explicit free needed, state_mem_ vector handles memory
    p_state_ = nullptr;
  }

  template <typename T>
  Status IIRFilterProcessorImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!p_state_ && state_size_bytes_ > 0)
      return Status::NotInitialized;  // Check if state should exist but doesn't
    if (input.size() != output.size()) return Status::SizeMismatch;
    if (input.empty()) return Status::Success;

    IppStatus status = ippStsErr;
    int len = static_cast<int>(input.size());
    // Cast pointers to IPP types
    const Details::GetIPPType<T>* p_src_ipp
        = reinterpret_cast<const Details::GetIPPType<T>*>(input.data());
    Details::GetIPPType<T>* p_dst_ipp
        = reinterpret_cast<Details::GetIPPType<T>*>(output.data());

    // Execute the IIR filter operation using internal wrapper
    status = internal::ippsIIR<T>(
        p_src_ipp,
        p_dst_ipp,
        len,
        p_state_  // Pass the typed state pointer
    );

    OMNI_CHECK_IPP_STATUS_RETURN(status, "IPP IIR: ippsIIR execution failed");
    return Status::Success;
  }

  template <typename T>
  Status IIRFilterProcessorImpl<T>::reset()
  {
    if (!p_state_ && state_size_bytes_ > 0) return Status::NotInitialized;
    if (!p_state_)
      return Status::Success;  // Nothing to reset if state doesn't exist

    IppStatus status = ippStsErr;
    int num_biquads = static_cast<int>(num_sections_);
    // Create a zero delay line buffer of the correct size (2 * num_biquads)
    std::vector<T> zero_delay_line(num_biquads * 2, T{0});
    // Cast pointer to IPP type
    const Details::GetIPPType<T>* p_dly_ipp
        = reinterpret_cast<const Details::GetIPPType<T>*>(
            zero_delay_line.data());

    // Set the delay line to zeros using internal wrapper
    status = internal::ippsIIRSetDlyLine<T>(
        p_state_,  // Pass typed state pointer
        p_dly_ipp  // Pass typed delay line pointer
    );

    OMNI_CHECK_IPP_STATUS_RETURN(
        status, "IPP IIR: Reset (ippsIIRSetDlyLine) failed");
    return Status::Success;
  }

  template <typename T>
  size_t IIRFilterProcessorImpl<T>::get_order() const /* noexcept */
  {
    return order_;
  }

  template <typename T>
  size_t IIRFilterProcessorImpl<T>::get_num_sections() const /* noexcept */
  {
    return num_sections_;
  }

  // --- Explicit Template Instantiations for IIR ---
  template class IIRFilterProcessorImpl<F32>;
  template class IIRFilterProcessorImpl<F64>;

}  // namespace OmniDSP::IntelIPP
