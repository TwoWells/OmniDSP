/**
 * @file filter.cpp (onemkl)
 * @brief Implements oneMKL backend FIRFilterPlanImpl and IIRFilterPlanImpl
 * classes using IPP.
 */

#include "filter.hpp"  // Include the corresponding header file

#include <ipps.h>       // IPP Signal Processing header
#include <mkl_types.h>  // For MKL types if needed indirectly

#include <algorithm>  // For std::min
#include <cstring>    // For std::memcpy, std::memset
#include <iostream>   // For error messages
#include <numeric>    // For std::fill
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

// Include OmniDSP headers
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/filter.hpp>

// Include the onemkl utility header
#include "utils.hpp"

namespace OmniDSP::backend {

  //--------------------------------------------------------------------------
  // OneMKLFIRFilterPlanImpl Method Definitions
  //--------------------------------------------------------------------------
  // ... (FIR Implementation remains the same) ...
  template <typename T>
  OneMKLFIRFilterPlanImpl<T>::OneMKLFIRFilterPlanImpl(
      const std::vector<T>& coefficients)
      : coefficients_(coefficients),
        num_taps_(coefficients.size()),
        order_(coefficients.empty() ? 0 : coefficients.size() - 1)
  {
    if (coefficients_.empty()) {
      throw std::invalid_argument("FIR filter coefficients cannot be empty.");
    }
    IppStatus status = ippStsErr;
    constexpr IppDataType data_type = IppwDataType<T>;
    if (static_cast<int>(data_type) == -1) {
      throw std::invalid_argument("Unsupported data type for IPP FIR filter.");
    }
    int taps_len = static_cast<int>(num_taps_);
    IppAlgType alg_type = ippAlgDirect;
    status = ippsFIRSRGetSize(taps_len, data_type, &spec_size_, &buffer_size_);
    if (status != ippStsNoErr) {
      throw std::runtime_error(
          "IPP Error (ippsFIRSRGetSize): "
          + std::string(ippGetStatusString(status)));
    }
    p_spec_ = ippsMalloc_8u(spec_size_);
    p_buffer_ = (buffer_size_ > 0) ? ippsMalloc_8u(buffer_size_) : nullptr;
    if (!p_spec_ || (buffer_size_ > 0 && !p_buffer_)) {
      ippsFree(p_spec_);
      ippsFree(p_buffer_);
      p_spec_ = nullptr;
      p_buffer_ = nullptr;
      throw std::bad_alloc();
    }
    T* p_taps_nonconst = const_cast<T*>(coefficients_.data());
    status = ipp_dispatch::ippsFIRSRInit<T>(
        p_taps_nonconst,
        taps_len,
        alg_type,
        reinterpret_cast<IppwFIRSpec<T>*>(p_spec_));
    if (status != ippStsNoErr) {
      ippsFree(p_spec_);
      ippsFree(p_buffer_);
      p_spec_ = nullptr;
      p_buffer_ = nullptr;
      throw std::runtime_error(
          "IPP Error (ippsFIRSRInit): "
          + std::string(ippGetStatusString(status)));
    }
  }
  template <typename T>
  OneMKLFIRFilterPlanImpl<T>::~OneMKLFIRFilterPlanImpl()
  {
    ippsFree(p_spec_);
    ippsFree(p_buffer_);
  }
  template <typename T>
  Status OneMKLFIRFilterPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!p_spec_) return Status::InvalidOperation;
    if (input.size() != output.size()) return Status::SizeMismatch;
    if (input.empty()) return Status::Success;
    IppStatus status = ippStsErr;
    int len = static_cast<int>(input.size());
    T* p_src = const_cast<T*>(input.data());
    T* p_dst = output.data();
    status = ipp_dispatch::ippsFIRSR<T>(
        p_src,
        p_dst,
        len,
        reinterpret_cast<IppwFIRSpec<T>*>(p_spec_),
        nullptr,
        nullptr,
        p_buffer_);
    return ipp_status_to_omnidsp_status(status);
  }
  template <typename T>
  Status OneMKLFIRFilterPlanImpl<T>::reset()
  {
    if (!p_spec_) return Status::InvalidOperation;
    IppStatus status = ippStsErr;
    int taps_len = static_cast<int>(num_taps_);
    IppAlgType alg_type = ippAlgDirect;
    T* p_taps_nonconst = const_cast<T*>(coefficients_.data());
    status = ipp_dispatch::ippsFIRSRInit<T>(
        p_taps_nonconst,
        taps_len,
        alg_type,
        reinterpret_cast<IppwFIRSpec<T>*>(p_spec_));
    return ipp_status_to_omnidsp_status(status);
  }
  template <typename T>
  size_t OneMKLFIRFilterPlanImpl<T>::get_order() const
  {
    return order_;
  }
  template <typename T>
  size_t OneMKLFIRFilterPlanImpl<T>::get_num_taps() const
  {
    return num_taps_;
  }

  //--------------------------------------------------------------------------
  // OneMKLIIRFilterPlanImpl Method Definitions
  //--------------------------------------------------------------------------

  // Constructor
  template <typename T>
  OneMKLIIRFilterPlanImpl<T>::OneMKLIIRFilterPlanImpl(
      const std::vector<IIRFilterCoef>& sos_coefficients)
      : num_sections_(sos_coefficients.size()),
        order_(sos_coefficients.empty() ? 0 : sos_coefficients.size() * 2)
  {
    if (num_sections_ == 0) {
      throw std::invalid_argument("IIR SOS coefficients cannot be empty.");
    }

    IppStatus status = ippStsErr;
    // constexpr IppDataType data_type = IppwDataType<T>; // Not needed directly
    // for GetStateSize dispatcher if (static_cast<int>(data_type) == -1) {
    //   throw std::invalid_argument("Unsupported data type for IPP IIR
    //   filter.");
    // }
    int num_biquads = static_cast<int>(num_sections_);

    // 1. Prepare taps in IPP format
    taps_interleaved_.resize(num_sections_ * 6);
    for (size_t i = 0; i < num_sections_; ++i) {
      const auto& sos = sos_coefficients[i];
      if (std::abs(sos.a0 - 1.0) > 1e-9) {
        throw std::invalid_argument(
            "IPP IIR requires SOS coefficients with a0=1.0 (normalize first).");
      }
      size_t base_idx = i * 6;
      taps_interleaved_[base_idx + 0] = static_cast<T>(sos.b0);
      taps_interleaved_[base_idx + 1] = static_cast<T>(sos.b1);
      taps_interleaved_[base_idx + 2] = static_cast<T>(sos.b2);
      taps_interleaved_[base_idx + 3] = static_cast<T>(1.0);  // a0 is 1.0
      taps_interleaved_[base_idx + 4] = static_cast<T>(sos.a1);
      taps_interleaved_[base_idx + 5] = static_cast<T>(sos.a2);
    }

    // 2. Get state buffer size
    // *** CORRECTED: Use the new ipp_dispatch template ***
    status = ipp_dispatch::ippsIIRGetStateSize_BiQuad<T>(
        num_biquads, &state_size_bytes_);
    // *** END CORRECTION ***

    if (status != ippStsNoErr) {
      throw std::runtime_error(
          "IPP Error (ippsIIRGetStateSize_BiQuad_...): "  // Updated error
                                                          // message slightly
          + std::string(ippGetStatusString(status)));
    }

    // 3. Allocate state memory as raw bytes
    Ipp8u* p_state_raw = ippsMalloc_8u(state_size_bytes_);
    if (!p_state_raw) {
      throw std::bad_alloc();
    }
    p_state_ = reinterpret_cast<IppwIIRState<T>*>(p_state_raw);

    // 4. Initialize state using the dispatch helper
    T* p_taps_nonconst = taps_interleaved_.data();
    status = ipp_dispatch::ippsIIRInit_BiQuad<T>(
        &p_state_,  // Pass address of the pointer
        p_taps_nonconst,
        num_biquads,
        nullptr,  // Initialize delay line to zero
        nullptr   // Pass nullptr for pBuf
    );

    if (status != ippStsNoErr) {
      ippsFree(p_state_);  // Free the raw buffer
      p_state_ = nullptr;
      throw std::runtime_error(
          "IPP Error (ippsIIRInit_BiQuad): "
          + std::string(ippGetStatusString(status)));
    }
  }

  // Destructor
  template <typename T>
  OneMKLIIRFilterPlanImpl<T>::~OneMKLIIRFilterPlanImpl()
  {
    ippsFree(p_state_);  // Free the state buffer using the stored pointer
  }

  // Execute
  template <typename T>
  Status OneMKLIIRFilterPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!p_state_) return Status::InvalidOperation;
    if (input.size() != output.size()) return Status::SizeMismatch;
    if (input.empty()) return Status::Success;

    IppStatus status = ippStsErr;
    int len = static_cast<int>(input.size());
    T* p_src = const_cast<T*>(input.data());
    T* p_dst = output.data();

    status = ipp_dispatch::ippsIIR<T>(
        p_src,
        p_dst,
        len,
        p_state_  // Pass the pointer directly
    );

    return ipp_status_to_omnidsp_status(status);
  }

  // Reset
  template <typename T>
  Status OneMKLIIRFilterPlanImpl<T>::reset()
  {
    if (!p_state_) return Status::InvalidOperation;

    IppStatus status = ippStsErr;
    int num_biquads = static_cast<int>(num_sections_);
    // Delay line size is 2 * numBiquads
    std::vector<T> zero_delay_line(num_biquads * 2, T{0});

    status = ipp_dispatch::ippsIIRSetDlyLine<T>(
        p_state_,  // Pass the pointer directly
        zero_delay_line.data());

    return ipp_status_to_omnidsp_status(status);
  }

  // Getters
  template <typename T>
  size_t OneMKLIIRFilterPlanImpl<T>::get_order() const
  {
    return order_;
  }
  template <typename T>
  size_t OneMKLIIRFilterPlanImpl<T>::get_num_sections() const
  {
    return num_sections_;
  }

  // --- Explicit Template Instantiations ---
  template class OneMKLFIRFilterPlanImpl<F32>;
  template class OneMKLFIRFilterPlanImpl<F64>;
  template class OneMKLFIRFilterPlanImpl<C32>;
  template class OneMKLFIRFilterPlanImpl<C64>;

  template class OneMKLIIRFilterPlanImpl<F32>;
  template class OneMKLIIRFilterPlanImpl<F64>;

}  // namespace OmniDSP::backend
