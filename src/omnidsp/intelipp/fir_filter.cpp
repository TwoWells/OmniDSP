/**
 * @file fir_filter.cpp (IntelIPP)
 * @brief Implements Intel IPP backend FIRFilterPlanImpl class using IPP.
 */

#include "fir_filter.hpp"  // Include the corresponding header file

#include <ippcore.h>  // For ippGetStatusString (included via ipp.h in fir_filter.hpp)
#include <ipps.h>  // IPP Signal Processing header (included via ipp.h in fir_filter.hpp)

#include <algorithm>    // For std::min (if needed)
#include <cmath>        // For std::abs
#include <cstring>      // For std::memcpy, std::memset (if needed)
#include <iostream>     // For error messages (used by utils)
#include <numeric>      // For std::fill (if needed)
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <string>       // For std::to_string
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "utils.hpp"  // Include the intelipp utility header (defines macros and type helpers)

// Include OmniDSP headers
#include <OmniDSP/core_types.hpp>  // Defines OmniException, Status etc.
// <OmniDSP/filter.hpp> is not directly needed for FIR implementation

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // IntelIPPFIRFilterPlanImpl Method Definitions
  //--------------------------------------------------------------------------

  template <typename T>
  IntelIPPFIRFilterPlanImpl<T>::IntelIPPFIRFilterPlanImpl(
      const std::vector<T>& coefficients)
      : coefficients_(coefficients),  // Store original coefficients
        num_taps_(coefficients.size()),
        order_(coefficients.empty() ? 0 : coefficients.size() - 1),
        p_spec_(
            nullptr),  // Initialize raw memory pointer (as declared in .hpp)
        p_buffer_(nullptr),  // Initialize buffer pointer
        spec_size_(0),
        buffer_size_(0),
        p_ipp_spec_(nullptr)  // Initialize typed pointer (as declared in .hpp)
  {
    if (coefficients_.empty()) {
      throw OmniException(
          "IPP FIR: Filter coefficients cannot be empty.",
          Status::InvalidArgument);
    }

    IppStatus status = ippStsErr;
    // Define IPP data type enum based on T
    constexpr IppDataType ipp_data_type = []
    {
      if constexpr (std::is_same_v<T, F32>)
        return ipp32f;
      else if constexpr (std::is_same_v<T, F64>)
        return ipp64f;
      else if constexpr (std::is_same_v<T, C32>)
        return ipp32fc;
      else if constexpr (std::is_same_v<T, C64>)
        return ipp64fc;
      else
        return ippUndef;
    }();

    int taps_len = static_cast<int>(num_taps_);
    if (taps_len <= 0) {
      throw OmniException(
          "IPP FIR: Invalid number of taps.", Status::InvalidArgument);
    }

    IppAlgType alg_type = ippAlgDirect;  // Use direct algorithm for FIR

    // Call IPP function directly with correct data type enum
    status = ::ippsFIRSRGetSize(
        taps_len, ipp_data_type, &spec_size_, &buffer_size_);
    OMNI_CHECK_IPP_STATUS_THROW(status, "IPP FIR: ippsFIRSRGetSize failed");

    // Allocate memory using IPP allocators
    p_spec_ = ippsMalloc_8u(spec_size_);
    p_buffer_ = (buffer_size_ > 0) ? ippsMalloc_8u(buffer_size_) : nullptr;
    if (!p_spec_ || (buffer_size_ > 0 && !p_buffer_)) {
      ippsFree(p_spec_);
      ippsFree(p_buffer_);
      p_spec_ = nullptr;  // Ensure members are null on failure
      p_buffer_ = nullptr;
      throw OmniException(
          "IPP FIR: Failed to allocate spec or buffer memory.",
          Status::AllocationError);
    }

    // Cast the raw spec memory to the correct IPP spec type pointer
    p_ipp_spec_ = reinterpret_cast<utils::GetIPPFIRSpec<T>*>(p_spec_);

    // Initialize the FIR state structure
    // Get a pointer to the coefficients data. IPP Init functions often take
    // const pointers for taps.
    const T* p_taps = coefficients_.data();

    // Cast the pointer to the type expected by the wrapper
    const utils::GetIPPType<T>* p_taps_ipp
        = reinterpret_cast<const utils::GetIPPType<T>*>(p_taps);

    // Use the internal templated wrapper for Init
    status = internal::ippsFIRSRInit<T>(
        p_taps_ipp,  // Pass the correctly casted pointer
        taps_len,
        alg_type,
        p_ipp_spec_);  // Pass the typed spec pointer

    if (status != ippStsNoErr) {
      ippsFree(p_spec_);
      ippsFree(p_buffer_);
      p_spec_ = nullptr;
      p_buffer_ = nullptr;
      p_ipp_spec_ = nullptr;
      OMNI_CHECK_IPP_STATUS_THROW(
          status, "IPP FIR: ippsFIRSRInit failed");  // Macro handles exception
                                                     // internally
    }
  }

  template <typename T>
  IntelIPPFIRFilterPlanImpl<T>::~IntelIPPFIRFilterPlanImpl()
  {
    ippsFree(p_spec_);    // Free the raw memory
    ippsFree(p_buffer_);  // Free the raw memory
  }

  template <typename T>
  Status IntelIPPFIRFilterPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!p_ipp_spec_) return Status::NotInitialized;
    if (input.size() != output.size()) return Status::SizeMismatch;
    if (input.empty()) return Status::Success;

    IppStatus status = ippStsErr;
    int len = static_cast<int>(input.size());
    // IPP expects non-const pointers for src/dst even if src isn't modified
    // Use reinterpret_cast for type conversion if necessary
    const utils::GetIPPType<T>* p_src_ipp
        = reinterpret_cast<const utils::GetIPPType<T>*>(input.data());
    utils::GetIPPType<T>* p_dst_ipp
        = reinterpret_cast<utils::GetIPPType<T>*>(output.data());

    // Execute FIR filter operation using internal wrapper
    status = internal::ippsFIRSR<T>(
        p_src_ipp,
        p_dst_ipp,
        len,
        p_ipp_spec_,  // Pass the typed spec pointer
        nullptr,      // pDlySrc - IPP manages delay internally for SR
        nullptr,      // pDlyDst - IPP manages delay internally for SR
        p_buffer_);   // Pass the work buffer

    OMNI_CHECK_IPP_STATUS_RETURN(status, "IPP FIR: ippsFIRSR execution failed");
    return Status::Success;
  }

  template <typename T>
  Status IntelIPPFIRFilterPlanImpl<T>::reset()
  {
    if (!p_ipp_spec_) return Status::NotInitialized;

    IppStatus status = ippStsErr;
    int taps_len = static_cast<int>(num_taps_);
    IppAlgType alg_type = ippAlgDirect;

    // Re-initialize the state using the same coefficients
    const T* p_taps = coefficients_.data();
    // Cast the pointer to the type expected by the wrapper
    const utils::GetIPPType<T>* p_taps_ipp
        = reinterpret_cast<const utils::GetIPPType<T>*>(p_taps);

    // Use internal wrapper for Init
    status = internal::ippsFIRSRInit<T>(
        p_taps_ipp,  // Pass the correctly casted pointer
        taps_len,
        alg_type,
        p_ipp_spec_);  // Pass the typed spec pointer

    OMNI_CHECK_IPP_STATUS_RETURN(
        status, "IPP FIR: Reset (ippsFIRSRInit) failed");
    return Status::Success;
  }

  template <typename T>
  size_t IntelIPPFIRFilterPlanImpl<T>::get_order() const /* noexcept */
  {
    return order_;
  }

  template <typename T>
  size_t IntelIPPFIRFilterPlanImpl<T>::get_num_taps() const /* noexcept */
  {
    return num_taps_;
  }

  // --- Explicit Template Instantiations for FIR ---
  template class IntelIPPFIRFilterPlanImpl<F32>;
  template class IntelIPPFIRFilterPlanImpl<F64>;
  template class IntelIPPFIRFilterPlanImpl<C32>;
  template class IntelIPPFIRFilterPlanImpl<C64>;

}  // namespace OmniDSP::IntelIPP
