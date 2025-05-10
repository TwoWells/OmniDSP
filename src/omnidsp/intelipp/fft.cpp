/**
 * @file fft.cpp (IntelIPP)
 * @brief Implements Intel IPP backend FFTPlanImpl and RFFTPlanImpl classes.
 */
#include "fft.hpp"  // Corresponding header

#include <ipps.h>  // Required for IPP functions and types (included via ipp.h in fft.hpp)

#include <OmniDSP/core_types.hpp>  // Include for Status, OmniException, etc.
#include <cmath>                   // For std::log2
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <vector>     // For buffer management

#include "utils.hpp"  // For ipp_status_to_omnidsp_status

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // FFTPlanImpl Method Definitions (Complex FFT)
  //--------------------------------------------------------------------------

  template <typename T_Complex>
  FFTPlanImpl<T_Complex>::FFTPlanImpl(size_t length)
      // Initialize declared members correctly
      : length_(length),
        p_spec_(nullptr),  // Initialize pointer members to nullptr
        spec_mem_(),       // Default-initialize vectors
        init_buf_(),
        work_buf_()
  {
    // --- Input Validation ---
    if (length == 0) {
      throw std::invalid_argument("FFT length cannot be zero.");
    }
    try {
      order_ = internal::calculate_fft_order(length);
    }
    catch (const std::invalid_argument& e) {
      // Re-throw or wrap if necessary, maybe as OmniException
      // Wrapping might be better to keep exception types consistent
      throw OmniException(e.what(), Status::InvalidArgument);
    }

    // --- IPP Initialization ---
    IppStatus ipp_status;
    int sizeSpec = 0, sizeInit = 0, sizeBuffer = 0;
    // Use IPP_FFT_DIV_INV_BY_N for standard FFT scaling (1/N on inverse)
    flag_ = IPP_FFT_DIV_INV_BY_N;

    // Use the 6-argument internal wrapper from fft.hpp
    ipp_status = internal::ippsFFTGetSize_C<T_Complex>(
        order_,
        flag_,
        ippAlgHintNone,
        &sizeSpec,
        &sizeInit,   // 5th arg (pSpecBufferSize)
        &sizeBuffer  // 6th arg (pBufferSize)
    );
    if (ipp_status != ippStsNoErr) {
      throw OmniException(
          "IPP Error: ippsFFTGetSize_C Failed",
          IntelIPP::Details::ipp_status_to_omnidsp_status(ipp_status));
    }

    // Allocate memory using std::vector<Ipp8u>
    try {
      spec_mem_.resize(sizeSpec);
      // Only resize init_buf_ if sizeInit > 0
      if (sizeInit > 0) {
        init_buf_.resize(sizeInit);
      }
      // Allocate work buffer here if size is fixed and known
      if (sizeBuffer > 0) {
        work_buf_.resize(sizeBuffer);
      }
    }
    catch (const std::bad_alloc&) {
      throw OmniException(
          "IPP Error: Failed to allocate memory for FFT buffers",
          Status::AllocationError);
    }

    // Get the raw pointer to the spec memory AFTER allocation/resize
    // Ensure spec_mem_ is not empty before getting data() pointer
    if (spec_mem_.empty()) {
      throw OmniException(
          "IPP Error: FFT spec size is zero", Status::BackendError);
    }
    p_spec_ = reinterpret_cast<IPP_Spec_Type*>(spec_mem_.data());

    // Initialize FFT spec using the allocated memory
    // Note: IPP Init functions write the initialized spec pointer back into
    // ppFFTSpec
    IPP_Spec_Type* p_spec_temp
        = p_spec_;  // Use a temporary pointer for the init call
    ipp_status = internal::ippsFFTInit_C<T_Complex>(
        &p_spec_temp,  // Address of the pointer
        order_,
        flag_,
        ippAlgHintNone,
        spec_mem_.data(),  // Pass the spec buffer itself
        // Pass init buffer data pointer (or nullptr if size was 0)
        init_buf_.empty() ? nullptr : init_buf_.data());
    // init_buf_ is no longer needed after Init returns
    init_buf_.clear();
    init_buf_.shrink_to_fit();

    if (ipp_status != ippStsNoErr) {
      // Clear potentially partially initialized memory
      spec_mem_.clear();
      spec_mem_.shrink_to_fit();
      work_buf_.clear();
      work_buf_.shrink_to_fit();
      p_spec_ = nullptr;
      throw OmniException(
          "IPP Error: ippsFFTInit_C Failed",
          IntelIPP::Details::ipp_status_to_omnidsp_status(ipp_status));
    }
    // Verify that the pointer returned by Init matches our buffer start
    // (optional sanity check)
    if (p_spec_temp != p_spec_) {
      // This indicates an unexpected issue with IPP Init
      spec_mem_.clear();
      spec_mem_.shrink_to_fit();
      work_buf_.clear();
      work_buf_.shrink_to_fit();
      p_spec_ = nullptr;
      throw OmniException(
          "IPP Error: FFTInit returned inconsistent spec pointer",
          Status::BackendError);
    }
    // p_spec_ should now point to the initialized spec within spec_mem_

    // Optional: Re-check work buffer size *after* init if needed by specific
    // IPP versions/funcs
  }

  // Destructor implementation remains default or empty if vectors handle memory

  template <typename T_Complex>
  Status FFTPlanImpl<T_Complex>::fft(
      std::span<const T_Complex> input, std::span<T_Complex> output) const
  {
    if (p_spec_ == nullptr) return Status::NotInitialized;
    if (input.size() != length_ || output.size() != length_)
      return Status::SizeMismatch;
    if (length_ == 0) return Status::Success;  // Nothing to do

    IppStatus ipp_status = internal::ippsFFTFwd_CToC<T_Complex>(
        reinterpret_cast<const IPP_C_Type*>(input.data()),
        reinterpret_cast<IPP_C_Type*>(output.data()),
        p_spec_,
        // Pass mutable buffer pointer, handle case where size is 0
        work_buf_.empty() ? nullptr : const_cast<Ipp8u*>(work_buf_.data()));
    return IntelIPP::Details::ipp_status_to_omnidsp_status(ipp_status);
  }

  template <typename T_Complex>
  Status FFTPlanImpl<T_Complex>::ifft(
      std::span<const T_Complex> input, std::span<T_Complex> output) const
  {
    if (p_spec_ == nullptr) return Status::NotInitialized;
    if (input.size() != length_ || output.size() != length_)
      return Status::SizeMismatch;
    if (length_ == 0) return Status::Success;  // Nothing to do

    IppStatus ipp_status = internal::ippsFFTInv_CToC<T_Complex>(
        reinterpret_cast<const IPP_C_Type*>(input.data()),
        reinterpret_cast<IPP_C_Type*>(output.data()),
        p_spec_,
        // Pass mutable buffer pointer, handle case where size is 0
        work_buf_.empty() ? nullptr : const_cast<Ipp8u*>(work_buf_.data()));
    return IntelIPP::Details::ipp_status_to_omnidsp_status(ipp_status);
  }

  // get_length() is already defined in the header

  //--------------------------------------------------------------------------
  // RFFTPlanImpl Method Definitions (Real FFT)
  //--------------------------------------------------------------------------
  template <typename T_Real>
  RFFTPlanImpl<T_Real>::RFFTPlanImpl(size_t length)
      // Initialize declared members correctly
      : length_(length),
        p_spec_(nullptr),
        spec_mem_(),
        init_buf_(),
        work_buf_(),
        temp_packed_buf_()  // Initialize the temp buffer vector
  {
    // --- Input Validation ---
    if (length == 0) {
      throw std::invalid_argument("RFFT length cannot be zero.");
    }
    // IPP RFFT often requires length >= 2
    if (length < 2) {
      throw std::invalid_argument("Intel IPP RFFT requires length >= 2.");
    }
    try {
      order_ = internal::calculate_fft_order(length);
    }
    catch (const std::invalid_argument& e) {
      throw OmniException(e.what(), Status::InvalidArgument);  // Wrap exception
    }

    // --- IPP Initialization ---
    IppStatus ipp_status;
    int sizeSpec = 0, sizeInit = 0, sizeBuffer = 0;
    flag_ = IPP_FFT_DIV_INV_BY_N;  // Use standard scaling

    // Use the 6-argument internal wrapper from fft.hpp
    ipp_status = internal::ippsFFTGetSize_R<T_Real>(
        order_, flag_, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer);
    if (ipp_status != ippStsNoErr) {
      throw OmniException(
          "IPP Error: ippsFFTGetSize_R Failed",
          IntelIPP::Details::ipp_status_to_omnidsp_status(ipp_status));
    }

    // Allocate memory
    try {
      spec_mem_.resize(sizeSpec);
      if (sizeInit > 0) {
        init_buf_.resize(sizeInit);
      }
      if (sizeBuffer > 0) {
        work_buf_.resize(sizeBuffer);
      }
      // Allocate temporary buffer for packed format (size = length real
      // elements)
      temp_packed_buf_.resize(length);
    }
    catch (const std::bad_alloc&) {
      throw OmniException(
          "IPP Error: Failed to allocate memory for RFFT buffers",
          Status::AllocationError);
    }

    // Get raw pointer
    if (spec_mem_.empty()) {
      throw OmniException(
          "IPP Error: RFFT spec size is zero", Status::BackendError);
    }
    p_spec_ = reinterpret_cast<IPP_Spec_Type*>(spec_mem_.data());

    // Initialize RFFT spec
    IPP_Spec_Type* p_spec_temp = p_spec_;
    ipp_status = internal::ippsFFTInit_R<T_Real>(
        &p_spec_temp,
        order_,
        flag_,
        ippAlgHintNone,
        spec_mem_.data(),
        init_buf_.empty() ? nullptr : init_buf_.data());
    init_buf_.clear();
    init_buf_.shrink_to_fit();

    if (ipp_status != ippStsNoErr) {
      spec_mem_.clear();
      spec_mem_.shrink_to_fit();
      work_buf_.clear();
      work_buf_.shrink_to_fit();
      temp_packed_buf_.clear();
      temp_packed_buf_.shrink_to_fit();
      p_spec_ = nullptr;
      throw OmniException(
          "IPP Error: ippsFFTInit_R Failed",
          IntelIPP::Details::ipp_status_to_omnidsp_status(ipp_status));
    }
    if (p_spec_temp != p_spec_) {
      spec_mem_.clear();
      spec_mem_.shrink_to_fit();
      work_buf_.clear();
      work_buf_.shrink_to_fit();
      temp_packed_buf_.clear();
      temp_packed_buf_.shrink_to_fit();
      p_spec_ = nullptr;
      throw OmniException(
          "IPP Error: RFFTInit returned inconsistent spec pointer",
          Status::BackendError);
    }
    // p_spec_ should now point to the initialized spec within spec_mem_
  }

  // Destructor implementation remains default or empty

  template <typename T_Real>
  Status RFFTPlanImpl<T_Real>::rfft(
      std::span<const T_Real> input, std::span<T_Complex> output) const
  {
    if (p_spec_ == nullptr) return Status::NotInitialized;
    size_t N = length_;
    size_t output_size_expected = (N / 2) + 1;
    if (N == 0 && input.empty() && output.empty()) return Status::Success;
    if (input.size() != N || output.size() != output_size_expected)
      return Status::SizeMismatch;

    // IPP RFFT outputs to a packed format (real array). We need a temporary
    // buffer. The size of the packed buffer is N real elements.
    if (temp_packed_buf_.size() != N) {
      // This should not happen if constructor logic is correct, but safety
      // check
      return Status::Failure;  // Indicate internal state error
    }
    IPP_R_Type* packed_output_ptr
        = const_cast<IPP_R_Type*>(temp_packed_buf_.data());

    IppStatus ipp_status = internal::ippsFFTFwd_RToPack<T_Real>(
        reinterpret_cast<const IPP_R_Type*>(input.data()),
        packed_output_ptr,  // Output to temporary packed buffer
        p_spec_,
        work_buf_.empty()
            ? nullptr
            : const_cast<Ipp8u*>(work_buf_.data())  // Mutable work buffer
    );

    if (ipp_status != ippStsNoErr) {
      return IntelIPP::Details::ipp_status_to_omnidsp_status(ipp_status);
    }

    // --- Convert IPP Packed format to standard Complex format ---
    T_Complex* complex_output_ptr = output.data();
    complex_output_ptr[0]
        = T_Complex(packed_output_ptr[0], 0.0);  // DC component

    for (size_t k = 1; k < N / 2; ++k) {
      complex_output_ptr[k]
          = T_Complex(packed_output_ptr[2 * k], packed_output_ptr[2 * k + 1]);
    }

    if (N % 2 == 0) {  // Check if N is even for Nyquist
      // Index 1 in packed format holds the real part of the Nyquist frequency
      complex_output_ptr[N / 2] = T_Complex(packed_output_ptr[1], 0.0);
    }
    // If N is odd, the loop up to N/2-1 covers all necessary components.

    return Status::Success;
  }

  template <typename T_Real>
  Status RFFTPlanImpl<T_Real>::irfft(  // Renamed from irfft to match base
      std::span<const T_Complex> input,
      std::span<T_Real> output) const
  {
    if (p_spec_ == nullptr) return Status::NotInitialized;
    size_t N = length_;
    size_t input_size_expected = (N / 2) + 1;
    if (N == 0 && input.empty() && output.empty()) return Status::Success;
    if (input.size() != input_size_expected || output.size() != N)
      return Status::SizeMismatch;

    // --- Convert standard Complex format to IPP Packed format ---
    if (temp_packed_buf_.size() != N) {
      return Status::Failure;  // Internal state error
    }
    const T_Complex* complex_input_ptr = input.data();
    IPP_R_Type* packed_input_ptr
        = const_cast<IPP_R_Type*>(temp_packed_buf_.data());

    packed_input_ptr[0] = complex_input_ptr[0].real();  // DC component

    for (size_t k = 1; k < N / 2; ++k) {
      packed_input_ptr[2 * k] = complex_input_ptr[k].real();
      packed_input_ptr[2 * k + 1] = complex_input_ptr[k].imag();
    }

    if (N % 2 == 0) {  // Check if N is even for Nyquist
      packed_input_ptr[1]
          = complex_input_ptr[N / 2].real();  // Nyquist component
    }

    // Perform inverse FFT from packed format
    IppStatus ipp_status = internal::ippsFFTInv_PackToR<T_Real>(
        packed_input_ptr,  // Input from temporary packed buffer
        reinterpret_cast<IPP_R_Type*>(output.data()),
        p_spec_,
        work_buf_.empty()
            ? nullptr
            : const_cast<Ipp8u*>(work_buf_.data())  // Mutable work buffer
    );

    return IntelIPP::Details::ipp_status_to_omnidsp_status(ipp_status);
  }

  // get_length() is already defined in the header

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // These ensure the compiler generates code for these specific types.
  template class FFTPlanImpl<C32>;
  template class FFTPlanImpl<C64>;
  template class RFFTPlanImpl<F32>;
  template class RFFTPlanImpl<F64>;

}  // namespace OmniDSP::IntelIPP
