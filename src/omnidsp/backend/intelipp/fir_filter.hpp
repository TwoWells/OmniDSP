/**
 * @file fir_filter.hpp (IntelIPP)
 * @brief Declares the Intel IPP implementation class for FIR filter plans.
 */

#ifndef OMNIDSP_INTELIPP_FIR_FILTER_HPP
#define OMNIDSP_INTELIPP_FIR_FILTER_HPP

#include <ipp.h>  // Main IPP header including ipps.h

#include <OmniDSP/core_types.hpp>  // For F32, C32, F64, C64, Status, OmniException, Utils::* etc.
// #include <OmniDSP/filter.hpp> // Not needed for FIR directly
#include <complex>
#include <memory>       // For std::unique_ptr
#include <span>         // For std::span in method signatures
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "details.hpp"
#include "interface/backend.hpp"  // Defines abstract::FIRFilterPlanImpl

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // Helper Functions / Type Traits (Internal) - FIR Wrappers
  //--------------------------------------------------------------------------
  namespace internal {

    // --- Templated wrappers for IPP Filter functions ---
    // FIR GetSize
    template <typename T>
    inline IppStatus ippsFIRSRGetSize(
        int tapsLen, int* pSpecSize, int* pBufferSize);
    template <>
    inline IppStatus ippsFIRSRGetSize<F32>(
        int tapsLen, int* pSpecSize, int* pBufferSize)
    {
      return ::ippsFIRSRGetSize(tapsLen, ipp32f, pSpecSize, pBufferSize);
    }
    template <>
    inline IppStatus ippsFIRSRGetSize<F64>(
        int tapsLen, int* pSpecSize, int* pBufferSize)
    {
      return ::ippsFIRSRGetSize(tapsLen, ipp64f, pSpecSize, pBufferSize);
    }
    template <>
    inline IppStatus ippsFIRSRGetSize<C32>(
        int tapsLen, int* pSpecSize, int* pBufferSize)
    {
      return ::ippsFIRSRGetSize(tapsLen, ipp32fc, pSpecSize, pBufferSize);
    }
    template <>
    inline IppStatus ippsFIRSRGetSize<C64>(
        int tapsLen, int* pSpecSize, int* pBufferSize)
    {
      return ::ippsFIRSRGetSize(tapsLen, ipp64fc, pSpecSize, pBufferSize);
    }

    // FIR Init
    template <typename T>
    inline IppStatus ippsFIRSRInit(
        const Details::GetIPPType<T>* pTaps,
        int tapsLen,
        IppAlgType algType,
        Details::GetIPPFIRSpec<T>* pSpec);  // Use Utils::
    template <>
    inline IppStatus ippsFIRSRInit<F32>(
        const Details::GetIPPType<F32>* pTaps,
        int tapsLen,
        IppAlgType algType,
        Details::GetIPPFIRSpec<F32>* pSpec)
    {  // Use Utils::
      return ::ippsFIRSRInit_32f(pTaps, tapsLen, algType, pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<F64>(
        const Details::GetIPPType<F64>* pTaps,
        int tapsLen,
        IppAlgType algType,
        Details::GetIPPFIRSpec<F64>* pSpec)
    {  // Use Utils::
      return ::ippsFIRSRInit_64f(pTaps, tapsLen, algType, pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<C32>(
        const Details::GetIPPType<C32>* pTaps,
        int tapsLen,
        IppAlgType algType,
        Details::GetIPPFIRSpec<C32>* pSpec)
    {  // Use Utils::
      return ::ippsFIRSRInit_32fc(pTaps, tapsLen, algType, pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<C64>(
        const Details::GetIPPType<C64>* pTaps,
        int tapsLen,
        IppAlgType algType,
        Details::GetIPPFIRSpec<C64>* pSpec)
    {  // Use Utils::
      return ::ippsFIRSRInit_64fc(pTaps, tapsLen, algType, pSpec);
    }

    // FIR Execute
    template <typename T>
    inline IppStatus ippsFIRSR(
        const Details::GetIPPType<T>* pSrc,
        Details::GetIPPType<T>* pDst,
        int numIters,
        Details::GetIPPFIRSpec<T>* pSpec,
        const Details::GetIPPType<T>* pDlySrc,
        Details::GetIPPType<T>* pDlyDst,
        Ipp8u* pBuffer);  // Use Utils::
    template <>
    inline IppStatus ippsFIRSR<F32>(
        const Details::GetIPPType<F32>* pSrc,
        Details::GetIPPType<F32>* pDst,
        int numIters,
        Details::GetIPPFIRSpec<F32>* pSpec,
        const Details::GetIPPType<F32>* pDlySrc,
        Details::GetIPPType<F32>* pDlyDst,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFIRSR_32f(
          pSrc,
          pDst,
          numIters,
          pSpec,
          const_cast<Ipp32f*>(pDlySrc),
          pDlyDst,
          pBuffer);
    }
    template <>
    inline IppStatus ippsFIRSR<F64>(
        const Details::GetIPPType<F64>* pSrc,
        Details::GetIPPType<F64>* pDst,
        int numIters,
        Details::GetIPPFIRSpec<F64>* pSpec,
        const Details::GetIPPType<F64>* pDlySrc,
        Details::GetIPPType<F64>* pDlyDst,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFIRSR_64f(
          pSrc,
          pDst,
          numIters,
          pSpec,
          const_cast<Ipp64f*>(pDlySrc),
          pDlyDst,
          pBuffer);
    }
    template <>
    inline IppStatus ippsFIRSR<C32>(
        const Details::GetIPPType<C32>* pSrc,
        Details::GetIPPType<C32>* pDst,
        int numIters,
        Details::GetIPPFIRSpec<C32>* pSpec,
        const Details::GetIPPType<C32>* pDlySrc,
        Details::GetIPPType<C32>* pDlyDst,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFIRSR_32fc(
          pSrc,
          pDst,
          numIters,
          pSpec,
          const_cast<Ipp32fc*>(pDlySrc),
          pDlyDst,
          pBuffer);
    }
    template <>
    inline IppStatus ippsFIRSR<C64>(
        const Details::GetIPPType<C64>* pSrc,
        Details::GetIPPType<C64>* pDst,
        int numIters,
        Details::GetIPPFIRSpec<C64>* pSpec,
        const Details::GetIPPType<C64>* pDlySrc,
        Details::GetIPPType<C64>* pDlyDst,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFIRSR_64fc(
          pSrc,
          pDst,
          numIters,
          pSpec,
          const_cast<Ipp64fc*>(pDlySrc),
          pDlyDst,
          pBuffer);
    }
  }  // namespace internal

  //--------------------------------------------------------------------------
  // IntelIPP FIR Filter Plan Implementation
  //--------------------------------------------------------------------------
  template <typename T>
  class FIRFilterProcessorImpl final
      : public Abstract::FIRFilterProcessorImpl<T> {
    // *** Use type helpers from Utils namespace ***
    using IPP_Spec_Type = Details::GetIPPFIRSpec<T>;

   public:
    explicit FIRFilterProcessorImpl(const std::vector<T>& coefficients);
    ~FIRFilterProcessorImpl() override;

    FIRFilterProcessorImpl(const FIRFilterProcessorImpl&) = delete;
    FIRFilterProcessorImpl& operator=(const FIRFilterProcessorImpl&) = delete;
    FIRFilterProcessorImpl(FIRFilterProcessorImpl&&) = delete;
    FIRFilterProcessorImpl& operator=(FIRFilterProcessorImpl&&) = delete;

    [[nodiscard]] OmniStatus execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] OmniStatus reset() override;

    size_t get_order() const override;
    size_t get_num_taps() const override;

   private:
    std::vector<T> coefficients_;
    size_t num_taps_;
    size_t order_;
    Ipp8u* p_spec_ = nullptr;    // Raw pointer to spec memory
    Ipp8u* p_buffer_ = nullptr;  // Raw pointer to buffer memory
    int spec_size_ = 0;
    int buffer_size_ = 0;
    IPP_Spec_Type* p_ipp_spec_ = nullptr;  // Typed pointer into p_spec_
  };

}  // namespace OmniDSP::IntelIPP

#endif  // OMNIDSP_INTELIPP_FIR_FILTER_HPP
