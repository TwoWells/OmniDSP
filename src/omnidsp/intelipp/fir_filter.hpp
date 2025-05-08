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

#include "../interface/backend.hpp"  // Defines abstract::FIRFilterPlanImpl
#include "utils.hpp"  // For IPP type helpers (utils::GetIPPType etc.) and macros

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
        const utils::GetIPPType<T>* pTaps,
        int tapsLen,
        IppAlgType algType,
        utils::GetIPPFIRSpec<T>* pSpec);  // Use utils::
    template <>
    inline IppStatus ippsFIRSRInit<F32>(
        const utils::GetIPPType<F32>* pTaps,
        int tapsLen,
        IppAlgType algType,
        utils::GetIPPFIRSpec<F32>* pSpec)
    {  // Use utils::
      return ::ippsFIRSRInit_32f(pTaps, tapsLen, algType, pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<F64>(
        const utils::GetIPPType<F64>* pTaps,
        int tapsLen,
        IppAlgType algType,
        utils::GetIPPFIRSpec<F64>* pSpec)
    {  // Use utils::
      return ::ippsFIRSRInit_64f(pTaps, tapsLen, algType, pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<C32>(
        const utils::GetIPPType<C32>* pTaps,
        int tapsLen,
        IppAlgType algType,
        utils::GetIPPFIRSpec<C32>* pSpec)
    {  // Use utils::
      return ::ippsFIRSRInit_32fc(pTaps, tapsLen, algType, pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<C64>(
        const utils::GetIPPType<C64>* pTaps,
        int tapsLen,
        IppAlgType algType,
        utils::GetIPPFIRSpec<C64>* pSpec)
    {  // Use utils::
      return ::ippsFIRSRInit_64fc(pTaps, tapsLen, algType, pSpec);
    }

    // FIR Execute
    template <typename T>
    inline IppStatus ippsFIRSR(
        const utils::GetIPPType<T>* pSrc,
        utils::GetIPPType<T>* pDst,
        int numIters,
        utils::GetIPPFIRSpec<T>* pSpec,
        const utils::GetIPPType<T>* pDlySrc,
        utils::GetIPPType<T>* pDlyDst,
        Ipp8u* pBuffer);  // Use utils::
    template <>
    inline IppStatus ippsFIRSR<F32>(
        const utils::GetIPPType<F32>* pSrc,
        utils::GetIPPType<F32>* pDst,
        int numIters,
        utils::GetIPPFIRSpec<F32>* pSpec,
        const utils::GetIPPType<F32>* pDlySrc,
        utils::GetIPPType<F32>* pDlyDst,
        Ipp8u* pBuffer)
    {  // Use utils::
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
        const utils::GetIPPType<F64>* pSrc,
        utils::GetIPPType<F64>* pDst,
        int numIters,
        utils::GetIPPFIRSpec<F64>* pSpec,
        const utils::GetIPPType<F64>* pDlySrc,
        utils::GetIPPType<F64>* pDlyDst,
        Ipp8u* pBuffer)
    {  // Use utils::
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
        const utils::GetIPPType<C32>* pSrc,
        utils::GetIPPType<C32>* pDst,
        int numIters,
        utils::GetIPPFIRSpec<C32>* pSpec,
        const utils::GetIPPType<C32>* pDlySrc,
        utils::GetIPPType<C32>* pDlyDst,
        Ipp8u* pBuffer)
    {  // Use utils::
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
        const utils::GetIPPType<C64>* pSrc,
        utils::GetIPPType<C64>* pDst,
        int numIters,
        utils::GetIPPFIRSpec<C64>* pSpec,
        const utils::GetIPPType<C64>* pDlySrc,
        utils::GetIPPType<C64>* pDlyDst,
        Ipp8u* pBuffer)
    {  // Use utils::
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
  class IntelIPPFIRFilterPlanImpl final
      : public Abstract::FIRFilterPlanImpl<T> {
    // *** Use type helpers from utils namespace ***
    using IPP_Spec_Type = utils::GetIPPFIRSpec<T>;

   public:
    explicit IntelIPPFIRFilterPlanImpl(const std::vector<T>& coefficients);
    ~IntelIPPFIRFilterPlanImpl() override;

    IntelIPPFIRFilterPlanImpl(const IntelIPPFIRFilterPlanImpl&) = delete;
    IntelIPPFIRFilterPlanImpl& operator=(const IntelIPPFIRFilterPlanImpl&)
        = delete;
    IntelIPPFIRFilterPlanImpl(IntelIPPFIRFilterPlanImpl&&) = delete;
    IntelIPPFIRFilterPlanImpl& operator=(IntelIPPFIRFilterPlanImpl&&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] Status reset() override;

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
