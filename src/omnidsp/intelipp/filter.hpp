/**
 * @file filter.hpp (intelipp)
 * @brief Declares the Intel IPP implementation classes for FIR and IIR filter
 * plans.
 */

#ifndef OMNIDSP_INTELIPP_FILTER_HPP
#define OMNIDSP_INTELIPP_FILTER_HPP

#include <ipp.h>  // Main IPP header including ipps.h

#include <OmniDSP/core_types.hpp>  // For F32, C32, F64, C64, Status, OmniException, Utils::* etc.
#include <OmniDSP/filter.hpp>  // For IIRFilterCoef definition
#include <complex>
#include <memory>       // For std::unique_ptr
#include <span>         // For std::span in method signatures
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "../interface/backend.hpp"  // Defines abstract::FIRFilterPlanImpl, abstract::IIRFilterPlanImpl
#include "utils.hpp"  // For IPP type helpers (utils::GetIPPType etc.) and macros

namespace OmniDSP::intelipp {

  //--------------------------------------------------------------------------
  // Helper Functions / Type Traits (Internal) - Wrappers only now
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

    // IIR GetStateSize
    template <typename T>
    inline IppStatus ippsIIRGetStateSize_BiQuad(int numBq, int* pStateSize);
    template <>
    inline IppStatus ippsIIRGetStateSize_BiQuad<F32>(int numBq, int* pStateSize)
    {
      return ::ippsIIRGetStateSize_BiQuad_32f(numBq, pStateSize);
    }
    template <>
    inline IppStatus ippsIIRGetStateSize_BiQuad<F64>(int numBq, int* pStateSize)
    {
      return ::ippsIIRGetStateSize_BiQuad_64f(numBq, pStateSize);
    }
    // Add complex if needed/supported by IPP for BiQuad size

    // IIR Init (5 Args)
    template <typename T>
    inline IppStatus ippsIIRInit_BiQuad(
        utils::GetIPPIIRState<T>** ppState,  // Use utils::
        const utils::GetIPPType<T>* pTaps,   // Use utils:: (Corrected type)
        int numBq,
        const utils::GetIPPType<T>* pDlyLine,  // Use utils::
        Ipp8u* pBuf);
    template <>
    inline IppStatus ippsIIRInit_BiQuad<F32>(
        utils::GetIPPIIRState<F32>** ppState,
        const utils::GetIPPType<F32>* pTaps,
        int numBq,
        const utils::GetIPPType<F32>* pDlyLine,
        Ipp8u* pBuf)
    {  // Use utils::
      return ::ippsIIRInit_BiQuad_32f(ppState, pTaps, numBq, pDlyLine, pBuf);
    }
    template <>
    inline IppStatus ippsIIRInit_BiQuad<F64>(
        utils::GetIPPIIRState<F64>** ppState,
        const utils::GetIPPType<F64>* pTaps,
        int numBq,
        const utils::GetIPPType<F64>* pDlyLine,
        Ipp8u* pBuf)
    {  // Use utils::
      return ::ippsIIRInit_BiQuad_64f(ppState, pTaps, numBq, pDlyLine, pBuf);
    }
    // Add complex if needed/supported by IPP for BiQuad init

    // IIR Execute
    // *** CORRECTED WRAPPER ***
    template <typename T>
    inline IppStatus ippsIIR(
        const utils::GetIPPType<T>* pSrc,
        utils::GetIPPType<T>* pDst,
        int len,
        utils::GetIPPIIRState<T>* pState);  // Use utils::
    template <>
    inline IppStatus ippsIIR<F32>(
        const utils::GetIPPType<F32>* pSrc,
        utils::GetIPPType<F32>* pDst,
        int len,
        utils::GetIPPIIRState<F32>* pState)
    {  // Use utils::
      // Call the out-of-place version
      return ::ippsIIR_32f(pSrc, pDst, len, pState);
    }
    template <>
    inline IppStatus ippsIIR<F64>(
        const utils::GetIPPType<F64>* pSrc,
        utils::GetIPPType<F64>* pDst,
        int len,
        utils::GetIPPIIRState<F64>* pState)
    {  // Use utils::
      // Call the out-of-place version
      return ::ippsIIR_64f(pSrc, pDst, len, pState);
    }
    // Add complex specializations if needed (assuming IppsIIRState_32fc/64fc
    // exist)
    /*
    template<> inline IppStatus ippsIIR<C32>(const utils::GetIPPType<C32>* pSrc,
    utils::GetIPPType<C32>* pDst, int len, utils::GetIPPIIRState<C32>* pState) {
        // Assuming IppsIIRState_32fc exists and is mapped correctly by
    GetIPPIIRState return ::ippsIIR_32fc(pSrc, pDst, len, pState);
    }
    template<> inline IppStatus ippsIIR<C64>(const utils::GetIPPType<C64>* pSrc,
    utils::GetIPPType<C64>* pDst, int len, utils::GetIPPIIRState<C64>* pState) {
        // Assuming IppsIIRState_64fc exists and is mapped correctly by
    GetIPPIIRState return ::ippsIIR_64fc(pSrc, pDst, len, pState);
    }
    */

    // IIR SetDlyLine
    template <typename T>
    inline IppStatus ippsIIRSetDlyLine(
        utils::GetIPPIIRState<T>* pState,
        const utils::GetIPPType<T>* pDlyLine);  // Use utils::
    template <>
    inline IppStatus ippsIIRSetDlyLine<F32>(
        utils::GetIPPIIRState<F32>* pState,
        const utils::GetIPPType<F32>* pDlyLine)
    {  // Use utils::
      return ::ippsIIRSetDlyLine_32f(pState, pDlyLine);
    }
    template <>
    inline IppStatus ippsIIRSetDlyLine<F64>(
        utils::GetIPPIIRState<F64>* pState,
        const utils::GetIPPType<F64>* pDlyLine)
    {  // Use utils::
      return ::ippsIIRSetDlyLine_64f(pState, pDlyLine);
    }
    // Add complex if needed/supported by IPP

  }  // namespace internal

  //--------------------------------------------------------------------------
  // IntelIPP FIR Filter Plan Implementation
  //--------------------------------------------------------------------------
  template <typename T>
  class IntelIPPFIRFilterPlanImpl final
      : public abstract::FIRFilterPlanImpl<T> {
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

  //--------------------------------------------------------------------------
  // IntelIPP IIR Filter Plan Implementation (Biquad SOS)
  //--------------------------------------------------------------------------
  template <typename T>  // T is F32 or F64 (IPP IIR typically real only)
  class IntelIPPIIRFilterPlanImpl final
      : public abstract::IIRFilterPlanImpl<T> {
    static_assert(
        std::is_same_v<T, F32> || std::is_same_v<T, F64>,
        "IntelIPPIIRFilterPlanImpl currently supports only F32 or F64.");
    // *** Use type helpers from utils namespace ***
    using IPP_State_Type = utils::GetIPPIIRState<T>;

   public:
    explicit IntelIPPIIRFilterPlanImpl(
        const std::vector<IIRFilterCoef>& sos_coefficients);
    ~IntelIPPIIRFilterPlanImpl() override;

    IntelIPPIIRFilterPlanImpl(const IntelIPPIIRFilterPlanImpl&) = delete;
    IntelIPPIIRFilterPlanImpl& operator=(const IntelIPPIIRFilterPlanImpl&)
        = delete;
    IntelIPPIIRFilterPlanImpl(IntelIPPIIRFilterPlanImpl&&) = delete;
    IntelIPPIIRFilterPlanImpl& operator=(IntelIPPIIRFilterPlanImpl&&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) override;
    [[nodiscard]] Status reset() override;

    size_t get_order() const override;
    size_t get_num_sections() const override;

   private:
    size_t num_sections_;
    size_t order_;
    IPP_State_Type* p_state_ = nullptr;  // Typed pointer into state_mem_
    std::vector<Ipp8u>
        state_mem_;             // Raw memory buffer for the IPP state structure
    int state_size_bytes_ = 0;  // Store size for debugging/verification
    // Store coefficients in the correct type T for passing to IPP Init
    std::vector<T> taps_interleaved_;
  };

}  // namespace OmniDSP::intelipp

#endif  // OMNIDSP_INTELIPP_FILTER_HPP
