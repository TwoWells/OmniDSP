/**
 * @file iir_filter.hpp (IntelIPP)
 * @brief Declares the Intel IPP implementation class for IIR filter plans.
 */

#ifndef OMNIDSP_INTELIPP_IIR_FILTER_HPP
#define OMNIDSP_INTELIPP_IIR_FILTER_HPP

#include <ipp.h>  // Main IPP header including ipps.h

#include <OmniDSP/coefs/iir_filter.hpp>  // For IIRFilterCoef definition
#include <OmniDSP/core_types.hpp>  // For F32, C32, F64, C64, Status, OmniException, Utils::* etc.
#include <complex>
#include <memory>       // For std::unique_ptr
#include <span>         // For std::span in method signatures
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "details.hpp"  // For IPP type helpers (utils::GetIPPType etc.) and macros
#include "interface/backend.hpp"  // Defines abstract::IIRFilterPlanImpl

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // Helper Functions / Type Traits (Internal) - IIR Wrappers
  //--------------------------------------------------------------------------
  namespace internal {

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
        Details::GetIPPIIRState<T>** ppState,  // Use Utils::
        const Details::GetIPPType<T>* pTaps,   // Use Utils:: (Corrected type)
        int numBq,
        const Details::GetIPPType<T>* pDlyLine,  // Use Utils::
        Ipp8u* pBuf);
    template <>
    inline IppStatus ippsIIRInit_BiQuad<F32>(
        Details::GetIPPIIRState<F32>** ppState,
        const Details::GetIPPType<F32>* pTaps,
        int numBq,
        const Details::GetIPPType<F32>* pDlyLine,
        Ipp8u* pBuf)
    {  // Use Utils::
      return ::ippsIIRInit_BiQuad_32f(ppState, pTaps, numBq, pDlyLine, pBuf);
    }
    template <>
    inline IppStatus ippsIIRInit_BiQuad<F64>(
        Details::GetIPPIIRState<F64>** ppState,
        const Details::GetIPPType<F64>* pTaps,
        int numBq,
        const Details::GetIPPType<F64>* pDlyLine,
        Ipp8u* pBuf)
    {  // Use Utils::
      return ::ippsIIRInit_BiQuad_64f(ppState, pTaps, numBq, pDlyLine, pBuf);
    }
    // Add complex if needed/supported by IPP for BiQuad init

    // IIR Execute
    // *** CORRECTED WRAPPER ***
    template <typename T>
    inline IppStatus ippsIIR(
        const Details::GetIPPType<T>* pSrc,
        Details::GetIPPType<T>* pDst,
        int len,
        Details::GetIPPIIRState<T>* pState);  // Use Utils::
    template <>
    inline IppStatus ippsIIR<F32>(
        const Details::GetIPPType<F32>* pSrc,
        Details::GetIPPType<F32>* pDst,
        int len,
        Details::GetIPPIIRState<F32>* pState)
    {  // Use Utils::
      // Call the out-of-place version
      return ::ippsIIR_32f(pSrc, pDst, len, pState);
    }
    template <>
    inline IppStatus ippsIIR<F64>(
        const Details::GetIPPType<F64>* pSrc,
        Details::GetIPPType<F64>* pDst,
        int len,
        Details::GetIPPIIRState<F64>* pState)
    {  // Use Utils::
      // Call the out-of-place version
      return ::ippsIIR_64f(pSrc, pDst, len, pState);
    }
    // Add complex specializations if needed (assuming IppsIIRState_32fc/64fc
    // exist)
    /*
    template<> inline IppStatus ippsIIR<C32>(const Utils::GetIPPType<C32>* pSrc,
    Utils::GetIPPType<C32>* pDst, int len, Utils::GetIPPIIRState<C32>* pState) {
        // Assuming IppsIIRState_32fc exists and is mapped correctly by
    GetIPPIIRState return ::ippsIIR_32fc(pSrc, pDst, len, pState);
    }
    template<> inline IppStatus ippsIIR<C64>(const Utils::GetIPPType<C64>* pSrc,
    Utils::GetIPPType<C64>* pDst, int len, Utils::GetIPPIIRState<C64>* pState) {
        // Assuming IppsIIRState_64fc exists and is mapped correctly by
    GetIPPIIRState return ::ippsIIR_64fc(pSrc, pDst, len, pState);
    }
    */

    // IIR SetDlyLine
    template <typename T>
    inline IppStatus ippsIIRSetDlyLine(
        Details::GetIPPIIRState<T>* pState,
        const Details::GetIPPType<T>* pDlyLine);  // Use Utils::
    template <>
    inline IppStatus ippsIIRSetDlyLine<F32>(
        Details::GetIPPIIRState<F32>* pState,
        const Details::GetIPPType<F32>* pDlyLine)
    {  // Use Utils::
      return ::ippsIIRSetDlyLine_32f(pState, pDlyLine);
    }
    template <>
    inline IppStatus ippsIIRSetDlyLine<F64>(
        Details::GetIPPIIRState<F64>* pState,
        const Details::GetIPPType<F64>* pDlyLine)
    {  // Use Utils::
      return ::ippsIIRSetDlyLine_64f(pState, pDlyLine);
    }
    // Add complex if needed/supported by IPP

  }  // namespace internal

  //--------------------------------------------------------------------------
  // IntelIPP IIR Filter Plan Implementation (Biquad SOS)
  //--------------------------------------------------------------------------
  template <typename T>  // T is F32 or F64 (IPP IIR typically real only)
  class IIRFilterProcessorImpl final
      : public Abstract::IIRFilterProcessorImpl<T> {
    static_assert(
        std::is_same_v<T, F32> || std::is_same_v<T, F64>,
        "IIRFilterPlanImpl currently supports only F32 or F64.");
    // *** Use type helpers from Utils namespace ***
    using IPP_State_Type = Details::GetIPPIIRState<T>;

   public:
    explicit IIRFilterProcessorImpl(
        const std::vector<IIRFilterCoef>& sos_coefficients);
    ~IIRFilterProcessorImpl() override;

    IIRFilterProcessorImpl(const IIRFilterProcessorImpl&) = delete;
    IIRFilterProcessorImpl& operator=(const IIRFilterProcessorImpl&) = delete;
    IIRFilterProcessorImpl(IIRFilterProcessorImpl&&) = delete;
    IIRFilterProcessorImpl& operator=(IIRFilterProcessorImpl&&) = delete;

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

}  // namespace OmniDSP::IntelIPP

#endif  // OMNIDSP_INTELIPP_IIR_FILTER_HPP
