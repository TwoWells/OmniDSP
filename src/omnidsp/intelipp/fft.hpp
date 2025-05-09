/**
 * @file fft.hpp (IntelIPP)
 * @brief Declares the Intel IPP implementation classes for FFT plans.
 */

#ifndef OMNIDSP_INTELIPP_FFT_HPP
#define OMNIDSP_INTELIPP_FFT_HPP

#include <ipp.h>  // Main IPP header including ipps.h

#include <OmniDSP/core_types.hpp>  // For F32, C32, F64, C64, Status, OmniException, Utils::* etc.
#include <cmath>  // For log2
#include <complex>
#include <memory>       // For std::unique_ptr
#include <span>         // For std::span in method signatures
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "../interface/backend.hpp"  // Defines abstract::FFTPlanImpl, abstract::RFFTPlanImpl
#include "details.hpp"

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // Helper Functions / Type Traits (Internal) - Wrappers only now
  //--------------------------------------------------------------------------
  namespace internal {

    // Check if a number is a power of two
    inline bool is_power_of_two(size_t n)
    {
      return (n > 0) && ((n & (n - 1)) == 0);
    }

    // Calculate base-2 logarithm (order) for FFT
    inline int calculate_fft_order(size_t n)
    {
      if (!is_power_of_two(n)) {
        // Consider throwing OmniException for consistency
        throw std::invalid_argument(
            "FFT length must be a power of two for this IPP implementation.");
      }
      // Using cmath functions - ensure correct type casting
      return static_cast<int>(std::log2(static_cast<double>(n)));
    }

    // *** REMOVED Type mapping helpers - Moved to Utils.hpp ***

    // --- Templated wrappers for IPP FFT functions ---
    // GetSize Cplx (Requires 6 arguments: order, flag, hint, pSpecSize,
    // pSpecBufferSize, pBufferSize)
    template <typename T_Complex>
    inline IppStatus ippsFFTGetSize_C(
        int order,
        int flag,
        IppHintAlgorithm hint,
        int* pSpecSize,
        int* pSpecBufferSize,
        int* pBufferSize);
    template <>
    inline IppStatus ippsFFTGetSize_C<C32>(
        int order,
        int flag,
        IppHintAlgorithm hint,
        int* pSpecSize,
        int* pSpecBufferSize,
        int* pBufferSize)
    {
      return ::ippsFFTGetSize_C_32fc(
          order, flag, hint, pSpecSize, pSpecBufferSize, pBufferSize);
    }
    template <>
    inline IppStatus ippsFFTGetSize_C<C64>(
        int order,
        int flag,
        IppHintAlgorithm hint,
        int* pSpecSize,
        int* pSpecBufferSize,
        int* pBufferSize)
    {
      return ::ippsFFTGetSize_C_64fc(
          order, flag, hint, pSpecSize, pSpecBufferSize, pBufferSize);
    }

    // GetSize Real (Requires 6 arguments)
    template <typename T_Real>
    inline IppStatus ippsFFTGetSize_R(
        int order,
        int flag,
        IppHintAlgorithm hint,
        int* pSpecSize,
        int* pSpecBufferSize,
        int* pBufferSize);
    template <>
    inline IppStatus ippsFFTGetSize_R<F32>(
        int order,
        int flag,
        IppHintAlgorithm hint,
        int* pSpecSize,
        int* pSpecBufferSize,
        int* pBufferSize)
    {
      return ::ippsFFTGetSize_R_32f(
          order, flag, hint, pSpecSize, pSpecBufferSize, pBufferSize);
    }
    template <>
    inline IppStatus ippsFFTGetSize_R<F64>(
        int order,
        int flag,
        IppHintAlgorithm hint,
        int* pSpecSize,
        int* pSpecBufferSize,
        int* pBufferSize)
    {
      return ::ippsFFTGetSize_R_64f(
          order, flag, hint, pSpecSize, pSpecBufferSize, pBufferSize);
    }

    // Init Cplx (Requires 6 arguments: ppFFTSpec, order, flag, hint, pSpec,
    // pSpecBuffer)
    template <typename T_Complex>
    inline IppStatus ippsFFTInit_C(
        Details::GetIPPFFTSpec<T_Complex>** ppFFTSpec,
        int order,
        int flag,
        IppHintAlgorithm hint,
        Ipp8u* pSpec,
        Ipp8u* pSpecBuffer);  // Use Utils::
    template <>
    inline IppStatus ippsFFTInit_C<C32>(
        Details::GetIPPFFTSpec<C32>** ppFFTSpec,
        int order,
        int flag,
        IppHintAlgorithm hint,
        Ipp8u* pSpec,
        Ipp8u* pSpecBuffer)
    {  // Use Utils::
      return ::ippsFFTInit_C_32fc(
          ppFFTSpec, order, flag, hint, pSpec, pSpecBuffer);
    }
    template <>
    inline IppStatus ippsFFTInit_C<C64>(
        Details::GetIPPFFTSpec<C64>** ppFFTSpec,
        int order,
        int flag,
        IppHintAlgorithm hint,
        Ipp8u* pSpec,
        Ipp8u* pSpecBuffer)
    {  // Use Utils::
      return ::ippsFFTInit_C_64fc(
          ppFFTSpec, order, flag, hint, pSpec, pSpecBuffer);
    }

    // Init Real (Requires 6 arguments)
    template <typename T_Real>
    inline IppStatus ippsFFTInit_R(
        Details::GetIPPFFTSpec<T_Real>** ppFFTSpec,
        int order,
        int flag,
        IppHintAlgorithm hint,
        Ipp8u* pSpec,
        Ipp8u* pSpecBuffer);  // Use Utils::
    template <>
    inline IppStatus ippsFFTInit_R<F32>(
        Details::GetIPPFFTSpec<F32>** ppFFTSpec,
        int order,
        int flag,
        IppHintAlgorithm hint,
        Ipp8u* pSpec,
        Ipp8u* pSpecBuffer)
    {  // Use Utils::
      return ::ippsFFTInit_R_32f(
          ppFFTSpec, order, flag, hint, pSpec, pSpecBuffer);
    }
    template <>
    inline IppStatus ippsFFTInit_R<F64>(
        Details::GetIPPFFTSpec<F64>** ppFFTSpec,
        int order,
        int flag,
        IppHintAlgorithm hint,
        Ipp8u* pSpec,
        Ipp8u* pSpecBuffer)
    {  // Use Utils::
      return ::ippsFFTInit_R_64f(
          ppFFTSpec, order, flag, hint, pSpec, pSpecBuffer);
    }

    // Forward Cplx (Requires 4 arguments: pSrc, pDst, pFFTSpec, pBuffer)
    template <typename T_Complex>
    inline IppStatus ippsFFTFwd_CToC(
        const Details::GetIPPType<T_Complex>* pSrc,
        Details::GetIPPType<T_Complex>* pDst,
        const Details::GetIPPFFTSpec<T_Complex>* pFFTSpec,
        Ipp8u* pBuffer);  // Use Utils::
    template <>
    inline IppStatus ippsFFTFwd_CToC<C32>(
        const Details::GetIPPType<C32>* pSrc,
        Details::GetIPPType<C32>* pDst,
        const Details::GetIPPFFTSpec<C32>* pFFTSpec,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFFTFwd_CToC_32fc(pSrc, pDst, pFFTSpec, pBuffer);
    }
    template <>
    inline IppStatus ippsFFTFwd_CToC<C64>(
        const Details::GetIPPType<C64>* pSrc,
        Details::GetIPPType<C64>* pDst,
        const Details::GetIPPFFTSpec<C64>* pFFTSpec,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFFTFwd_CToC_64fc(pSrc, pDst, pFFTSpec, pBuffer);
    }

    // Inverse Cplx (Requires 4 arguments)
    template <typename T_Complex>
    inline IppStatus ippsFFTInv_CToC(
        const Details::GetIPPType<T_Complex>* pSrc,
        Details::GetIPPType<T_Complex>* pDst,
        const Details::GetIPPFFTSpec<T_Complex>* pFFTSpec,
        Ipp8u* pBuffer);  // Use Utils::
    template <>
    inline IppStatus ippsFFTInv_CToC<C32>(
        const Details::GetIPPType<C32>* pSrc,
        Details::GetIPPType<C32>* pDst,
        const Details::GetIPPFFTSpec<C32>* pFFTSpec,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFFTInv_CToC_32fc(pSrc, pDst, pFFTSpec, pBuffer);
    }
    template <>
    inline IppStatus ippsFFTInv_CToC<C64>(
        const Details::GetIPPType<C64>* pSrc,
        Details::GetIPPType<C64>* pDst,
        const Details::GetIPPFFTSpec<C64>* pFFTSpec,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFFTInv_CToC_64fc(pSrc, pDst, pFFTSpec, pBuffer);
    }

    // Forward Real (Using Pack format - Requires 4 arguments)
    template <typename T_Real>
    inline IppStatus ippsFFTFwd_RToPack(
        const Details::GetIPPType<T_Real>* pSrc,
        Details::GetIPPType<T_Real>* pDst,
        const Details::GetIPPFFTSpec<T_Real>* pFFTSpec,
        Ipp8u* pBuffer);  // Use Utils::
    template <>
    inline IppStatus ippsFFTFwd_RToPack<F32>(
        const Details::GetIPPType<F32>* pSrc,
        Details::GetIPPType<F32>* pDst,
        const Details::GetIPPFFTSpec<F32>* pFFTSpec,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFFTFwd_RToPack_32f(pSrc, pDst, pFFTSpec, pBuffer);
    }
    template <>
    inline IppStatus ippsFFTFwd_RToPack<F64>(
        const Details::GetIPPType<F64>* pSrc,
        Details::GetIPPType<F64>* pDst,
        const Details::GetIPPFFTSpec<F64>* pFFTSpec,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFFTFwd_RToPack_64f(pSrc, pDst, pFFTSpec, pBuffer);
    }

    // Inverse Real (Using Pack format - Requires 4 arguments)
    template <typename T_Real>
    inline IppStatus ippsFFTInv_PackToR(
        const Details::GetIPPType<T_Real>* pSrc,
        Details::GetIPPType<T_Real>* pDst,
        const Details::GetIPPFFTSpec<T_Real>* pFFTSpec,
        Ipp8u* pBuffer);  // Use Utils::
    template <>
    inline IppStatus ippsFFTInv_PackToR<F32>(
        const Details::GetIPPType<F32>* pSrc,
        Details::GetIPPType<F32>* pDst,
        const Details::GetIPPFFTSpec<F32>* pFFTSpec,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFFTInv_PackToR_32f(pSrc, pDst, pFFTSpec, pBuffer);
    }
    template <>
    inline IppStatus ippsFFTInv_PackToR<F64>(
        const Details::GetIPPType<F64>* pSrc,
        Details::GetIPPType<F64>* pDst,
        const Details::GetIPPFFTSpec<F64>* pFFTSpec,
        Ipp8u* pBuffer)
    {  // Use Utils::
      return ::ippsFFTInv_PackToR_64f(pSrc, pDst, pFFTSpec, pBuffer);
    }

  }  // namespace internal

  //--------------------------------------------------------------------------
  // IntelIPP Complex FFT Plan Implementation
  //--------------------------------------------------------------------------
  template <typename T_Complex>
  class FFTPlanImpl final : public Abstract::FFTPlanImpl<T_Complex> {
    static_assert(
        std::is_same_v<T_Complex, C32> || std::is_same_v<T_Complex, C64>,
        "FFTPlanImpl supports only C32 or C64.");
    // *** Use type helpers from Utils namespace ***
    using IPP_C_Type = Details::GetIPPType<T_Complex>;
    using IPP_Spec_Type = Details::GetIPPFFTSpec<T_Complex>;

   public:
    explicit FFTPlanImpl(size_t length);
    ~FFTPlanImpl() override = default;

    FFTPlanImpl(const FFTPlanImpl&) = delete;
    FFTPlanImpl& operator=(const FFTPlanImpl&) = delete;
    FFTPlanImpl(FFTPlanImpl&&) = delete;
    FFTPlanImpl& operator=(FFTPlanImpl&&) = delete;

    [[nodiscard]] Status fft(  // Use 'fft' to match abstract base
        std::span<const T_Complex> input,
        std::span<T_Complex> output) const override;

    [[nodiscard]] Status ifft(  // Use 'ifft' to match abstract base
        std::span<const T_Complex> input,
        std::span<T_Complex> output) const override;

    size_t get_length() const noexcept override { return length_; }

   private:
    size_t length_;
    int order_;
    int flag_;
    IPP_Spec_Type* p_spec_ = nullptr;
    std::vector<Ipp8u> spec_mem_;
    std::vector<Ipp8u> init_buf_;
    mutable std::vector<Ipp8u> work_buf_;
  };

  //--------------------------------------------------------------------------
  // IntelIPP Real FFT Plan Implementation
  //--------------------------------------------------------------------------
  template <typename T_Real>
  class IntelIPPRFFTPlanImpl final : public Abstract::RFFTPlanImpl<T_Real> {
    static_assert(
        std::is_same_v<T_Real, F32> || std::is_same_v<T_Real, F64>,
        "IntelIPPRFFTPlanImpl supports only F32 or F64.");
    // *** Use type helpers from Utils namespace ***
    using T_Complex = ::OmniDSP::Utils::GetComplexType<
        T_Real>;  // Get corresponding complex type from core_types
    using IPP_R_Type = Details::GetIPPType<T_Real>;
    using IPP_C_Type
        = Details::GetIPPType<T_Complex>;  // Needed for output span type
    using IPP_Spec_Type = Details::GetIPPFFTSpec<T_Real>;

   public:
    explicit IntelIPPRFFTPlanImpl(size_t length);
    ~IntelIPPRFFTPlanImpl() override = default;

    IntelIPPRFFTPlanImpl(const IntelIPPRFFTPlanImpl&) = delete;
    IntelIPPRFFTPlanImpl& operator=(const IntelIPPRFFTPlanImpl&) = delete;
    IntelIPPRFFTPlanImpl(IntelIPPRFFTPlanImpl&&) = delete;
    IntelIPPRFFTPlanImpl& operator=(IntelIPPRFFTPlanImpl&&) = delete;

    [[nodiscard]] Status rfft(  // Use 'rfft' to match abstract base
        std::span<const T_Real> input,
        std::span<T_Complex> output) const override;

    [[nodiscard]] Status irfft(  // Use 'irfft' to match abstract base
        std::span<const T_Complex> input,
        std::span<T_Real> output) const override;

    size_t get_length() const noexcept override { return length_; }

   private:
    size_t length_;
    int order_;
    int flag_;
    IPP_Spec_Type* p_spec_ = nullptr;
    std::vector<Ipp8u> spec_mem_;
    std::vector<Ipp8u> init_buf_;
    mutable std::vector<Ipp8u> work_buf_;
    mutable std::vector<IPP_R_Type> temp_packed_buf_;
  };

}  // namespace OmniDSP::IntelIPP

#endif  // OMNIDSP_INTELIPP_FFT_HPP
