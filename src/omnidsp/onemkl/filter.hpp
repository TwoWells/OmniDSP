/**
 * @file filter.hpp (onemkl)
 * @brief Declares the oneMKL backend FIRFilterPlanImpl and IIRFilterPlanImpl
 * classes using IPP.
 */

#ifndef OMNIDSP_ONEMKL_FILTER_HPP
#define OMNIDSP_ONEMKL_FILTER_HPP

#ifdef OMNIDSP_USE_ONEMKL  // Compile guard

#include <ipps.h>  // IPP Signal Processing header

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/filter.hpp>  // For IIRFilterCoef definition
#include <complex>
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr
#include <span>
#include <type_traits>  // For std::conditional, std::is_same_v
#include <vector>

#include "../interface/backend.hpp"  // Base PlanImpl interfaces

namespace OmniDSP::backend {

  // --- IPP Type Mapping Helpers ---
  namespace ipp_helpers {
    // ... (Helpers remain the same) ...
    template <typename T>
    struct IppDataTypeHelper;
    template <>
    struct IppDataTypeHelper<F32> {
      static constexpr IppDataType value = ipp32f;
    };
    template <>
    struct IppDataTypeHelper<F64> {
      static constexpr IppDataType value = ipp64f;
    };
    template <>
    struct IppDataTypeHelper<C32> {
      static constexpr IppDataType value = ipp32fc;
    };
    template <>
    struct IppDataTypeHelper<C64> {
      static constexpr IppDataType value = ipp64fc;
    };

    template <typename T>
    struct IppComplexTypeHelper;
    template <>
    struct IppComplexTypeHelper<C32> {
      using type = Ipp32fc;
    };
    template <>
    struct IppComplexTypeHelper<C64> {
      using type = Ipp64fc;
    };

    template <typename T>
    struct IppsFIRSpecTypeHelper;
    template <>
    struct IppsFIRSpecTypeHelper<F32> {
      using type = IppsFIRSpec_32f;
    };
    template <>
    struct IppsFIRSpecTypeHelper<F64> {
      using type = IppsFIRSpec_64f;
    };
    template <>
    struct IppsFIRSpecTypeHelper<C32> {
      using type = IppsFIRSpec_32fc;
    };
    template <>
    struct IppsFIRSpecTypeHelper<C64> {
      using type = IppsFIRSpec_64fc;
    };

    template <typename T>
    struct IppsIIRStateTypeHelper;
    template <>
    struct IppsIIRStateTypeHelper<F32> {
      using type = IppsIIRState_32f;
    };
    template <>
    struct IppsIIRStateTypeHelper<F64> {
      using type = IppsIIRState_64f;
    };

  }  // namespace ipp_helpers

  // --- Consistent Wrapper Aliases ---
  template <typename T>
  constexpr IppDataType IppwDataType = ipp_helpers::IppDataTypeHelper<T>::value;
  template <typename T>
  using IppwFIRSpec = typename ipp_helpers::IppsFIRSpecTypeHelper<T>::type;
  template <typename T>
  using IppwIIRState = typename ipp_helpers::IppsIIRStateTypeHelper<T>::type;
  template <typename T>
  using IppwComplex = typename ipp_helpers::IppComplexTypeHelper<T>::type;

  // --- IPP Function Dispatch Templates (using new aliases) ---
  namespace ipp_dispatch {

    // ... (ippsFIRSRInit and ippsFIRSR remain the same) ...
    template <typename T>
    inline IppStatus ippsFIRSRInit(
        const T* pTaps, int tapsLen, IppAlgType algType, IppwFIRSpec<T>* pSpec);
    template <>
    inline IppStatus ippsFIRSRInit<F32>(
        const F32* pTaps,
        int tapsLen,
        IppAlgType algType,
        IppwFIRSpec<F32>* pSpec)
    {
      return ippsFIRSRInit_32f(pTaps, tapsLen, algType, pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<F64>(
        const F64* pTaps,
        int tapsLen,
        IppAlgType algType,
        IppwFIRSpec<F64>* pSpec)
    {
      return ippsFIRSRInit_64f(pTaps, tapsLen, algType, pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<C32>(
        const C32* pTaps,
        int tapsLen,
        IppAlgType algType,
        IppwFIRSpec<C32>* pSpec)
    {
      return ippsFIRSRInit_32fc(
          reinterpret_cast<const IppwComplex<C32>*>(pTaps),
          tapsLen,
          algType,
          pSpec);
    }
    template <>
    inline IppStatus ippsFIRSRInit<C64>(
        const C64* pTaps,
        int tapsLen,
        IppAlgType algType,
        IppwFIRSpec<C64>* pSpec)
    {
      return ippsFIRSRInit_64fc(
          reinterpret_cast<const IppwComplex<C64>*>(pTaps),
          tapsLen,
          algType,
          pSpec);
    }

    template <typename T>
    inline IppStatus ippsFIRSR(
        const T* pSrc,
        T* pDst,
        int numIters,
        IppwFIRSpec<T>* pSpec,
        const T* pDlySrc,
        T* pDlyDst,
        Ipp8u* pBuf);
    template <>
    inline IppStatus ippsFIRSR<F32>(
        const F32* pSrc,
        F32* pDst,
        int numIters,
        IppwFIRSpec<F32>* pSpec,
        const F32* pDlySrc,
        F32* pDlyDst,
        Ipp8u* pBuf)
    {
      return ippsFIRSR_32f(pSrc, pDst, numIters, pSpec, pDlySrc, pDlyDst, pBuf);
    }
    template <>
    inline IppStatus ippsFIRSR<F64>(
        const F64* pSrc,
        F64* pDst,
        int numIters,
        IppwFIRSpec<F64>* pSpec,
        const F64* pDlySrc,
        F64* pDlyDst,
        Ipp8u* pBuf)
    {
      return ippsFIRSR_64f(pSrc, pDst, numIters, pSpec, pDlySrc, pDlyDst, pBuf);
    }
    template <>
    inline IppStatus ippsFIRSR<C32>(
        const C32* pSrc,
        C32* pDst,
        int numIters,
        IppwFIRSpec<C32>* pSpec,
        const C32* pDlySrc,
        C32* pDlyDst,
        Ipp8u* pBuf)
    {
      return ippsFIRSR_32fc(
          reinterpret_cast<const IppwComplex<C32>*>(pSrc),
          reinterpret_cast<IppwComplex<C32>*>(pDst),
          numIters,
          pSpec,
          reinterpret_cast<const IppwComplex<C32>*>(pDlySrc),
          reinterpret_cast<IppwComplex<C32>*>(pDlyDst),
          pBuf);
    }
    template <>
    inline IppStatus ippsFIRSR<C64>(
        const C64* pSrc,
        C64* pDst,
        int numIters,
        IppwFIRSpec<C64>* pSpec,
        const C64* pDlySrc,
        C64* pDlyDst,
        Ipp8u* pBuf)
    {
      return ippsFIRSR_64fc(
          reinterpret_cast<const IppwComplex<C64>*>(pSrc),
          reinterpret_cast<IppwComplex<C64>*>(pDst),
          numIters,
          pSpec,
          reinterpret_cast<const IppwComplex<C64>*>(pDlySrc),
          reinterpret_cast<IppwComplex<C64>*>(pDlyDst),
          pBuf);
    }

    // ippsIIRInit_BiQuad
    // *** UPDATED Template Declaration: Takes ppState** and pBuf ***
    template <typename T>
    inline IppStatus ippsIIRInit_BiQuad(
        IppwIIRState<T>** ppState,
        const T* pTaps,
        int numBq,
        const T* pDlyLine,
        Ipp8u* pBuf);

    // *** UPDATED Specializations: Match the 5-argument signature with
    // ppState** ***
    template <>
    inline IppStatus ippsIIRInit_BiQuad<F32>(
        IppwIIRState<F32>** ppState,
        const F32* pTaps,
        int numBq,
        const F32* pDlyLine,
        Ipp8u* pBuf)
    {
      // Call the actual IPP function which expects IppsIIRState_32f**
      return ippsIIRInit_BiQuad_32f(ppState, pTaps, numBq, pDlyLine, pBuf);
    }
    template <>
    inline IppStatus ippsIIRInit_BiQuad<F64>(
        IppwIIRState<F64>** ppState,
        const F64* pTaps,
        int numBq,
        const F64* pDlyLine,
        Ipp8u* pBuf)
    {
      // Call the actual IPP function which expects IppsIIRState_64f**
      return ippsIIRInit_BiQuad_64f(ppState, pTaps, numBq, pDlyLine, pBuf);
    }

    // ippsIIR (signature remains correct)
    template <typename T>
    inline IppStatus ippsIIR(
        const T* pSrc, T* pDst, int len, IppwIIRState<T>* pState);
    // ... specializations ...
    template <>
    inline IppStatus ippsIIR<F32>(
        const F32* pSrc, F32* pDst, int len, IppwIIRState<F32>* pState)
    {
      return ippsIIR_32f(pSrc, pDst, len, pState);
    }
    template <>
    inline IppStatus ippsIIR<F64>(
        const F64* pSrc, F64* pDst, int len, IppwIIRState<F64>* pState)
    {
      return ippsIIR_64f(pSrc, pDst, len, pState);
    }

    // ippsIIRSetDlyLine (signature remains correct)
    template <typename T>
    inline IppStatus ippsIIRSetDlyLine(
        IppwIIRState<T>* pState, const T* pDlyLine);
    // ... specializations ...
    template <>
    inline IppStatus ippsIIRSetDlyLine<F32>(
        IppwIIRState<F32>* pState, const F32* pDlyLine)
    {
      return ippsIIRSetDlyLine_32f(pState, pDlyLine);
    }
    template <>
    inline IppStatus ippsIIRSetDlyLine<F64>(
        IppwIIRState<F64>* pState, const F64* pDlyLine)
    {
      return ippsIIRSetDlyLine_64f(pState, pDlyLine);
    }

  }  // namespace ipp_dispatch

  // ... (Rest of the class definitions remain the same) ...
  template <typename T>
  class OneMKLFIRFilterPlanImpl final : public FIRFilterPlanImpl<T> {
    // ... members ...
  };

  template <typename T>  // T is real type here (F32, F64)
  class OneMKLIIRFilterPlanImpl final : public IIRFilterPlanImpl<T> {
    // ... members ...
   private:
    size_t num_sections_;
    size_t order_;
    std::vector<T> taps_interleaved_;
    IppwIIRState<T>* p_state_ = nullptr;  // Use the alias
    int state_size_bytes_ = 0;
  };

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_USE_ONEMKL
#endif  // OMNIDSP_ONEMKL_FILTER_HPP
