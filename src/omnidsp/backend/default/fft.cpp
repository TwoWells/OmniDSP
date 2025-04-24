/**
 * @file fft.cpp (default)
 * @brief Implements Default backend FFTPlanImpl and RFFTPlanImpl classes using
 * standard C++ and Google Highway for SIMD acceleration with dynamic dispatch.
 * @details Provides portable FFT implementations based on the Cooley-Tukey
 * algorithm, accelerated with SIMD where possible using Google Highway. This
 * includes the core FFT butterflies, RFFT unpacking, IRFFT packing, and the
 * final real copy in IRFFT. Dynamic dispatch selects the best SIMD instruction
 * set at runtime. Requires FFT lengths to be powers of two.
 */

// Define HWY_TARGET_INCLUDE before including Highway headers
#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE                                                     \
  "src/omnidsp/backend/default/fft.cpp"  // Path to this file
#endif

#include <algorithm>  // For std::reverse, std::swap, std::copy
#include <cmath>      // For sin, cos, log2, pow
#include <complex>
#include <iostream>  // For debug/error messages
#include <numbers>
#include <numeric>  // For std::iota
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.h"  // For Status, RealT, ComplexT etc.
#include "backend.h"  // Corresponding header for Default backend declarations
#include "hwy/contrib/complex/complex-inl.h"  // For complex operations
#include "hwy/contrib/math/math-inl.h"
#include "hwy/foreach_target.h"  // Must be first Highway include
#include "hwy/highway.h"

// Highway namespace alias within the compilation unit for the active target
HWY_BEFORE_NAMESPACE();
namespace OmniDSP {
  namespace backend {
    namespace HWY_NAMESPACE {  // Start Highway's target-specific namespace

      namespace hn
          = hwy;  // Alias for Highway types/functions within this namespace

      //--------------------------------------------------------------------------
      // Highway-Accelerated FFT Implementation (Core Logic)
      //--------------------------------------------------------------------------

      /**
       * @brief Computes one stage of the Cooley-Tukey FFT using Highway SIMD.
       * @details Performs the butterfly operations for a given stage length
       * (`len`). This function is intended to be called within the
       * HWY_NAMESPACE. Uses scalar twiddle factor loading for
       * simplicity/correctness, which could be a bottleneck for further
       * optimization.
       * @tparam T The underlying real floating-point type (float or double).
       * @param data Pointer to the complex data array (modified in-place).
       * @param N Total FFT length (must be power of two).
       * @param len Current stage length (must be power of two, 2 <= len <= N).
       * @param twiddles_base Pointer to the precomputed W_N^k twiddle factors
       * (k=0..N/2-1).
       * @param inverse True if computing inverse FFT (applies conjugate
       * twiddle).
       */
      template <typename T> HWY_NOINLINE void CooleyTukeyStage_HWY(
          std::complex<T>* HWY_RESTRICT data,
          size_t N,
          size_t len,
          const std::complex<T>* HWY_RESTRICT twiddles_base,
          bool inverse)
      {
        const hn::ScalableTag<T> d;
        using CplxV = hn::Vec2<decltype(d)>;

        const size_t half_len = len >> 1;
        const size_t twiddle_step_N = N / len;

        for (size_t i = 0; i < N; i += len) {
          std::complex<T>* HWY_RESTRICT p1 = data + i;
          std::complex<T>* HWY_RESTRICT p2 = data + i + half_len;

          size_t j = 0;
          // Vectorized loop using Highway
          for (; j + hn::Lanes(d) <= half_len; j += hn::Lanes(d)) {
            const size_t current_twiddle_idx = (j * twiddle_step_N);
            const CplxV u
                = hn::LoadInterleaved2(d, reinterpret_cast<const T*>(p1 + j));
            const CplxV v_un
                = hn::LoadInterleaved2(d, reinterpret_cast<const T*>(p2 + j));

            // --- Scalar twiddle loading ---
            std::complex<T> twiddle_scalar = twiddles_base[current_twiddle_idx];
            if (inverse) {
              twiddle_scalar = std::conj(twiddle_scalar);
            }
            const CplxV twiddle
                = hn::Set(d, twiddle_scalar.real(), twiddle_scalar.imag());

            const CplxV v = hn::ComplexMul(v_un, twiddle);
            const CplxV u_new = hn::Add(u, v);
            const CplxV v_new = hn::Sub(u, v);

            hn::StoreInterleaved2(u_new, d, reinterpret_cast<T*>(p1 + j));
            hn::StoreInterleaved2(v_new, d, reinterpret_cast<T*>(p2 + j));
          }

          // Scalar remainder loop
          for (; j < half_len; ++j) {
            size_t current_twiddle_idx = (j * twiddle_step_N);
            std::complex<T> twiddle = twiddles_base[current_twiddle_idx];
            if (inverse) {
              twiddle = std::conj(twiddle);
            }
            std::complex<T> u_s = p1[j];
            std::complex<T> v_s = p2[j] * twiddle;
            p1[j] = u_s + v_s;
            p2[j] = u_s - v_s;
          }
        }
      }

      /**
       * @brief Performs an in-place iterative Cooley-Tukey FFT using Highway
       * SIMD.
       * @details Requires the input length N to be a power of two. Uses Highway
       * for accelerating butterfly computations. This function lives inside the
       * HWY_NAMESPACE and is called by the exported wrappers.
       * @tparam T_Complex Complex floating-point type (e.g.,
       * std::complex<float>).
       * @param data The data span to transform (modified in-place).
       * @param inverse If true, computes the inverse FFT.
       * @param twiddles Precomputed twiddle factors (W_N^k for k=0 to N/2-1).
       * @param bit_reverse_indices Precomputed bit-reversal permutation
       * indices.
       */
      template <typename T_Complex> void CooleyTukeyFFT_HWY(
          std::span<T_Complex> data,
          bool inverse,
          const std::vector<T_Complex>& twiddles,
          const std::vector<size_t>& bit_reverse_indices)
      {
        size_t N = data.size();
        if (N == 0 || N == 1) return;

        // 1. Bit-reversal permutation (Scalar)
        for (size_t i = 0; i < N; ++i) {
          size_t reversed_i = bit_reverse_indices[i];
          if (i < reversed_i) {
            std::swap(data[i], data[reversed_i]);
          }
        }

        // 2. Iterative butterfly stages
        using T_Real = typename T_Complex::value_type;
        for (size_t len = 2; len <= N; len <<= 1) {
          CooleyTukeyStage_HWY<T_Real>(
              data.data(), N, len, twiddles.data(), inverse);
        }
      }

      /**
       * @brief Unpacks N/2 complex FFT results into N/2+1 RFFT format using
       * Highway.
       * @details Implements the unpacking formula using SIMD complex
       * arithmetic. This function lives inside the HWY_NAMESPACE.
       * @tparam T The underlying real floating-point type (float or double).
       * @param packed_fft_ptr Pointer to the N/2 complex FFT result (input).
       * @param WN_twiddles_ptr Pointer to precomputed W_N^k factors
       * (k=0..N/2-1).
       * @param output_ptr Pointer to the output array (size N/2+1).
       * @param N The original real FFT length (must be power of two >= 2).
       */
      template <typename T> HWY_NOINLINE void RFFT_Unpack_HWY(
          const std::complex<T>* HWY_RESTRICT packed_fft_ptr,
          const std::complex<T>* HWY_RESTRICT WN_twiddles_ptr,
          std::complex<T>* HWY_RESTRICT output_ptr,
          size_t N)
      {
        const hn::ScalableTag<T> d;
        using V = hn::Vec<decltype(d)>;
        using CplxV = hn::Vec2<decltype(d)>;
        const size_t N_over_2 = N / 2;

        // Handle DC and Nyquist separately (scalar)
        output_ptr[0] = std::complex<T>{
            packed_fft_ptr[0].real() + packed_fft_ptr[0].imag(), T{0.0}};
        output_ptr[N_over_2] = std::complex<T>{
            packed_fft_ptr[0].real() - packed_fft_ptr[0].imag(), T{0.0}};

        const CplxV v_half
            = hn::Set(d, T{0.5}, T{0.0});  // Vector containing 0.5
        const CplxV v_j
            = hn::Set(d, T{0.0}, T{1.0});  // Vector containing j (0+1i)

        size_t k = 1;
        // Vectorized loop from k=1 up to N/4
        for (; k + hn::Lanes(d) <= N / 4 + 1; k += hn::Lanes(d)) {
          // Load fft[k] and fft[N/2-k] blocks
          const CplxV fft_k = hn::LoadInterleaved2(
              d, reinterpret_cast<const T*>(packed_fft_ptr + k));
          // Need to load fft[N/2-k] in reverse order for SIMD. Use Gather or
          // scalar loop. Using scalar loading for simplicity here.
          std::vector<T> temp_N2mk_real(hn::Lanes(d)),
              temp_N2mk_imag(hn::Lanes(d));
          for (size_t lane = 0; lane < hn::Lanes(d); ++lane) {
            temp_N2mk_real[lane] = packed_fft_ptr[N_over_2 - (k + lane)].real();
            temp_N2mk_imag[lane] = packed_fft_ptr[N_over_2 - (k + lane)].imag();
          }
          const CplxV fft_N2_minus_k = hn::InterleaveLower(
              d,
              hn::LoadU(d, temp_N2mk_real.data()),
              hn::LoadU(d, temp_N2mk_imag.data()));

          // Load WN[k] block
          const CplxV WN_k = hn::LoadInterleaved2(
              d, reinterpret_cast<const T*>(WN_twiddles_ptr + k));

          // term1 = fft_k + conj(fft_N2_minus_k)
          const CplxV term1 = hn::Add(fft_k, hn::Conj(fft_N2_minus_k));
          // term2 = fft_k - conj(fft_N2_minus_k)
          const CplxV term2 = hn::Sub(fft_k, hn::Conj(fft_N2_minus_k));
          // term2_twiddle = -j * term2 * WN_k = j * term2 * (-WN_k)
          // Note: ComplexMul(j, X) = (-X.imag, X.real)
          CplxV neg_WN_k;
          neg_WN_k.val[0] = hn::Neg(WN_k.val[0]);
          neg_WN_k.val[1] = hn::Neg(WN_k.val[1]);
          CplxV term2_times_neg_WNk = hn::ComplexMul(term2, neg_WN_k);
          CplxV term2_twiddle;
          term2_twiddle.val[0]
              = hn::Neg(term2_times_neg_WNk.val[1]);          // -imag part
          term2_twiddle.val[1] = term2_times_neg_WNk.val[0];  // real part

          // output[k] = 0.5 * (term1 + term2_twiddle)
          CplxV out_k = hn::ComplexMul(v_half, hn::Add(term1, term2_twiddle));
          hn::StoreInterleaved2(out_k, d, reinterpret_cast<T*>(output_ptr + k));

          // output[N/2-k] = 0.5 * conj(term1 - term2_twiddle)
          // Only store if k != N/2-k (i.e., k != N/4)
          // This condition is tricky to handle efficiently in SIMD.
          // For simplicity, we might compute and store, then fix the middle
          // element later if N/4 is a multiple of lanes.
          CplxV out_N2mk
              = hn::ComplexMul(v_half, hn::Conj(hn::Sub(term1, term2_twiddle)));
          // Store requires scatter or scalar loop because indices N/2-k are
          // descending. Using scalar store for simplicity.
          for (size_t lane = 0; lane < hn::Lanes(d); ++lane) {
            if (k + lane
                < N_over_2 - (k + lane)) {  // Check if not the middle element
              output_ptr[N_over_2 - (k + lane)] = std::complex<T>{
                  hn::ExtractLane(out_N2mk.val[0], lane),
                  hn::ExtractLane(out_N2mk.val[1], lane)};
            }
          }
        }

        // Scalar remainder loop
        for (; k < N_over_2; ++k) {
          std::complex<T> fft_k = packed_fft_ptr[k];
          std::complex<T> fft_N2_minus_k = packed_fft_ptr[N_over_2 - k];
          std::complex<T> WN_k = WN_twiddles_ptr[k];

          std::complex<T> term1 = fft_k + std::conj(fft_N2_minus_k);
          std::complex<T> term2 = fft_k - std::conj(fft_N2_minus_k);
          std::complex<T> term2_twiddle
              = std::complex<T>{0.0, -1.0} * term2 * WN_k;

          output_ptr[k] = static_cast<T>(0.5) * (term1 + term2_twiddle);
          // Check k != N/2-k before assigning symmetric part (only needed if
          // N/4 was hit)
          if (k < N_over_2 - k) {
            output_ptr[N_over_2 - k]
                = static_cast<T>(0.5) * std::conj(term1 - term2_twiddle);
          }
        }
      }

      /**
       * @brief Reconstructs full N-point complex spectrum from N/2+1 RFFT input
       * using Highway.
       * @details Handles DC and Nyquist components separately, then uses SIMD
       * to fill k and N-k indices using conjugate symmetry. This function lives
       * inside the HWY_NAMESPACE.
       * @tparam T The underlying real floating-point type (float or double).
       * @param rfft_input_ptr Pointer to the N/2+1 complex RFFT input array.
       * @param full_spectrum_ptr Pointer to the N complex output array.
       * @param N The target full FFT length (must be power of two >= 2).
       */
      template <typename T> HWY_NOINLINE void IRFFT_Pack_HWY(
          const std::complex<T>* HWY_RESTRICT rfft_input_ptr,
          std::complex<T>* HWY_RESTRICT full_spectrum_ptr,
          size_t N)
      {
        const hn::ScalableTag<T> d;
        using CplxV = hn::Vec2<decltype(d)>;
        const size_t N_over_2 = N / 2;

        // Handle DC and Nyquist (scalar)
        full_spectrum_ptr[0] = rfft_input_ptr[0];
        full_spectrum_ptr[N_over_2] = std::complex<T>{
            rfft_input_ptr[N_over_2].real(), T{0.0}};  // Nyquist is real

        size_t k = 1;
        // Vectorized loop from k=1 to N/2-1
        for (; k + hn::Lanes(d) <= N_over_2; k += hn::Lanes(d)) {
          // Load input[k] block
          const CplxV input_k = hn::LoadInterleaved2(
              d, reinterpret_cast<const T*>(rfft_input_ptr + k));
          // Compute conjugate block
          const CplxV conj_input_k = hn::Conj(input_k);

          // Store input[k] at full_spectrum[k]
          hn::StoreInterleaved2(
              input_k, d, reinterpret_cast<T*>(full_spectrum_ptr + k));
          // Store conj(input[k]) at full_spectrum[N-k] (needs scatter or scalar
          // loop) Using scalar store for simplicity
          for (size_t lane = 0; lane < hn::Lanes(d); ++lane) {
            full_spectrum_ptr[N - (k + lane)] = std::complex<T>{
                hn::ExtractLane(conj_input_k.val[0], lane),
                hn::ExtractLane(conj_input_k.val[1], lane)};
          }
        }

        // Scalar remainder loop
        for (; k < N_over_2; ++k) {
          full_spectrum_ptr[k] = rfft_input_ptr[k];
          full_spectrum_ptr[N - k] = std::conj(rfft_input_ptr[k]);
        }
      }

      /**
       * @brief Copies the real part of a complex array to a real array using
       * Highway.
       * @details Extracts the real component using SIMD loads and stores.
       * This function lives inside the HWY_NAMESPACE.
       * @tparam T The underlying real floating-point type (float or double).
       * @param complex_input_ptr Pointer to the N complex input array.
       * @param real_output_ptr Pointer to the N real output array.
       * @param N The number of elements.
       */
      template <typename T> HWY_NOINLINE void IRFFT_RealCopy_HWY(
          const std::complex<T>* HWY_RESTRICT complex_input_ptr,
          T* HWY_RESTRICT real_output_ptr,
          size_t N)
      {
        const hn::ScalableTag<T> d;
        using V = hn::Vec<decltype(d)>;
        using CplxV = hn::Vec2<decltype(d)>;

        size_t i = 0;
        // Vectorized loop
        for (; i + hn::Lanes(d) <= N; i += hn::Lanes(d)) {
          CplxV complex_vec = hn::LoadInterleaved2(
              d, reinterpret_cast<const T*>(complex_input_ptr + i));
          // Extract real part (val[0])
          V real_vec = complex_vec.val[0];
          // Store real part
          hn::StoreU(
              real_vec,
              d,
              real_output_ptr
                  + i);  // Use StoreU for potentially unaligned output
        }

        // Scalar remainder loop
        for (; i < N; ++i) {
          real_output_ptr[i] = complex_input_ptr[i].real();
        }
      }

      // Explicit instantiations for exported functions
      void CFFT_F32_HWY(
          std::span<float_c> d,
          bool inv,
          const std::vector<float_c>& t,
          const std::vector<size_t>& bi)
      {
        CooleyTukeyFFT_HWY(d, inv, t, bi);
      }
      void CFFT_F64_HWY(
          std::span<double_c> d,
          bool inv,
          const std::vector<double_c>& t,
          const std::vector<size_t>& bi)
      {
        CooleyTukeyFFT_HWY(d, inv, t, bi);
      }
      void RFFT_Unpack_F32_HWY(
          const float_c* p, const float_c* w, float_c* o, size_t n)
      {
        RFFT_Unpack_HWY<float>(p, w, o, n);
      }
      void RFFT_Unpack_F64_HWY(
          const double_c* p, const double_c* w, double_c* o, size_t n)
      {
        RFFT_Unpack_HWY<double>(p, w, o, n);
      }
      void IRFFT_Pack_F32_HWY(const float_c* r, float_c* f, size_t n)
      {
        IRFFT_Pack_HWY<float>(r, f, n);
      }
      void IRFFT_Pack_F64_HWY(const double_c* r, double_c* f, size_t n)
      {
        IRFFT_Pack_HWY<double>(r, f, n);
      }
      void IRFFT_RealCopy_F32_HWY(const float_c* c, float* r, size_t n)
      {
        IRFFT_RealCopy_HWY<float>(c, r, n);
      }
      void IRFFT_RealCopy_F64_HWY(const double_c* c, double* r, size_t n)
      {
        IRFFT_RealCopy_HWY<double>(c, r, n);
      }

    }  // namespace HWY_NAMESPACE
  }  // namespace backend
}  // namespace OmniDSP
HWY_AFTER_NAMESPACE();

//==========================================================================
// Exported Wrapper Functions & Dispatch Logic (Compiled Once)
//==========================================================================
/**
 * @brief This block is compiled only once, regardless of the number of Highway
 * targets.
 * @details Contains exported functions for dynamic dispatch and helpers.
 * Headers needed by this block must be included again within the #if HWY_ONCE
 * scope.
 */
#if HWY_ONCE

#include <complex>
#include <numeric>  // For std::iota in DefaultRFFTPlanImpl if needed
#include <span>
#include <stdexcept>  // For Plan constructors
#include <vector>

#include "OmniDSP/core_types.h"
#include "backend.h"               // Include again for HWY_ONCE block
#include "hwy/dynamic_dispatch.h"  // Include again for HWY_ONCE block

namespace OmniDSP {
  namespace backend {

    // Define types for brevity
    using float_c = OmniDSP::ComplexT<float>;
    using double_c = OmniDSP::ComplexT<double>;
    using float_r = OmniDSP::RealT<float>;
    using double_r = OmniDSP::RealT<double>;

    // --- Exported Wrappers for CFFT/IFFT ---
    void CFFT_F32_Export(
        float_c* d, size_t N, bool inv, const float_c* t, const size_t* bi)
    {
      HWY_NAMESPACE::CFFT_F32_HWY({d, N}, inv, {t, N / 2}, {bi, N});
    }
    HWY_EXPORT(CFFT_F32_Export);
    void CFFT_F64_Export(
        double_c* d, size_t N, bool inv, const double_c* t, const size_t* bi)
    {
      HWY_NAMESPACE::CFFT_F64_HWY({d, N}, inv, {t, N / 2}, {bi, N});
    }
    HWY_EXPORT(CFFT_F64_Export);

    // --- Exported Wrappers for RFFT Unpacking ---
    void RFFT_Unpack_F32_Export(
        const float_c* p, const float_c* w, float_c* o, size_t n)
    {
      HWY_NAMESPACE::RFFT_Unpack_F32_HWY(p, w, o, n);
    }
    HWY_EXPORT(RFFT_Unpack_F32_Export);
    void RFFT_Unpack_F64_Export(
        const double_c* p, const double_c* w, double_c* o, size_t n)
    {
      HWY_NAMESPACE::RFFT_Unpack_F64_HWY(p, w, o, n);
    }
    HWY_EXPORT(RFFT_Unpack_F64_Export);

    // --- Exported Wrappers for IRFFT Packing ---
    void IRFFT_Pack_F32_Export(const float_c* r, float_c* f, size_t n)
    {
      HWY_NAMESPACE::IRFFT_Pack_F32_HWY(r, f, n);
    }
    HWY_EXPORT(IRFFT_Pack_F32_Export);
    void IRFFT_Pack_F64_Export(const double_c* r, double_c* f, size_t n)
    {
      HWY_NAMESPACE::IRFFT_Pack_F64_HWY(r, f, n);
    }
    HWY_EXPORT(IRFFT_Pack_F64_Export);

    // --- Exported Wrappers for IRFFT Real Copy ---
    void IRFFT_RealCopy_F32_Export(const float_c* c, float* r, size_t n)
    {
      HWY_NAMESPACE::IRFFT_RealCopy_F32_HWY(c, r, n);
    }
    HWY_EXPORT(IRFFT_RealCopy_F32_Export);
    void IRFFT_RealCopy_F64_Export(const double_c* c, double* r, size_t n)
    {
      HWY_NAMESPACE::IRFFT_RealCopy_F64_HWY(c, r, n);
    }
    HWY_EXPORT(IRFFT_RealCopy_F64_Export);

    //--------------------------------------------------------------------------
    // Dispatcher Functions (non-namespaced, called by Plan methods)
    //--------------------------------------------------------------------------

    /** @brief Dispatches complex FFT/IFFT (float). */
    Status DispatchComplexFFT(
        std::span<float_c> data,
        bool inverse,
        const std::vector<float_c>& twiddles,
        const std::vector<size_t>& bit_reverse_indices)
    {
      if (data.empty()) return Status::Success;
      auto fft_func = HWY_DYNAMIC_DISPATCH(CFFT_F32_Export);
      fft_func(
          data.data(),
          data.size(),
          inverse,
          twiddles.data(),
          bit_reverse_indices.data());
      return Status::Success;
    }
    /** @brief Dispatches complex FFT/IFFT (double). */
    Status DispatchComplexFFT(
        std::span<double_c> data,
        bool inverse,
        const std::vector<double_c>& twiddles,
        const std::vector<size_t>& bit_reverse_indices)
    {
      if (data.empty()) return Status::Success;
      auto fft_func = HWY_DYNAMIC_DISPATCH(CFFT_F64_Export);
      fft_func(
          data.data(),
          data.size(),
          inverse,
          twiddles.data(),
          bit_reverse_indices.data());
      return Status::Success;
    }

    /** @brief Dispatches RFFT unpacking step (float). */
    Status DispatchRFFTUnpack(
        std::span<const float_c> packed_fft,
        const std::vector<float_c>& WN_twiddles,
        std::span<float_c> output,
        size_t N)
    {
      if (packed_fft.empty()) return Status::Success;
      auto unpack_func = HWY_DYNAMIC_DISPATCH(RFFT_Unpack_F32_Export);
      unpack_func(packed_fft.data(), WN_twiddles.data(), output.data(), N);
      return Status::Success;
    }
    /** @brief Dispatches RFFT unpacking step (double). */
    Status DispatchRFFTUnpack(
        std::span<const double_c> packed_fft,
        const std::vector<double_c>& WN_twiddles,
        std::span<double_c> output,
        size_t N)
    {
      if (packed_fft.empty()) return Status::Success;
      auto unpack_func = HWY_DYNAMIC_DISPATCH(RFFT_Unpack_F64_Export);
      unpack_func(packed_fft.data(), WN_twiddles.data(), output.data(), N);
      return Status::Success;
    }

    /** @brief Dispatches IRFFT packing (spectrum reconstruction) step (float).
     */
    Status DispatchIRFFTPack(
        std::span<const float_c> rfft_input,
        std::span<float_c> full_spectrum,
        size_t N)
    {
      if (rfft_input.empty()) return Status::Success;
      auto pack_func = HWY_DYNAMIC_DISPATCH(IRFFT_Pack_F32_Export);
      pack_func(rfft_input.data(), full_spectrum.data(), N);
      return Status::Success;
    }
    /** @brief Dispatches IRFFT packing (spectrum reconstruction) step (double).
     */
    Status DispatchIRFFTPack(
        std::span<const double_c> rfft_input,
        std::span<double_c> full_spectrum,
        size_t N)
    {
      if (rfft_input.empty()) return Status::Success;
      auto pack_func = HWY_DYNAMIC_DISPATCH(IRFFT_Pack_F64_Export);
      pack_func(rfft_input.data(), full_spectrum.data(), N);
      return Status::Success;
    }

    /** @brief Dispatches IRFFT final real part copy step (float). */
    Status DispatchIRFFTRealCopy(
        std::span<const float_c> complex_input,
        std::span<float> real_output,
        size_t N)
    {
      if (complex_input.empty()) return Status::Success;
      auto copy_func = HWY_DYNAMIC_DISPATCH(IRFFT_RealCopy_F32_Export);
      copy_func(complex_input.data(), real_output.data(), N);
      return Status::Success;
    }
    /** @brief Dispatches IRFFT final real part copy step (double). */
    Status DispatchIRFFTRealCopy(
        std::span<const double_c> complex_input,
        std::span<double> real_output,
        size_t N)
    {
      if (complex_input.empty()) return Status::Success;
      auto copy_func = HWY_DYNAMIC_DISPATCH(IRFFT_RealCopy_F64_Export);
      copy_func(complex_input.data(), real_output.data(), N);
      return Status::Success;
    }

    //--------------------------------------------------------------------------
    // Helper Functions (Standard C++)
    //--------------------------------------------------------------------------
    inline bool is_power_of_two(size_t n)
    {
      return (n > 0) && ((n & (n - 1)) == 0);
    }
    inline size_t reverse_bits(size_t n, size_t num_bits)
    { /* ... implementation ... */
      size_t reversed_n = 0;
      for (size_t i = 0; i < num_bits; ++i) {
        if ((n >> i) & 1)
          reversed_n |= (static_cast<size_t>(1) << (num_bits - 1 - i));
      }
      return reversed_n;
    }

    //--------------------------------------------------------------------------
    // DefaultFFTPlanImpl Method Definitions (using Dynamic Dispatch)
    //--------------------------------------------------------------------------
    template <typename T>
    DefaultFFTPlanImpl<T>::DefaultFFTPlanImpl(size_t length) : length_(length)
    { /* ... constructor ... */
      if (length == 0) return;
      if (!is_power_of_two(length_))
        throw std::invalid_argument("FFT requires power-of-two length.");
      size_t num_bits
          = (length > 0)
                ? static_cast<size_t>(std::log2(static_cast<double>(length_)))
                : 0;
      bit_reverse_indices_.resize(length_);
      for (size_t i = 0; i < length_; ++i)
        bit_reverse_indices_[i] = reverse_bits(i, num_bits);
      twiddle_factors_.resize(length_ / 2);
      using ValueType = typename T::value_type;
      for (size_t k = 0; k < length_ / 2; ++k) {
        ValueType angle = -2.0 * std::numbers::pi * static_cast<ValueType>(k)
                          / static_cast<ValueType>(length_);
        twiddle_factors_[k] = T{std::cos(angle), std::sin(angle)};
      }
    }
    template <typename T> Status DefaultFFTPlanImpl<T>::fft(
        std::span<const T> input, std::span<T> output) const
    {
      if (input.size() != length_ || output.size() != length_)
        return Status::SizeMismatch;
      if (length_ == 0) return Status::Success;
      std::copy(input.begin(), input.end(), output.begin());
      return DispatchComplexFFT(
          output, false, twiddle_factors_, bit_reverse_indices_);
    }
    template <typename T> Status DefaultFFTPlanImpl<T>::ifft(
        std::span<const T> input, std::span<T> output) const
    {
      if (input.size() != length_ || output.size() != length_)
        return Status::SizeMismatch;
      if (length_ == 0) return Status::Success;
      std::copy(input.begin(), input.end(), output.begin());
      return DispatchComplexFFT(
          output, true, twiddle_factors_, bit_reverse_indices_);
    }
    template <typename T> size_t DefaultFFTPlanImpl<T>::get_length() const
    {
      return length_;
    }

    //--------------------------------------------------------------------------
    // DefaultRFFTPlanImpl Method Definitions (using Dynamic Dispatch)
    //--------------------------------------------------------------------------
    template <typename T>
    DefaultRFFTPlanImpl<T>::DefaultRFFTPlanImpl(size_t length) : length_(length)
    { /* ... constructor ... */
      if (length == 0) return;
      if (!is_power_of_two(length_))
        throw std::invalid_argument("RFFT requires power-of-two length.");
      if (length_ < 2 && length_ != 0)
        throw std::invalid_argument("RFFT length must be >= 2.");
      size_t N_over_2 = length_ / 2;
      size_t num_bits
          = (N_over_2 > 0)
                ? static_cast<size_t>(std::log2(static_cast<double>(N_over_2)))
                : 0;
      bit_reverse_indices_.resize(N_over_2);
      for (size_t i = 0; i < N_over_2; ++i)
        bit_reverse_indices_[i] = reverse_bits(i, num_bits);
      twiddle_factors_.resize(N_over_2 / 2);
      for (size_t k = 0; k < N_over_2 / 2; ++k) {
        RealT<T> angle = -2.0 * std::numbers::pi * static_cast<RealT<T>>(k)
                         / static_cast<RealT<T>>(N_over_2);
        twiddle_factors_[k] = ComplexT<T>{std::cos(angle), std::sin(angle)};
      }
    }

    template <typename T> Status DefaultRFFTPlanImpl<T>::rfft(
        std::span<const RealT<T>> input, std::span<ComplexT<T>> output) const
    {
      size_t N = length_;
      if (N == 0) return Status::Success;
      size_t N_over_2 = N / 2;
      size_t output_size_expected = N_over_2 + 1;
      if (input.size() != N || output.size() != output_size_expected)
        return Status::SizeMismatch;

      // 1. Pack (Scalar)
      std::vector<ComplexT<T>> packed_input(N_over_2);
      for (size_t k = 0; k < N_over_2; ++k)
        packed_input[k] = ComplexT<T>{input[2 * k], input[2 * k + 1]};

      // 2. Compute N/2 complex FFT (in-place, using Highway dispatch)
      Status status = DispatchComplexFFT(
          std::span<ComplexT<T>>(packed_input),
          false,
          twiddle_factors_,
          bit_reverse_indices_);
      if (status != Status::Success) return status;

      // 3. Unpack (using Highway dispatch)
      std::vector<ComplexT<T>> WN_twiddles(N_over_2);  // Need W_N not W_{N/2}
      for (size_t k = 0; k < N_over_2; ++k) {
        RealT<T> angle = -2.0 * std::numbers::pi * static_cast<RealT<T>>(k)
                         / static_cast<RealT<T>>(N);
        WN_twiddles[k] = ComplexT<T>{std::cos(angle), std::sin(angle)};
      }
      status = DispatchRFFTUnpack(
          std::span<const ComplexT<T>>(packed_input), WN_twiddles, output, N);
      return status;
    }

    template <typename T> Status DefaultRFFTPlanImpl<T>::irfft(
        std::span<const ComplexT<T>> input, std::span<RealT<T>> output) const
    {
      size_t N = length_;
      if (N == 0) return Status::Success;
      size_t N_over_2 = N / 2;
      size_t input_size_expected = N_over_2 + 1;
      if (input.size() != input_size_expected || output.size() != N)
        return Status::SizeMismatch;

      // 1. Reconstruct full spectrum (using Highway dispatch)
      std::vector<ComplexT<T>> full_spectrum(N);
      Status status
          = DispatchIRFFTPack(input, std::span<ComplexT<T>>(full_spectrum), N);
      if (status != Status::Success) return status;

      // 2. Perform N-point complex IFFT (using Highway dispatch)
      DefaultFFTPlanImpl<ComplexT<T>> temp_cfft_plan(N);
      status = DispatchComplexFFT(
          std::span<ComplexT<T>>(full_spectrum),
          true,
          temp_cfft_plan.twiddle_factors_,
          temp_cfft_plan.bit_reverse_indices_);
      if (status != Status::Success) return status;
      // Result is now in full_spectrum

      // 3. Copy real parts (using Highway dispatch)
      status = DispatchIRFFTRealCopy(
          std::span<const ComplexT<T>>(full_spectrum), output, N);
      return status;
    }
    template <typename T> size_t DefaultRFFTPlanImpl<T>::get_length() const
    {
      return length_;
    }

    //--------------------------------------------------------------------------
    // Explicit Template Instantiations
    //--------------------------------------------------------------------------
    template class OmniDSP::backend::DefaultFFTPlanImpl<float_c>;
    template class OmniDSP::backend::DefaultFFTPlanImpl<double_c>;
    template class OmniDSP::backend::DefaultRFFTPlanImpl<float>;
    template class OmniDSP::backend::DefaultRFFTPlanImpl<double>;

  }  // namespace backend
}  // namespace OmniDSP

#endif  // HWY_ONCE
