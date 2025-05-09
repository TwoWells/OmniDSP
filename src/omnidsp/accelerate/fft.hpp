/**
 * @file fft.hpp (Accelerate)
 * @brief Declares the Accelerate backend FFTPlanImpl and RFFTPlanImpl classes
 * using vDSP DFT.
 */

#ifndef OMNIDSP_ACCELERATE_FFT_HPP
#define OMNIDSP_ACCELERATE_FFT_HPP

#include <OmniDSP/omnidsp_config.hpp>
#ifdef OMNIDSP_INTERNAL_USE_ACCELERATE

#include <Accelerate/Accelerate.h>  // Specifically vDSP.h is needed

#include <OmniDSP/core_types.hpp>  // For Status, RealT, ComplexT, C32, C64, F32, F64 etc.
#include <complex>
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr
#include <span>     // For std::span

#include "../interface/backend.hpp"  // Base PlanImpl interfaces

namespace OmniDSP::Accelerate {

  // --- Constants ---
  // Maximum length supported by vDSP_DFT_Interleaved_CreateSetup (for the
  // Length parameter)
  static constexpr size_t MAX_VDSP_DFT_INTERLEAVED_SETUP_LENGTH = 4096;

  /**
   * @brief Accelerate implementation for complex-to-complex FFT plans using
   * vDSP DFT.
   * @tparam T Complex type (C32 or C64).
   */
  template <typename T>  // T is complex type here (C32, C64)
  class FFTPlanImpl final : public Abstract::FFTPlanImpl<T> {
    static_assert(
        Utils::IsComplex_v<T>, "FFTPlanImpl requires a complex type.");
    using Real = typename T::value_type;

    using vDSP_DFT_Setup_Type = typename std::conditional<
        std::is_same_v<Real, float>,
        vDSP_DFT_Interleaved_Setup,
        vDSP_DFT_Interleaved_SetupD>::type;

    using vDSP_Complex_Type = typename std::conditional<
        std::is_same_v<Real, float>,
        DSPComplex,
        DSPDoubleComplex>::type;

   public:
    explicit FFTPlanImpl(size_t length);
    ~FFTPlanImpl() override;

    FFTPlanImpl(const FFTPlanImpl&) = delete;
    FFTPlanImpl& operator=(const FFTPlanImpl&) = delete;
    FFTPlanImpl(FFTPlanImpl&&) = delete;
    FFTPlanImpl& operator=(FFTPlanImpl&&) = delete;

    [[nodiscard]] Status fft(
        std::span<const T> input, std::span<T> output) const override;
    [[nodiscard]] Status ifft(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_length() const override;

    // *** Static helper to check length validity ***
    [[nodiscard]] static bool is_vdsp_dft_supported_length(size_t N_complex);

   private:
    size_t length_;  // Number of complex elements (N)
    vDSP_DFT_Setup_Type fft_setup_forward_ = nullptr;
    vDSP_DFT_Setup_Type fft_setup_inverse_ = nullptr;
  };

  /**
   * @brief Accelerate implementation for real FFT plans using vDSP DFT.
   * @tparam T Real type (F32 or F64).
   */
  template <typename T>  // T is real type here (F32, F64)
  class RFFTPlanImpl final : public Abstract::RFFTPlanImpl<T> {
    static_assert(!Utils::IsComplex_v<T>, "RFFTPlanImpl requires a real type.");
    using Complex = Utils::GetComplexType<T>;

    using vDSP_DFT_Setup_Type = typename std::conditional<
        std::is_same_v<T, float>,
        vDSP_DFT_Interleaved_Setup,
        vDSP_DFT_Interleaved_SetupD>::type;

    using vDSP_Complex_Type = typename std::conditional<
        std::is_same_v<T, float>,
        DSPComplex,
        DSPDoubleComplex>::type;

   public:
    explicit RFFTPlanImpl(size_t length);
    ~RFFTPlanImpl() override;

    RFFTPlanImpl(const RFFTPlanImpl&) = delete;
    RFFTPlanImpl& operator=(const RFFTPlanImpl&) = delete;
    RFFTPlanImpl(RFFTPlanImpl&&) = delete;
    RFFTPlanImpl& operator=(RFFTPlanImpl&&) = delete;

    [[nodiscard]] Status rfft(
        std::span<const T> input, std::span<Complex> output) const override;
    [[nodiscard]] Status irfft(
        std::span<const Complex> input, std::span<T> output) const override;
    size_t get_length() const override;

    // *** Static helper to check length validity ***
    [[nodiscard]] static bool is_vdsp_dft_supported_length(size_t N_real);

   private:
    size_t length_;  // Number of real elements (N)
    vDSP_DFT_Setup_Type fft_setup_forward_ = nullptr;
    vDSP_DFT_Setup_Type fft_setup_inverse_ = nullptr;
    mutable std::vector<T> temp_packed_buffer_;
  };

}  // namespace OmniDSP::Accelerate

#endif  // OMNIDSP_INTERNAL_USE_ACCELERATE

#endif  // OMNIDSP_ACCELERATE_FFT_HPP
