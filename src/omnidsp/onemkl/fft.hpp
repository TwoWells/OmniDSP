/**
 * @file fft.hpp (OneMKL)
 * @brief Declares the oneMKL backend FFTPlanImpl and RFFTPlanImpl classes.
 */

#ifndef OMNIDSP_ONEMKL_FFT_HPP
#define OMNIDSP_ONEMKL_FFT_HPP

#include <mkl.h>  // Include MKL header for DFTI types

#include <OmniDSP/core_types.hpp>  // For Status, RealT, ComplexT etc.
#include <complex>
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr (if needed internally)
#include <span>     // For std::span
#include <vector>   // For std::vector (used internally)

#include "../interface/backend.hpp"  // Base PlanImpl interfaces

namespace OmniDSP::OneMKL {

  /**
   * @brief oneMKL implementation for complex-to-complex FFT plans using DFTI.
   * @tparam T Complex type (e.g., C32, C64).
   */
  template <typename T>  // T is complex type here (C32, C64)
  class FFTPlanImpl final : public Abstract::FFTPlanImpl<T> {
    static_assert(
        Utils::IsComplex_v<T>, "FFTPlanImpl requires a complex type.");
    using Real = typename T::value_type;  // Get underlying real type

   public:
    /**
     * @brief Constructor. Creates and configures the DFTI descriptor.
     * @param length The length of the FFT. Must be > 0.
     * @throws std::invalid_argument If length is 0 or precision is unsupported.
     * @throws std::runtime_error If DFTI descriptor creation or configuration
     * fails.
     */
    explicit FFTPlanImpl(size_t length);

    /**
     * @brief Destructor. Frees the DFTI descriptor.
     */
    ~FFTPlanImpl() override;

    // --- Deleted Copy/Move ---
    FFTPlanImpl(const FFTPlanImpl&) = delete;
    FFTPlanImpl& operator=(const FFTPlanImpl&) = delete;
    FFTPlanImpl(FFTPlanImpl&&) = delete;
    FFTPlanImpl& operator=(FFTPlanImpl&&) = delete;

    // --- Interface Methods ---
    [[nodiscard]] Status fft(
        std::span<const T> input, std::span<T> output) const override;
    [[nodiscard]] Status ifft(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_length() const override;

    // Public member to allow backend factory to check status after construction
    // (Alternatively, constructor could return expected or throw more specific
    // exceptions)
    MKL_LONG mkl_status_;

   private:
    size_t length_;
    DFTI_DESCRIPTOR_HANDLE descriptor_handle_ = nullptr;
    // Store MKL status if needed for error checking after construction
    // mutable MKL_LONG last_mkl_status_ = DFTI_NO_ERROR; // If execute needs to
    // report errors
  };

  /**
   * @brief oneMKL implementation for real-to-complex/complex-to-real FFT plans
   * using DFTI.
   * @tparam T Real type (e.g., F32, F64).
   */
  template <typename T>  // T is real type here (F32, F64)
  class OneMKLRFFTPlanImpl final : public Abstract::RFFTPlanImpl<T> {
    static_assert(
        !Utils::IsComplex_v<T>, "OneMKLRFFTPlanImpl requires a real type.");
    using Complex = Utils::GetComplexType<T>;  // Corresponding complex type

   public:
    /**
     * @brief Constructor. Creates and configures the DFTI descriptor for real
     * transforms.
     * @param length The length of the real FFT. Must be >= 2.
     * @throws std::invalid_argument If length < 2 or precision is unsupported.
     * @throws std::runtime_error If DFTI descriptor creation or configuration
     * fails.
     */
    explicit OneMKLRFFTPlanImpl(size_t length);

    /**
     * @brief Destructor. Frees the DFTI descriptor.
     */
    ~OneMKLRFFTPlanImpl() override;

    // --- Deleted Copy/Move ---
    OneMKLRFFTPlanImpl(const OneMKLRFFTPlanImpl&) = delete;
    OneMKLRFFTPlanImpl& operator=(const OneMKLRFFTPlanImpl&) = delete;
    OneMKLRFFTPlanImpl(OneMKLRFFTPlanImpl&&) = delete;
    OneMKLRFFTPlanImpl& operator=(OneMKLRFFTPlanImpl&&) = delete;

    // --- Interface Methods ---
    [[nodiscard]] Status rfft(
        std::span<const T> input, std::span<Complex> output) const override;
    [[nodiscard]] Status irfft(
        std::span<const Complex> input, std::span<T> output) const override;
    size_t get_length() const override;

    // Public member to allow backend factory to check status after construction
    MKL_LONG mkl_status_;

   private:
    size_t length_;
    DFTI_DESCRIPTOR_HANDLE descriptor_handle_ = nullptr;
    MKL_LONG input_strides_[1] = {0};   // For 1D transform
    MKL_LONG output_strides_[1] = {0};  // For 1D transform
    // mutable MKL_LONG last_mkl_status_ = DFTI_NO_ERROR;
  };

}  // namespace OmniDSP::OneMKL

#endif  // OMNIDSP_ONEMKL_FFT_HPP
