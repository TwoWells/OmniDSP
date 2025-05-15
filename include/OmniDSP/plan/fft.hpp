/**
 * @file fft.hpp
 * @brief Defines interfaces for FFT (Fast Fourier Transform) Plan objects
 * under the OmniDSP::Plan namespace.
 */

#ifndef OMNIDSP_PLAN_FFT_HPP  // Conventionally, .hpp for C++ headers
#define OMNIDSP_PLAN_FFT_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>  // For std::move
#include <vector>   // Kept for potential transitive includes or future use

#include "OmniDSP/core_types.hpp"  // For OmniStatus, OmniExpected, F32, C32, Utils::IsComplex_v, Utils::GetComplexType
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT

// Forward declare Abstract Impl classes. These remain in the Abstract
// namespace.
namespace OmniDSP::Abstract {
  template <typename T>  // T is Complex for FFTPlanImpl (C32, C64)
  class FFTPlanImpl;     // Implementation for Plan::FFT
  template <typename T>  // T is Real for RFFTPlanImpl (F32, F64)
  class RFFTPlanImpl;    // Implementation for Plan::RFFT
}  // namespace OmniDSP::Abstract

namespace OmniDSP::Plan {

  /**
   * @brief Plan object for complex-to-complex FFTs.
   * @tparam T The underlying COMPLEX floating-point type (e.g., C32, C64).
   *
   * Formerly OmniDSP::FFTPlan.
   */
  template <typename T>  // T is COMPLEX type here (C32, C64)
  class FFT {  // Renamed from FFTPlan. OMNIDSP_EXPORT removed from class
               // template.
    static_assert(
        Utils::IsComplex_v<T>,
        "Plan::FFT requires a complex type (C32 or C64).");

    // friend class OmniDSP; // Removed: No longer a friend

   public:
    ~FFT();                                // Destructor
    FFT(FFT&& other) noexcept;             // Move constructor
    FFT& operator=(FFT&& other) noexcept;  // Move assignment
    FFT(const FFT&) = delete;              // Disable copy constructor
    FFT& operator=(const FFT&) = delete;   // Disable copy assignment

    /**
     * @brief Performs a forward FFT (complex-to-complex).
     * @param input The input complex data span.
     * @param output The output complex data span.
     * @return OmniStatus indicating success or failure.
     */
    [[nodiscard]] OmniStatus fft(
        std::span<const T> input, std::span<T> output) const;

    /**
     * @brief Performs an inverse FFT (complex-to-complex).
     * @param input The input complex data span.
     * @param output The output complex data span.
     * @return OmniStatus indicating success or failure.
     */
    [[nodiscard]] OmniStatus ifft(
        std::span<const T> input, std::span<T> output) const;

    /** @return The length of the FFT. */
    size_t get_length() const;

    /**
     * @brief (Internal) Creates a public FFT plan from a private
     * implementation.
     * @param pimpl A unique_ptr to the backend-specific abstract implementation
     * object.
     * @return A unique_ptr to the newly created OmniDSP::Plan::FFT. Returns
     * nullptr if pimpl is null.
     */
    static std::unique_ptr<FFT<T>> create_from_impl(  // Updated return type
        std::unique_ptr<Abstract::FFTPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      // Use private constructor for controlled creation
      return std::unique_ptr<FFT<T>>(new FFT<T>(std::move(pimpl)));
    }

   private:
    // Private constructor, called by create_from_impl.
    explicit FFT(std::unique_ptr<Abstract::FFTPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::FFTPlanImpl<T>>
        pimpl_;  // Pointer to implementation
  };

  /**
   * @brief Plan object for real-to-complex FFTs (RFFT) and complex-to-real
   * IFFTs (IRFFT).
   * @tparam T The underlying REAL floating-point type (e.g., F32, F64).
   *
   * Formerly OmniDSP::RFFTPlan.
   */
  template <typename T>  // T is the REAL type here (F32, F64)
  class RFFT {  // Renamed from RFFTPlan. OMNIDSP_EXPORT removed from class
                // template.
    static_assert(
        !Utils::IsComplex_v<T>,
        "Plan::RFFT requires a real type (F32 or F64).");

    // Define the corresponding complex type based on T
    using Complex = Utils::GetComplexType<T>;

    // friend class OmniDSP; // Removed: No longer a friend

   public:
    ~RFFT();                                 // Destructor
    RFFT(RFFT&& other) noexcept;             // Move constructor
    RFFT& operator=(RFFT&& other) noexcept;  // Move assignment
    RFFT(const RFFT&) = delete;              // Disable copy constructor
    RFFT& operator=(const RFFT&) = delete;   // Disable copy assignment

    /**
     * @brief Performs a forward real-to-complex FFT.
     * @param input The input real data span.
     * @param output The output complex data span (typically N/2 + 1 elements).
     * @return OmniStatus indicating success or failure.
     */
    [[nodiscard]] OmniStatus rfft(
        std::span<const T> input, std::span<Complex> output) const;

    /**
     * @brief Performs an inverse complex-to-real FFT.
     * @param input The input complex data span (typically N/2 + 1 elements).
     * @param output The output real data span.
     * @return OmniStatus indicating success or failure.
     */
    [[nodiscard]] OmniStatus irfft(
        std::span<const Complex> input, std::span<T> output) const;

    /** @return The length of the real FFT (length of the real signal). */
    size_t get_length() const;

    /**
     * @brief (Internal) Creates a public RFFT plan from a private
     * implementation.
     * @param pimpl A unique_ptr to the backend-specific abstract implementation
     * object.
     * @return A unique_ptr to the newly created OmniDSP::Plan::RFFT. Returns
     * nullptr if pimpl is null.
     */
    static std::unique_ptr<RFFT<T>> create_from_impl(  // Updated return type
        std::unique_ptr<Abstract::RFFTPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      // Use private constructor for controlled creation
      return std::unique_ptr<RFFT<T>>(new RFFT<T>(std::move(pimpl)));
    }

   private:
    // Private constructor, called by create_from_impl.
    explicit RFFT(std::unique_ptr<Abstract::RFFTPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::RFFTPlanImpl<T>>
        pimpl_;  // Pointer to implementation
  };

}  // namespace OmniDSP::Plan

#endif  // OMNIDSP_PLAN_FFT_HPP
