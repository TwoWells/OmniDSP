/**
 * @file fft.hpp
 * @brief Defines interfaces for FFT (Fast Fourier Transform) Plan objects.
 */

#ifndef OMNIDSP_FFT_H
#define OMNIDSP_FFT_H

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/core_types.hpp"
#include "OmniDSP/omnidsp_export.hpp"

// Forward declare backend Impl class
namespace OmniDSP::Abstract {
  template <typename T>
  class FFTPlanImpl;

  template <typename T>
  class RFFTPlanImpl;
}  // namespace OmniDSP::Abstract

namespace OmniDSP {

  class OmniDSP;  // Forward declare for friend

  /**
   * @brief Plan object for complex-to-complex FFTs.
   * @tparam T The underlying COMPLEX floating-point type (e.g., C32, C64).
   */
  template <typename T>  // T is COMPLEX type here (C32, C64)
  class OMNIDSP_EXPORT FFTPlan {
    // *** UPDATED Namespace ***
    static_assert(
        Utils::IsComplex_v<T>, "FFTPlan requires a complex type (C32 or C64).");
    // Only OmniDSP needs friend access if it calls the private constructor
    // directly
    friend class OmniDSP;
    // Removed friend declarations for Backend and Backend

   public:
    ~FFTPlan();
    FFTPlan(FFTPlan&& other) noexcept;
    FFTPlan& operator=(FFTPlan&& other) noexcept;
    FFTPlan(const FFTPlan&) = delete;
    FFTPlan& operator=(const FFTPlan&) = delete;

    [[nodiscard]] OmniStatus fft(
        std::span<const T> input, std::span<T> output) const;
    [[nodiscard]] OmniStatus ifft(
        std::span<const T> input, std::span<T> output) const;
    size_t get_length() const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @details This is the intended way for backends to construct the public
     * Plan handle.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public FFTPlan. Returns nullptr
     * if pimpl is null.
     */
    static std::unique_ptr<FFTPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::FFTPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;  // Or handle error appropriately
      }
      // Directly call the private constructor using 'new' and wrap in
      // unique_ptr This avoids the need for the MakeUniqueEnabler struct.
      return std::unique_ptr<FFTPlan<T>>(new FFTPlan<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called by create_from_impl. */
    explicit FFTPlan(std::unique_ptr<Abstract::FFTPlanImpl<T>> pimpl);

    // Removed the private static helper and MakeUniqueEnabler struct

    std::unique_ptr<Abstract::FFTPlanImpl<T>> pimpl_;
  };

  /**
   * @brief Plan object for real FFTs (RFFT/IRFFT).
   * @tparam T The underlying REAL floating-point type (e.g., F32, F64).
   */
  template <typename T>  // T is the REAL type here (F32, F64)
  class OMNIDSP_EXPORT RFFTPlan {
    // *** UPDATED Namespace ***
    static_assert(
        !Utils::IsComplex_v<T>, "RFFTPlan requires a real type (F32 or F64).");
    // *** UPDATED Namespace ***
    using Complex = Utils::GetComplexType<T>;
    // Only OmniDSP needs friend access if it calls the private constructor
    // directly
    friend class OmniDSP;
    // Removed friend declarations for Backend and Backend

   public:
    ~RFFTPlan();
    RFFTPlan(RFFTPlan&& other) noexcept;
    RFFTPlan& operator=(RFFTPlan&& other) noexcept;
    RFFTPlan(const RFFTPlan&) = delete;
    RFFTPlan& operator=(const RFFTPlan&) = delete;

    [[nodiscard]] OmniStatus rfft(
        std::span<const T> input, std::span<Complex> output) const;
    [[nodiscard]] OmniStatus irfft(
        std::span<const Complex> input, std::span<T> output) const;
    size_t get_length() const;

    /**
     * @brief Public static helper for factory methods to create instances.
     * @details This is the intended way for backends to construct the public
     * Plan handle.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     * @return A unique_ptr to the newly created public RFFTPlan. Returns
     * nullptr if pimpl is null.
     */
    static std::unique_ptr<RFFTPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::RFFTPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<RFFTPlan<T>>(new RFFTPlan<T>(std::move(pimpl)));
    }

   private:
    /** @brief Private constructor, called by create_from_impl. */
    explicit RFFTPlan(std::unique_ptr<Abstract::RFFTPlanImpl<T>> pimpl);

    // Removed the private static helper and MakeUniqueEnabler struct

    std::unique_ptr<Abstract::RFFTPlanImpl<T>> pimpl_;
  };

  // Definitions MUST be provided in a .cpp file

}  // namespace OmniDSP

#endif  // OMNIDSP_FFT_H
