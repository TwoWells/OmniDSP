#ifndef OMNIDSP_DEFAULT_FFT_HPP
#define OMNIDSP_DEFAULT_FFT_HPP

// Include the internal backend interface which defines FFTPlanImpl/RFFTPlanImpl
#include "../interface/backend.hpp"
// Include public types needed (like complex types, Status)
#include <OmniDSP/core_types.hpp>  // For Status, F32, C32, F64, C64 etc.
#include <OmniDSP/fft.hpp>  // For fft_direction enum (though not directly used in Impl constructor)
#include <complex>
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::logic_error
#include <vector>
// Removed Highway includes for standard C++ version
// #include <hwy/aligned_allocator.h>

namespace OmniDSP::Default {

  // --- Default BackendType FFT Plan Implementations (Standard C++) ---
  // Inherit from internal Impl classes defined in ../interface/backend.hpp
  // Match names used in Backend factory methods (e.g.,
  // create_fft_plan_impl_c32)

  /**
   * @brief Default backend implementation for Complex-to-Complex FFT Plan
   * (Standard C++).
   * @tparam T_Complex Complex type (e.g., C32 or C64).
   */
  template <typename T_Complex>  // T_Complex will be C32 or C64
  class FFTPlanImpl final : public Abstract::FFTPlanImpl<T_Complex> {
    // Infer scalar type T from T_Complex (assuming T_Complex is
    // std::complex<T>)
    using T = typename T_Complex::value_type;

   public:
    /**
     * @brief Constructor.
     * @param n FFT length (must be power of 2).
     * @throws std::invalid_argument if n is not a positive power of 2.
     * @throws std::bad_alloc if memory allocation fails.
     */
    explicit FFTPlanImpl(size_t n);
    ~FFTPlanImpl() override = default;  // Use default destructor

    /**
     * @brief Executes the forward FFT.
     * @param input Input complex data.
     * @param output Output complex data.
     * @return Status code (Success or SizeMismatch).
     */
    // FIX: Added const qualifier to match base class
    OmniStatus fft(
        std::span<const T_Complex> input,
        std::span<T_Complex> output) const override;

    /**
     * @brief Executes the inverse FFT (output is complex, scaled by 1/N).
     * @param input Input complex data.
     * @param output Output complex data.
     * @return Status code (Success or SizeMismatch).
     */
    // FIX: Added const qualifier to match base class
    OmniStatus ifft(
        std::span<const T_Complex> input,
        std::span<T_Complex> output) const override;

    /**
     * @brief Gets the FFT length N.
     * @return size_t FFT length.
     */
    // This was already const, matching the base class
    size_t get_length() const override;

    // Disable copy/move operations
    FFTPlanImpl(const FFTPlanImpl&) = delete;
    FFTPlanImpl& operator=(const FFTPlanImpl&) = delete;
    FFTPlanImpl(FFTPlanImpl&&) = delete;
    FFTPlanImpl& operator=(FFTPlanImpl&&) = delete;

   private:
    size_t n_;  // FFT Length
    // Precomputed twiddle factors for both directions
    std::vector<T_Complex> forward_twiddles_;
    std::vector<T_Complex> inverse_twiddles_;
    // Standard C++ temporary buffer
    std::unique_ptr<T_Complex[]> temp_buffer_;
  };

  /**
   * @brief Default backend implementation for Real FFT Plan (Standard C++).
   * @tparam T_Real Real type (e.g., F32 or F64).
   */
  template <typename T_Real>  // T_Real will be F32 or F64
  class RFFTPlanImpl final : public Abstract::RFFTPlanImpl<T_Real> {
    // Define complex type corresponding to T_Real
    // Use the Utils::GetComplexType from core_types.hpp as seen in
    // interface/backend.hpp
    // *** UPDATED Namespace ***
    using T_Complex = Utils::GetComplexType<T_Real>;

   public:
    /**
     * @brief Constructor.
     * @param n Real FFT length (must be power of 2 and >= 2).
     * @throws std::invalid_argument if n is not power of 2 or < 2.
     * @throws std::bad_alloc if memory allocation fails.
     */
    explicit RFFTPlanImpl(size_t n);  // n is the real data length
    ~RFFTPlanImpl() override = default;

    /**
     * @brief Executes the forward Real FFT (Real input -> Complex output).
     * @param input Input real data (size N).
     * @param output Output complex data (size N/2 + 1).
     * @return Status code (Success or SizeMismatch).
     */
    // FIX: Added const qualifier to match base class
    OmniStatus rfft(std::span<const T_Real> input, std::span<T_Complex> output)
        const override;

    /**
     * @brief Executes the inverse Real FFT (Complex input -> Real output).
     * @param input Input complex data (size N/2 + 1).
     * @param output Output real data (size N).
     * @return Status code (Success or SizeMismatch).
     */
    // FIX: Added const qualifier to match base class
    OmniStatus irfft(std::span<const T_Complex> input, std::span<T_Real> output)
        const override;

    /**
     * @brief Gets the real FFT length N.
     * @return size_t Real FFT length.
     */
    // This was already const, matching the base class
    size_t get_length()
        const override;  // Returns the original real data length N

    // Disable copy/move operations
    RFFTPlanImpl(const RFFTPlanImpl&) = delete;
    RFFTPlanImpl& operator=(const RFFTPlanImpl&) = delete;
    RFFTPlanImpl(RFFTPlanImpl&&) = delete;
    RFFTPlanImpl& operator=(RFFTPlanImpl&&) = delete;

   private:
    size_t n_;       // Real data length
    size_t half_n_;  // Complex FFT length (N/2)
    // Use a single CFFT plan for the underlying N/2 transform (for standard C++
    // version) This needs to be const-correct if its methods are called from
    // const methods here
    FFTPlanImpl<T_Complex> cfft_plan_;  // Plan for N/2 complex FFT
    // Buffers needed for packing/unpacking stages (for standard C++ version)
    // These buffers are modified during execution, so they can remain non-const
    // members if allocated/managed correctly, OR they need to be passed as
    // mutable args/locals. Let's keep them as members for now, assuming execute
    // methods handle them.
    std::unique_ptr<T_Complex[]>
        packed_buffer_;  // Buffer for N/2 complex data (used in RFFT/IRFFT)
  };

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_FFT_HPP
