/**
 * @file fft.h
 * @brief Defines interfaces for FFT (Fast Fourier Transform) Plan objects.
 * @details Plan objects are created via OmniDSP factory methods (e.g.,
 * OmniDSP::create_fft_plan) and provide optimized execution of FFTs for
 * specific configurations and backends.
 */

#ifndef OMNIDSP_FFT_H
#define OMNIDSP_FFT_H

#include <complex>  // For std::complex
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr (Pimpl)
#include <span>     // For input/output views (requires C++20)
#include <vector>   // For std::vector (potentially used internally)

#include "core_types.h"  // Core types like RealT, ComplexT, Status

// Include the generated export header
#include "OmniDSP/omnidsp_export.h"

namespace OmniDSP {

  // Forward declare the main OmniDSP class for friend declaration
  class OmniDSP;

  // Forward declarations for implementation classes (Pimpl idiom)
  namespace backend {
    template <typename T>
    class FFTPlanImpl;
    template <typename T>
    class RFFTPlanImpl;
  }  // namespace backend

  /**
   * @brief Plan object for executing complex-to-complex Fast Fourier Transforms
   * (FFT/IFFT).
   * @details This class encapsulates the pre-computed data and backend-specific
   * context required for efficient FFT execution. Instances are created via
   * OmniDSP::create_fft_plan. This class is non-copyable but movable.
   * @tparam T The underlying floating-point type (e.g., float, double).
   */
  template <typename T>
  class OMNIDSP_EXPORT FFTPlan {
    // Friend declaration allows OmniDSP factory methods to call the private
    // constructor
    friend class OmniDSP;

   public:
    /**
     * @brief Destructor. Cleans up implementation resources.
     * @details Defined in the source file to handle Pimpl destruction.
     */
    ~FFTPlan();

    /**
     * @brief Move constructor.
     * @details Defined in the source file.
     */
    FFTPlan(FFTPlan&& other) noexcept;

    /**
     * @brief Move assignment operator.
     * @details Defined in the source file.
     */
    FFTPlan& operator=(FFTPlan&& other) noexcept;

    /** @brief Deleted copy constructor. FFTPlan instances are non-copyable. */
    FFTPlan(const FFTPlan&) = delete;
    /** @brief Deleted copy assignment operator. FFTPlan instances are
     * non-copyable. */
    FFTPlan& operator=(const FFTPlan&) = delete;

    /**
     * @brief Executes the forward complex-to-complex FFT.
     * @param input A span representing the complex input buffer of size
     * get_length().
     * @param output A span representing the complex output buffer of size
     * get_length().
     * @return Status::Success on success, or an error code on failure.
     * @note Input and output spans must have size equal to get_length().
     * @note Depending on the backend, execution might be in-place if input ==
     * output. Check backend specifics.
     */
    [[nodiscard]] Status fft(
        std::span<const ComplexT<T>> input,
        std::span<ComplexT<T>> output) const;

    /**
     * @brief Executes the inverse complex-to-complex FFT (IFFT).
     * @param input A span representing the complex input buffer of size
     * get_length().
     * @param output A span representing the complex output buffer of size
     * get_length().
     * @return Status::Success on success, or an error code on failure.
     * @note Input and output spans must have size equal to get_length().
     * @note The output is typically unnormalized; multiply by (1.0 /
     * get_length()) if needed.
     * @note Depending on the backend, execution might be in-place if input ==
     * output. Check backend specifics.
     */
    [[nodiscard]] Status ifft(
        std::span<const ComplexT<T>> input,
        std::span<ComplexT<T>> output) const;

    /**
     * @brief Gets the length (number of complex points) of the FFT transform
     * this plan handles.
     * @return The length of the FFT.
     */
    size_t get_length() const;

   private:
    /**
     * @brief Private constructor, called by OmniDSP factory methods.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     */
    explicit FFTPlan(std::unique_ptr<backend::FFTPlanImpl<T>> pimpl);

    /**
     * @brief Pointer to the implementation object (Pimpl idiom).
     */
    std::unique_ptr<backend::FFTPlanImpl<T>> pimpl_;
  };

  /**
   * @brief Plan object for executing real-to-complex and complex-to-real FFTs
   * (RFFT/IRFFT).
   * @details This class encapsulates the pre-computed data and backend-specific
   * context required for efficient real FFT execution, taking advantage of
   * conjugate symmetry. Instances are created via OmniDSP::create_rfft_plan.
   * This class is non-copyable but movable.
   * @tparam T The underlying floating-point type (e.g., float, double).
   */
  template <typename T>
  class OMNIDSP_EXPORT RFFTPlan {
    // Friend declaration allows OmniDSP factory methods to call the private
    // constructor
    friend class OmniDSP;

   public:
    /**
     * @brief Destructor. Cleans up implementation resources.
     * @details Defined in the source file to handle Pimpl destruction.
     */
    ~RFFTPlan();

    /**
     * @brief Move constructor.
     * @details Defined in the source file.
     */
    RFFTPlan(RFFTPlan&& other) noexcept;

    /**
     * @brief Move assignment operator.
     * @details Defined in the source file.
     */
    RFFTPlan& operator=(RFFTPlan&& other) noexcept;

    /** @brief Deleted copy constructor. RFFTPlan instances are non-copyable. */
    RFFTPlan(const RFFTPlan&) = delete;
    /** @brief Deleted copy assignment operator. RFFTPlan instances are
     * non-copyable. */
    RFFTPlan& operator=(const RFFTPlan&) = delete;

    /**
     * @brief Executes the forward real-to-complex FFT (RFFT).
     * @param input A span representing the real input buffer of size
     * get_length().
     * @param output A span representing the complex output buffer. Its size
     * must be (get_length() / 2) + 1.
     * @return Status::Success on success, or an error code on failure.
     * @note Input span must have size get_length().
     * @note Output span must have size (get_length() / 2 + 1). It stores the
     * non-redundant complex coefficients.
     */
    [[nodiscard]] Status rfft(
        std::span<const RealT<T>> input, std::span<ComplexT<T>> output) const;

    /**
     * @brief Executes the inverse complex-to-real FFT (IRFFT).
     * @param input A span representing the complex input buffer of size
     * (get_length() / 2) + 1.
     * @param output A span representing the real output buffer of size
     * get_length().
     * @return Status::Success on success, or an error code on failure.
     * @note Input span must have size (get_length() / 2 + 1), containing the
     * non-redundant complex coefficients.
     * @note Output span must have size get_length().
     * @note The output is typically unnormalized; multiply by (1.0 /
     * get_length()) if needed.
     */
    [[nodiscard]] Status irfft(
        std::span<const ComplexT<T>> input, std::span<RealT<T>> output) const;

    /**
     * @brief Gets the logical length (number of real points) of the FFT
     * transform this plan handles.
     * @return The length N of the transform.
     */
    size_t get_length() const;

   private:
    /**
     * @brief Private constructor, called by OmniDSP factory methods.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     */
    explicit RFFTPlan(std::unique_ptr<backend::RFFTPlanImpl<T>> pimpl);

    /**
     * @brief Pointer to the implementation object (Pimpl idiom).
     */
    std::unique_ptr<backend::RFFTPlanImpl<T>> pimpl_;
  };

  // --- Template Implementations (Definitions) ---
  // Template classes often require definitions to be available in the header
  // or an included "-inl.h" file. Since Pimpl requires the Impl class to be
  // defined only in the .cpp file, we only define simple getters here.
  // The constructor, destructor, move ops, and execution methods MUST be
  // defined in a .cpp file where the Impl class is fully defined.

}  // namespace OmniDSP

#endif  // OMNIDSP_FFT_H
