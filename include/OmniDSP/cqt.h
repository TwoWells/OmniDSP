/**
 * @file cqt.h
 * @brief Defines the interface for Constant-Q Transform (CQT) Plan objects.
 * @details Plan objects are created via OmniDSP::create_cqt_plan and provide
 * optimized execution of CQT for specific configurations and backends. Now uses
 * Pimpl.
 */

#ifndef OMNIDSP_CQT_H
#define OMNIDSP_CQT_H

#include <complex>  // For std::complex
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr (Pimpl)
#include <span>     // For input/output views (requires C++20)
#include <vector>   // For std::vector (potentially used internally)

#include "core_types.h"  // Core types like RealT, ComplexT, Status
#include "window.h"      // Include WindowSpec for potential use in CQT

// Include the generated export header
#include "OmniDSP/omnidsp_export.h"

namespace OmniDSP {

  // Forward declare the main OmniDSP class (needed for friend declaration in
  // factory)
  class OmniDSP;

  // Forward declarations for implementation classes (Pimpl idiom)
  namespace backend {
    template <typename T>
    class CQTPlanImpl;  // Forward declare the Impl interface
  }  // namespace backend

  /**
   * @brief Plan object for executing Constant-Q Transforms (CQT).
   * @details This class encapsulates the pre-computed data (like CQT kernels)
   * and backend-specific context required for efficient CQT execution. Now uses
   * Pimpl. Instances are created via OmniDSP::create_cqt_plan. This class is
   * non-copyable but movable.
   * @tparam T The underlying real floating-point type (e.g., float, double).
   */
  template <typename T>  // T is real type here
  class OMNIDSP_EXPORT CQTPlan {
    // Friend declaration allows OmniDSP factory methods to call the private
    // constructor
    friend class OmniDSP;
    // Allow backend implementations to call the private constructor too? Maybe
    // not needed. friend class backend::DefaultOmniDSPImpl; // Example

   public:
    /**
     * @brief Destructor. Cleans up implementation resources.
     * @details Defined in the source file to handle Pimpl destruction.
     */
    ~CQTPlan();

    /**
     * @brief Move constructor.
     * @details Defined in the source file.
     */
    CQTPlan(CQTPlan&& other) noexcept;

    /**
     * @brief Move assignment operator.
     * @details Defined in the source file.
     */
    CQTPlan& operator=(CQTPlan&& other) noexcept;

    /** @brief Deleted copy constructor. CQTPlan instances are non-copyable. */
    CQTPlan(const CQTPlan&) = delete;
    /** @brief Deleted copy assignment operator. CQTPlan instances are
     * non-copyable. */
    CQTPlan& operator=(const CQTPlan&) = delete;

    /**
     * @brief Executes the Constant-Q Transform on an input signal.
     * @param input A span representing the real input time-domain signal.
     * @param output A span representing the complex output buffer for the CQT
     * result (time-frequency). The required size is get_num_bins() *
     * get_num_output_frames(input.size()). The layout is typically (num_bins x
     * num_frames), stored row-major or column-major depending on the
     * implementation.
     * @return Status::Success on success, or an error code on failure.
     * @note The output span must be pre-allocated with sufficient size before
     * calling execute.
     */
    [[nodiscard]] Status execute(
        std::span<const RealT<T>> input, std::span<ComplexT<T>> output) const;

    /**
     * @brief Gets the number of frequency bins configured for this CQT plan.
     * @return The number of CQT bins.
     */
    size_t get_num_bins() const;

    /**
     * @brief Calculates the number of output time frames for a given input
     * signal length.
     * @details This depends on the internal hop length used by the CQT
     * implementation.
     * @param input_length The length of the input signal in samples.
     * @return The expected number of time frames in the CQT output.
     */
    size_t get_num_output_frames(size_t input_length) const;

    /**
     * @brief Gets the hop length (in samples) used between consecutive CQT
     * frames.
     * @return The hop length in samples.
     */
    size_t get_hop_length() const;

    // Add other potential getters as needed: get_sample_rate(), get_min_freq(),
    // get_max_freq(), etc. by forwarding to pimpl_

   private:
    /**
     * @brief Private constructor, called by OmniDSP factory methods.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     */
    explicit CQTPlan(
        std::unique_ptr<backend::CQTPlanImpl<T>> pimpl);  // Takes Impl ptr

    /**
     * @brief Pointer to the implementation object (Pimpl idiom).
     */
    std::unique_ptr<backend::CQTPlanImpl<T>> pimpl_;  // Pimpl member

    // Remove direct members that are now handled by Impl
    // const OmniDSP* owner_dsp_ = nullptr; // Removed
    // RealT<T> sample_rate_ = 0.0; // Removed (or get from pimpl_)
    // size_t num_bins_ = 0; // Removed (get from pimpl_)
    // size_t hop_length_ = 0; // Removed (get from pimpl_)
    // std::unique_ptr<FFTPlan<ComplexT<T>>> fft_plan_; // Removed (managed by
    // Impl) std::vector<std::vector<ComplexT<T>>> kernels_; // Removed (managed
    // by Impl)
  };

  // --- Template Implementations (Definitions) ---
  // Definitions for template class methods that depend on the Impl class
  // (constructor, destructor, move ops, execute, getters) MUST be provided
  // in src/omnidsp/cqt.cpp where the CQTPlanImpl<T> is fully defined.

  // Example placeholder getters (real definitions MUST be in .cpp):
  template <typename T>
  size_t CQTPlan<T>::get_num_bins() const
  {
    // return pimpl_ ? pimpl_->get_num_bins() : 0; // Requires Impl definition
    return 0;  // Placeholder
  }

  template <typename T>
  size_t CQTPlan<T>::get_num_output_frames(size_t /*input_length*/) const
  {
    // return pimpl_ ? pimpl_->get_num_output_frames(input_length) : 0; //
    // Requires Impl definition
    return 0;  // Placeholder
  }

  template <typename T>
  size_t CQTPlan<T>::get_hop_length() const
  {
    // return pimpl_ ? pimpl_->get_hop_length() : 0; // Requires Impl definition
    return 0;  // Placeholder
  }

}  // namespace OmniDSP

#endif  // OMNIDSP_CQT_H
