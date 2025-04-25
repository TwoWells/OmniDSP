/**
 * @file resample.h
 * @brief Defines the interface for Resample Plan objects and related
 * specifications.
 * @details Plan objects are created via OmniDSP::create_resample_plan using a
 * ResampleSpec and provide optimized execution for resampling signals between
 * specified input and output rates.
 */

#ifndef OMNIDSP_RESAMPLE_H
#define OMNIDSP_RESAMPLE_H

#include <cmath>      // For std::ceil (used in placeholder getter)
#include <cstddef>    // For size_t
#include <memory>     // For std::unique_ptr (Pimpl)
#include <span>       // For input/output views (requires C++20)
#include <stdexcept>  // For invalid_argument in ResampleSpec validation
#include <vector>     // For std::vector (potentially used internally)

#include "core_types.h"  // Core types like RealT, Status

namespace OmniDSP {

  // Forward declare the main OmniDSP class for friend declaration
  class OmniDSP;

  // Forward declarations for implementation classes (Pimpl idiom)
  namespace backend {
    template <typename T>
    class ResamplePlanImpl;
  }  // namespace backend

  /**
   * @brief Specification for creating a resampling plan.
   * @details Defines the input and output sample rates, and a quality parameter
   * that influences the underlying anti-aliasing/anti-imaging filter design.
   */
  struct ResampleSpec {
    double input_rate = 0.0;  ///< Input sample rate in Hz. Must be positive.
    double output_rate
        = 0.0;  ///< Desired output sample rate in Hz. Must be positive.
    int quality
        = 12;  ///< Quality factor (e.g., related to filter complexity/taps).
               ///< Higher values generally mean better quality but more
               ///< computation. Default is 12. Must be positive.
    // size_t max_input_size = 0; ///< Optional hint for max input block size
    // (potentially removed or handled differently)

    /**
     * @brief Validates the resampling specification parameters.
     * @return True if the parameters are valid, false otherwise.
     */
    bool validate() const
    {
      if (input_rate <= 0.0 || output_rate <= 0.0) {
        // std::cerr << "ResampleSpec Error: Input and output rates must be
        // positive." << std::endl;
        return false;
      }
      if (quality <= 0) {
        // std::cerr << "ResampleSpec Error: Quality factor must be positive."
        // << std::endl;
        return false;
      }
      // Add other checks if necessary (e.g., reasonable ratio?)
      return true;
    }
  };

  /**
   * @brief Plan object for executing signal resampling operations.
   * @details This class encapsulates the pre-computed data (e.g., polyphase
   * filter coefficients) and backend-specific context required for efficient
   * resampling. Instances are created via OmniDSP::create_resample_plan using a
   * ResampleSpec. This class is non-copyable but movable.
   * @tparam T The underlying floating-point type (e.g., float, double).
   * Typically used with RealT<T>.
   */
  template <typename T>
  class ResamplePlan {
    // Friend declaration allows OmniDSP factory methods to call the private
    // constructor
    friend class OmniDSP;

   public:
    /**
     * @brief Destructor. Cleans up implementation resources.
     * @details Defined in the source file to handle Pimpl destruction.
     */
    ~ResamplePlan();

    /**
     * @brief Move constructor.
     * @details Defined in the source file.
     */
    ResamplePlan(ResamplePlan&& other) noexcept;

    /**
     * @brief Move assignment operator.
     * @details Defined in the source file.
     */
    ResamplePlan& operator=(ResamplePlan&& other) noexcept;

    /** @brief Deleted copy constructor. ResamplePlan instances are
     * non-copyable.
     */
    ResamplePlan(const ResamplePlan&) = delete;
    /** @brief Deleted copy assignment operator. ResamplePlan instances are
     * non-copyable. */
    ResamplePlan& operator=(const ResamplePlan&) = delete;

    /**
     * @brief Executes the pre-planned resampling operation on an input signal.
     * @param input A span representing the input signal buffer.
     * @param output A span representing the output buffer for the resampled
     * signal. The required size can be estimated via get_output_length().
     * @return Status::Success on success, or an error code on failure.
     * @note The output span must be pre-allocated with sufficient size before
     * calling execute. The exact output size might depend on internal filter
     * delays and boundary handling.
     */
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;

    /**
     * @brief Resets the internal state of the resampler (e.g., filter delay
     * lines).
     * @details Call this when processing discontinuous blocks of data to avoid
     * artifacts.
     * @return Status::Success on success, or an error code if resetting fails.
     */
    Status reset();  // Added based on TODO refactoring plan

    /**
     * @brief Gets the input sample rate configured for this plan.
     * @return The input sample rate in Hz.
     */
    double get_input_rate() const;

    /**
     * @brief Gets the output sample rate configured for this plan.
     * @return The output sample rate in Hz.
     */
    double get_output_rate() const;

    /**
     * @brief Estimates the required output length for a given input length
     * based on the resampling ratio.
     * @details The actual output length might vary slightly due to filter
     * delays and implementation details. This provides a suitable size for
     * allocating the output buffer.
     * @param input_length The length of the input signal.
     * @return An estimated length for the output signal buffer.
     */
    size_t get_output_length(size_t input_length) const;

    // Consider adding get_quality() if needed

   private:
    /**
     * @brief Private constructor, called by OmniDSP factory methods.
     * @param pimpl A unique_ptr to the backend-specific implementation object.
     */
    explicit ResamplePlan(std::unique_ptr<backend::ResamplePlanImpl<T>> pimpl);

    /**
     * @brief Pointer to the implementation object (Pimpl idiom).
     */
    std::unique_ptr<backend::ResamplePlanImpl<T>> pimpl_;
  };

  // --- Template Implementations (Definitions) ---
  // Definitions for template class methods that depend on the Impl class
  // (constructor, destructor, move ops, execute, getters) MUST be provided
  // in src/omnidsp/resample.cpp where the ResamplePlanImpl<T> is fully defined.

}  // namespace OmniDSP

#endif  // OMNIDSP_RESAMPLE_H
