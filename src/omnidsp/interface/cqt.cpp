/**
 * @file cqt.cpp
 * @brief Implements the CQTPlan class methods, forwarding calls to the
 * backend implementation (Pimpl pattern).
 */

#include <OmniDSP/cqt.hpp>  // Corresponding header (defines CQTPlan and Complex alias)

// Include the backend interface definition which declares CQTPlanImpl
#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move

#include "backend.hpp"  // Defines backend::CQTPlanImpl

// Include core types for aliases like F32, F64 used in instantiations
#include <OmniDSP/core_types.hpp>

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // CQTPlan Method Definitions
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   * Takes ownership of the backend-specific implementation object.
   * @param pimpl A unique_ptr to the backend-specific CQTPlanImpl.
   * @throws std::runtime_error if pimpl is null.
   */
  template <typename T>  // T is REAL type
  CQTPlan<T>::CQTPlan(std::unique_ptr<backend::CQTPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "CQTPlan cannot be created with a null implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  CQTPlan<T>::~CQTPlan() = default;

  /**
   * @brief Move constructor.
   */
  template <typename T>
  CQTPlan<T>::CQTPlan(CQTPlan&& other) noexcept = default;

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  CQTPlan<T>& CQTPlan<T>::operator=(CQTPlan&& other) noexcept = default;

  /**
   * @brief Executes the Constant-Q Transform by calling the backend
   * implementation.
   * @param input A span representing the real input time-domain signal.
   * @param output A span representing the complex output buffer for the CQT
   * result.
   * @return Status::Success on success, or an error code on failure.
   */
  template <typename T>  // T is REAL type
  [[nodiscard]] Status CQTPlan<T>::execute(
      std::span<const T> input,
      std::span<Complex> output) const  // Use T and Complex alias
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Add size checks for robustness
    size_t expected_output_size
        = get_num_bins() * get_num_output_frames(input.size());
    if (output.size() < expected_output_size) {
      // Potentially log warning or return SizeMismatch immediately
      // std::cerr << "Warning: CQTPlan execute output span size (" <<
      // output.size()
      //           << ") is smaller than expected (" << expected_output_size <<
      //           ")." << std::endl;
      // return Status::SizeMismatch; // Or let the backend handle it
    }
    // Forward the call to the actual backend implementation.
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Gets the number of frequency bins by calling the backend
   * implementation.
   */
  template <typename T>
  size_t CQTPlan<T>::get_num_bins() const
  {
    if (!pimpl_) {
      throw std::runtime_error("Invalid CQTPlan instance in get_num_bins.");
    }
    return pimpl_->get_num_bins();
  }

  /**
   * @brief Calculates the number of output time frames by calling the backend
   * implementation.
   */
  template <typename T>
  size_t CQTPlan<T>::get_num_output_frames(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CQTPlan instance in get_num_output_frames.");
    }
    return pimpl_->get_num_output_frames(input_length);
  }

  /**
   * @brief Gets the hop length by calling the backend implementation.
   */
  template <typename T>
  size_t CQTPlan<T>::get_hop_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error("Invalid CQTPlan instance in get_hop_length.");
    }
    return pimpl_->get_hop_length();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate templates for common types (float, double) to ensure code
  // generation for the public CQTPlan class.

  template class CQTPlan<F32>;  // float
  template class CQTPlan<F64>;  // double

}  // namespace OmniDSP
