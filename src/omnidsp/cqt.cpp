/**
 * @file cqt.cpp
 * @brief Implements the CQTPlan class methods, forwarding calls to the
 * backend implementation (Pimpl pattern).
 */

#include "OmniDSP/cqt.h"  // Corresponding header

// Include the backend interface definition which declares CQTPlanImpl
#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move

#include "backend/backend.h"

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
template <typename T>
CQTPlan<T>::CQTPlan(std::unique_ptr<backend::CQTPlanImpl<T>> pimpl)
    : pimpl_(std::move(pimpl)) {
  if (!pimpl_) {
    // This should ideally be caught by the factory creating the Impl,
    // but check here as a safeguard.
    throw std::runtime_error(
        "CQTPlan cannot be created with a null implementation pointer.");
  }
}

/**
 * @brief Destructor.
 * The unique_ptr pimpl_ automatically deletes the managed implementation
 * object.
 */
template <typename T>
CQTPlan<T>::~CQTPlan() = default;

/**
 * @brief Move constructor.
 * Transfers ownership of the implementation pointer.
 */
template <typename T>
CQTPlan<T>::CQTPlan(CQTPlan&& other) noexcept = default;

/**
 * @brief Move assignment operator.
 * Transfers ownership of the implementation pointer.
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
 * Returns Status::InvalidOperation if the plan's implementation is missing.
 */
template <typename T>
[[nodiscard]] Status CQTPlan<T>::execute(std::span<const RealT<T>> input,
                                         std::span<ComplexT<T>> output) const {
  if (!pimpl_) {
    return Status::InvalidOperation;
  }
  // Forward the call to the actual backend implementation.
  return pimpl_->execute(input, output);
}

/**
 * @brief Gets the number of frequency bins by calling the backend
 * implementation.
 * @return The number of CQT bins.
 * @throws std::runtime_error if the plan's implementation is missing.
 */
template <typename T>
size_t CQTPlan<T>::get_num_bins() const {
  if (!pimpl_) {
    throw std::runtime_error(
        "Invalid CQTPlan instance: Implementation pointer is null in "
        "get_num_bins.");
  }
  return pimpl_->get_num_bins();
}

/**
 * @brief Calculates the number of output time frames by calling the backend
 * implementation.
 * @param input_length The length of the input signal in samples.
 * @return The expected number of time frames in the CQT output.
 * @throws std::runtime_error if the plan's implementation is missing.
 */
template <typename T>
size_t CQTPlan<T>::get_num_output_frames(size_t input_length) const {
  if (!pimpl_) {
    throw std::runtime_error(
        "Invalid CQTPlan instance: Implementation pointer is null in "
        "get_num_output_frames.");
  }
  return pimpl_->get_num_output_frames(input_length);
}

/**
 * @brief Gets the hop length by calling the backend implementation.
 * @return The hop length in samples.
 * @throws std::runtime_error if the plan's implementation is missing.
 */
template <typename T>
size_t CQTPlan<T>::get_hop_length() const {
  if (!pimpl_) {
    throw std::runtime_error(
        "Invalid CQTPlan instance: Implementation pointer is null in "
        "get_hop_length.");
  }
  return pimpl_->get_hop_length();
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) to ensure code
// generation for the public CQTPlan class.

template class OmniDSP::CQTPlan<float>;
template class OmniDSP::CQTPlan<double>;

}  // namespace OmniDSP
