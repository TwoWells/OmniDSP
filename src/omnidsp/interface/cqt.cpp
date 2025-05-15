/**
 * @file cqt.cpp
 * @brief Implements the Processor::CQT class methods, forwarding calls to the
 * backend implementation (Pimpl pattern).
 */

#include <OmniDSP/processor/cqt.hpp>  // Corresponding header (defines CQTPlan and Complex alias)

// Include the backend interface definition which declares CQTProcessorImpl
#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move

#include "backend.hpp"  // Defines abstract::CQTPlanImpl

// Include core types for aliases like F32, F64 used in instantiations
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/omnidsp_export.hpp>  // For OMNIDSP_EXPORT

namespace OmniDSP::Processor {

  //--------------------------------------------------------------------------
  // Processor::CQT Method Definitions
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   * Takes ownership of the backend-specific implementation object.
   * @param pimpl A unique_ptr to the backend-specific CQTProcessorImpl.
   * @throws std::runtime_error if pimpl is null.
   */
  template <typename T>  // T is REAL type
  CQT<T>::CQT(std::unique_ptr<Abstract::CQTProcessorImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Processor::CQT cannot be created with a null implementation "
          "pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  CQT<T>::~CQT() = default;

  /**
   * @brief Move constructor.
   */
  template <typename T>
  CQT<T>::CQT(CQT&& other) noexcept = default;

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  CQT<T>& CQT<T>::operator=(CQT&& other) noexcept = default;

  /**
   * @brief Executes the Constant-Q Transform by calling the backend
   * implementation.
   * @param input A span representing the real input time-domain signal.
   * @param output A span representing the complex output buffer for the CQT
   * result.
   * @return Status::Success on success, or an error code on failure.
   */
  template <typename T>  // T is REAL type
  [[nodiscard]] OmniStatus CQT<T>::execute(
      std::span<const T> input, std::span<Complex> output) const
  {  // Use T and Complex alias
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    // Add size checks for robustness (optional, could be in Impl)
    size_t expected_output_size
        = get_num_bins() * get_num_output_frames(input.size());
    if (output.size() < expected_output_size
        && expected_output_size
               > 0) {  // Check if expected_output_size is non-zero
      // spdlog or other logging mechanism might be useful here.
      // For now, rely on backend Impl to potentially return SizeMismatch.
      // Or, return Status::SizeMismatch directly:
      // return Status::SizeMismatch;
    }
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Gets the number of frequency bins by calling the backend
   * implementation.
   */
  template <typename T>
  size_t CQT<T>::get_num_bins() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Processor::CQT instance in get_num_bins.");
    }
    return pimpl_->get_num_bins();
  }

  /**
   * @brief Calculates the number of output time frames by calling the backend
   * implementation.
   */
  template <typename T>
  size_t CQT<T>::get_num_output_frames(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Processor::CQT instance in get_num_output_frames.");
    }
    return pimpl_->get_num_output_frames(input_length);
  }

  /**
   * @brief Gets the hop length by calling the backend implementation.
   */
  template <typename T>
  size_t CQT<T>::get_hop_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Processor::CQT instance in get_hop_length.");
    }
    return pimpl_->get_hop_length();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations for Processor::CQT class
  //--------------------------------------------------------------------------
  // This ensures that the code for Processor::CQT<F32> and Processor::CQT<F64>
  // is generated in this translation unit. The OMNIDSP_EXPORT macro will handle
  // visibility if this is part of a shared library.

  template class OMNIDSP_EXPORT CQT<F32>;  // float
  template class OMNIDSP_EXPORT CQT<F64>;  // double

}  // namespace OmniDSP::Processor
