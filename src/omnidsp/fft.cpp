/**
 * @file fft.cpp
 * @brief Implementation file for FFT functions and plans.
 * Delegates actual computation to backend implementations via PIMPL.
 */
#include "OmniDSP/fft.h"  // Public header declarations

#include <complex>
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <vector>

#include "backend/backend_impl.h"  // Defines Backend::createFFTPlan etc. and PlanImpl interfaces

namespace OmniDSP {

//--------------------------------------------------------------------------
// FFTPlan (Complex-to-Complex)
//--------------------------------------------------------------------------

/**
 * @brief Constructs FFTPlan, creating the backend implementation.
 * Stores the plan size.
 */
template <typename T>
FFTPlan<T>::FFTPlan(size_t size)
    : size_(size),  // <<< Initialize size_ member
      pimpl_(
          Backend::createFFTPlan<T>(size)) {  // Create backend implementation
  // Check if backend creation was successful
  if (!pimpl_) {
    throw std::runtime_error(
        "Failed to create FFTPlan backend implementation. No suitable backend "
        "found or backend failed to initialize.");
  }
  // Check for invalid size
  if (size == 0) {
    // While backend might handle 0, it's often problematic or undefined.
    throw std::invalid_argument("FFTPlan size cannot be zero.");
  }
}

/**
 * @brief Destructor. Default implementation is sufficient because
 * std::unique_ptr automatically cleans up the implementation object.
 */
template <typename T>
FFTPlan<T>::~FFTPlan() = default;

/**
 * @brief Move constructor. Needed because copy operations are deleted and we
 * use unique_ptr.
 */
template <typename T>
FFTPlan<T>::FFTPlan(FFTPlan&&) noexcept = default;

/**
 * @brief Move assignment operator. Needed because copy operations are deleted
 * and we use unique_ptr.
 */
template <typename T>
FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan&&) noexcept = default;

/**
 * @brief Executes the forward FFT, delegating to the backend implementation.
 */
template <typename T>
void FFTPlan<T>::fft(const std::vector<std::complex<T>>& input,
                     std::vector<std::complex<T>>& output, FFTNorm norm) {
  // Validate input/output sizes against the stored plan size
  if (input.size() != size_ || output.size() != size_) {
    throw std::invalid_argument(
        "FFTPlan::fft input/output vector size (" +
        std::to_string(input.size()) + "/" + std::to_string(output.size()) +
        ") does not match plan size (" + std::to_string(size_) + ")");
  }
  // Ensure the implementation pointer is valid before dereferencing
  if (pimpl_) {
    // Delegate to the backend implementation
    pimpl_->fft(input.data(), output.data(), norm);
  } else {
    // This should ideally not happen if constructor succeeded, but check for
    // safety
    throw std::runtime_error(
        "FFTPlan::fft called on a null backend implementation.");
  }
}

/**
 * @brief Executes the inverse FFT, delegating to the backend implementation.
 */
template <typename T>
void FFTPlan<T>::ifft(const std::vector<std::complex<T>>& input,
                      std::vector<std::complex<T>>& output, FFTNorm norm) {
  // Validate input/output sizes
  if (input.size() != size_ || output.size() != size_) {
    throw std::invalid_argument(
        "FFTPlan::ifft input/output vector size (" +
        std::to_string(input.size()) + "/" + std::to_string(output.size()) +
        ") does not match plan size (" + std::to_string(size_) + ")");
  }
  // Delegate to the backend implementation
  if (pimpl_) {
    pimpl_->ifft(input.data(), output.data(), norm);
  } else {
    throw std::runtime_error(
        "FFTPlan::ifft called on a null backend implementation.");
  }
}

/**
 * @brief Gets the size N (number of complex samples) for which this plan was
 * created.
 */
template <typename T>
size_t FFTPlan<T>::getSize() const {
  return size_;  // Return the stored size
}

//--------------------------------------------------------------------------
// RFFTPlan (Real-to-Complex / Complex-to-Real)
//--------------------------------------------------------------------------

/**
 * @brief Constructs RFFTPlan, creating the backend implementation.
 * Stores the real signal size N.
 */
template <typename T>
RFFTPlan<T>::RFFTPlan(size_t size)
    : size_(size),  // <<< Initialize size_ member (Real size N)
      pimpl_(
          Backend::createRFFTPlan<T>(size)) {  // Create backend implementation
  // Check if backend creation was successful
  if (!pimpl_) {
    throw std::runtime_error(
        "Failed to create RFFTPlan backend implementation. No suitable backend "
        "found or backend failed to initialize.");
  }
  // Check for invalid size
  if (size == 0) {
    throw std::invalid_argument("RFFTPlan size cannot be zero.");
  }
}

/**
 * @brief Destructor. Default implementation is sufficient.
 */
template <typename T>
RFFTPlan<T>::~RFFTPlan() = default;

/**
 * @brief Move constructor.
 */
template <typename T>
RFFTPlan<T>::RFFTPlan(RFFTPlan&&) noexcept = default;

/**
 * @brief Move assignment operator.
 */
template <typename T>
RFFTPlan<T>& RFFTPlan<T>::operator=(RFFTPlan&&) noexcept = default;

/**
 * @brief Executes the forward RFFT, delegating to the backend implementation.
 */
template <typename T>
void RFFTPlan<T>::rfft(const std::vector<T>& input,
                       std::vector<std::complex<T>>& output, FFTNorm norm) {
  // Validate input/output sizes based on the stored real size N
  size_t expected_output_size = size_ / 2 + 1;
  if (input.size() != size_ || output.size() != expected_output_size) {
    throw std::invalid_argument(
        "RFFTPlan::rfft input/output vector size (" +
        std::to_string(input.size()) + "/" + std::to_string(output.size()) +
        ") does not match plan size (Real N: " + std::to_string(size_) +
        ", Complex N/2+1: " + std::to_string(expected_output_size) + ")");
  }
  // Delegate to the backend implementation
  if (pimpl_) {
    pimpl_->rfft(input.data(), output.data(), norm);
  } else {
    throw std::runtime_error(
        "RFFTPlan::rfft called on a null backend implementation.");
  }
}

/**
 * @brief Executes the inverse IRFFT, delegating to the backend implementation.
 */
template <typename T>
void RFFTPlan<T>::irfft(const std::vector<std::complex<T>>& input,
                        std::vector<T>& output, FFTNorm norm) {
  // Validate input/output sizes based on the stored real size N
  size_t expected_input_size = size_ / 2 + 1;
  if (input.size() != expected_input_size || output.size() != size_) {
    throw std::invalid_argument("RFFTPlan::irfft input/output vector size (" +
                                std::to_string(input.size()) + "/" +
                                std::to_string(output.size()) +
                                ") does not match plan size (Complex N/2+1: " +
                                std::to_string(expected_input_size) +
                                ", Real N: " + std::to_string(size_) + ")");
  }
  // Delegate to the backend implementation
  if (pimpl_) {
    pimpl_->irfft(input.data(), output.data(), norm);
  } else {
    throw std::runtime_error(
        "RFFTPlan::irfft called on a null backend implementation.");
  }
}

/**
 * @brief Gets the real signal size N for which this plan was created.
 */
template <typename T>
size_t RFFTPlan<T>::getSize() const {
  return size_;  // Return the stored size
}

//--------------------------------------------------------------------------
// Convenience Functions (Stateless)
//--------------------------------------------------------------------------
// These create temporary plans internally.

template <typename T>
void fft(const std::vector<std::complex<T>>& input,
         std::vector<std::complex<T>>& output, FFTNorm norm) {
  if (input.empty()) {
    output.clear();
    return;  // Handle empty input
  }
  size_t size = input.size();
  if (output.size() != size) {
    output.resize(size);  // Ensure output is correct size
  }
  // Create a temporary plan for one-off use
  FFTPlan<T> plan(size);
  plan.fft(input, output, norm);
}

template <typename T>
void ifft(const std::vector<std::complex<T>>& input,
          std::vector<std::complex<T>>& output, FFTNorm norm) {
  if (input.empty()) {
    output.clear();
    return;
  }
  size_t size = input.size();
  if (output.size() != size) {
    output.resize(size);
  }
  FFTPlan<T> plan(size);
  plan.ifft(input, output, norm);
}

template <typename T>
void rfft(const std::vector<T>& input, std::vector<std::complex<T>>& output,
          FFTNorm norm) {
  if (input.empty()) {
    output.clear();
    return;
  }
  size_t size = input.size();
  size_t expected_output_size = size / 2 + 1;
  if (output.size() != expected_output_size) {
    output.resize(expected_output_size);
  }
  RFFTPlan<T> plan(size);
  plan.rfft(input, output, norm);
}

template <typename T>
void irfft(const std::vector<std::complex<T>>& input, std::vector<T>& output,
           size_t N,  // N = target real output size
           FFTNorm norm) {
  if (N == 0) {  // Handle N=0 case
    output.clear();
    if (!input.empty()) {
      throw std::invalid_argument(
          "irfft: Input must be empty if target size N is 0.");
    }
    return;
  }
  size_t expected_input_size = N / 2 + 1;
  if (input.size() != expected_input_size) {
    throw std::invalid_argument("irfft: Input vector size (" +
                                std::to_string(input.size()) +
                                ") does not match expected size (" +
                                std::to_string(expected_input_size) +
                                ") for output size N=" + std::to_string(N));
  }
  if (output.size() != N) {
    output.resize(N);
  }
  RFFTPlan<T> plan(N);
  plan.irfft(input, output, norm);
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Required because the definitions are in this .cpp file.

// FFTPlan Class
template class FFTPlan<float>;
template class FFTPlan<double>;
// FFTPlan Methods (including new getSize)
template size_t FFTPlan<float>::getSize() const;
template size_t FFTPlan<double>::getSize() const;
template void FFTPlan<float>::fft(const std::vector<std::complex<float>>&,
                                  std::vector<std::complex<float>>&, FFTNorm);
template void FFTPlan<double>::fft(const std::vector<std::complex<double>>&,
                                   std::vector<std::complex<double>>&, FFTNorm);
template void FFTPlan<float>::ifft(const std::vector<std::complex<float>>&,
                                   std::vector<std::complex<float>>&, FFTNorm);
template void FFTPlan<double>::ifft(const std::vector<std::complex<double>>&,
                                    std::vector<std::complex<double>>&,
                                    FFTNorm);

// RFFTPlan Class
template class RFFTPlan<float>;
template class RFFTPlan<double>;
// RFFTPlan Methods (including new getSize)
template size_t RFFTPlan<float>::getSize() const;
template size_t RFFTPlan<double>::getSize() const;
template void RFFTPlan<float>::rfft(const std::vector<float>&,
                                    std::vector<std::complex<float>>&, FFTNorm);
template void RFFTPlan<double>::rfft(const std::vector<double>&,
                                     std::vector<std::complex<double>>&,
                                     FFTNorm);
template void RFFTPlan<float>::irfft(const std::vector<std::complex<float>>&,
                                     std::vector<float>&, FFTNorm);
template void RFFTPlan<double>::irfft(const std::vector<std::complex<double>>&,
                                      std::vector<double>&, FFTNorm);

// Convenience Functions
template void fft<float>(const std::vector<std::complex<float>>&,
                         std::vector<std::complex<float>>&, FFTNorm);
template void fft<double>(const std::vector<std::complex<double>>&,
                          std::vector<std::complex<double>>&, FFTNorm);
template void ifft<float>(const std::vector<std::complex<float>>&,
                          std::vector<std::complex<float>>&, FFTNorm);
template void ifft<double>(const std::vector<std::complex<double>>&,
                           std::vector<std::complex<double>>&, FFTNorm);
template void rfft<float>(const std::vector<float>&,
                          std::vector<std::complex<float>>&, FFTNorm);
template void rfft<double>(const std::vector<double>&,
                           std::vector<std::complex<double>>&, FFTNorm);
template void irfft<float>(const std::vector<std::complex<float>>&,
                           std::vector<float>&, size_t, FFTNorm);
template void irfft<double>(const std::vector<std::complex<double>>&,
                            std::vector<double>&, size_t, FFTNorm);

}  // namespace OmniDSP
