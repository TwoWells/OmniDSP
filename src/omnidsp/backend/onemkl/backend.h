/**
 * @file backend.h (onemkl)
 * @brief Declares the concrete oneMKL backend implementation classes.
 * @details These classes inherit from the abstract interfaces defined in
 * src/omnidsp/backend/backend.h and implement the functionality using
 * Intel oneAPI Math Kernel Library (oneMKL) functions (DFT, VML, etc.).
 */

#ifndef OMNIDSP_ONEMKL_BACKEND_H
#define OMNIDSP_ONEMKL_BACKEND_H

// Only compile this file if oneMKL backend is enabled via CMake
#ifdef USE_ONEMKL

#include <mkl.h>  // Main oneMKL header (includes DFTI, VML, etc.)

#include <complex>
#include <memory>
#include <stdexcept>  // For potential exceptions during setup
#include <vector>

#include "../backend.h"  // Include the main backend interface definitions

// Helper to map std::complex to MKL_Complex types
template <typename T>
struct MKLComplexType;
template <>
struct MKLComplexType<std::complex<float>> {
  using type = MKL_Complex8;
};
template <>
struct MKLComplexType<std::complex<double>> {
  using type = MKL_Complex16;
};

namespace OmniDSP {
namespace backend {

//--------------------------------------------------------------------------
// oneMKL Plan Implementation Classes
//--------------------------------------------------------------------------

/** @brief oneMKL implementation for complex FFT plans using DFTI. */
template <typename T>
class OneMKLFFTPlanImpl final : public FFTPlanImpl<T> {
 public:
  // Constructor: Creates DFTI descriptor
  OneMKLFFTPlanImpl(size_t length);
  // Destructor: Frees DFTI descriptor
  ~OneMKLFFTPlanImpl() override;

  // Execute methods
  Status fft(std::span<const T> input, std::span<T> output) const override;
  Status ifft(std::span<const T> input, std::span<T> output) const override;

  // Getter
  size_t get_length() const override;

 private:
  size_t length_;
  DFTI_DESCRIPTOR_HANDLE descriptor_handle_ = nullptr;
  // Store MKL_LONG status for error checking
  MKL_LONG mkl_status_;
};

/** @brief oneMKL implementation for real FFT plans using DFTI. */
template <typename T>
class OneMKLRFFTPlanImpl final : public RFFTPlanImpl<T> {
 public:
  // Constructor: Creates DFTI descriptor for real transforms
  OneMKLRFFTPlanImpl(size_t length);
  // Destructor: Frees DFTI descriptor
  ~OneMKLRFFTPlanImpl() override;

  // Execute methods
  Status rfft(std::span<const RealT<T>> input,
              std::span<ComplexT<T>> output) const override;
  Status irfft(std::span<const ComplexT<T>> input,
               std::span<RealT<T>> output) const override;

  // Getter
  size_t get_length() const override;

 private:
  size_t length_;
  DFTI_DESCRIPTOR_HANDLE descriptor_handle_ = nullptr;
  MKL_LONG mkl_status_;
  // Store configuration values needed by DFTI ComputeForward/Backward for real
  // FFTs
  MKL_LONG number_of_transforms_ = 1;
  MKL_LONG input_distance_ = 0;
  MKL_LONG output_distance_ = 0;
  MKL_LONG input_strides_[1] = {0};   // For real input/output
  MKL_LONG output_strides_[1] = {0};  // For complex conjugate-even output/input
};

/** @brief oneMKL implementation for resampling plans. */
template <typename T>
class OneMKLResamplePlanImpl final : public ResamplePlanImpl<T> {
 public:
  // Constructor: Sets up resampling (potentially using MKL VSL
  // convolution/filter design?)
  OneMKLResamplePlanImpl(double input_rate, double output_rate,
                         size_t max_input_size);
  // Destructor
  ~OneMKLResamplePlanImpl() override =
      default;  // May need custom if resources allocated

  // Execute method
  Status execute(std::span<const T> input, std::span<T> output) const override;

  // Getters
  double get_input_rate() const override;
  double get_output_rate() const override;
  size_t get_output_length(size_t input_length) const override;

 private:
  double input_rate_;
  double output_rate_;
  // Store pre-computed filter coefficients, state, etc.
  // MKL VSL might offer convolution tasks useful here.
  // Implementation might involve polyphase filtering similar to Accelerate.
};

/** @brief oneMKL implementation for convolution plans using DFTI or VSL
 * Convolution. */
template <typename T>
class OneMKLConvolutionPlanImpl final : public ConvolutionPlanImpl<T> {
 public:
  // Constructor: Stores kernel, mode, potentially pre-computes kernel FFT using
  // DFTI or sets up VSL task
  OneMKLConvolutionPlanImpl(const std::vector<T>& kernel, ConvolutionMode mode);
  // Destructor
  ~OneMKLConvolutionPlanImpl()
      override;  // May need to free DFTI descriptor or VSL task

  // Execute method
  Status execute(std::span<const T> input, std::span<T> output) const override;

  // Getters
  size_t get_kernel_length() const override;
  ConvolutionMode get_mode() const override;
  size_t get_output_length(size_t input_length) const override;

 private:
  ConvolutionMode mode_;
  size_t kernel_length_;
  size_t input_length_hint_ = 0;  // May be needed for VSL task setup
  // Store either VSL Conv task handle or DFTI descriptor and FFT'd kernel
  // VSLTaskPtr vsl_task_ = nullptr; // Example if using VSL
  DFTI_DESCRIPTOR_HANDLE dfti_descriptor_ = nullptr;  // Example if using DFTI
  std::vector<ComplexT<typename Detail::ValueType<T>::type>>
      kernel_fft_;  // Example
  MKL_LONG mkl_status_;
};

/** @brief oneMKL implementation for correlation plans using DFTI or VSL
 * Convolution. */
template <typename T>
class OneMKLCorrelationPlanImpl final : public CorrelationPlanImpl<T> {
 public:
  // Constructor: Stores template, mode, potentially pre-computes template FFT
  // or sets up VSL task
  OneMKLCorrelationPlanImpl(const std::vector<T>& kernel, ConvolutionMode mode);
  // Destructor
  ~OneMKLCorrelationPlanImpl()
      override;  // May need to free DFTI descriptor or VSL task

  // Execute method
  Status execute(std::span<const T> input, std::span<T> output) const override;

  // Getters
  size_t get_template_length() const override;
  ConvolutionMode get_mode() const override;
  size_t get_output_length(size_t input_length) const override;

 private:
  ConvolutionMode mode_;
  size_t template_length_;
  // Store either VSL Corr task handle or DFTI descriptor and FFT'd (conjugated)
  // template VSLTaskPtr vsl_task_ = nullptr; // Example
  DFTI_DESCRIPTOR_HANDLE dfti_descriptor_ = nullptr;  // Example
  std::vector<ComplexT<typename Detail::ValueType<T>::type>>
      template_fft_conj_;  // Example
  MKL_LONG mkl_status_;
};

// Add OneMKLFilterPlanImpl when IIR filtering is added

//--------------------------------------------------------------------------
// oneMKL Main Backend Implementation Class
//--------------------------------------------------------------------------

/** @brief Concrete oneMKL backend implementation for OmniDSPImpl. */
class OneMKLOmniDSPImpl final : public OmniDSPImpl {
 public:
  OneMKLOmniDSPImpl();  // Constructor for any one-time setup (e.g., MKL
                        // threading mode)
  ~OneMKLOmniDSPImpl() override = default;

  Backend get_backend() const override;

  // --- DSP Operations ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> convolve(
      const std::vector<RealT<T>>& input, const std::vector<RealT<T>>& kernel,
      ConvolutionMode mode) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> convolve(
      const std::vector<ComplexT<T>>& input,
      const std::vector<ComplexT<T>>& kernel,
      ConvolutionMode mode) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> correlate(
      const std::vector<RealT<T>>& input, const std::vector<RealT<T>>& kernel,
      ConvolutionMode mode) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> correlate(
      const std::vector<ComplexT<T>>& input,
      const std::vector<ComplexT<T>>& kernel,
      ConvolutionMode mode) const override;

  // --- One-off FFTs ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> fft(
      const std::vector<ComplexT<T>>& input) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> ifft(
      const std::vector<ComplexT<T>>& input) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> rfft(
      const std::vector<RealT<T>>& input) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> irfft(
      const std::vector<ComplexT<T>>& input,
      size_t output_length) const override;

  // --- Window Generation ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> bartlett_window(
      size_t length) const override;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> blackman_window(
      size_t length) const override;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> flattop_window(
      size_t length) const override;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> gaussian_window(
      size_t length, RealT<T> stddev) const override;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> hamming_window(
      size_t length) const override;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> hann_window(
      size_t length) const override;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> kaiser_window(
      size_t length, RealT<T> beta) const override;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> rectangular_window(
      size_t length) const override;
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> triangular_window(
      size_t length) const override;

  // --- Plan Factories ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>> create_fft_plan(
      size_t length) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>> create_rfft_plan(
      size_t length) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>> create_cqt_plan(
      const OmniDSP* owner,  // Pass owner for sub-plan creation
      RealT<T> sample_rate, RealT<T> min_freq, RealT<T> max_freq,
      int bins_per_octave) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
  create_resample_plan(double input_rate, double output_rate,
                       size_t max_input_size) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
  create_convolution_plan(const std::vector<T>& kernel,
                          ConvolutionMode mode) const override;

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
  create_correlation_plan(const std::vector<T>& kernel,
                          ConvolutionMode mode) const override;

  // Add overrides for FilterPlan factory and filter design methods when added
};

}  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ONEMKL
#endif  // OMNIDSP_ONEMKL_BACKEND_H
