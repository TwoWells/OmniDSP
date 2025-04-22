/**
 * @file backend.h (stub)
 * @brief Declares the concrete Stub backend implementation classes.
 * @details These classes inherit from the abstract interfaces defined in
 * src/omnidsp/backend/backend.h and provide basic, portable implementations
 * using standard C++ (loops, std::vector, std::complex, etc.).
 */

#ifndef OMNIDSP_STUB_BACKEND_H
#define OMNIDSP_STUB_BACKEND_H

#include <cmath>  // For basic math functions
#include <complex>
#include <memory>
#include <vector>

#include "../backend.h"  // Include the main backend interface definitions

namespace OmniDSP {
namespace backend {

//--------------------------------------------------------------------------
// Stub Plan Implementation Classes
//--------------------------------------------------------------------------

/** @brief Stub implementation for complex FFT plans. */
template <typename T>
class StubFFTPlanImpl final : public FFTPlanImpl<T> {
 public:
  // Constructor: May precompute twiddle factors or do nothing.
  StubFFTPlanImpl(size_t length);
  // Destructor
  ~StubFFTPlanImpl() override = default;

  // Execute methods (implement basic FFT algorithm like Cooley-Tukey)
  Status fft(std::span<const T> input, std::span<T> output) const override;
  Status ifft(std::span<const T> input, std::span<T> output) const override;

  // Getter
  size_t get_length() const override;

 private:
  size_t length_;
  // Optional: Store precomputed twiddle factors
  std::vector<T> twiddle_factors_;
  std::vector<size_t> bit_reverse_indices_;

  // Helper for basic Cooley-Tukey FFT (or other algorithm)
  void compute_fft(std::span<const T> input, std::span<T> output,
                   bool inverse) const;
};

/** @brief Stub implementation for real FFT plans. */
template <typename T>
class StubRFFTPlanImpl final : public RFFTPlanImpl<T> {
 public:
  // Constructor: May setup based on complex FFT or specific real FFT algorithm.
  StubRFFTPlanImpl(size_t length);
  // Destructor
  ~StubRFFTPlanImpl() override = default;

  // Execute methods (implement real FFT algorithm, e.g., using complex FFT of
  // half length)
  Status rfft(std::span<const RealT<T>> input,
              std::span<ComplexT<T>> output) const override;
  Status irfft(std::span<const ComplexT<T>> input,
               std::span<RealT<T>> output) const override;

  // Getter
  size_t get_length() const override;

 private:
  size_t length_;
  // May internally use a complex FFT plan of size N/2
  // std::unique_ptr<StubFFTPlanImpl<ComplexT<T>>> internal_fft_plan_; //
  // Example Or store twiddle factors directly
  std::vector<ComplexT<T>> twiddle_factors_;
  std::vector<size_t> bit_reverse_indices_;
};

/** @brief Stub implementation for resampling plans. */
template <typename T>
class StubResamplePlanImpl final : public ResamplePlanImpl<T> {
 public:
  // Constructor: Designs and stores polyphase filter coefficients.
  StubResamplePlanImpl(double input_rate, double output_rate,
                       size_t max_input_size);
  // Destructor
  ~StubResamplePlanImpl() override = default;

  // Execute method (implement polyphase FIR filtering loop)
  Status execute(std::span<const T> input, std::span<T> output) const override;

  // Getters
  double get_input_rate() const override;
  double get_output_rate() const override;
  size_t get_output_length(size_t input_length) const override;

 private:
  double input_rate_;
  double output_rate_;
  int upsample_factor_L_;
  int downsample_factor_M_;
  size_t filter_length_;
  std::vector<T> prototype_filter_;
  // Polyphase filter bank (optional precomputation)
  // std::vector<std::vector<T>> polyphase_filters_;
  // Filter state (delay line) - needs to be mutable for execute
  mutable std::vector<T> filter_state_;
  mutable size_t current_phase_;  // Keep track of phase for polyphase index
};

/** @brief Stub implementation for convolution plans. */
template <typename T>
class StubConvolutionPlanImpl final : public ConvolutionPlanImpl<T> {
 public:
  // Constructor: Stores reversed kernel and mode.
  StubConvolutionPlanImpl(const std::vector<T>& kernel, ConvolutionMode mode);
  // Destructor
  ~StubConvolutionPlanImpl() override = default;

  // Execute method (implement direct convolution loop)
  Status execute(std::span<const T> input, std::span<T> output) const override;

  // Getters
  size_t get_kernel_length() const override;
  ConvolutionMode get_mode() const override;
  size_t get_output_length(size_t input_length) const override;

 private:
  std::vector<T>
      reversed_kernel_;  // Store reversed kernel for direct convolution
  ConvolutionMode mode_;
  size_t kernel_length_;
};

/** @brief Stub implementation for correlation plans. */
template <typename T>
class StubCorrelationPlanImpl final : public CorrelationPlanImpl<T> {
 public:
  // Constructor: Stores template (kernel) and mode.
  StubCorrelationPlanImpl(const std::vector<T>& kernel, ConvolutionMode mode);
  // Destructor
  ~StubCorrelationPlanImpl() override = default;

  // Execute method (implement direct correlation loop)
  Status execute(std::span<const T> input, std::span<T> output) const override;

  // Getters
  size_t get_template_length() const override;
  ConvolutionMode get_mode() const override;
  size_t get_output_length(size_t input_length) const override;

 private:
  std::vector<T> template_;  // Store template directly
  ConvolutionMode mode_;
  size_t template_length_;
};

// Add StubFilterPlanImpl when IIR filtering is added

//--------------------------------------------------------------------------
// Stub Main Backend Implementation Class
//--------------------------------------------------------------------------

/** @brief Concrete Stub backend implementation for OmniDSPImpl. */
class StubOmniDSPImpl final : public OmniDSPImpl {
 public:
  StubOmniDSPImpl();  // Constructor
  ~StubOmniDSPImpl() override = default;

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

#endif  // OMNIDSP_STUB_BACKEND_H
