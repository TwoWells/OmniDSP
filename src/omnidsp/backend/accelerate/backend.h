/**
 * @file backend.h (accelerate)
 * @brief Declares the concrete Accelerate backend implementation classes.
 * @details These classes inherit from the abstract interfaces defined in
 * src/omnidsp/backend/backend.h and implement the functionality using
 * Apple's Accelerate framework (vDSP, vForce).
 */

#ifndef OMNIDSP_ACCELERATE_BACKEND_H
#define OMNIDSP_ACCELERATE_BACKEND_H

// Only compile this file if Accelerate backend is enabled via CMake
#ifdef USE_ACCELERATE

#include <Accelerate/Accelerate.h>  // Main Accelerate framework header

#include <complex>
#include <memory>
#include <vector>

#include "../backend.h"  // Include the main backend interface definitions

namespace OmniDSP {
  namespace backend {

    //--------------------------------------------------------------------------
    // Accelerate Plan Implementation Classes
    //--------------------------------------------------------------------------

    /** @brief Accelerate implementation for complex FFT plans. */
    template <typename T>
    class AccelerateFFTPlanImpl final : public FFTPlanImpl<T> {
     public:
      // Constructor: Creates vDSP_DFT_Setup
      AccelerateFFTPlanImpl(size_t length);
      // Destructor: Destroys vDSP_DFT_Setup
      ~AccelerateFFTPlanImpl() override;

      // Execute methods
      Status fft(std::span<const T> input, std::span<T> output) const override;
      Status ifft(std::span<const T> input, std::span<T> output) const override;

      // Getter
      size_t get_length() const override;

     private:
      size_t length_;
      // Store vDSP_DFT_Setup (needs different types for float/double)
      typename std::conditional<
          std::is_same_v<T, std::complex<float>>,
          vDSP_DFT_Setup,
          vDSP_DFT_SetupD>::type setup_;
      // Potentially store temporary buffers if needed by vDSP_DFT_Execute
    };

    /** @brief Accelerate implementation for real FFT plans. */
    template <typename T>
    class AccelerateRFFTPlanImpl final : public RFFTPlanImpl<T> {
     public:
      // Constructor: Creates vDSP_FFTSetup
      AccelerateRFFTPlanImpl(size_t length);
      // Destructor: Destroys vDSP_FFTSetup
      ~AccelerateRFFTPlanImpl() override;

      // Execute methods
      Status rfft(
          std::span<const RealT<T>> input,
          std::span<ComplexT<T>> output) const override;
      Status irfft(
          std::span<const ComplexT<T>> input,
          std::span<RealT<T>> output) const override;

      // Getter
      size_t get_length() const override;

     private:
      size_t length_;
      // Store vDSP_FFTSetup (needs different types for float/double)
      typename std::conditional<
          std::is_same_v<T, float>,
          vDSP_FFTSetup,
          vDSP_FFTSetupD>::type setup_;
      // Temporary buffer often needed for vDSP real FFTs
      typename std::conditional<
          std::is_same_v<T, float>,
          DSPSplitComplex,
          DSPDoubleSplitComplex>::type temp_split_complex_;
    };

    /** @brief Accelerate implementation for resampling plans. */
    template <typename T>
    class AccelerateResamplePlanImpl final : public ResamplePlanImpl<T> {
     public:
      // Constructor: Designs polyphase filter, sets up state
      AccelerateResamplePlanImpl(
          double input_rate, double output_rate, size_t max_input_size);
      // Destructor
      ~AccelerateResamplePlanImpl() override
          = default;  // May need custom if resources allocated

      // Execute method
      Status execute(
          std::span<const T> input, std::span<T> output) const override;

      // Getters
      double get_input_rate() const override;
      double get_output_rate() const override;
      size_t get_output_length(size_t input_length) const override;

     private:
      double input_rate_;
      double output_rate_;
      // Store pre-computed filter coefficients, state, etc.
      // Accelerate doesn't have a direct resampler, implementation is complex.
      // May involve vDSP_conv or FFT-based overlap-add/save.
    };

    /** @brief Accelerate implementation for convolution plans. */
    template <typename T>
    class AccelerateConvolutionPlanImpl final : public ConvolutionPlanImpl<T> {
     public:
      // Constructor: Stores kernel, mode, potentially pre-computes kernel FFT
      AccelerateConvolutionPlanImpl(
          const std::vector<T>& kernel, ConvolutionMode mode);
      // Destructor
      ~AccelerateConvolutionPlanImpl() override
          = default;  // May need custom if FFT setup stored

      // Execute method
      Status execute(
          std::span<const T> input, std::span<T> output) const override;

      // Getters
      size_t get_kernel_length() const override;
      ConvolutionMode get_mode() const override;
      size_t get_output_length(size_t input_length) const override;

     private:
      std::vector<T> kernel_;  // Store kernel
      ConvolutionMode mode_;
      size_t kernel_length_;
      // Potentially store FFT setup and FFT'd kernel if using FFT-based
      // convolution std::unique_ptr<AccelerateFFTPlanImpl<T>> fft_plan_; //
      // Example std::vector<T> kernel_fft_; // Example
    };

    /** @brief Accelerate implementation for correlation plans. */
    template <typename T>
    class AccelerateCorrelationPlanImpl final : public CorrelationPlanImpl<T> {
     public:
      // Constructor: Stores template, mode, potentially pre-computes template
      // FFT
      AccelerateCorrelationPlanImpl(
          const std::vector<T>& kernel, ConvolutionMode mode);
      // Destructor
      ~AccelerateCorrelationPlanImpl() override
          = default;  // May need custom if FFT setup stored

      // Execute method
      Status execute(
          std::span<const T> input, std::span<T> output) const override;

      // Getters
      size_t get_template_length() const override;
      ConvolutionMode get_mode() const override;
      size_t get_output_length(size_t input_length) const override;

     private:
      std::vector<T> template_;  // Store template (kernel)
      ConvolutionMode mode_;
      size_t template_length_;
      // Potentially store FFT setup and FFT'd template (conjugated)
    };

    // Add AccelerateFilterPlanImpl when IIR filtering is added

    //--------------------------------------------------------------------------
    // Accelerate Main Backend Implementation Class
    //--------------------------------------------------------------------------

    /** @brief Concrete Accelerate backend implementation for OmniDSPImpl. */
    class AccelerateOmniDSPImpl final : public OmniDSPImpl {
     public:
      AccelerateOmniDSPImpl();  // Constructor for any one-time setup
      ~AccelerateOmniDSPImpl() override = default;

      Backend get_backend() const override;

      // --- DSP Operations ---
      template <typename T>
      [[nodiscard]] OmniExpected<std::vector<RealT<T>>> convolve(
          const std::vector<RealT<T>>& input,
          const std::vector<RealT<T>>& kernel,
          ConvolutionMode mode) const override;

      template <typename T>
      [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> convolve(
          const std::vector<ComplexT<T>>& input,
          const std::vector<ComplexT<T>>& kernel,
          ConvolutionMode mode) const override;

      template <typename T>
      [[nodiscard]] OmniExpected<std::vector<RealT<T>>> correlate(
          const std::vector<RealT<T>>& input,
          const std::vector<RealT<T>>& kernel,
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
          RealT<T> sample_rate,
          RealT<T> min_freq,
          RealT<T> max_freq,
          int bins_per_octave) const override;

      template <typename T>
      [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
      create_resample_plan(
          double input_rate,
          double output_rate,
          size_t max_input_size) const override;

      template <typename T>
      [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
      create_convolution_plan(
          const std::vector<T>& kernel, ConvolutionMode mode) const override;

      template <typename T>
      [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
      create_correlation_plan(
          const std::vector<T>& kernel, ConvolutionMode mode) const override;

      // Add overrides for FilterPlan factory and filter design methods when
      // added
    };

  }  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ACCELERATE
#endif  // OMNIDSP_ACCELERATE_BACKEND_H
