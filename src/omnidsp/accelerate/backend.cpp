/**
 * @file backend.cpp (accelerate)
 * @brief Implements the AccelerateOmniDSPImpl class methods using Accelerate
 * framework.
 */

// Only compile this file if Accelerate backend is enabled via CMake
#ifdef USE_ACCELERATE

#include "backend.hpp"  // Corresponding header for Accelerate backend declarations

// Include headers for the public Plan classes (needed for factory return types
// and constructors)
#include "OmniDSP/convolution.hpp"
#include "OmniDSP/cqt.hpp"
#include "OmniDSP/fft.hpp"
#include "OmniDSP/omnidsp.hpp"  // Needed for OmniDSP class definition (for owner pointer)
#include "OmniDSP/resample.hpp"
// #include "OmniDSP/filter.hpp" // Include when FilterPlan is added

// Include implementation headers for the Plan Impls (optional but good
// practice) #include "fft.cpp" // Or specific headers if implementations are
// split #include "convolution.cpp" #include "resample.cpp"

#include <Accelerate/Accelerate.h>

#include <complex>
#include <iostream>  // For debug/error messages
#include <memory>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

namespace OmniDSP {
  namespace backend {

    //--------------------------------------------------------------------------
    // AccelerateOmniDSPImpl Method Definitions
    //--------------------------------------------------------------------------

    AccelerateOmniDSPImpl::AccelerateOmniDSPImpl()
    {
      // Perform any one-time setup specific to the Accelerate backend, if
      // needed. Often, Accelerate doesn't require explicit global
      // initialization.
      std::cout << "Accelerate Backend Initialized."
                << std::endl;  // Debug message
    }

    Backend AccelerateOmniDSPImpl::get_backend() const
    {
      return Backend::Accelerate;
    }

    // --- DSP Operations (Placeholders - Need Accelerate Implementation) ---

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::convolve(
        const std::vector<RealT<T>>& input,
        const std::vector<RealT<T>>& kernel,
        ConvolutionMode mode) const
    {
      // TODO: Implement 1D real convolution using Accelerate (e.g., vDSP_conv
      // or FFT based) Consider different modes (Full, Same, Valid)
      std::cerr << "AccelerateOmniDSPImpl::convolve (real) not yet implemented."
                << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    AccelerateOmniDSPImpl::convolve(
        const std::vector<ComplexT<T>>& input,
        const std::vector<ComplexT<T>>& kernel,
        ConvolutionMode mode) const
    {
      // TODO: Implement 1D complex convolution using Accelerate (likely FFT
      // based)
      std::cerr
          << "AccelerateOmniDSPImpl::convolve (complex) not yet implemented."
          << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::correlate(
        const std::vector<RealT<T>>& input,
        const std::vector<RealT<T>>& kernel,
        ConvolutionMode mode) const
    {
      // TODO: Implement 1D real correlation using Accelerate (e.g., vDSP_conv
      // or FFT based, kernel not reversed)
      std::cerr
          << "AccelerateOmniDSPImpl::correlate (real) not yet implemented."
          << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    AccelerateOmniDSPImpl::correlate(
        const std::vector<ComplexT<T>>& input,
        const std::vector<ComplexT<T>>& kernel,
        ConvolutionMode mode) const
    {
      // TODO: Implement 1D complex correlation using Accelerate (likely FFT
      // based, conjugate kernel)
      std::cerr
          << "AccelerateOmniDSPImpl::correlate (complex) not yet implemented."
          << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    // --- One-off FFTs (Placeholders - Need Accelerate Implementation) ---
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    AccelerateOmniDSPImpl::fft(const std::vector<ComplexT<T>>& input) const
    {
      // TODO: Implement one-off complex FFT using Accelerate (e.g., create
      // temporary plan)
      std::cerr << "AccelerateOmniDSPImpl::fft not yet implemented."
                << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    AccelerateOmniDSPImpl::ifft(const std::vector<ComplexT<T>>& input) const
    {
      // TODO: Implement one-off complex IFFT using Accelerate (e.g., create
      // temporary plan)
      std::cerr << "AccelerateOmniDSPImpl::ifft not yet implemented."
                << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    AccelerateOmniDSPImpl::rfft(const std::vector<RealT<T>>& input) const
    {
      // TODO: Implement one-off real RFFT using Accelerate (e.g., create
      // temporary plan)
      std::cerr << "AccelerateOmniDSPImpl::rfft not yet implemented."
                << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::irfft(
        const std::vector<ComplexT<T>>& input, size_t output_length) const
    {
      // TODO: Implement one-off real IRFFT using Accelerate (e.g., create
      // temporary plan)
      std::cerr << "AccelerateOmniDSPImpl::irfft not yet implemented."
                << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    // --- Window Generation (Implementations using vDSP) ---
    // Helper to convert OmniDSP Status to vDSP status if needed, or handle
    // errors directly
    Status check_vDSP_status(Status /* vdsp_status */)
    {
      // Placeholder - map vDSP errors if necessary
      return Status::Success;
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::bartlett_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();  // Handle zero length
      std::vector<RealT<T>> coeffs(length);
      // vDSP doesn't have Bartlett directly, use vDSP_vgen or vDSP_vramp
      // Simplified implementation (might need adjustment for exact definition)
      if constexpr (std::is_same_v<T, float>) {
        float N_minus_1 = static_cast<float>(length - 1);
        if (N_minus_1 <= 0) {  // length 1 case
          if (length > 0) coeffs[0] = 1.0f;
          return coeffs;
        }
        float scale = 2.0f / N_minus_1;
        float start = 0.0f;
        vDSP_vramp(&start, &scale, coeffs.data(), 1, length / 2);  // Ramp up
        if (length % 2 != 0) {  // Odd length - middle sample is 1.0
          coeffs[length / 2] = 1.0f;
        }
        // Ramp down - more complex, maybe easier with direct formula loop or
        // vDSP_vgen
        for (size_t i = (length + 1) / 2; i < length; ++i) {
          coeffs[i] = 2.0f - static_cast<float>(i) * scale;
        }
      }
      else {  // Double
        double N_minus_1 = static_cast<double>(length - 1);
        if (N_minus_1 <= 0) {  // length 1 case
          if (length > 0) coeffs[0] = 1.0;
          return coeffs;
        }
        double scale = 2.0 / N_minus_1;
        double start = 0.0;
        vDSP_vrampd(&start, &scale, coeffs.data(), 1, length / 2);  // Ramp up
        if (length % 2 != 0) {  // Odd length - middle sample is 1.0
          coeffs[length / 2] = 1.0;
        }
        // Ramp down
        for (size_t i = (length + 1) / 2; i < length; ++i) {
          coeffs[i] = 2.0 - static_cast<double>(i) * scale;
        }
      }
      return coeffs;
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::blackman_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      if constexpr (std::is_same_v<T, float>) {
        vDSP_blkman_window(
            coeffs.data(), length, 0);  // 0 for standard Blackman
      }
      else {
        vDSP_blkman_windowD(coeffs.data(), length, 0);
      }
      return coeffs;
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::flattop_window(size_t length) const
    {
      // vDSP does not have a direct flat top window. Needs manual
      // implementation. Formula: a0 - a1*cos(2pi*n/N) + a2*cos(4pi*n/N) -
      // a3*cos(6pi*n/N) + a4*cos(8pi*n/N) Use standard coefficients (e.g., from
      // SciPy) Can be implemented using vDSP vector math (vramp, vsmul, cos,
      // etc.)
      std::cerr << "AccelerateOmniDSPImpl::flattop_window not yet implemented."
                << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::gaussian_window(size_t length, RealT<T> stddev) const
    {
      // vDSP does not have a direct Gaussian window. Needs manual
      // implementation. Formula: exp(-0.5 * ((n - (N-1)/2) / (sigma *
      // (N-1)/2))^2) Use vDSP vector math (vramp, vsadd, vsdiv, vsq, vsmul,
      // vsadd, vexp)
      std::cerr << "AccelerateOmniDSPImpl::gaussian_window not yet implemented."
                << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::hamming_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      if constexpr (std::is_same_v<T, float>) {
        vDSP_hamm_window(coeffs.data(), length, 0);  // 0 for standard Hamming
      }
      else {
        vDSP_hamm_windowD(coeffs.data(), length, 0);
      }
      return coeffs;
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::hann_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      if constexpr (std::is_same_v<T, float>) {
        vDSP_hann_window(coeffs.data(), length, 0);  // 0 for standard Hann
      }
      else {
        vDSP_hann_windowD(coeffs.data(), length, 0);
      }
      return coeffs;
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::kaiser_window(size_t length, RealT<T> beta) const
    {
      // vDSP does not have a direct Kaiser window. Needs manual implementation.
      // Formula involves Bessel function I0. Requires implementing I0 or using
      // an approximation.
      std::cerr << "AccelerateOmniDSPImpl::kaiser_window not yet implemented."
                << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::rectangular_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      RealT<T> one = 1.0;
      if constexpr (std::is_same_v<T, float>) {
        vDSP_vfill(&one, coeffs.data(), 1, length);
      }
      else {
        vDSP_vfillD(&one, coeffs.data(), 1, length);
      }
      return coeffs;
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::triangular_window(size_t length) const
    {
      // Similar to Bartlett, but often defined slightly differently (ends at
      // zero vs Bartlett ends near zero). Needs manual implementation using
      // vDSP vector math.
      std::cerr
          << "AccelerateOmniDSPImpl::triangular_window not yet implemented."
          << std::endl;
      return std::unexpected(Status::UnsupportedFeature);
    }

    // --- Plan Factories ---
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>>
    AccelerateOmniDSPImpl::create_fft_plan(size_t length) const
    {
      try {
        // 1. Create the backend-specific implementation object
        auto pimpl_backend = std::make_unique<AccelerateFFTPlanImpl<T>>(length);

        // 2. Create the public Plan object using its private constructor,
        // passing the impl Need access to the private constructor (friend class
        // OmniDSP in FFTPlan) We achieve this by calling the constructor
        // directly. Note: Using std::make_unique requires a public constructor.
        // We construct manually and wrap.
        FFTPlan<T>* plan_ptr = new FFTPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<FFTPlan<T>>(plan_ptr);
      }
      catch (const std::bad_alloc&) {
        return std::unexpected(Status::AllocationError);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate FFTPlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(
            Status::BackendError);  // Or InvalidArgument if length is bad
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>>
    AccelerateOmniDSPImpl::create_rfft_plan(size_t length) const
    {
      try {
        auto pimpl_backend
            = std::make_unique<AccelerateRFFTPlanImpl<T>>(length);
        RFFTPlan<T>* plan_ptr = new RFFTPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<RFFTPlan<T>>(plan_ptr);
      }
      catch (const std::bad_alloc&) {
        return std::unexpected(Status::AllocationError);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate RFFTPlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(Status::BackendError);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>>
    AccelerateOmniDSPImpl::create_cqt_plan(
        const OmniDSP* owner,  // Pass owner for sub-plan creation
        RealT<T> sample_rate,
        RealT<T> min_freq,
        RealT<T> max_freq,
        int bins_per_octave) const
    {
      // CQTPlan doesn't use Pimpl directly. It needs the OmniDSP owner
      // to create its internal FFT/Resample plans using the *correct* backend
      // (Accelerate in this case). We just call the public CQTPlan constructor
      // (which is friended by OmniDSP).
      try {
        // Note: CQTPlan constructor is private/protected, but OmniDSP is a
        // friend. This factory method, being part of OmniDSPImpl (called by
        // OmniDSP), can facilitate. A clean way is to have the public
        // OmniDSP::create_cqt_plan call the private CQTPlan constructor
        // directly, passing 'this' as the owner. This backend implementation
        // doesn't need to do much specific here, unless backend choice affects
        // CQT setup parameters. Let's assume the main OmniDSP::create_cqt_plan
        // forwards to this, and we just construct.

        // The CQTPlan constructor itself will call back to
        // owner->create_fft_plan etc.
        CQTPlan<T>* plan_ptr = new CQTPlan<T>(
            owner, sample_rate, min_freq, max_freq, bins_per_octave);
        return std::unique_ptr<CQTPlan<T>>(plan_ptr);
      }
      catch (const std::bad_alloc&) {
        return std::unexpected(Status::AllocationError);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating CQTPlan: " << e.what() << std::endl;
        // Map specific CQT setup errors if possible
        return std::unexpected(
            Status::Failure);  // Or BackendError if sub-plan creation failed
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
    AccelerateOmniDSPImpl::create_resample_plan(
        double input_rate, double output_rate, size_t max_input_size) const
    {
      try {
        auto pimpl_backend = std::make_unique<AccelerateResamplePlanImpl<T>>(
            input_rate, output_rate, max_input_size);
        ResamplePlan<T>* plan_ptr
            = new ResamplePlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<ResamplePlan<T>>(plan_ptr);
      }
      catch (const std::bad_alloc&) {
        return std::unexpected(Status::AllocationError);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate ResamplePlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(Status::BackendError);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
    AccelerateOmniDSPImpl::create_convolution_plan(
        const std::vector<T>& kernel, ConvolutionMode mode) const
    {
      try {
        auto pimpl_backend
            = std::make_unique<AccelerateConvolutionPlanImpl<T>>(kernel, mode);
        ConvolutionPlan<T>* plan_ptr
            = new ConvolutionPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<ConvolutionPlan<T>>(plan_ptr);
      }
      catch (const std::bad_alloc&) {
        return std::unexpected(Status::AllocationError);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate ConvolutionPlanImpl: "
                  << e.what() << std::endl;
        return std::unexpected(Status::BackendError);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
    AccelerateOmniDSPImpl::create_correlation_plan(
        const std::vector<T>& kernel, ConvolutionMode mode) const
    {
      try {
        auto pimpl_backend
            = std::make_unique<AccelerateCorrelationPlanImpl<T>>(kernel, mode);
        CorrelationPlan<T>* plan_ptr
            = new CorrelationPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<CorrelationPlan<T>>(plan_ptr);
      }
      catch (const std::bad_alloc&) {
        return std::unexpected(Status::AllocationError);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate CorrelationPlanImpl: "
                  << e.what() << std::endl;
        return std::unexpected(Status::BackendError);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    // Add implementations for FilterPlan factory and filter design methods when
    // added

    //--------------------------------------------------------------------------
    // Explicit Template Instantiations
    //--------------------------------------------------------------------------
    // Instantiate templates for common types (float, double) for
    // AccelerateOmniDSPImpl methods

    // Define types for brevity
    using float_c = OmniDSP::ComplexT<float>;
    using double_c = OmniDSP::ComplexT<double>;

    // DSP Operations
    template OmniExpected<std::vector<float>> AccelerateOmniDSPImpl::convolve(
        const std::vector<float>&,
        const std::vector<float>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<double>> AccelerateOmniDSPImpl::convolve(
        const std::vector<double>&,
        const std::vector<double>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<float_c>> AccelerateOmniDSPImpl::convolve(
        const std::vector<float_c>&,
        const std::vector<float_c>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<double_c>>
    AccelerateOmniDSPImpl::convolve(
        const std::vector<double_c>&,
        const std::vector<double_c>&,
        ConvolutionMode) const;

    template OmniExpected<std::vector<float>> AccelerateOmniDSPImpl::correlate(
        const std::vector<float>&,
        const std::vector<float>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<double>> AccelerateOmniDSPImpl::correlate(
        const std::vector<double>&,
        const std::vector<double>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<float_c>>
    AccelerateOmniDSPImpl::correlate(
        const std::vector<float_c>&,
        const std::vector<float_c>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<double_c>>
    AccelerateOmniDSPImpl::correlate(
        const std::vector<double_c>&,
        const std::vector<double_c>&,
        ConvolutionMode) const;

    // One-off FFTs
    template OmniExpected<std::vector<float_c>> AccelerateOmniDSPImpl::fft(
        const std::vector<float_c>&) const;
    template OmniExpected<std::vector<double_c>> AccelerateOmniDSPImpl::fft(
        const std::vector<double_c>&) const;
    template OmniExpected<std::vector<float_c>> AccelerateOmniDSPImpl::ifft(
        const std::vector<float_c>&) const;
    template OmniExpected<std::vector<double_c>> AccelerateOmniDSPImpl::ifft(
        const std::vector<double_c>&) const;
    template OmniExpected<std::vector<float_c>> AccelerateOmniDSPImpl::rfft(
        const std::vector<float>&) const;
    template OmniExpected<std::vector<double_c>> AccelerateOmniDSPImpl::rfft(
        const std::vector<double>&) const;
    template OmniExpected<std::vector<float>> AccelerateOmniDSPImpl::irfft(
        const std::vector<float_c>&, size_t) const;
    template OmniExpected<std::vector<double>> AccelerateOmniDSPImpl::irfft(
        const std::vector<double_c>&, size_t) const;

    // Window Generation
    template OmniExpected<std::vector<float>>
        AccelerateOmniDSPImpl::bartlett_window(size_t) const;
    template OmniExpected<std::vector<double>>
        AccelerateOmniDSPImpl::bartlett_window(size_t) const;
    template OmniExpected<std::vector<float>>
        AccelerateOmniDSPImpl::blackman_window(size_t) const;
    template OmniExpected<std::vector<double>>
        AccelerateOmniDSPImpl::blackman_window(size_t) const;
    template OmniExpected<std::vector<float>>
        AccelerateOmniDSPImpl::flattop_window(size_t) const;
    template OmniExpected<std::vector<double>>
        AccelerateOmniDSPImpl::flattop_window(size_t) const;
    template OmniExpected<std::vector<float>>
    AccelerateOmniDSPImpl::gaussian_window(size_t, float) const;
    template OmniExpected<std::vector<double>>
    AccelerateOmniDSPImpl::gaussian_window(size_t, double) const;
    template OmniExpected<std::vector<float>>
        AccelerateOmniDSPImpl::hamming_window(size_t) const;
    template OmniExpected<std::vector<double>>
        AccelerateOmniDSPImpl::hamming_window(size_t) const;
    template OmniExpected<std::vector<float>>
        AccelerateOmniDSPImpl::hann_window(size_t) const;
    template OmniExpected<std::vector<double>>
        AccelerateOmniDSPImpl::hann_window(size_t) const;
    template OmniExpected<std::vector<float>>
    AccelerateOmniDSPImpl::kaiser_window(size_t, float) const;
    template OmniExpected<std::vector<double>>
    AccelerateOmniDSPImpl::kaiser_window(size_t, double) const;
    template OmniExpected<std::vector<float>>
        AccelerateOmniDSPImpl::rectangular_window(size_t) const;
    template OmniExpected<std::vector<double>>
        AccelerateOmniDSPImpl::rectangular_window(size_t) const;
    template OmniExpected<std::vector<float>>
        AccelerateOmniDSPImpl::triangular_window(size_t) const;
    template OmniExpected<std::vector<double>>
        AccelerateOmniDSPImpl::triangular_window(size_t) const;

    // Plan Factories
    template OmniExpected<std::unique_ptr<FFTPlan<float_c>>>
        AccelerateOmniDSPImpl::create_fft_plan(size_t) const;
    template OmniExpected<std::unique_ptr<FFTPlan<double_c>>>
        AccelerateOmniDSPImpl::create_fft_plan(size_t) const;

    template OmniExpected<std::unique_ptr<RFFTPlan<float>>>
        AccelerateOmniDSPImpl::create_rfft_plan(size_t) const;
    template OmniExpected<std::unique_ptr<RFFTPlan<double>>>
        AccelerateOmniDSPImpl::create_rfft_plan(size_t) const;

    template OmniExpected<std::unique_ptr<CQTPlan<float>>>
    AccelerateOmniDSPImpl::create_cqt_plan(
        const OmniDSP*, float, float, float, int) const;
    template OmniExpected<std::unique_ptr<CQTPlan<double>>>
    AccelerateOmniDSPImpl::create_cqt_plan(
        const OmniDSP*, double, double, double, int) const;

    template OmniExpected<std::unique_ptr<ResamplePlan<float>>>
    AccelerateOmniDSPImpl::create_resample_plan(double, double, size_t) const;
    template OmniExpected<std::unique_ptr<ResamplePlan<double>>>
    AccelerateOmniDSPImpl::create_resample_plan(double, double, size_t) const;

    template OmniExpected<std::unique_ptr<ConvolutionPlan<float>>>
    AccelerateOmniDSPImpl::create_convolution_plan(
        const std::vector<float>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<ConvolutionPlan<double>>>
    AccelerateOmniDSPImpl::create_convolution_plan(
        const std::vector<double>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<ConvolutionPlan<float_c>>>
    AccelerateOmniDSPImpl::create_convolution_plan(
        const std::vector<float_c>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<ConvolutionPlan<double_c>>>
    AccelerateOmniDSPImpl::create_convolution_plan(
        const std::vector<double_c>&, ConvolutionMode) const;

    template OmniExpected<std::unique_ptr<CorrelationPlan<float>>>
    AccelerateOmniDSPImpl::create_correlation_plan(
        const std::vector<float>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<CorrelationPlan<double>>>
    AccelerateOmniDSPImpl::create_correlation_plan(
        const std::vector<double>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<CorrelationPlan<float_c>>>
    AccelerateOmniDSPImpl::create_correlation_plan(
        const std::vector<float_c>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<CorrelationPlan<double_c>>>
    AccelerateOmniDSPImpl::create_correlation_plan(
        const std::vector<double_c>&, ConvolutionMode) const;

  }  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ACCELERATE
