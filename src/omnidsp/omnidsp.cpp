/**
 * @file omnidsp.cpp
 * @brief Implements the OmniDSP class methods, handling backend selection and
 * forwarding calls.
 */

#include "OmniDSP/omnidsp.h"  // Corresponding header

// Include Pimpl interface definition (defined below or in a separate backend.h)
// #include "backend/backend.h"

// Include concrete backend implementation headers (Placeholders)
// These headers should define classes inheriting from OmniDSPImpl
#include "backend/stub/stub_backend.h"              // Defines StubOmniDSPImpl
#ifdef USE_ACCELERATE                               // Use actual CMake flag
#include "backend/accelerate/accelerate_backend.h"  // Defines AccelerateOmniDSPImpl
#endif
#ifdef USE_ONEMKL                           // Use actual CMake flag
#include "backend/onemkl/onemkl_backend.h"  // Defines OneMKLOmniDSPImpl
#endif

#include <iostream>   // Temporary for debug messages (remove later)
#include <memory>     // For std::unique_ptr, std::make_unique
#include <stdexcept>  // For std::runtime_error (potentially)
#include <utility>    // For std::move, std::pair
#include <vector>

namespace OmniDSP {
  namespace backend {

    /**
     * @brief Abstract base class defining the interface for backend
     * implementations (Pimpl).
     * @details Concrete backend implementations (Default, Accelerate, oneMKL)
     * inherit from this class.
     */
    class OmniDSPImpl {
     public:
      /** @brief Virtual destructor is required for base classes with virtual
       * functions. */
      virtual ~OmniDSPImpl() = default;

      /** @brief Gets the backend type associated with this implementation. */
      virtual Backend get_backend() const = 0;

      // === Virtual methods corresponding to OmniDSP public API ===

      // --- DSP Operations ---
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>>
      convolve(  // Renamed
          const std::vector<RealT<T>>& input,
          const std::vector<RealT<T>>& kernel,
          ConvolutionMode mode) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>>
      convolve(  // Renamed
          const std::vector<ComplexT<T>>& input,
          const std::vector<ComplexT<T>>& kernel,
          ConvolutionMode mode) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>>
      correlate(  // Renamed
          const std::vector<RealT<T>>& input,
          const std::vector<RealT<T>>& kernel,
          ConvolutionMode mode) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>>
      correlate(  // Renamed
          const std::vector<ComplexT<T>>& input,
          const std::vector<ComplexT<T>>& kernel,
          ConvolutionMode mode) const
          = 0;

      // --- One-off FFTs ---
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>>
      fft(  // Added
          const std::vector<ComplexT<T>>& input) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>>
      ifft(  // Added
          const std::vector<ComplexT<T>>& input) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>>
      rfft(  // Added
          const std::vector<RealT<T>>& input) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> irfft(  // Added
          const std::vector<ComplexT<T>>& input,
          size_t output_length) const
          = 0;

      // --- Window Generation ---
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> bartlett_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> blackman_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> flattop_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> gaussian_window(
          size_t length, RealT<T> stddev) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> hamming_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> hann_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> kaiser_window(
          size_t length, RealT<T> beta) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>>
      rectangular_window(size_t length) const = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>>
      triangular_window(size_t length) const = 0;

      // --- Plan Factories ---
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlan<T>>>
      create_fft_plan(size_t length) const = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlan<T>>>
      create_rfft_plan(size_t length) const = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlan<T>>>
      create_cqt_plan(
          RealT<T> sample_rate,
          RealT<T> min_freq,
          RealT<T> max_freq,
          int bins_per_octave) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlan<T>>>
      create_resample_plan(
          double input_rate, double output_rate, size_t max_input_size) const
          = 0;

      // Add virtual methods for ConvolutionPlan/CorrelationPlan factories when
      // added to OmniDSP template<typename T>
      // [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
      // create_convolution_plan(...) const = 0; template<typename T>
      // [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
      // create_correlation_plan(...) const = 0;

      // Add virtual methods for FilterPlan factory when added to OmniDSP
      // template<typename T>
      // [[nodiscard]] virtual OmniExpected<std::unique_ptr<FilterPlan<T>>>
      // create_iir_filter_plan(...) const = 0;

    };  // class OmniDSPImpl

  }  // namespace backend

  //--------------------------------------------------------------------------
  // OmniDSP Method Definitions
  //--------------------------------------------------------------------------

  // --- Factory ---

  [[nodiscard]] /* static */ OmniExpected<OmniDSP> OmniDSP::create(
      Backend backend)
  {
    std::unique_ptr<backend::OmniDSPImpl> pimpl = nullptr;

    try {
      switch (backend) {
        case Backend::Accelerate:
#ifdef USE_ACCELERATE  // Use actual CMake flag
          // std::cout << "Attempting to create Accelerate backend..." <<
          // std::endl; // Debug
          pimpl = std::make_unique<backend::AccelerateOmniDSPImpl>();
          // std::cout << "Accelerate backend created." << std::endl; // Debug
#else
          std::cerr << "Warning: Accelerate backend requested but not enabled "
                       "during build (USE_ACCELERATE=OFF)."
                    << std::endl;  // Informative message
          return std::unexpected(Status::UnsupportedFeature);
#endif
          break;

        case Backend::OneMKL:
#ifdef USE_ONEMKL  // Use actual CMake flag
          // std::cout << "Attempting to create oneMKL backend..." << std::endl;
          // // Debug
          pimpl = std::make_unique<backend::OneMKLOmniDSPImpl>();
          // std::cout << "oneMKL backend created." << std::endl; // Debug
#else
          std::cerr
              << "Warning: oneMKL backend requested but not enabled during "
                 "build (USE_ONEMKL=OFF)."
              << std::endl;  // Informative message
          return std::unexpected(Status::UnsupportedFeature);
#endif
          break;

        case Backend::Default:
        default:  // Fallback to Default
          // std::cout << "Creating Default backend..." << std::endl; // Debug
          pimpl = std::make_unique<backend::DefaultOmniDSPImpl>();
          // std::cout << "Default backend created." << std::endl; // Debug
          break;
      }

      // Check if pimpl was successfully created (make_unique throws on
      // allocation failure)
      if (!pimpl) {
        // This case might be redundant if make_unique throws bad_alloc
        // Or if backend constructors can fail and return nullptr (bad practice)
        // std::cerr << "Backend implementation creation failed unexpectedly."
        // << std::endl; // Debug
        return std::unexpected(Status::BackendError);
      }

      // If backend impl creation succeeded, create OmniDSP instance
      OmniDSP dsp_instance(std::move(pimpl));  // Calls private constructor
      return dsp_instance;  // Return the created instance (implicitly moves)
    }
    catch (const std::bad_alloc& e) {
      std::cerr << "Error: Memory allocation failed during backend creation: "
                << e.what() << std::endl;  // Error message
      return std::unexpected(Status::AllocationError);
    }
    catch (const std::exception& e) {
      // Catch potential exceptions from backend Impl constructors
      std::cerr << "Error: Exception during backend initialization: "
                << e.what() << std::endl;  // Error message
      return std::unexpected(
          Status::BackendError);  // Or a more specific error?
    }
    catch (...) {
      std::cerr << "Error: Unknown exception during backend creation."
                << std::endl;                   // Error message
      return std::unexpected(Status::Failure);  // Generic failure
    }
  }

  // --- Constructor / Destructor / Move Operations ---

  // Private Constructor: Takes ownership of the implementation pointer
  OmniDSP::OmniDSP(std::unique_ptr<backend::OmniDSPImpl> impl)
      : pimpl_(std::move(impl))
  {}

  // Destructor: Needs to be defined here where OmniDSPImpl is complete.
  OmniDSP::~OmniDSP() = default;

  // Move Constructor: Needs to be defined here. = default is sufficient.
  OmniDSP::OmniDSP(OmniDSP&& other) noexcept = default;

  // Move Assignment Operator: Needs to be defined here. = default is
  // sufficient.
  OmniDSP& OmniDSP::operator=(OmniDSP&& other) noexcept = default;

  // --- Public Member Functions (Forwarding to Pimpl) ---

  Backend OmniDSP::get_backend() const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in get_backend.");
    return pimpl_->get_backend();
  }

  // --- DSP Operations ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
  OmniDSP::convolve(  // Renamed
      const std::vector<RealT<T>>& input,
      const std::vector<RealT<T>>& kernel,
      ConvolutionMode mode) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in convolve.");
    return pimpl_->convolve(input, kernel, mode);  // Renamed
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
  OmniDSP::convolve(  // Renamed
      const std::vector<ComplexT<T>>& input,
      const std::vector<ComplexT<T>>& kernel,
      ConvolutionMode mode) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in convolve.");
    return pimpl_->convolve(input, kernel, mode);  // Renamed
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
  OmniDSP::correlate(  // Renamed
      const std::vector<RealT<T>>& input,
      const std::vector<RealT<T>>& kernel,
      ConvolutionMode mode) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in correlate.");
    return pimpl_->correlate(input, kernel, mode);  // Renamed
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
  OmniDSP::correlate(  // Renamed
      const std::vector<ComplexT<T>>& input,
      const std::vector<ComplexT<T>>& kernel,
      ConvolutionMode mode) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in correlate.");
    return pimpl_->correlate(input, kernel, mode);  // Renamed
  }

  // --- One-off FFTs ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OmniDSP::fft(  // Added
      const std::vector<ComplexT<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance in fft.");
    return pimpl_->fft(input);  // Added
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OmniDSP::ifft(  // Added
      const std::vector<ComplexT<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance in ifft.");
    return pimpl_->ifft(input);  // Added
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OmniDSP::rfft(  // Added
      const std::vector<RealT<T>>& input) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance in rfft.");
    return pimpl_->rfft(input);  // Added
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::irfft(  // Added
      const std::vector<ComplexT<T>>& input,
      size_t output_length) const
  {
    if (!pimpl_) throw std::runtime_error("Invalid OmniDSP instance in irfft.");
    return pimpl_->irfft(input, output_length);  // Added
  }

  // --- Window Generation ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::bartlett_window(
      size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in bartlett_window.");
    return pimpl_->bartlett_window<T>(length);
  }
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::blackman_window(
      size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in blackman_window.");
    return pimpl_->blackman_window<T>(length);
  }
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::flattop_window(
      size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in flattop_window.");
    return pimpl_->flattop_window<T>(length);
  }
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::gaussian_window(
      size_t length, RealT<T> stddev) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in gaussian_window.");
    return pimpl_->gaussian_window<T>(length, stddev);
  }
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::hamming_window(
      size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in hamming_window.");
    return pimpl_->hamming_window<T>(length);
  }
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::hann_window(
      size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in hann_window.");
    return pimpl_->hann_window<T>(length);
  }
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::kaiser_window(
      size_t length, RealT<T> beta) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in kaiser_window.");
    return pimpl_->kaiser_window<T>(length, beta);
  }
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::rectangular_window(
      size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in rectangular_window.");
    return pimpl_->rectangular_window<T>(length);
  }
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<RealT<T>>> OmniDSP::triangular_window(
      size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in triangular_window.");
    return pimpl_->triangular_window<T>(length);
  }

  // --- Plan Factories ---
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>>
  OmniDSP::create_fft_plan(size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in create_fft_plan.");
    return pimpl_->create_fft_plan<T>(length);
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>>
  OmniDSP::create_rfft_plan(size_t length) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in create_rfft_plan.");
    return pimpl_->create_rfft_plan<T>(length);
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>>
  OmniDSP::create_cqt_plan(
      RealT<T> sample_rate,
      RealT<T> min_freq,
      RealT<T> max_freq,
      int bins_per_octave) const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in create_cqt_plan.");
    return pimpl_->create_cqt_plan<T>(
        sample_rate, min_freq, max_freq, bins_per_octave);
  }

  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
  OmniDSP::create_resample_plan(
      double input_rate, double output_rate, size_t max_input_size) const
  {
    if (!pimpl_)
      throw std::runtime_error(
          "Invalid OmniDSP instance in create_resample_plan.");
    return pimpl_->create_resample_plan<T>(
        input_rate, output_rate, max_input_size);
  }

  // Add implementations for create_convolution_plan, create_correlation_plan,
  // create_iir_filter_plan when declared

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate templates for common types (float, double) to ensure code
  // generation.

  // Define types for brevity
  using float_c = OmniDSP::ComplexT<float>;
  using double_c = OmniDSP::ComplexT<double>;

  // DSP Operations
  template OmniExpected<std::vector<float>> OmniDSP::convolve(
      const std::vector<float>&,
      const std::vector<float>&,
      ConvolutionMode) const;  // Renamed
  template OmniExpected<std::vector<double>> OmniDSP::convolve(
      const std::vector<double>&,
      const std::vector<double>&,
      ConvolutionMode) const;  // Renamed
  template OmniExpected<std::vector<float_c>> OmniDSP::convolve(
      const std::vector<float_c>&,
      const std::vector<float_c>&,
      ConvolutionMode) const;  // Renamed
  template OmniExpected<std::vector<double_c>> OmniDSP::convolve(
      const std::vector<double_c>&,
      const std::vector<double_c>&,
      ConvolutionMode) const;  // Renamed

  template OmniExpected<std::vector<float>> OmniDSP::correlate(
      const std::vector<float>&,
      const std::vector<float>&,
      ConvolutionMode) const;  // Renamed
  template OmniExpected<std::vector<double>> OmniDSP::correlate(
      const std::vector<double>&,
      const std::vector<double>&,
      ConvolutionMode) const;  // Renamed
  template OmniExpected<std::vector<float_c>> OmniDSP::correlate(
      const std::vector<float_c>&,
      const std::vector<float_c>&,
      ConvolutionMode) const;  // Renamed
  template OmniExpected<std::vector<double_c>> OmniDSP::correlate(
      const std::vector<double_c>&,
      const std::vector<double_c>&,
      ConvolutionMode) const;  // Renamed

  // One-off FFTs
  template OmniExpected<std::vector<float_c>> OmniDSP::fft(
      const std::vector<float_c>&) const;  // Added
  template OmniExpected<std::vector<double_c>> OmniDSP::fft(
      const std::vector<double_c>&) const;  // Added
  template OmniExpected<std::vector<float_c>> OmniDSP::ifft(
      const std::vector<float_c>&) const;  // Added
  template OmniExpected<std::vector<double_c>> OmniDSP::ifft(
      const std::vector<double_c>&) const;  // Added
  template OmniExpected<std::vector<float_c>> OmniDSP::rfft(
      const std::vector<float>&) const;  // Added
  template OmniExpected<std::vector<double_c>> OmniDSP::rfft(
      const std::vector<double>&) const;  // Added
  template OmniExpected<std::vector<float>> OmniDSP::irfft(
      const std::vector<float_c>&, size_t) const;  // Added
  template OmniExpected<std::vector<double>> OmniDSP::irfft(
      const std::vector<double_c>&, size_t) const;  // Added

  // Window Generation
  template OmniExpected<std::vector<float>> OmniDSP::bartlett_window(
      size_t) const;
  template OmniExpected<std::vector<double>> OmniDSP::bartlett_window(
      size_t) const;
  template OmniExpected<std::vector<float>> OmniDSP::blackman_window(
      size_t) const;
  template OmniExpected<std::vector<double>> OmniDSP::blackman_window(
      size_t) const;
  template OmniExpected<std::vector<float>> OmniDSP::flattop_window(
      size_t) const;
  template OmniExpected<std::vector<double>> OmniDSP::flattop_window(
      size_t) const;
  template OmniExpected<std::vector<float>> OmniDSP::gaussian_window(
      size_t, float) const;
  template OmniExpected<std::vector<double>> OmniDSP::gaussian_window(
      size_t, double) const;
  template OmniExpected<std::vector<float>> OmniDSP::hamming_window(
      size_t) const;
  template OmniExpected<std::vector<double>> OmniDSP::hamming_window(
      size_t) const;
  template OmniExpected<std::vector<float>> OmniDSP::hann_window(size_t) const;
  template OmniExpected<std::vector<double>> OmniDSP::hann_window(size_t) const;
  template OmniExpected<std::vector<float>> OmniDSP::kaiser_window(
      size_t, float) const;
  template OmniExpected<std::vector<double>> OmniDSP::kaiser_window(
      size_t, double) const;
  template OmniExpected<std::vector<float>> OmniDSP::rectangular_window(
      size_t) const;
  template OmniExpected<std::vector<double>> OmniDSP::rectangular_window(
      size_t) const;
  template OmniExpected<std::vector<float>> OmniDSP::triangular_window(
      size_t) const;
  template OmniExpected<std::vector<double>> OmniDSP::triangular_window(
      size_t) const;

  // Plan Factories
  template OmniExpected<std::unique_ptr<FFTPlan<float_c>>>
      OmniDSP::create_fft_plan(size_t) const;
  template OmniExpected<std::unique_ptr<FFTPlan<double_c>>>
      OmniDSP::create_fft_plan(size_t) const;

  template OmniExpected<std::unique_ptr<RFFTPlan<float>>>
      OmniDSP::create_rfft_plan(size_t) const;
  template OmniExpected<std::unique_ptr<RFFTPlan<double>>>
      OmniDSP::create_rfft_plan(size_t) const;

  template OmniExpected<std::unique_ptr<CQTPlan<float>>>
  OmniDSP::create_cqt_plan(float, float, float, int) const;
  template OmniExpected<std::unique_ptr<CQTPlan<double>>>
  OmniDSP::create_cqt_plan(double, double, double, int) const;

  template OmniExpected<std::unique_ptr<ResamplePlan<float>>>
  OmniDSP::create_resample_plan(double, double, size_t) const;
  template OmniExpected<std::unique_ptr<ResamplePlan<double>>>
  OmniDSP::create_resample_plan(double, double, size_t) const;

  // Add instantiations for ConvolutionPlan/CorrelationPlan/FilterPlan factories
  // when added

}  // namespace OmniDSP
