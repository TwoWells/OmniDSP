/**
 * @file omnidsp.h
 * @brief Main public header file for the OmniDSP library, defining the main
 * OmniDSP class interface.
 */

#ifndef OMNIDSP_H
#define OMNIDSP_H

#include <complex>      // For std::complex parameter and return types
#include <cstddef>      // For size_t
#include <memory>       // For std::unique_ptr (Pimpl and Plan factories)
#include <string_view>  // Potentially for future parameters
#include <vector>       // For std::vector parameter and return types

// Include core types first, as they are used throughout the interface.
#include "core_types.h"

// Include Plan interface definitions
#include "convolution.h"
#include "cqt.h"
#include "fft.h"
#include "filter.h"  // Includes FIRFilterPlan, IIRFilterPlan, FIRFilterSpec, IIRFilterSpec, SecondOrderSection
#include "resample.h"  // Includes ResampleSpec

// Include the generated export header
#include "OmniDSP/omnidsp_export.h"

namespace OmniDSP {

  // Forward declaration for the implementation class (Pimpl idiom)
  namespace backend {
    class OmniDSPImpl;
  }  // namespace backend

  /**
   * @brief Main class providing access to OmniDSP functionalities.
   * @details This class acts as the primary entry point for using the library.
   * It manages the selected backend and provides methods for performing common
   * DSP operations directly or creating stateful Plan objects for optimized
   * repeated operations.
   * Use the static OmniDSP::create() factory function to instantiate this
   * class. This class is non-copyable but movable.
   */
  class OMNIDSP_EXPORT OmniDSP {
   public:
    /**
     * @brief Factory function to create an OmniDSP instance.
     * @param backend The desired computation backend. Defaults to
     * Backend::Default.
     * @return An OmniExpected<OmniDSP> containing the created instance on
     * success, or a Status error code.
     */
    [[nodiscard]] static OmniExpected<OmniDSP> create(
        Backend backend = Backend::Default);

    ~OmniDSP();                                    // Destructor
    OmniDSP(OmniDSP&& other) noexcept;             // Move constructor
    OmniDSP& operator=(OmniDSP&& other) noexcept;  // Move assignment

    // Delete Copy Semantics
    OmniDSP(const OmniDSP&) = delete;
    OmniDSP& operator=(const OmniDSP&) = delete;

    /** @brief Gets the backend currently used by this OmniDSP instance. */
    Backend get_backend() const;

    //-------------------------------------------------------------------------
    /** @defgroup DspOps DSP Operations (Member Methods)
     * @{ */
    //-------------------------------------------------------------------------

    /** @name Convolution and Correlation */
    ///@{
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> convolve(
        const std::vector<RealT<T>>& input,
        const std::vector<RealT<T>>& kernel,
        ConvolutionType type) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> convolve(
        const std::vector<ComplexT<T>>& input,
        const std::vector<ComplexT<T>>& kernel,
        ConvolutionType type) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> correlate(
        const std::vector<RealT<T>>& input,
        const std::vector<RealT<T>>& kernel,
        ConvolutionType type) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> correlate(
        const std::vector<ComplexT<T>>& input,
        const std::vector<ComplexT<T>>& kernel,
        ConvolutionType type) const;
    ///@}

    /** @name Fourier Transforms (One-Off) */
    ///@{
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> fft(
        const std::vector<ComplexT<T>>& input) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> ifft(
        const std::vector<ComplexT<T>>& input) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> rfft(
        const std::vector<RealT<T>>& input) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> irfft(
        const std::vector<ComplexT<T>>& input, size_t output_length) const;
    ///@}

    /** @name Window Coefficient Generation */
    ///@{
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> bartlett_window(
        size_t length) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> blackman_window(
        size_t length) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> flattop_window(
        size_t length) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> gaussian_window(
        size_t length, RealT<T> stddev) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> hamming_window(
        size_t length) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> hann_window(
        size_t length) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> kaiser_window(
        size_t length, RealT<T> beta) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> rectangular_window(
        size_t length) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> triangular_window(
        size_t length) const;
    ///@}
    /** @} */  // End of DspOps group

    //-------------------------------------------------------------------------
    /** @defgroup PlanFactories Plan Factory Methods
     * @brief Methods to create optimized Plan objects for repeated operations.
     * @{ */
    //-------------------------------------------------------------------------

    /** @brief Creates a plan for complex-to-complex FFTs. */
    template <typename T>  // T is complex type
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>> create_fft_plan(
        size_t length) const;

    /** @brief Creates a plan for real-to-complex FFTs. */
    template <typename T>  // T is real type
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>> create_rfft_plan(
        size_t length) const;

    /** @brief Creates a plan for Constant-Q Transforms (CQTs). */
    template <typename T>  // T is real type
    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>> create_cqt_plan(
        RealT<T> sample_rate,
        RealT<T> min_freq,
        RealT<T> max_freq,
        int bins_per_octave,
        const WindowSpec<T>& window_spec = WindowSpec<T>()) const;

    /**
     * @brief Creates a plan for resampling signals.
     * @tparam T The floating-point type (e.g., float, double).
     * @param spec The ResampleSpec defining input/output rates and quality.
     * @return An OmniExpected containing a unique_ptr to the ResamplePlan
     * interface on success, or a Status error code.
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
    create_resample_plan(const ResampleSpec& spec) const;

    /** @brief Creates a plan for 1D convolution. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
    create_convolution_plan(
        const std::vector<T>& kernel, ConvolutionType type) const;

    /** @brief Creates a plan for 1D cross-correlation. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
    create_correlation_plan(
        const std::vector<T>& kernel, ConvolutionType type) const;

    /** @brief Creates a plan for FIR filtering. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
    create_fir_filter_plan(const std::vector<T>& coefficients) const;

    /** @brief Creates a plan for IIR filtering using Second-Order Sections. */
    template <typename T>  // T is typically real
    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
    create_iir_filter_plan(
        const std::vector<SecondOrderSection<T>>& sos_coefficients) const;

    /** @} */  // End of PlanFactories group

    //-------------------------------------------------------------------------
    /** @defgroup FilterDesign Filter Design Methods
     * @brief Methods to design standard digital filters.
     * @{ */
    //-------------------------------------------------------------------------
    // TODO: Add design_fir_filter and design_iir_filter declarations here
    // Example:
    // template <typename T> // T is real type
    // [[nodiscard]] OmniExpected<std::vector<RealT<T>>> design_fir_filter(
    //      const FIRFilterSpec<T>& spec) const;
    // template <typename T> // T is real type
    // [[nodiscard]] OmniExpected<std::vector<SecondOrderSection<T>>>
    // design_iir_filter(
    //      const IIRFilterSpec<T>& spec) const;
    /** @} */  // End of FilterDesign group

   private:
    /** @brief Private constructor used by the static create factory function.
     */
    OmniDSP(std::unique_ptr<backend::OmniDSPImpl> impl);

    /** @brief Pointer to the implementation object (Pimpl idiom). */
    std::unique_ptr<backend::OmniDSPImpl> pimpl_;
  };

  // Template method definitions for member functions are typically defined in
  // the header file itself (if simple and not dependent on Impl) or in a
  // separate
  // "-inl.h" header file included at the end of this header, or explicitly
  // instantiated in a .cpp file. Since these depend on pimpl_, their
  // definitions MUST be in omnidsp.cpp.

}  // namespace OmniDSP

#endif  // OMNIDSP_H
