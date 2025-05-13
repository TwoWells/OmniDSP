/**
 * @file omnidsp.hpp
 * @brief Main public header file for the OmniDSP library, defining the main
 * OmniDSP class interface and its configuration Builder.
 */

#ifndef OMNIDSP_HPP
#define OMNIDSP_HPP

#include <complex>
#include <cstddef>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"
// Include individual operation headers - these define Plan/Processor classes
// and their associated Params/Design/Coefs structs or forward declare them.
#include "OmniDSP/convolution.hpp"
#include "OmniDSP/core_types.hpp"
#include "OmniDSP/cqt.hpp"
#include "OmniDSP/fft.hpp"
#include "OmniDSP/fir_filter.hpp"
#include "OmniDSP/iir_filter.hpp"
#include "OmniDSP/resample.hpp"
#include "OmniDSP/window.hpp"

// Include Params headers specifically if they are not fully covered by
// operation headers
#include "OmniDSP/params/convolution.hpp"
#include "OmniDSP/params/cqt.hpp"
#include "OmniDSP/params/fft.hpp"
#include "OmniDSP/params/fir_filter.hpp"
#include "OmniDSP/params/iir_filter.hpp"
#include "OmniDSP/params/resample.hpp"

// Include the backend interface definition
#include "interface/backend.hpp"  // Defines Abstract::Backend

namespace OmniDSP {

  template <typename T>
  struct always_false : std::false_type {};

  class OMNIDSP_EXPORT OmniDSP {
   public:
    class OMNIDSP_EXPORT Builder {
     public:
      explicit Builder(std::optional<BackendType> primary_type = std::nullopt);
      Builder& add_override(OperationCategory category, BackendType type);
      [[nodiscard]] OmniExpected<OmniDSP> build() const;

     private:
      BackendType primary_backend_type_;
      std::map<OperationCategory, BackendType> category_overrides_map_;
    };

    [[nodiscard]] static OmniExpected<OmniDSP> create(
        BackendType backend_type = BackendType::Default);
    [[nodiscard]] static OmniExpected<OmniDSP> create(
        BackendType primary_backend_type,
        const std::map<OperationCategory, BackendType>& category_overrides);

    ~OmniDSP();
    OmniDSP(OmniDSP&& other) noexcept;
    OmniDSP& operator=(OmniDSP&& other) noexcept;
    OmniDSP(const OmniDSP&) = delete;
    OmniDSP& operator=(const OmniDSP&) = delete;

    BackendType get_backend() const;

    // One-off DSP Operations (Declarations only)
    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetRealType<T>>> convolve(
        const std::vector<Utils::GetRealType<T>>& input,
        const std::vector<Utils::GetRealType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>  // T is C32 or C64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> convolve(
        const std::vector<Utils::GetComplexType<T>>& input,
        const std::vector<Utils::GetComplexType<T>>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;

    template <typename T>  // T can be F32, F64 (real types)
    [[nodiscard]] OmniExpected<std::vector<Utils::GetRealType<T>>> correlate(
        const std::vector<Utils::GetRealType<T>>& input,
        const std::vector<Utils::GetRealType<T>>& kernel,  // or template
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;
    template <typename T>  // T can be C32, C64 (complex types)
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> correlate(
        const std::vector<Utils::GetComplexType<T>>& input,
        const std::vector<Utils::GetComplexType<T>>& kernel,  // or template
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const;

    template <typename T>  // T is C32 or C64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> fft(
        const std::vector<Utils::GetComplexType<T>>& input) const;
    template <typename T>  // T is C32 or C64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>> ifft(
        const std::vector<Utils::GetComplexType<T>>& input) const;
    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::vector<Utils::GetComplexType<T>>>
    rfft(  // Output is complex
        const std::vector<Utils::GetRealType<T>>& input) const;
    template <typename T>  // T is F32 or F64 (real type for output)
    [[nodiscard]] OmniExpected<std::vector<T>> irfft(        // Output is real
        const std::vector<Utils::GetComplexType<T>>& input,  // Input is complex
        size_t output_length) const;

    template <typename T>  // T is F32 or F64
    [[nodiscard]] OmniExpected<std::vector<T>> generate_window(
        const WindowSetup& setup) const;

    // Deprecated specific window functions (kept for backward compatibility if
    // desired)
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> bartlett_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> blackman_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> flattop_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> gaussian_window(
        size_t length, double stddev) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> hamming_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> hann_window(size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> kaiser_window(
        size_t length, double beta) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> rectangular_window(
        size_t length) const;
    /** @deprecated Use generate_window(const WindowSetup&) instead. */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<T>> triangular_window(
        size_t length) const;

    // Plan and Processor Factory Methods (Declarations only)
    template <typename T_Complex>
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T_Complex>>> create_plan(
        const FFTParams& params) const;
    template <typename T_Real>
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T_Real>>> create_plan(
        const RFFTParams& params) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>> create_plan(
        const ConvolutionParams& params,
        const std::vector<T>& kernel_coeffs) const;
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>> create_plan(
        const CorrelationParams& params,
        const std::vector<T>& kernel_coeffs) const;

    template <typename T>
    [[nodiscard]] OmniExpected<
        std::unique_ptr<FIRFilterProcessor<T>>> /* TODO: FIRFilterProcessor */
    create_processor(const FIRFilterParams& params) const;
    template <typename T>
    [[nodiscard]] OmniExpected<
        std::unique_ptr<FIRFilterProcessor<T>>> /* TODO: FIRFilterProcessor */
    create_processor(const FIRCoefs<T>& coeffs) const;
    template <typename T_Real>
    [[nodiscard]] OmniExpected<std::unique_ptr<
        IIRFilterProcessor<T_Real>>> /* TODO: IIRFilterProcessor */
    create_processor(const IIRFilterParams& params) const;
    template <typename T_Real>
    [[nodiscard]] OmniExpected<std::unique_ptr<
        IIRFilterProcessor<T_Real>>> /* TODO: IIRFilterProcessor */
    create_processor(const std::vector<IIRFilterCoef>& sos_coeffs) const;
    template <typename T_Real>
    [[nodiscard]] OmniExpected<std::unique_ptr<
        ResampleProcessor<T_Real>>> /* TODO: ResampleProcessor */
    create_processor(const ResampleParams& params) const;
    template <typename T_Real>
    [[nodiscard]] OmniExpected<std::unique_ptr<
        ResampleProcessor<T_Real>>> /* TODO: ResampleProcessor */
    create_processor(const Design::Resample& spec) const;
    template <typename T_Real>
    [[nodiscard]] OmniExpected<
        std::unique_ptr<CQTProcessor<T_Real>>> /* TODO: CQTProcessor */
    create_processor(const CQTParams& params) const;
    template <typename T_Real>
    [[nodiscard]] OmniExpected<
        std::unique_ptr<CQTProcessor<T_Real>>> /* TODO: CQTProcessor */
    create_processor(const Design::CQT& spec) const;

    // Filter Design Methods (Declarations only)
    template <typename T>
    [[nodiscard]] OmniExpected<FIRCoefs<T>> design_fir_filter(
        const FIRFilterParams& params) const;
    template <typename T_Real>
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>> design_iir_filter(
        const IIRFilterParams& params) const;

   private:
    OmniDSP(std::unique_ptr<Abstract::Backend> impl);  // Private constructor
    std::unique_ptr<Abstract::Backend> pimpl_;
  };

}  // namespace OmniDSP

// Include the template implementations at the end of the header
#include "OmniDSP/omnidsp.tpp"

#endif  // OMNIDSP_HPP
