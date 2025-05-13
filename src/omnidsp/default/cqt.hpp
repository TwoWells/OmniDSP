/**
 * @file cqt.hpp (Default)
 * @brief Declares the Default backend CQTProcessorImpl class.
 */

#ifndef OMNIDSP_DEFAULT_CQT_HPP
#define OMNIDSP_DEFAULT_CQT_HPP

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>       // For Abstract::CQTPlanImpl and Design::CQT
#include <OmniDSP/fft.hpp>       // For RFFTPlan (used internally)
#include <OmniDSP/resample.hpp>  // For ResamplePlan (used internally)
#include <OmniDSP/window.hpp>    // For WindowSetup
#include <memory>                // For std::unique_ptr
#include <span>
#include <vector>

// Forward declare Abstract::Backend
namespace OmniDSP::Abstract {
  class Backend;
}

namespace OmniDSP::Default {

  /**
   * @brief Default backend implementation for a Constant-Q Transform (CQT)
   * Plan.
   * @tparam T The REAL data type (F32 or F64).
   */
  template <typename T>
  class CQTProcessorImpl final : public Abstract::CQTProcessorImpl<T> {
    using Complex = Utils::GetComplexType<T>;

   public:
    /**
     * @brief Constructs a CQTProcessorImpl using a fully resolved Design::CQT.
     * @param owner_backend Pointer to the backend instance creating this plan
     * (for creating FFT/Resample sub-plans).
     * @param spec The fully resolved Constant-Q Transform specification.
     * @throws OmniException if internal plan creation or setup fails.
     */
    CQTProcessorImpl(
        const Abstract::Backend* owner_backend, const Design::CQT& spec);

    ~CQTProcessorImpl() override;

    // Disable copy/move
    CQTProcessorImpl(const CQTProcessorImpl&) = delete;
    CQTProcessorImpl& operator=(const CQTProcessorImpl&) = delete;
    CQTProcessorImpl(CQTProcessorImpl&&) = delete;
    CQTProcessorImpl& operator=(CQTProcessorImpl&&) = delete;

    // Interface methods
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<Complex> output) const override;
    [[nodiscard]] Status reset() override;  // Added reset method
    size_t get_num_bins() const override;
    size_t get_num_output_frames(size_t input_length) const override;
    size_t get_hop_length() const override;

   private:
    const Abstract::Backend* owner_backend_;  // Non-owning pointer
    Design::CQT spec_;  // Store the resolved specification

    // Internal structure to hold processed data for each octave
    struct ProcessedCQTOctave {
      double octave_sample_rate;
      std::unique_ptr<ResampleProcessor<T>>
          resampler;  // Plan to resample to this octave's rate (nullable if
                      // original SR)
      std::unique_ptr<RFFTPlan<T>> rfft_plan;  // FFT plan for this octave
      std::vector<std::vector<Complex>>
          sparse_kernels_fft;  // Pre-computed FFTs of CQT analysis kernels for
                               // this octave
      size_t fft_length;       // FFT length used for this octave (from spec)
      std::vector<double>
          bin_frequencies;  // Frequencies processed in this octave (from spec)
      // kernel_lengths_samples are in
      // spec_.octave_specs[i].kernel_lengths_samples

      // Constructor for ProcessedCQTOctave
      ProcessedCQTOctave(
          double osr,
          std::unique_ptr<ResampleProcessor<T>> rs_processor,
          std::unique_ptr<RFFTPlan<T>> rf_plan,
          std::vector<std::vector<Complex>> sk_fft,
          size_t fl,
          std::vector<double> bf)
          : octave_sample_rate(osr),
            resampler(std::move(rs_processor)),
            rfft_plan(std::move(rf_plan)),
            sparse_kernels_fft(std::move(sk_fft)),
            fft_length(fl),
            bin_frequencies(std::move(bf))
      {}
    };
    std::vector<ProcessedCQTOctave> processed_octaves_;

    // Internal buffers (mutable for use in const execute method)
    // These are for processing a single input block to execute().
    // If execute() itself becomes stateful (e.g. overlap-add), more state might
    // be needed here.
    mutable std::vector<T> resampled_buffer_;
    mutable std::vector<Complex> frame_fft_buffer_;
    // temp_kernel_real_buffer_ was used during construction, not a persistent
    // state for execute/reset.
  };

  // Explicit template instantiations
  extern template class CQTProcessorImpl<F32>;
  extern template class CQTProcessorImpl<F64>;

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_CQT_HPP
