/**
 * @file cqt.hpp (Default)
 * @brief Declares the Default backend CQTPlanImpl class.
 */

#ifndef OMNIDSP_DEFAULT_CQT_HPP
#define OMNIDSP_DEFAULT_CQT_HPP

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>  // For Abstract::CQTPlanImpl and public CQTPlan (if needed for context)
#include <OmniDSP/fft.hpp>  // For RFFTPlan (used internally)
#include <OmniDSP/resample.hpp>  // For ResamplePlan (used internally) and ResampleSpec
#include <OmniDSP/window.hpp>  // For WindowSetup
#include <memory>              // For std::unique_ptr
#include <vector>

// Forward declare Abstract::Backend if owner_backend_ is of that type
// and not Default::Backend specifically.
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
  class CQTPlanImpl final : public Abstract::CQTPlanImpl<T> {
    using Complex = Utils::GetComplexType<T>;

   public:
    /**
     * @brief Constructs a CQTPlanImpl.
     * @param owner_backend Pointer to the backend instance creating this plan
     * (for creating FFT/Resample plans).
     * @param sample_rate Sample rate of the input signal.
     * @param min_freq Minimum frequency for CQT bins.
     * @param max_freq Maximum frequency for CQT bins.
     * @param bins_per_octave Number of CQT bins per octave.
     * @param window_setup Window setup for the CQT analysis windows.
     * @throws std::invalid_argument if parameters are invalid.
     * @throws OmniException if internal plan creation or setup fails.
     */
    CQTPlanImpl(
        const Abstract::Backend* owner_backend,
        T sample_rate,
        T min_freq,
        T max_freq,
        int bins_per_octave,
        const WindowSetup&
            window_setup);  // WindowSetup for CQT analysis windows

    ~CQTPlanImpl() override;

    // Disable copy/move
    CQTPlanImpl(const CQTPlanImpl&) = delete;
    CQTPlanImpl& operator=(const CQTPlanImpl&) = delete;
    CQTPlanImpl(CQTPlanImpl&&) = delete;
    CQTPlanImpl& operator=(CQTPlanImpl&&) = delete;

    // Interface methods
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<Complex> output) const override;
    size_t get_num_bins() const override;
    size_t get_num_output_frames(size_t input_length) const override;
    size_t get_hop_length() const override;

   private:
    const Abstract::Backend* owner_backend_;  // Non-owning pointer
    T sample_rate_;
    int bins_per_octave_;
    size_t hop_length_;

    struct CQTOctave {
      T octave_sample_rate;
      std::unique_ptr<ResamplePlan<T>>
          resampler;  // Plan to resample to this octave's rate
      std::unique_ptr<RFFTPlan<T>> rfft_plan;  // FFT plan for this octave
      std::vector<T> kernel_lengths;  // Length of each CQT kernel (window) in
                                      // samples at this octave_sample_rate
      std::vector<std::vector<Complex>>
          sparse_kernels;    // Pre-computed FFTs of CQT windows
      std::vector<T> freqs;  // Center frequencies for this octave
      size_t fft_length;     // FFT length used for this octave
    };
    std::vector<CQTOctave> octaves_;
    std::vector<T>
        all_bin_frequencies_;  // All CQT bin frequencies across octaves

    // Internal buffer for resampled audio per octave
    mutable std::vector<T> resampled_buffer_;
    // Internal buffer for FFT output per octave
    mutable std::vector<Complex> fft_output_buffer_;
  };

  // Explicit template instantiations
  extern template class CQTPlanImpl<F32>;
  extern template class CQTPlanImpl<F64>;

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_CQT_HPP
