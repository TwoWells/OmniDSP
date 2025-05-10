/**
 * @file cqt.cpp (Default)
 * @brief Implements the Default backend CQTPlanImpl class.
 */

#include "cqt.hpp"  // Corresponding header

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/fft.hpp>
#include <OmniDSP/resample.hpp>
#include <OmniDSP/window.hpp>  // For WindowSetup, WindowParams, and OmniDSP::generate_window
#include <algorithm>  // For std::sort, std::min, std::max, std::transform
#include <bit>        // For std::bit_ceil (C++20)
#include <cmath>    // For std::log2, std::pow, std::round, std::ceil, std::abs
#include <numeric>  // For std::iota
#include <stdexcept>
#include <string>  // For exception messages
#include <vector>

#include "interface/backend.hpp"  // For Abstract::Backend
#include "spdlog/spdlog.h"        // For logging

namespace OmniDSP::Default {

  using ::OmniDSP::OmniException;  // Bring OmniException into scope

  //--------------------------------------------------------------------------
  // CQTPlanImpl Method Implementations
  //--------------------------------------------------------------------------
  template <typename T>
  CQTPlanImpl<T>::CQTPlanImpl(
      const Abstract::Backend* owner_backend,
      T sample_rate,
      T min_freq,
      T max_freq,
      int bins_per_octave,
      const WindowSetup& cqt_window_setup)
      : owner_backend_(owner_backend),
        sample_rate_(sample_rate),
        bins_per_octave_(bins_per_octave),
        hop_length_(0)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner_backend_) {
      logger->error("CQTPlanImpl: owner_backend is null.");
      throw OmniException(
          "CQTPlanImpl requires a non-null owner_backend.",
          Status::InvalidArgument);
    }
    if (sample_rate <= 0) {
      logger->error(
          "CQTPlanImpl: Sample rate ({}) must be positive.", sample_rate);
      throw OmniException(
          "Sample rate must be positive.", Status::InvalidArgument);
    }
    if (min_freq <= 0 || max_freq <= 0 || min_freq >= max_freq) {
      logger->error(
          "CQTPlanImpl: Invalid frequency range [{}, {}].", min_freq, max_freq);
      throw OmniException(
          "Invalid CQT frequency range.", Status::InvalidArgument);
    }
    if (bins_per_octave <= 0) {
      logger->error(
          "CQTPlanImpl: Bins per octave ({}) must be positive.",
          bins_per_octave);
      throw OmniException(
          "Bins per octave must be positive.", Status::InvalidArgument);
    }

    logger->debug(
        "Creating Default::CQTPlanImpl: SR={}, MinF={}, MaxF={}, Bins/Oct={}, "
        "WinType={}",
        sample_rate,
        min_freq,
        max_freq,
        bins_per_octave,
        get_window_type_name(cqt_window_setup.type));

    int num_octaves
        = static_cast<int>(std::ceil(std::log2(max_freq / min_freq)));
    if (num_octaves <= 0) num_octaves = 1;

    T current_min_freq = min_freq;
    T quality_factor_q = static_cast<T>(
        1.0
        / (std::pow(static_cast<T>(2.0), static_cast<T>(1.0) / bins_per_octave_)
           - static_cast<T>(1.0)));

    T min_kernel_len_approx_samples
        = quality_factor_q * sample_rate_ / min_freq;
    hop_length_ = static_cast<size_t>(std::max(
        static_cast<T>(1.0),
        min_kernel_len_approx_samples / static_cast<T>(4.0)));
    logger->debug("CQTPlanImpl: Estimated hop_length = {}", hop_length_);

    const int resampler_quality = 10;
    WindowSetup default_resampler_window_setup(
        WindowType::Kaiser, 64, WindowParams{{"beta", 5.0}});

    for (int i = 0; i < num_octaves; ++i) {
      CQTOctave
          octave_data;  // Use a temporary local struct to build up octave data
      octave_data.octave_sample_rate
          = sample_rate_ / std::pow(static_cast<T>(2.0), static_cast<T>(i));

      if (i > 0) {
        try {
          ResampleSpec resample_spec(
              sample_rate_,
              octave_data.octave_sample_rate,
              resampler_quality,
              default_resampler_window_setup);

          // Declare and assign resampler_impl_expected inside the correct
          // constexpr branch
          if constexpr (std::is_same_v<T, F32>) {
            OmniExpected<std::unique_ptr<Abstract::ResamplePlanImpl<F32>>>
                resampler_impl_expected
                = owner_backend_->create_resample_plan_impl_f32(resample_spec);
            if (!resampler_impl_expected) {
              logger->error(
                  "CQTPlanImpl: Failed to create F32 resampler impl for octave "
                  "{}, target SR {}. Status: {}",
                  i,
                  octave_data.octave_sample_rate,
                  static_cast<int>(resampler_impl_expected.error()));
              throw OmniException(
                  "Failed to create internal F32 resampler plan impl.",
                  resampler_impl_expected.error());
            }
            octave_data.resampler = ResamplePlan<F32>::create_from_impl(
                std::move(resampler_impl_expected.value()));
          }
          else if constexpr (std::is_same_v<T, F64>) {
            OmniExpected<std::unique_ptr<Abstract::ResamplePlanImpl<F64>>>
                resampler_impl_expected
                = owner_backend_->create_resample_plan_impl_f64(resample_spec);
            if (!resampler_impl_expected) {
              logger->error(
                  "CQTPlanImpl: Failed to create F64 resampler impl for octave "
                  "{}, target SR {}. Status: {}",
                  i,
                  octave_data.octave_sample_rate,
                  static_cast<int>(resampler_impl_expected.error()));
              throw OmniException(
                  "Failed to create internal F64 resampler plan impl.",
                  resampler_impl_expected.error());
            }
            octave_data.resampler = ResamplePlan<F64>::create_from_impl(
                std::move(resampler_impl_expected.value()));
          }

          if (!octave_data.resampler) {
            logger->error(
                "CQTPlanImpl: Failed to create public resampler plan for "
                "octave {}, target SR {}.",
                i,
                octave_data.octave_sample_rate);
            throw OmniException(
                "Failed to create internal resampler plan.", Status::Failure);
          }
        }
        catch (const std::invalid_argument& e) {
          logger->error(
              "CQTPlanImpl: Invalid ResampleSpec for octave {}: {}",
              i,
              e.what());
          throw OmniException(
              "Invalid ResampleSpec for CQT octave: " + std::string(e.what()),
              Status::InvalidArgument);
        }
      }

      T f_min_oct = current_min_freq;
      T f_max_oct = std::min(max_freq, current_min_freq * static_cast<T>(2.0));
      if (i == num_octaves - 1) f_max_oct = max_freq;

      size_t max_kernel_len_this_octave = 0;

      for (int k = 0; k < bins_per_octave_; ++k) {
        T freq_k = f_min_oct
                   * std::pow(
                       static_cast<T>(2.0),
                       static_cast<T>(k) / static_cast<T>(bins_per_octave_));
        if (freq_k > max_freq + static_cast<T>(1e-6)) break;
        if (freq_k > octave_data.octave_sample_rate / static_cast<T>(2.0))
          break;

        octave_data.freqs.push_back(freq_k);
        all_bin_frequencies_.push_back(freq_k);

        size_t N_k = static_cast<size_t>(std::round(
            quality_factor_q * octave_data.octave_sample_rate / freq_k));
        if (N_k == 0) N_k = 1;
        octave_data.kernel_lengths.push_back(N_k);
        if (N_k > max_kernel_len_this_octave) {
          max_kernel_len_this_octave = N_k;
        }
      }
      if (octave_data.freqs.empty()) continue;

      if (max_kernel_len_this_octave > 0) {
        octave_data.fft_length = std::bit_ceil(max_kernel_len_this_octave);
        if (octave_data.fft_length < 2 * max_kernel_len_this_octave) {
          octave_data.fft_length *= 2;
        }
        if (octave_data.fft_length < 32) {
          octave_data.fft_length = 32;
        }
      }
      else {
        octave_data.fft_length = 256;
      }
      logger->debug(
          "CQT Octave {}: MaxKernelLen={}, Calculated FFTLen={}",
          i,
          max_kernel_len_this_octave,
          octave_data.fft_length);

      // Create RFFT plan for this octave
      if constexpr (std::is_same_v<T, F32>) {
        auto rfft_pimpl_f32_e
            = owner_backend_->create_rfft_plan_impl_f32(octave_data.fft_length);
        if (!rfft_pimpl_f32_e) {
          logger->error(
              "CQTPlanImpl: Failed to create RFFT plan impl F32 for octave {}. "
              "FFTLen={}. Status: {}",
              i,
              octave_data.fft_length,
              static_cast<int>(rfft_pimpl_f32_e.error()));
          throw OmniException(
              "Failed to create internal RFFT plan impl F32.",
              rfft_pimpl_f32_e.error());
        }
        octave_data.rfft_plan = RFFTPlan<F32>::create_from_impl(
            std::move(rfft_pimpl_f32_e.value()));
      }
      else if constexpr (std::is_same_v<T, F64>) {
        auto rfft_pimpl_f64_e
            = owner_backend_->create_rfft_plan_impl_f64(octave_data.fft_length);
        if (!rfft_pimpl_f64_e) {
          logger->error(
              "CQTPlanImpl: Failed to create RFFT plan impl F64 for octave {}. "
              "FFTLen={}. Status: {}",
              i,
              octave_data.fft_length,
              static_cast<int>(rfft_pimpl_f64_e.error()));
          throw OmniException(
              "Failed to create internal RFFT plan impl F64.",
              rfft_pimpl_f64_e.error());
        }
        octave_data.rfft_plan = RFFTPlan<F64>::create_from_impl(
            std::move(rfft_pimpl_f64_e.value()));
      }
      if (!octave_data.rfft_plan) {
        logger->error(
            "CQTPlanImpl: Failed to create public RFFT plan for octave {}. "
            "FFTLen={}",
            i,
            octave_data.fft_length);
        throw OmniException(
            "Failed to create internal RFFT plan.", Status::Failure);
      }

      std::vector<T> temp_kernel_real(octave_data.fft_length);

      for (size_t k_idx = 0; k_idx < octave_data.freqs.size(); ++k_idx) {
        T freq_k = octave_data.freqs[k_idx];
        size_t N_k = octave_data.kernel_lengths[k_idx];

        std::fill(temp_kernel_real.begin(), temp_kernel_real.end(), T{0});

        WindowSetup current_kernel_window_setup = cqt_window_setup;
        current_kernel_window_setup.length = static_cast<int>(N_k);

        std::vector<T> window_values(N_k);
        auto gen_win_status
            = generate_window<T>(current_kernel_window_setup, window_values);
        if (!gen_win_status) {
          logger->error(
              "CQTPlanImpl: Failed to generate CQT analysis window for f={} "
              "Hz, N_k={}. Status: {}",
              freq_k,
              N_k,
              static_cast<int>(gen_win_status.error()));
          throw OmniException(
              "Failed to generate CQT analysis window.",
              gen_win_status.error());
        }

        T norm_factor = static_cast<T>(1.0) / static_cast<T>(N_k);
        for (size_t n = 0; n < N_k; ++n) {
          double time_n
              = static_cast<double>(n) / octave_data.octave_sample_rate;
          Complex exp_val = std::exp(Complex(
              0,
              static_cast<T>(-2.0 * std::numbers::pi_v<double>) * freq_k
                  * static_cast<T>(time_n)));
          Complex kernel_val = norm_factor * window_values[n] * exp_val;
          if (n < temp_kernel_real.size()) {
            temp_kernel_real[n] = kernel_val.real();
          }
        }

        std::vector<Complex> complex_kernel_fft_placeholder(
            octave_data.fft_length / 2 + 1);
        // TODO: Correctly prepare and FFT the complex CQT kernel.
        octave_data.sparse_kernels.push_back(complex_kernel_fft_placeholder);
      }
      if (!octave_data.freqs.empty()) {
        octaves_.push_back(std::move(octave_data));
      }
      current_min_freq *= static_cast<T>(2.0);
    }
    if (octaves_.empty() || all_bin_frequencies_.empty()) {
      logger->error(
          "CQTPlanImpl: No valid CQT bins or octaves were generated.");
      throw OmniException(
          "No CQT bins generated for the given parameters.",
          Status::InvalidArgument);
    }
    std::sort(all_bin_frequencies_.begin(), all_bin_frequencies_.end());

    logger->info(
        "Default::CQTPlanImpl created successfully with {} octaves and {} "
        "total bins.",
        octaves_.size(),
        all_bin_frequencies_.size());
  }

  template <typename T>
  CQTPlanImpl<T>::~CQTPlanImpl()
  {
    // auto logger = spdlog::get("OmniDSP");
    // if(logger) logger->trace("Default::CQTPlanImpl destructed.");
  }

  template <typename T>
  Status CQTPlanImpl<T>::execute(
      std::span<const T> input, std::span<Complex> output) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (octaves_.empty()) {
      logger->warn("CQTPlanImpl::execute: Plan not initialized (no octaves).");
      return Status::NotInitialized;
    }

    size_t num_input_samples = input.size();
    size_t num_output_frames = get_num_output_frames(num_input_samples);

    if (output.size() < num_output_frames * get_num_bins()) {
      logger->warn(
          "CQTPlanImpl::execute: Output span too small. Provided: {}, "
          "Required: {} ({} frames x {} bins).",
          output.size(),
          num_output_frames * get_num_bins(),
          num_output_frames,
          get_num_bins());
      return Status::SizeMismatch;
    }
    if (num_output_frames == 0) {
      return Status::Success;
    }

    logger->warn(
        "CQTPlanImpl::execute: Default CQT execution is a placeholder and not "
        "fully implemented.");
    for (size_t i = 0; i < output.size();
         ++i) {  // Iterate up to output.size() to avoid writing out of bounds
      output[i] = Complex{0, 0};
    }

    return Status::NotImplemented;
  }

  template <typename T>
  size_t CQTPlanImpl<T>::get_num_bins() const
  {
    return all_bin_frequencies_.size();
  }

  template <typename T>
  size_t CQTPlanImpl<T>::get_num_output_frames(size_t input_length) const
  {
    if (hop_length_ == 0) return 0;
    if (input_length == 0) return 0;
    // A more robust calculation considering the longest kernel to ensure enough
    // samples for the last frame.
    size_t longest_kernel_overall = 0;
    for (const auto& oct : octaves_) {
      for (size_t len : oct.kernel_lengths) {
        if (len > longest_kernel_overall) longest_kernel_overall = len;
      }
    }
    if (input_length < longest_kernel_overall
        && longest_kernel_overall
               > 0) {  // if input is shorter than the longest analysis window
      return (input_length > 0)
                 ? 1
                 : 0;  // produce 1 frame if there's any input, else 0
    }
    if (input_length
        <= longest_kernel_overall) {  // handles case where input_length ==
                                      // longest_kernel_overall
      return (input_length > 0) ? 1 : 0;
    }
    return (input_length - longest_kernel_overall) / hop_length_ + 1;
  }

  template <typename T>
  size_t CQTPlanImpl<T>::get_hop_length() const
  {
    return hop_length_;
  }

  // Explicit Template Instantiations
  template class CQTPlanImpl<F32>;
  template class CQTPlanImpl<F64>;

}  // namespace OmniDSP::Default
