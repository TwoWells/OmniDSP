/**
 * @file cqt.cpp (Default)
 * @brief Implements the Default backend CQTProcessorImpl class.
 */

#include "cqt.hpp"  // Corresponding header

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>              // For Design::CQT, Design::CQTOctave
#include <OmniDSP/fft.hpp>              // For RFFTPlan, FFTPlan
#include <OmniDSP/params/resample.hpp>  // For Params::Resample
#include <OmniDSP/resample.hpp>  // For ResamplePlan, Design::Resample, ResampleProcessor
#include <OmniDSP/utils.hpp>
#include <OmniDSP/window.hpp>
#include <algorithm>  // For std::fill
#include <bit>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "interface/backend.hpp"  // For Abstract::Backend
#include "spdlog/spdlog.h"

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // CQTProcessorImpl Method Implementations
  //--------------------------------------------------------------------------
  template <typename T>
  CQTProcessorImpl<T>::CQTProcessorImpl(
      const Abstract::Backend* owner_backend, const Design::CQT& design_spec)
      : owner_backend_(owner_backend), spec_(design_spec)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner_backend_) {
      logger->error("Default::CQTProcessorImpl: owner_backend is null.");
      throw OmniException(
          "CQTProcessorImpl requires a non-null owner_backend.",
          OmniStatus::InvalidArgument);
    }

    logger->debug(
        "Creating Default::CQTProcessorImpl: SR={}, MinF={}, MaxF={}, "
        "Bins/Oct={}, "
        "Hop={}, NumOctavesInDesign={}",
        spec_.original_sample_rate,
        spec_.min_freq,
        spec_.max_freq,
        spec_.bins_per_octave,
        spec_.hop_length,
        spec_.num_octaves_processed);

    processed_octaves_.reserve(spec_.octave_designs.size());

    size_t max_fft_len_for_frame_buf = 0;

    for (const auto& octave_design_item : spec_.octave_designs) {
      logger->debug(
          "Processing OctaveDesign: OctaveSR={}, FFTLen={}, NumBins={}",
          octave_design_item.octave_sample_rate,
          octave_design_item.fft_length,
          octave_design_item.bin_frequencies.size());

      std::unique_ptr<ResampleProcessor<T>> resampler_processor = nullptr;
      if (std::abs(
              octave_design_item.octave_sample_rate
              - spec_.original_sample_rate)
          > 1e-6) {
        WindowParams kaiser_params;
        kaiser_params["beta"] = 5.0;
        Params::Resample resample_params(
            spec_.original_sample_rate,
            octave_design_item.octave_sample_rate,
            10,
            WindowSetup{WindowType::Kaiser, 0, kaiser_params});

        auto internal_resample_design_e = Utils::create_spec(resample_params);
        if (!internal_resample_design_e) {
          logger->error(
              "Failed to create internal Design::Resample for CQT octave. "
              "Error: {}",
              static_cast<int>(internal_resample_design_e.error()));
          throw OmniException(
              "Failed to create internal Design::Resample for CQT.",
              internal_resample_design_e.error());
        }

        auto resampler_expected = ResampleProcessor<T>::create(
            *owner_backend_, internal_resample_design_e.value());
        if (!resampler_expected) {
          logger->error(
              "Failed to create ResampleProcessor for CQT octave. Error: {}",
              static_cast<int>(resampler_expected.error()));
          throw OmniException(
              "Failed to create internal ResampleProcessor for CQT.",
              resampler_expected.error());
        }
        resampler_processor = std::move(resampler_expected.value());
      }

      OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<T>>>
          rfft_plan_impl_expected;
      if constexpr (std::is_same_v<T, F32>) {
        rfft_plan_impl_expected = owner_backend_->create_rfft_plan_impl_f32(
            octave_design_item.fft_length);
      }
      else {
        rfft_plan_impl_expected = owner_backend_->create_rfft_plan_impl_f64(
            octave_design_item.fft_length);
      }

      if (!rfft_plan_impl_expected) {
        logger->error(
            "Failed to create RFFTPlanImpl for CQT octave. FFTLen={}, Error: "
            "{}",
            octave_design_item.fft_length,
            static_cast<int>(rfft_plan_impl_expected.error()));
        throw OmniException(
            "Failed to create internal RFFTPlanImpl for CQT.",
            rfft_plan_impl_expected.error());
      }
      std::unique_ptr<RFFTPlan<T>> rfft_plan = RFFTPlan<T>::create_from_impl(
          std::move(rfft_plan_impl_expected.value()));
      if (!rfft_plan) {
        logger->error(
            "Failed to wrap RFFTPlanImpl for CQT octave. FFTLen={}",
            octave_design_item.fft_length);
        throw OmniException(
            "Failed to wrap internal RFFTPlanImpl for CQT.",
            OmniStatus::Failure);
      }

      if (octave_design_item.fft_length / 2 + 1 > max_fft_len_for_frame_buf) {
        max_fft_len_for_frame_buf = octave_design_item.fft_length / 2 + 1;
      }

      std::vector<std::vector<Complex>> octave_sparse_kernels_fft;
      octave_sparse_kernels_fft.reserve(
          octave_design_item.bin_frequencies.size());

      // This buffer is transient for kernel FFT generation, not part of
      // persistent state. std::vector<T>
      // temp_kernel_real_time_domain(octave_design_item.fft_length);

      for (size_t k_idx = 0; k_idx < octave_design_item.bin_frequencies.size();
           ++k_idx) {
        double bin_freq = octave_design_item.bin_frequencies[k_idx];
        int N_k = octave_design_item.kernel_lengths_samples[k_idx];

        if (N_k <= 0) {
          logger->warn(
              "Kernel length N_k is {} for freq {}. Skipping kernel.",
              N_k,
              bin_freq);
          octave_sparse_kernels_fft.emplace_back();
          continue;
        }
        if (static_cast<size_t>(N_k) > octave_design_item.fft_length) {
          logger->error(
              "Kernel length N_k ({}) > FFT length ({}) for freq {}. This "
              "should not happen if Design::CQT is correct.",
              N_k,
              octave_design_item.fft_length,
              bin_freq);
          throw OmniException(
              "CQT Kernel length exceeds FFT length.",
              OmniStatus::InvalidArgument);
        }

        WindowSetup kernel_win_setup
            = octave_design_item.window_setup_for_octave;
        kernel_win_setup.length = N_k;

        std::vector<T> window_values(N_k);
        OmniExpected<void> gen_win_status
            = generate_window<T>(kernel_win_setup, window_values);
        if (!gen_win_status) {
          logger->error(
              "CQTProcessorImpl: Failed to generate CQT analysis window for "
              "f={} Hz, N_k={}. Status: {}",
              bin_freq,
              N_k,
              static_cast<int>(gen_win_status.error()));
          throw OmniException(
              "Failed to generate CQT analysis window.",
              gen_win_status.error());
        }

        T norm_factor = static_cast<T>(1.0) / static_cast<T>(N_k);

        std::vector<Complex> complex_kernel_time_domain(
            octave_design_item.fft_length, Complex{T{0.0}, T{0.0}});
        for (int n = 0; n < N_k; ++n) {
          double time_n
              = static_cast<double>(n) / octave_design_item.octave_sample_rate;
          Complex exp_val = std::exp(Complex(
              T{0.0},
              static_cast<T>(
                  -2.0 * std::numbers::pi_v<double> * bin_freq * time_n)));
          complex_kernel_time_domain[n]
              = norm_factor * window_values[n] * exp_val;
        }

        std::vector<Complex> kernel_fft_result(octave_design_item.fft_length);
        OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<Complex>>>
            kernel_cfft_impl_e;
        if constexpr (std::is_same_v<T, F32>) {
          kernel_cfft_impl_e = owner_backend_->create_fft_plan_impl_c32(
              octave_design_item.fft_length);
        }
        else {
          kernel_cfft_impl_e = owner_backend_->create_fft_plan_impl_c64(
              octave_design_item.fft_length);
        }

        if (!kernel_cfft_impl_e) {
          logger->error(
              "Failed to create temporary CFFTPlanImpl for CQT kernel. "
              "FFTLen={}, Error: {}",
              octave_design_item.fft_length,
              static_cast<int>(kernel_cfft_impl_e.error()));
          throw OmniException(
              "Failed to create internal CFFTPlanImpl for CQT kernel.",
              kernel_cfft_impl_e.error());
        }
        auto kernel_cfft_plan = FFTPlan<Complex>::create_from_impl(
            std::move(kernel_cfft_impl_e.value()));
        if (!kernel_cfft_plan) {
          logger->error(
              "Failed to wrap CFFTPlanImpl for CQT kernel. FFTLen={}",
              octave_design_item.fft_length);
          throw OmniException(
              "Failed to wrap internal CFFTPlanImpl for CQT kernel.",
              OmniStatus::Failure);
        }

        OmniStatus kernel_fft_status = kernel_cfft_plan->fft(
            complex_kernel_time_domain, kernel_fft_result);
        if (kernel_fft_status != OmniStatus::Success) {
          logger->error(
              "FFT of CQT kernel failed for freq {}. Status: {}",
              bin_freq,
              static_cast<int>(kernel_fft_status));
          throw OmniException("FFT of CQT kernel failed.", kernel_fft_status);
        }
        octave_sparse_kernels_fft.push_back(std::move(kernel_fft_result));
      }

      processed_octaves_.emplace_back(
          octave_design_item.octave_sample_rate,
          std::move(resampler_processor),
          std::move(rfft_plan),
          std::move(octave_sparse_kernels_fft),
          octave_design_item.fft_length,
          octave_design_item.bin_frequencies);
    }

    if (max_fft_len_for_frame_buf > 0) {
      frame_fft_buffer_.resize(max_fft_len_for_frame_buf);
    }

    size_t max_octave_fft_len = 0;
    for (const auto& od : spec_.octave_designs) {
      if (od.fft_length > max_octave_fft_len) {
        max_octave_fft_len = od.fft_length;
      }
    }
    if (max_octave_fft_len > 0) {
      resampled_buffer_.resize(max_octave_fft_len);
    }

    logger->info(
        "Default::CQTProcessorImpl created successfully using Design::CQT. "
        "HopLength={}, TotalBins={}",
        spec_.hop_length,
        spec_.all_bin_frequencies.size());
  }

  template <typename T>
  CQTProcessorImpl<T>::~CQTProcessorImpl() = default;

  template <typename T>
  OmniStatus CQTProcessorImpl<T>::execute(
      std::span<const T> input, std::span<Complex> output) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (processed_octaves_.empty()) {
      logger->warn(
          "Default::CQTProcessorImpl::execute: Plan not initialized (no "
          "processed "
          "octaves). Output will be zeroed.");
      std::fill(output.begin(), output.end(), Complex{T{0}, T{0}});
      return OmniStatus::NotInitialized;
    }

    // TODO: Implement the actual CQT processing logic using overlap-add or
    // other framing strategy. This will involve:
    // 1. Framing the input signal.
    // 2. For each frame:
    //    a. For each octave in processed_octaves_:
    //       i. Resample the current audio frame to octave_sample_rate (if
    //       resampler exists for this octave).
    //          Store in resampled_buffer_.
    //       ii. Take FFT of the (resampled) audio frame using rfft_plan. Store
    //       in frame_fft_buffer_.
    //           (Ensure frame_fft_buffer_ is appropriate size for RFFT output).
    //       iii. For each CQT kernel in sparse_kernels_fft for this octave:
    //            - Multiply the frame_fft_buffer_ with the pre-computed
    //            kernel_fft (element-wise).
    //            - Sum the result (or IFFT and take specific sample if
    //            time-domain CQT).
    //              This is spectral convolution, so result is one complex
    //              number per (frame, bin).
    //    b. Store/accumulate results into the output span, considering
    //    hop_length.

    logger->warn(
        "Default::CQTProcessorImpl::execute is not fully implemented beyond "
        "setup. Output will be zeroed.");
    std::fill(output.begin(), output.end(), Complex{T{0}, T{0}});
    return OmniStatus::NotImplemented;  // Placeholder
  }

  template <typename T>
  OmniStatus CQTProcessorImpl<T>::reset()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    logger->trace("Default::CQTProcessorImpl::reset() called.");

    for (auto& octave_data : processed_octaves_) {
      if (octave_data.resampler) {
        OmniStatus resampler_status = octave_data.resampler->reset();
        if (resampler_status != OmniStatus::Success) {
          logger->error(
              "Failed to reset resampler for octave with SR {}. Status: {}",
              octave_data.octave_sample_rate,
              static_cast<int>(resampler_status));
          // Depending on desired strictness, could return resampler_status or
          // continue resetting others. For now, let's log and continue.
        }
      }
    }
    // If execute() had other state (e.g., overlap-add buffers), clear them
    // here. std::fill(resampled_buffer_.begin(), resampled_buffer_.end(),
    // T{0}); // Optional: clear processing buffers
    // std::fill(frame_fft_buffer_.begin(), frame_fft_buffer_.end(),
    // Complex{T{0},T{0}}); // Optional

    return OmniStatus::Success;
  }

  template <typename T>
  size_t CQTProcessorImpl<T>::get_num_bins() const
  {
    return spec_.get_total_num_bins();
  }

  template <typename T>
  size_t CQTProcessorImpl<T>::get_num_output_frames(size_t input_length) const
  {
    if (spec_.hop_length == 0) return 0;
    if (input_length == 0) return 0;

    size_t max_samples_needed_at_original_sr = 0;
    if (!spec_.octave_designs.empty()) {
      for (const auto& oct_design : spec_.octave_designs) {
        size_t samples_for_this_octave_fft = oct_design.fft_length;
        double samples_at_original_sr_double
            = static_cast<double>(samples_for_this_octave_fft)
              * (spec_.original_sample_rate / oct_design.octave_sample_rate);
        size_t samples_at_original_sr
            = static_cast<size_t>(std::ceil(samples_at_original_sr_double));
        if (samples_at_original_sr > max_samples_needed_at_original_sr) {
          max_samples_needed_at_original_sr = samples_at_original_sr;
        }
      }
    }

    if (max_samples_needed_at_original_sr == 0
        && !spec_.all_bin_frequencies.empty()) {
      max_samples_needed_at_original_sr = static_cast<size_t>(std::ceil(
          spec_.quality_factor_q * spec_.original_sample_rate
          / spec_.min_freq));
      if (max_samples_needed_at_original_sr == 0)
        max_samples_needed_at_original_sr = 1;
    }
    else if (
        max_samples_needed_at_original_sr == 0
        && spec_.all_bin_frequencies.empty()) {
      return 0;
    }

    if (input_length < max_samples_needed_at_original_sr) {
      return 0;
    }
    return (input_length - max_samples_needed_at_original_sr) / spec_.hop_length
           + 1;
  }

  template <typename T>
  size_t CQTProcessorImpl<T>::get_hop_length() const
  {
    return spec_.hop_length;
  }

  // Explicit template instantiations
  template class CQTProcessorImpl<float>;
  template class CQTProcessorImpl<double>;

}  // namespace OmniDSP::Default
