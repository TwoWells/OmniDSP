/**
 * @file cqt.cpp (Default)
 * @brief Implements the Default backend CQTPlanImpl class.
 */

#include "cqt.hpp"  // Corresponding header

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>              // For Design::CQT, Design::CQTOctave
#include <OmniDSP/fft.hpp>              // For RFFTPlan
#include <OmniDSP/params/resample.hpp>  // For ResampleParams
#include <OmniDSP/resample.hpp>         // For ResamplePlan, Design::Resample
#include <OmniDSP/utils.hpp>
#include <OmniDSP/window.hpp>
#include <algorithm>
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
  // CQTPlanImpl Method Implementations
  //--------------------------------------------------------------------------
  template <typename T>
  CQTPlanImpl<T>::CQTPlanImpl(  // TODO: Rename to CQTProcessorImpl
      const Abstract::Backend* owner_backend,
      const Design::CQT& design_spec)
      : owner_backend_(owner_backend), spec_(design_spec)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner_backend_) {
      logger->error("Default::CQTPlanImpl: owner_backend is null.");
      throw OmniException(
          "CQTPlanImpl requires a non-null owner_backend.",
          Status::InvalidArgument);
    }

    logger->debug(
        "Creating Default::CQTPlanImpl: SR={}, MinF={}, MaxF={}, Bins/Oct={}, "
        "Hop={}, NumOctavesInDesign={}",
        spec_.original_sample_rate,
        spec_.min_freq,
        spec_.max_freq,
        spec_.bins_per_octave,
        spec_.hop_length,
        spec_.num_octaves_processed);

    processed_octaves_.reserve(spec_.octave_designs.size());  // CHANGED HERE

    size_t max_resample_buf_len = 0;
    size_t max_fft_len_for_frame = 0;
    size_t max_kernel_real_len = 0;

    for (const auto& octave_design_item :
         spec_.octave_designs) {  // CHANGED HERE and loop var name
      logger->debug(
          "Processing OctaveDesign: OctaveSR={}, FFTLen={}, NumBins={}",
          octave_design_item.octave_sample_rate,
          octave_design_item.fft_length,
          octave_design_item.bin_frequencies.size());

      std::unique_ptr<ResamplePlan<T>> resampler_plan = nullptr;
      if (std::abs(
              octave_design_item.octave_sample_rate
              - spec_.original_sample_rate)
          > 1e-6) {
        WindowParams kaiser_params;
        kaiser_params["beta"] = 5.0;
        ResampleParams resample_params(  // CORRECTED ARGUMENT ORDER
            spec_.original_sample_rate,
            octave_design_item.octave_sample_rate,
            10,  // quality
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

        auto resampler_expected = ResamplePlan<T>::create(
            *owner_backend_, internal_resample_design_e.value());
        if (!resampler_expected) {
          logger->error(
              "Failed to create ResamplePlan for CQT octave. Error: {}",
              static_cast<int>(resampler_expected.error()));
          throw OmniException(
              "Failed to create internal ResamplePlan for CQT.",
              resampler_expected.error());
        }
        resampler_plan = std::move(resampler_expected.value());
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
            "Failed to wrap internal RFFTPlanImpl for CQT.", Status::Failure);
      }

      std::vector<std::vector<Complex>> octave_sparse_kernels_fft;
      octave_sparse_kernels_fft.reserve(
          octave_design_item.bin_frequencies.size());

      if (octave_design_item.fft_length > max_kernel_real_len) {
        max_kernel_real_len = octave_design_item.fft_length;
      }
      if (octave_design_item.fft_length / 2 + 1 > max_fft_len_for_frame) {
        max_fft_len_for_frame = octave_design_item.fft_length / 2 + 1;
      }

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
              "CQT Kernel length exceeds FFT length.", Status::InvalidArgument);
        }

        temp_kernel_real_buffer_.assign(octave_design_item.fft_length, T{0});

        WindowSetup kernel_win_setup
            = octave_design_item.window_setup_for_octave;
        kernel_win_setup.length = N_k;

        std::vector<T> window_values(N_k);
        OmniExpected<void> gen_win_status
            = generate_window<T>(kernel_win_setup, window_values);
        if (!gen_win_status) {
          logger->error(
              "CQTPlanImpl: Failed to generate CQT analysis window for f={} "
              "Hz, N_k={}. Status: {}",
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
                  2.0 * std::numbers::pi_v<double> * bin_freq * time_n)));
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
              Status::Failure);
        }

        Status kernel_fft_status = kernel_cfft_plan->fft(
            complex_kernel_time_domain, kernel_fft_result);
        if (kernel_fft_status != Status::Success) {
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
          std::move(resampler_plan),
          std::move(rfft_plan),
          std::move(octave_sparse_kernels_fft),
          octave_design_item.fft_length,
          octave_design_item.bin_frequencies);
    }

    if (max_kernel_real_len > 0) {
      // temp_kernel_real_buffer_ might be unused now.
    }
    if (max_fft_len_for_frame > 0) {
      frame_fft_buffer_.resize(max_fft_len_for_frame);
    }

    logger->info(
        "Default::CQTPlanImpl created successfully using Design::CQT. "
        "HopLength={}, TotalBins={}",
        spec_.hop_length,
        spec_.all_bin_frequencies.size());
  }

  template <typename T>
  CQTPlanImpl<T>::~CQTPlanImpl() = default;

  template <typename T>
  Status CQTPlanImpl<T>::execute(
      std::span<const T> input, std::span<Complex> output) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (processed_octaves_.empty()) {
      logger->warn(
          "Default::CQTPlanImpl::execute: Plan not initialized (no processed "
          "octaves).");
      return Status::NotInitialized;
    }
    logger->warn(
        "Default::CQTPlanImpl::execute is not fully implemented beyond setup.");
    return Status::NotImplemented;
  }

  template <typename T>
  size_t CQTPlanImpl<T>::get_num_bins() const
  {
    return spec_.get_total_num_bins();
  }

  template <typename T>
  size_t CQTPlanImpl<T>::get_num_output_frames(size_t input_length) const
  {
    if (spec_.hop_length == 0) return 0;
    if (input_length == 0) return 0;

    size_t longest_kernel_at_original_sr = 0;
    for (const auto& oct_design : spec_.octave_designs) {  // CHANGED HERE
      for (int kernel_len_oct_sr : oct_design.kernel_lengths_samples) {
        double len_at_orig_sr_double
            = static_cast<double>(kernel_len_oct_sr)
              * (spec_.original_sample_rate / oct_design.octave_sample_rate);
        size_t len_at_orig_sr
            = static_cast<size_t>(std::round(len_at_orig_sr_double));
        if (len_at_orig_sr > longest_kernel_at_original_sr) {
          longest_kernel_at_original_sr = len_at_orig_sr;
        }
      }
    }
    if (longest_kernel_at_original_sr == 0
        && !spec_.all_bin_frequencies.empty()) {
      longest_kernel_at_original_sr = static_cast<size_t>(std::round(
          spec_.quality_factor_q * spec_.original_sample_rate
          / spec_.all_bin_frequencies[0]));
      if (longest_kernel_at_original_sr == 0) longest_kernel_at_original_sr = 1;
    }

    if (input_length <= longest_kernel_at_original_sr) {
      return (input_length > 0) ? 1 : 0;
    }
    return (input_length - longest_kernel_at_original_sr) / spec_.hop_length
           + 1;
  }

  template <typename T>
  size_t CQTPlanImpl<T>::get_hop_length() const
  {
    return spec_.hop_length;
  }

  template class CQTPlanImpl<float>;
  template class CQTPlanImpl<double>;

}  // namespace OmniDSP::Default
