/**
 * @file cqt.cpp (Default)
 * @brief Implements the Default backend CQTPlanImpl class.
 */

#include "cqt.hpp"  // Corresponding header

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>              // For CQTSpec, CQTOctaveSpec
#include <OmniDSP/fft.hpp>              // For RFFTPlan
#include <OmniDSP/params/resample.hpp>  // For ResampleParams
#include <OmniDSP/resample.hpp>         // For ResamplePlan, ResampleSpec
#include <OmniDSP/utils.hpp>  // For Utils::create_spec (used for internal ResampleSpec)
#include <OmniDSP/window.hpp>  // For WindowSetup, WindowType, OmniDSP::generate_window
#include <algorithm>  // For std::sort, std::min, std::max, std::transform
#include <bit>        // For std::bit_ceil (C++20)
#include <cmath>  // For std::log2, std::pow, std::round, std::ceil, std::abs, std::exp
#include <numeric>    // For std::iota
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <utility>    // For std::move
#include <vector>

#include "interface/backend.hpp"  // For Abstract::Backend
#include "spdlog/spdlog.h"

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // CQTPlanImpl Method Implementations
  //--------------------------------------------------------------------------
  template <typename T>
  CQTPlanImpl<T>::CQTPlanImpl(
      const Abstract::Backend* owner_backend, const CQTSpec& spec)
      : owner_backend_(owner_backend), spec_(spec)  // Store the provided spec
  // resampled_buffer_, frame_fft_buffer_, temp_kernel_real_buffer_ are default
  // initialized
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
    // CQTSpec is assumed to be valid if constructed by Utils::create_spec

    logger->debug(
        "Creating Default::CQTPlanImpl: SR={}, MinF={}, MaxF={}, Bins/Oct={}, "
        "Hop={}, NumOctavesInSpec={}",
        spec_.original_sample_rate,
        spec_.min_freq,
        spec_.max_freq,
        spec_.bins_per_octave,
        spec_.hop_length,
        spec_.num_octaves_processed);

    processed_octaves_.reserve(spec_.octave_specs.size());

    size_t max_resample_buf_len = 0;   // To size resampled_buffer_
    size_t max_fft_len_for_frame = 0;  // To size frame_fft_buffer_
    size_t max_kernel_real_len = 0;    // To size temp_kernel_real_buffer_

    for (const auto& octave_spec : spec_.octave_specs) {
      logger->debug(
          "Processing OctaveSpec: OctaveSR={}, FFTLen={}, NumBins={}",
          octave_spec.octave_sample_rate,
          octave_spec.fft_length,
          octave_spec.bin_frequencies.size());

      std::unique_ptr<ResamplePlan<T>> resampler_plan = nullptr;
      if (std::abs(octave_spec.octave_sample_rate - spec_.original_sample_rate)
          > 1e-6) {
        ResampleParams resample_params(
            spec_.original_sample_rate,
            octave_spec.octave_sample_rate,
            10,  // Default quality for internal resampler
            // Use a default window for internal resampling, or make it
            // configurable via CQTSpec/Params
            WindowSetup{WindowType::Kaiser, 0, WindowParams{{"beta", 5.0}}}
            // Explicit WindowParams
        );
        // Create ResampleSpec for this internal resampling
        auto internal_resample_spec_e = Utils::create_spec(resample_params);
        if (!internal_resample_spec_e) {
          logger->error(
              "Failed to create internal ResampleSpec for CQT octave. Error: "
              "{}",
              static_cast<int>(internal_resample_spec_e.error()));
          throw OmniException(
              "Failed to create internal ResampleSpec for CQT.",
              internal_resample_spec_e.error());
        }
        // Create the ResamplePlan using the owner_backend_ and the generated
        // internal_resample_spec
        auto resampler_expected = ResamplePlan<T>::create(
            *owner_backend_, internal_resample_spec_e.value());
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

      // Create RFFTPlanImpl via owner_backend_
      OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<T>>>
          rfft_plan_impl_expected;
      if constexpr (std::is_same_v<T, F32>) {
        rfft_plan_impl_expected
            = owner_backend_->create_rfft_plan_impl_f32(octave_spec.fft_length);
      }
      else {  // F64
        rfft_plan_impl_expected
            = owner_backend_->create_rfft_plan_impl_f64(octave_spec.fft_length);
      }

      if (!rfft_plan_impl_expected) {
        logger->error(
            "Failed to create RFFTPlanImpl for CQT octave. FFTLen={}, Error: "
            "{}",
            octave_spec.fft_length,
            static_cast<int>(rfft_plan_impl_expected.error()));
        throw OmniException(
            "Failed to create internal RFFTPlanImpl for CQT.",
            rfft_plan_impl_expected.error());
      }
      // Create the public RFFTPlan by wrapping the implementation
      std::unique_ptr<RFFTPlan<T>> rfft_plan = RFFTPlan<T>::create_from_impl(
          std::move(rfft_plan_impl_expected.value()));
      if (!rfft_plan) {
        logger->error(
            "Failed to wrap RFFTPlanImpl for CQT octave. FFTLen={}",
            octave_spec.fft_length);
        throw OmniException(
            "Failed to wrap internal RFFTPlanImpl for CQT.", Status::Failure);
      }

      std::vector<std::vector<Complex>> octave_sparse_kernels_fft;
      octave_sparse_kernels_fft.reserve(octave_spec.bin_frequencies.size());

      if (octave_spec.fft_length > max_kernel_real_len) {
        max_kernel_real_len = octave_spec.fft_length;
      }
      if (octave_spec.fft_length / 2 + 1 > max_fft_len_for_frame) {
        max_fft_len_for_frame = octave_spec.fft_length / 2 + 1;
      }

      for (size_t k_idx = 0; k_idx < octave_spec.bin_frequencies.size();
           ++k_idx) {
        double bin_freq = octave_spec.bin_frequencies[k_idx];
        int N_k = octave_spec.kernel_lengths_samples[k_idx];  // Length of this
                                                              // specific kernel

        if (N_k <= 0) {
          logger->warn(
              "Kernel length N_k is {} for freq {}. Skipping kernel.",
              N_k,
              bin_freq);
          octave_sparse_kernels_fft
              .emplace_back();  // Add empty FFT if kernel is invalid
          continue;
        }
        if (static_cast<size_t>(N_k) > octave_spec.fft_length) {
          logger->error(
              "Kernel length N_k ({}) > FFT length ({}) for freq {}. This "
              "should not happen if CQTSpec is correct.",
              N_k,
              octave_spec.fft_length,
              bin_freq);
          throw OmniException(
              "CQT Kernel length exceeds FFT length.", Status::InvalidArgument);
        }

        temp_kernel_real_buffer_.assign(
            octave_spec.fft_length, T{0});  // Zero out before use

        WindowSetup kernel_win_setup
            = octave_spec
                  .window_setup_for_octave;  // Get window type and params
        kernel_win_setup.length
            = N_k;  // Set the correct length for this specific kernel

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

        T norm_factor = static_cast<T>(1.0)
                        / static_cast<T>(N_k);  // Normalization for energy

        // Prepare complex kernel for FFT
        std::vector<Complex> complex_kernel_time_domain(
            octave_spec.fft_length, Complex{0, 0});
        for (int n = 0; n < N_k; ++n) {
          double time_n
              = static_cast<double>(n) / octave_spec.octave_sample_rate;
          // CQT Kernel: (1/N_k) * window[n] * exp(j * 2*pi * f_k * n / fs_oct)
          // For convolution via FFT: use FFT(signal) * conj(FFT(kernel))
          // So we need FFT of the (non-conjugated) kernel.
          Complex exp_val = std::exp(Complex(
              0,
              static_cast<T>(2.0 * std::numbers::pi_v<double>)
                  * static_cast<T>(bin_freq) * static_cast<T>(time_n)));
          complex_kernel_time_domain[n]
              = norm_factor * window_values[n] * exp_val;
        }

        std::vector<Complex> kernel_fft_result(
            octave_spec.fft_length);  // For C-C FFT
        // Create a temporary Complex-to-Complex FFTPlan for the kernel
        OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<Complex>>>
            kernel_cfft_impl_e;
        if constexpr (std::is_same_v<T, F32>) {  // Complex type is C32
          kernel_cfft_impl_e = owner_backend_->create_fft_plan_impl_c32(
              octave_spec.fft_length);
        }
        else {  // Complex type is C64
          kernel_cfft_impl_e = owner_backend_->create_fft_plan_impl_c64(
              octave_spec.fft_length);
        }

        if (!kernel_cfft_impl_e) {
          logger->error(
              "Failed to create temporary CFFTPlanImpl for CQT kernel. "
              "FFTLen={}, Error: {}",
              octave_spec.fft_length,
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
              octave_spec.fft_length);
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

        // The sparse_kernels_fft should store the FFT of the kernel, possibly
        // conjugated depending on the execute strategy. If execute does
        // FFT(signal) * conj(FFT(kernel)), then store FFT(kernel) here.
        octave_sparse_kernels_fft.push_back(std::move(kernel_fft_result));
      }

      processed_octaves_.emplace_back(
          octave_spec.octave_sample_rate,
          std::move(resampler_plan),
          std::move(rfft_plan),
          std::move(octave_sparse_kernels_fft),
          octave_spec.fft_length,
          octave_spec.bin_frequencies);
    }

    if (max_kernel_real_len > 0) {
      // temp_kernel_real_buffer_ was intended for constructing real part of
      // kernel if using RFFT for kernel. Since kernel is complex and we use
      // CFFT, this buffer might not be needed in its current form, or it could
      // be repurposed for the complex time-domain kernel before FFT. For now,
      // let's assume complex_kernel_time_domain serves that purpose locally.
      // temp_kernel_real_buffer_.resize(max_kernel_real_len); // This might be
      // unused now.
    }
    if (max_fft_len_for_frame
        > 0) {  // This is for the output of RFFT of audio frame.
      frame_fft_buffer_.resize(max_fft_len_for_frame);
    }

    logger->info(
        "Default::CQTPlanImpl created successfully using CQTSpec. "
        "HopLength={}, TotalBins={}",
        spec_.hop_length,
        spec_.all_bin_frequencies.size());
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

    if (processed_octaves_.empty()) {
      logger->warn(
          "Default::CQTPlanImpl::execute: Plan not initialized (no processed "
          "octaves).");
      return Status::NotInitialized;
    }

    size_t num_input_samples = input.size();
    size_t total_cqt_bins = spec_.get_total_num_bins();
    size_t num_output_frames = get_num_output_frames(num_input_samples);

    if (output.size() < num_output_frames * total_cqt_bins) {
      logger->warn(
          "Default::CQTPlanImpl::execute: Output span too small. Provided: {}, "
          "Required: {} ({} frames x {} bins).",
          output.size(),
          num_output_frames * total_cqt_bins,
          num_output_frames,
          total_cqt_bins);
      return Status::SizeMismatch;
    }
    if (num_output_frames == 0) {
      if (!output.empty())
        std::fill(output.begin(), output.end(), Complex{0, 0});
      return Status::Success;
    }

    // --- Actual CQT computation ---
    // This requires careful framing, resampling, FFT of audio, spectral
    // multiplication, and combining results.

    // Initialize output to zero
    std::fill(
        output.begin(),
        output.begin() + (num_output_frames * total_cqt_bins),
        Complex{0, 0});
    if (output.size() > num_output_frames * total_cqt_bins) {
      std::fill(
          output.begin() + (num_output_frames * total_cqt_bins),
          output.end(),
          Complex{0, 0});
    }

    std::vector<T>
        current_frame_real;  // To hold the current time-domain audio frame
    size_t output_bin_idx_offset = 0;

    for (const auto& octave : processed_octaves_) {
      if (!octave.rfft_plan) {  // Should not happen if constructor succeeded
        logger->error("Null RFFT plan in processed octave. Skipping octave.");
        output_bin_idx_offset += octave.bin_frequencies.size();
        continue;
      }

      // Determine max frame size needed for this octave's FFT
      size_t frame_len_for_octave_fft = octave.fft_length;
      current_frame_real.resize(frame_len_for_octave_fft);

      // Resize resampled_buffer_ if needed (only once per octave processing)
      // The input to resampler is always a block of original signal.
      // The output of resampler needs to be large enough for one FFT frame at
      // octave_sample_rate.
      if (octave.resampler) {
        // Max input to resampler for one output FFT frame:
        // ceil(frame_len_for_octave_fft * original_SR / octave_SR)
        size_t max_input_to_resampler = static_cast<size_t>(std::ceil(
            static_cast<double>(frame_len_for_octave_fft)
            * spec_.original_sample_rate / octave.octave_sample_rate));
        max_input_to_resampler = std::max(
            max_input_to_resampler,
            frame_len_for_octave_fft);  // Ensure it's at least the FFT len
        if (resampled_buffer_.size()
            < frame_len_for_octave_fft) {  // resampled_buffer_ holds output of
                                           // resampler
          resampled_buffer_.resize(frame_len_for_octave_fft);
        }
      }

      for (size_t frame_idx = 0; frame_idx < num_output_frames; ++frame_idx) {
        size_t frame_start_original_sr = frame_idx * spec_.hop_length;
        // We need enough samples from input for the longest CQT kernel at its
        // original SR equivalent. This is complex due to varying kernel lengths
        // and potential resampling. For simplicity, let's assume the frame
        // length for FFT is `frame_len_for_octave_fft`.

        if (frame_start_original_sr + frame_len_for_octave_fft
                > num_input_samples
            && octave.resampler == nullptr) {
          // If no resampling, and we don't have a full frame_len_for_octave_fft
          // from original input, pad or break. For now, this simplified execute
          // will break. A full implementation needs careful padding. This
          // condition is too simple; CQT kernels are centered. We need to
          // extract a segment of input that, after resampling, gives
          // `frame_len_for_octave_fft`. The actual segment of input depends on
          // the longest kernel in the *octave*. This part of the execute logic
          // is highly complex for a production CQT. The placeholder warning is
          // still very relevant.
          continue;
        }

        std::span<const T> audio_segment_to_process;
        if (octave.resampler) {
          // Determine segment of original input to resample for this frame.
          // This is tricky: need to know how many original samples map to
          // frame_len_for_octave_fft at octave_sr.
          size_t input_segment_len_for_resampler
              = octave.resampler->get_output_length(
                  frame_len_for_octave_fft);  // This is backwards.
                                              // We need input to resampler that
                                              // *produces*
                                              // frame_len_for_octave_fft.
                                              // Approx:
                                              // frame_len_for_octave_fft *
                                              // (SR_orig / SR_oct)
          input_segment_len_for_resampler = static_cast<size_t>(std::ceil(
              static_cast<double>(frame_len_for_octave_fft)
              * spec_.original_sample_rate / octave.octave_sample_rate));

          if (frame_start_original_sr + input_segment_len_for_resampler
              > num_input_samples) {
            // Not enough original input for this resampled frame. Pad or skip.
            // For now, skip. Proper CQT needs careful boundary handling.
            continue;
          }
          std::span<const T> original_input_segment = input.subspan(
              frame_start_original_sr, input_segment_len_for_resampler);
          resampled_buffer_.assign(
              frame_len_for_octave_fft,
              T{0});  // Ensure correct size for output of resampler

          Status resample_status = octave.resampler->execute(
              original_input_segment, resampled_buffer_);
          if (resample_status != Status::Success) {
            logger->error(
                "Resampling failed for CQT octave. Frame {}. Status {}",
                frame_idx,
                static_cast<int>(resample_status));
            continue;
          }
          audio_segment_to_process = resampled_buffer_;
        }
        else {
          // No resampling, use original signal segment directly.
          // Ensure we don't read past the end of the input for the FFT frame.
          size_t available_samples
              = num_input_samples - frame_start_original_sr;
          size_t samples_for_this_frame
              = std::min(frame_len_for_octave_fft, available_samples);

          std::fill(
              current_frame_real.begin(),
              current_frame_real.end(),
              T{0});  // Zero pad
          std::copy_n(
              input.data() + frame_start_original_sr,
              samples_for_this_frame,
              current_frame_real.begin());
          audio_segment_to_process = current_frame_real;  // Process this frame
        }

        // FFT of the audio segment for this octave
        // frame_fft_buffer_ is Complex, sized max_fft_len_for_frame (which is
        // N/2 + 1 for RFFT)
        std::span<Complex> current_frame_fft_span = std::span<Complex>(
            frame_fft_buffer_.data(), octave.fft_length / 2 + 1);
        Status audio_fft_status = octave.rfft_plan->rfft(
            audio_segment_to_process.first(octave.fft_length),
            current_frame_fft_span);

        if (audio_fft_status != Status::Success) {
          logger->error(
              "RFFT of audio frame failed for CQT octave. Frame {}. Status {}",
              frame_idx,
              static_cast<int>(audio_fft_status));
          continue;
        }

        // Spectral multiplication with pre-computed kernel FFTs
        for (size_t bin_k_idx = 0; bin_k_idx < octave.bin_frequencies.size();
             ++bin_k_idx) {
          const auto& kernel_fft = octave.sparse_kernels_fft[bin_k_idx];
          if (kernel_fft.empty()) continue;  // Skip if kernel was invalid

          Complex cqt_coeff = Complex{0, 0};
          // The kernel_fft is C-C FFT of length octave.fft_length.
          // The current_frame_fft_span is R-C FFT of length octave.fft_length/2
          // + 1. This multiplication needs to be handled correctly. Standard
          // CQT: sum_freq_domain( FrameFFT[f] * conj(KernelFFT[f]) ) / N_fft
          // Or, if KernelFFT already stores
          // conj(FFT(actual_kernel_for_convolution)): sum_freq_domain(
          // FrameFFT[f] * KernelFFT[f] ) / N_fft This sum is effectively an
          // IFFT at time 0, or just a dot product.

          // For simplicity, assuming kernel_fft is already
          // conj(FFT(kernel_for_conv)) And that it's full complex spectrum (not
          // RFFT packed) This part of the placeholder is still very rough.
          for (size_t f_idx = 0; f_idx < current_frame_fft_span.size();
               ++f_idx) {  // Iterate up to N/2+1
            if (f_idx
                < kernel_fft
                      .size()) {  // Ensure kernel_fft is also indexed correctly
              if (f_idx == 0
                  || (octave.fft_length % 2 == 0
                      && f_idx == octave.fft_length / 2)) {  // DC and Nyquist
                cqt_coeff += current_frame_fft_span[f_idx] * kernel_fft[f_idx];
              }
              else {
                cqt_coeff += current_frame_fft_span[f_idx] * kernel_fft[f_idx]
                             * Complex(2.0, 0.0);  // x2 for positive freqs
              }
            }
          }
          cqt_coeff /= static_cast<T>(
              octave.fft_length);  // Normalization by FFT length

          // Place the coefficient in the global output array
          output
              [(frame_idx * total_cqt_bins)
               + (output_bin_idx_offset + bin_k_idx)]
              = cqt_coeff;
        }
      }
      output_bin_idx_offset += octave.bin_frequencies.size();
    }

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
    for (const auto& oct_spec : spec_.octave_specs) {
      for (int kernel_len_oct_sr : oct_spec.kernel_lengths_samples) {
        double len_at_orig_sr_double
            = static_cast<double>(kernel_len_oct_sr)
              * (spec_.original_sample_rate / oct_spec.octave_sample_rate);
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

    if (input_length < longest_kernel_at_original_sr
        && longest_kernel_at_original_sr > 0) {
      return (input_length > 0) ? 1 : 0;
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

  // Explicit Template Instantiations
  template class CQTPlanImpl<F32>;
  template class CQTPlanImpl<F64>;

}  // namespace OmniDSP::Default
