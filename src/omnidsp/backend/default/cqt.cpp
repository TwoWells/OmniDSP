/**
 * @file cqt.cpp (Default)
 * @brief Implements the Default backend CQTProcessorImpl class.
 */

#include "cqt.hpp"  // Corresponding header

#include <spdlog/fmt/ostr.h>  // For logging custom types with operator<<

#include <OmniDSP/core_types.hpp>  // For OmniStatus, OmniException, F32, F64, etc.
#include <OmniDSP/design.hpp>  // For Design::create (used for internal resampler design)
#include <OmniDSP/design/cqt.hpp>  // For Design::CQT, Design::CQTOctave and their operator<<
#include <OmniDSP/params/resample.hpp>  // For Params::Resample and its operator<<
#include <OmniDSP/plan/fft.hpp>         // For Plan::RFFT, Plan::FFT
#include <OmniDSP/processor/resample.hpp>  // For Processor::Resample
#include <OmniDSP/window.hpp>  // For WindowSetup, generate_window, and their operator<<
#include <algorithm>  // For std::fill, std::min, std::max, std::transform
#include <bit>        // For std::bit_ceil (C++20)
#include <cmath>  // For std::log2, std::pow, std::round, std::ceil, std::abs, std::floor, std::exp
#include <numeric>  // For std::iota
#include <sstream>  // For std::ostringstream
#include <stdexcept>  // For std::invalid_argument, std::runtime_error, std::logic_error
#include <string>   // For std::string in exceptions
#include <utility>  // For std::move
#include <vector>   // For std::vector

#include "interface/backend.hpp"  // For Abstract::Backend
#include "spdlog/spdlog.h"

/**
 * @namespace OmniDSP::Default
 * @brief Contains the default backend implementations for OmniDSP operations.
 */
namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // CQTProcessorImpl Method Implementations
  //--------------------------------------------------------------------------

  /**
   * @brief Constructor for the Default CQTProcessorImpl.
   * @details Initializes the CQT processor based on the provided design
   * specification. This involves setting up resamplers for each octave (if
   * multi-rate), creating FFT plans, and pre-computing the FFTs of the CQT
   * analysis kernels.
   * @param owner_backend Pointer to the Abstract::Backend instance that owns
   * this processor. Used for creating internal FFT and Resample
   * plans/processors. Must not be null.
   * @param design_spec The fully resolved Design::CQT specification.
   * @throws OmniException if owner_backend is null, or if creation of internal
   * resamplers, FFT plans, or kernel FFTs fails.
   * @throws std::invalid_argument if design_spec contains inconsistent
   * parameters (though most validation should occur during Design::CQT
   * construction).
   */
  template <typename T>
  CQTProcessorImpl<T>::CQTProcessorImpl(
      const Abstract::Backend* owner_backend, const Design::CQT& design_spec)
      : owner_backend_(owner_backend), spec_(design_spec)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Creating Default::CQTProcessorImpl with Design: {}", design_spec);
    }

    if (!owner_backend_) {
      if (logger)
        logger->error("Default::CQTProcessorImpl: owner_backend is null.");
      throw OmniException(
          "CQTProcessorImpl requires a non-null owner_backend.",
          OmniStatus::InvalidArgument);
    }

    if (logger && logger->should_log(spdlog::level::debug)) {
      logger->debug(
          "Initializing Default::CQTProcessorImpl. Original SR: {}, MinF: {}, "
          "MaxF: {}, Bins/Oct: {}, Hop: {}, NumOctavesInDesign: {}",
          spec_.original_sample_rate,
          spec_.min_freq,
          spec_.max_freq,
          spec_.bins_per_octave,
          spec_.hop_length,
          spec_.num_octaves_processed);
    }

    processed_octaves_.reserve(spec_.octave_designs.size());
    size_t max_fft_len_for_frame_buf = 0;

    for (const auto& octave_design_item : spec_.octave_designs) {
      if (logger && logger->should_log(spdlog::level::debug)) {
        logger->debug("Processing OctaveDesign: {}", octave_design_item);
      }

      std::unique_ptr<Processor::Resample<T>> resampler_processor = nullptr;
      if (std::abs(
              octave_design_item.octave_sample_rate
              - spec_.original_sample_rate)
          > 1e-6) {
        Params::Resample resample_params(
            spec_.original_sample_rate,
            octave_design_item.octave_sample_rate,
            10,  // Default quality for internal resampler
            WindowSetup{Type::Window::Kaiser, 0, WindowParams{{"beta", 5.0}}}
            // Default window
        );
        if (logger && logger->should_log(spdlog::level::trace)) {
          logger->trace(
              "Creating internal resampler with Params: {}", resample_params);
        }

        auto internal_resample_design_e = Design::create(resample_params);
        if (!internal_resample_design_e) {
          std::ostringstream msg_stream;
          msg_stream << "Failed to create internal Design::Resample for CQT "
                        "octave. Error: "
                     << internal_resample_design_e.error();
          if (logger) logger->error(msg_stream.str());
          throw OmniException(
              msg_stream.str(), internal_resample_design_e.error());
        }
        if (logger && logger->should_log(spdlog::level::trace)) {
          logger->trace(
              "Internal Resample Design created: {}",
              internal_resample_design_e.value());
        }

        auto resampler_expected = Processor::Resample<T>::create(
            *owner_backend_, internal_resample_design_e.value());
        if (!resampler_expected) {
          std::ostringstream msg_stream;
          msg_stream
              << "Failed to create ResampleProcessor for CQT octave. Error: "
              << resampler_expected.error();
          if (logger) logger->error(msg_stream.str());
          throw OmniException(msg_stream.str(), resampler_expected.error());
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
        std::ostringstream msg_stream;
        msg_stream << "Failed to create RFFTPlanImpl for CQT octave. FFTLen="
                   << octave_design_item.fft_length
                   << ", Error: " << rfft_plan_impl_expected.error();
        if (logger) logger->error(msg_stream.str());
        throw OmniException(msg_stream.str(), rfft_plan_impl_expected.error());
      }
      std::unique_ptr<Plan::RFFT<T>> rfft_plan
          = Plan::RFFT<T>::create_from_impl(
              std::move(rfft_plan_impl_expected.value()));
      if (!rfft_plan) {
        std::ostringstream msg_stream;
        msg_stream << "Failed to wrap RFFTPlanImpl for CQT octave. FFTLen="
                   << octave_design_item.fft_length;
        if (logger) logger->error(msg_stream.str());
        throw OmniException(msg_stream.str(), OmniStatus::Failure);
      }

      if (octave_design_item.fft_length / 2 + 1 > max_fft_len_for_frame_buf) {
        max_fft_len_for_frame_buf = octave_design_item.fft_length / 2 + 1;
      }

      std::vector<std::vector<Complex>> octave_sparse_kernels_fft;
      octave_sparse_kernels_fft.reserve(
          octave_design_item.bin_frequencies.size());

      for (size_t k_idx = 0; k_idx < octave_design_item.bin_frequencies.size();
           ++k_idx) {
        double bin_freq = octave_design_item.bin_frequencies[k_idx];
        int N_k = octave_design_item.kernel_lengths_samples[k_idx];

        if (N_k <= 0) {
          if (logger)
            logger->warn(
                "Kernel length N_k is {} for freq {}. Skipping kernel.",
                N_k,
                bin_freq);
          octave_sparse_kernels_fft
              .emplace_back();  // Add empty vector for this kernel
          continue;
        }
        if (static_cast<size_t>(N_k) > octave_design_item.fft_length) {
          std::ostringstream msg_stream;
          msg_stream << "Kernel length N_k (" << N_k << ") > FFT length ("
                     << octave_design_item.fft_length << ") for freq "
                     << bin_freq
                     << ". This should not happen if Design::CQT is correct.";
          if (logger) logger->error(msg_stream.str());
          throw OmniException(msg_stream.str(), OmniStatus::InvalidArgument);
        }

        WindowSetup kernel_win_setup
            = octave_design_item.window_setup_for_octave;
        kernel_win_setup.length = N_k;
        if (logger && logger->should_log(spdlog::level::trace)) {
          logger->trace(
              "Generating window for CQT kernel: Freq={}, N_k={}, Window: {}",
              bin_freq,
              N_k,
              kernel_win_setup);
        }

        std::vector<T> window_values(N_k);
        OmniExpected<void> gen_win_status
            = generate_window<T>(kernel_win_setup, window_values);
        if (!gen_win_status) {
          std::ostringstream msg_stream;
          msg_stream << "CQTProcessorImpl: Failed to generate CQT analysis "
                        "window for f="
                     << bin_freq << " Hz, N_k=" << N_k
                     << ". Status: " << gen_win_status.error();
          if (logger) logger->error(msg_stream.str());
          throw OmniException(msg_stream.str(), gen_win_status.error());
        }

        T norm_factor
            = static_cast<T>(1.0)
              / static_cast<T>(N_k);  // Normalization for energy/amplitude

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
        if constexpr (std::is_same_v<T, F32>) {  // T is Real, so Complex is C32
          kernel_cfft_impl_e = owner_backend_->create_fft_plan_impl_c32(
              octave_design_item.fft_length);
        }
        else {  // T is F64, so Complex is C64
          kernel_cfft_impl_e = owner_backend_->create_fft_plan_impl_c64(
              octave_design_item.fft_length);
        }

        if (!kernel_cfft_impl_e) {
          std::ostringstream msg_stream;
          msg_stream << "Failed to create temporary CFFTPlanImpl for CQT "
                        "kernel. FFTLen="
                     << octave_design_item.fft_length
                     << ", Error: " << kernel_cfft_impl_e.error();
          if (logger) logger->error(msg_stream.str());
          throw OmniException(msg_stream.str(), kernel_cfft_impl_e.error());
        }
        auto kernel_cfft_plan = Plan::FFT<Complex>::create_from_impl(
            std::move(kernel_cfft_impl_e.value()));
        if (!kernel_cfft_plan) {
          std::ostringstream msg_stream;
          msg_stream << "Failed to wrap FFTPlanImpl for CQT kernel. FFTLen="
                     << octave_design_item.fft_length;
          if (logger) logger->error(msg_stream.str());
          throw OmniException(msg_stream.str(), OmniStatus::Failure);
        }

        OmniStatus kernel_fft_status = kernel_cfft_plan->fft(
            complex_kernel_time_domain, kernel_fft_result);
        if (kernel_fft_status != OmniStatus::Success) {
          std::ostringstream msg_stream;
          msg_stream << "FFT of CQT kernel failed for freq " << bin_freq
                     << ". Status: " << kernel_fft_status;
          if (logger) logger->error(msg_stream.str());
          throw OmniException(msg_stream.str(), kernel_fft_status);
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

    if (logger && logger->should_log(spdlog::level::info)) {
      logger->info(
          "Default::CQTProcessorImpl created successfully. HopLength={}, "
          "TotalBins={}",
          spec_.hop_length,
          spec_.all_bin_frequencies.size());
    }
  }

  /**
   * @brief Destructor for CQTProcessorImpl.
   * @tparam T Data type of the samples.
   */
  template <typename T>
  CQTProcessorImpl<T>::~CQTProcessorImpl()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Default::CQTProcessorImpl<{}> destructed.", typeid(T).name());
    }
  }

  /**
   * @brief Executes the Constant-Q Transform.
   * @details This method processes the input audio signal frame by frame,
   * applying resampling (if multi-rate), FFT, and spectral convolution with
   * pre-computed CQT kernel FFTs for each octave.
   * @warning This method is currently a placeholder and will zero the output.
   * Full CQT execution logic needs to be implemented.
   * @tparam T Data type of the input samples (real).
   * @param input A span of constant input samples.
   * @param output A span for the complex CQT coefficients. The layout is
   * typically [num_frames x num_bins].
   * @return OmniStatus::NotImplemented as the core processing logic is a
   * placeholder.
   * @return OmniStatus::NotInitialized if the processor was not properly set
   * up.
   */
  template <typename T>
  OmniStatus CQTProcessorImpl<T>::execute(
      std::span<const T> input, std::span<Complex> output) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Default::CQTProcessorImpl::execute called. Input size: {}, Output "
          "span size: {}",
          input.size(),
          output.size());
    }

    if (processed_octaves_.empty()
        && !spec_.all_bin_frequencies
                .empty()) {  // Check if there should be octaves but none were
                             // processed
      if (logger)
        logger->warn(
            "Default::CQTProcessorImpl::execute: Plan not initialized (no "
            "processed octaves, but bins expected). Output will be zeroed.");
      std::fill(output.begin(), output.end(), Complex{T{0}, T{0}});
      return OmniStatus::NotInitialized;
    }
    if (processed_octaves_.empty()
        && spec_.all_bin_frequencies.empty()) {  // No bins to process
      if (logger && logger->should_log(spdlog::level::debug)) {
        logger->debug(
            "Default::CQTProcessorImpl::execute: No CQT bins to process. "
            "Output will be zeroed.");
      }
      std::fill(output.begin(), output.end(), Complex{T{0}, T{0}});
      return OmniStatus::Success;  // Or NotInitialized if this state is
                                   // considered invalid
    }

    // TODO: Implement the actual CQT processing logic using overlap-add or
    // other framing strategy. This will involve:
    // 1. Framing the input signal based on hop_length and kernel requirements.
    // 2. For each frame:
    //    a. For each octave in processed_octaves_:
    //       i. Resample the current audio frame to octave_sample_rate (if
    //       resampler exists). ii. Take FFT of the (resampled) audio frame
    //       using rfft_plan. iii. For each CQT kernel FFT in this octave's
    //       sparse_kernels_fft:
    //            - Perform spectral multiplication: FrameFFT[f] * KernelFFT[f].
    //            - Sum the result of the spectral multiplication (dot product
    //            in frequency domain).
    //              This gives one complex CQT coefficient for this bin and
    //              frame.
    //    b. Store/accumulate results into the output span.

    if (logger)
      logger->warn(
          "Default::CQTProcessorImpl::execute is not fully implemented beyond "
          "setup. Output will be zeroed.");
    std::fill(output.begin(), output.end(), Complex{T{0}, T{0}});
    return OmniStatus::NotImplemented;
  }

  /**
   * @brief Resets the internal state of the CQT processor.
   * @details This includes resetting any internal resamplers used for
   * multi-rate processing. Any overlap-add buffers or other frame-processing
   * state would also be cleared here.
   * @tparam T Data type of the samples.
   * @return OmniStatus::Success if reset is successful, or an error status from
   * an internal component.
   */
  template <typename T>
  OmniStatus CQTProcessorImpl<T>::reset()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::debug)) {
      logger->debug("Default::CQTProcessorImpl::reset() called.");
    }

    for (auto& octave_data : processed_octaves_) {
      if (octave_data.resampler) {
        OmniStatus resampler_status = octave_data.resampler->reset();
        if (resampler_status != OmniStatus::Success) {
          if (logger)
            logger->error(
                "Failed to reset resampler for octave with SR {}. Status: {}",
                octave_data.octave_sample_rate,
                resampler_status);  // Log OmniStatus directly
          // Depending on desired strictness, could return resampler_status or
          // continue resetting others. For now, let's log and continue.
        }
      }
    }
    // If execute() had other state (e.g., overlap-add buffers), clear them
    // here. std::fill(resampled_buffer_.begin(), resampled_buffer_.end(),
    // T{0}); // Optional std::fill(frame_fft_buffer_.begin(),
    // frame_fft_buffer_.end(), Complex{T{0},T{0}}); // Optional

    return OmniStatus::Success;
  }

  /**
   * @brief Gets the total number of CQT frequency bins.
   * @tparam T Data type of the samples.
   * @return The total number of bins.
   */
  template <typename T>
  size_t CQTProcessorImpl<T>::get_num_bins() const
  {
    return spec_.get_total_num_bins();
  }

  /**
   * @brief Calculates the number of output CQT frames for a given input length.
   * @details This depends on the longest effective analysis window (considering
   * resampling) and the hop length.
   * @tparam T Data type of the samples.
   * @param input_length The length of the input signal in samples.
   * @return The number of CQT frames that will be produced.
   */
  template <typename T>
  size_t CQTProcessorImpl<T>::get_num_output_frames(size_t input_length) const
  {
    if (spec_.hop_length == 0) return 0;  // Avoid division by zero
    if (input_length == 0 && spec_.all_bin_frequencies.empty())
      return 0;  // No input, no bins
    if (input_length == 0 && !spec_.all_bin_frequencies.empty())
      return 1;  // Potentially one frame from initial state/padding if defined
                 // that way

    // Determine the maximum effective window length at the original sample rate
    size_t max_effective_window_at_original_sr = 0;
    if (!spec_.octave_designs.empty()) {
      for (const auto& oct_design : spec_.octave_designs) {
        // Find the max kernel length in this octave
        size_t max_kernel_len_this_octave = 0;
        for (int len : oct_design.kernel_lengths_samples) {
          if (static_cast<size_t>(len) > max_kernel_len_this_octave) {
            max_kernel_len_this_octave = static_cast<size_t>(len);
          }
        }
        // Convert this max kernel length back to original sample rate domain
        double samples_at_original_sr_double = 0.0;
        if (oct_design.octave_sample_rate > 1e-9) {  // Avoid division by zero
          samples_at_original_sr_double
              = static_cast<double>(max_kernel_len_this_octave)
                * (spec_.original_sample_rate / oct_design.octave_sample_rate);
        }

        size_t samples_at_original_sr
            = static_cast<size_t>(std::ceil(samples_at_original_sr_double));
        if (samples_at_original_sr > max_effective_window_at_original_sr) {
          max_effective_window_at_original_sr = samples_at_original_sr;
        }
      }
    }
    else if (!spec_.all_bin_frequencies.empty() && spec_.min_freq > 1e-9) {
      // Fallback if octave_designs is empty but bins exist (e.g. single-rate
      // CQT not fully using octave structure) This is a rough estimate based on
      // the lowest frequency.
      max_effective_window_at_original_sr = static_cast<size_t>(std::ceil(
          spec_.quality_factor_q * spec_.original_sample_rate
          / spec_.min_freq));
    }

    if (max_effective_window_at_original_sr == 0
        && !spec_.all_bin_frequencies.empty()) {
      // If still zero, but we have bins, it implies a very short effective
      // window or an issue. For a single frame output, we need at least one
      // window length. This might happen if all kernel_lengths_samples were 0.
      // To produce at least one frame if input_length is non-zero:
      return (input_length > 0) ? 1 : 0;
    }
    if (max_effective_window_at_original_sr == 0
        && spec_.all_bin_frequencies.empty()) {
      return 0;  // No bins, no frames
    }

    if (input_length < max_effective_window_at_original_sr) {
      return 0;  // Not enough input for even one full frame
    }
    return (input_length - max_effective_window_at_original_sr)
               / spec_.hop_length
           + 1;
  }

  /**
   * @brief Gets the hop length between CQT frames.
   * @tparam T Data type of the samples.
   * @return The hop length in samples.
   */
  template <typename T>
  size_t CQTProcessorImpl<T>::get_hop_length() const
  {
    return spec_.hop_length;
  }

  // Explicit template instantiations
  template class CQTProcessorImpl<float>;
  template class CQTProcessorImpl<double>;

}  // namespace OmniDSP::Default
