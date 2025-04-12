/**
 * @file cqt.cpp
 * @brief Implementation of the CQTPlan class using the recursive downsampling method
 * with precomputed sparse kernels.
 */

#include <OmniDSP/cqt.h>     // Corresponding header for CQTPlan
#include <OmniDSP/omnidsp.h> // For FFTPlan
#include <OmniDSP/windows.h> // For M_PI definition via cmath (though cmath is included below too)
#include <cmath>             // Standard math functions (cos, sin, log2, pow, ceil, round, abs)
#include <vector>            // For std::vector
#include <complex>           // For std::complex
#include <stdexcept>         // For std::invalid_argument, std::runtime_error, std::out_of_range
#include <numeric>           // For std::accumulate
#include <limits>            // For std::numeric_limits
#include <algorithm>         // For std::min, std::max, std::copy, std::fill
#include <string>            // For std::to_string
#include <memory>            // For std::unique_ptr
#include <cstdint>           // For int64_t (used in hop length validation)
#include <iostream>          // For std::cerr warnings
#include <map>               // For std::map (used for sparse kernel storage)

// Define M_PI if it's not already defined (e.g., by <cmath> in some environments)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Backend Forward Declaration ---
// Declares the backend function needed for downsampling without including the specific backend header.
// This assumes backend_impl.h exists and declares these functions.
namespace OmniDSP
{
    namespace Backend
    {
        // Function signature for the backend implementation of filter + downsample
        template <typename T>
        std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal,
                                                  const std::vector<T> &kernel,
                                                  int factor);
    } // namespace Backend
} // namespace OmniDSP

namespace OmniDSP
{

    // --- Helper: Next Power of 2 ---
    // Calculates the smallest power of 2 greater than or equal to n.
    // Useful for determining efficient FFT lengths.
    inline size_t nextPowerOf2(size_t n)
    {
        if (n == 0)
            return 1; // Handle n=0 case
        n--;          // Bit manipulation trick: decrement first
        // Set all bits lower than the most significant bit
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        if constexpr (sizeof(size_t) > 4) // Handle 64-bit size_t
        {
            n |= n >> 32;
        }
        n++; // Increment to get the next power of 2
        return n;
    }

    // --- Constructor ---
    template <typename T>
    CQTPlan<T>::CQTPlan(double sample_rate, size_t hop_length, double lowest_freq, double highest_freq,
                        int bins_per_octave, WindowFuncType window_function, T sparsity_threshold,
                        int fir_filter_order) // Added parameter for FIR filter order
        : sample_rate_(sample_rate), hop_length_(hop_length), lowest_freq_(lowest_freq), highest_freq_(highest_freq),
          bins_per_octave_(bins_per_octave), window_function_(window_function),
          sparsity_threshold_(sparsity_threshold)
    {
        // --- Parameter Validation ---
        // Ensure all input parameters are within valid ranges.
        if (sample_rate <= 0.0)
            throw std::invalid_argument("Sample rate must be positive.");
        if (lowest_freq <= 0.0)
            throw std::invalid_argument("Lowest frequency must be positive.");
        if (highest_freq <= lowest_freq)
            throw std::invalid_argument("Highest frequency must be greater than lowest frequency.");
        // Check highest frequency against Nyquist frequency (sr/2)
        if (highest_freq > sample_rate / 2.0)
        {
            // Allow a small tolerance for floating point inaccuracies
            if (highest_freq > (sample_rate / 2.0) * (1.0 + std::numeric_limits<double>::epsilon() * 10))
            {
                throw std::invalid_argument("Highest frequency cannot exceed Nyquist frequency (sr/2).");
            }
            else
            {
                // Clamp to Nyquist if slightly above due to precision issues
                highest_freq_ = sample_rate / 2.0;
                std::cerr << "Warning: Highest frequency exceeded Nyquist. Clamped to " << highest_freq_ << " Hz." << std::endl;
            }
        }
        if (bins_per_octave <= 0)
            throw std::invalid_argument("Bins per octave must be positive.");
        if (!window_function_)
            throw std::invalid_argument("A valid window function must be provided.");
        if (hop_length_ == 0)
            throw std::invalid_argument("Hop length must be positive.");
        if (fir_filter_order <= 0)
            throw std::invalid_argument("FIR filter order must be positive.");
        if (fir_filter_order % 2 == 0)
            throw std::invalid_argument("FIR filter order must be odd."); // Windowed-sinc design often assumes odd
        fir_filter_order_ = fir_filter_order;                             // Store the validated filter order
        if (sparsity_threshold_ < 0)
            throw std::invalid_argument("Sparsity threshold cannot be negative.");

        // --- Calculate Total Bin Count & Number of Octaves ---
        // Total bins needed to cover the frequency range logarithmically.
        num_bins_ = static_cast<size_t>(std::ceil(bins_per_octave_ * std::log2(highest_freq_ / lowest_freq_)));
        if (num_bins_ == 0)
        { // Handle cases where the range is too small or invalid
            if (lowest_freq > 0)
            { // Tiny valid range resulted in 0 bins
                std::cerr << "Warning: Calculated number of CQT bins is zero based on frequency range. CQT output will be empty." << std::endl;
            }
            else
            { // Invalid frequency range led to 0 bins
                throw std::invalid_argument("Calculated number of CQT bins is zero due to invalid frequency range.");
            }
        }
        // Number of full or partial octaves required.
        num_octaves_ = static_cast<int>(std::ceil(static_cast<double>(num_bins_) / bins_per_octave_));
        if (num_octaves_ < 1 && num_bins_ > 0)
            num_octaves_ = 1; // Ensure at least one octave if bins exist

        // --- Validate Hop Length Divisibility ---
        // For the recursive approach, the hop length must be divisible by 2^(num_octaves - 1)
        // to ensure correct time alignment after downsampling.
        int64_t divisor = (num_octaves_ > 0) ? (1LL << (num_octaves_ - 1)) : 1LL; // Use 64-bit int for power calculation
        if (divisor > 1 && hop_length_ % divisor != 0)
        {
            throw std::invalid_argument("Hop length (" + std::to_string(hop_length_) + ") must be divisible by 2^(" + std::to_string(num_octaves_ - 1) + ") = " + std::to_string(divisor) + " for recursive CQT.");
        }

        // --- Precompute Resources for EACH Octave ---
        // This is the main setup part, calculating FFT lengths, plans, and sparse kernels per octave.
        if (num_octaves_ > 0)
        { // Only proceed if octaves are needed
            // Resize storage vectors for per-octave data
            octave_fft_lens_.resize(num_octaves_);
            octave_spectrum_lens_.resize(num_octaves_);
            octave_signal_fft_plans_.resize(num_octaves_);    // FFT plans for signal frames
            octave_kernel_fft_plans_.resize(num_octaves_);    // FFT plans for kernel generation
            precomputed_sparse_kernels_.resize(num_octaves_); // Sparse spectral kernels

            double current_sr = sample_rate_;                                                  // Start with the original sample rate
            Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE; // Determine precision
            // Calculate the constant Q factor based on bins per octave
            double Q = 1.0 / (std::pow(2.0, 1.0 / bins_per_octave_) - 1.0);

            std::vector<std::complex<T>> temp_kernel_fft_output; // Reusable buffer for kernel FFT calculation

            // Loop through each octave, starting from the lowest frequencies (octave 0)
            for (int octave_idx = 0; octave_idx < num_octaves_; ++octave_idx)
            {

                // 1. Determine required FFT length for this octave
                //    Based on the lowest frequency within this octave and the Q factor.
                //    The lowest frequency *in absolute terms* for this octave.
                double octave_base_freq_abs = lowest_freq_ * std::pow(2.0, static_cast<double>(octave_idx)); // Freq of first bin in octave
                size_t max_nk_this_octave = 0;                                                               // Required window length (samples) for the lowest freq bin
                if (octave_base_freq_abs > std::numeric_limits<double>::epsilon())
                {
                    // Nk = Q * fs / f (window length required for resolution)
                    max_nk_this_octave = static_cast<size_t>(std::max(1.0, std::round(Q * current_sr / octave_base_freq_abs)));
                }
                else
                {
                    max_nk_this_octave = 1; // Avoid division by zero
                }
                // FFT length must be >= max_nk_this_octave and preferably a power of 2.
                octave_fft_lens_[octave_idx] = nextPowerOf2(max_nk_this_octave);
                size_t fft_len_oct = octave_fft_lens_[octave_idx];

                // 2. Create FFT Plans for this octave's FFT length
                try
                {
                    // Plan for Real-to-Complex FFT of signal frames (using BACKWARD norm for convenience)
                    octave_signal_fft_plans_[octave_idx] = std::make_unique<FFTPlan<T>>(
                        fft_len_oct, prec, Direction::FORWARD, Domain::REAL, NormMode::BACKWARD);
                    // Plan for Complex-to-Complex FFT used during kernel generation (norm mode can be BACKWARD)
                    octave_kernel_fft_plans_[octave_idx] = std::make_unique<FFTPlan<T>>(
                        fft_len_oct, prec, Direction::FORWARD, Domain::COMPLEX, NormMode::BACKWARD);

                    // Validate plan creation
                    if (!octave_signal_fft_plans_[octave_idx] || !octave_kernel_fft_plans_[octave_idx])
                    {
                        throw std::runtime_error("Failed to allocate FFTPlan for octave " + std::to_string(octave_idx));
                    }
                    // Store the complex spectrum length (N/2+1) for this octave's FFT plan
                    octave_spectrum_lens_[octave_idx] = octave_signal_fft_plans_[octave_idx]->getComplexLength();
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error("Failed to create FFTPlan for octave " + std::to_string(octave_idx) + ": " + e.what());
                }

                // 3. Generate Sparse Kernels for each bin within this octave
                // Determine how many bins belong to this octave (usually bins_per_octave, except maybe the last octave)
                int bins_in_this_octave = (octave_idx == num_octaves_ - 1)
                                              ? (static_cast<int>(num_bins_) - octave_idx * bins_per_octave_) // Remaining bins
                                              : bins_per_octave_;
                if (bins_in_this_octave <= 0)
                    continue; // Skip if no bins (shouldn't happen with valid num_bins_)

                precomputed_sparse_kernels_[octave_idx].resize(bins_in_this_octave); // Resize kernel storage
                temp_kernel_fft_output.resize(fft_len_oct);                          // Resize temporary FFT buffer

                // Calculate the downsampling factor relative to the original sample rate
                double downsampling_factor = sample_rate_ / current_sr;

                // Loop through each bin (k) within the current octave
                for (int k = 0; k < bins_in_this_octave; ++k)
                {
                    // Calculate the absolute frequency of this bin (relative to original sr)
                    int absolute_bin_index = octave_idx * bins_per_octave_ + k;
                    double freq_k_abs = lowest_freq_ * std::pow(2.0, static_cast<double>(absolute_bin_index) / bins_per_octave_);
                    // Calculate the frequency relative to the current (potentially downsampled) sample rate
                    double freq_k_relative_to_current_sr = freq_k_abs / downsampling_factor;

                    // Calculate the required window length (Nk) for this specific bin's frequency
                    size_t Nk = 0;
                    if (freq_k_relative_to_current_sr > std::numeric_limits<double>::epsilon())
                    {
                        Nk = static_cast<size_t>(std::max(1.0, std::round(Q * current_sr / freq_k_relative_to_current_sr)));
                    }
                    else
                    {
                        Nk = 1;
                    }

                    // Ensure Nk is valid and doesn't exceed the FFT length for this octave
                    if (Nk == 0 || Nk > fft_len_oct)
                    {
                        std::cerr << "Warning: Clamping Nk (" << Nk << ") to FFT length (" << fft_len_oct
                                  << ") for octave " << octave_idx << ", bin " << k << std::endl;
                        Nk = std::min(Nk, fft_len_oct);
                        if (Nk == 0)
                            continue; // Skip bin if Nk becomes 0
                    }

                    // a. Generate time-domain window using the provided function
                    std::vector<T> time_window;
                    try
                    {
                        time_window = window_function_(Nk); // Call user-provided function
                    }
                    catch (const std::exception &e)
                    {
                        throw std::runtime_error("Window function failed during kernel generation for octave " + std::to_string(octave_idx) + ", bin " + std::to_string(k) + ": " + e.what());
                    }
                    // Validate the returned window size
                    if (time_window.size() != Nk)
                    {
                        throw std::runtime_error("Window func returned incorrect size (" + std::to_string(time_window.size()) + ", expected " + std::to_string(Nk) + ") in constructor kernel gen.");
                    }

                    // Normalize the time-domain window (e.g., L1 norm) - crucial for consistent CQT magnitudes
                    T l1_norm = T(0.0);
                    for (T val : time_window)
                        l1_norm += std::abs(val);
                    if (l1_norm > std::numeric_limits<T>::epsilon())
                    { // Avoid division by zero
                        for (T &val : time_window)
                            val /= l1_norm;
                    }
                    else
                    {
                        std::cerr << "Warning: Window L1 norm is zero for octave " << octave_idx << ", bin " << k << ". Kernel might be zero." << std::endl;
                    }

                    // b. Pad the window with zeros to match the octave's FFT length and center it
                    std::vector<T> padded_time_window(fft_len_oct, 0.0);
                    // Calculate starting index to center the Nk-length window in the fft_len_oct buffer
                    long long start_idx_ll = static_cast<long long>(fft_len_oct / 2) - static_cast<long long>(Nk / 2);
                    size_t start_idx = (start_idx_ll < 0) ? 0 : static_cast<size_t>(start_idx_ll);
                    size_t copy_len = std::min(Nk, fft_len_oct - start_idx); // Elements to actually copy
                    std::copy(time_window.begin(), time_window.begin() + copy_len, padded_time_window.begin() + start_idx);

                    // c. Apply complex frequency shift to the padded window
                    // This effectively centers the filter in the frequency domain at freq_k_relative_to_current_sr.
                    std::vector<std::complex<T>> complex_shifted_window(fft_len_oct);
                    // Phase increment for the complex exponential shift
                    double shift_phase_increment = -2.0 * M_PI * freq_k_relative_to_current_sr / current_sr;
                    for (size_t n = 0; n < fft_len_oct; ++n)
                    {
                        double phase = shift_phase_increment * static_cast<double>(n);
                        // Multiply padded window value by complex exponential e^(j*phase)
                        complex_shifted_window[n] = padded_time_window[n] * std::complex<T>(std::cos(phase), std::sin(phase));
                    }

                    // d. Compute FFT of the complex, shifted, padded window
                    try
                    {
                        // Use the C2C FFT plan for kernel generation
                        octave_kernel_fft_plans_[octave_idx]->execute(complex_shifted_window.data(), temp_kernel_fft_output.data());
                    }
                    catch (const std::exception &e)
                    {
                        throw std::runtime_error("Kernel FFT failed for octave " + std::to_string(octave_idx) + ", bin " + std::to_string(k) + ": " + e.what());
                    }

                    // e. Sparsify and Store the CONJUGATED kernel in the frequency domain
                    // We store the conjugate because CQT involves correlation (or multiplication by conjugate in freq domain).
                    size_t spectrum_len_oct = octave_spectrum_lens_[octave_idx]; // N/2+1 (length of RFFT output)
                    precomputed_sparse_kernels_[octave_idx][k].clear();          // Clear any previous map data for this bin
                    // Iterate through the FFT output of the kernel
                    for (size_t j = 0; j < fft_len_oct; ++j)
                    {
                        // We only need to store kernel components up to the length of the RFFT spectrum
                        if (j >= spectrum_len_oct)
                            continue;

                        // Take the complex conjugate of the kernel's FFT result
                        std::complex<T> kernel_val_conj = std::conj(temp_kernel_fft_output[j]);
                        // Check if the magnitude exceeds the sparsity threshold
                        if (std::abs(kernel_val_conj) >= sparsity_threshold_)
                        {
                            // Store the non-zero conjugated value in the sparse map (frequency index -> value)
                            precomputed_sparse_kernels_[octave_idx][k][j] = kernel_val_conj;
                        }
                    } // end kernel sparsification loop (j)
                } // end bin loop (k)

                // Prepare for the next octave (lower sample rate) if not the last octave
                if (octave_idx < num_octaves_ - 1)
                {
                    current_sr /= 2.0;
                }
            } // end octave loop (octave_idx)
        } // end if(num_octaves_ > 0)
    }

    // --- Destructor ---
    template <typename T>
    CQTPlan<T>::~CQTPlan() = default; // Default destructor is sufficient because unique_ptrs manage FFTPlan resources.

    // --- Move Semantics ---
    // Define move constructor and assignment operator for efficient resource transfer.
    template <typename T>
    CQTPlan<T>::CQTPlan(CQTPlan &&other) noexcept = default; // Use compiler-generated move constructor
    template <typename T>
    CQTPlan<T> &CQTPlan<T>::operator=(CQTPlan<T> &&other) noexcept = default; // Use compiler-generated move assignment

    // --- Execute Method ---
    // Performs the CQT on an input signal using the precomputed resources.
    template <typename T>
    void CQTPlan<T>::execute(const std::vector<T> &input, std::vector<std::vector<std::complex<T>>> &output) const
    {
        // Handle cases where the plan has no bins/octaves (e.g., invalid frequency range)
        if (num_bins_ == 0 || num_octaves_ == 0)
        {
            output.clear(); // Return empty output
            return;
        }

        // Handle empty input signal
        if (input.empty())
        {
            // Return vectors of the correct size but filled with zeros (or empty vectors)
            output.assign(num_bins_, {}); // Assigns num_bins_ empty vectors
            return;
        }

        std::vector<T> current_y = input; // Start with the original signal
        double current_sr = sample_rate_; // Start with the original sample rate
        size_t current_hop = hop_length_; // Start with the original hop length
        size_t num_frames = 0;            // Number of output time frames (determined later)
        // Storage for CQT coefficients calculated per octave
        std::vector<std::vector<std::vector<std::complex<T>>>> all_octave_coeffs(num_octaves_);

        // --- Process Octaves Recursively (Highest Frequency First) ---
        // Loop from the highest octave index down to the lowest (0).
        for (int octave_idx = num_octaves_ - 1; octave_idx >= 0; --octave_idx)
        {
            // Check if the signal became empty after downsampling in previous iterations
            if (current_y.empty())
            {
                std::cerr << "Warning: Signal became empty during CQT processing at octave " << octave_idx << ". Skipping lower octaves." << std::endl;
                // Mark remaining lower octaves as having no data
                for (int rem_idx = octave_idx; rem_idx >= 0; --rem_idx)
                {
                    int bins_in_rem_octave = (rem_idx == num_octaves_ - 1)
                                                 ? (static_cast<int>(num_bins_) - rem_idx * bins_per_octave_)
                                                 : bins_per_octave_;
                    all_octave_coeffs[rem_idx].assign(std::max(0, bins_in_rem_octave), {}); // Assign empty vectors
                }
                break; // Stop processing octaves
            }

            // Calculate CQT coefficients for the current octave using the current signal (current_y)
            std::vector<std::vector<std::complex<T>>> octave_cqt_coeffs =
                calculateSingleOctaveCQT(current_y, current_sr, current_hop, octave_idx);

            // Store the results for this octave
            all_octave_coeffs[octave_idx] = std::move(octave_cqt_coeffs); // Move to avoid copy

            // Determine the total number of time frames from the first valid octave processed
            // (Assumes all octaves will yield the same number of frames if input is long enough)
            if (num_frames == 0 && !all_octave_coeffs[octave_idx].empty() && !all_octave_coeffs[octave_idx][0].empty())
            {
                num_frames = all_octave_coeffs[octave_idx][0].size();
            }

            // --- Prepare for the Next (Lower Frequency) Octave ---
            // Filter and downsample the *current* signal (current_y) by 2 if this isn't the lowest octave.
            if (octave_idx > 0)
            {
                try
                {
                    // Call the helper function which uses the backend implementation
                    current_y = filterAndDownsampleBy2(current_y, current_sr);
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error("Filtering/Downsampling failed preparing for octave " + std::to_string(octave_idx - 1) + ": " + e.what());
                }
                current_sr /= 2.0; // Halve the sample rate for the next iteration
                current_hop /= 2;  // Halve the hop length for the next iteration

                // Sanity check for hop length (should not happen if constructor validation is correct)
                if (current_hop == 0 && num_octaves_ > 1)
                {
                    throw std::logic_error("Internal error: CQT hop length became zero during recursion.");
                }
            }
        } // End octave processing loop

        // --- Combine Results from All Octaves ---
        // Handle case where num_frames might still be 0 (e.g., input signal was too short for even the highest octave FFT)
        if (num_frames == 0 && !input.empty() && !octave_fft_lens_.empty())
        {
            // Estimate frames based on input length and original hop length of the highest octave
            size_t first_fft_len = octave_fft_lens_.back(); // FFT length of the highest octave processed (last in loop)
            size_t first_hop_len = hop_length_;             // Original hop length
            num_frames = (input.size() >= first_fft_len) ? (1 + (input.size() - first_fft_len) / first_hop_len) : (input.size() > 0 ? 1 : 0);
        }

        // Resize the final output matrix to [total_bins][num_frames] and initialize with zeros
        output.assign(num_bins_, std::vector<std::complex<T>>(num_frames, {0.0, 0.0}));

        size_t current_bin_offset = 0; // Keep track of the starting bin index for the current octave
        // Loop through octaves again (lowest to highest) to assemble the final output matrix
        for (int octave_idx = 0; octave_idx < num_octaves_; ++octave_idx)
        {
            const auto &octave_data = all_octave_coeffs[octave_idx]; // Get the computed CQT data for this octave
            // Calculate the expected number of bins in this octave
            int bins_in_this_octave = (octave_idx == num_octaves_ - 1)
                                          ? (static_cast<int>(num_bins_) - octave_idx * bins_per_octave_)
                                          : bins_per_octave_;
            bins_in_this_octave = std::max(0, bins_in_this_octave); // Ensure non-negative

            // Check if this octave actually produced data
            if (octave_data.empty() || octave_data[0].empty())
            {
                current_bin_offset += bins_in_this_octave; // Advance offset even if no data
                continue;                                  // Skip to the next octave
            }

            size_t bins_in_oct_data = octave_data.size();         // Actual number of bins with data
            size_t frames_in_this_octave = octave_data[0].size(); // Actual number of frames with data

            // Check for consistency between expected and actual bin count
            if (static_cast<size_t>(bins_in_this_octave) != bins_in_oct_data)
            {
                std::cerr << "Warning: CQT bin count mismatch for octave " << octave_idx
                          << " (expected " << bins_in_this_octave << ", got " << bins_in_oct_data << "). Using actual data size for offset." << std::endl;
                bins_in_this_octave = static_cast<int>(bins_in_oct_data); // Adjust offset based on actual data
            }

            // Handle potential frame count discrepancies (e.g., signal got shorter)
            size_t frames_to_copy = std::min(num_frames, frames_in_this_octave); // Copy up to the minimum number of frames
            if (frames_to_copy == 0)
            {
                current_bin_offset += bins_in_this_octave;
                continue; // Skip if no frames to copy
            }

            // Copy data from the octave results to the final output matrix
            for (size_t bin = 0; bin < bins_in_oct_data; ++bin)
            {
                size_t target_bin = current_bin_offset + bin; // Calculate the target bin index in the final output
                if (target_bin < num_bins_)
                { // Ensure target bin is within the total number of bins
                    if (octave_data[bin].size() >= frames_to_copy)
                    { // Check source frame count
                        // Copy the relevant frame data
                        std::copy(octave_data[bin].begin(),
                                  octave_data[bin].begin() + frames_to_copy,
                                  output[target_bin].begin());
                    }
                    else
                    {
                        std::cerr << "Warning: CQT frame count inconsistency within octave " << octave_idx << ", bin " << bin << std::endl;
                    }
                }
            }
            current_bin_offset += bins_in_this_octave; // Update the offset for the next octave
        } // End combining loop

        // Final safety check: ensure all output vectors have the correct number of frames
        for (auto &bin_vec : output)
        {
            if (bin_vec.size() != num_frames)
            {
                bin_vec.resize(num_frames, {0.0, 0.0}); // Resize and pad with zeros if necessary
            }
        }
    }

    // --- Helper Function Implementations ---

    /**
     * @brief Calculates CQT for a single octave using precomputed resources.
     * Implements framing, RFFT per frame, and sparse kernel correlation.
     */
    template <typename T>
    std::vector<std::vector<std::complex<T>>> CQTPlan<T>::calculateSingleOctaveCQT(
        const std::vector<T> &signal, double current_sample_rate /* unused? */, size_t current_hop_length,
        int octave_idx // Pass octave index to retrieve precomputed resources
    ) const
    {
        // --- Basic Validation and Resource Retrieval ---
        if (octave_idx < 0 || octave_idx >= num_octaves_)
        {
            throw std::out_of_range("calculateSingleOctaveCQT: Invalid octave index.");
        }
        if (signal.empty() || current_hop_length == 0)
        {
            int bins_in_this_octave = (octave_idx == num_octaves_ - 1)
                                          ? (static_cast<int>(num_bins_) - octave_idx * bins_per_octave_)
                                          : bins_per_octave_;
            return std::vector<std::vector<std::complex<T>>>(std::max(0, bins_in_this_octave)); // Return empty frames
        }

        // Retrieve precomputed resources for this octave
        const FFTPlan<T> *fft_plan = octave_signal_fft_plans_[octave_idx].get();         // The RFFT plan for signal frames
        const SparseKernelOctave &kernels_oct = precomputed_sparse_kernels_[octave_idx]; // Sparse kernels for this octave
        size_t fft_len_oct = octave_fft_lens_[octave_idx];                               // FFT length for this octave
        size_t spectrum_len_oct = octave_spectrum_lens_[octave_idx];                     // Spectrum length (N/2+1)
        int octave_num_bins = static_cast<int>(kernels_oct.size());                      // Number of CQT bins in this octave

        if (!fft_plan || octave_num_bins <= 0)
        {
            if (octave_num_bins == 0)
                return {}; // Valid case if no bins
            throw std::runtime_error("Internal CQT error: Resources not initialized for octave " + std::to_string(octave_idx));
        }

        size_t num_samples = signal.size();
        // Calculate number of frames based on signal length, FFT length, and hop length
        size_t num_frames = (num_samples >= fft_len_oct) ? (1 + (num_samples - fft_len_oct) / current_hop_length) : (num_samples > 0 ? 1 : 0);
        if (num_frames == 0)
        {
            return std::vector<std::vector<std::complex<T>>>(octave_num_bins); // Return empty frames if no frames possible
        }

        // Allocate output matrix [bin][frame]
        std::vector<std::vector<std::complex<T>>> output_coeffs(octave_num_bins, std::vector<std::complex<T>>(num_frames, {0.0, 0.0}));
        // Reusable buffers for frame data and its spectrum
        std::vector<T> frame(fft_len_oct);
        std::vector<std::complex<T>> frame_spectrum(spectrum_len_oct);

        // --- Framing Loop ---
        // Iterate through each frame based on the current_hop_length
        for (size_t t = 0; t < num_frames; ++t)
        {
            size_t frame_start = t * current_hop_length;
            // Ensure frame doesn't read past the end of the signal
            size_t frame_end = std::min(frame_start + fft_len_oct, num_samples);
            size_t current_frame_len = frame_end - frame_start;

            // Copy frame data into the buffer
            std::copy(signal.begin() + frame_start, signal.begin() + frame_end, frame.begin());
            // Zero-pad the frame buffer if it's shorter than the FFT length (e.g., at the end of the signal)
            if (current_frame_len < fft_len_oct)
            {
                std::fill(frame.begin() + current_frame_len, frame.end(), T(0));
            }

            // a. Compute RFFT of the frame using the precomputed plan
            try
            {
                // Call the execute_rfft method of the FFTPlan object for this octave
                fft_plan->execute_rfft(frame.data(), frame_spectrum.data());
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Signal frame FFT failed for octave " + std::to_string(octave_idx) + ", frame " + std::to_string(t) + ": " + e.what());
            }

            // b. Correlate spectrum with sparse kernels for this octave
            // Iterate through each CQT bin (k) within this octave
            for (int k = 0; k < octave_num_bins; ++k)
            {
                std::complex<T> cqt_val = {0.0, 0.0};           // Accumulator for the dot product
                const auto &sparse_kernel_bin = kernels_oct[k]; // Get the sparse kernel (std::map) for this bin

                // --- Sparse Dot Product Implementation ---
                // Iterate through the non-zero elements (frequency index -> complex value pairs)
                // stored in the sparse kernel map for the current bin k.
                for (const auto &pair : sparse_kernel_bin)
                {
                    size_t freq_idx = pair.first;                             // Frequency index (key in the map)
                    const std::complex<T> &kernel_element_conj = pair.second; // Pre-conjugated kernel value (value in the map)

                    // Check bounds: Ensure the frequency index from the kernel exists in the frame spectrum
                    if (freq_idx < spectrum_len_oct)
                    {
                        // Multiply the frame's spectrum value at this index by the (conjugated) kernel value
                        // and add it to the accumulator. This performs the dot product efficiently.
                        cqt_val += frame_spectrum[freq_idx] * kernel_element_conj;
                    }
                } // End sparse kernel element loop (pair)

                // c. Apply final scaling factor
                // [TODO - Task I.cqt.cpp.TuneScalingFactor] This might need adjustment based on window L1 norm and FFT scaling.
                T scale_factor = T(1.0) / static_cast<T>(fft_len_oct); // Placeholder scaling (e.g., 1/N FFT scaling)
                output_coeffs[k][t] = cqt_val * scale_factor;          // Store the final CQT coefficient for this bin/frame

            } // End CQT bin loop (k)
        } // End frame loop (t)

        return output_coeffs; // Return the computed coefficients for this octave
    }

    /**
     * @brief Calculates FIR filter coefficients for anti-aliasing using the windowed-sinc method.
     * Designs a low-pass filter with cutoff at fs/4 suitable for downsampling by 2.
     * @param current_sample_rate The sample rate for which the filter is designed (Hz).
     * @param N The desired filter order (length, must be odd and positive).
     * @return A vector containing the normalized filter coefficients.
     */
    template <typename T>
    std::vector<T> CQTPlan<T>::calculateFirCoefficients(double current_sample_rate, int N) const
    {
        // --- Input Validation ---
        if (N <= 0)
            throw std::invalid_argument("FIR filter order N must be positive.");
        if (N % 2 == 0)
            std::cerr << "Warning: FIR filter order N (" << N << ") should preferably be odd for windowed-sinc." << std::endl;
        if (current_sample_rate <= 0.0)
            throw std::invalid_argument("Current sample rate must be positive.");

        // --- Parameters ---
        double cutoff_freq = current_sample_rate / 4.0; // Cutoff at fs/4 for downsampling by 2
        double fc = cutoff_freq / current_sample_rate;  // Normalized cutoff frequency (0.25)
        // Center index of the filter (works for odd and even N)
        double M_double = static_cast<double>(N - 1) / 2.0;

        std::vector<T> h_coeffs(N);      // Stores the final filter coefficients
        std::vector<T> window_coeffs(N); // Stores the window coefficients (e.g., Hann)

        // --- Generate Window (Hann window used here) ---
        // Denominator for cosine argument, handle N=1 case.
        double window_denom = (N > 1) ? static_cast<double>(N - 1) : 1.0;
        for (int n = 0; n < N; ++n)
        {
            window_coeffs[n] = static_cast<T>(0.5 - 0.5 * cos(2.0 * M_PI * static_cast<double>(n) / window_denom));
        }

        // --- Calculate Windowed-Sinc Coefficients ---
        T sum_h = T(0.0); // Accumulator for normalization
        for (int n = 0; n < N; ++n)
        {
            T ideal_response; // Ideal sinc response
            // Calculate distance from center index (using double for precision)
            double n_minus_M = static_cast<double>(n) - M_double;
            // Check if we are exactly at the center point (n == M -> n_minus_M == 0)
            if (std::abs(n_minus_M) < std::numeric_limits<double>::epsilon() * 10)
            { // Use tolerance for float comparison
                // The limit of sinc(x) as x->0 is 1. The value is 2*fc * 1.
                ideal_response = static_cast<T>(2.0 * fc);
            }
            else
            {
                // Calculate the argument for the sinc function
                double x = 2.0 * M_PI * fc * n_minus_M;
                // Calculate sinc(x) = sin(pi*x)/(pi*x). Here, sin(2*pi*fc*(n-M)) / (pi*(n-M))
                // Simplified to sin(x) / (pi * (n-M)) where x = pi * 2*fc * (n-M)
                // Check denominator is non-zero before division
                if (std::abs(M_PI * n_minus_M) < std::numeric_limits<double>::epsilon())
                {
                    // Should not be hit if center point check above is correct, but as fallback:
                    ideal_response = static_cast<T>(2.0 * fc);
                }
                else
                {
                    ideal_response = static_cast<T>(sin(x) / (M_PI * n_minus_M));
                }
            }

            // Multiply ideal sinc response by the window coefficient
            h_coeffs[n] = ideal_response * window_coeffs[n];
            sum_h += h_coeffs[n]; // Accumulate sum for normalization
        }

        // --- Normalize Coefficients (Ensure DC gain is 1) ---
        if (std::abs(sum_h) > std::numeric_limits<T>::epsilon() * N)
        { // Check if sum is significantly non-zero
            for (int n = 0; n < N; ++n)
            {
                h_coeffs[n] /= sum_h; // Divide each coefficient by the sum
            }
        }
        else
        {
            std::cerr << "Warning: FIR filter coefficient sum is close to zero. Normalization skipped. Filter might be ineffective." << std::endl;
            // Fallback: Create a delta function (pass-through) if normalization fails
            if (N > 0)
            {
                std::fill(h_coeffs.begin(), h_coeffs.end(), T(0.0));
                int center_idx = (N - 1) / 2; // Center index
                if (center_idx >= 0 && center_idx < N)
                {
                    h_coeffs[center_idx] = T(1.0); // Set center tap to 1
                }
            }
        }

        return h_coeffs; // Return the calculated coefficients
    }

    /**
     * @brief Applies the anti-aliasing FIR filter and downsamples the signal by 2.
     * This function calculates the necessary FIR coefficients and then calls the
     * appropriate backend implementation (`filter_and_downsample_impl`) to perform
     * the combined operation.
     * @param signal The input signal vector.
     * @param current_sample_rate The sample rate of the input signal (Hz).
     * @return The filtered and downsampled signal vector.
     */
    template <typename T>
    std::vector<T> CQTPlan<T>::filterAndDownsampleBy2(
        const std::vector<T> &signal, double current_sample_rate) const
    {
        // Handle empty input
        if (signal.empty())
        {
            return {};
        }

        // 1. Calculate FIR coefficients using the stored filter order
        std::vector<T> coefficients;
        try
        {
            // Use the member variable 'fir_filter_order_' set during construction
            coefficients = calculateFirCoefficients(current_sample_rate, fir_filter_order_);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Failed to calculate FIR coefficients for downsampling: " + std::string(e.what()));
        }

        // 2. Call the backend implementation function
        const int downsample_factor = 2;
        std::vector<T> result;
        try
        {
            // Call the function pointer/template defined in Backend namespace
            result = Backend::filter_and_downsample_impl(signal, coefficients, downsample_factor);
        }
        catch (const std::exception &e)
        {
            // Catch potential errors from the backend (e.g., stub implementation, MKL/Accelerate errors)
            throw std::runtime_error("Backend filter_and_downsample failed: " + std::string(e.what()));
        }

        // 3. Return the result from the backend
        return result;
    }

    // --- Explicit Template Instantiations ---
    // Instantiate the CQTPlan class template for float and double.
    // This forces the compiler to generate the code for both types, making them available for linking.
    template class CQTPlan<float>;
    template class CQTPlan<double>;

} // namespace OmniDSP