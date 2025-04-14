/**
 * @file cqt.cpp
 * @brief Implementation of the CQTPlan class using the recursive downsampling method
 * with precomputed sparse kernels. Includes compile-time log guards.
 * Added check to prevent calling missing double-precision downsample backend.
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
     // Initializes CQT parameters and precomputes FFT plans and sparse spectral kernels.
     template <typename T>
     CQTPlan<T>::CQTPlan(double sample_rate, size_t hop_length, double lowest_freq, double highest_freq,
                         int bins_per_octave, WindowFuncType window_function, T sparsity_threshold,
                         int fir_filter_order)
         : sample_rate_(sample_rate), hop_length_(hop_length), lowest_freq_(lowest_freq), highest_freq_(highest_freq),
           bins_per_octave_(bins_per_octave), window_function_(window_function),
           sparsity_threshold_(sparsity_threshold)
     {
         // --- Parameter Validation ---
         if (sample_rate <= 0.0)
             throw std::invalid_argument("Sample rate must be positive.");
         if (lowest_freq <= 0.0)
             throw std::invalid_argument("Lowest frequency must be positive.");
         if (highest_freq <= lowest_freq)
             throw std::invalid_argument("Highest frequency must be greater than lowest frequency.");
         if (highest_freq > sample_rate / 2.0)
         {
             // Allow a small tolerance for floating point inaccuracies
             if (highest_freq > (sample_rate / 2.0) * (1.0 + std::numeric_limits<double>::epsilon() * 10))
             {
                 throw std::invalid_argument("Highest frequency (" + std::to_string(highest_freq) + ") cannot exceed Nyquist frequency (" + std::to_string(sample_rate / 2.0) + ").");
             }
             else
             {
                 // Clamp to Nyquist if slightly above due to precision issues
                 highest_freq_ = sample_rate / 2.0;
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS // Guard the warning
                 std::cerr << "Warning: Highest frequency exceeded Nyquist. Clamped to " << highest_freq_ << " Hz." << std::endl;
 #endif
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
             throw std::invalid_argument("FIR filter order must be odd for windowed-sinc design.");
         fir_filter_order_ = fir_filter_order; // Store the validated filter order
         if (sparsity_threshold_ < 0)
             throw std::invalid_argument("Sparsity threshold cannot be negative.");
 
         // --- Calculate Total Bin Count & Number of Octaves ---
         num_bins_ = static_cast<size_t>(std::ceil(bins_per_octave_ * std::log2(highest_freq_ / lowest_freq_)));
         if (num_bins_ == 0)
         {
             if (lowest_freq > 0)
             {
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS // Guard the warning
                 std::cerr << "Warning: Calculated number of CQT bins is zero based on frequency range. CQT output will be empty." << std::endl;
 #endif
             }
             else
             {
                 throw std::invalid_argument("Calculated number of CQT bins is zero due to invalid frequency range.");
             }
         }
         num_octaves_ = static_cast<int>(std::ceil(static_cast<double>(num_bins_) / bins_per_octave_));
         if (num_octaves_ < 1 && num_bins_ > 0)
             num_octaves_ = 1; // Ensure at least one octave if bins exist
 
         // --- Validate Hop Length Divisibility ---
         int64_t divisor = (num_octaves_ > 0) ? (1LL << (num_octaves_ - 1)) : 1LL;
         if (divisor > 1 && hop_length_ % divisor != 0)
         {
             throw std::invalid_argument("Hop length (" + std::to_string(hop_length_) + ") must be divisible by 2^(" + std::to_string(num_octaves_ - 1) + ") = " + std::to_string(divisor) + " for recursive CQT.");
         }
 
         // --- Precompute Resources for EACH Octave ---
         if (num_octaves_ > 0)
         {
             // Resize storage vectors
             octave_fft_lens_.resize(num_octaves_);
             octave_spectrum_lens_.resize(num_octaves_);
             octave_signal_fft_plans_.resize(num_octaves_);
             octave_kernel_fft_plans_.resize(num_octaves_);
             precomputed_sparse_kernels_.resize(num_octaves_);
 
             double current_sr = sample_rate_;
             Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
             double Q = 1.0 / (std::pow(2.0, 1.0 / bins_per_octave_) - 1.0);
             std::vector<std::complex<T>> temp_kernel_fft_output; // Reusable buffer
 
             // Loop through each octave (0 = lowest frequencies)
             for (int octave_idx = 0; octave_idx < num_octaves_; ++octave_idx)
             {
                 // 1. Determine required FFT length for this octave
                 double octave_base_freq_abs = lowest_freq_ * std::pow(2.0, static_cast<double>(octave_idx));
                 size_t max_nk_this_octave = 1; // Default to 1
                 if (octave_base_freq_abs > std::numeric_limits<double>::epsilon())
                 {
                     // Nk = Q * fs / f (window length required for resolution)
                     max_nk_this_octave = static_cast<size_t>(std::max(1.0, std::round(Q * current_sr / octave_base_freq_abs)));
                 }
                 octave_fft_lens_[octave_idx] = nextPowerOf2(max_nk_this_octave);
                 size_t fft_len_oct = octave_fft_lens_[octave_idx];
 
                 // 2. Create FFT Plans for this octave's FFT length
                 try
                 {
                     // Plan for Real-to-Complex FFT of signal frames
                     octave_signal_fft_plans_[octave_idx] = std::make_unique<FFTPlan<T>>(
                         fft_len_oct, prec, Direction::FORWARD, Domain::REAL, NormMode::BACKWARD);
                     // Plan for Complex-to-Complex FFT used during kernel generation
                     octave_kernel_fft_plans_[octave_idx] = std::make_unique<FFTPlan<T>>(
                         fft_len_oct, prec, Direction::FORWARD, Domain::COMPLEX, NormMode::BACKWARD);
 
                     if (!octave_signal_fft_plans_[octave_idx] || !octave_kernel_fft_plans_[octave_idx])
                     {
                         throw std::runtime_error("Failed to allocate FFTPlan.");
                     }
                     octave_spectrum_lens_[octave_idx] = octave_signal_fft_plans_[octave_idx]->getComplexLength();
                 }
                 catch (const std::exception &e)
                 {
                     throw std::runtime_error("Failed to create FFTPlan for octave " + std::to_string(octave_idx) + ": " + e.what());
                 }
 
                 // 3. Generate Sparse Kernels for each bin within this octave
                 int bins_in_this_octave = (octave_idx == num_octaves_ - 1)
                                               ? (static_cast<int>(num_bins_) - octave_idx * bins_per_octave_)
                                               : bins_per_octave_;
                 bins_in_this_octave = std::max(0, bins_in_this_octave); // Ensure non-negative
                 if (bins_in_this_octave <= 0)
                     continue;
 
                 precomputed_sparse_kernels_[octave_idx].resize(bins_in_this_octave);
                 temp_kernel_fft_output.resize(fft_len_oct);
                 double downsampling_factor = sample_rate_ / current_sr;
 
                 // Loop through each bin (k) within the current octave
                 for (int k = 0; k < bins_in_this_octave; ++k)
                 {
                     int absolute_bin_index = octave_idx * bins_per_octave_ + k;
                     double freq_k_abs = lowest_freq_ * std::pow(2.0, static_cast<double>(absolute_bin_index) / bins_per_octave_);
                     double freq_k_relative_to_current_sr = freq_k_abs / downsampling_factor;
 
                     // Calculate required window length (Nk)
                     size_t Nk = 1; // Default to 1
                     if (freq_k_relative_to_current_sr > std::numeric_limits<double>::epsilon())
                     {
                         Nk = static_cast<size_t>(std::max(1.0, std::round(Q * current_sr / freq_k_relative_to_current_sr)));
                     }
 
                     // Ensure Nk is valid and clamp if necessary
                     if (Nk == 0 || Nk > fft_len_oct)
                     {
 // --- START Log Guard ---
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS
                         std::cerr << "Warning: Clamping Nk (" << Nk << ") to FFT length (" << fft_len_oct
                                   << ") for octave " << octave_idx << ", bin " << k << std::endl;
 #endif
                         // --- END Log Guard ---
                         Nk = std::min(Nk, fft_len_oct);
                         if (Nk == 0)
                             continue; // Skip bin if Nk becomes 0
                     }
 
                     // a. Generate time-domain window using the provided function
                     std::vector<T> time_window;
                     try
                     {
                         time_window = window_function_(Nk);
                     }
                     catch (const std::exception &e)
                     {
                         throw std::runtime_error("Window function failed during kernel generation for octave " + std::to_string(octave_idx) + ", bin " + std::to_string(k) + ": " + e.what());
                     }
                     if (time_window.size() != Nk)
                     {
                         throw std::runtime_error("Window func returned incorrect size (" + std::to_string(time_window.size()) + ", expected " + std::to_string(Nk) + ") in constructor kernel gen.");
                     }
 
                     // Normalize the time-domain window (L1 norm)
                     T l1_norm = T(0.0);
                     for (T val : time_window)
                         l1_norm += std::abs(val);
                     if (l1_norm > std::numeric_limits<T>::epsilon())
                     {
                         for (T &val : time_window)
                             val /= l1_norm;
                     }
                     else
                     {
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS // Guard the warning
                         std::cerr << "Warning: Window L1 norm is zero for octave " << octave_idx << ", bin " << k << ". Kernel might be zero." << std::endl;
 #endif
                     }
 
                     // b. Pad the window with zeros and center it
                     std::vector<T> padded_time_window(fft_len_oct, 0.0);
                     long long start_idx_ll = static_cast<long long>(fft_len_oct / 2) - static_cast<long long>(Nk / 2);
                     size_t start_idx = (start_idx_ll < 0) ? 0 : static_cast<size_t>(start_idx_ll);
                     size_t copy_len = std::min(Nk, fft_len_oct - start_idx);
                     std::copy(time_window.begin(), time_window.begin() + copy_len, padded_time_window.begin() + start_idx);
 
                     // c. Apply complex frequency shift
                     std::vector<std::complex<T>> complex_shifted_window(fft_len_oct);
                     double shift_phase_increment = -2.0 * M_PI * freq_k_relative_to_current_sr / current_sr;
                     for (size_t n = 0; n < fft_len_oct; ++n)
                     {
                         double phase = shift_phase_increment * static_cast<double>(n);
                         complex_shifted_window[n] = padded_time_window[n] * std::complex<T>(std::cos(phase), std::sin(phase));
                     }
 
                     // d. Compute FFT of the complex, shifted, padded window
                     try
                     {
                         octave_kernel_fft_plans_[octave_idx]->execute(complex_shifted_window.data(), temp_kernel_fft_output.data());
                     }
                     catch (const std::exception &e)
                     {
                         throw std::runtime_error("Kernel FFT failed for octave " + std::to_string(octave_idx) + ", bin " + std::to_string(k) + ": " + e.what());
                     }
 
                     // e. Sparsify and Store the CONJUGATED kernel
                     size_t spectrum_len_oct = octave_spectrum_lens_[octave_idx];
                     precomputed_sparse_kernels_[octave_idx][k].clear();
                     for (size_t j = 0; j < spectrum_len_oct; ++j)
                     { // Only iterate up to spectrum length
                         std::complex<T> kernel_val_conj = std::conj(temp_kernel_fft_output[j]);
                         if (std::abs(kernel_val_conj) >= sparsity_threshold_)
                         {
                             precomputed_sparse_kernels_[octave_idx][k][j] = kernel_val_conj;
                         }
                     } // end kernel sparsification loop (j)
                 } // end bin loop (k)
 
                 // Prepare for the next octave
                 if (octave_idx < num_octaves_ - 1)
                 {
                     current_sr /= 2.0;
                 }
             } // end octave loop (octave_idx)
         } // end if(num_octaves_ > 0)
     }
 
     // --- Destructor ---
     template <typename T>
     CQTPlan<T>::~CQTPlan() = default; // Default destructor is sufficient
 
     // --- Move Semantics ---
     template <typename T>
     CQTPlan<T>::CQTPlan(CQTPlan &&other) noexcept = default; // Use compiler-generated move constructor
     template <typename T>
     CQTPlan<T> &CQTPlan<T>::operator=(CQTPlan<T> &&other) noexcept = default; // Use compiler-generated move assignment
 
     // --- Execute Method ---
     // Performs the CQT on an input signal using the precomputed resources.
     template <typename T>
     void CQTPlan<T>::execute(const std::vector<T> &input, std::vector<std::vector<std::complex<T>>> &output) const
     {
         // Handle cases where the plan has no bins/octaves
         if (num_bins_ == 0 || num_octaves_ == 0)
         {
             output.clear();
             return;
         }
         // Handle empty input signal
         if (input.empty())
         {
             output.assign(num_bins_, {}); // Assigns num_bins_ empty vectors
             return;
         }
 
         std::vector<T> current_y = input; // Start with the original signal
         double current_sr = sample_rate_;
         size_t current_hop = hop_length_;
         size_t num_frames = 0; // Determined later
         std::vector<std::vector<std::vector<std::complex<T>>>> all_octave_coeffs(num_octaves_);
 
         // --- Process Octaves Recursively (Highest Frequency First) ---
         for (int octave_idx = num_octaves_ - 1; octave_idx >= 0; --octave_idx)
         {
             if (current_y.empty())
             {
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS // Guard the warning
                 std::cerr << "Warning: Signal became empty during CQT processing at octave " << octave_idx << ". Skipping lower octaves." << std::endl;
 #endif
                 for (int rem_idx = octave_idx; rem_idx >= 0; --rem_idx)
                 {
                     int bins_in_rem_octave = (rem_idx == num_octaves_ - 1) ? (static_cast<int>(num_bins_) - rem_idx * bins_per_octave_) : bins_per_octave_;
                     all_octave_coeffs[rem_idx].assign(std::max(0, bins_in_rem_octave), {});
                 }
                 break;
             }
 
             // Calculate CQT coefficients for the current octave
             std::vector<std::vector<std::complex<T>>> octave_cqt_coeffs =
                 calculateSingleOctaveCQT(current_y, current_sr, current_hop, octave_idx); // Pass current_sr
 
             all_octave_coeffs[octave_idx] = std::move(octave_cqt_coeffs);
 
             // Determine the total number of time frames from the first valid octave processed
             if (num_frames == 0 && !all_octave_coeffs[octave_idx].empty() && !all_octave_coeffs[octave_idx][0].empty())
             {
                 num_frames = all_octave_coeffs[octave_idx][0].size();
             }
 
             // --- Prepare for the Next (Lower Frequency) Octave ---
             if (octave_idx > 0)
             {
                 try
                 {
                     current_y = filterAndDownsampleBy2(current_y, current_sr);
                 }
                 catch (const std::exception &e)
                 {
                     throw std::runtime_error("Filtering/Downsampling failed preparing for octave " + std::to_string(octave_idx - 1) + ": " + e.what());
                 }
                 current_sr /= 2.0;
                 current_hop /= 2;
                 if (current_hop == 0 && num_octaves_ > 1)
                 {
                     throw std::logic_error("Internal error: CQT hop length became zero during recursion.");
                 }
             }
         } // End octave processing loop
 
         // --- Combine Results from All Octaves ---
         // Estimate frames if still zero (e.g., very short input)
         if (num_frames == 0 && !input.empty() && !octave_fft_lens_.empty())
         {
             size_t first_fft_len = octave_fft_lens_.back();
             size_t first_hop_len = hop_length_;
             num_frames = (input.size() >= first_fft_len) ? (1 + (input.size() - first_fft_len) / first_hop_len) : (input.size() > 0 ? 1 : 0);
         }
 
         // Resize the final output matrix and initialize with zeros
         output.assign(num_bins_, std::vector<std::complex<T>>(num_frames, {0.0, 0.0}));
 
         size_t current_bin_offset = 0;
         // Loop through octaves again (lowest to highest) to assemble
         for (int octave_idx = 0; octave_idx < num_octaves_; ++octave_idx)
         {
             const auto &octave_data = all_octave_coeffs[octave_idx];
             int bins_in_this_octave = (octave_idx == num_octaves_ - 1) ? (static_cast<int>(num_bins_) - octave_idx * bins_per_octave_) : bins_per_octave_;
             bins_in_this_octave = std::max(0, bins_in_this_octave);
 
             if (octave_data.empty() || (num_frames > 0 && octave_data[0].empty()))
             { // Check if frames exist if expected
                 current_bin_offset += bins_in_this_octave;
                 continue;
             }
 
             size_t bins_in_oct_data = octave_data.size();
             size_t frames_in_this_octave = (num_frames > 0) ? octave_data[0].size() : 0; // Use num_frames check
             if (static_cast<size_t>(bins_in_this_octave) != bins_in_oct_data)
             {
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS // Guard the warning
                 std::cerr << "Warning: CQT bin count mismatch for octave " << octave_idx
                           << " (expected " << bins_in_this_octave << ", got " << bins_in_oct_data << "). Using actual data size for offset." << std::endl;
 #endif
                 bins_in_this_octave = static_cast<int>(bins_in_oct_data);
             }
             size_t frames_to_copy = std::min(num_frames, frames_in_this_octave);
             if (frames_to_copy == 0)
             {
                 current_bin_offset += bins_in_this_octave;
                 continue;
             }
 
             // Copy data to the final output matrix
             for (size_t bin = 0; bin < bins_in_oct_data; ++bin)
             {
                 size_t target_bin = current_bin_offset + bin;
                 if (target_bin < num_bins_)
                 {
                     if (octave_data[bin].size() >= frames_to_copy)
                     {
                         std::copy(octave_data[bin].begin(), octave_data[bin].begin() + frames_to_copy, output[target_bin].begin());
                     }
                     else
                     {
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS // Guard the warning
                         std::cerr << "Warning: CQT frame count inconsistency within octave " << octave_idx << ", bin " << bin << std::endl;
 #endif
                     }
                 }
             }
             current_bin_offset += bins_in_this_octave;
         } // End combining loop
 
         // Final safety check for frame counts
         for (auto &bin_vec : output)
         {
             if (bin_vec.size() != num_frames)
             {
                 bin_vec.resize(num_frames, {0.0, 0.0});
             }
         }
     }
 
     // --- Getter Definitions (Moved from Header) ---
     template <typename T>
     size_t CQTPlan<T>::getNumBins() const
     {
         return num_bins_;
     }
 
     template <typename T>
     double CQTPlan<T>::getSampleRate() const
     {
         return sample_rate_;
     }
 
     template <typename T>
     size_t CQTPlan<T>::getHopLength() const
     {
         return hop_length_;
     }
 
     template <typename T>
     double CQTPlan<T>::getLowestFrequency() const
     {
         return lowest_freq_;
     }
 
     template <typename T>
     double CQTPlan<T>::getHighestFrequency() const
     {
         return highest_freq_;
     }
 
     template <typename T>
     int CQTPlan<T>::getBinsPerOctave() const
     {
         return bins_per_octave_;
     }
 
     template <typename T>
     int CQTPlan<T>::getNumOctaves() const
     {
         return num_octaves_;
     }
 
     template <typename T>
     T CQTPlan<T>::getSparsityThreshold() const
     {
         return sparsity_threshold_;
     }
 
     template <typename T>
     int CQTPlan<T>::getFirFilterOrder() const
     {
         return fir_filter_order_;
     }
 
     // --- Helper Function Implementations ---
 
     /**
      * @brief Calculates CQT for a single octave using precomputed resources.
      */
     template <typename T>
     std::vector<std::vector<std::complex<T>>> CQTPlan<T>::calculateSingleOctaveCQT(
         const std::vector<T> &signal, double current_sr, size_t current_hop_length, // Added current_sr
         int octave_idx) const
     {
         // --- Basic Validation and Resource Retrieval ---
         if (octave_idx < 0 || octave_idx >= num_octaves_)
         {
             throw std::out_of_range("calculateSingleOctaveCQT: Invalid octave index.");
         }
         int bins_in_this_octave = (octave_idx == num_octaves_ - 1) ? (static_cast<int>(num_bins_) - octave_idx * bins_per_octave_) : bins_per_octave_;
         bins_in_this_octave = std::max(0, bins_in_this_octave);
         if (signal.empty() || current_hop_length == 0 || bins_in_this_octave == 0)
         {
             return std::vector<std::vector<std::complex<T>>>(bins_in_this_octave); // Return empty frames
         }
 
         // Retrieve precomputed resources
         const FFTPlan<T> *fft_plan = octave_signal_fft_plans_[octave_idx].get();
         const SparseKernelOctave &kernels_oct = precomputed_sparse_kernels_[octave_idx];
         size_t fft_len_oct = octave_fft_lens_[octave_idx];
         size_t spectrum_len_oct = octave_spectrum_lens_[octave_idx];
         int octave_num_bins = static_cast<int>(kernels_oct.size());
 
         if (!fft_plan || octave_num_bins <= 0 || static_cast<size_t>(octave_num_bins) != static_cast<size_t>(bins_in_this_octave))
         {
             if (octave_num_bins == 0 && bins_in_this_octave == 0)
                 return {}; // Valid case if no bins
             throw std::runtime_error("Internal CQT error: Resources mismatch or not initialized for octave " + std::to_string(octave_idx));
         }
 
         size_t num_samples = signal.size();
         // Calculate number of frames
         size_t num_frames = (num_samples >= fft_len_oct) ? (1 + (num_samples - fft_len_oct) / current_hop_length) : (num_samples > 0 ? 1 : 0);
         if (num_frames == 0)
         {
             return std::vector<std::vector<std::complex<T>>>(octave_num_bins); // Return empty frames
         }
 
         // Allocate output matrix [bin][frame]
         std::vector<std::vector<std::complex<T>>> output_coeffs(octave_num_bins, std::vector<std::complex<T>>(num_frames, {0.0, 0.0}));
         std::vector<T> frame(fft_len_oct);
         std::vector<std::complex<T>> frame_spectrum(spectrum_len_oct);
 
         // --- Framing Loop ---
         for (size_t t = 0; t < num_frames; ++t)
         {
             size_t frame_start = t * current_hop_length;
             size_t frame_end = std::min(frame_start + fft_len_oct, num_samples);
             size_t current_frame_len = frame_end - frame_start;
 
             // Copy frame data and zero-pad if necessary
             std::copy(signal.begin() + frame_start, signal.begin() + frame_end, frame.begin());
             if (current_frame_len < fft_len_oct)
             {
                 std::fill(frame.begin() + current_frame_len, frame.end(), T(0));
             }
 
             // a. Compute RFFT of the frame
             try
             {
                 fft_plan->execute_rfft(frame.data(), frame_spectrum.data());
             }
             catch (const std::exception &e)
             {
                 throw std::runtime_error("Signal frame FFT failed for octave " + std::to_string(octave_idx) + ", frame " + std::to_string(t) + ": " + e.what());
             }
 
             // b. Correlate spectrum with sparse kernels
             for (int k = 0; k < octave_num_bins; ++k)
             {
                 std::complex<T> cqt_val = {0.0, 0.0};
                 const auto &sparse_kernel_bin = kernels_oct[k];
 
                 // Sparse Dot Product
                 for (const auto &pair : sparse_kernel_bin)
                 {
                     size_t freq_idx = pair.first;
                     const std::complex<T> &kernel_element_conj = pair.second;
                     if (freq_idx < spectrum_len_oct)
                     { // Bounds check
                         cqt_val += frame_spectrum[freq_idx] * kernel_element_conj;
                     }
                 } // End sparse kernel element loop
 
                 // c. Apply final scaling factor (Set to 1.0 based on previous findings)
                 T scale_factor = T(1.0);
                 output_coeffs[k][t] = cqt_val * scale_factor;
 
             } // End CQT bin loop (k)
         } // End frame loop (t)
 
         return output_coeffs;
     }
 
     /**
      * @brief Calculates FIR filter coefficients for anti-aliasing using the windowed-sinc method.
      */
     template <typename T>
     std::vector<T> CQTPlan<T>::calculateFirCoefficients(double current_sample_rate, int N) const
     {
         // --- Input Validation ---
         if (N <= 0)
             throw std::invalid_argument("FIR filter order N must be positive.");
         if (N % 2 == 0)
         {
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS // Guard the warning
             std::cerr << "Warning: FIR filter order N (" << N << ") should preferably be odd for windowed-sinc." << std::endl;
 #endif
         }
         if (current_sample_rate <= 0.0)
             throw std::invalid_argument("Current sample rate must be positive.");
 
         // --- Parameters ---
         double cutoff_freq = current_sample_rate / 4.0;     // Cutoff at fs/4 for downsampling by 2
         double fc = cutoff_freq / current_sample_rate;      // Normalized cutoff frequency (0.25)
         double M_double = static_cast<double>(N - 1) / 2.0; // Center index
 
         std::vector<T> h_coeffs(N);
         std::vector<T> window_coeffs(N);
 
         // --- Generate Window (Hann) ---
         double window_denom = (N > 1) ? static_cast<double>(N - 1) : 1.0;
         for (int n = 0; n < N; ++n)
         {
             window_coeffs[n] = static_cast<T>(0.5 - 0.5 * cos(2.0 * M_PI * static_cast<double>(n) / window_denom));
         }
 
         // --- Calculate Windowed-Sinc Coefficients ---
         T sum_h = T(0.0);
         for (int n = 0; n < N; ++n)
         {
             T ideal_response;
             double n_minus_M = static_cast<double>(n) - M_double;
             if (std::abs(n_minus_M) < std::numeric_limits<double>::epsilon() * 10)
             {
                 ideal_response = static_cast<T>(2.0 * fc); // Limit of sinc(x) as x->0 is 1
             }
             else
             {
                 double x = 2.0 * M_PI * fc * n_minus_M;
                 ideal_response = static_cast<T>(sin(x) / (M_PI * n_minus_M));
             }
             h_coeffs[n] = ideal_response * window_coeffs[n];
             sum_h += h_coeffs[n];
         }
 
         // --- Normalize Coefficients (Ensure DC gain is 1) ---
         if (std::abs(sum_h) > std::numeric_limits<T>::epsilon() * N)
         {
             for (int n = 0; n < N; ++n)
             {
                 h_coeffs[n] /= sum_h;
             }
         }
         else
         {
 #ifdef OMNIDSP_ENABLE_CQT_WARNINGS // Guard the warning
             std::cerr << "Warning: FIR filter coefficient sum is close to zero. Normalization skipped. Filter might be ineffective." << std::endl;
 #endif
             // Fallback: Create a delta function
             if (N > 0)
             {
                 std::fill(h_coeffs.begin(), h_coeffs.end(), T(0.0));
                 int center_idx = (N - 1) / 2;
                 if (center_idx >= 0 && center_idx < N)
                 {
                     h_coeffs[center_idx] = T(1.0);
                 }
             }
         }
         return h_coeffs;
     }
 
     /**
      * @brief Applies the anti-aliasing FIR filter and downsamples the signal by 2.
      * Throws an error if called for double precision, as the backend is missing.
      */
     template <typename T>
     std::vector<T> CQTPlan<T>::filterAndDownsampleBy2(
         const std::vector<T> &signal, double current_sample_rate) const
     {
         // --- NEW: Add check for double precision ---
         if constexpr (std::is_same_v<T, double>) {
             throw std::runtime_error("CQTPlan internal downsampling is not supported for double precision with the current backend configuration (missing IPP _64f implementation).");
         }
         // --- End New Check ---
 
         if (signal.empty())
             return {};
 
         // 1. Calculate FIR coefficients
         std::vector<T> coefficients;
         try
         {
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
             // This call will now only happen for T=float
             result = Backend::filter_and_downsample_impl(signal, coefficients, downsample_factor);
         }
         catch (const std::exception &e)
         {
             // Catch potential errors from the float backend implementation
             throw std::runtime_error("Backend filter_and_downsample failed: " + std::string(e.what()));
         }
         return result;
     }
 
     // --- Explicit Template Instantiations ---
     // Instantiate the CQTPlan class template for float and double.
     template class OMNIDSP_EXPORT CQTPlan<float>;
     template class OMNIDSP_EXPORT CQTPlan<double>;
 
 } // namespace OmniDSP
 