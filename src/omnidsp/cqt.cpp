/**
 * @file cqt.cpp
 * @brief Implementation of the CQTPlan class using the efficient frequency-domain method.
 *
 * This file contains the implementation details for calculating the Constant Q Transform,
 * including kernel generation and FFT processing. It now uses a window function signature
 * that explicitly receives the required window length.
 *
 * @version 1.1.0
 * @date 2025-04-11 // Updated date
 */

 #include <OmniDSP/cqt.h>     // Include the updated header
 #include <OmniDSP/omnidsp.h> // For FFTPlan
 #include <OmniDSP/windows.h> // For default window functions if ever needed, and M_PI definition via cmath
 #include <cmath>
 #include <vector>
 #include <complex>
 #include <stdexcept>
 #include <numeric>   // For std::accumulate (if used for normalization)
 #include <limits>    // For std::numeric_limits
 #include <algorithm> // For std::max, std::copy
 #include <string>    // For exception messages
 #include <memory>    // For std::unique_ptr (used internally for FFTPlan)
 
 #ifndef M_PI // Ensure M_PI is defined (usually via <cmath>)
 #define M_PI 3.14159265358979323846
 #endif
 
 
 namespace OmniDSP {
 
 /**
  * @brief Constructor for the CQTPlan class (efficient version).
  *
  * Initializes the CQT transform by creating a single FFT plan and
  * precomputing frequency-domain kernels for each CQT bin. Uses the updated
  * window function signature taking size_t.
  *
  * @tparam T The floating-point type of the input data (float or double).
  * @param sample_rate The sample rate of the input signal in Hz.
  * @param lowest_freq The lowest frequency of interest for the CQT in Hz. Must be > 0.
  * @param highest_freq The highest frequency of interest for the CQT in Hz. Must be <= sample_rate / 2.
  * @param bins_per_octave The number of frequency bins per octave. Must be > 0.
  * @param window_function A function matching CQTPlan<T>::WindowFuncType signature (std::vector<T>(size_t)).
  * @throws std::invalid_argument If parameters are invalid or window_function is null.
  * @throws std::runtime_error If FFT plan creation or kernel generation fails.
  */
 template <typename T>
 CQTPlan<T>::CQTPlan(double sample_rate, double lowest_freq, double highest_freq, int bins_per_octave,
                     WindowFuncType window_function) // Constructor uses new WindowFuncType
     : sample_rate_(sample_rate), lowest_freq_(lowest_freq), highest_freq_(highest_freq),
       bins_per_octave_(bins_per_octave), window_function_(window_function) // Store new type
 {
     // --- Parameter Validation ---
     if (sample_rate <= 0.0) throw std::invalid_argument("Sample rate must be positive.");
     if (lowest_freq <= 0.0) throw std::invalid_argument("Lowest frequency must be positive.");
     if (highest_freq <= lowest_freq) throw std::invalid_argument("Highest frequency must be greater than lowest frequency.");
     if (highest_freq > sample_rate / 2.0) throw std::invalid_argument("Highest frequency cannot exceed Nyquist frequency (sample_rate / 2).");
     if (bins_per_octave <= 0) throw std::invalid_argument("Bins per octave must be positive.");
     if (!window_function_) throw std::invalid_argument("A valid window function must be provided."); // Check if function object is valid
 
 
     // --- Calculate Bin Count ---
     num_bins_ = static_cast<size_t>(std::ceil(bins_per_octave_ * std::log2(highest_freq_ / lowest_freq_)));
      if (num_bins_ == 0) {
          throw std::invalid_argument("Calculated number of CQT bins is zero. Check frequency range and bins per octave.");
      }
 
     // --- Determine Required FFT Length ---
     size_t max_window_length = 0;
     for (size_t k = 0; k < num_bins_; ++k) {
         double freq = lowest_freq_ * std::pow(2.0, static_cast<double>(k) / bins_per_octave_);
         if (freq <= std::numeric_limits<double>::epsilon()) {
              throw std::runtime_error("Calculated frequency for bin " + std::to_string(k) + " is too close to zero.");
         }
         double Q = 1.0 / (std::pow(2.0, 1.0 / bins_per_octave_) - 1.0);
         size_t window_length = static_cast<size_t>(std::max(1.0, std::round(Q * sample_rate_ / freq)));
         max_window_length = std::max(max_window_length, window_length);
     }
 
     // Choose FFT length (next power of 2 >= max_window_length)
     fft_len_ = 1;
     while (fft_len_ < max_window_length) {
         fft_len_ *= 2;
     }
     // Consider adding a minimum FFT length check here if desired
 
 
     // --- Create the Single Main FFT Plan ---
     Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
     try {
         // Use REAL domain for efficiency if input signal is real
         fft_plan_ = std::make_unique<FFTPlan<T>>(fft_len_, prec, Direction::FORWARD, Domain::REAL, NormMode::BACKWARD);
     } catch (const std::exception& e) {
         throw std::runtime_error("Failed to create main FFTPlan for CQT: " + std::string(e.what()));
     }
     const size_t spectrum_len = fft_plan_->getComplexLength(); // N/2 + 1 for REAL plan
 
     // --- Precompute Frequency-Domain Kernels ---
     cqt_kernels_.resize(num_bins_);
     std::vector<std::complex<T>> temp_fft_output(fft_len_); // Buffer for temporary C2C FFT
 
     // Create a temporary C2C FFT plan for kernel generation (FFT of complex shifted window)
     std::unique_ptr<FFTPlan<T>> temp_kernel_fft_plan;
      try {
          temp_kernel_fft_plan = std::make_unique<FFTPlan<T>>(fft_len_, prec, Direction::FORWARD, Domain::COMPLEX, NormMode::BACKWARD);
      } catch (const std::exception& e) {
          throw std::runtime_error("Failed to create temporary C2C FFTPlan for kernel generation: " + std::string(e.what()));
      }
 
 
     for (size_t k = 0; k < num_bins_; ++k) {
         double freq_k = lowest_freq_ * std::pow(2.0, static_cast<double>(k) / bins_per_octave_);
         double Q = 1.0 / (std::pow(2.0, 1.0 / bins_per_octave_) - 1.0);
         size_t window_length = static_cast<size_t>(std::max(1.0, std::round(Q * sample_rate_ / freq_k)));
 
         // a. Generate time-domain window for this bin by calling function with length
         std::vector<T> time_window;
         try {
             // Call the stored std::function with the required length
             time_window = window_function_(window_length);
         } catch (const std::exception& e) {
              // Catch exceptions from the provided callable (e.g., Python function via pybind11)
              throw std::runtime_error("Provided window function failed during call for bin " + std::to_string(k) + " (requested length " + std::to_string(window_length) + "): " + e.what());
         }
 
         // b. Check returned size (crucial validation)
         if (time_window.size() != window_length) {
              throw std::runtime_error("Window function returned incorrect size (" + std::to_string(time_window.size()) + ") for bin " + std::to_string(k) + ", expected " + std::to_string(window_length));
         }
 
         // c. Pad & Center the time-domain window
         std::vector<T> padded_time_window(fft_len_, 0.0);
         long long start_idx_ll = static_cast<long long>(fft_len_ / 2) - static_cast<long long>(window_length / 2);
         size_t start_idx = (start_idx_ll < 0) ? 0 : static_cast<size_t>(start_idx_ll);
         size_t end_idx = start_idx + window_length;
         if (end_idx > fft_len_) end_idx = fft_len_; // Prevent writing out of bounds if window is large
         for (size_t n = 0; n < (end_idx - start_idx); ++n) {
              padded_time_window[start_idx + n] = time_window[n];
         }
 
         // d. Apply frequency shift in time domain
         std::vector<std::complex<T>> complex_shifted_window(fft_len_);
         double shift_phase_increment = -2.0 * M_PI * freq_k / sample_rate_;
         for (size_t n = 0; n < fft_len_; ++n) {
             double phase = shift_phase_increment * static_cast<double>(n);
             complex_shifted_window[n] = padded_time_window[n] * std::complex<T>(std::cos(phase), std::sin(phase));
         }
 
         // e. Compute FFT of the shifted, padded window using the temporary C2C plan
          try {
             temp_kernel_fft_plan->execute(complex_shifted_window.data(), temp_fft_output.data());
          } catch (const std::exception& e) {
             throw std::runtime_error("Failed execute temporary FFT for kernel bin " + std::to_string(k) + ": " + std::string(e.what()));
          }
 
 
         // f. Normalize the kernel and store relevant part (conjugated)
         // Normalization by window_length is common for filterbank interpretation.
         T normalization_factor = static_cast<T>(window_length);
         if (normalization_factor < std::numeric_limits<T>::epsilon()) {
              normalization_factor = static_cast<T>(1.0); // Avoid division by zero
         }
 
         cqt_kernels_[k].resize(spectrum_len);
         for (size_t i = 0; i < spectrum_len; ++i) {
             // Store conjugated and normalized kernel frequency response
             cqt_kernels_[k][i] = std::conj(temp_fft_output[i]) / normalization_factor;
         }
     }
     // Temporary FFT plan goes out of scope and is destroyed automatically by unique_ptr
 }
 
 /**
  * @brief Destructor for the CQTPlan class.
  */
 template <typename T>
 CQTPlan<T>::~CQTPlan() = default; // unique_ptr handles fft_plan_ cleanup
 
 // --- Rule of Five: Move Semantics ---
 /** @brief Move constructor definition. */
 template <typename T>
 CQTPlan<T>::CQTPlan(CQTPlan&& other) noexcept = default; // Default move is sufficient
 /** @brief Move assignment definition. */
 template <typename T>
 CQTPlan<T>& CQTPlan<T>::operator=(CQTPlan<T>&& other) noexcept = default; // Default move is sufficient
 
 
 /**
  * @brief Executes the Constant Q Transform using the efficient frequency-domain method.
  *
  * Performs a single long FFT on the input signal, then multiplies the spectrum
  * by precomputed kernels and sums the results to obtain CQT coefficients.
  *
  * @tparam T The floating-point type of the input data (float or double).
  * @param input The input signal vector (real-valued).
  * @param output A vector of complex numbers where each element represents the CQT coefficient
  * for a specific frequency bin. The vector will be resized to the number of bins.
  * @throws std::runtime_error If FFT execution or subsequent processing fails, or if the plan is invalid.
  */
 template <typename T>
 void CQTPlan<T>::execute(const std::vector<T>& input, std::vector<std::complex<T>>& output) const {
     if (!fft_plan_) {
         throw std::runtime_error("CQTPlan is not valid (likely moved from or init failed).");
     }
     if (input.empty()) {
         output.assign(num_bins_, {0.0, 0.0}); // Return vector of zeros if input is empty
         return;
     }
 
     output.resize(num_bins_); // Ensure output vector has correct size
 
     // 1. Pad or truncate input signal to match internal FFT length
     std::vector<T> processed_input = input; // Make a mutable copy
     if (processed_input.size() < fft_len_) {
          processed_input.resize(fft_len_, 0.0);  // Pad with zeros
     } else if (processed_input.size() > fft_len_) {
          processed_input.resize(fft_len_); // Truncate
          // Consider adding a warning or optional error for truncation
     }
 
     // 2. Compute the single large FFT using the member plan (RFFT)
     std::vector<std::complex<T>> full_spectrum(fft_plan_->getComplexLength()); // Size N/2 + 1
     try {
         fft_plan_->execute_rfft(processed_input.data(), full_spectrum.data());
     } catch (const std::exception& e) {
          throw std::runtime_error("Failed to execute main FFT in CQT: " + std::string(e.what()));
     }
 
     // 3. Compute CQT coefficients via kernel correlation (spectrum * conj(kernel_spectrum))
     // The overall scaling factor applied here might need adjustment depending
     // on the exact kernel normalization used during construction.
     // Currently, kernels are normalized by 1/window_length.
     // If fft_plan uses BACKWARD norm (forward unscaled), this direct sum might be correct.
     T output_scale_factor = static_cast<T>(1.0); // Usually 1.0 or maybe 1.0/fft_len_
 
     for (size_t k = 0; k < num_bins_; ++k) {
         // Ensure kernel size matches spectrum size (defensive check)
         if (cqt_kernels_[k].size() != full_spectrum.size()) {
              throw std::runtime_error("Internal error: Mismatch between spectrum size ("
                  + std::to_string(full_spectrum.size()) + ") and kernel size ("
                  + std::to_string(cqt_kernels_[k].size()) + ") for bin " + std::to_string(k));
         }
 
         // Perform complex dot product: sum(full_spectrum[i] * cqt_kernels_[k][i])
         // Since kernels were stored pre-conjugated, this is the correlation.
         std::complex<T> cqt_coefficient = {0.0, 0.0};
         for (size_t i = 0; i < full_spectrum.size(); ++i) {
             cqt_coefficient += full_spectrum[i] * cqt_kernels_[k][i];
         }
         output[k] = cqt_coefficient * output_scale_factor; // Apply final scaling
     }
 }
 
 
 // --- Explicit Template Instantiations ---
 // Ensures the class and its methods are compiled for float and double.
 template class CQTPlan<float>;
 template class CQTPlan<double>;
 
 } // namespace OmniDSP