#include <OmniDSP/cqt.h> // Use the updated header
#include <OmniDSP/omnidsp.h>
#include <cmath>
#include <vector>
#include <complex>
#include <stdexcept>
#include <numeric>   // For std::accumulate
#include <limits>    // For std::numeric_limits
#include <algorithm> // For std::max, std::copy
#include <string>    // For exception messages

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


namespace OmniDSP {

/**
 * @brief Constructor for the CQTPlan class (efficient version).
 *
 * Initializes the CQT transform by creating a single FFT plan and
 * precomputing frequency-domain kernels for each CQT bin.
 *
 * @tparam T The floating-point type of the input data (float or double).
 * @param sample_rate The sample rate of the input signal in Hz.
 * @param lowest_freq The lowest frequency of interest for the CQT in Hz. Must be > 0.
 * @param highest_freq The highest frequency of interest for the CQT in Hz. Must be <= sample_rate / 2.
 * @param bins_per_octave The number of frequency bins per octave. Must be > 0.
 * @param window_function A function that takes a vector<T> (representing a window of size 1)
 * and returns the window coefficients as a vector<T>. This function
 * will be called internally with appropriate lengths for each CQT bin's
 * time-domain window during kernel generation.
 * @throws std::invalid_argument If sample_rate, lowest_freq, highest_freq, or bins_per_octave are invalid.
 * @throws std::runtime_error If FFT plan creation or kernel generation fails.
 */
template <typename T>
CQTPlan<T>::CQTPlan(double sample_rate, double lowest_freq, double highest_freq, int bins_per_octave,
                    std::function<std::vector<T>(const std::vector<T>&)> window_function)
    : sample_rate_(sample_rate), lowest_freq_(lowest_freq), highest_freq_(highest_freq),
      bins_per_octave_(bins_per_octave), window_function_(window_function)
{
    // --- Parameter Validation ---
    if (sample_rate <= 0.0) {
        throw std::invalid_argument("Sample rate must be positive.");
    }
    if (lowest_freq <= 0.0) {
         throw std::invalid_argument("Lowest frequency must be positive.");
    }
     if (highest_freq <= lowest_freq) {
         throw std::invalid_argument("Highest frequency must be greater than lowest frequency.");
     }
    if (highest_freq > sample_rate / 2.0) {
         throw std::invalid_argument("Highest frequency cannot exceed Nyquist frequency (sample_rate / 2).");
    }
    if (bins_per_octave <= 0) {
         throw std::invalid_argument("Bins per octave must be positive.");
    }
    if (!window_function_) {
         throw std::invalid_argument("A valid window function must be provided.");
    }

    // --- Calculate Bin Count ---
    num_bins_ = static_cast<size_t>(std::ceil(bins_per_octave_ * std::log2(highest_freq_ / lowest_freq_)));
     if (num_bins_ == 0) {
         throw std::invalid_argument("Calculated number of CQT bins is zero. Check frequency range and bins per octave.");
     }

    // --- Determine Required FFT Length ---
    size_t max_window_length = 0;
    for (size_t k = 0; k < num_bins_; ++k) {
        double freq = lowest_freq_ * std::pow(2.0, static_cast<double>(k) / bins_per_octave_);
        // Ensure frequency is positive before division
         if (freq <= std::numeric_limits<double>::epsilon()) {
            // This might happen if lowest_freq is extremely small.
            // Skip this bin or throw an error. Throwing is safer.
             throw std::runtime_error("Calculated frequency for bin " + std::to_string(k) + " is too close to zero.");
        }
        double Q = 1.0 / (std::pow(2.0, 1.0 / bins_per_octave_) - 1.0);
        // Calculate window length, ensure it's at least 1
        size_t window_length = static_cast<size_t>(std::max(1.0, std::round(Q * sample_rate_ / freq)));
        max_window_length = std::max(max_window_length, window_length);
    }

    // Choose FFT length (e.g., next power of 2 >= max_window_length)
    fft_len_ = 1;
    while (fft_len_ < max_window_length) {
        fft_len_ *= 2;
    }
     // Also ensure FFT length is sufficient for reasonable resolution if input length is known/large
     // fft_len_ = std::max(fft_len_, some_reasonable_minimum_fft_length_or_input_length);


    // --- Create the Single Main FFT Plan ---
    // Use REAL domain for efficiency with real input signals
    Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
    try {
        fft_plan_ = std::make_unique<FFTPlan<T>>(fft_len_, prec, Direction::FORWARD, Domain::REAL, NormMode::BACKWARD);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create main FFTPlan for CQT: " + std::string(e.what()));
    }
    const size_t spectrum_len = fft_plan_->getComplexLength(); // N/2 + 1 for REAL plan

    // --- Precompute Frequency-Domain Kernels ---
    cqt_kernels_.resize(num_bins_);
    std::vector<std::complex<T>> temp_fft_output(fft_len_); // Buffer for temporary C2C FFT

    // Create a temporary C2C FFT plan for kernel generation
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

        // a. Generate time-domain window for this bin
        std::vector<T> time_window_input(window_length, 1.0); // Vector of ones to get window coeffs
        std::vector<T> time_window = window_function_(time_window_input);
        if (time_window.size() != window_length) {
             throw std::runtime_error("Window function returned incorrect size for bin " + std::to_string(k));
        }

        // b. Pad & Center the time-domain window
        std::vector<T> padded_time_window(fft_len_, 0.0);
        // Calculate start index to center the window (handle odd/even lengths)
        long long start_idx_ll = static_cast<long long>(fft_len_ / 2) - static_cast<long long>(window_length / 2);
        size_t start_idx = (start_idx_ll < 0) ? 0 : static_cast<size_t>(start_idx_ll);
        for (size_t n = 0; n < window_length; ++n) {
            if (start_idx + n < fft_len_) { // Check bounds
                padded_time_window[start_idx + n] = time_window[n];
            }
        }

        // d. Apply frequency shift in time domain (multiply by complex exponential)
        std::vector<std::complex<T>> complex_shifted_window(fft_len_);
        double shift_phase_increment = -2.0 * M_PI * freq_k / sample_rate_; // Negative for forward shift convention often used
        for (size_t n = 0; n < fft_len_; ++n) {
            double phase = shift_phase_increment * static_cast<double>(n);
            // Use padded_time_window value
            complex_shifted_window[n] = padded_time_window[n] * std::complex<T>(std::cos(phase), std::sin(phase));
        }

        // c. Compute FFT of the shifted, padded window using the temporary C2C plan
         try {
            temp_kernel_fft_plan->execute(complex_shifted_window.data(), temp_fft_output.data());
         } catch (const std::exception& e) {
            throw std::runtime_error("Failed execute temporary FFT for kernel bin " + std::to_string(k) + ": " + std::string(e.what()));
         }


        // e. Normalize the kernel (using L1 norm here, L2 might be better for energy)
        // Normalization is crucial for correct amplitude/energy representation.
        // Dividing by window_length here is one common approach, related to Parseval's theorem
        // and how the sum correlates to the original time-domain windowed energy.
        T normalization_factor = static_cast<T>(window_length); // Or calculate L1/L2 norm of kernel
         if (normalization_factor < std::numeric_limits<T>::epsilon()) {
             normalization_factor = static_cast<T>(1.0); // Avoid division by zero
         }

        // f. Store the relevant part of the kernel (first N/2 + 1 points for REAL FFT correlation)
        cqt_kernels_[k].resize(spectrum_len);
        for (size_t i = 0; i < spectrum_len; ++i) {
            // Conjugate the kernel here because we are correlating X(f) with K(f)^*
            // where K(f) is the FFT of the time-domain kernel k(t).
            // Also apply normalization.
            cqt_kernels_[k][i] = std::conj(temp_fft_output[i]) / normalization_factor;
        }
    }
    // Temporary FFT plan goes out of scope and is destroyed
}

/**
 * @brief Destructor for the CQTPlan class.
 */
template <typename T>
CQTPlan<T>::~CQTPlan() = default; // unique_ptr handles fft_plan_ cleanup

// --- Rule of Five: Move Semantics ---
/** @brief Move constructor definition. */
template <typename T>
CQTPlan<T>::CQTPlan(CQTPlan&& other) noexcept = default;
/** @brief Move assignment definition. */
template <typename T>
CQTPlan<T>& CQTPlan<T>::operator=(CQTPlan<T>&& other) noexcept = default;


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
 * @throws std::runtime_error If FFT execution or subsequent processing fails.
 */
template <typename T>
void CQTPlan<T>::execute(const std::vector<T>& input, std::vector<std::complex<T>>& output) const {
    if (!fft_plan_) {
        throw std::runtime_error("CQTPlan is not valid (likely moved from or init failed).");
    }
     if (input.empty()) {
        output.clear();
        output.resize(num_bins_, {0.0, 0.0}); // Return vector of zeros
        return;
     }

    output.resize(num_bins_); // Ensure output vector has correct size

    // 1. Pad input signal
    // Create padded input vector - consider performance for very large inputs
    std::vector<T> padded_input = input; // Copy
    if (padded_input.size() < fft_len_) {
         padded_input.resize(fft_len_, 0.0);  // Pad with zeros
    } else if (padded_input.size() > fft_len_) {
         // Option 1: Truncate (may lose information)
         padded_input.resize(fft_len_);
         // Option 2: Throw error (safer if unexpected)
         // throw std::runtime_error("Input signal size (" + std::to_string(input.size()) + ") exceeds internal FFT length (" + std::to_string(fft_len_) + ").");
    }


    // 2. Compute the single large FFT using the member plan
    std::vector<std::complex<T>> full_spectrum(fft_plan_->getComplexLength());
    try {
        // Assuming fft_plan_ was created with Domain::REAL
        fft_plan_->execute_rfft(padded_input.data(), full_spectrum.data());
    } catch (const std::exception& e) {
         throw std::runtime_error("Failed to execute main FFT in CQT: " + std::string(e.what()));
    }


    // 3. Compute CQT coefficients via kernel multiplication & sum
    // The final scaling depends heavily on the kernel normalization used.
    // If kernels were normalized appropriately (e.g., by window_length or L2 norm),
    // this direct sum might be correct. Otherwise, a factor like 1/fft_len_ might be needed.
    T output_scale_factor = static_cast<T>(1.0); // Adjust if necessary based on kernel normalization

    for (size_t k = 0; k < num_bins_; ++k) {
        std::complex<T> cqt_coefficient = {0.0, 0.0};
        // Ensure kernel size matches spectrum size
        if (cqt_kernels_[k].size() != full_spectrum.size()) {
             throw std::runtime_error("Internal error: Mismatch between spectrum size ("
                 + std::to_string(full_spectrum.size()) + ") and kernel size ("
                 + std::to_string(cqt_kernels_[k].size()) + ") for bin " + std::to_string(k));
        }
        // Multiply spectrum by kernel and sum
        for (size_t i = 0; i < full_spectrum.size(); ++i) {
            // Correlation involves multiplying spectrum X(f) by K(f)^*
            // Kernels were stored pre-conjugated.
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