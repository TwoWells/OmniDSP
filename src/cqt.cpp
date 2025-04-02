#include "OmniDSP/cqt.h"
#include <cmath>

namespace OmniDSP {

/**
 * @brief Constructor for the CQTPlan class.
 *
 * Initializes the CQT transform by precomputing FFT plans and window functions.
 *
 * @tparam T The floating-point type of the input data (float or double).
 * @param sample_rate The sample rate of the input signal in Hz.
 * @param lowest_freq The lowest frequency of interest for the CQT in Hz.
 * @param highest_freq The highest frequency of interest for the CQT in Hz.
 * @param bins_per_octave The number of frequency bins per octave.
 * @param window_function A function that takes a vector of T and returns a windowed vector of T.
 * @throws std::invalid_argument If sample_rate, lowest_freq, or bins_per_octave are invalid.
 */
template <typename T>
CQTPlan<T>::CQTPlan(double sample_rate, double lowest_freq, double highest_freq, int bins_per_octave,
                    std::function<std::vector<T>(const std::vector<T>&)> window_function)
    : sample_rate_(sample_rate), lowest_freq_(lowest_freq), highest_freq_(highest_freq),
      bins_per_octave_(bins_per_octave), window_function_(window_function) {
    if (sample_rate <= 0.0 || lowest_freq <= 0.0 || highest_freq <= 0.0 || bins_per_octave <= 0) {
        throw std::invalid_argument("Invalid CQT parameters.");
    }

    // 1. Calculate the number of bins
    num_bins_ = static_cast<size_t>(bins_per_octave_ * std::ceil(std::log2(highest_freq_ / lowest_freq_)));

    // 2. Precompute FFT plans and window functions
    fft_plans_.resize(num_bins_);
    windows_.resize(num_bins_);

    for (size_t k = 0; k < num_bins_; ++k) {
        // Calculate frequency for this bin
        double freq = lowest_freq_ * std::pow(2.0, static_cast<double>(k) / bins_per_octave_);

        // Calculate window length (dependent on frequency and Q factor)
        double Q = 1.0 / (std::pow(2.0, 1.0 / bins_per_octave_) - 1.0);
        size_t window_length = static_cast<size_t>(Q * sample_rate_ / freq);

        // Generate window function (e.g., Hann window)
        std::vector<T> window_input(window_length, 1.0); // Create a vector of ones for windowing
        windows_[k] = window_function(window_input);     // Apply the window function
        
        // Create FFT plan
        Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
        fft_plans_[k] = std::make_unique<FFTPlan<T>>(window_length, prec, Direction::FORWARD, Domain::COMPLEX);
    }
}

/**
 * @brief Destructor for the CQTPlan class.
 *
 * Releases any resources associated with the CQT plan, including the precomputed FFT plans.
 */
template <typename T>
CQTPlan<T>::~CQTPlan() = default;

/**
 * @brief Executes the Constant Q Transform.
 *
 * Performs the CQT on the input signal and stores the result in the output.
 *
 * @tparam T The floating-point type of the input data (float or double).
 * @param input The input signal vector.
 * @param output A vector of complex vector. Each inner vector represents the CQT result for a specific frequency bin.
 * The outer vector has a size equal to the number of frequency bins.
 * @throws std::runtime_error If FFT execution fails.
 */
template <typename T>
void CQTPlan<T>::execute(const std::vector<T>& input, std::vector<std::vector<std::complex<T>>>& output) const {
    output.resize(num_bins_);
    for (size_t k = 0; k < num_bins_; ++k) {
        size_t window_length = windows_[k].size();
        output[k].resize(window_length);

        // 1. Window the input signal
        std::vector<std::complex<T>> windowed_input(window_length);
        for (size_t n = 0; n < window_length; ++n) {
            size_t input_index = 0; // Initialize to 0

            // CQT often centers the window around the time
            // corresponding to the output sample.
            // For simplicity, let's assume we're taking the CQT
            // at time t=0 (beginning of the signal).

            // If we wanted to take the CQT at time t, we would calculate:
            // input_index = static_cast<size_t>(t * sample_rate_ - window_length / 2);

            // In this simplified case, we'll just grab the beginning of the signal:
            input_index = 0;

            // Ensure input_index is within bounds
            if (n < window_length && input_index + n < input.size()) {
                windowed_input[n] = input[input_index + n] * windows_[k][n];
            } else {
                windowed_input[n] = 0.0; // Or handle boundary conditions as needed
            }
        }

        // 2. Perform FFT using precomputed plan
        fft_plans_[k]->execute(windowed_input.data(), output[k].data());
    }
}

// ... other methods ...

// Explicit Instantiations
template class CQTPlan<float>;
template class CQTPlan<double>;

} // namespace OmniDSP