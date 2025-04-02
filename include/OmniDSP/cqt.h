#ifndef OMNIDSP_CQT_H
#define OMNIDSP_CQT_H

#include <OmniDSP/omnidsp.h> // Include OmniDSP header
#include <OmniDSP/window.h>  // Include Window header
#include <vector>
#include <complex>
#include <functional> // For std::function

namespace OmniDSP {

/**
 * @brief Provides functionality for performing the Constant Q Transform (CQT).
 *
 * The CQT is a transform similar to the Fourier Transform, but with a
 * frequency resolution that varies logarithmically. This class uses
 * the existing FFTPlan class to perform the underlying FFT calculations.
 *
 * @tparam T The floating-point type of the input data (float or double).
 */
template <typename T>
class CQTPlan {
public:
    /**
     * @brief Constructor for the CQTPlan class.
     *
     * Initializes the CQT transform by precomputing FFT plans and window functions.
     *
     * @param sample_rate The sample rate of the input signal in Hz.
     * @param lowest_freq The lowest frequency of interest for the CQT in Hz.
     * @param highest_freq The highest frequency of interest for the CQT in Hz.
     * @param bins_per_octave The number of frequency bins per octave.
     * @param window_function A function that takes a vector of T and returns a windowed vector of T.
     * @throws std::invalid_argument If sample_rate, lowest_freq, or bins_per_octave are invalid.
     */
    CQTPlan(double sample_rate, double lowest_freq, double highest_freq, int bins_per_octave,
            std::function<std::vector<T>(const std::vector<T>&)> window_function);

    /**
     * @brief Destructor for the CQTPlan class.
     *
     * Releases any resources associated with the CQT plan, including the precomputed FFT plans.
     */
    ~CQTPlan();

    /**
     * @brief Executes the Constant Q Transform.
     *
     * Performs the CQT on the input signal and stores the result in the output.
     *
     * @param input The input signal vector.
     * @param output A vector of complex vector. Each inner vector represents the CQT result for a specific frequency bin.
     * The outer vector has a size equal to the number of frequency bins.
     * @throws std::runtime_error If FFT execution fails.
     */
    void execute(const std::vector<T>& input, std::vector<std::vector<std::complex<T>>>& output) const;

    // ... other methods ...

private:
    // Precomputed FFT plans
    std::vector<std::unique_ptr<FFTPlan<T>>> fft_plans_;

    // Precomputed window functions
    std::vector<std::vector<T>> windows_;

    // CQT parameters
    double sample_rate_;
    double lowest_freq_;
    double highest_freq_;
    int bins_per_octave_;
    size_t num_bins_;

    // Window function
    std::function<std::vector<T>(const std::vector<T>&)> window_function_;

    // ... other data members ...
};

} // namespace OmniDSP

#endif // OMNIDSP_CQT_H