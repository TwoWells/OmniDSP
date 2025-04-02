#ifndef OMNIDSP_CQT_H
#define OMNIDSP_CQT_H

#include <OmniDSP/omnidsp.h> // Include OmniDSP header for FFTPlan
#include <OmniDSP/windows.h> // Include Window header
#include <vector>
#include <complex>
#include <functional> // For std::function
#include <memory>     // For std::unique_ptr
#include <cstddef>    // For size_t

namespace OmniDSP {

/**
 * @brief Provides functionality for performing the Constant Q Transform (CQT)
 * using an efficient frequency-domain approach.
 *
 * The CQT is a transform similar to the Fourier Transform, but with a
 * frequency resolution that varies logarithmically. This implementation uses
 * a single long FFT and precomputed frequency-domain kernels.
 *
 * @tparam T The floating-point type of the input data (float or double).
 */
template <typename T>
class CQTPlan {
    // Ensure T is either float or double at compile time
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "CQTPlan only supports float or double precision.");
public:
    /**
     * @brief Constructor for the CQTPlan class (efficient version).
     *
     * Initializes the CQT transform by creating a single FFT plan and
     * precomputing frequency-domain kernels for each CQT bin.
     *
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
    CQTPlan(double sample_rate, double lowest_freq, double highest_freq, int bins_per_octave,
            std::function<std::vector<T>(const std::vector<T>&)> window_function);

    /**
     * @brief Destructor for the CQTPlan class.
     *
     * Releases resources associated with the CQT plan, including the FFT plan.
     */
    ~CQTPlan();

    // --- Rule of Five: Move-Only ---
    /** @brief Deleted copy constructor. CQTPlan is not copyable. */
    CQTPlan(const CQTPlan&) = delete;
    /** @brief Deleted copy assignment operator. CQTPlan is not copyable. */
    CQTPlan& operator=(const CQTPlan&) = delete;
    /** @brief Move constructor. Transfers ownership of the plan resources. */
    CQTPlan(CQTPlan&&) noexcept;
    /** @brief Move assignment operator. Transfers ownership of the plan resources. */
    CQTPlan& operator=(CQTPlan&&) noexcept;


    /**
     * @brief Executes the Constant Q Transform using the efficient frequency-domain method.
     *
     * Performs a single long FFT on the input signal, then multiplies the spectrum
     * by precomputed kernels and sums the results to obtain CQT coefficients.
     *
     * @param input The input signal vector (real-valued).
     * @param output A vector of complex numbers where each element represents the CQT coefficient
     * for a specific frequency bin. The vector will be resized to the number of bins.
     * @throws std::runtime_error If FFT execution or subsequent processing fails.
     */
    void execute(const std::vector<T>& input, std::vector<std::complex<T>>& output) const;

    // --- Getters ---

    /** @brief Gets the number of CQT frequency bins. */
    size_t getNumBins() const { return num_bins_; }

    /** @brief Gets the sample rate used for this plan (Hz). */
    double getSampleRate() const { return sample_rate_; }

    /** @brief Gets the lowest frequency configured for this plan (Hz). */
    double getLowestFrequency() const { return lowest_freq_; }

    /** @brief Gets the highest frequency configured for this plan (Hz). */
    double getHighestFrequency() const { return highest_freq_; }

    /** @brief Gets the number of bins per octave configured for this plan. */
    int getBinsPerOctave() const { return bins_per_octave_; }

    /** @brief Gets the length of the internal FFT used by the plan. */
    size_t getFFTLength() const { return fft_len_; }


private:
    // --- Configuration & Parameters ---
    double sample_rate_;
    double lowest_freq_;
    double highest_freq_;
    int bins_per_octave_;
    size_t num_bins_;
    std::function<std::vector<T>(const std::vector<T>&)> window_function_;

    // --- Internal Resources for Efficient Method ---
    size_t fft_len_;                             // Length of the single long FFT
    std::unique_ptr<FFTPlan<T>> fft_plan_;       // Single plan for the long FFT (likely REAL domain)
    std::vector<std::vector<std::complex<T>>> cqt_kernels_; // Precomputed frequency-domain kernels
};

// --- Explicit Template Instantiations (Declarations) ---
// Ensures the class is instantiated for float and double in the implementation file.
/** @cond OMNIDSP_INTERNAL */
extern template class CQTPlan<float>;
extern template class CQTPlan<double>;
/** @endcond */


} // namespace OmniDSP

#endif // OMNIDSP_CQT_H