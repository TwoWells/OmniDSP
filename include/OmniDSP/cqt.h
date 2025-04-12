/**
 * @file cqt.h
 * @brief Public API header for the CQTPlan class (Recursive Implementation).
 * Precomputes sparse kernels per octave in the constructor.
 * @version 2.2.1 // Version bump for friend declaration
 * @date 2025-04-12 // Or current date - updated to reflect changes
 */

#ifndef OMNIDSP_CQT_H
#define OMNIDSP_CQT_H

#include <OmniDSP/omnidsp.h> // For FFTPlan
#include <vector>
#include <complex>
#include <functional>
#include <memory>      // For unique_ptr
#include <cstddef>     // For size_t
#include <type_traits> // For std::is_same_v
#include <map>         // For sparse kernel storage

// Forward declaration of the test fixture class
// (Alternatively, include the test header if structure allows, but forward declaration is safer)
class PrecomputedRecursiveCQTTest;

namespace OmniDSP
{

    template <typename T>
    class CQTPlan
    {
        // Ensure T is either float or double at compile time
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "CQTPlan only supports float or double precision.");

        // Default anti-aliasing filter order (must be odd)
        static constexpr int DEFAULT_RECURSIVE_FIR_ORDER = 101;

        // Grant access to the test fixture
        friend class ::PrecomputedRecursiveCQTTest; // Use global scope :: if test class is not in OmniDSP namespace

    public:
        // Type alias for the window function signature expected by CQTPlan
        // It receives the required length and should return a vector of coefficients.
        using WindowFuncType = std::function<std::vector<T>(size_t required_length)>;
        // Type alias for sparse kernel representation for one bin (FreqIndex -> ComplexValue)
        using SparseKernelBin = std::map<size_t, std::complex<T>>;
        // Type alias for sparse kernels for one octave (vector of bins)
        using SparseKernelOctave = std::vector<SparseKernelBin>;

        /**
         * @brief Constructor for the CQTPlan class (recursive, precomputed kernels).
         *
         * Initializes CQT parameters and precomputes necessary FFT plans and
         * sparse spectral kernels for each octave. This involves significant setup time.
         *
         * @param sample_rate Initial sample rate in Hz.
         * @param hop_length Hop size in samples (must be divisible by 2^(num_octaves-1)).
         * @param lowest_freq Lowest frequency in Hz (> 0).
         * @param highest_freq Highest frequency in Hz (<= sample_rate / 2).
         * @param bins_per_octave Number of bins per octave (> 0).
         * @param window_function Function to generate time-domain windows for kernels.
         * @param sparsity_threshold Threshold below which kernel values are treated as zero (e.g., 1e-5).
         * @param fir_filter_order Order (length) of the FIR anti-aliasing filter used in recursion (must be positive and odd). Defaults to DEFAULT_RECURSIVE_FIR_ORDER.
         * @throws std::invalid_argument If parameters are invalid.
         * @throws std::runtime_error If precomputation fails (e.g., FFT plan creation).
         */
        CQTPlan(double sample_rate, size_t hop_length, double lowest_freq, double highest_freq, int bins_per_octave,
                WindowFuncType window_function, T sparsity_threshold = 1e-5,
                int fir_filter_order = DEFAULT_RECURSIVE_FIR_ORDER); // Added parameter with default

        /** @brief Destructor. */
        ~CQTPlan();

        // --- Rule of Five: Move-Only ---
        CQTPlan(const CQTPlan &) = delete;
        CQTPlan &operator=(const CQTPlan &) = delete;
        CQTPlan(CQTPlan &&) noexcept;
        CQTPlan &operator=(CQTPlan &&) noexcept;

        /**
         * @brief Executes the Constant Q Transform using precomputed resources.
         *
         * The output format is a vector of vectors (bins x frames).
         * `output[bin_index][frame_index]` gives the complex CQT coefficient.
         *
         * @param input The input signal vector (real-valued).
         * @param output Resulting CQT coefficients [bin][frame]. Resized internally.
         * @throws std::runtime_error If execution fails.
         */
        void execute(const std::vector<T> &input, std::vector<std::vector<std::complex<T>>> &output) const;

        // --- Getters ---
        size_t getNumBins() const { return num_bins_; }
        double getSampleRate() const { return sample_rate_; } // Initial sample rate
        size_t getHopLength() const { return hop_length_; }   // Initial hop length
        double getLowestFrequency() const { return lowest_freq_; }
        double getHighestFrequency() const { return highest_freq_; }
        int getBinsPerOctave() const { return bins_per_octave_; }
        int getNumOctaves() const { return num_octaves_; }
        T getSparsityThreshold() const { return sparsity_threshold_; }
        int getFirFilterOrder() const { return fir_filter_order_; } // Getter for the filter order
        // Optional: Get FFT lengths per octave if needed for external info
        // const std::vector<size_t>& getOctaveFFTLengths() const { return octave_fft_lens_; }

    private: // Keep helpers private, accessible via friend declaration above
        // --- Configuration & Parameters ---
        double sample_rate_;
        size_t hop_length_;
        double lowest_freq_;
        double highest_freq_;
        int bins_per_octave_;
        size_t num_bins_;
        int num_octaves_;
        WindowFuncType window_function_;
        int fir_filter_order_; // Store the filter order
        T sparsity_threshold_;

        // --- Precomputed Resources (per octave) ---
        std::vector<size_t> octave_fft_lens_;                              // FFT length for each octave
        std::vector<size_t> octave_spectrum_lens_;                         // Spectrum length (N/2+1) for each octave
        std::vector<std::unique_ptr<FFTPlan<T>>> octave_signal_fft_plans_; // RFFT plans for signal frames
        std::vector<std::unique_ptr<FFTPlan<T>>> octave_kernel_fft_plans_; // C2C plans for kernel FFT gen
        std::vector<SparseKernelOctave> precomputed_sparse_kernels_;       // Kernels[octave][bin_in_octave]

        // --- Private Helper Function Declarations ---

        /**
         * @brief Calculates the CQT for a single octave using precomputed FFT plan and sparse kernels.
         * @param signal The input signal for this octave (potentially downsampled).
         * @param current_sample_rate The sample rate corresponding to the input signal.
         * @param current_hop_length The hop length corresponding to the input signal.
         * @param octave_idx The index of the octave to process (used to retrieve precomputed resources).
         * @return CQT coefficients for this octave [bin_in_octave][frame].
         */
        std::vector<std::vector<std::complex<T>>> calculateSingleOctaveCQT(
            const std::vector<T> &signal, double current_sample_rate, size_t current_hop_length,
            int octave_idx) const;

        /**
         * @brief Calculates FIR filter coefficients for anti-aliasing using the windowed-sinc method.
         * @param current_sample_rate The sample rate for which to design the filter.
         * @param N The desired filter order (length, must be odd).
         * @return A vector containing the normalized filter coefficients.
         */
        std::vector<T> calculateFirCoefficients(double current_sample_rate, int N) const;

        /**
         * @brief Applies the anti-aliasing FIR filter and downsamples the signal by a factor of 2.
         * Uses the backend implementation selected at compile time.
         * @param signal The input signal to filter and downsample.
         * @param current_sample_rate The sample rate of the input signal.
         * @return The filtered and downsampled signal.
         */
        std::vector<T> filterAndDownsampleBy2(const std::vector<T> &signal, double current_sample_rate) const;
    };

    // --- Explicit Template Instantiations (Declarations) ---
    // These tell the compiler that the full definitions exist elsewhere (in cqt.cpp)
    /** @cond OMNIDSP_INTERNAL */
    extern template class CQTPlan<float>;
    extern template class CQTPlan<double>;
    /** @endcond */

} // namespace OmniDSP

#endif // OMNIDSP_CQT_H