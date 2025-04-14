/**
 * @file cqt.h
 * @brief Public API header for the CQTPlan class (Recursive Implementation).
 * Precomputes sparse kernels per octave in the constructor. Includes export macros.
 * @version 2.2.2 // Version bump for getter definition separation
 * @date 2025-04-13 // Or current date - updated to reflect changes
 */

#ifndef OMNIDSP_CQT_H
#define OMNIDSP_CQT_H

// --- Include the generated export header ---
// Defines OMNIDSP_EXPORT for DLL symbol handling
#include <OmniDSP/omnidsp_export.h> // Adjust path/name if EXPORT_FILE_NAME was used in CMake

#include <OmniDSP/omnidsp.h> // For FFTPlan
#include <vector>
#include <complex>
#include <functional>
#include <memory>      // For unique_ptr
#include <cstddef>     // For size_t
#include <type_traits> // For std::is_same_v
#include <map>         // For sparse kernel storage

// Forward declaration of the test fixture class
// (Allows friend declaration without including the test header here)
class PrecomputedRecursiveCQTTest;

namespace OmniDSP
{
    /**
     * @brief Manages precomputation and execution for the Constant Q Transform (Recursive Method).
     *
     * This class implements the CQT using a recursive, multi-resolution approach.
     * It precomputes sparse spectral kernels during construction for efficient execution.
     * Marked with OMNIDSP_EXPORT for DLL usage.
     *
     * @tparam T The floating-point type (float or double).
     */
    template <typename T>
    class OMNIDSP_EXPORT CQTPlan // <--- OMNIDSP_EXPORT macro added here
    {
        // Ensure T is either float or double at compile time
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "CQTPlan only supports float or double precision.");

        // Default anti-aliasing filter order (must be odd)
        static constexpr int DEFAULT_RECURSIVE_FIR_ORDER = 101;

        // Grant access to the test fixture for testing private helpers
        friend class ::PrecomputedRecursiveCQTTest; // Use global scope ::

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
                int fir_filter_order = DEFAULT_RECURSIVE_FIR_ORDER);

        /** @brief Destructor. Releases internal resources. */
        ~CQTPlan();

        // --- Rule of Five: Move-Only ---
        CQTPlan(const CQTPlan &) = delete;            // Disable copy constructor
        CQTPlan &operator=(const CQTPlan &) = delete; // Disable copy assignment
        CQTPlan(CQTPlan &&) noexcept;                 // Enable move constructor
        CQTPlan &operator=(CQTPlan &&) noexcept;      // Enable move assignment

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

        // --- Getters (Declarations ONLY) ---
        // Definitions moved to cqt.cpp to ensure proper DLL export/import
        /** @brief Gets the total number of CQT frequency bins calculated by the plan. */
        size_t getNumBins() const;
        /** @brief Gets the initial sample rate (in Hz) the plan was configured with. */
        double getSampleRate() const;
        /** @brief Gets the hop length (in samples) between CQT frames. */
        size_t getHopLength() const;
        /** @brief Gets the lowest frequency (in Hz) represented by the CQT bins. */
        double getLowestFrequency() const;
        /** @brief Gets the highest frequency (in Hz) the CQT attempts to cover. */
        double getHighestFrequency() const;
        /** @brief Gets the number of CQT bins per octave. */
        int getBinsPerOctave() const;
        /** @brief Gets the total number of octaves covered by the CQT plan. */
        int getNumOctaves() const;
        /** @brief Gets the threshold used for sparsifying the CQT kernels during precomputation. */
        T getSparsityThreshold() const;
        /** @brief Gets the order (length) of the FIR anti-aliasing filter used. */
        int getFirFilterOrder() const;

    private: // Keep implementation details private
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
    /** @cond OMNIDSP_INTERNAL */                         // Hide from Doxygen index
    extern template class OMNIDSP_EXPORT CQTPlan<float>;  // Add export macro here too
    extern template class OMNIDSP_EXPORT CQTPlan<double>; // Add export macro here too
    /** @endcond */

} // namespace OmniDSP

#endif // OMNIDSP_CQT_H