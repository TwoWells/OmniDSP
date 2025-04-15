/**
 * @file filter.h
 * @brief Public API header for filter design functions in OmniDSP.
 *
 * This header declares functions for designing various digital filters (FIR, IIR).
 * Filter *application* is typically done using OmniDSP::correlate1d (for FIR)
 * or potentially dedicated IIR filtering functions if added later.
 *
 * @version 1.0.0 (Placeholder)
 * @date 2025-04-15
 */

 #ifndef OMNIDSP_FILTER_H
 #define OMNIDSP_FILTER_H
 
 #include <vector>
 #include <string>    // For window type string etc.
 #include <stdexcept> // For invalid_argument
 
 // Include the generated export header for DLL symbol handling if needed for future classes/funcs
 // #include <OmniDSP/omnidsp_export.h>
 
 namespace OmniDSP
 {
     // --- Filter Design Functions (Placeholders) ---
     // These are examples and are not yet implemented.
 
     /**
      * @brief Designs a lowpass FIR filter using the window method. (Placeholder)
      *
      * @tparam T float or double.
      * @param num_taps The number of filter coefficients (filter order + 1). Should be odd for Type I linear phase.
      * @param cutoff_freq The normalized cutoff frequency (0 to 0.5, where 0.5 is Nyquist).
      * @param window_type The type of window to use (e.g., "hann", "hamming", "kaiser").
      * @param window_param Optional parameter for the window (e.g., beta for Kaiser).
      * @return std::vector<T> The calculated filter coefficients.
      * @throws std::invalid_argument If parameters are invalid.
      * @throws std::runtime_error If design fails.
      */
     template <typename T>
     std::vector<T> designLowpassFIR(int num_taps, double cutoff_freq,
                                     const std::string& window_type = "hann",
                                     T window_param = T{}); // Default param might vary
 
     /**
      * @brief Designs a Butterworth IIR filter. (Placeholder)
      * Returns coefficients in Second-Order Sections (SOS) format for stability.
      *
      * @tparam T float or double.
      * @param order The filter order.
      * @param cutoff_freqs Normalized cutoff frequency (or [low, high] for bandpass/stop).
      * @param filter_type Type: "lowpass", "highpass", "bandpass", "bandstop".
      * @return std::vector<std::vector<T>> SOS matrix [[b0, b1, b2, a0, a1, a2], ...]. a0 is typically 1.
      * @throws std::invalid_argument If parameters are invalid.
      * @throws std::runtime_error If design fails.
      */
     template <typename T>
     std::vector<std::vector<T>> designButterworth(int order, const std::vector<double>& cutoff_freqs,
                                                  const std::string& filter_type = "lowpass");
 
 
     // Add declarations for other filter design functions (e.g., Chebyshev, Elliptic, other FIR methods) here.
 
     // Note: Explicit template instantiations would go in a future filter.cpp
 
 } // namespace OmniDSP
 
 #endif // OMNIDSP_FILTER_H
 