/**
 * @file windows.cpp
 * @brief Example usage of OmniDSP window functions.
 *
 * Demonstrates applying Hann, Hamming, Kaiser, and Flat-top windows.
 */

#include <cmath>          // Included for M_PI via window.h, good to be explicit
#include <iostream>
#include <vector>
#include <numeric>        // For std::iota if needed
#include <stdexcept>      // For catching potential errors
#include <OmniDSP/omnidsp.h>
#include <OmniDSP/windows.h>

int main() {
    using Real = double;

    std::cout << "OmniDSP Window Function Example" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    // --- Prepare Input Data ---
    std::vector<Real> signal_to_window = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0}; // Example signal
    size_t window_len = signal_to_window.size();

    std::cout << "Original Signal: ";
    for (size_t i=0; i<window_len; ++i) std::cout << signal_to_window[i] << (i == window_len-1 ? "" : " ");
    std::cout << std::endl;

    try {
        // ==========================================================
        // Apply Window Functions
        // ==========================================================
        std::cout << "\n--- Applying Window Functions ---" << std::endl;

        // Apply Hann window
        auto hann_window_func = [](const std::vector<Real>& win_in) { return OmniDSP::Window::hann(win_in); };
        std::vector<Real> hann_windowed_signal = hann_window_func(signal_to_window); // Use helper directly
        std::cout << "Hann Windowed Signal:      ";
        for (size_t i=0; i<window_len; ++i) std::cout << hann_windowed_signal[i] << (i == window_len-1 ? "" : " ");
        std::cout << std::endl;

        // Apply Hamming window
        std::vector<Real> hamming_windowed_signal = OmniDSP::Window::hamming(signal_to_window);
        std::cout << "Hamming Windowed Signal:   ";
         for (size_t i=0; i<window_len; ++i) std::cout << hamming_windowed_signal[i] << (i == window_len-1 ? "" : " ");
        std::cout << std::endl;

        // Apply Kaiser window
        Real kaiser_beta = 8.6; // Example beta value
        std::vector<Real> kaiser_windowed_signal = OmniDSP::Window::kaiser(signal_to_window, kaiser_beta);
        std::cout << "Kaiser Windowed Signal:    ";
         for (size_t i=0; i<window_len; ++i) std::cout << kaiser_windowed_signal[i] << (i == window_len-1 ? "" : " ");
        std::cout << std::endl;

        // Apply Flat-top window
        std::vector<Real> flattop_windowed_signal = OmniDSP::Window::flattop(signal_to_window);
        std::cout << "Flat-top Windowed Signal:  ";
         for (size_t i=0; i<window_len; ++i) std::cout << flattop_windowed_signal[i] << (i == window_len-1 ? "" : " ");
        std::cout << std::endl << std::endl;


    } catch (const std::exception& e) {
         // Catch potential errors (e.g., std::invalid_argument if input were empty)
        std::cerr << "\nERROR during Window operation: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "OmniDSP Window Example Finished Successfully." << std::endl;
    return 0;
}