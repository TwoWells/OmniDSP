/**
 * @file main.cpp
 * @brief Example usage of the OmniDSP library.
 *
 * Demonstrates Complex-to-Complex (C2C) transforms using both the
 * FFTPlan class and convenience functions, as well as Real-to-Complex (R2C)
 * and Complex-to-Real (C2R) transforms using convenience functions.
 * Also demonstrates the use of window functions and the CQT.
 */

#include <OmniDSP/omnidsp.h>
#include <OmniDSP/window.h>
#include <OmniDSP/cqt.h>
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <numeric>

int main() {
    // --- Configuration ---
    const size_t N = 16;
    using Real = double;
    using Complex = std::complex<Real>;
    const auto CURRENT_PRECISION = OmniDSP::Precision::DOUBLE;

    std::cout << "OmniDSP Example (N=" << N << ")" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    // --- Prepare Input Data ---
    std::vector<Complex> input_signal_c2c(N);
    std::vector<Real> input_signal_real(N);

    double frequency = 3.0;
    for (size_t i = 0; i < N; ++i) {
        Real real_val = std::cos(2.0 * M_PI * frequency * static_cast<Real>(i) / N);
        Real imag_val = std::sin(1.5 * M_PI * frequency * static_cast<Real>(i) / N);
        input_signal_c2c[i] = Complex(real_val, imag_val);
        input_signal_real[i] = real_val;
    }

    std::cout << "Input Real Signal (for RFFT):" << std::endl;
    for (size_t i = 0; i < N; ++i)
        std::cout << input_signal_real[i] << (i == N - 1 ? "" : ", ");
    std::cout << std::endl << std::endl;

    // --- Execute DSP functions within a try-catch block ---
    try {
        // ==========================================================
        // Method 1: Using FFTPlan for C2C (Out-of-Place)
        // Recommended for repeated transforms of the same configuration.
        // ==========================================================
        std::cout << "--- Method 1: Using FFTPlan (C2C OOP) ---" << std::endl;
        OmniDSP::FFTPlan<Real> forward_plan_c2c(N, CURRENT_PRECISION,
                                                OmniDSP::Direction::FORWARD,
                                                OmniDSP::Domain::COMPLEX,
                                                OmniDSP::NormMode::BACKWARD);
        OmniDSP::FFTPlan<Real> inverse_plan_c2c(N, CURRENT_PRECISION,
                                                OmniDSP::Direction::INVERSE,
                                                OmniDSP::Domain::COMPLEX,
                                                OmniDSP::NormMode::BACKWARD);

        std::vector<Complex> spectrum_c2c_plan(N);
        std::vector<Complex> reconstructed_c2c_plan(N);

        // Forward FFT
        forward_plan_c2c.execute(input_signal_c2c.data(),
                                 spectrum_c2c_plan.data());

        std::cout << "C2C Spectrum (Plan OOP):" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, N); ++i)
            std::cout << spectrum_c2c_plan[i] << " ";
        if (N > 5)
            std::cout << "...";
        std::cout << std::endl;

        // Inverse FFT
        inverse_plan_c2c.execute(spectrum_c2c_plan.data(),
                                 reconstructed_c2c_plan.data());

        std::cout << "\nC2C Reconstructed (Plan OOP):" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, N); ++i)
            std::cout << reconstructed_c2c_plan[i] << " ";
        if (N > 5)
            std::cout << "...";
        std::cout << std::endl;

        // Verification (optional, needs scaling consideration for non-MKL
        // backends/norms)
        Real max_diff_oop = 0.0;
        for (size_t i = 0; i < N; ++i)
            max_diff_oop = std::max(
                max_diff_oop, std::abs(input_signal_c2c[i] - reconstructed_c2c_plan[i]));
        std::cout << "Max Difference (OOP): " << max_diff_oop << std::endl
                  << std::endl;

        // ==========================================================
        // Method 2: Using Convenience Functions (C2C OOP)
        // Simple for one-off transforms, uses NormMode::BACKWARD.
        // ==========================================================
        std::cout << "--- Method 2: Using Convenience Functions (C2C OOP) "
                     "---"
                  << std::endl;
        std::vector<Complex> spectrum_conv_c2c;
        std::vector<Complex> reconstructed_conv_c2c;

        // Convenience functions use the correct constructor internally
        OmniDSP::fft(input_signal_c2c, spectrum_conv_c2c); // Forward
        OmniDSP::ifft(spectrum_conv_c2c,
                      reconstructed_conv_c2c); // Inverse

        std::cout << "C2C Spectrum (Convenience Func):" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, N); ++i)
            std::cout << spectrum_conv_c2c[i] << " ";
        if (N > 5)
            std::cout << "...";
        std::cout << std::endl;

        std::cout << "\nC2C Reconstructed (Convenience Func):" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, N); ++i)
            std::cout << reconstructed_conv_c2c[i] << " ";
        if (N > 5)
            std::cout << "...";
        std::cout << std::endl
                  << std::endl;

        // ==========================================================
        // Method 3: Using In-place Transform (C2C)
        // Uses NormMode::BACKWARD.
        // ==========================================================
        std::cout << "--- Method 3: Using In-place Transform (C2C) ---"
                  << std::endl;
        std::vector<Complex> data_inplace_c2c =
            input_signal_c2c; // Copy input data

        // Use convenience function for simplicity
        OmniDSP::fft_inplace(data_inplace_c2c); // Forward transform in-place

        std::cout << "C2C Spectrum (In-place Forward):" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, N); ++i)
            std::cout << data_inplace_c2c[i] << " ";
        if (N > 5)
            std::cout << "...";
        std::cout << std::endl;

        OmniDSP::ifft_inplace(data_inplace_c2c); // Inverse transform in-place

        std::cout << "\nC2C Reconstructed (In-place Inverse):" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, N); ++i)
            std::cout << data_inplace_c2c[i] << " ";
        if (N > 5)
            std::cout << "...";
        std::cout << std::endl;

        // Verification (optional)
        Real max_diff_ip = 0.0;
        for (size_t i = 0; i < N; ++i)
            max_diff_ip = std::max(
                max_diff_ip, std::abs(input_signal_c2c[i] - data_inplace_c2c[i]));
        std::cout << "Max Difference (In-Place): " << max_diff_ip << std::endl
                  << std::endl;

        // ==========================================================
        // Method 4: Real-to-Complex / Complex-to-Real (RFFT / IRFFT)
        // Using convenience functions (NormMode::BACKWARD)
        // ==========================================================
        std::cout << "--- Method 4: Using RFFT / IRFFT ---" << std::endl;

        std::vector<Complex> rfft_spectrum;   // Size N/2 + 1
        std::vector<Real> irfft_reconstructed; // Size N

        // RFFT Forward Transform (using convenience function)
        OmniDSP::rfft(input_signal_real, rfft_spectrum);

        std::cout << "RFFT Spectrum (Size " << rfft_spectrum.size() << "):"
                  << std::endl;
        for (size_t i = 0; i < rfft_spectrum.size(); ++i)
            std::cout << rfft_spectrum[i]
                      << (i == rfft_spectrum.size() - 1 ? "" : ", ");
        std::cout << std::endl;

        // IRFFT Inverse Transform (using convenience function)
        OmniDSP::irfft(rfft_spectrum, irfft_reconstructed);

        std::cout << "\nIRFFT Reconstructed Signal (Size "
                  << irfft_reconstructed.size() << "):" << std::endl;
        // Note: Manual scaling might be needed here depending on backend/norm
        // mode if not BACKWARD. The convenience functions use BACKWARD, where
        // MKL scales by 1/N and Accelerate impl attempts internal scaling.
        for (size_t i = 0; i < irfft_reconstructed.size(); ++i)
            std::cout << irfft_reconstructed[i]
                      << (i == irfft_reconstructed.size() - 1 ? "" : ", ");
        std::cout << std::endl;

        // Verification (optional)
        Real max_diff_real = 0.0;
        if (irfft_reconstructed.size() == N) { // Check size just in case
            for (size_t i = 0; i < N; ++i)
                max_diff_real = std::max(
                    max_diff_real,
                    std::abs(input_signal_real[i] - irfft_reconstructed[i]));
        }
        std::cout << "Max Difference (Real): " << max_diff_real << std::endl
                  << std::endl;

        // ==========================================================
        // Example: Using ORTHO normalization via FFTPlan
        // ==========================================================
        std::cout << "--- Example: Using FFTPlan with ORTHO Norm "
                     "(RFFT/IRFFT) ---"
                  << std::endl;
        try {
            OmniDSP::FFTPlan<Real> plan_rfft_ortho(
                N, CURRENT_PRECISION, OmniDSP::Direction::FORWARD,
                OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);
            OmniDSP::FFTPlan<Real> plan_irfft_ortho(
                N, CURRENT_PRECISION, OmniDSP::Direction::INVERSE,
                OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);

            std::vector<Complex> spectrum_ortho(
                plan_rfft_ortho.getComplexLength());
            std::vector<Real> recon_ortho(plan_rfft_ortho.getLength());

            plan_rfft_ortho.execute_rfft(input_signal_real.data(),
                                         spectrum_ortho.data());
            plan_irfft_ortho.execute_irfft(spectrum_ortho.data(),
                                          recon_ortho.data());

            std::cout << "ORTHO Norm - RFFT/IRFFT completed." << std::endl;
            Real max_diff_ortho = 0.0;
            for (size_t i = 0; i < N; ++i)
                max_diff_ortho = std::max(
                    max_diff_ortho,
                    std::abs(input_signal_real[i] - recon_ortho[i]));
            std::cout << "Max Difference (ORTHO): " << max_diff_ortho << std::endl
                      << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error during ORTHO test: " << e.what() << std::endl
                      << std::endl;
            // Continue with other tests if possible
        }

        // ==========================================================
        // Example: Using Window Functions
        // ==========================================================
        std::cout << "--- Example: Using Window Functions ---" << std::endl;

        std::vector<Real> signal = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0,
                                    1.0}; // Example signal
        std::vector<Real> hann_windowed_signal = OmniDSP::Window::hann(signal);
        std::vector<Real> hamming_windowed_signal =
            OmniDSP::Window::hamming(signal);
        std::vector<Real> kaiser_windowed_signal =
            OmniDSP::Window::kaiser(signal, 8.6); // Example beta value
        std::vector<Real> flattop_windowed_signal =
            OmniDSP::Window::flattop(signal);

        std::cout << "Original Signal: ";
        for (Real val : signal)
            std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Hann Windowed Signal: ";
        for (Real val : hann_windowed_signal)
            std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Hamming Windowed Signal: ";
        for (Real val : hamming_windowed_signal)
            std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Kaiser Windowed Signal: ";
        for (Real val : kaiser_windowed_signal)
            std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Flat-top Windowed Signal: ";
        for (Real val : flattop_windowed_signal)
            std::cout << val << " ";
        std::cout << std::endl;

        // ==========================================================
        // Example: Using CQT
        // ==========================================================
        std::cout << "--- Example: Using CQT ---" << std::endl;

        // CQT parameters
        double cqt_sample_rate = 44100.0;
        double cqt_lowest_freq = 20.0;
        double cqt_highest_freq = 20000.0;
        int cqt_bins_per_octave = 12;

        // Create CQTPlan
        auto hann_window = [](const std::vector<Real>& input) {
            return OmniDSP::Window::hann(input);
        };
        OmniDSP::CQTPlan<Real> cqt_plan(cqt_sample_rate, cqt_lowest_freq,
                                        cqt_highest_freq, cqt_bins_per_octave, hann_window);

        // Generate a test signal (e.g., a simple chord)
        std::vector<Real> cqt_input_signal(4096);
        for (size_t i = 0; i < cqt_input_signal.size(); ++i) {
            double time = static_cast<double>(i) / cqt_sample_rate;
            cqt_input_signal[i] =
                std::cos(2.0 * M_PI * 220.0 * time) + // A3
                std::cos(2.0 * M_PI * 440.0 * time);  // A4
        }

        // Execute CQT
        std::vector<std::vector<Complex>> cqt_output;
        cqt_plan.execute(cqt_input_signal, cqt_output);

        // Print some CQT output (first 5 bins)
        std::cout << "CQT Output (first 5 bins):" << std::endl;
        for (size_t i = 0; i < std::min(cqt_output.size(), (size_t)5); ++i) {
            std::cout << "Bin " << i << ": ";
            for (const auto& val : cqt_output[i]) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "OmniDSP Example Finished." << std::endl;
    return 0;
}