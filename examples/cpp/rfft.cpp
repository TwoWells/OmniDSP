/**
 * @file rfft.cpp
 * @brief Example usage of OmniDSP for Real FFTs (RFFT/IRFFT).
 *
 * Demonstrates Real-to-Complex and Complex-to-Real transforms using
 * convenience functions (BACKWARD norm) and FFTPlan (ORTHO norm).
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <algorithm> // For std::min, std::max
#include <OmniDSP/omnidsp.h>


int main() {
    // --- Configuration ---
    // Note: Accelerate backend requires N to be power-of-2 for REAL domain FFTs.
    const size_t N = 16; // FFT Size (must be power of 2 for Accelerate backend)
    using Real = double;
    using Complex = std::complex<Real>;
    const auto CURRENT_PRECISION = OmniDSP::Precision::DOUBLE;

    std::cout << "OmniDSP Real FFT Example (N=" << N << ")" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    // --- Prepare Input Data ---
    std::vector<Real> input_signal_real(N);
    double frequency = 3.0; // Example frequency for RFFT input
    for (size_t i = 0; i < N; ++i) {
        input_signal_real[i] = std::cos(2.0 * M_PI * frequency * static_cast<Real>(i) / N);
    }

    std::cout << "Input Real Signal:" << std::endl;
    for (size_t i = 0; i < N; ++i) std::cout << input_signal_real[i] << (i == N - 1 ? "" : ", ");
    std::cout << std::endl << std::endl;

    // --- Execute DSP functions within a try-catch block ---
    try {
        // ==========================================================
        // Method 1: RFFT / IRFFT using Convenience Functions
        // Uses NormMode::BACKWARD.
        // ==========================================================
        std::cout << "--- Method 1: Using RFFT / IRFFT Convenience Functions ---" << std::endl;

        std::vector<Complex> rfft_spectrum;   // Size N/2 + 1
        std::vector<Real> irfft_reconstructed; // Size N

        OmniDSP::rfft(input_signal_real, rfft_spectrum);
        std::cout << "RFFT Spectrum (Size " << rfft_spectrum.size() << "):" << std::endl;
        for (size_t i = 0; i < rfft_spectrum.size(); ++i) std::cout << rfft_spectrum[i] << (i == rfft_spectrum.size() - 1 ? "" : ", ");
        std::cout << std::endl;

        OmniDSP::irfft(rfft_spectrum, irfft_reconstructed);
        std::cout << "\nIRFFT Reconstructed Signal (Size " << irfft_reconstructed.size() << "):" << std::endl;
        for (size_t i = 0; i < irfft_reconstructed.size(); ++i) std::cout << irfft_reconstructed[i] << (i == irfft_reconstructed.size() - 1 ? "" : ", ");
        std::cout << std::endl;

        Real max_diff_real = 0.0;
        if (irfft_reconstructed.size() == N) {
            for (size_t i = 0; i < N; ++i) max_diff_real = std::max(max_diff_real, std::abs(input_signal_real[i] - irfft_reconstructed[i]));
        }
        std::cout << "Max Difference (Convenience): " << max_diff_real << std::endl << std::endl;

        // ==========================================================
        // Method 2: RFFT / IRFFT using FFTPlan with ORTHO Norm
        // ==========================================================
        std::cout << "--- Method 2: Using FFTPlan with ORTHO Norm (RFFT/IRFFT) ---" << std::endl;
        try {
            OmniDSP::FFTPlan<Real> plan_rfft_ortho(N, CURRENT_PRECISION, OmniDSP::Direction::FORWARD, OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);
            OmniDSP::FFTPlan<Real> plan_irfft_ortho(N, CURRENT_PRECISION, OmniDSP::Direction::INVERSE, OmniDSP::Domain::REAL, OmniDSP::NormMode::ORTHO);

            std::vector<Complex> spectrum_ortho(plan_rfft_ortho.getComplexLength());
            std::vector<Real> recon_ortho(plan_rfft_ortho.getLength());

            plan_rfft_ortho.execute_rfft(input_signal_real.data(), spectrum_ortho.data());
             std::cout << "RFFT Spectrum ORTHO (Size " << spectrum_ortho.size() << "):" << std::endl;
             for (size_t i = 0; i < spectrum_ortho.size(); ++i) std::cout << spectrum_ortho[i] << (i == spectrum_ortho.size() - 1 ? "" : ", ");
             std::cout << std::endl;

            plan_irfft_ortho.execute_irfft(spectrum_ortho.data(), recon_ortho.data());
             std::cout << "\nIRFFT Reconstructed Signal ORTHO (Size " << recon_ortho.size() << "):" << std::endl;
             for (size_t i = 0; i < recon_ortho.size(); ++i) std::cout << recon_ortho[i] << (i == recon_ortho.size() - 1 ? "" : ", ");
             std::cout << std::endl;


            Real max_diff_ortho = 0.0;
            for (size_t i = 0; i < N; ++i) max_diff_ortho = std::max(max_diff_ortho, std::abs(input_signal_real[i] - recon_ortho[i]));
            std::cout << "Max Difference (ORTHO): " << max_diff_ortho << std::endl << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error during ORTHO test: " << e.what() << std::endl << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR during Real FFT operation: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "OmniDSP Real FFT Example Finished Successfully." << std::endl;
    return 0;
}