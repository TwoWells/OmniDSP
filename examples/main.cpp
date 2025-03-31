#define _USE_MATH_DEFINES // Required for M_PI with MSVC <cmath>
#include <cmath>          // Now includes constants like M_PI on MSVC

#include "omnifft.h"      // Include the library header
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>      // Include for std::exception

int main() {
    const size_t N = 16; // FFT size (Must be power of 2 for Accelerate REAL domain)
    using Complex = std::complex<double>;
    using Real = double;
    const auto CURRENT_PRECISION = OmniFFT::Precision::DOUBLE; // Set precision based on types used

    // --- Prepare Input Data (e.g., a cosine wave) ---
    std::vector<Complex> input_signal_c2c(N); // For C2C examples
    std::vector<Real> input_signal_real(N);   // For R2C example

    double frequency = 3.0; // Cycles per N samples
    for (size_t i = 0; i < N; ++i) {
        Real real_val = std::cos(2.0 * M_PI * frequency * static_cast<Real>(i) / N);
        input_signal_c2c[i] = Complex(real_val, 0.0); // Complex signal for C2C
        input_signal_real[i] = real_val;              // Real signal for R2C
    }

    std::cout << "Input Real Signal:" << std::endl;
    for(const auto& val : input_signal_real) std::cout << val << " ";
    std::cout << std::endl << std::endl;

    try {
        // --- Method 1: Using FFTPlan (C2C Example) ---
        std::cout << "--- Method 1: Using FFTPlan (C2C) ---" << std::endl;
        // *** CORRECTED CONSTRUCTOR CALLS ***
        OmniFFT::FFTPlan<Real> forward_plan_c2c(N, CURRENT_PRECISION, OmniFFT::Direction::FORWARD, OmniFFT::Domain::COMPLEX);
        OmniFFT::FFTPlan<Real> inverse_plan_c2c(N, CURRENT_PRECISION, OmniFFT::Direction::INVERSE, OmniFFT::Domain::COMPLEX);

        std::vector<Complex> spectrum_c2c_plan(N);
        std::vector<Complex> reconstructed_c2c_plan(N);

        // Forward FFT (out-of-place)
        forward_plan_c2c.execute(input_signal_c2c.data(), spectrum_c2c_plan.data());

        std::cout << "C2C Spectrum (Plan):" << std::endl;
        for(const auto& val : spectrum_c2c_plan) std::cout << val << " ";
        std::cout << std::endl;

        // Inverse FFT (out-of-place)
        inverse_plan_c2c.execute(spectrum_c2c_plan.data(), reconstructed_c2c_plan.data());

        // Note: Check scaling for C2C inverse! oneMKL usually scales, Accelerate DFT may not.

        std::cout << "\nC2C Reconstructed Signal (Plan):" << std::endl;
         for(const auto& val : reconstructed_c2c_plan) std::cout << val << " ";
        std::cout << std::endl << std::endl;


        // --- Method 2: Using Convenience Functions (C2C Example) ---
         std::cout << "--- Method 2: Using Convenience Functions (C2C) ---" << std::endl;
        std::vector<Complex> spectrum_conv_c2c;
        std::vector<Complex> reconstructed_conv_c2c;

        // Convenience functions were updated internally to use the correct constructor
        OmniFFT::fft(input_signal_c2c, spectrum_conv_c2c); // Forward FFT
        OmniFFT::ifft(spectrum_conv_c2c, reconstructed_conv_c2c); // Inverse FFT

        std::cout << "C2C Spectrum (Convenience Func):" << std::endl;
        for(const auto& val : spectrum_conv_c2c) std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "\nC2C Reconstructed Signal (Convenience Func):" << std::endl;
         for(const auto& val : reconstructed_conv_c2c) std::cout << val << " ";
        std::cout << std::endl << std::endl;


        // --- Method 3: In-place Transform (C2C Example) ---
         std::cout << "--- Method 3: In-place Transform (C2C) ---" << std::endl;
         std::vector<Complex> data_inplace_c2c = input_signal_c2c; // Copy input data

         // *** CORRECTED CONSTRUCTOR CALLS ***
         OmniFFT::FFTPlan<Real> inplace_plan_fwd_c2c(N, CURRENT_PRECISION, OmniFFT::Direction::FORWARD, OmniFFT::Domain::COMPLEX);
         inplace_plan_fwd_c2c.execute(data_inplace_c2c.data()); // Forward transform in-place

         std::cout << "C2C Spectrum (In-place Forward):" << std::endl;
         for(const auto& val : data_inplace_c2c) std::cout << val << " ";
         std::cout << std::endl;

         // *** CORRECTED CONSTRUCTOR CALLS ***
         OmniFFT::FFTPlan<Real> inplace_plan_inv_c2c(N, CURRENT_PRECISION, OmniFFT::Direction::INVERSE, OmniFFT::Domain::COMPLEX);
         inplace_plan_inv_c2c.execute(data_inplace_c2c.data()); // Inverse transform in-place

         // Add scaling if needed

         std::cout << "\nC2C Reconstructed Signal (In-place Inverse):" << std::endl;
         for(const auto& val : data_inplace_c2c) std::cout << val << " ";
         std::cout << std::endl << std::endl;

        // --- Method 4: Real-to-Complex / Complex-to-Real ---
        std::cout << "\n--- Method 4: Using R2C / C2R Transform ---" << std::endl;

        std::vector<Complex> r2c_spectrum; // Size N/2 + 1
        std::vector<Real> c2r_reconstructed; // Size N

        // R2C Forward Transform (using convenience function)
        OmniFFT::fft_r2c(input_signal_real, r2c_spectrum);

        std::cout << "R2C Spectrum (Size " << r2c_spectrum.size() << "):" << std::endl;
        for(const auto& val : r2c_spectrum) std::cout << val << " ";
        std::cout << std::endl;

        // C2R Inverse Transform (using convenience function)
        OmniFFT::ifft_c2r(r2c_spectrum, c2r_reconstructed);

        std::cout << "\nC2R Reconstructed Signal (Size " << c2r_reconstructed.size() << "):" << std::endl;
        // Apply scaling here if needed, especially for Accelerate backend
        // double scale = 1.0 / N;
        // for(auto& val : c2r_reconstructed) val *= scale;
        for(const auto& val : c2r_reconstructed) std::cout << val << " ";
        std::cout << std::endl << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}