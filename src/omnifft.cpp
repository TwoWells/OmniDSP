// src/fft_lib.cpp

#include "omnifft.h" // Includes FFTPlan declaration + convenience func declarations
#include <vector>
#include <complex>
#include <stdexcept> // For convenience functions if they throw indirectly via FFTPlan
#include <cmath>     // For std::ceil, std::floor if needed (though size is N/2+1)

// NOTE: FFTPlan<T> methods (constructor, destructor, execute, getters, etc.)
//       are defined and instantiated in the conditionally compiled
//       fft_impl_onemkl.cpp, fft_impl_accelerate.cpp, or fft_impl_stub.cpp files.
//       This file only contains the platform-independent convenience functions.

namespace OmniFFT {

// --- Implementation of Convenience Functions ---

// C2C Forward (Out-of-place)
template <typename T>
void fft(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output) {
    if (input.empty()) { output.clear(); return; }
    output.resize(input.size());
    FFTPlan<T> plan(input.size(),
                    std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE,
                    Direction::FORWARD,
                    Domain::COMPLEX);
    plan.execute(input.data(), output.data());
}

// C2C Inverse (Out-of-place)
template <typename T>
void ifft(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output) {
     if (input.empty()) { output.clear(); return; }
    output.resize(input.size());
    FFTPlan<T> plan(input.size(),
                    std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE,
                    Direction::INVERSE,
                    Domain::COMPLEX);
    plan.execute(input.data(), output.data());
}

// C2C Forward (In-place)
template <typename T>
void fft_inplace(std::vector<std::complex<T>>& data) {
     if (data.empty()) return;
     FFTPlan<T> plan(data.size(),
                    std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE,
                    Direction::FORWARD,
                    Domain::COMPLEX);
     plan.execute(data.data());
}

// C2C Inverse (In-place)
template <typename T>
void ifft_inplace(std::vector<std::complex<T>>& data) {
     if (data.empty()) return;
     FFTPlan<T> plan(data.size(),
                    std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE,
                    Direction::INVERSE,
                    Domain::COMPLEX);
     plan.execute(data.data());
}

// R2C Forward (Out-of-place)
template <typename T>
void fft_r2c(const std::vector<T>& real_input, std::vector<std::complex<T>>& complex_output) {
    if (real_input.empty()) {
        complex_output.clear();
        return;
    }
    const size_t N = real_input.size();
    const size_t Nc = N / 2 + 1; // Complex output size
    complex_output.resize(Nc);

    FFTPlan<T> plan(N, // Length is real size N
                    std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE,
                    Direction::FORWARD,
                    Domain::REAL);
    plan.execute_r2c(real_input.data(), complex_output.data());
}

// C2R Inverse (Out-of-place)
template <typename T>
void ifft_c2r(const std::vector<std::complex<T>>& complex_input, std::vector<T>& real_output) {
    if (complex_input.empty()) {
        real_output.clear();
        return;
    }
    // Infer N from complex size Nc = N/2 + 1 => N = 2*(Nc-1)
    const size_t Nc = complex_input.size();
    if (Nc == 0) { // Or Nc == 1 ? A single DC value could imply N=0 or N=1? Let's assume Nc>=1
         real_output.clear(); // Or throw? Or handle N=0/1? Let's clear for Nc=0,1
         if (Nc == 1) real_output.resize(1, complex_input[0].real()); // DC only -> N=1 real?
         return;
    }
    const size_t N = 2 * (Nc - 1);
    real_output.resize(N);

    FFTPlan<T> plan(N, // Length is real size N
                    std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE,
                    Direction::INVERSE,
                    Domain::REAL);
    plan.execute_c2r(complex_input.data(), real_output.data());
    // Reminder: Scaling might need to be applied by caller depending on backend.
}


// --- Explicit Template Instantiations (Definition) ---
// Only instantiate the convenience functions in this file.

// Existing C2C instantiations
template void fft<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
template void fft<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
template void ifft<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
template void ifft<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
template void fft_inplace<float>(std::vector<std::complex<float>>&);
template void fft_inplace<double>(std::vector<std::complex<double>>&);
template void ifft_inplace<float>(std::vector<std::complex<float>>&);
template void ifft_inplace<double>(std::vector<std::complex<double>>&);

// New R2C/C2R instantiations
template void fft_r2c<float>(const std::vector<float>&, std::vector<std::complex<float>>&);
template void fft_r2c<double>(const std::vector<double>&, std::vector<std::complex<double>>&);
template void ifft_c2r<float>(const std::vector<std::complex<float>>&, std::vector<float>&);
template void ifft_c2r<double>(const std::vector<std::complex<double>>&, std::vector<double>&);


} // namespace OmniFFT