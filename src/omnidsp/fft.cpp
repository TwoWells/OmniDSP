/**
 * @file src/omnidsp/fft.cpp
 * @brief Implementation of FFT convenience functions for the OmniDSP library.
 *
 * Defines the public C++ convenience functions (fft, ifft, rfft, irfft)
 * which internally create and use FFTPlan objects. Also includes the
 * explicit template instantiations for these convenience functions.
 */

 #include <OmniDSP/omnidsp.h> // Main public header (declares FFTPlan and convenience funcs)
 #include <vector>
 #include <complex>
 #include <stdexcept>        // For potential exceptions from FFTPlan
 #include <type_traits>      // For std::is_same_v
 #include <cmath>            // For std::sqrt (used indirectly by FFTPlan ORTHO)
 #include <OmniDSP/omnidsp_export.h> // For OMNIDSP_EXPORT macro
 
 namespace OmniDSP
 {
 
     // --- Implementation of Convenience Functions ---
     // These functions provide a simpler interface using std::vector.
     // They internally create, use, and destroy an FFTPlan object using
     // the default NormMode::BACKWARD.
 
     /**
      * @brief Performs a Complex-to-Complex forward FFT (out-of-place) using NormMode::BACKWARD.
      */
     template <typename T>
     void fft(const std::vector<std::complex<T>> &input, std::vector<std::complex<T>> &output)
     {
         if (input.empty()) {
             output.clear();
             return;
         }
         const size_t N = input.size();
         output.resize(N);
         const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
 
         // Create and execute plan using default NormMode::BACKWARD
         FFTPlan<T> plan(N, prec, Direction::FORWARD, Domain::COMPLEX, NormMode::BACKWARD);
         plan.execute(input.data(), output.data());
     }
 
     /**
      * @brief Performs a Complex-to-Complex inverse FFT (out-of-place) using NormMode::BACKWARD.
      */
     template <typename T>
     void ifft(const std::vector<std::complex<T>> &input, std::vector<std::complex<T>> &output)
     {
         if (input.empty()) {
             output.clear();
             return;
         }
         const size_t N = input.size();
         output.resize(N);
         const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
 
         // Create and execute plan using default NormMode::BACKWARD
         FFTPlan<T> plan(N, prec, Direction::INVERSE, Domain::COMPLEX, NormMode::BACKWARD);
         plan.execute(input.data(), output.data());
     }
 
     /**
      * @brief Performs a Complex-to-Complex forward FFT (in-place) using NormMode::BACKWARD.
      */
     template <typename T>
     void fft_inplace(std::vector<std::complex<T>> &data)
     {
         if (data.empty()) return;
         const size_t N = data.size();
         const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
 
         // Create and execute plan using default NormMode::BACKWARD
         FFTPlan<T> plan(N, prec, Direction::FORWARD, Domain::COMPLEX, NormMode::BACKWARD);
         plan.execute(data.data()); // Call in-place execute
     }
 
     /**
      * @brief Performs a Complex-to-Complex inverse FFT (in-place) using NormMode::BACKWARD.
      */
     template <typename T>
     void ifft_inplace(std::vector<std::complex<T>> &data)
     {
         if (data.empty()) return;
         const size_t N = data.size();
         const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
 
         // Create and execute plan using default NormMode::BACKWARD
         FFTPlan<T> plan(N, prec, Direction::INVERSE, Domain::COMPLEX, NormMode::BACKWARD);
         plan.execute(data.data()); // Call in-place execute
     }
 
     /**
      * @brief Performs a Real-to-Complex forward FFT (out-of-place) using NormMode::BACKWARD.
      */
     template <typename T>
     void rfft(const std::vector<T> &real_input, std::vector<std::complex<T>> &complex_output)
     {
         if (real_input.empty()) {
             complex_output.clear();
             return;
         }
         const size_t N = real_input.size();
         const size_t Nc = N / 2 + 1; // Complex output size
         complex_output.resize(Nc);
         const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
 
         // Create and execute plan using default NormMode::BACKWARD
         FFTPlan<T> plan(N, prec, Direction::FORWARD, Domain::REAL, NormMode::BACKWARD);
         plan.execute_rfft(real_input.data(), complex_output.data());
     }
 
     /**
      * @brief Performs a Complex-to-Real inverse FFT (out-of-place) using NormMode::BACKWARD.
      */
     template <typename T>
     void irfft(const std::vector<std::complex<T>> &complex_input, std::vector<T> &real_output)
     {
         if (complex_input.empty()) {
             real_output.clear();
             return;
         }
         const size_t Nc = complex_input.size();
         if (Nc < 1) {
             real_output.clear();
             return;
         }
         const size_t N = (Nc == 1) ? 1 : 2 * (Nc - 1);
         real_output.resize(N);
 
         if (Nc == 1 && N == 1) {
             // Handle N=1 case directly (requires NormMode::BACKWARD scale=1/N=1.0)
             real_output[0] = complex_input[0].real();
             return;
         }
 
         const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;
 
         // Create and execute plan using default NormMode::BACKWARD
         FFTPlan<T> plan(N, prec, Direction::INVERSE, Domain::REAL, NormMode::BACKWARD);
         plan.execute_irfft(complex_input.data(), real_output.data());
     }
 
     // --- Explicit Template Instantiations (Definition) ---
     // Instantiate convenience functions for float and double.
     // OMNIDSP_EXPORT ensures these are exported from a DLL build.
 
     // C2C Functions
     template OMNIDSP_EXPORT void fft<float>(const std::vector<std::complex<float>> &, std::vector<std::complex<float>> &);
     template OMNIDSP_EXPORT void fft<double>(const std::vector<std::complex<double>> &, std::vector<std::complex<double>> &);
     template OMNIDSP_EXPORT void ifft<float>(const std::vector<std::complex<float>> &, std::vector<std::complex<float>> &);
     template OMNIDSP_EXPORT void ifft<double>(const std::vector<std::complex<double>> &, std::vector<std::complex<double>> &);
     template OMNIDSP_EXPORT void fft_inplace<float>(std::vector<std::complex<float>> &);
     template OMNIDSP_EXPORT void fft_inplace<double>(std::vector<std::complex<double>> &);
     template OMNIDSP_EXPORT void ifft_inplace<float>(std::vector<std::complex<float>> &);
     template OMNIDSP_EXPORT void ifft_inplace<double>(std::vector<std::complex<double>> &);
     // R2C / C2R Functions
     template OMNIDSP_EXPORT void rfft<float>(const std::vector<float> &, std::vector<std::complex<float>> &);
     template OMNIDSP_EXPORT void rfft<double>(const std::vector<double> &, std::vector<std::complex<double>> &);
     template OMNIDSP_EXPORT void irfft<float>(const std::vector<std::complex<float>> &, std::vector<float> &);
     template OMNIDSP_EXPORT void irfft<double>(const std::vector<std::complex<double>> &, std::vector<double> &);
 
 } // namespace OmniDSP
 