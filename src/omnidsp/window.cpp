/**
 * @file src/omnidsp/window.cpp
 * @brief Implementation of public OmniDSP::Window static methods.
 *
 * These methods dispatch window coefficient generation to the backend
 * and then apply the coefficients to the input signal.
 */

 #include <OmniDSP/window.h> // Public API header (renamed)
 #include <vector>
 #include <stdexcept>        // For std::invalid_argument
 #include <cmath>            // Included via window.h, but good practice
 #include <limits>           // Included via window.h, but good practice
 
 // Include the internal header that declares the backend implementation functions
 #include "backend/backend_impl.h" // Adjust path if necessary
 
 namespace OmniDSP {
 
     /**
      * @brief Applies the Hann window by getting coefficients from backend and multiplying.
      */
     template <typename T>
     std::vector<T> Window::hann(const std::vector<T>& input) {
         if (input.empty()) {
             // Match original behavior: throw for empty input
             throw std::invalid_argument("Input vector cannot be empty for Window::hann.");
             // Alternatively, return {}; to match other refactored functions
         }
         size_t length = input.size();
         // Get window coefficients from the backend
         std::vector<T> coeffs = Backend::generate_hann_window_impl<T>(length);
 
         // Apply coefficients to the input signal
         std::vector<T> output = input; // Create a copy to modify
         for (size_t i = 0; i < length; ++i) {
             output[i] *= coeffs[i];
         }
         return output;
     }
 
     /**
      * @brief Applies the Hamming window by getting coefficients from backend and multiplying.
      */
     template <typename T>
     std::vector<T> Window::hamming(const std::vector<T>& input) {
          if (input.empty()) {
             throw std::invalid_argument("Input vector cannot be empty for Window::hamming.");
          }
         size_t length = input.size();
         std::vector<T> coeffs = Backend::generate_hamming_window_impl<T>(length);
         std::vector<T> output = input;
         for (size_t i = 0; i < length; ++i) {
             output[i] *= coeffs[i];
         }
         return output;
     }
 
     /**
      * @brief Applies the Kaiser window by getting coefficients from backend and multiplying.
      */
     template <typename T>
     std::vector<T> Window::kaiser(const std::vector<T>& input, T beta) {
          if (input.empty()) {
             throw std::invalid_argument("Input vector cannot be empty for Window::kaiser.");
          }
          if (beta < 0) {
              // Add validation for beta parameter
              throw std::invalid_argument("Kaiser window beta parameter cannot be negative.");
          }
         size_t length = input.size();
         std::vector<T> coeffs = Backend::generate_kaiser_window_impl<T>(length, beta);
         std::vector<T> output = input;
         for (size_t i = 0; i < length; ++i) {
             output[i] *= coeffs[i];
         }
         return output;
     }
 
     /**
      * @brief Applies the Flattop window by getting coefficients from backend and multiplying.
      */
     template <typename T>
     std::vector<T> Window::flattop(const std::vector<T>& input) {
          if (input.empty()) {
             throw std::invalid_argument("Input vector cannot be empty for Window::flattop.");
          }
         size_t length = input.size();
         std::vector<T> coeffs = Backend::generate_flattop_window_impl<T>(length);
         std::vector<T> output = input;
         for (size_t i = 0; i < length; ++i) {
             output[i] *= coeffs[i];
         }
         return output;
     }
 
 
     // --- Explicit Template Instantiations (for public API) ---
     // Instantiate the static methods of the Window class for float and double.
     template std::vector<float> Window::hann<float>(const std::vector<float>& input);
     template std::vector<double> Window::hann<double>(const std::vector<double>& input);
 
     template std::vector<float> Window::hamming<float>(const std::vector<float>& input);
     template std::vector<double> Window::hamming<double>(const std::vector<double>& input);
 
     template std::vector<float> Window::kaiser<float>(const std::vector<float>& input, float beta);
     template std::vector<double> Window::kaiser<double>(const std::vector<double>& input, double beta);
 
     template std::vector<float> Window::flattop<float>(const std::vector<float>& input);
     template std::vector<double> Window::flattop<double>(const std::vector<double>& input);
 
 } // namespace OmniDSP
 