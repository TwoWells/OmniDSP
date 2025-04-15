/**
 * @file cqt.cpp
 * @brief Example usage of the OmniDSP CQTPlan class.
 *
 * Demonstrates performing a Constant Q Transform using the efficient
 * recursive frequency-domain implementation.
 * Updated includes after header refactoring. Added M_PI guard.
 */

 #include <cmath>
 #include <iostream>
 #include <vector>
 #include <complex>
 #include <stdexcept>
 #include <algorithm> // For std::min
 #include <OmniDSP/omnidsp.h> // Main header includes enums
 #include <OmniDSP/window.h>  // Use the renamed window header
 #include <OmniDSP/cqt.h>     // Include the CQT header

 // Define M_PI if it's not already defined (e.g., by <cmath> in some environments/standards)
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif


 // Helper function to generate window coefficients (e.g., Hann)
 // This matches the signature expected by the updated CQTPlan constructor binding
 // NOTE: This is a manual implementation for demonstration. One could potentially
 //       call OmniDSP::Window::hann here if only coefficients were needed,
 //       but the public API applies the window. This manual version is fine.
 template <typename T>
 std::vector<T> generate_hann_coeffs(size_t required_length) {
     if (required_length == 0) return {};
     std::vector<T> coeffs(required_length);
     // Use M_PI from <cmath> or the definition above
     double denom = (required_length > 1) ? static_cast<double>(required_length - 1) : 1.0;
     for (size_t n = 0; n < required_length; ++n) {
         coeffs[n] = static_cast<T>(0.5 - 0.5 * std::cos(2.0 * M_PI * n / denom));
     }
     // CQTPlan internally handles normalization during kernel generation.
     // No explicit L1 normalization needed here for the generator function.
     return coeffs;
 }


 int main() {
     using Real = double;
     using Complex = std::complex<Real>;

     std::cout << "OmniDSP CQT Example" << std::endl;
     std::cout << "-------------------" << std::endl;

     try {
         // ==========================================================
         // CQT Example
         // ==========================================================
         std::cout << "--- Setting up and Running CQT ---" << std::endl;

         // CQT parameters
         double cqt_sample_rate = 44100.0;
         double cqt_lowest_freq = 55.0;    // A1 note
         double cqt_highest_freq = 1760.0; // A6 note
         int cqt_bins_per_octave = 12;     // Semitones
         // Hop length must be compatible with the number of octaves.
         // A1-A6 is 5 octaves (0..4), so hop must be divisible by 2^(5-1) = 16.
         size_t cqt_hop_length = 512; // Example valid hop length

         // Define the window function generator using the helper
         auto hann_window_gen_func = generate_hann_coeffs<Real>;

         // Create CQTPlan
         std::cout << "Creating CQT Plan..." << std::endl;
         // Optional args for sparsity_threshold and fir_filter_order use defaults here.
         OmniDSP::CQTPlan<Real> cqt_plan(cqt_sample_rate, cqt_hop_length,
                                         cqt_lowest_freq, cqt_highest_freq,
                                         cqt_bins_per_octave,
                                         hann_window_gen_func); // Pass the generator function

         std::cout << "CQT Plan created (Num Bins: " << cqt_plan.getNumBins()
                   << ", Num Octaves: " << cqt_plan.getNumOctaves()
                   << ", Hop Length: " << cqt_plan.getHopLength() << ")" << std::endl;


         // Generate a test signal (e.g., a simple chord A3 + A4)
         size_t cqt_signal_len = static_cast<size_t>(cqt_sample_rate * 2); // 2 seconds of audio
         std::cout << "Generating test signal (Length: " << cqt_signal_len << ")..." << std::endl;
         std::vector<Real> cqt_input_signal(cqt_signal_len);
         double freq1 = 220.0; // A3
         double freq2 = 440.0; // A4
         for (size_t i = 0; i < cqt_input_signal.size(); ++i) {
             double time = static_cast<double>(i) / cqt_sample_rate;
             cqt_input_signal[i] = 0.5 * std::cos(2.0 * M_PI * freq1 * time) +
                                   0.5 * std::cos(2.0 * M_PI * freq2 * time);
         }

         // Execute CQT
         std::cout << "Executing CQT..." << std::endl;
         // Output is std::vector<std::vector<std::complex<Real>>> [bin][frame]
         std::vector<std::vector<Complex>> cqt_output;
         cqt_plan.execute(cqt_input_signal, cqt_output);
         std::cout << "CQT Execution complete." << std::endl;

         // --- Print CQT Output ---
         if (cqt_output.empty() || cqt_output[0].empty()) {
              std::cout << "\nCQT Output is empty (possibly due to short input signal)." << std::endl;
         } else {
             size_t num_bins_out = cqt_output.size();
             size_t num_frames_out = cqt_output[0].size();

             // Print some CQT output (magnitudes for first few bins of the *first frame*)
             size_t num_bins_to_print = std::min((size_t)15, num_bins_out);
             std::cout << "\nCQT Output Magnitudes (First Frame, first " << num_bins_to_print
                       << " bins of " << num_bins_out << ", Total Frames: " << num_frames_out << "):" << std::endl;

             for (size_t i = 0; i < num_bins_to_print; ++i) {
                  // Calculate approximate center frequency for context
                  double bin_freq = cqt_plan.getLowestFrequency() * std::pow(2.0, static_cast<double>(i) / cqt_plan.getBinsPerOctave());
                  // Access the first frame [i][0]
                  std::cout << "Bin " << i << " (~" << static_cast<int>(std::round(bin_freq)) << " Hz): "
                            << std::abs(cqt_output[i][0]) // Get magnitude of complex coeff for bin i, frame 0
                            << std::endl;
             }
             if (num_bins_out > num_bins_to_print) std::cout << "..." << std::endl;
         }


     } catch (const std::exception& e) {
         std::cerr << "\nFATAL ERROR during CQT operation: " << e.what() << std::endl;
         return 1;
     }

     std::cout << "\nOmniDSP CQT Example Finished Successfully." << std::endl;
     return 0;
 }
