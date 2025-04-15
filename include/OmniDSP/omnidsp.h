/**
 * @file omnidsp.h
 * @brief Main public API include header for the OmniDSP library.
 *
 * This header includes the core enums and brings in the declarations
 * for all major functionalities (FFT, CQT, Convolution, Window, Resample)
 * from their respective headers. Including this single header is often
 * sufficient for using the library.
 *
 * @version 2.0.0 (Refactored API)
 * @date 2025-04-15
 */

 #ifndef OMNIDSP_H
 #define OMNIDSP_H
 
 // --- Include the generated export header ---
 // Defines OMNIDSP_EXPORT for DLL symbol handling
 #include <OmniDSP/omnidsp_export.h> // Adjust path/name if needed
 
 /**
  * @brief Main namespace for the OmniDSP library.
  */
 namespace OmniDSP
 {
     // --- Core Enums ---
     // (Could potentially be moved to a separate types.h header)
 
     /** @brief Specifies the direction of the Fourier Transform. */
     enum class Direction { FORWARD, INVERSE };
 
     /** @brief Specifies the floating-point precision for calculations. */
     enum class Precision { SINGLE, DOUBLE };
 
     /** @brief Specifies the domain of the input/output signals for FFT. */
     enum class Domain { COMPLEX, REAL };
 
     /** @brief Specifies the normalization/scaling mode applied to FFTs. */
     enum class NormMode { BACKWARD, ORTHO, FORWARD };
 
 } // namespace OmniDSP
 
 
 // --- Include Feature-Specific Public Headers ---
 
 #include <OmniDSP/fft.h>         // For FFTPlan and FFT convenience functions
 #include <OmniDSP/cqt.h>         // For CQTPlan
 #include <OmniDSP/convolution.h> // For convolve1d, correlate1d
 #include <OmniDSP/window.h>      // For Window class (applying windows)
 #include <OmniDSP/resample.h>    // For filter_and_downsample
 
 
 #endif // OMNIDSP_H
 