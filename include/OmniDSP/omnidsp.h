/**
 * @file omnidsp.h
 * @brief Main public API include header for the OmniDSP library.
 *
 * This header includes the core types/status codes and brings in the
 * declarations for all major functionalities (FFT, CQT, Convolution, Window,
 * Resample) from their respective headers. Including this single header is
 * often sufficient for using the library.
 *
 * @version 2.0.2 (Moved FFT enums to fft.h)
 * @date 2025-04-18
 */

#ifndef OMNIDSP_H
#define OMNIDSP_H

// --- Include the generated export header ---
// Defines OMNIDSP_API for DLL symbol handling
#include <OmniDSP/omnidsp_export.h>  // Adjust path/name if needed

/**
 * @brief Main namespace for the OmniDSP library.
 */
namespace OmniDSP {

// --- Core Types and Status Codes ---

/**
 * @brief Status code returned by some OmniDSP functions.
 * Typically 0 for success, non-zero for failure.
 */
typedef int OMNIDSP_STATUS;

/** @brief Indicates successful execution. */
const OMNIDSP_STATUS OMNIDSP_SUCCESS = 0;
/** @brief Indicates general failure during execution. */
const OMNIDSP_STATUS OMNIDSP_FAILURE = -1;
/** @brief Indicates an invalid argument was provided. */
const OMNIDSP_STATUS OMNIDSP_INVALID_ARGUMENT = -2;
/** @brief Indicates backend-specific error occurred. */
const OMNIDSP_STATUS OMNIDSP_BACKEND_ERROR = -3;
// Add more specific error codes as needed

// --- Core Enums (General) ---

/** @brief Specifies the floating-point precision for calculations. */
enum class Precision { SINGLE, DOUBLE };

// Removed FFT-specific enums (FFTNorm, Direction, Domain) - Moved to fft.h

}  // namespace OmniDSP

// --- Include Feature-Specific Public Headers ---
// These headers rely on the types defined above and define their own specific
// types/classes.

#include <OmniDSP/convolution.h>  // For convolve1d, correlate1d, ConvMode
#include <OmniDSP/cqt.h>          // For CQTPlan
#include <OmniDSP/fft.h>  // For FFTPlan, RFFTPlan, FFTNorm, Direction, Domain etc.
#include <OmniDSP/resample.h>  // For filter_and_downsample
#include <OmniDSP/window.h>    // For Window class (applying/getting windows)

#endif  // OMNIDSP_H
