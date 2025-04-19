/**
 * @file omnidsp.h
 * @brief Main public API include header for the OmniDSP library.
 *
 * Includes core types and all feature-specific headers.
 *
 * @version 2.0.6 (Using core_types.h)
 * @date 2025-04-18
 */

#ifndef OMNIDSP_H
#define OMNIDSP_H

// --- Include Core Types First ---
#include <OmniDSP/core_types.h>  // Defines Precision, OMNIDSP_STATUS, etc. in namespace OmniDSP

// --- Include Feature-Specific Public Headers ---
// These headers define their contents within the OmniDSP namespace internally.
// Includes are placed OUTSIDE any namespace block here. Order matters.
#include <OmniDSP/convolution.h>  // Defines convolution functions inside namespace OmniDSP
#include <OmniDSP/cqt.h>  // Defines CQTPlan inside namespace OmniDSP (depends on fft.h)
#include <OmniDSP/fft.h>  // Defines FFTNorm, FFTPlan, etc. inside namespace OmniDSP
#include <OmniDSP/resample.h>  // Defines resample functions inside namespace OmniDSP
#include <OmniDSP/window.h>  // Defines Window class inside namespace OmniDSP

// Include export header if needed for functions declared directly in omnidsp.h
// (if any) #include <OmniDSP/omnidsp_export.h>

#endif  // OMNIDSP_H
