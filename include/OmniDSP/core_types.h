/**
 * @file core_types.h
 * @brief Defines core types and status codes for the OmniDSP library.
 */
#ifndef OMNIDSP_CORE_TYPES_H
#define OMNIDSP_CORE_TYPES_H

// Include export header if needed for types (usually not)
// #include <OmniDSP/omnidsp_export.h>

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
enum class Precision { Single, Double };

}  // namespace OmniDSP

#endif  // OMNIDSP_CORE_TYPES_H
