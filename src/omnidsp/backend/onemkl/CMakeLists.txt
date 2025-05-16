# OmniDSP/src/omnidsp/backends/onemkl/CMakeLists.txt
# Configures the oneMKL backend for the 'OmniDSP' library.
# Adapted from the original structure for direct interaction with OmniDSP target.

# This script assumes OPTION_OMNIDSP_ENABLE_ONEMKL is ON if it's processed.
message(STATUS "    Configuring oneMKL backend sources and dependencies (Original Adapted)...")

# Local status variable, will be propagated to parent scope.
set(OMNIDSP_BACKEND_ONEMKL_ENABLED_LOCAL 0)

# --- Find oneMKL Package ---
# find_package(MKL ...) sets MKL_FOUND and MKL_VERSION_STRING (or MKL_VERSION)
find_package(MKL QUIET)

include(FindPackageHandleStandardArgs)
# FPHSA will check MKL_FOUND (as specified by FOUND_VAR)
# and also any variables listed in REQUIRED_VARS.
# The overall result (whether MKL is considered "found" by FPHSA)
# is implicitly reflected in the value of MKL_FOUND after this call.
find_package_handle_standard_args(MKL
    FOUND_VAR MKL_FOUND             # Tell FPHSA to check the standard MKL_FOUND variable
    REQUIRED_VARS MKL_FOUND         # We require MKL_FOUND to be true.
                                    # If MKL sets specific component found vars, list them here.
    VERSION_VAR MKL_VERSION_LOCAL   # Store the version in MKL_VERSION_LOCAL
                                    # FPHSA will use MKL_VERSION or MKL_VERSION_STRING if set by find_package(MKL)
    FAIL_MESSAGE "oneMKL core components not found by find_package(MKL). Please ensure MKL is installed and MKLROOT or CMAKE_PREFIX_PATH is set."
)
# After this, MKL_FOUND reflects the outcome of FPHSA.
# MKL_VERSION_LOCAL will hold the version string.

if(NOT MKL_FOUND) # Check the standard MKL_FOUND which FPHSA would have finalized.
    message(WARNING "      oneMKL package not successfully found or requirements not met (MKL_FOUND=${MKL_FOUND}). oneMKL backend will be disabled.")
    set(OMNIDSP_BACKEND_ONEMKL_ENABLED ${OMNIDSP_BACKEND_ONEMKL_ENABLED_LOCAL} PARENT_SCOPE) # Propagate 0
    return()
endif()

# The original script specifically checked for TARGET MKL::MKL.
if(NOT TARGET MKL::MKL)
    message(WARNING "      oneMKL package was found (MKL_FOUND=TRUE), but the specific target MKL::MKL was NOT defined by find_package(MKL). This backend version requires this target. oneMKL backend will be disabled. Consider trying `find_package(MKL CONFIG ...)` if your MKL installation supports it and provides standard CMake targets like MKL::mkl_intel_lp64 or MKL::MKL.")
    set(OMNIDSP_BACKEND_ONEMKL_ENABLED ${OMNIDSP_BACKEND_ONEMKL_ENABLED_LOCAL} PARENT_SCOPE) # Propagate 0
    return()
endif()

message(STATUS "      Found oneMKL and target MKL::MKL. Version: ${MKL_VERSION_LOCAL}") # Use MKL_VERSION_LOCAL

# --- Define oneMKL Backend Source Files ---
set(ONEMKL_BACKEND_SOURCES_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp"
)

# --- Add Sources to the 'OmniDSP' library target ---
target_sources(OmniDSP PRIVATE ${ONEMKL_BACKEND_SOURCES_LIST})
message(STATUS "        Added oneMKL backend sources to OmniDSP target.")

# --- Include Directories for oneMKL Backend ---
# The MKL::MKL target should propagate its include directories.
# Adding current source dir for this backend's own headers (e.g., utils.hpp).
target_include_directories(OmniDSP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Link oneMKL libraries to OmniDSP target ---
# Directly link the MKL::MKL target to the main OmniDSP library.
target_link_libraries(OmniDSP PRIVATE MKL::MKL)
message(STATUS "        Linked OmniDSP target with MKL::MKL.")

# --- Compile Definition for oneMKL Backend internal C++ code ---
target_compile_definitions(OmniDSP PRIVATE OMNIDSP_INTERNAL_USE_ONEMKL)

# --- Signal Success and Set Configuration Variable for omnidsp_config.hpp ---
set(OMNIDSP_BACKEND_ONEMKL_ENABLED_LOCAL 1)
set(OMNIDSP_BACKEND_ONEMKL_ENABLED ${OMNIDSP_BACKEND_ONEMKL_ENABLED_LOCAL} PARENT_SCOPE)

message(STATUS "      oneMKL backend configured and enabled successfully.")
message(STATUS "    Finished configuring oneMKL backend.")
