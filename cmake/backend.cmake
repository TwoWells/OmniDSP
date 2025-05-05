# ==============================================================================
# cmake/backend.cmake (Refactored v2)
# ==============================================================================
# Defines options for enabling backends and includes their configuration scripts.
# The included scripts are responsible for checking options/platform, finding
# dependencies, setting the OMNIDSP_ENABLED_<BACKEND_NAME> variable, and
# creating the OmniDSP::<backend> interface target.

message(STATUS "Configuring OmniDSP Backends...")

# Include module for dependent options
include(CMakeDependentOption)

# --- Backend Options ---
# Define options to enable/disable optional backends.
# Defaults depend on the platform.

cmake_dependent_option(
    OMNIDSP_ENABLE_ACCELERATE
    "Enable Accelerate Backend (macOS only)"
    ON      # Default value IF dependency met (ON if APPLE)
    "APPLE" # Dependency variable(s)
    OFF     # Default value IF dependency NOT met (OFF if NOT APPLE)
)

cmake_dependent_option(
    OMNIDSP_ENABLE_ONEMKL
    "Enable oneMKL Backend (requires non-Apple OS and MKL library)"
    ON          # Default value IF dependency met (ON if NOT APPLE)
    "NOT APPLE" # Dependency variable(s)
    OFF         # Default value IF dependency NOT met (OFF if APPLE)
)

cmake_dependent_option(
    OMNIDSP_ENABLE_INTELIPP
    "Enable Intel IPP Backend (requires non-Apple OS and IPP library)"
    ON          # Default value IF dependency met (ON if NOT APPLE)
    "NOT APPLE" # Dependency variable(s)
    OFF         # Default value IF dependency NOT met (OFF if APPLE)
)

# Add options for other backends as needed...

# --- Include Backend Configuration Files ---
# These files will find dependencies and set variables like:
# - OMNIDSP_ENABLED_<BACKEND_NAME> (e.g., OMNIDSP_ENABLED_DEFAULT) -> TRUE/FALSE (CACHE BOOL)
# And create targets like:
# - OmniDSP::<backend_name> (e.g., OmniDSP::onemkl) -> INTERFACE IMPORTED target

# Default Backend (Always included - its script should set OMNIDSP_ENABLED_DEFAULT)
message(STATUS "  Including Default backend configuration...")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/default.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/backend/default.cmake")
else()
    message(FATAL_ERROR "Required Default backend configuration file cmake/backend/default.cmake not found.")
endif()
# Ensure the variable is set even if the include fails somehow (should be FATAL above)
if(NOT DEFINED OMNIDSP_ENABLED_DEFAULT)
    set(OMNIDSP_ENABLED_DEFAULT FALSE CACHE BOOL "Flag indicating Default backend is enabled" FORCE)
endif()


# Accelerate Backend (Include unconditionally - script handles enable/platform check)
message(STATUS "  Including Accelerate backend configuration...")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/accelerate.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/backend/accelerate.cmake")
else()
    message(WARNING "Optional Accelerate backend configuration file cmake/backend/accelerate.cmake not found.")
    set(OMNIDSP_ENABLED_ACCELERATE FALSE CACHE BOOL "Flag indicating Accelerate backend is enabled" FORCE) # Ensure flag is false if file missing
endif()
# Ensure the variable is set if the script failed to set it
if(NOT DEFINED OMNIDSP_ENABLED_ACCELERATE)
    set(OMNIDSP_ENABLED_ACCELERATE FALSE CACHE BOOL "Flag indicating Accelerate backend is enabled" FORCE)
endif()


# oneMKL Backend (Include unconditionally - script handles enable/platform check)
message(STATUS "  Including oneMKL backend configuration...")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/onemkl.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/backend/onemkl.cmake")
else()
    message(WARNING "Optional oneMKL backend configuration file cmake/backend/onemkl.cmake not found.")
    set(OMNIDSP_ENABLED_ONEMKL FALSE CACHE BOOL "Flag indicating Intel MKL backend is enabled" FORCE) # Ensure flag is false if file missing
endif()
# Ensure the variable is set if the script failed to set it
if(NOT DEFINED OMNIDSP_ENABLED_ONEMKL)
    set(OMNIDSP_ENABLED_ONEMKL FALSE CACHE BOOL "Flag indicating Intel MKL backend is enabled" FORCE)
endif()


# IntelIPP Backend (Include unconditionally - script handles enable/platform check)
message(STATUS "  Including IntelIPP backend configuration...")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/intelipp.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/backend/intelipp.cmake")
else()
    message(WARNING "Optional IntelIPP backend configuration file cmake/backend/intelipp.cmake not found.")
    set(OMNIDSP_ENABLED_INTELIPP FALSE CACHE BOOL "Flag indicating Intel IPP backend is enabled" FORCE) # Ensure flag is false if file missing
endif()
# Ensure the variable is set if the script failed to set it
if(NOT DEFINED OMNIDSP_ENABLED_INTELIPP)
    set(OMNIDSP_ENABLED_INTELIPP FALSE CACHE BOOL "Flag indicating Intel IPP backend is enabled" FORCE)
endif()


# --- Add includes for other optional backends here ---


# --- Report Enabled Backends ---
# Report based on OMNIDSP_ENABLED_* boolean flags
set(OMNIDSP_ENABLED_BACKEND_NAMES "")
if(OMNIDSP_ENABLED_DEFAULT) # Check the final ENABLED flag
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "Default")
endif()
if(OMNIDSP_ENABLED_ACCELERATE) # Check the final ENABLED flag
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "Accelerate")
endif()
if(OMNIDSP_ENABLED_ONEMKL) # Check the final ENABLED flag
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "oneMKL")
endif()
if(OMNIDSP_ENABLED_INTELIPP) # Check the final ENABLED flag
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "IntelIPP")
endif()
# Add checks for other OMNIDSP_ENABLED_* flags

if(NOT OMNIDSP_ENABLED_BACKEND_NAMES)
    message(FATAL_ERROR "No OmniDSP backends were configured successfully. Default backend failed?")
endif()

# Ensure the list is unique
list(REMOVE_DUPLICATES OMNIDSP_ENABLED_BACKEND_NAMES)

message(STATUS "Enabled OmniDSP Backends: ${OMNIDSP_ENABLED_BACKEND_NAMES}")
message(STATUS "Finished configuring OmniDSP Backends.")
