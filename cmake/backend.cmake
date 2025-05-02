# ==============================================================================
# cmake/backend.cmake
# ==============================================================================
# Defines options for enabling backends and includes their configuration scripts.
# The included scripts are responsible for checking options/platform and setting
# variables indicating availability and build information.

message(STATUS "Configuring OmniDSP Backends...")

# Include module for dependent options
include(CMakeDependentOption)

# --- Backend Options ---
# Define options to enable/disable optional backends.
# Defaults depend on the platform.

cmake_dependent_option(
    OMNIDSP_ENABLE_ACCELERATE # Renamed option for clarity
    "Enable Accelerate Backend (macOS only)"
    ON      # Default value IF dependency met (ON if APPLE)
    "APPLE" # Dependency variable(s)
    OFF     # Default value IF dependency NOT met (OFF if NOT APPLE)
)

cmake_dependent_option(
    OMNIDSP_ENABLE_ONEMKL # Renamed option for clarity
    "Enable oneMKL Backend (requires non-Apple OS)"
    ON          # Default value IF dependency met (ON if NOT APPLE)
    "NOT APPLE" # Dependency variable(s)
    OFF         # Default value IF dependency NOT met (OFF if APPLE)
)

# Add options for other backends as needed...

# --- Include Backend Configuration Files ---
# These files will find dependencies and set variables like:
# - OMNIDSP_HAS_<BACKEND_NAME> (e.g., OMNIDSP_HAS_DEFAULT) -> TRUE/FALSE
# - <BACKEND_NAME>_BACKEND_SOURCES -> List of source files
# - <BACKEND_NAME>_BACKEND_INCLUDE_DIRS -> List of include directories
# - <BACKEND_NAME>_BACKEND_LINK_LIBS -> List of link libraries/targets
# - OMNIDSP_HAS_BACKEND_*_VALUE -> 1/0 for omnidsp_config.h

# Default Backend (Always included)
message(STATUS "  Including Default backend configuration...")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/default.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/backend/default.cmake")
else()
    message(FATAL_ERROR "Required Default backend configuration file cmake/backend/default.cmake not found.")
endif()

# Accelerate Backend (Include unconditionally - script handles enable/platform check)
message(STATUS "  Including Accelerate backend configuration (will proceed only if enabled/supported)...")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/accelerate.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/backend/accelerate.cmake")
else()
    message(WARNING "Optional Accelerate backend configuration file cmake/backend/accelerate.cmake not found.")
    set(OMNIDSP_HAS_ACCELERATE FALSE) # Ensure flag is false if file missing
    set(OMNIDSP_HAS_BACKEND_ACCELERATE_VALUE 0) # Ensure value is 0
endif()

# oneMKL Backend (Include unconditionally - script handles enable/platform check)
message(STATUS "  Including oneMKL backend configuration (will proceed only if enabled/supported)...")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/onemkl.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/backend/onemkl.cmake")
else()
    message(WARNING "Optional oneMKL backend configuration file cmake/backend/onemkl.cmake not found.")
    set(OMNIDSP_HAS_ONEMKL FALSE) # Ensure flag is false if file missing
    set(OMNIDSP_HAS_BACKEND_ONEMKL_VALUE 0) # Ensure value is 0
endif()

# --- Add includes for other optional backends here ---

# --- Report Enabled Backends ---
# (This section remains the same, reporting based on OMNIDSP_HAS_* flags)
set(OMNIDSP_ENABLED_BACKEND_NAMES "")
if(OMNIDSP_HAS_DEFAULT)
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "Default")
endif()
if(OMNIDSP_HAS_ACCELERATE)
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "Accelerate")
endif()
if(OMNIDSP_HAS_ONEMKL)
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "oneMKL")
endif()
# Add checks for other OMNIDSP_HAS_* flags

if(NOT OMNIDSP_ENABLED_BACKEND_NAMES)
    message(FATAL_ERROR "No OmniDSP backends were configured successfully. Default backend failed?")
endif()

message(STATUS "Enabled OmniDSP Backends: ${OMNIDSP_ENABLED_BACKEND_NAMES}")
message(STATUS "Finished configuring OmniDSP Backends.")
