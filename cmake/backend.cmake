# cmake/backend.cmake
# Detects and configures available compute backends.
# The Default backend is always included.
# Optional backends (Accelerate, oneMKL) are configured based on platform-dependent options.
# Sets OMNIDSP_SELECTED_BACKEND_TARGETS list.

message(STATUS "Configuring OmniDSP Backends...")

# Include module for dependent options
include(CMakeDependentOption)

# --- Backend Options ---

# Default backend is REQUIRED, no option provided.

# Accelerate backend: Defaults to ON only if APPLE is TRUE.
cmake_dependent_option(
    OMNIDSP_ENABLE_BACKEND_ACCELERATE # Option name
    "Enable Accelerate Backend (macOS only)" # Description
    ON          # Default value IF dependency met (ON if APPLE)
    "APPLE"     # Dependency variable(s)
    OFF         # Default value IF dependency NOT met (OFF if NOT APPLE)
)

# oneMKL backend: Defaults to ON only if APPLE is FALSE.
cmake_dependent_option(
    OMNIDSP_ENABLE_BACKEND_ONEMKL # Option name
    "Enable oneMKL Backend (requires non-Apple OS)" # Description
    ON          # Default value IF dependency met (ON if NOT APPLE)
    "NOT APPLE" # Dependency variable(s)
    OFF         # Default value IF dependency NOT met (OFF if APPLE)
)

# Add options for other backends as needed...

# List to hold the targets for the selected backends
set(OMNIDSP_SELECTED_BACKEND_TARGETS "")
set(OMNIDSP_ENABLED_BACKEND_NAMES "") # Keep track of names for user feedback

# --- Default Backend (Required) ---
message(STATUS "Including Required Default backend configuration...")
# Ensure the backend file exists before including
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/Default.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/backend/Default.cmake")
    # The Default.cmake script should add its target (e.g., omnidsp_backend_default)
    # to the OMNIDSP_SELECTED_BACKEND_TARGETS list.
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "Default")
else()
    # If the required backend is missing, it's a fatal error.
    message(FATAL_ERROR "Required Default backend configuration file cmake/backend/Default.cmake not found.")
endif()

# --- Accelerate Backend (Optional, macOS) ---
# The include logic remains the same, but the option's default state is now platform-dependent.
# The accelerate.cmake file itself also contains an APPLE check for safety.
if(OMNIDSP_ENABLE_BACKEND_ACCELERATE)
    # The check inside accelerate.cmake is the primary guard, but this outer check avoids unnecessary messages.
    if(APPLE)
        message(STATUS "Including Accelerate backend configuration...")
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/Accelerate.cmake")
            include("${CMAKE_CURRENT_LIST_DIR}/backend/Accelerate.cmake")
            # Accelerate.cmake should add its target to OMNIDSP_SELECTED_BACKEND_TARGETS
             list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "Accelerate")
        else()
            message(WARNING "Accelerate backend enabled, but cmake/backend/Accelerate.cmake not found.")
        endif()
    else()
         # This message should ideally not be reached if cmake_dependent_option works correctly,
         # but serves as a fallback message.
         message(STATUS "Accelerate backend disabled (platform is not Apple).")
    endif()
else()
     message(STATUS "Accelerate backend disabled (OMNIDSP_ENABLE_BACKEND_ACCELERATE=OFF).")
endif()

# --- oneMKL Backend (Optional, non-Apple) ---
# The include logic remains the same, but the option's default state is now platform-dependent.
# The onemkl.cmake file itself also contains an APPLE check.
if(OMNIDSP_ENABLE_BACKEND_ONEMKL)
     # Similar to Accelerate, the check inside onemkl.cmake is the primary guard.
     if(NOT APPLE)
        message(STATUS "Including oneMKL backend configuration...")
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/backend/oneMKL.cmake")
            include("${CMAKE_CURRENT_LIST_DIR}/backend/oneMKL.cmake")
            # oneMKL.cmake should add its target to OMNIDSP_SELECTED_BACKEND_TARGETS
             list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES "oneMKL")
        else()
            message(WARNING "oneMKL backend enabled, but cmake/backend/oneMKL.cmake not found.")
        endif()
     else()
         message(STATUS "oneMKL backend disabled (platform is Apple).")
     endif()
else()
     message(STATUS "oneMKL backend disabled (OMNIDSP_ENABLE_BACKEND_ONEMKL=OFF).")
endif()

# --- Add other backends similarly ---

# --- Validation ---
# This check remains valid, as OMNIDSP_SELECTED_BACKEND_TARGETS should always
# contain at least "omnidsp_backend_default".
if(NOT OMNIDSP_SELECTED_BACKEND_TARGETS)
    message(FATAL_ERROR "No OmniDSP backends were configured successfully. This should not happen as the Default backend is required.")
endif()

message(STATUS "Enabled OmniDSP Backends: ${OMNIDSP_ENABLED_BACKEND_NAMES}")
message(STATUS "Selected backend targets: ${OMNIDSP_SELECTED_BACKEND_TARGETS}")
message(STATUS "Finished configuring OmniDSP Backends.")
