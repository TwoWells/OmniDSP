# OmniDSP/src/omnidsp/backends/CMakeLists.txt
# This file manages the inclusion of individual backend CMake configurations.

message(STATUS "  Configuring OmniDSP backends...")

# --- Default Backend (Mandatory) ---
# The OMNIDSP_BACKEND_DEFAULT_ENABLED variable is set within default/CMakeLists.txt
# with PARENT_SCOPE, making it available in this (backends/CMakeLists.txt) scope.
add_subdirectory(default)

# --- Dispatcher Backend (Core Component) ---
# The dispatcher is not optional in the same way as optimized backends;
# it's a core part of the backend selection mechanism.
add_subdirectory(dispatcher)

# --- Accelerate Backend (Conditional) ---
if(OPTION_OMNIDSP_ENABLE_ACCELERATE)
    message(STATUS "    Attempting to include Accelerate backend configuration (backends/CMakeLists.txt)...")
    add_subdirectory(accelerate) # Sets OMNIDSP_BACKEND_ACCELERATE_ENABLED in this scope
else()
    message(STATUS "    Accelerate backend disabled by option (backends/CMakeLists.txt).")
endif()

# --- oneMKL Backend (Conditional) ---
if(OPTION_OMNIDSP_ENABLE_ONEMKL)
    message(STATUS "    Attempting to include oneMKL backend configuration (backends/CMakeLists.txt)...")
    add_subdirectory(onemkl) # Sets OMNIDSP_BACKEND_ONEMKL_ENABLED in this scope
else()
    message(STATUS "    oneMKL backend disabled by option (backends/CMakeLists.txt).")
endif()

# --- Intel IPP Backend (Conditional) ---
if(OPTION_OMNIDSP_ENABLE_INTELIPP)
    message(STATUS "    Attempting to include Intel IPP backend configuration (backends/CMakeLists.txt)...")
    add_subdirectory(intelipp) # Sets OMNIDSP_BACKEND_INTELIPP_ENABLED in this scope
else()
    message(STATUS "    Intel IPP backend disabled by option (backends/CMakeLists.txt).")
endif()

message(STATUS "  Finished configuring OmniDSP backends subdirectories.")

# --- Propagate Backend Enabled Status to Parent Scope ---
# The OMNIDSP_BACKEND_*_ENABLED variables were set by the individual backend
# CMakeLists.txt files into *this* file's scope (backends/CMakeLists.txt)
# using PARENT_SCOPE. Now, we explicitly set them again with PARENT_SCOPE
# to make them visible in this file's parent (src/omnidsp/CMakeLists.txt).

if(DEFINED OMNIDSP_BACKEND_DEFAULT_ENABLED)
    set(OMNIDSP_BACKEND_DEFAULT_ENABLED ${OMNIDSP_BACKEND_DEFAULT_ENABLED} PARENT_SCOPE)
endif()
if(DEFINED OMNIDSP_BACKEND_ACCELERATE_ENABLED)
    set(OMNIDSP_BACKEND_ACCELERATE_ENABLED ${OMNIDSP_BACKEND_ACCELERATE_ENABLED} PARENT_SCOPE)
endif()
if(DEFINED OMNIDSP_BACKEND_ONEMKL_ENABLED)
    set(OMNIDSP_BACKEND_ONEMKL_ENABLED ${OMNIDSP_BACKEND_ONEMKL_ENABLED} PARENT_SCOPE)
endif()
if(DEFINED OMNIDSP_BACKEND_INTELIPP_ENABLED)
    set(OMNIDSP_BACKEND_INTELIPP_ENABLED ${OMNIDSP_BACKEND_INTELIPP_ENABLED} PARENT_SCOPE)
endif()
