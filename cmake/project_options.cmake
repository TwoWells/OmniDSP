# cmake/project_options.cmake
# ===========================
# Defines project-specific CMake options using the option() command. These options control various
# build configurations like enabling Python bindings, setting the library type, and toggling debug
# features.
#
# Variables Set:
#   - OMNIDSP_BUILD_PYTHON_BINDINGS (BOOL): Controls Python binding build. Default: OFF.
#   - BUILD_SHARED_LIBS (BOOL): Controls shared/static library build. Default: ON.
#   - OMNIDSP_LIBRARY_TYPE (STRING): Set to SHARED or STATIC based on BUILD_SHARED_LIBS.
#   - OMNIDSP_ENABLE_MKL_DEBUG_LOGS (BOOL): Enables MKL backend debug logs. Default: OFF.
#   - OMNIDSP_ENABLE_CQT_WARNINGS (BOOL): Enables CQT Nk clamping warnings. Default: OFF.
# ===========================

# Option to control whether Python bindings are built
option(OMNIDSP_BUILD_PYTHON_BINDINGS "Build the Python language bindings" OFF)

# Option to control shared/static library build type Note: BUILD_SHARED_LIBS is a standard CMake
# variable, often set externally. Defining it here provides a default if not set by the
# user/environment.
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so/.dylib)" ON)

# --- Determine Library Type based on BUILD_SHARED_LIBS ---
# Moved this logic from target_definitions.cmake
if(BUILD_SHARED_LIBS)
    set(OMNIDSP_LIBRARY_TYPE SHARED)
    message(STATUS "  Building omnidsp as SHARED library.")
else()
    set(OMNIDSP_LIBRARY_TYPE STATIC)
    message(STATUS "  Building omnidsp as STATIC library.")
endif()
# --- End Library Type Determination ---

# Options to control verbose logging
option(OMNIDSP_ENABLE_MKL_DEBUG_LOGS "Enable verbose MKL backend debug logs" OFF)
option(OMNIDSP_ENABLE_CQT_WARNINGS "Enable CQT Nk clamping warnings" OFF)

message(STATUS "Loaded project options from cmake/project_options.cmake")
