# cmake/config.cmake
# ===================================
# Project-specific CMake options and global compiler settings.
# ===================================

message(STATUS "Loading project configuration (options and compiler settings from cmake/config.cmake)...")

# ==============================================================================
# Project Options
# ==============================================================================
# Defines project-specific CMake options using the option() command. These options control various
# build configurations like enabling Python bindings, setting the library type, and toggling debug
# features.
#
# Variables Set:
#   - OMNIDSP_BUILD_PYTHON_BINDINGS (BOOL): Controls Python binding build.
#   - OMNIDSP_BUILD_TESTS (BOOL): Controls test builds.
#   - OMNIDSP_BUILD_EXAMPLES (BOOL): Controls C++ example builds.
#   - BUILD_SHARED_LIBS (BOOL): Controls shared/static library build. Default: ON.
#   - OMNIDSP_LIBRARY_TYPE (STRING): Set to SHARED or STATIC based on BUILD_SHARED_LIBS.
#   - OMNIDSP_ENABLE_MKL_DEBUG_LOGS (BOOL): Enables MKL backend debug logs. Default: OFF.
#   - OMNIDSP_ENABLE_CQT_WARNINGS (BOOL): Enables CQT Nk clamping warnings. Default: OFF.
# ==============================================================================

message(STATUS "  Setting project build options...")

# Option to control whether Python bindings are built
option(OMNIDSP_BUILD_PYTHON_BINDINGS "Build the Python language bindings" OFF)

# Option to control whether tests are built
option(OMNIDSP_BUILD_TESTS "Build tests for OmniDSP" OFF)

# Option to control whether C++ examples are built
option(OMNIDSP_BUILD_EXAMPLES "Build C++ examples for OmniDSP" OFF)

# Option to control shared/static library build type
# Note: BUILD_SHARED_LIBS is a standard CMake variable, often set externally.
# Defining it here provides a default if not set by the user/environment.
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so/.dylib)" ON)

# --- Determine Library Type based on BUILD_SHARED_LIBS ---
if(BUILD_SHARED_LIBS)
    set(OMNIDSP_LIBRARY_TYPE SHARED)
    message(STATUS "    Building OmniDSP as SHARED library.")
else()
    set(OMNIDSP_LIBRARY_TYPE STATIC)
    message(STATUS "    Building OmniDSP as STATIC library.")
endif()
# --- End Library Type Determination ---

# Options to control verbose logging
option(OMNIDSP_ENABLE_MKL_DEBUG_LOGS "Enable verbose MKL backend debug logs" OFF)
option(OMNIDSP_ENABLE_CQT_WARNINGS "Enable CQT Nk clamping warnings" OFF)

message(STATUS "  Finished setting project build options.")

# ==============================================================================
# Compiler Settings
# ==============================================================================
message(STATUS "  Setting global compiler configurations...")

# Enable generation of compile_commands.json for IDEs/tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message(STATUS "    Enabled export of compile commands (compile_commands.json).")

# Set C++ standard requirements
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard features over compiler extensions
message(STATUS "    Set CMAKE_CXX_STANDARD to ${CMAKE_CXX_STANDARD} (Required, No Extensions).")

message(STATUS "  Finished setting global compiler configurations.")
message(STATUS "Finished loading project configuration from cmake/config.cmake.")
