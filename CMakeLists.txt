# OmniDSP/CMakeLists.txt (Root - Refactored)
# Main project file. Sets up basic project structure and includes sub-modules.

cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0148 NEW) # Policy for modern FindPython behavior

project(
  OmniDSP
  VERSION 1.0.0
  LANGUAGES CXX)

# --- Include Project Utility Modules from ./cmake ---

include(cmake/project_options.cmake) # Sets OMNIDSP_LIBRARY_TYPE, BUILD_SHARED_LIBS, OMNIDSP_BUILD_PYTHON_BINDINGS, OMNIDSP_BUILD_TESTS, OMNIDSP_BUILD_EXAMPLES etc.
include(cmake/compiler_settings.cmake)
include(cmake/depend.cmake) # Handles finding/fetching common dependencies. Should NOT find Python/pybind11.

# --- Add Core Library Subdirectory ---
add_subdirectory(src/omnidsp) # Defines the 'OmniDSP' C++ library target

# --- Include Standard CMake Modules (if needed at root level) ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# =============================================================================
# Testing Configuration
# =============================================================================
# The OMNIDSP_BUILD_TESTS option should be defined (e.g., in cmake/project_options.cmake)
if(OMNIDSP_BUILD_TESTS)
    message(STATUS "Testing enabled by option (OMNIDSP_BUILD_TESTS=ON).")
    # Enable CTest support globally if tests are to be built.
    # This needs to be done before any add_test() calls, which might occur in subdirectories.
    enable_testing()
    message(STATUS "  CTest support enabled.")

    # Add the main tests subdirectory.
    # Its CMakeLists.txt will handle adding specific test type subdirectories (cpp, python).
    message(STATUS "  Attempting to add main tests subdirectory (tests)...")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests ${PROJECT_BINARY_DIR}/tests)
    else()
        message(WARNING "  Directory tests or its CMakeLists.txt not found. Skipping all test configurations.")
    endif()
else()
    message(STATUS "Testing disabled (OMNIDSP_BUILD_TESTS=OFF).")
endif()

# =============================================================================
# Conditionally Add Python Bindings Subdirectory
# =============================================================================
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
    message(STATUS "Python bindings enabled by option (OMNIDSP_BUILD_PYTHON_BINDINGS=ON). Adding src/omnidsp_py subdirectory.")
    add_subdirectory(src/omnidsp_py ${PROJECT_BINARY_DIR}/src/omnidsp_py)
else()
    message(STATUS "Python bindings disabled (OMNIDSP_BUILD_PYTHON_BINDINGS=OFF). Skipping src/omnidsp_py.")
endif()

# =============================================================================
# Examples Configuration
# =============================================================================
# The examples/cpp/CMakeLists.txt is responsible for checking OMNIDSP_BUILD_EXAMPLES
# and OMNIDSP_BUILD_PYTHON_BINDINGS to determine if it should actually configure anything.
message(STATUS "Attempting to add examples/cpp subdirectory (configuration depends on options in its CMakeLists.txt)...")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/CMakeLists.txt")
    add_subdirectory(examples/cpp ${PROJECT_BINARY_DIR}/examples/cpp)
else()
    message(STATUS "  Directory examples/cpp or its CMakeLists.txt not found. Skipping examples configuration.")
endif()


# =============================================================================
# Installation Rules
# =============================================================================
include(cmake/installation.cmake)

message(STATUS "Finished processing root CMakeLists.txt")
