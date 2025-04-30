# CMakeLists.txt for OmniDSP project
# Uses Conda for managing CMake, pybind11, GTest, Boost, Highway, and MKL/IPP dependencies
# Supports compiling multiple backends (Default + available Accelerate/oneMKL) for runtime selection.

cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0148 NEW) # Policy for modern FindPython behavior
cmake_policy(SET CMP0167 NEW) # Policy for FindBoost module removal

# --- Automatically Add Conda Prefix to CMAKE_PREFIX_PATH ---
# Check if the CONDA_PREFIX environment variable is set
if(DEFINED ENV{CONDA_PREFIX})
  # Read the value of the CONDA_PREFIX environment variable
  set(CONDA_ENV_PREFIX "$ENV{CONDA_PREFIX}")
  message(STATUS "Detected Conda environment prefix: ${CONDA_ENV_PREFIX}")

  # Append the Conda prefix to CMAKE_PREFIX_PATH
  # Use list(APPEND ...) to handle paths correctly, especially on Windows
  list(APPEND CMAKE_PREFIX_PATH ${CONDA_ENV_PREFIX})
  # Optionally, make it unique in case it was already passed or present
  list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
  message(STATUS "Appended Conda prefix to CMAKE_PREFIX_PATH.")
else()
  message(STATUS "CONDA_PREFIX environment variable not set. Assuming system paths or manually provided CMAKE_PREFIX_PATH.")
endif()
# --- End Conda Prefix Handling ---

project(
  OmniDSP
  VERSION 1.0.0
  LANGUAGES CXX)

# Note: enable_testing() moved to cmake/testing.cmake

# --- Include Project Modules ---
# Create the cmake directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/cmake)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Include utility functions first
include(cmake/utils.cmake)

# --- Include Project Modules (Continued) ---
# Include project options
include(cmake/project_options.cmake)
# Include compiler settings
include(cmake/compiler_settings.cmake)
# Include core dependency finding/fetching orchestrator
include(cmake/depend.cmake)
# Include backend detection and configuration
# Requires cmake/backend/ directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/cmake/backend)
include(cmake/backend.cmake)

# --- Include Standard CMake Modules ---
# GNUInstallDirs is needed *before* installation.cmake
include(GNUInstallDirs)
# CMakePackageConfigHelpers is needed *before* installation.cmake
include(CMakePackageConfigHelpers)
# GenerateExportHeader is needed *before* target_definitions.cmake
include(GenerateExportHeader)
# FetchContent is included within depend.cmake

# --- Define and Configure Core Library Target ---
# This module uses variables set by previous includes (especially backend.cmake)
# to define the 'omnidsp' target and set its properties (sources, includes, links, etc.)
include(cmake/target_definitions.cmake)

# =============================================================================
# Conditionally Add Subdirectories (Python Bindings Source)
# =============================================================================
# Include the module that adds the Python bindings source subdirectory if enabled.
include(cmake/omnidsp_py.cmake)

# =============================================================================
# Examples Configuration
# =============================================================================
# Include the module that adds example subdirectories (currently only C++ examples).
# include(cmake/examples.cmake)

# =============================================================================
# Testing Configuration
# =============================================================================
# Include the module that enables testing and adds test subdirectories.
# include(cmake/testing.cmake)

# =============================================================================
# Installation Rules
# =============================================================================
# Include the module that defines all installation rules for the project.
# This should generally come after all targets that need to be installed are defined.
include(cmake/installation.cmake)

message(STATUS "Finished processing root CMakeLists.txt")
