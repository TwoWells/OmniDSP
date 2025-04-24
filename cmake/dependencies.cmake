# cmake/dependencies.cmake
# ========================
# Finds core, non-backend-specific dependencies required by the OmniDSP project.
# It uses find_package() to locate libraries like pybind11, Python, GTest,
# Boost, and Highway. The results (found status, paths, versions) are typically
# stored in variables set by the respective find modules/configs (e.g.,
# pybind11_FOUND).
#
# Dependencies Expected: - Conda environment providing: pybind11, Python (if
# bindings ON), GTest (if bindings OFF), Boost, Highway
#
# Variables Read: - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake)
# ========================

message(STATUS "Finding core project dependencies...")

# --- Find Core Dependencies (Managed by Conda) ---

# Find pybind11 (Required for Python bindings, but might be useful even if
# bindings are off) Using CONFIG mode as recommended for modern pybind11
find_package(pybind11 CONFIG REQUIRED)
message(
  STATUS
    "Found pybind11 version ${pybind11_VERSION} via find_package at ${pybind11_DIR}"
)

# Find Python Interpreter and Development libraries if Python bindings are
# enabled
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  find_package(
    Python
    COMPONENTS Interpreter Development # Find both interpreter and headers/libs
    REQUIRED)
  message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
  message(STATUS "Found Python Development Libraries: ${Python_LIBRARIES}")
  message(STATUS "Found Python Include Directories: ${Python_INCLUDE_DIRS}")
else()
  message(
    STATUS "Python bindings are disabled (OMNIDSP_BUILD_PYTHON_BINDINGS=OFF).")
endif()

# Find Google Test if Python bindings are *not* enabled (assuming C++ tests are
# built then)
if(NOT OMNIDSP_BUILD_PYTHON_BINDINGS)
  find_package(GTest REQUIRED)
  message(STATUS "Found GoogleTest for C++ tests.")
  # Note: Linking GTest::gtest and GTest::gtest_main should happen later in the
  # test targets defined in tests/cpp/CMakeLists.txt
endif()

# Find Boost (Headers are usually sufficient, but find the whole package)
# CMP0167 NEW policy means FindBoost is deprecated, but it's often still
# used/needed. If your Boost installation provides config files,
# `find_package(Boost CONFIG REQUIRED)` is preferred.
find_package(Boost REQUIRED)
message(STATUS "Found Boost version ${Boost_VERSION}")
# Note: Linking Boost::headers is usually done later via target_link_libraries

# Find Google Highway library using its config file
find_package(highway CONFIG REQUIRED)
message(STATUS "Found Google Highway library.")
# Note: Linking hwy::hwy is done later via target_link_libraries

message(STATUS "Finished finding core dependencies.")
