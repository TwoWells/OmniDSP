
# cmake/depend/googletest.cmake
# ==============================
# Finds the pre-installed GoogleTest library using find_package.
# Assumes GoogleTest is installed via Conda or system package manager.
# This is typically needed only if C++ tests are being built
# (i.e., OMNIDSP_BUILD_PYTHON_BINDINGS is OFF).
#
# Variables Read:
# - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake)
# ==============================

# Add top-level configuration message
message(STATUS "   Configuring GoogleTest dependency...")

# Find Google Test only if building C++ tests
if(NOT OMNIDSP_BUILD_PYTHON_BINDINGS)
    message(STATUS "     Finding pre-installed GoogleTest using find_package...")
    # Find the pre-installed GoogleTest package provided by Conda/system.
    # GoogleTest's CMake config files typically use 'GTest' as the package name.
    # REQUIRED will cause CMake to error out if the package is not found.
    find_package(GTest REQUIRED)

    # Check if the package was found (somewhat redundant with REQUIRED)
    if(GTest_FOUND)
        message(STATUS "       Found pre-installed GoogleTest version ${GTEST_VERSION}")
        # The necessary targets (like GTest::gtest, GTest::gmock) should now be available.
    else()
        # This part should not be reached if REQUIRED is used.
        message(FATAL_ERROR "     Could not find required GoogleTest package. Please ensure it is installed in your Conda environment or system.")
    endif()

    # --- FetchContent and related configurations removed ---
    # FetchContent_Declare(googletest ...)
    # set(INSTALL_GTEST OFF ...)
    # set(BUILD_GMOCK OFF ...)
    # set(BUILD_GTEST_SAMPLES OFF ...)
    # FetchContent_MakeAvailable(googletest)
    # --- End Removed Section ---

    # Note: Linking against GTest targets should happen in tests/cpp/CMakeLists.txt.
else()
    message(STATUS "     Skipping GoogleTest configuration (Python bindings enabled).")
endif()
