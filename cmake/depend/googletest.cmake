# cmake/depend/googletest.cmake
# ==============================
# Fetches and configures GoogleTest using FetchContent.
# This is typically needed only if C++ tests are being built
# (i.e., OMNIDSP_BUILD_PYTHON_BINDINGS is OFF).
#
# Variables Read:
# - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake)
# ==============================

# Add top-level configuration message
message(STATUS "  Configuring GoogleTest dependency...")

# Fetch Google Test only if building C++ tests
if(NOT OMNIDSP_BUILD_PYTHON_BINDINGS)
    # Adjust indentation for fetching message
    message(STATUS "    Fetching GoogleTest v1.16.0 using FetchContent...")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.16.0 # Pin to a specific release tag for reproducibility
      GIT_SHALLOW    TRUE
    )
    # Set GTest build options *before* FetchContent_MakeAvailable
    # Prevent GTest from installing itself and building samples/gmock
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE) # Optionally disable gmock if only gtest is needed
    set(BUILD_GTEST_SAMPLES OFF CACHE BOOL "" FORCE)
    # Fetch, configure, build (if necessary), and make GTest targets available
    FetchContent_MakeAvailable(googletest)
    # Adjust indentation for made available message
    message(STATUS "      Made googletest targets available (e.g., GTest::gtest).")
    # Note: Linking against GTest targets should happen in tests/cpp/CMakeLists.txt.
else()
    message(STATUS "    Skipping GoogleTest FetchContent (Python bindings enabled).")
endif()
