# tests/cpp/CMakeLists.txt
# CMakeLists.txt for OmniDSP C++ Tests.
# This script is processed if OMNIDSP_BUILD_TESTS is ON,
# and the parent tests/CMakeLists.txt decides to include C++ tests
# (i.e., OMNIDSP_BUILD_PYTHON_BINDINGS is OFF).
# Assumes the OmniDSP library supports multiple runtime-selectable backends.

# OMNIDSP_BUILD_TESTS is assumed to be ON if this script is reached.
# OMNIDSP_BUILD_PYTHON_BINDINGS is assumed to be OFF if this script is reached.

message(STATUS "  C++ Tests: Configuring C++ tests (OMNIDSP_BUILD_TESTS=ON, OMNIDSP_BUILD_PYTHON_BINDINGS=OFF).")

# --- Find Dependencies ---

# Find the main library target (should be OmniDSP)
if(NOT TARGET OmniDSP) # Ensure this matches your main library target name
    message(FATAL_ERROR "OmniDSP library target (OmniDSP) not found. Build the main library first.")
endif()

# Find GoogleTest.
# Attempting to find it here if not already found by a parent script.
if(NOT TARGET GTest::gtest_main)
    message(STATUS "  C++ Tests: GTest::gtest_main target not found. Attempting to find GTest...")
    find_package(GTest REQUIRED) # Attempt to find GTest if not already found
    if(NOT TARGET GTest::gtest_main) # Re-check after find_package
        message(FATAL_ERROR "GoogleTest target (GTest::gtest_main) still not found. Cannot build C++ tests.")
    endif()
endif()
message(STATUS "  C++ Tests: GoogleTest target GTest::gtest_main found.")

# --- Define the test executable target ---
# The omnidsp_config.hpp header (included via OmniDSP target's include directories)
# will inform the test code about which backends are available.

add_executable(
    OmniDSP_cpp_tests
    # Data Loader Utility
    TestDataLoader.cpp
    # General Test Suites
    fft.cpp
    window.cpp
    cqt.cpp
    # Common backend tests or tests that iterate through available backends.
    # This file should be adapted to use the OmniDSP API to select/test different backends.
    backend/backend.cpp
    # Add other general test suite files here.
)

# --- Define Path to Reference Data Directory ---
set(TEST_DATA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
message(STATUS "  C++ Tests: Test data source directory: ${TEST_DATA_SRC_DIR}")

if(NOT EXISTS "${TEST_DATA_SRC_DIR}")
    message(WARNING "  C++ Tests: Test data directory not found: ${TEST_DATA_SRC_DIR}")
endif()

# Escape backslashes for Windows paths in C++ string literals if needed
if(WIN32)
    string(REPLACE "\\" "\\\\" TEST_DATA_SRC_DIR_ESC ${TEST_DATA_SRC_DIR})
else()
    set(TEST_DATA_SRC_DIR_ESC ${TEST_DATA_SRC_DIR})
endif()

# Pass the data directory path definition to the C++ code
target_compile_definitions(OmniDSP_cpp_tests PRIVATE OMNIDSP_TEST_DATA_DIR="${TEST_DATA_SRC_DIR_ESC}")

# --- Dependencies and Linking ---
target_link_libraries(OmniDSP_cpp_tests PRIVATE GTest::gtest_main OmniDSP) # Link against OmniDSP
message(STATUS "  C++ Tests: Linking OmniDSP_cpp_tests against GTest::gtest_main and OmniDSP.")

# Add include directories needed for the tests
target_include_directories(
    OmniDSP_cpp_tests
    PRIVATE
        # For OmniDSP/omnidsp.hpp, OmniDSP/omnidsp_config.hpp etc.
        # These are made available because OmniDSP target has PUBLIC include directories.
        ${CMAKE_CURRENT_SOURCE_DIR}    # For TestDataLoader.hpp and other local headers in tests/cpp
)
message(STATUS "  C++ Tests: Added include directories for OmniDSP_cpp_tests.")

# --- CTest Integration ---
include(GoogleTest) # Provides gtest_discover_tests
message(STATUS "  C++ Tests: Discovering tests using gtest_discover_tests...")
gtest_discover_tests(OmniDSP_cpp_tests)
message(STATUS "  C++ Tests: Finished discovering tests.")

message(STATUS "Configured C++ tests target: OmniDSP_cpp_tests")
