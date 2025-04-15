# CMakeLists.txt for OmniDSP C++ Tests
# Uses GoogleTest found by the main CMakeLists.txt
# Defines path to reference data file for tests.

# Find the main library target (defined in parent scope)
if(NOT TARGET omnidsp)
    message(FATAL_ERROR "OmniDSP library target (omnidsp) not found. Build the main library first.")
endif()

# Find GoogleTest (should be found by parent CMakeLists.txt)
if(NOT TARGET GTest::gtest_main)
    message(WARNING "GoogleTest target (GTest::gtest_main) not found by parent CMakeLists.txt. Attempting find_package again.")
    find_package(GTest REQUIRED)
    if(NOT TARGET GTest::gtest_main)
         message(FATAL_ERROR "GoogleTest target (GTest::gtest_main) still not found. Cannot build C++ tests.")
    endif()
endif()

# Define the test executable target
add_executable(omnidsp_tests
    fft.cpp
    ifft.cpp
    rfft.cpp
    irfft.cpp
    backend_conv_test.cpp
    cqt.cpp
    window.cpp # Renamed from windows.cpp
    # Add other test cpp files here
)

# --- Define Path to Reference Data ---
# Define path relative to this build directory (tests/cpp)
set(TEST_REF_FILE_BUILD_PATH "${CMAKE_CURRENT_BINARY_DIR}/test_references.txt")
# Escape backslashes for Windows compile definitions
if(WIN32)
    string(REPLACE "\\" "\\\\" TEST_REF_FILE_BUILD_PATH_ESC ${TEST_REF_FILE_BUILD_PATH})
else()
    set(TEST_REF_FILE_BUILD_PATH_ESC ${TEST_REF_FILE_BUILD_PATH})
endif()

# Pass the path as a compile definition
target_compile_definitions(omnidsp_tests PRIVATE
    OMNIDSP_TEST_REF_FILE_PATH="${TEST_REF_FILE_BUILD_PATH_ESC}"
)
# Removed OMNIDSP_TEST_DATA_DIR definition as it's less reliable for runtime

# --- Copy Reference Data ---
# Copy the reference data file to the build directory where the test executable will run
# Ensures the path defined above is valid after build.
add_custom_command(
    TARGET omnidsp_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/test_references.txt"
            "${CMAKE_CURRENT_BINARY_DIR}/test_references.txt" # Copy to CMAKE_CURRENT_BINARY_DIR
            # Using "$<TARGET_FILE_DIR:omnidsp_tests>/test_references.txt" might place it
            # inside a config subdir (e.g., Release/) which complicates the fixed path definition.
            # Copying to CMAKE_CURRENT_BINARY_DIR is simpler if tests run from there.
    COMMENT "Copying test_references.txt to build directory (${CMAKE_CURRENT_BINARY_DIR})"
)


# Link the test executable against GTest and the OmniDSP library
target_link_libraries(omnidsp_tests PRIVATE
    GTest::gtest_main # Links gtest and gtest_main
    omnidsp
)

# Add include directories needed for the tests
target_include_directories(omnidsp_tests PRIVATE
    # Add path to find <OmniDSP/header.h>
    ${PROJECT_SOURCE_DIR}/include
    # Add path to find internal backend header if tests need friend access that pulls it in
    ${PROJECT_SOURCE_DIR}/src
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(omnidsp_tests)

message(STATUS "Configured C++ tests target: omnidsp_tests")
message(STATUS "Reference file path definition: OMNIDSP_TEST_REF_FILE_PATH=${TEST_REF_FILE_BUILD_PATH_ESC}")

