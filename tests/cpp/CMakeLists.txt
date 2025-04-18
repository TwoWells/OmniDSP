# CMakeLists.txt for OmniDSP C++ Tests Uses GoogleTest found by the main
# CMakeLists.txt Defines path to reference data file for tests. Conditionally
# compiles backend-specific test files from backend/ subdirectory.

message(STATUS "Processing tests/cpp/CMakeLists.txt...")

# Find the main library target
if(NOT TARGET omnidsp)
  message(
    FATAL_ERROR
      "OmniDSP library target (omnidsp) not found. Build the main library first."
  )
endif()

# Find GoogleTest
if(NOT TARGET GTest::gtest_main)
  message(
    WARNING
      "GoogleTest target (GTest::gtest_main) not found by parent CMakeLists.txt. Attempting find_package again."
  )
  find_package(GTest REQUIRED)
  if(NOT TARGET GTest::gtest_main)
    message(
      FATAL_ERROR
        "GoogleTest target (GTest::gtest_main) still not found. Cannot build C++ tests." #
    )
  endif()
else()
  message(STATUS "tests/cpp: GoogleTest target GTest::gtest_main found.") #
endif()

# --- Determine which backend-specific test file to include ---
set(BACKEND_SPECIFIC_TEST_SRC "") # Initialize empty
message(
  STATUS
    "tests/cpp: Checking BACKEND variable for conditional test compilation: ${BACKEND}"
)

# Construct the path relative to the current CMakeLists.txt file
if(BACKEND STREQUAL "ONEMKL")
  set(BACKEND_SPECIFIC_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/backend/onemkl.cpp
  )# Use relative path
  message(
    STATUS
      "tests/cpp: Including MKL-specific tests: ${BACKEND_SPECIFIC_TEST_SRC}")
elseif(BACKEND STREQUAL "ACCELERATE")
  set(BACKEND_SPECIFIC_TEST_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/backend/accelerate.cpp) # Use relative path
  message(
    STATUS
      "tests/cpp: Including Accelerate-specific tests: ${BACKEND_SPECIFIC_TEST_SRC}" #
  )
else() # Assuming "NONE" or empty means Stub backend
  set(BACKEND_SPECIFIC_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/backend/stub.cpp
  )# Use relative path
  message(
    STATUS
      "tests/cpp: Including Stub-specific tests: ${BACKEND_SPECIFIC_TEST_SRC}")
endif()

# Define the test executable target, including the conditionally selected file
add_executable(
  omnidsp_tests
  # Common tests (sources relative to this CMakeLists.txt)
  fft.cpp # Now includes ifft, rfft, irfft tests
  # ifft.cpp        # Removed - tests moved to fft.cpp
  # rfft.cpp        # Removed - tests moved to fft.cpp
  # irfft.cpp       # Removed - tests moved to fft.cpp
  cqt.cpp
  window.cpp
  test_data_utils.cpp # Utility for loading reference data
  # Common backend tests + fixture (source relative to this CMakeLists.txt)
  ${CMAKE_CURRENT_SOURCE_DIR}/backend/backend.cpp # Use relative path
  # Backend-specific tests (variable now holds the full relative path) #
  ${BACKEND_SPECIFIC_TEST_SRC})

# --- Define Path to Reference Data ---
set(TEST_REF_FILE_BUILD_PATH "${CMAKE_CURRENT_BINARY_DIR}/test_references.txt")
if(WIN32)
  # Escape backslashes for Windows paths in C++ string literals
  string(REPLACE "\\" "\\\\" TEST_REF_FILE_BUILD_PATH_ESC
                 ${TEST_REF_FILE_BUILD_PATH})
else()
  set(TEST_REF_FILE_BUILD_PATH_ESC ${TEST_REF_FILE_BUILD_PATH})
endif()

# Pass the reference file path definition
target_compile_definitions(
  omnidsp_tests
  PRIVATE OMNIDSP_TEST_REF_FILE_PATH="${TEST_REF_FILE_BUILD_PATH_ESC}")
get_target_property(TEST_DEFS omnidsp_tests COMPILE_DEFINITIONS)
message(
  STATUS "tests/cpp: Final COMPILE_DEFINITIONS for omnidsp_tests: ${TEST_DEFS}")

# --- Copy Reference Data ---
# Ensure the reference data file is copied to the build directory where the test
# executable runs
add_custom_command(
  TARGET omnidsp_tests
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/test_references.txt" # Source file location
    "${CMAKE_CURRENT_BINARY_DIR}/test_references.txt" # Destination in build dir
  COMMENT
    "Copying test_references.txt to build directory (${CMAKE_CURRENT_BINARY_DIR})"
)

# Link the test executable #
target_link_libraries(omnidsp_tests PRIVATE GTest::gtest_main omnidsp) #
message(
  STATUS
    "tests/cpp: Linking omnidsp_tests against GTest::gtest_main and omnidsp.") #

# Add include directories needed for the tests
target_include_directories(
  omnidsp_tests
  PRIVATE ${PROJECT_SOURCE_DIR}/include # For OmniDSP headers
          ${PROJECT_SOURCE_DIR}/src # Potentially for internal headers if needed
                                    # by tests
)
message(STATUS "tests/cpp: Added include directories for omnidsp_tests.")

# Add tests to CTest
include(GoogleTest)
message(STATUS "tests/cpp: Discovering tests using gtest_discover_tests...")
gtest_discover_tests(omnidsp_tests)
message(STATUS "tests/cpp: Finished discovering tests.")

message(STATUS "Configured C++ tests target: omnidsp_tests")
