# CMakeLists.txt for OmniDSP C++ Tests (Refactored) Uses GoogleTest found by the
# main CMakeLists.txt. Defines path to reference data directory
# (OMNIDSP_TEST_DATA_DIR) for tests. Assumes reference data files (.txt) are
# pre-generated and exist in tests/cpp/data/. Conditionally compiles
# backend-specific test files from backend/ subdirectory.

message(
  STATUS
    "Processing tests/cpp/CMakeLists.txt (Refactored - No Python Execution)...")

# --- Find Dependencies ---

# Find the main library target
if(NOT TARGET omnidsp)
  message(
    FATAL_ERROR
      "OmniDSP library target (omnidsp) not found. Build the main library first."
  )
endif()

# Find GoogleTest (assuming it's found by the parent CMakeLists.txt)
if(NOT TARGET GTest::gtest_main)
  message(
    FATAL_ERROR
      "GoogleTest target (GTest::gtest_main) not found by parent CMakeLists.txt. Cannot build C++ tests."
  )
else()
  message(STATUS "tests/cpp: GoogleTest target GTest::gtest_main found.")
endif()

# --- Determine which backend-specific test file to include ---
set(BACKEND_SPECIFIC_TEST_SRC "") # Initialize empty
message(
  STATUS
    "tests/cpp: Checking BACKEND variable for conditional test compilation: ${BACKEND}"
)

# Construct the path relative to the current CMakeLists.txt file
if(BACKEND STREQUAL "ONEMKL")
  set(BACKEND_SPECIFIC_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/backend/onemkl.cpp)
  message(
    STATUS
      "tests/cpp: Including MKL-specific tests: ${BACKEND_SPECIFIC_TEST_SRC}")
elseif(BACKEND STREQUAL "ACCELERATE")
  set(BACKEND_SPECIFIC_TEST_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/backend/accelerate.cpp)
  message(
    STATUS
      "tests/cpp: Including Accelerate-specific tests: ${BACKEND_SPECIFIC_TEST_SRC}"
  )
else() # Assuming "NONE" or empty means Stub backend
  set(BACKEND_SPECIFIC_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/backend/stub.cpp)
  message(
    STATUS
      "tests/cpp: Including Stub-specific tests: ${BACKEND_SPECIFIC_TEST_SRC}")
endif()

# --- Define the test executable target ---

add_executable(
  omnidsp_tests
  # Data Loader Utility (relative to this CMakeLists.txt)
  TestDataLoader.cpp
  # Test Suites (sources relative to this CMakeLists.txt, now in tests/ subdir)
  tests/fft.cpp
  tests/window.cpp
  tests/cqt.cpp
  # Add other test suite files from tests/ subdir here...
  # Common backend tests + fixture (relative path)
  backend/backend.cpp
  # Backend-specific tests (variable holds the full relative path)
  ${BACKEND_SPECIFIC_TEST_SRC})

# --- Define Path to Reference Data Directory ---

# Define path relative to the source directory of tests/cpp
set(TEST_DATA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
message(STATUS "tests/cpp: Test data source directory: ${TEST_DATA_SRC_DIR}")

# Escape backslashes for Windows paths in C++ string literals if needed
if(WIN32)
  string(REPLACE "\\" "\\\\" TEST_DATA_SRC_DIR_ESC ${TEST_DATA_SRC_DIR})
else()
  set(TEST_DATA_SRC_DIR_ESC ${TEST_DATA_SRC_DIR})
endif()

# Pass the data directory path definition to the C++ code
target_compile_definitions(
  omnidsp_tests PRIVATE OMNIDSP_TEST_DATA_DIR="${TEST_DATA_SRC_DIR_ESC}")

# Conditionally add the backend definition if it's set
if(BACKEND_DEFINITION)
  target_compile_definitions(omnidsp_tests PRIVATE ${BACKEND_DEFINITION})
  message(
    STATUS
      "tests/cpp: Added backend definition ${BACKEND_DEFINITION} to omnidsp_tests"
  )
endif()

get_target_property(TEST_DEFS omnidsp_tests COMPILE_DEFINITIONS)
message(
  STATUS "tests/cpp: Final COMPILE_DEFINITIONS for omnidsp_tests: ${TEST_DEFS}")

# --- Dependencies and Linking ---

# Removed dependency on generate_test_data target

# Link the test executable
target_link_libraries(omnidsp_tests PRIVATE GTest::gtest_main omnidsp)
message(
  STATUS
    "tests/cpp: Linking omnidsp_tests against GTest::gtest_main and omnidsp.")

# Add include directories needed for the tests
target_include_directories(
  omnidsp_tests
  PRIVATE ${PROJECT_SOURCE_DIR}/include # For OmniDSP headers
          ${CMAKE_CURRENT_SOURCE_DIR} # For TestDataLoader.h
          # ${PROJECT_SOURCE_DIR}/src # Potentially for internal headers if
          # needed
)
message(STATUS "tests/cpp: Added include directories for omnidsp_tests.")

# --- CTest Integration ---
include(GoogleTest)
message(STATUS "tests/cpp: Discovering tests using gtest_discover_tests...")
gtest_discover_tests(omnidsp_tests)
message(STATUS "tests/cpp: Finished discovering tests.")

message(STATUS "Configured C++ tests target: omnidsp_tests")
