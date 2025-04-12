# tests/cpp/CMakeLists.txt
# Updated to find GTest, include backend tests conditionally, and set paths/definitions.
# Find GoogleTest dependency (managed by Conda)
find_package(GTest REQUIRED)
message(STATUS "Found GoogleTest for C++ tests.")

# Automatically find all .cpp source files in this directory
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp)

# Check if any test source files were found
if(NOT TEST_SOURCES)
    message(WARNING "No C++ test source files (.cpp) found in tests/cpp/ directory.")
    return() # Exit early if no sources
endif()

message(STATUS "Found C++ test sources: ${TEST_SOURCES}")

# Create the test executable using the automatically discovered source files
add_executable(omnidsp_tests ${TEST_SOURCES})

# --- Apply Backend Compile Definition (Inherited from Root) ---
# This variable is set in the parent CMakeLists.txt
if(DEFINED BACKEND_DEFINITION AND NOT BACKEND_DEFINITION STREQUAL "") # Use renamed variable
    target_compile_definitions(omnidsp_tests PRIVATE ${BACKEND_DEFINITION}) # Use renamed variable
    message(STATUS "Applying inherited backend compile definition to tests: -D${BACKEND_DEFINITION}") # Use renamed variable
else()
    message(STATUS "No specific backend compile definition inherited (Stub backend active).")
endif()

# --- Add Include Directories ---
target_include_directories(omnidsp_tests PRIVATE
    # Add path to find <OmniDSP/header.h>
    ${PROJECT_SOURCE_DIR}/include

    # Add path to find internal backend header "../../src/omnidsp/backend/backend_impl.h"
    ${PROJECT_SOURCE_DIR}/src
)

# --- Link Libraries ---
# Link the test executable against GoogleTest and the OmniDSP library
target_link_libraries(omnidsp_tests PRIVATE
    GTest::gtest_main # Links GTest and provides main()
    omnidsp # Link against the library target built in the parent scope
)

# --- Ensure reference data file is available at test runtime ---
# Check if the reference file exists in the source directory
set(REFERENCE_FILE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/test_references.txt")
set(REFERENCE_FILE_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/test_references.txt") # Destination in build dir

if(EXISTS "${REFERENCE_FILE_SOURCE}")
    # Copy the file from the source dir to the build dir where the test runs
    configure_file("${REFERENCE_FILE_SOURCE}" "${REFERENCE_FILE_DESTINATION}" COPYONLY)
    message(STATUS "Configured copying of ${REFERENCE_FILE_SOURCE} to ${REFERENCE_FILE_DESTINATION} for tests.")
else()
    message(WARNING "Reference data file not found: ${REFERENCE_FILE_SOURCE}. backend_conv_test will likely fail.")
    # Optionally, create an empty file to prevent runtime file-not-found errors,
    # though the tests would still fail without data.
    # file(WRITE "${REFERENCE_FILE_DESTINATION}" "# Reference file missing!\n")
endif()

# --- CTest Integration ---
# Add the executable to CTest's list of tests
include(GoogleTest) # Include this to get gtest_discover_tests
gtest_discover_tests(omnidsp_tests)

# --- CTest Environment Setup for Shared Libraries ---
# (Keep the existing block for handling shared library paths if needed)
if(BUILD_SHARED_LIBS) # Check option defined in the root CMakeLists.txt
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        # Use ENVIRONMENT_MODIFICATION (CMake 3.21+)
        set(OMNIDSP_LIB_BUILD_DIR "$<TARGET_FILE_DIR:omnidsp>")
        set(ENV_VAR_MOD "") # Default empty
        if(UNIX AND NOT APPLE)
            set(ENV_VAR_MOD "LD_LIBRARY_PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        elseif(APPLE)
            set(ENV_VAR_MOD "DYLD_LIBRARY_PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        elseif(WIN32)
            set(ENV_VAR_MOD "PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        endif()
        if(NOT ENV_VAR_MOD STREQUAL "")
            message(VERBOSE "CTest: Setting ENVIRONMENT_MODIFICATION for 'omnidsp_tests': ${ENV_VAR_MOD}")
            set_tests_properties(omnidsp_tests PROPERTIES ENVIRONMENT_MODIFICATION "${ENV_VAR_MOD}")
        endif()
    else()
        message(WARNING "CTest environment setup requires CMake 3.21+ for robust ENVIRONMENT_MODIFICATION.")
    endif()
endif()
