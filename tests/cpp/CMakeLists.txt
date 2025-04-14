# tests/cpp/CMakeLists.txt
# Updated to find GTest, include backend tests conditionally, set paths/definitions,
# propagate log control options, and handle DLL copying for test discovery.
# Reordered set_tests_properties call.
# WORKAROUND: Commented out CTest environment setup block due to persistent CMake error.

# Find GoogleTest dependency (managed by Conda)
# Assumes GTest is found in the root CMakeLists.txt if not building Python bindings
if(NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
    message(STATUS "Found GoogleTest for C++ tests.")
endif()


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

# --- Apply Inherited Backend Compile Definition ---
# This variable is set in the parent CMakeLists.txt
# Ensure this check uses the variable name defined in your root CMakeLists.txt (e.g., BACKEND_DEFINITION)
if(DEFINED BACKEND_DEFINITION AND NOT BACKEND_DEFINITION STREQUAL "")
    target_compile_definitions(omnidsp_tests PRIVATE ${BACKEND_DEFINITION})
    message(STATUS "Applying inherited backend compile definition to tests: -D${BACKEND_DEFINITION}")
else()
    message(STATUS "No specific backend compile definition inherited (Stub backend active).")
endif()

# --- Propagate Log Control Definitions ---
# Check options defined in the root CMakeLists.txt
if(OMNIDSP_ENABLE_MKL_DEBUG_LOGS)
    target_compile_definitions(omnidsp_tests PRIVATE OMNIDSP_ENABLE_MKL_DEBUG_LOGS)
    message(STATUS "Enabling MKL debug logs for omnidsp_tests target.")
endif()
if(OMNIDSP_ENABLE_CQT_WARNINGS)
    target_compile_definitions(omnidsp_tests PRIVATE OMNIDSP_ENABLE_CQT_WARNINGS)
    message(STATUS "Enabling CQT Nk clamping warnings for omnidsp_tests target.")
endif()

# --- Copy Shared Library for Test Discovery (Windows) ---
# This ensures gtest_discover_tests can run the executable at build time.
if(WIN32 AND BUILD_SHARED_LIBS) # Check option defined in root CMakeLists.txt
    add_custom_command(TARGET omnidsp_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:omnidsp> # Source: the omnidsp library file (.dll)
        $<TARGET_FILE_DIR:omnidsp_tests> # Destination: directory of omnidsp_tests.exe
        COMMENT "Copying omnidsp DLL for test discovery"
        VERBATIM)
endif()

# --- Add Include Directories ---
target_include_directories(omnidsp_tests PRIVATE
    # Add path to find <OmniDSP/header.h>
    ${PROJECT_SOURCE_DIR}/include

    # Add path to find internal backend header "../../src/omnidsp/backend/backend_impl.h"
    # Adjust if your backend_impl.h is elsewhere relative to src/
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
    # COPYONLY prevents CMake from trying to configure variables inside the file
    configure_file("${REFERENCE_FILE_SOURCE}" "${REFERENCE_FILE_DESTINATION}" COPYONLY)
    message(STATUS "Configured copying of ${REFERENCE_FILE_SOURCE} to ${REFERENCE_FILE_DESTINATION} for tests.")
else()
    message(WARNING "Reference data file not found: ${REFERENCE_FILE_SOURCE}. backend_conv_test and cqt tests will likely fail.")
    # Optionally, create an empty file to prevent runtime file-not-found errors,
    # though the tests would still fail without data.
    # file(WRITE "${REFERENCE_FILE_DESTINATION}" "# Reference file missing!\n")
endif()

# --- CTest Integration ---
# Add the executable to CTest's list of tests
include(GoogleTest) # Include this to get gtest_discover_tests
gtest_discover_tests(omnidsp_tests) # This adds the test 'omnidsp_tests' to CTest

# --- CTest Environment Setup for Shared Libraries ---
# Sets PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH for CTest *runtime* execution
# This helps CTest find omnidsp.dll/so/dylib when running the tests.
# ***** MOVED THIS BLOCK AFTER gtest_discover_tests *****
# ***** WORKAROUND: Temporarily commented out due to CMake config error *****
# if(BUILD_SHARED_LIBS) # Check option defined in the root CMakeLists.txt
#     if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
#         # Use ENVIRONMENT_MODIFICATION (CMake 3.21+)
#         set(OMNIDSP_LIB_BUILD_DIR "$<TARGET_FILE_DIR:omnidsp>")
#         set(ENV_VAR_MOD "") # Default empty
#         if(UNIX AND NOT APPLE)
#             set(ENV_VAR_MOD "LD_LIBRARY_PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
#         elseif(APPLE)
#             set(ENV_VAR_MOD "DYLD_LIBRARY_PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
#         elseif(WIN32)
#             # On Windows, also ensure Conda env's Library/bin is in PATH if using MKL
#             # This is complex to do robustly here; relying on activated env is better.
#             # We primarily ensure the locally built omnidsp.dll is found.
#             set(ENV_VAR_MOD "PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
#         endif()
#         if(NOT ENV_VAR_MOD STREQUAL "")
#             message(VERBOSE "CTest: Setting ENVIRONMENT_MODIFICATION for 'omnidsp_tests': ${ENV_VAR_MOD}")
#             # Apply properties *after* the test has been discovered/added
#             set_tests_properties(omnidsp_tests PROPERTIES ENVIRONMENT_MODIFICATION "${ENV_VAR_MOD}")
#         endif()
#     else()
#         message(WARNING "CTest environment setup requires CMake 3.21+ for robust ENVIRONMENT_MODIFICATION.")
#         # Older CMake versions might require custom scripts or setting properties differently.
#     endif()
# endif()

