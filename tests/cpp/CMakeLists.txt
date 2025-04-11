# tests/cpp/CMakeLists.txt
# Updated to find its own GTest dependency

# Find the GoogleTest dependency (provided by Conda environment)
find_package(GTest REQUIRED)
message(STATUS "Found GoogleTest for C++ tests.")

# Automatically find all .cpp source files in this directory
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp) # Use CONFIGURE_DEPENDS for robustness

# Check if any test source files were found
if(NOT TEST_SOURCES)
    message(WARNING "No C++ test source files (.cpp) found in tests/cpp/ directory.")
    return() # Exit early if no sources, nothing to build
endif()

message(STATUS "Found C++ test sources: ${TEST_SOURCES}")

# Create the test executable using the automatically discovered source files
add_executable(omnidsp_tests ${TEST_SOURCES}) # Use the variable here

# Link the test executable against GoogleTest and the OmniDSP library
# Assumes GTest includes needed headers via target_link_libraries
target_link_libraries(omnidsp_tests PRIVATE
    GTest::gtest_main  # Links GTest and provides main()
    omnidsp            # Link against the library target built in the parent scope
)

# Add the executable to CTest's list of tests
# gtest_discover_tests requires enable_testing() in a parent scope (root CMakeLists.txt)
include(GoogleTest) # Include this to get gtest_discover_tests
gtest_discover_tests(omnidsp_tests)

# --- CTest Environment Setup for Shared Libraries ---
# This ensures the tests load the omnidsp shared library from the build
# directory, not a potentially installed version. Only needed if BUILD_SHARED_LIBS is ON.
if(BUILD_SHARED_LIBS) # Check the option defined in the root CMakeLists.txt
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        # Use ENVIRONMENT_MODIFICATION (CMake 3.21+) for cleaner path manipulation

        # Get the directory containing the built omnidsp library artifact.
        set(OMNIDSP_LIB_BUILD_DIR "$<TARGET_FILE_DIR:omnidsp>")

        # Determine the platform-specific environment variable and modification string
        if(UNIX AND NOT APPLE) # Linux
            set(ENV_VAR_MOD "LD_LIBRARY_PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        elseif(APPLE) # macOS
            set(ENV_VAR_MOD "DYLD_LIBRARY_PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        elseif(WIN32) # Windows
             set(ENV_VAR_MOD "PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        else()
            set(ENV_VAR_MOD "") # Unknown platform
        endif()

        # Apply the modification to the test properties if a modification string was set
        if(NOT ENV_VAR_MOD STREQUAL "")
            message(VERBOSE "CTest: Setting ENVIRONMENT_MODIFICATION for test 'omnidsp_tests': ${ENV_VAR_MOD}")
            set_tests_properties(omnidsp_tests PROPERTIES ENVIRONMENT_MODIFICATION "${ENV_VAR_MOD}")
        endif()

    else()
        # Fallback for CMake < 3.21 (less robust)
        message(WARNING "CTest environment setup for shared libraries requires CMake 3.21+ for robust ENVIRONMENT_MODIFICATION. Tests might load installed library.")
    endif()
endif()