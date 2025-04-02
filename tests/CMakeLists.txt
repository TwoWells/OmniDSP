# tests/CMakeLists.txt

# Create the test executable
add_executable(omnidsp_tests
    fft.cpp
    ifft.cpp
    rfft.cpp
    irfft.cpp
    window.cpp
)

# Link the test executable against GoogleTest and the OmniDSP library
target_link_libraries(omnidsp_tests PRIVATE
    gtest_main # Provides main() function and links gtest
    omnidsp    # Link against the library target built in the parent scope
)

# Add the executable to CTest's list of tests
include(GoogleTest)
gtest_discover_tests(omnidsp_tests)

# --- Configure CTest Environment for Shared Libraries ---
# This ensures the tests load the omnidsp shared library from the build
# directory, not a potentially installed version. Only needed if BUILD_SHARED_LIBS is ON.

if(BUILD_SHARED_LIBS) # Check the option defined in the root CMakeLists.txt
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        # Use ENVIRONMENT_MODIFICATION (CMake 3.21+) for cleaner path manipulation

        # Get the directory containing the built omnidsp library artifact.
        # $<TARGET_FILE_DIR:omnidsp> points to the directory of omnidsp.lib/.a/.so/.dylib
        # For Windows DLLs, they might be elsewhere (CMAKE_RUNTIME_OUTPUT_DIRECTORY).
        # However, build systems like VS often place DLLs next to EXEs or where
        # the linker finds them. $<TARGET_FILE_DIR:omnidsp> often works for finding
        # the import lib's directory, which might be sufficient if DLL is copied nearby
        # or if PATH logic below handles it. A more complex approach might query
        # CMAKE_RUNTIME_OUTPUT_DIRECTORY or target properties if needed.
        set(OMNIDSP_LIB_BUILD_DIR "$<TARGET_FILE_DIR:omnidsp>")

        # Determine the platform-specific environment variable and modification string
        if(UNIX AND NOT APPLE) # Linux
            set(ENV_VAR_MOD "LD_LIBRARY_PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        elseif(APPLE) # macOS
            set(ENV_VAR_MOD "DYLD_LIBRARY_PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        elseif(WIN32) # Windows
            # Prepending to PATH is needed for DLLs. PATH uses semicolons.
            set(ENV_VAR_MOD "PATH=PREPEND:${OMNIDSP_LIB_BUILD_DIR}")
        else()
            set(ENV_VAR_MOD "") # Unknown platform
        endif()

        # Apply the modification to the test properties if a modification string was set
        if(NOT ENV_VAR_MOD STREQUAL "")
            message(STATUS "CTest: Setting ENVIRONMENT_MODIFICATION for test 'omnidsp_tests': ${ENV_VAR_MOD}")
            set_tests_properties(omnidsp_tests PROPERTIES ENVIRONMENT_MODIFICATION "${ENV_VAR_MOD}")
        endif()

    else()
        # Fallback for CMake < 3.21 (less robust)
        message(WARNING "CTest environment setup for shared libraries requires CMake 3.21+ for robust ENVIRONMENT_MODIFICATION. Tests might load installed library.")
        # Manually setting ENVIRONMENT requires getting the existing variable value,
        # which is difficult to do reliably and portably within CMake configuration.
        # It's better to require CMake 3.21+ or use static linking for testing.
    endif()
endif()