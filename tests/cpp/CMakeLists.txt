# tests/cpp/CMakeLists.txt

# Automatically find all .cpp source files in this directory
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp) # Use CONFIGURE_DEPENDS for robustness

# Check if any test source files were found
if(NOT TEST_SOURCES)
    message(WARNING "No C++ test source files (.cpp) found in tests/ directory.")
    return() # Exit early if no sources, nothing to build
endif()

message(STATUS "Found C++ test sources: ${TEST_SOURCES}")

# Create the test executable using the automatically discovered source files
add_executable(omnidsp_tests ${TEST_SOURCES}) # Use the variable here

# Link the test executable against GoogleTest and the OmniDSP library
target_link_libraries(omnidsp_tests PRIVATE
    gtest_main # Provides main() function and links gtest
    omnidsp # Link against the library target built in the parent scope
)

# --- Copy MKL DLLs for Test Execution (Windows) ---
if(WIN32 AND MKL_FOUND)
    message(STATUS "Tests: MKL backend detected, configuring DLL copy for omnidsp_tests.")

    # --- Start: More Portable MKL/OMP DLL Finding Logic ---
    set(MKL_BIN_HINTS "")
    set(OMP_BIN_HINTS "$ENV{PATH}") # Prioritize system PATH to find OpenMP DLL
    set(MKL_RELATIVE_BIN_PATHS "latest/bin/intel64" "latest/bin" "bin/intel64" "bin")
    set(OMP_RELATIVE_BIN_PATHS "latest/bin") # OpenMP DLL often directly in compiler/latest/bin

    # Check MKLROOT environment variable (primarily for MKL DLLs)
    if(DEFINED ENV{MKLROOT})
        foreach(REL_PATH ${MKL_RELATIVE_BIN_PATHS})
            list(APPEND MKL_BIN_HINTS "$ENV{MKLROOT}/${REL_PATH}")
        endforeach()
        message(STATUS "Tests: Added hints based on MKLROOT: $ENV{MKLROOT}")
    endif()

    # Check ONEAPI_ROOT environment variable (useful for both MKL and Compiler/OpenMP)
    if(DEFINED ENV{ONEAPI_ROOT})
        # Add hints for MKL relative to ONEAPI_ROOT
        foreach(REL_PATH ${MKL_RELATIVE_BIN_PATHS})
            list(APPEND MKL_BIN_HINTS "$ENV{ONEAPI_ROOT}/mkl/${REL_PATH}")
        endforeach()
        # Add hints for OpenMP/Compiler relative to ONEAPI_ROOT
        foreach(REL_PATH ${OMP_RELATIVE_BIN_PATHS})
            list(APPEND OMP_BIN_HINTS "$ENV{ONEAPI_ROOT}/compiler/${REL_PATH}") # <--- Key Change
        endforeach()
        message(STATUS "Tests: Added hints based on ONEAPI_ROOT: $ENV{ONEAPI_ROOT}")
    else()
        message(WARNING "Tests: ONEAPI_ROOT environment variable not found. DLL search may be less reliable.")
    endif()

    # Add MKL_RUNTIME_LIBRARY_DIRS if set (unlikely based on warnings)
    if(MKL_RUNTIME_LIBRARY_DIRS)
        list(APPEND MKL_BIN_HINTS ${MKL_RUNTIME_LIBRARY_DIRS})
        list(APPEND OMP_BIN_HINTS ${MKL_RUNTIME_LIBRARY_DIRS}) # Add here too? Might not be relevant.
    endif()
    # Always include the system PATH as a fallback hint for both
    list(APPEND MKL_BIN_HINTS "$ENV{PATH}")
    list(APPEND OMP_BIN_HINTS "$ENV{PATH}")

    message(STATUS "Tests: Searching for MKL runtime DLL with hints: ${MKL_BIN_HINTS}")
    find_file(MKL_RT_DLL_PATH NAMES mkl_rt.2.dll mkl_rt.1.dll mkl_rt.dll
        HINTS ${MKL_BIN_HINTS} NO_DEFAULT_PATH)

    message(STATUS "Tests: Searching for OpenMP runtime DLL with hints: ${OMP_BIN_HINTS}")
    find_file(IOMP5_DLL_PATH NAMES libiomp5md.dll
        HINTS ${OMP_BIN_HINTS} NO_DEFAULT_PATH)

    # Collect found DLLs
    set(MKL_DLLS_TO_COPY "")
    if(MKL_RT_DLL_PATH)
        list(APPEND MKL_DLLS_TO_COPY "${MKL_RT_DLL_PATH}")
        message(STATUS "Tests: Found MKL runtime DLL: ${MKL_RT_DLL_PATH}")
    else()
        message(WARNING "Tests: Could not find mkl_rt DLL using hints ${MKL_BIN_HINTS}. Test execution might fail.")
    endif()

    if(IOMP5_DLL_PATH)
        list(APPEND MKL_DLLS_TO_COPY "${IOMP5_DLL_PATH}")
        message(STATUS "Tests: Found OpenMP runtime DLL: ${IOMP5_DLL_PATH}")
    else()
        message(WARNING "Tests: Could not find libiomp5md.dll using hints ${OMP_BIN_HINTS}. Test execution might fail if OpenMP is needed.")
    endif()
    # --- End: More Portable MKL/OMP DLL Finding Logic ---

    # Use add_custom_command to copy DLLs if found
    if(MKL_DLLS_TO_COPY)
        # ... (add_custom_command remains the same) ...
        add_custom_command(
            TARGET omnidsp_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MKL_DLLS_TO_COPY}
            $<TARGET_FILE_DIR:omnidsp_tests>
            COMMENT "Copying MKL DLLs for test execution: ${MKL_DLLS_TO_COPY}"
            VERBATIM
        )
        message(STATUS "Tests: Added custom command to copy MKL DLLs to $<TARGET_FILE_DIR:omnidsp_tests>")
    else()
        message(WARNING "Tests: No MKL/OpenMP runtime DLLs found to copy for test execution.")
    endif()
elseif(WIN32)
    message(STATUS "Tests: MKL backend not selected or MKL not found, skipping MKL DLL copy step.")
endif()

# Add the executable to CTest's list of tests
include(GoogleTest)
# This command runs the test executable, so DLLs must be present
gtest_discover_tests(omnidsp_tests)

# --- Configure CTest Environment for Shared Libraries ---
# This ensures the tests load the omnidsp shared library from the build
# directory, not a potentially installed version. Only needed if BUILD_SHARED_LIBS is ON.
if(BUILD_SHARED_LIBS) # Check the option defined in the root CMakeLists.txt
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        # Use ENVIRONMENT_MODIFICATION (CMake 3.21+) for cleaner path manipulation

        # Get the directory containing the built omnidsp library artifact.
        # $<TARGET_FILE_DIR:omnidsp> points to the directory of omnidsp.lib/.a/.so/.dylib
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
            message(STATUS "CTest: Setting ENVIRONMENT_MODIFICATION for test 'omnidsp_tests': ${ENV_VAR_MOD}")
            set_tests_properties(omnidsp_tests PROPERTIES ENVIRONMENT_MODIFICATION "${ENV_VAR_MOD}")
        endif()

    else()
        # Fallback for CMake < 3.21 (less robust)
        message(WARNING "CTest environment setup for shared libraries requires CMake 3.21+ for robust ENVIRONMENT_MODIFICATION. Tests might load installed library.")
    endif()
endif()
