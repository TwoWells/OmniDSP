# CMakeLists.txt for OmniDSP project
cmake_minimum_required(VERSION 3.15)

# Set project name and version (used for installation and config files)
project(OmniDSP VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add math defines globally for MSVC compiler
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
    message(STATUS "MSVC detected, adding _USE_MATH_DEFINES compile definition.")
endif()

# --- Include Standard CMake Modules ---
include(GNUInstallDirs) # For standard installation paths (CMAKE_INSTALL_*)
include(CMakePackageConfigHelpers) # For generating *Config.cmake files
include(FetchContent) # For downloading dependencies like GoogleTest

# --- Unit Testing Setup (GoogleTest) ---
enable_testing() # Enable CTest support

# Declare GoogleTest dependency using FetchContent (Live at Head)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main # Fetch the latest commit from the main branch
)

# Configure GoogleTest: Disable installation of gtest itself when building OmniDSP
set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of googletest itself")
set(BUILD_GMOCK OFF CACHE BOOL "Disable building gmock") # Only need gtest

# Make GoogleTest available (downloads and prepares its build targets)
FetchContent_MakeAvailable(googletest)

# --- Option to control backend preference ---
# Default ON for Accelerate on Apple, OFF otherwise
option(FFT_PREFER_ACCELERATE "Prefer Apple Accelerate if available" ${APPLE})
option(FFT_PREFER_ONEMKL "Prefer Intel oneMKL if available" OFF)

# --- Backend Detection and Source File Selection ---
set(FFT_BACKEND "NONE")
set(FFT_IMPL_SOURCE "") # Variable to hold the platform-specific implementation file

# Prefer Accelerate on Apple unless MKL is explicitly preferred
if(APPLE AND FFT_PREFER_ACCELERATE AND NOT FFT_PREFER_ONEMKL)
    # Use find_package for Accelerate for better target handling if available
    find_package(Accelerate QUIET)
    if(Accelerate_FOUND)
        message(STATUS "Found Apple Accelerate framework (via find_package).")
        set(FFT_BACKEND "ACCELERATE")
        set(FFT_IMPL_SOURCE src/fft_impl_accelerate.cpp)
    else()
        # Fallback to find_library if find_package doesn't work
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            message(STATUS "Found Apple Accelerate framework (via find_library).")
            set(FFT_BACKEND "ACCELERATE")
            set(FFT_IMPL_SOURCE src/fft_impl_accelerate.cpp)
        else()
            message(WARNING "Apple Accelerate framework requested but not found on Apple system.")
        endif()
        unset(ACCELERATE_FRAMEWORK CACHE) # Clean up variable
    endif()
endif()

# Check for oneMKL if not using Accelerate or if MKL is preferred
if(NOT FFT_BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
    # Use find_package(MKL) - Requires environment setup (e.g., MKLROOT or setvars script)
    find_package(MKL QUIET) # Use QUIET initially

    if(MKL_FOUND)
        message(STATUS "Found Intel oneMKL.")
        # If Accelerate wasn't chosen, or MKL is preferred, use MKL
        if(NOT FFT_BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
            message(STATUS "Selecting Intel oneMKL backend.")
            set(FFT_BACKEND "ONEMKL")
            set(FFT_IMPL_SOURCE src/fft_impl_onemkl.cpp)
        else()
            message(STATUS "Intel oneMKL found, but Accelerate backend was already selected.")
        endif()
    else()
        # If MKL wasn't found AND Accelerate wasn't chosen, report status
        if(NOT FFT_BACKEND STREQUAL "ACCELERATE")
            message(STATUS "Intel oneMKL not found.")
        endif()
        if(FFT_PREFER_ONEMKL)
            message(WARNING "Preferred oneMKL backend not found.")
        endif()
    endif()
endif()

# --- Select Stub Implementation if no backend was successfully selected ---
if(FFT_BACKEND STREQUAL "NONE")
    message(WARNING "No suitable FFT backend (Accelerate or oneMKL) found or configured. Using stub implementation (will throw errors at runtime).")
    set(FFT_IMPL_SOURCE src/fft_impl_stub.cpp)
endif()

# Check that the selected implementation source file actually exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${FFT_IMPL_SOURCE}")
    message(FATAL_ERROR "Selected FFT implementation source file not found: ${CMAKE_CURRENT_SOURCE_DIR}/${FFT_IMPL_SOURCE}")
endif()


# --- Library Target Definition ---
add_library(omnidsp
    src/omnidsp.cpp # Platform-independent wrapper/convenience functions
    ${FFT_IMPL_SOURCE} # The selected platform-specific implementation
)
# Set C++ standard for the target explicitly (good practice)
target_compile_features(omnidsp PRIVATE cxx_std_17)

# Define include directories needed to build and use the library
target_include_directories(omnidsp PUBLIC
    # Needed when building the library itself from the source tree
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # Exported for users consuming the installed library via $<INSTALL_INTERFACE:...>
    $<INSTALL_INTERFACE:include> # Corresponds to <prefix>/include
)

# --- Apply backend-specific settings to the library target ---
if(FFT_BACKEND STREQUAL "ACCELERATE")
    if(NOT Accelerate_FOUND)
        # Ensure Accelerate was actually found if we selected it via find_library fallback
        find_package(Accelerate REQUIRED)
    endif()
    # Link publicly so consuming targets inherit the dependency if OmniDSP is static
    target_link_libraries(omnidsp PUBLIC Accelerate::Accelerate)
    # This define controls conditional compilation within the source code
    target_compile_definitions(omnidsp PRIVATE USE_ACCELERATE)

elseif(FFT_BACKEND STREQUAL "ONEMKL")
    if(NOT MKL_FOUND)
        find_package(MKL REQUIRED) # Should have been found earlier, but double-check
    endif()
    # Add MKL include directory needed *only* for compiling omnidsp implementation
    target_include_directories(omnidsp PRIVATE ${MKL_INCLUDE_DIRS})
    # Link publicly so consuming targets inherit the MKL dependency
    if(TARGET MKL::MKL)
        target_link_libraries(omnidsp PUBLIC MKL::MKL)
    elseif(MKL_LIBRARIES)
        # Fallback if imported target isn't defined by FindMKL
        target_link_libraries(omnidsp PUBLIC ${MKL_LIBRARIES})
    else()
        message(FATAL_ERROR "FindMKL found MKL but did not define MKL::MKL target or MKL_LIBRARIES. Cannot link OmniDSP.")
    endif()
    # This define controls conditional compilation within the source code
    target_compile_definitions(omnidsp PRIVATE USE_ONEMKL)

    # elseif(FFT_BACKEND STREQUAL "NONE")
    # No specific link libraries or compile definitions needed for the stub backend
endif()


# --- Example Executable Target ---
# This is typically not installed, just used for testing/demonstration within this build
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/main.cpp")
    add_executable(omnidsp_example examples/main.cpp)
    # Link the example privately against our library target
    # No need to link example against MKL/Accelerate directly, happens via omnidsp target
    target_link_libraries(omnidsp_example PRIVATE omnidsp)
    # If example needs includes directly (it shouldn't if omnidsp includes are PUBLIC)
    # target_include_directories(omnidsp_example PRIVATE include)
endif()


# --- Tests Subdirectory ---
# Add the 'tests' subdirectory which contains its own CMakeLists.txt
# defining the 'omnidsp_tests' executable and linking it against 'omnidsp' and 'gtest_main'
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
else()
    message(STATUS "Tests subdirectory not found, skipping test configuration.")
endif()


# =============================================================================
# Installation Rules
# =============================================================================

# Use a variable for the CMake config destination for consistency
set(OMNIDSP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# 1. Install the public header file(s)
# Destination: <prefix>/include/OmniDSP/omnidsp.h
install(FILES include/OmniDSP/omnidsp.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
    COMPONENT Devel # Component grouping (optional)
)

# 2. Prepare package configuration files for find_package(OmniDSP)
# Create OmniDSPConfigVersion.cmake based on project VERSION
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion # Or SameMajorVersion, ExactVersion, etc.
)

# Create OmniDSPConfig.cmake from the template file Config.cmake.in
# Assumes Config.cmake.in is in the same directory as this CMakeLists.txt
configure_package_config_file(
    Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
    INSTALL_DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    # PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR # Usually not needed if using relative paths
)

# 3. Install the library target and associate it with an export set
install(TARGETS omnidsp
    EXPORT OmniDSPTargets # Associate with export set named OmniDSPTargets
    # Destinations use variables from GNUInstallDirs
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib # Static libs (.a, .lib)
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib # Shared libs (.so) / Import libs (.lib)
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime # DLLs (Windows) / Executables
    # This exports the include directories associated with the target's INSTALL_INTERFACE
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT Devel
)

# 4. Install the export set definitions file (OmniDSPTargets.cmake)
# This file allows CMake to create the imported target OmniDSP::omnidsp
install(EXPORT OmniDSPTargets
    FILE OmniDSPTargets.cmake
    NAMESPACE OmniDSP:: # Defines the namespace for imported targets
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    COMPONENT Devel
)

# 5. Install the generated configuration files (OmniDSPConfig.cmake, OmniDSPConfigVersion.cmake)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    COMPONENT Devel
)
