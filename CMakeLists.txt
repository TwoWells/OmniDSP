# CMakeLists.txt for OmniDSP project
cmake_minimum_required(VERSION 3.15)

# Set project name and version (used for installation and config files)
project(OmniDSP VERSION 1.0.0 LANGUAGES CXX)

option(OMNIDSP_BUILD_PYTHON_BINDINGS "Build the Python language bindings" OFF)

# Show FetchContent progress
set(FETCHCONTENT_QUIET FALSE)

# Enable generation of compile_commands.json for IDEs/tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
include(FetchContent) # For downloading dependencies like pybind11 and GoogleTest

# --- pybind11 Dependency (FetchContent) ---
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
# For newer CMake versions (3.14+), pybind11 uses INTERFACE targets,
# so setting PYBIND11_INSTALL is often not needed when using FetchContent directly.
# set(PYBIND11_INSTALL OFF CACHE BOOL "Don't install pybind11") # Optional
FetchContent_MakeAvailable(pybind11)

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
option(FFT_PREFER_ACCELERATE "Prefer Apple Accelerate if available" ${APPLE}) # Keep FFT_ prefix for user option? Or change to BACKEND_? Let's keep for now.
option(FFT_PREFER_ONEMKL "Prefer Intel oneMKL if available" OFF) # Keep FFT_ prefix for user option?

# --- Backend Detection and Source File Selection ---
# RENAMED Variables
set(BACKEND "NONE")
set(BACKEND_SOURCE "") # Variable to hold the platform-specific implementation file

# Prefer Accelerate on Apple unless MKL is explicitly preferred
if(APPLE AND FFT_PREFER_ACCELERATE AND NOT FFT_PREFER_ONEMKL)
    # Use find_package for Accelerate for better target handling if available
    find_package(Accelerate QUIET)
    if(Accelerate_FOUND)
        message(STATUS "Found Apple Accelerate framework (via find_package).")
        set(BACKEND "ACCELERATE")
        # UPDATED Path and Name
        set(BACKEND_SOURCE src/backend/accelerate.cpp)
    else()
        # Fallback to find_library if find_package doesn't work
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            message(STATUS "Found Apple Accelerate framework (via find_library).")
            set(BACKEND "ACCELERATE")
            # UPDATED Path and Name
            set(BACKEND_SOURCE src/backend/accelerate.cpp)
        else()
            message(WARNING "Apple Accelerate framework requested but not found on Apple system.")
        endif()
        unset(ACCELERATE_FRAMEWORK CACHE) # Clean up variable
    endif()
endif()

# Check for oneMKL if not using Accelerate or if MKL is preferred
# UPDATED Variable Name Check
if(NOT BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
    # Use find_package(MKL) - Requires environment setup (e.g., MKLROOT or setvars script)
    find_package(MKL QUIET) # Use QUIET initially

    if(MKL_FOUND)
        message(STATUS "Found Intel oneMKL.")
        # If Accelerate wasn't chosen, or MKL is preferred, use MKL
        # UPDATED Variable Name Check
        if(NOT BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
            message(STATUS "Selecting Intel oneMKL backend.")
            # UPDATED Variable Name
            set(BACKEND "ONEMKL")
            # UPDATED Path and Name
            set(BACKEND_SOURCE src/backend/onemkl.cpp)
        else()
            message(STATUS "Intel oneMKL found, but Accelerate backend was already selected.")
        endif()
    else()
        # If MKL wasn't found AND Accelerate wasn't chosen, report status
        # UPDATED Variable Name Check
        if(NOT BACKEND STREQUAL "ACCELERATE")
            message(STATUS "Intel oneMKL not found.")
        endif()
        if(FFT_PREFER_ONEMKL)
            message(WARNING "Preferred oneMKL backend not found.")
        endif()
    endif()
endif()

# --- Select Stub Implementation if no backend was successfully selected ---
# UPDATED Variable Name Check
if(BACKEND STREQUAL "NONE")
    message(WARNING "No suitable backend (Accelerate or oneMKL) found or configured. Using stub implementation (will throw errors at runtime).")
    # UPDATED Path and Name
    set(BACKEND_SOURCE src/backend/stub.cpp)
endif()

# Check that the selected implementation source file actually exists
# UPDATED Variable Name
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${BACKEND_SOURCE}")
    # UPDATED Variable Name
    message(FATAL_ERROR "Selected backend implementation source file not found: ${CMAKE_CURRENT_SOURCE_DIR}/${BACKEND_SOURCE}")
endif()


# --- Library Target Definition ---
# Use explicit source file listing (Recommended)
add_library(omnidsp
    src/omnidsp.cpp # Platform-independent wrapper/convenience functions
    src/cqt.cpp # CQT implementation
    # Add other .cpp files directly in src/ here if any

    ${BACKEND_SOURCE} # The selected platform-specific backend implementation
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
# UPDATED Variable Name Check
if(BACKEND STREQUAL "ACCELERATE")
    if(NOT Accelerate_FOUND)
        # Ensure Accelerate was actually found if we selected it via find_library fallback
        find_package(Accelerate REQUIRED)
    endif()
    # Link publicly so consuming targets inherit the dependency if OmniDSP is static
    target_link_libraries(omnidsp PUBLIC Accelerate::Accelerate)
    # This define controls conditional compilation within the source code
    target_compile_definitions(omnidsp PRIVATE USE_ACCELERATE)

    # UPDATED Variable Name Check
elseif(BACKEND STREQUAL "ONEMKL")
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

    # elseif(BACKEND STREQUAL "NONE")
    # No specific link libraries or compile definitions needed for the stub backend
endif()


# --- C++ Examples Subdirectory ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/CMakeLists.txt")
    add_subdirectory(examples/cpp)
else()
    message(STATUS "C++ examples subdirectory 'examples/cpp' or its CMakeLists.txt not found, skipping C++ example configuration.")
endif()


# --- Tests Subdirectory ---
# Add the 'tests' subdirectory which contains its own CMakeLists.txt
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
else()
    message(STATUS "Tests subdirectory not found, skipping test configuration.")
endif()


# --- Python Bindings & Python Tests Subdirectories ---
# Add the python and tests/python subdirectories only if the option is enabled.
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
    message(STATUS "Python bindings requested (OMNIDSP_BUILD_PYTHON_BINDINGS=ON).")

    # Add python bindings subdirectory
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt")
        add_subdirectory(python)
    else()
        message(WARNING "Python bindings subdirectory 'python' or its CMakeLists.txt not found, skipping Python module configuration.")
    endif()

    # Add python tests subdirectory (depends on Python bindings)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/CMakeLists.txt")
        add_subdirectory(tests/python)
    else()
        # This message is optional, depends if you always expect python tests
        message(STATUS "Python tests subdirectory 'tests/python' or its CMakeLists.txt not found, skipping Python test configuration.")
    endif()

else()
     message(STATUS "Skipping Python bindings and Python tests configuration (OMNIDSP_BUILD_PYTHON_BINDINGS=OFF).")
endif()


# =============================================================================
# Installation Rules
# =============================================================================

# Use a variable for the CMake config destination for consistency
set(OMNIDSP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# 1. Install the public header file(s) using install(DIRECTORY)
# This automatically handles all *.h files in include/OmniDSP/
install(DIRECTORY include/OmniDSP/ # Source directory relative to CMakeLists.txt
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # Destination directory relative to install prefix
    COMPONENT Devel # Installation component group
    FILES_MATCHING PATTERN "*.h" # Ensures only files ending in .h are installed
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
    # PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR # Usually not needed
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

# --- Install Python Examples ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/python")
    install(DIRECTORY examples/python/ # <--- Use the new path here
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/examples_python" # e.g., <prefix>/share/OmniDSP/examples_python
        COMPONENT Examples
        FILES_MATCHING PATTERN "*.py" PATTERN "*.txt" # Optional: filter specific files
    )
    message(STATUS "Configuring installation for Python examples directory.")
else()
    message(STATUS "Python examples directory 'examples/python' not found, skipping installation.")
endif()
