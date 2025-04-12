# CMakeLists.txt for OmniDSP project
# Updated to use Conda for managing CMake, pybind11, GTest, and libsamplerate dependencies

cmake_minimum_required(VERSION 3.21)

# Set project name and version (used for installation and config files)
project(OmniDSP VERSION 1.0.0 LANGUAGES CXX)

enable_testing() # Enable CTest support

option(OMNIDSP_BUILD_PYTHON_BINDINGS "Build the Python language bindings" OFF)

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

# --- Find Dependencies (Managed by Conda) ---

# Find pybind11 (Needed regardless of OMNIDSP_BUILD_PYTHON_BINDINGS for conditional logic)
# Assumes Conda environment provides pybind11.
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 version ${pybind11_VERSION} via find_package at ${pybind11_DIR}")

# --- Option to control backend preference ---
option(FFT_PREFER_ACCELERATE "Prefer Apple Accelerate if available" ${APPLE})
option(FFT_PREFER_ONEMKL "Prefer Intel oneMKL if available" OFF)

# --- Backend Detection and Source File Selection ---
# Initialize backend variables
set(BACKEND "NONE")
set(BACKEND_SOURCE "")
set(BACKEND_LINK_LIBS "") # Renamed variable
set(BACKEND_DEFINITION "") # Renamed variable, initialized here

# Prefer Accelerate on Apple unless MKL is explicitly preferred
if(APPLE AND FFT_PREFER_ACCELERATE AND NOT FFT_PREFER_ONEMKL)
    find_package(Accelerate QUIET)
    if(Accelerate_FOUND)
        message(STATUS "Found Apple Accelerate framework (via find_package).")
        set(BACKEND "ACCELERATE")
        set(BACKEND_SOURCE src/omnidsp/backend/accelerate.cpp)
        set(BACKEND_LINK_LIBS Accelerate::Accelerate) # Set renamed variable
        set(BACKEND_DEFINITION "USE_ACCELERATE") # Set renamed variable here
    else()
        # Fallback check
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            message(STATUS "Found Apple Accelerate framework (via find_library).")
            set(BACKEND "ACCELERATE")
            set(BACKEND_SOURCE src/omnidsp/backend/accelerate.cpp)
            set(BACKEND_LINK_LIBS ${ACCELERATE_FRAMEWORK}) # Set renamed variable
            set(BACKEND_DEFINITION "USE_ACCELERATE") # Set renamed variable here
        else()
            message(WARNING "Apple Accelerate framework requested but not found on Apple system.")
        endif()
        unset(ACCELERATE_FRAMEWORK CACHE) # Clean up variable
    endif()
endif()

# Check for oneMKL if not using Accelerate or if MKL is preferred
if(NOT BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
    find_package(MKL QUIET) # Assumes Conda environment provides MKL correctly.
    if(MKL_FOUND)
        message(STATUS "Found Intel oneMKL (version ${MKL_VERSION}).")
        if(NOT BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
            message(STATUS "Selecting Intel oneMKL backend.")
            set(BACKEND "ONEMKL")
            set(BACKEND_SOURCE src/omnidsp/backend/onemkl.cpp)
            set(BACKEND_DEFINITION "USE_ONEMKL") # Set renamed variable here
            # Store MKL link information
            if(TARGET MKL::MKL)
                set(BACKEND_LINK_LIBS MKL::MKL) # Set renamed variable
            elseif(MKL_LIBRARIES)
                set(BACKEND_LINK_LIBS ${MKL_LIBRARIES}) # Set renamed variable
            else()
                message(WARNING "FindMKL found MKL but did not define MKL::MKL target or MKL_LIBRARIES.")
            endif()
        else()
            message(STATUS "Intel oneMKL found, but Accelerate backend was already selected.")
        endif()
    else()
        if(NOT BACKEND STREQUAL "ACCELERATE")
            message(STATUS "Intel oneMKL not found.")
        endif()
        if(FFT_PREFER_ONEMKL)
            message(WARNING "Preferred oneMKL backend not found.")
        endif()
    endif()
endif()

# --- Select Stub Implementation if no backend was successfully selected ---
if(BACKEND STREQUAL "NONE")
    message(WARNING "No suitable backend (Accelerate or oneMKL) found or configured. Using stub implementation (will throw errors at runtime).")
    set(BACKEND_SOURCE src/omnidsp/backend/stub.cpp)
    # BACKEND_DEFINITION and BACKEND_LINK_LIBS remain empty
endif()

# Check that the selected implementation source file actually exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${BACKEND_SOURCE}")
    message(FATAL_ERROR "Selected backend implementation source file not found: ${CMAKE_CURRENT_SOURCE_DIR}/${BACKEND_SOURCE}")
endif()

message(STATUS "OmniDSP selected backend: ${BACKEND}") # Log final backend choice
message(STATUS "OmniDSP backend source: ${BACKEND_SOURCE}")
message(STATUS "OmniDSP backend definition: ${BACKEND_DEFINITION}") # Log definition
message(STATUS "OmniDSP backend link libs: ${BACKEND_LINK_LIBS}") # Log link libs

# --- Library Target Definition ---
add_library(omnidsp
    src/omnidsp/omnidsp.cpp
    src/omnidsp/cqt.cpp
    src/omnidsp/convolution.cpp
    ${BACKEND_SOURCE}
)

# Set C++ standard for the target explicitly
target_compile_features(omnidsp PRIVATE cxx_std_17)

# Define include directories needed to build and use the library
target_include_directories(omnidsp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add backend-specific compile definitions TO THE LIBRARY TARGET
if(NOT BACKEND_DEFINITION STREQUAL "")
    target_compile_definitions(omnidsp PRIVATE ${BACKEND_DEFINITION})
endif()
# Add MKL include directory needed *only* for compiling omnidsp implementation
if(BACKEND STREQUAL "ONEMKL" AND DEFINED MKL_INCLUDE_DIRS)
    target_include_directories(omnidsp PRIVATE ${MKL_INCLUDE_DIRS})
endif()

# Link omnidsp library target against its dependencies
target_link_libraries(omnidsp PUBLIC
    # Link against libsamplerate (Assuming it's found earlier and provides SampleRate::samplerate)
    # find_package(SampleRate REQUIRED) # Make sure this is present earlier
    # SampleRate::samplerate
    # Link against backend libraries (using the renamed variable)
    ${BACKEND_LINK_LIBS}
)

# =============================================================================
# Conditionally Add Subdirectories based on Build Type
# =============================================================================

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
    # --- Add Python Bindings & Python Tests Subdirectories ---
    message(STATUS "Configuring Python bindings and tests (OMNIDSP_BUILD_PYTHON_BINDINGS=ON).")

    # Add python bindings subdirectory (will link against pybind11::module)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp_py/CMakeLists.txt")
        add_subdirectory(src/omnidsp_py)
    else()
        message(WARNING "Python bindings subdirectory 'src/omnidsp_py' or its CMakeLists.txt not found, skipping Python module configuration.")
    endif()

    # Add python tests subdirectory (depends on Python bindings)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/CMakeLists.txt")
        # enable_testing() called earlier
        add_subdirectory(tests/python)
    else()
        message(STATUS "Python tests subdirectory 'tests/python' or its CMakeLists.txt not found, skipping Python test configuration.")
    endif()

else()
    # --- Add C++ Examples & C++ Tests Subdirectories ---
    message(STATUS "Configuring C++ examples and tests (OMNIDSP_BUILD_PYTHON_BINDINGS=OFF).")

    # Add C++ Examples Subdirectory
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/CMakeLists.txt")
        add_subdirectory(examples/cpp) # Process C++ examples
    else()
        message(STATUS "C++ examples subdirectory 'examples/cpp' or its CMakeLists.txt not found, skipping C++ example configuration.")
    endif()

    # Add C++ Tests Subdirectory (will link against GTest::gtest_main)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
        # enable_testing() called earlier
        add_subdirectory(tests/cpp) # Process C++ tests
    else()
        message(STATUS "C++ tests subdirectory 'tests/cpp' or its CMakeLists.txt not found, skipping C++ test configuration.")
    endif()

endif() # End conditional add_subdirectory

# =============================================================================
# Installation Rules
# =============================================================================

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
    # --- Install Python Examples ---
    # (Installation of the python module itself is handled in src/omnidsp_py/CMakeLists.txt
    # and by scikit-build-core / pip)
    message(STATUS "Configuring Python examples installation (Python bindings ON)")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/notebooks") # Check for notebooks dir
        install(DIRECTORY examples/notebooks/ # Install notebooks dir
            DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/examples_python"
            COMPONENT Examples
            FILES_MATCHING PATTERN "*.ipynb" # Include notebooks
        )
    else()
        message(STATUS "Python examples directory 'examples/notebooks' not found, skipping installation.")
    endif()
else()
    # --- Install C++ Development Files ---
    message(STATUS "Configuring C++ library installation rules (Python bindings OFF)")

    # Use a variable for the CMake config destination for consistency
    set(OMNIDSP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

    # 1. Install the public header files
    install(DIRECTORY include/OmniDSP/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # <prefix>/include/OmniDSP
        COMPONENT Devel
        FILES_MATCHING PATTERN "*.h"
    )

    # 2. Prepare package configuration files (Ensure Config.cmake.in exists)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in")
        message(WARNING "Config.cmake.in not found in project root. Cannot generate CMake package configuration files for installation.")
    else()
        # 3. Generate version file
        write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion
        )
        # 4. Generate main config file from template
        configure_package_config_file(
            Config.cmake.in
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
            INSTALL_DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
            # PATH_VARS # Add path variables here if Config.cmake.in needs them
        )

        # 5. Install the generated configuration files
        install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
            DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
            COMPONENT Devel
        )
    endif() # End check for Config.cmake.in

    # 6. Install the C++ library target and export set
    install(TARGETS omnidsp
        EXPORT OmniDSPTargets # Associate target with an export set name
        # Define destinations for different artifact types and components
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib # Static libs (.a, .lib)
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib # Shared libs (.so, .dylib)
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime # Executables or DLLs (Windows)
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # Headers associated via target_include_directories
    )

    # 7. Install the export set definitions file
    install(EXPORT OmniDSPTargets # The export set name defined above
        FILE OmniDSPTargets.cmake # The file to generate
        NAMESPACE OmniDSP:: # Namespace for imported targets
        DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}" # Install alongside config files
        COMPONENT Devel
    )

endif() # End conditional install rules
