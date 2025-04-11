# CMakeLists.txt for OmniDSP project
# Updated to use Conda for managing CMake, pybind11, and GTest dependencies

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

# --- Find pybind11 Dependency (Managed by Conda) ---
# This replaces FetchContent. Assumes Conda environment provides pybind11.
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 version ${pybind11_VERSION} via find_package at ${pybind11_DIR}")

# --- Option to control backend preference ---
option(FFT_PREFER_ACCELERATE "Prefer Apple Accelerate if available" ${APPLE})
option(FFT_PREFER_ONEMKL "Prefer Intel oneMKL if available" OFF)

# --- Backend Detection and Source File Selection ---
set(BACKEND "NONE")
set(BACKEND_SOURCE "") # Variable to hold the platform-specific implementation file

# Prefer Accelerate on Apple unless MKL is explicitly preferred
if(APPLE AND FFT_PREFER_ACCELERATE AND NOT FFT_PREFER_ONEMKL)
    # Use find_package for Accelerate (OS-provided dependency)
    find_package(Accelerate QUIET)
    if(Accelerate_FOUND)
        message(STATUS "Found Apple Accelerate framework (via find_package).")
        set(BACKEND "ACCELERATE")
        set(BACKEND_SOURCE src/omnidsp/backend/accelerate.cpp)
    else()
        # Fallback check (less common now with find_package)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            message(STATUS "Found Apple Accelerate framework (via find_library).")
            set(BACKEND "ACCELERATE")
            set(BACKEND_SOURCE src/omnidsp/backend/accelerate.cpp)
        else()
            message(WARNING "Apple Accelerate framework requested but not found on Apple system.")
        endif()
        unset(ACCELERATE_FRAMEWORK CACHE) # Clean up variable
    endif()
endif()

# Check for oneMKL if not using Accelerate or if MKL is preferred
if(NOT BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
    # Use find_package(MKL). Assumes Conda environment provides MKL correctly.
    find_package(MKL QUIET)

    if(MKL_FOUND)
        message(STATUS "Found Intel oneMKL (version ${MKL_VERSION}).")
        if(NOT BACKEND STREQUAL "ACCELERATE" OR FFT_PREFER_ONEMKL)
            message(STATUS "Selecting Intel oneMKL backend.")
            set(BACKEND "ONEMKL")
            set(BACKEND_SOURCE src/omnidsp/backend/onemkl.cpp)
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
endif()

# Check that the selected implementation source file actually exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${BACKEND_SOURCE}")
    message(FATAL_ERROR "Selected backend implementation source file not found: ${CMAKE_CURRENT_SOURCE_DIR}/${BACKEND_SOURCE}")
endif()


# --- Library Target Definition ---
add_library(omnidsp
    src/omnidsp/omnidsp.cpp # Platform-independent wrapper/convenience functions
    src/omnidsp/cqt.cpp     # CQT implementation
    # Add other .cpp files directly in src/ here if any

    ${BACKEND_SOURCE}       # The selected platform-specific backend implementation
)

# Set C++ standard for the target explicitly
target_compile_features(omnidsp PRIVATE cxx_std_17)

# Define include directories needed to build and use the library
target_include_directories(omnidsp PUBLIC
    # Needed when building the library itself from the source tree
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # Exported for users consuming the installed library via $<INSTALL_INTERFACE:...>
    $<INSTALL_INTERFACE:include> # Corresponds to <prefix>/include
)
# Ensure pybind11 headers are available if needed by omnidsp target directly
# (though usually only needed by the python binding module)
# target_link_libraries(omnidsp PRIVATE pybind11::headers) # Usually not needed here


# =============================================================================
# Apply backend-specific settings to the library target ---
# =============================================================================
if(BACKEND STREQUAL "ACCELERATE")
    if(NOT Accelerate_FOUND)
        find_package(Accelerate REQUIRED) # Should have been found earlier
    endif()
    # Link publicly so consuming targets inherit the dependency if OmniDSP is static
    target_link_libraries(omnidsp PUBLIC Accelerate::Accelerate)
    target_compile_definitions(omnidsp PRIVATE USE_ACCELERATE)

elseif(BACKEND STREQUAL "ONEMKL")
    if(NOT MKL_FOUND)
        find_package(MKL REQUIRED) # Should have been found earlier
    endif()
    # Add MKL include directory needed *only* for compiling omnidsp implementation
    # find_package(MKL) usually sets MKL_INCLUDE_DIRS
    target_include_directories(omnidsp PRIVATE ${MKL_INCLUDE_DIRS})
    # Link publicly so consuming targets inherit the MKL dependency
    # find_package(MKL) usually defines MKL::MKL imported target
    if(TARGET MKL::MKL)
        target_link_libraries(omnidsp PUBLIC MKL::MKL)
    elseif(MKL_LIBRARIES)
        # Fallback if imported target isn't defined by FindMKL
        target_link_libraries(omnidsp PUBLIC ${MKL_LIBRARIES})
    else()
        message(FATAL_ERROR "FindMKL found MKL but did not define MKL::MKL target or MKL_LIBRARIES. Cannot link OmniDSP.")
    endif()
    target_compile_definitions(omnidsp PRIVATE USE_ONEMKL)

    # elseif(BACKEND STREQUAL "NONE")
    # No specific link libraries or compile definitions needed for the stub backend
endif()

# =============================================================================
# Conditionally Add Subdirectories based on Build Type ---
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
        enable_testing() # Enable CTest for Python tests too
        add_subdirectory(tests/python)
    else()
        message(STATUS "Python tests subdirectory 'tests/python' or its CMakeLists.txt not found, skipping Python test configuration.")
    endif()

else()
    # --- Add C++ Examples & C++ Tests Subdirectories ---
    message(STATUS "Configuring C++ examples and tests (OMNIDSP_BUILD_PYTHON_BINDINGS=OFF).")

    # Add C++ Examples Subdirectory
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/CMakeLists.txt")
        add_subdirectory(examples/cpp)
    else()
        message(STATUS "C++ examples subdirectory 'examples/cpp' or its CMakeLists.txt not found, skipping C++ example configuration.")
    endif()

    # Add C++ Tests Subdirectory (will link against GTest::gtest_main)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
        add_subdirectory(tests/cpp)
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
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/python")
        install(DIRECTORY examples/python/
            DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/examples_python"
            COMPONENT Examples
            FILES_MATCHING PATTERN "*.py" PATTERN "*.txt" PATTERN "*.ipynb" # Include notebooks
        )
    else()
        message(STATUS "Python examples directory 'examples/python' not found, skipping installation.")
    endif()
else()
    # --- Install C++ Development Files ---
    message(STATUS "Configuring C++ library installation rules (Python bindings OFF)")

    # Use a variable for the CMake config destination for consistency
    set(OMNIDSP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

    # 1. Install the public header file(s)
    install(DIRECTORY include/OmniDSP/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT Devel
        FILES_MATCHING PATTERN "*.h"
    )

    # 2. Prepare package configuration files (Ensure Config.cmake.in exists)
    # Assumes Config.cmake.in is in the root project directory
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in")
        message(WARNING "Config.cmake.in not found in project root. Cannot generate CMake package configuration files for installation.")
    else()
        write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion
        )
        configure_package_config_file(
            Config.cmake.in
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
            INSTALL_DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
        )

        # 5. Install the generated configuration files
        install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
            DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
            COMPONENT Devel
        )
    endif() # End check for Config.cmake.in

    # 3. Install the C++ library target
    install(TARGETS omnidsp
        EXPORT OmniDSPTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT Devel # Public headers via target properties
    )

    # 4. Install the export set definitions file
    install(EXPORT OmniDSPTargets
        FILE OmniDSPTargets.cmake
        NAMESPACE OmniDSP::
        DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
        COMPONENT Devel
    )

    # 6. Install C++ Examples (Optional)
    # If examples/cpp/CMakeLists.txt contains install rules, they will be processed here.
    # Ensure they use install(TARGETS ...) for executables with appropriate DESTINATION and COMPONENT.
endif() # End conditional install rules