# CMakeLists.txt for OmniDSP project Updated to use Conda for managing CMake,
# pybind11, GTest, and MKL/IPP dependencies Includes options for controlling
# debug logs and CMP0148 policy. Includes GenerateExportHeader for Windows DLL
# builds with PUBLIC include fix. Uses BUILD_INTERFACE generator expression for
# build directory include path. Explicitly adds Conda include paths (using
# $ENV{CONDA_PREFIX}) when MKL backend is active. Renamed backend preference
# options. Links IPP libraries by name, removing find_package(IPP). Added
# $CONDA_PREFIX/include/ipp path for MKL backend. Added target_link_directories
# for IPP .lib files with PUBLIC scope. Refactored Conda include path logic for
# potentially better IntelliSense parsing. Updated library sources for backend
# refactoring. Corrected if()/command() syntax errors. # <-- Change Description
cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0148 NEW) # Policy for modern FindPython behavior
                              # (suppresses warning)

# Set project name and version (used for installation and config files)
project(
  OmniDSP
  VERSION 1.0.0
  LANGUAGES CXX)

enable_testing() # Enable CTest support

# Option to control whether Python bindings are built
option(OMNIDSP_BUILD_PYTHON_BINDINGS "Build the Python language bindings" OFF)
# Option to control shared/static library build type
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so/.dylib)" ON)

# Options to control verbose logging (added)
option(OMNIDSP_ENABLE_MKL_DEBUG_LOGS "Enable verbose MKL backend debug logs"
       OFF)
option(OMNIDSP_ENABLE_CQT_WARNINGS "Enable CQT Nk clamping warnings" OFF)

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
include(GenerateExportHeader) # For handling DLL exports/imports

# --- Find Dependencies (Managed by Conda) ---

# Find pybind11 (Needed regardless of OMNIDSP_BUILD_PYTHON_BINDINGS for
# conditional logic)
find_package(pybind11 CONFIG REQUIRED)
message(
  STATUS
    "Found pybind11 version ${pybind11_VERSION} via find_package at ${pybind11_DIR}"
)

# Find Python (needed for Python tests if bindings are ON)
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  find_package(
    Python
    COMPONENTS Interpreter Development
    REQUIRED)
  message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
endif()

# Find GTest (needed for C++ tests if bindings are OFF)
if(NOT OMNIDSP_BUILD_PYTHON_BINDINGS)
  find_package(GTest REQUIRED)
  message(STATUS "Found GoogleTest for C++ tests.")
endif()

# --- Option to control backend preference ---
option(PREFER_ACCELERATE "Prefer Apple Accelerate if available" ${APPLE})
option(PREFER_ONEMKL "Prefer Intel oneMKL if available" OFF)

# --- Backend Detection and Source File Selection ---
set(BACKEND "NONE")
set(BACKEND_SOURCES "") # Changed to plural, expect a list
set(BACKEND_LINK_LIBS "")
set(BACKEND_DEFINITION "")
set(BACKEND_INCLUDE_DIRS "")
set(BACKEND_LINK_DIRS "") # Initialize link dirs

if(APPLE
   AND PREFER_ACCELERATE
   AND NOT PREFER_ONEMKL)
  find_package(Accelerate QUIET)
  if(Accelerate_FOUND)
    message(STATUS "Found Apple Accelerate framework (via find_package).")
    set(BACKEND "ACCELERATE")
    # Use GLOB to find all .cpp files in the accelerate backend directory
    file(GLOB ACCELERATE_SRCS CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/*.cpp")
    set(BACKEND_SOURCES ${ACCELERATE_SRCS})
    set(BACKEND_LINK_LIBS Accelerate::Accelerate)
    set(BACKEND_DEFINITION "USE_ACCELERATE")
  else()
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
      message(STATUS "Found Apple Accelerate framework (via find_library).")
      set(BACKEND "ACCELERATE")
      file(GLOB ACCELERATE_SRCS CONFIGURE_DEPENDS
           "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/*.cpp")
      set(BACKEND_SOURCES ${ACCELERATE_SRCS})
      set(BACKEND_LINK_LIBS ${ACCELERATE_FRAMEWORK})
      set(BACKEND_DEFINITION "USE_ACCELERATE")
    else()
      message(
        WARNING
          "Apple Accelerate framework requested but not found on Apple system.")
    endif()
    unset(ACCELERATE_FRAMEWORK CACHE)
  endif()
endif()

if(NOT BACKEND STREQUAL "ACCELERATE" OR PREFER_ONEMKL)
  find_package(MKL QUIET)
  if(MKL_FOUND)
    message(STATUS "Found Intel oneMKL (version ${MKL_VERSION}).")
    if(NOT BACKEND STREQUAL "ACCELERATE" OR PREFER_ONEMKL)
      message(STATUS "Selecting Intel oneMKL backend.")
      set(BACKEND "ONEMKL")
      # Use GLOB to find all .cpp files in the onemkl backend directory
      file(GLOB ONEMKL_SRCS CONFIGURE_DEPENDS
           "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/*.cpp")
      set(BACKEND_SOURCES ${ONEMKL_SRCS})
      set(BACKEND_DEFINITION "USE_ONEMKL")
      if(TARGET MKL::MKL)
        set(BACKEND_LINK_LIBS MKL::MKL)
      elseif(MKL_LIBRARIES)
        set(BACKEND_LINK_LIBS ${MKL_LIBRARIES})
      else()
        message(
          WARNING
            "FindMKL found MKL but did not define MKL::MKL target or MKL_LIBRARIES."
        )
      endif()
      # Link IPP libraries by name
      list(APPEND BACKEND_LINK_LIBS ippcore ipps)
      message(STATUS "Adding IPP link libraries by name: ippcore ipps")
      # Handle include directories
      if(DEFINED MKL_INCLUDE_DIRS AND NOT MKL_INCLUDE_DIRS STREQUAL "")
        set(BACKEND_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
        message(
          STATUS
            "Adding MKL include directory found by find_package: ${BACKEND_INCLUDE_DIRS}"
        )
      else()
        message(
          STATUS
            "MKL_INCLUDE_DIRS not set by find_package(MKL). Will rely on CONDA_PREFIX."
        )
      endif()
      # --- Set Backend Link Directories for MKL/IPP ---
      if(DEFINED ENV{CONDA_PREFIX})
        set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
        if(WIN32)
          set(conda_library_lib_dir "${CONDA_PREFIX_PATH}/Library/lib")
        else() # Linux/macOS
          set(conda_library_lib_dir "${CONDA_PREFIX_PATH}/lib")
        endif()
        if(EXISTS "${conda_library_lib_dir}")
          set(BACKEND_LINK_DIRS ${conda_library_lib_dir})
          message(
            STATUS
              "Setting backend link directory from CONDA_PREFIX: ${BACKEND_LINK_DIRS}"
          )
        else()
          message(
            WARNING
              "Guessed Conda library directory does not exist: ${conda_library_lib_dir}. Linker might not find MKL/IPP libs."
          )
        endif()
      else()
        message(
          WARNING
            "CONDA_PREFIX environment variable not set! Cannot automatically set linker path for MKL/IPP."
        )
      endif()
      # --- END Set ---
    else()
      message(
        STATUS
          "Intel oneMKL found, but Accelerate backend was already selected.")
    endif()
  else()
    if(NOT BACKEND STREQUAL "ACCELERATE")
      message(STATUS "Intel oneMKL not found.")
    endif()
    if(PREFER_ONEMKL)
      message(WARNING "Preferred oneMKL backend not found.")
    endif()
  endif()
endif()

if(BACKEND STREQUAL "NONE")
  message(
    WARNING
      "No suitable backend (Accelerate or oneMKL) found or configured. Using stub implementation (will throw errors at runtime)."
  )
  # Use GLOB to find all .cpp files in the stub backend directory
  file(GLOB STUB_SRCS CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/stub/*.cpp")
  set(BACKEND_SOURCES ${STUB_SRCS})
  # No specific definition or link libs needed for stub
endif()

# --- Check if Backend Sources were found ---
if(NOT BACKEND_SOURCES)
  message(
    FATAL_ERROR
      "No backend source files were found for the selected backend (${BACKEND}). Check backend subdirectories and CMake GLOB patterns."
  )
endif()

message(STATUS "OmniDSP selected backend: ${BACKEND}")
message(STATUS "OmniDSP backend sources: ${BACKEND_SOURCES}") # Now a list
message(STATUS "OmniDSP backend definition: ${BACKEND_DEFINITION}")
message(STATUS "OmniDSP backend link libs: ${BACKEND_LINK_LIBS}")
message(STATUS "OmniDSP backend link dirs: ${BACKEND_LINK_DIRS}")

# --- Library Target Definition ---
add_library(
  omnidsp # Target name
  # Public API Implementation Files
  src/omnidsp/fft.cpp
  src/omnidsp/cqt.cpp
  src/omnidsp/convolution.cpp
  src/omnidsp/window.cpp
  src/omnidsp/resample.cpp # Add the new resample API file
  # Backend Implementation Files (List from GLOB)
  ${BACKEND_SOURCES})
# Removed src/omnidsp/omnidsp.cpp

generate_export_header(omnidsp BASE_NAME OMNIDSP EXPORT_FILE_NAME
                       OmniDSP/omnidsp_export.h)
target_compile_features(omnidsp PRIVATE cxx_std_17)

# --- Define Include Directories ---
target_include_directories(
  omnidsp
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
# Add include dir for backend_impl.h
target_include_directories(omnidsp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp)

# Add backend-specific include directories as PRIVATE
if(BACKEND STREQUAL "ONEMKL")
  if(DEFINED BACKEND_INCLUDE_DIRS AND NOT BACKEND_INCLUDE_DIRS STREQUAL "")
    target_include_directories(omnidsp PRIVATE ${BACKEND_INCLUDE_DIRS})
  endif()
  # Add Conda include paths explicitly for MKL/IPP headers
  if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    set(conda_library_include_dir "")
    set(conda_main_include_dir "")
    set(conda_ipp_include_dir "")
    if(WIN32)
      set(conda_library_include_dir "${CONDA_PREFIX_PATH}/Library/include")
      set(conda_main_include_dir "${CONDA_PREFIX_PATH}/include")
      set(conda_ipp_include_dir "${CONDA_PREFIX_PATH}/include/ipp")
    else()
      set(conda_library_include_dir "${CONDA_PREFIX_PATH}/include")
      set(conda_main_include_dir "${CONDA_PREFIX_PATH}/include")
      set(conda_ipp_include_dir "${CONDA_PREFIX_PATH}/include/ipp")
    endif()
    # Add paths if they exist
    if(EXISTS "${conda_library_include_dir}")
      target_include_directories(omnidsp PRIVATE "${conda_library_include_dir}")
      message(
        STATUS
          "Explicitly adding Conda Library include path from CONDA_PREFIX: ${conda_library_include_dir}"
      )
    else()
      # Corrected syntax: message on newline
      if(WIN32)
        message(
          WARNING
            "Guessed Conda Library include path does not exist: ${conda_library_include_dir}"
        )
      endif()
    endif()
    if(EXISTS "${conda_main_include_dir}")
      # Corrected syntax: command on newline
      if(NOT conda_main_include_dir STREQUAL conda_library_include_dir)
        target_include_directories(omnidsp PRIVATE "${conda_main_include_dir}")
        message(
          STATUS
            "Explicitly adding Conda main include path from CONDA_PREFIX: ${conda_main_include_dir}"
        )
      endif()
    else()
      message(
        WARNING
          "Guessed Conda main include path does not exist: ${conda_main_include_dir}"
      )
    endif()
    if(EXISTS "${conda_ipp_include_dir}")
      target_include_directories(omnidsp PRIVATE "${conda_ipp_include_dir}")
      message(
        STATUS
          "Explicitly adding Conda IPP include path from CONDA_PREFIX: ${conda_ipp_include_dir}"
      )
    else()
      message(
        FATAL_ERROR
          "Required Conda IPP include path does not exist: ${conda_ipp_include_dir}. Check IPP installation."
      )
    endif()
  else()
    message(
      FATAL_ERROR
        "CONDA_PREFIX environment variable not set! Cannot find Conda includes. Activate Conda environment."
    )
  endif()
endif()

# Add backend-specific compile definitions TO THE LIBRARY TARGET Corrected
# syntax: command on newline
if(NOT BACKEND_DEFINITION STREQUAL "")
  target_compile_definitions(omnidsp PRIVATE ${BACKEND_DEFINITION})
endif()

# Add compile definitions based on logging options Corrected syntax: command on
# newline
if(OMNIDSP_ENABLE_MKL_DEBUG_LOGS)
  target_compile_definitions(omnidsp PRIVATE OMNIDSP_ENABLE_MKL_DEBUG_LOGS)
  message(STATUS "Enabling MKL debug logs for omnidsp target.")
endif()
# Corrected syntax: command on newline
if(OMNIDSP_ENABLE_CQT_WARNINGS)
  target_compile_definitions(omnidsp PRIVATE OMNIDSP_ENABLE_CQT_WARNINGS)
  message(STATUS "Enabling CQT Nk clamping warnings for omnidsp target.")
endif()

# --- Add Linker Directories ---
# Corrected syntax: command on newline
if(NOT BACKEND_LINK_DIRS STREQUAL "")
  target_link_directories(
    omnidsp PUBLIC ${BACKEND_LINK_DIRS}) # Use PUBLIC if downstream targets need
                                         # it
  message(
    STATUS "Adding PUBLIC link directories for omnidsp: ${BACKEND_LINK_DIRS}")
endif()

# Link omnidsp library target against its dependencies
target_link_libraries(
  omnidsp
  PUBLIC # Use PUBLIC if backend libs need to be propagated (e.g., MKL
         # interface)
         ${BACKEND_LINK_LIBS})

# =============================================================================
# Conditionally Add Subdirectories based on Build Type
# =============================================================================
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  message(
    STATUS
      "Configuring Python bindings and tests (OMNIDSP_BUILD_PYTHON_BINDINGS=ON)."
  )
  # Corrected syntax: command on newline
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp_py/CMakeLists.txt")
    add_subdirectory(src/omnidsp_py)
  else()
    message(
      WARNING
        "Python bindings subdirectory 'src/omnidsp_py' or its CMakeLists.txt not found, skipping Python module configuration."
    )
  endif()
  # Corrected syntax: command on newline
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/CMakeLists.txt")
    add_subdirectory(tests/python)
  else()
    message(
      STATUS
        "Python tests subdirectory 'tests/python' or its CMakeLists.txt not found, skipping Python test configuration."
    )
  endif()
else()
  message(
    STATUS
      "Configuring C++ examples and tests (OMNIDSP_BUILD_PYTHON_BINDINGS=OFF).")
  # Corrected syntax: command on newline
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/CMakeLists.txt")
    add_subdirectory(examples/cpp)
  else()
    message(
      STATUS
        "C++ examples subdirectory 'examples/cpp' or its CMakeLists.txt not found, skipping C++ example configuration."
    )
  endif()
  # Corrected syntax: command on newline
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
    add_subdirectory(tests/cpp)
  else()
    message(
      STATUS
        "C++ tests subdirectory 'tests/cpp' or its CMakeLists.txt not found, skipping C++ test configuration."
    )
  endif()
endif()

# =============================================================================
# Installation Rules
# =============================================================================
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  message(
    STATUS "Configuring Python examples installation (Python bindings ON)")
  # Corrected syntax: command on newline
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/notebooks")
    install(
      DIRECTORY examples/notebooks/
      DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/examples_python"
      COMPONENT Examples
      FILES_MATCHING
      PATTERN "*.ipynb")
  else()
    message(
      STATUS
        "Python examples directory 'examples/notebooks' not found, skipping installation."
    )
  endif()
else()
  message(
    STATUS "Configuring C++ library installation rules (Python bindings OFF)")
  set(OMNIDSP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
  # Install public headers
  install(
    DIRECTORY include/OmniDSP/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT Devel
    FILES_MATCHING
    PATTERN "*.h")
  # Install generated export header
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/OmniDSP/omnidsp_export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/OmniDSP"
    COMPONENT Devel)
  # Generate and install CMake config files Corrected syntax: command on newline
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in")
    message(
      WARNING
        "Config.cmake.in not found. Cannot generate CMake package configuration files."
    )
  else()
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY AnyNewerVersion)
    configure_package_config_file(
      Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
      INSTALL_DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}")
    install(
      FILES "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
      DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
      COMPONENT Devel)
  endif()
  # Install library targets and export set
  install(
    TARGETS omnidsp
    EXPORT OmniDSPTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime
    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # For include directories
                                              # associated with the target
  )
  install(
    EXPORT OmniDSPTargets
    FILE OmniDSPTargets.cmake
    NAMESPACE OmniDSP::
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    COMPONENT Devel)
endif()
