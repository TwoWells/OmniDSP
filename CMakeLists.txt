# CMakeLists.txt for OmniDSP project Uses Conda for managing CMake, pybind11,
# GTest, Boost, Highway, and MKL/IPP dependencies Supports compiling multiple
# backends (Default + available Accelerate/oneMKL) for runtime selection.

cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0148 NEW) # Policy for modern FindPython behavior
cmake_policy(SET CMP0167 NEW) # Policy for FindBoost module removal

project(
  OmniDSP
  VERSION 1.0.0
  LANGUAGES CXX)

enable_testing() # Enable CTest support

# Option to control whether Python bindings are built
option(OMNIDSP_BUILD_PYTHON_BINDINGS "Build the Python language bindings" OFF)
# Option to control shared/static library build type
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so/.dylib)" ON)

# Options to control verbose logging
option(OMNIDSP_ENABLE_MKL_DEBUG_LOGS "Enable verbose MKL backend debug logs"
       OFF)
option(OMNIDSP_ENABLE_CQT_WARNINGS "Enable CQT Nk clamping warnings" OFF)

# Enable generation of compile_commands.json for IDEs/tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
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
find_package(pybind11 CONFIG REQUIRED)
message(
  STATUS
    "Found pybind11 version ${pybind11_VERSION} via find_package at ${pybind11_DIR}"
)
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  find_package(
    Python
    COMPONENTS Interpreter Development
    REQUIRED)
  message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
endif()
if(NOT OMNIDSP_BUILD_PYTHON_BINDINGS)
  find_package(GTest REQUIRED)
  message(STATUS "Found GoogleTest for C++ tests.")
endif()
find_package(Boost REQUIRED)
message(STATUS "Found Boost version ${Boost_VERSION}")
find_package(highway CONFIG REQUIRED)
message(STATUS "Found Google Highway library.")

# --- Backend Detection and Configuration ---
# We will detect all available backends and compile support for them. The C++
# factory `OmniDSP::create` will choose at runtime.

set(OMNIDSP_HAS_ACCELERATE FALSE)
set(OMNIDSP_HAS_ONEMKL FALSE)
set(OMNIDSP_COMPILE_DEFINITIONS "") # Accumulate compile definitions
set(OMNIDSP_LINK_LIBS "") # Accumulate link libraries
set(OMNIDSP_INCLUDE_DIRS "") # Accumulate include dirs
set(OMNIDSP_LINK_DIRS "") # Accumulate link dirs

# Always include Default backend sources
set(DEFAULT_BACKEND_SOURCES "")
list(
  APPEND
  DEFAULT_BACKEND_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/fft.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/window.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/convolution.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/resample.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/cqt.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/backend.cpp")
message(
  STATUS "Default backend sources will be compiled: ${DEFAULT_BACKEND_SOURCES}")

# Check for Accelerate
if(APPLE)
  find_package(Accelerate QUIET)
  if(Accelerate_FOUND OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Apple Accelerate backend available.")
    set(OMNIDSP_HAS_ACCELERATE TRUE)
    list(APPEND OMNIDSP_COMPILE_DEFINITIONS "USE_ACCELERATE")
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    list(APPEND OMNIDSP_LINK_LIBS ${ACCELERATE_FRAMEWORK})
    # Add Accelerate source files
    list(
      APPEND
      ACCELERATE_BACKEND_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/fft.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/window.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/convolution.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/resample.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/backend.cpp")
    message(
      STATUS
        "Accelerate backend sources will be compiled: ${ACCELERATE_BACKEND_SOURCES}"
    )
  else()
    message(STATUS "Apple Accelerate framework not found.")
  endif()
endif()

# Check for oneMKL
find_package(MKL QUIET)
if(MKL_FOUND)
  message(STATUS "Intel oneMKL backend available (version ${MKL_VERSION}).")
  set(OMNIDSP_HAS_ONEMKL TRUE)
  list(APPEND OMNIDSP_COMPILE_DEFINITIONS "USE_ONEMKL")
  # Set Link Libraries for MKL
  if(TARGET MKL::MKL)
    list(APPEND OMNIDSP_LINK_LIBS MKL::MKL)
  elseif(MKL_LIBRARIES)
    list(APPEND OMNIDSP_LINK_LIBS ${MKL_LIBRARIES})
  else()
    message(
      WARNING
        "FindMKL found MKL but did not define MKL::MKL target or MKL_LIBRARIES."
    )
  endif()
  # Add IPP libraries
  list(APPEND OMNIDSP_LINK_LIBS ippcore ipps)
  message(STATUS "Adding IPP link libraries by name: ippcore ipps")
  # Set Include Dirs for MKL/IPP (handled by Conda path logic below)
  if(DEFINED MKL_INCLUDE_DIRS AND NOT MKL_INCLUDE_DIRS STREQUAL "")
    list(APPEND OMNIDSP_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
  endif()
  # Set Link Dirs for MKL/IPP (handled by Conda path logic below)
  if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    if(WIN32)
      set(conda_library_lib_dir "${CONDA_PREFIX_PATH}/Library/lib")
    else()
      set(conda_library_lib_dir "${CONDA_PREFIX_PATH}/lib")
    endif()
    if(EXISTS "${conda_library_lib_dir}")
      list(APPEND OMNIDSP_LINK_DIRS ${conda_library_lib_dir})
    endif()
  else()
    message(
      WARNING
        "CONDA_PREFIX environment variable not set! Cannot determine Conda link path for MKL/IPP."
    )
  endif()
  # Add oneMKL source files
  list(
    APPEND
    ONEMKL_BACKEND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/window.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/resample.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/cqt.cpp" # If oneMKL
                                                                     # provides
                                                                     # CQT
                                                                     # override
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/backend.cpp")
  message(
    STATUS "oneMKL backend sources will be compiled: ${ONEMKL_BACKEND_SOURCES}")
else()
  message(STATUS "Intel oneMKL not found.")
endif()

# --- Library Target Definition ---
add_library(
  omnidsp # Target name
  # Public API Implementation Files (Forwarders)
  src/omnidsp/fft.cpp
  src/omnidsp/cqt.cpp
  src/omnidsp/convolution.cpp
  # src/omnidsp/window.cpp # WindowSpec is header-only
  src/omnidsp/resample.cpp
  src/omnidsp/omnidsp.cpp # The main OmniDSP class impl
  # Default Backend Implementation Files (Always included)
  ${DEFAULT_BACKEND_SOURCES}
  # Optimized Backend Implementation Files (Conditionally included)
  $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${ACCELERATE_BACKEND_SOURCES}>
  $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${ONEMKL_BACKEND_SOURCES}>)

generate_export_header(omnidsp BASE_NAME OMNIDSP EXPORT_FILE_NAME
                       OmniDSP/omnidsp_export.h)
target_compile_features(omnidsp PRIVATE cxx_std_23)

# --- Define Include Directories ---
target_include_directories(
  omnidsp
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default
    # Conditionally add optimized backend include paths if they exist
    $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate>
    $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl>
    ${OMNIDSP_INCLUDE_DIRS} # MKL includes found by find_package
)
# Add Conda include paths explicitly for MKL/IPP headers if MKL is enabled
if(OMNIDSP_HAS_ONEMKL)
  # ... (MKL/IPP Conda include path logic remains the same as before) ...
  if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    set(conda_library_include_dir "")
    set(conda_main_include_dir "")
    set(conda_ipp_include_dir_specific "")
    set(conda_ipp_include_dir_parent "")
    set(found_ipp_include FALSE)
    set(found_mkl_include FALSE)

    set(conda_main_include_dir "${CONDA_PREFIX_PATH}/include")
    set(conda_library_include_dir "${CONDA_PREFIX_PATH}/Library/include"
    )# Windows specific
    set(conda_ipp_include_dir_specific "${conda_main_include_dir}/ipp")
    set(conda_ipp_include_dir_parent "${conda_main_include_dir}")

    if(EXISTS "${conda_library_include_dir}")
      target_include_directories(omnidsp PRIVATE "${conda_library_include_dir}")
      message(
        STATUS
          "[MKL] Explicitly adding Conda Library include path: ${conda_library_include_dir}"
      )
      if(EXISTS "${conda_library_include_dir}/mkl.h")
        set(found_mkl_include TRUE)
      endif()
    endif()
    if(EXISTS "${conda_main_include_dir}"
       AND NOT conda_main_include_dir STREQUAL conda_library_include_dir)
      target_include_directories(omnidsp PRIVATE "${conda_main_include_dir}")
      message(
        STATUS
          "[MKL] Explicitly adding Conda main include path: ${conda_main_include_dir}"
      )
      if(EXISTS "${conda_main_include_dir}/mkl.h")
        set(found_mkl_include TRUE)
      endif()
    elseif(EXISTS "${conda_main_include_dir}"
           AND conda_main_include_dir STREQUAL conda_library_include_dir)
      if(EXISTS "${conda_main_include_dir}/mkl.h")
        set(found_mkl_include TRUE)
      endif() # Already added if paths equal
    endif()
    if(NOT found_mkl_include)
      message(WARNING "[MKL] Could not find mkl.h in Conda include paths.")
    endif()

    if(EXISTS "${conda_ipp_include_dir_specific}")
      target_include_directories(omnidsp
                                 PRIVATE "${conda_ipp_include_dir_specific}")
      message(
        STATUS
          "[IPP] Explicitly adding Conda IPP include path (specific): ${conda_ipp_include_dir_specific}"
      )
      set(found_ipp_include TRUE)
    elseif(EXISTS "${conda_ipp_include_dir_parent}/ipps.h")
      target_include_directories(omnidsp
                                 PRIVATE "${conda_ipp_include_dir_parent}")
      message(
        STATUS
          "[IPP] Explicitly adding Conda IPP include path (parent): ${conda_ipp_include_dir_parent}"
      )
      set(found_ipp_include TRUE)
    endif()
    if(NOT found_ipp_include)
      message(
        FATAL_ERROR
          "[IPP] Required Conda IPP include path not found. Check IPP installation."
      )
    endif()
  else()
    message(
      FATAL_ERROR
        "[MKL/IPP] CONDA_PREFIX environment variable not set! Cannot find Conda includes. Activate Conda environment."
    )
  endif()
endif()

# --- Add Compile Definitions ---
target_compile_definitions(omnidsp PRIVATE ${OMNIDSP_COMPILE_DEFINITIONS}
)# Add USE_ACCELERATE / USE_ONEMKL
if(OMNIDSP_ENABLE_MKL_DEBUG_LOGS AND OMNIDSP_HAS_ONEMKL) # Only define if MKL is
                                                         # used
  target_compile_definitions(omnidsp PRIVATE OMNIDSP_ENABLE_MKL_DEBUG_LOGS)
  message(STATUS "Enabling MKL debug logs for omnidsp target.")
endif()
if(OMNIDSP_ENABLE_CQT_WARNINGS)
  target_compile_definitions(omnidsp PRIVATE OMNIDSP_ENABLE_CQT_WARNINGS)
  message(STATUS "Enabling CQT Nk clamping warnings for omnidsp target.")
endif()

# --- Add Linker Directories ---
# Add only if the list is not empty
if(OMNIDSP_LINK_DIRS)
  target_link_directories(omnidsp PUBLIC ${OMNIDSP_LINK_DIRS})
  message(
    STATUS "Adding PUBLIC link directories for omnidsp: ${OMNIDSP_LINK_DIRS}")
endif()

# --- Link Libraries ---
target_link_libraries(
  omnidsp
  PUBLIC # Propagate necessary libs
         Boost::headers # Link Boost headers
         hwy::hwy # Link Highway library target
         ${OMNIDSP_LINK_LIBS} # Add Accelerate/MKL/IPP libs
  PRIVATE # Link specific Boost components if needed, e.g., Boost::math
)

# --- Highway Dynamic Dispatch Configuration ---
hwy_dynamic_dispatch(omnidsp)
message(STATUS "Configured Highway dynamic dispatch for omnidsp target.")

# =============================================================================
# Conditionally Add Subdirectories based on Build Type
# =============================================================================
# ... (Logic for adding src/omnidsp_py, tests/python, examples/cpp, tests/cpp
# remains the same) ...
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  message(
    STATUS
      "Configuring Python bindings and tests (OMNIDSP_BUILD_PYTHON_BINDINGS=ON)."
  )
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp_py/CMakeLists.txt")
    add_subdirectory(src/omnidsp_py)
  else()
    message(
      WARNING
        "Python bindings subdirectory 'src/omnidsp_py' or its CMakeLists.txt not found, skipping Python module configuration."
    )
  endif()
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
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/CMakeLists.txt")
    add_subdirectory(examples/cpp)
  else()
    message(
      STATUS
        "C++ examples subdirectory 'examples/cpp' or its CMakeLists.txt not found, skipping C++ example configuration."
    )
  endif()
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
# ... (Installation logic remains largely the same) ...
# Ensure find_dependency calls in Config.cmake.in reflect ALL required public
# dependencies (Boost, Highway, potentially MKL if its interface is needed by
# consumers)
if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  message(
    STATUS "Configuring Python examples installation (Python bindings ON)")
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
  install(
    DIRECTORY include/OmniDSP/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT Devel
    FILES_MATCHING
    PATTERN "*.h")
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/OmniDSP/omnidsp_export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/OmniDSP"
    COMPONENT Devel)

  # --- Generate and install CMake config files ---
  # Assume Config.cmake.in exists in the source tree and is maintained by the
  # developer
  set(CONFIG_CMAKE_IN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in")
  if(NOT EXISTS "${CONFIG_CMAKE_IN_PATH}")
    message(
      FATAL_ERROR
        "Config.cmake.in not found at '${CONFIG_CMAKE_IN_PATH}'. This file is required for installation."
    )
  else()
    message(STATUS "Using Config.cmake.in found at: ${CONFIG_CMAKE_IN_PATH}")
    # Developers MUST ensure Config.cmake.in includes: find_dependency(Boost
    # REQUIRED) find_dependency(highway CONFIG REQUIRED) Potentially
    # find_dependency(MKL) if MKL interface is needed by consumers
  endif()

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)
  configure_package_config_file(
    "${CONFIG_CMAKE_IN_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
    INSTALL_DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    COMPONENT Devel)
  install(
    TARGETS omnidsp
    EXPORT OmniDSPTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime)
  install(
    EXPORT OmniDSPTargets
    FILE OmniDSPTargets.cmake
    NAMESPACE OmniDSP::
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    COMPONENT Devel)
endif()

message(STATUS "Finished processing root CMakeLists.txt")
