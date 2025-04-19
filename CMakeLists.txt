# CMakeLists.txt for OmniDSP project Uses Conda for managing CMake, pybind11,
# GTest, Boost, and MKL/IPP dependencies Includes options for controlling debug
# logs and CMP0148 policy. Includes GenerateExportHeader for Windows DLL builds
# with PUBLIC include fix. Uses BUILD_INTERFACE generator expression for build
# directory include path. Explicitly adds Conda include paths (using
# $ENV{CONDA_PREFIX}) when MKL backend is active. Links IPP libraries by name.
# Added Boost dependency finding and linking. Uses explicit source file listing
# for backends instead of file(GLOB). Corrected Windows IPP include path check.
# Corrected Boost include directory handling (use target_link_libraries).

cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0148 NEW) # Policy for modern FindPython behavior
                              # (suppresses warning)
cmake_policy(SET CMP0167 NEW) # Policy for FindBoost module removal (suppresses
                              # warning)

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

# Find the Boost library (required for Bessel function in windowing) Headers are
# usually sufficient for basic math functions.
find_package(Boost REQUIRED)
message(STATUS "Found Boost version ${Boost_VERSION}")

# --- Option to control backend preference ---
option(PREFER_ACCELERATE "Prefer Apple Accelerate if available" ${APPLE})
option(PREFER_ONEMKL "Prefer Intel oneMKL if available" OFF)

# --- Backend Detection ---
set(BACKEND "NONE")
set(BACKEND_DEFINITION "")
set(BACKEND_LINK_LIBS "")
set(BACKEND_INCLUDE_DIRS "")
set(BACKEND_LINK_DIRS "")
set(OMNIDSP_BACKEND_SOURCES "") # Initialize list for backend sources

if(APPLE
   AND PREFER_ACCELERATE
   AND NOT PREFER_ONEMKL)
  find_package(Accelerate QUIET)
  # Also check system name as fallback detection on macOS
  if(Accelerate_FOUND OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Selecting Apple Accelerate backend.")
    set(BACKEND "ACCELERATE")
    set(BACKEND_DEFINITION "USE_ACCELERATE")
    # Link Accelerate framework
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    set(BACKEND_LINK_LIBS ${ACCELERATE_FRAMEWORK})
    # Add specific Accelerate sources directly to the list
    list(
      APPEND
      OMNIDSP_BACKEND_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/fft.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/window.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/convolution.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/resample.cpp"
      # Add other accelerate backend files here if any
    )
  else()
    message(
      WARNING
        "Apple Accelerate framework requested but not found on Apple system.")
  endif()
endif()

# Check for MKL only if Accelerate wasn't chosen or MKL is preferred
if(NOT BACKEND STREQUAL "ACCELERATE" OR PREFER_ONEMKL)
  find_package(MKL QUIET)
  if(MKL_FOUND)
    message(STATUS "Found Intel oneMKL (version ${MKL_VERSION}).")
    # Select MKL if Accelerate wasn't chosen OR if MKL is preferred
    if(NOT BACKEND STREQUAL "ACCELERATE" OR PREFER_ONEMKL)
      message(STATUS "Selecting Intel oneMKL backend.")
      set(BACKEND "ONEMKL")
      set(BACKEND_DEFINITION "USE_ONEMKL")
      # Set Link Libraries
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
      # Add IPP libraries by name
      list(APPEND BACKEND_LINK_LIBS ippcore ipps)
      message(STATUS "Adding IPP link libraries by name: ippcore ipps")
      # Set Include Dirs (rely on Conda paths added later)
      if(DEFINED MKL_INCLUDE_DIRS AND NOT MKL_INCLUDE_DIRS STREQUAL "")
        set(BACKEND_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
      endif()
      # Set Link Dirs (rely on Conda paths added later)
      if(DEFINED ENV{CONDA_PREFIX})
        set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
        if(WIN32)
          set(conda_library_lib_dir "${CONDA_PREFIX_PATH}/Library/lib")
        else() # Linux/macOS
          set(conda_library_lib_dir "${CONDA_PREFIX_PATH}/lib")
        endif()
        if(EXISTS "${conda_library_lib_dir}")
          set(BACKEND_LINK_DIRS ${conda_library_lib_dir})
        endif()
      endif()
      # Add specific MKL sources directly to the list
      list(
        APPEND
        OMNIDSP_BACKEND_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/fft.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/window.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/convolution.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/resample.cpp"
        # Add other onemkl backend files here if any
      )
    else()
      message(
        STATUS
          "Intel oneMKL found, but Accelerate backend was already selected and preferred."
      )
    endif()
  else()
    # MKL not found, only issue warning if it was preferred
    if(NOT BACKEND STREQUAL "ACCELERATE")
      message(STATUS "Intel oneMKL not found.")
    endif()
    if(PREFER_ONEMKL)
      message(WARNING "Preferred oneMKL backend not found.")
    endif()
  endif()
endif()

# Fallback to Stub if no other backend was selected
if(BACKEND STREQUAL "NONE")
  message(
    WARNING
      "No suitable backend (Accelerate or oneMKL) found or configured. Using stub implementation."
  )
  set(BACKEND "STUB") # Explicitly set backend name
  # Add specific Stub sources directly to the list
  list(
    APPEND
    OMNIDSP_BACKEND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/stub/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/stub/window.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/stub/convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/stub/resample.cpp"
    # Add other stub backend files here if any
  )
endif()

# --- Verify that some backend sources were added ---
if(NOT OMNIDSP_BACKEND_SOURCES)
  message(
    FATAL_ERROR
      "No backend source files were selected. Check backend detection logic.")
endif()

message(STATUS "OmniDSP selected backend: ${BACKEND}")
message(STATUS "OmniDSP backend sources: ${OMNIDSP_BACKEND_SOURCES}")
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
  src/omnidsp/resample.cpp
  # Backend Dispatcher
  src/omnidsp/backend/backend.cpp
  # Backend Implementation Files (Explicit List)
  ${OMNIDSP_BACKEND_SOURCES})

generate_export_header(omnidsp BASE_NAME OMNIDSP EXPORT_FILE_NAME
                       OmniDSP/omnidsp_export.h)
target_compile_features(omnidsp PRIVATE cxx_std_17)

# --- Define Include Directories ---
target_include_directories(
  omnidsp
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # Public headers
                                                                # (build tree)
         $<INSTALL_INTERFACE:include> # Public headers (install tree)
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # For generated export
                                                        # header
         # Boost::headers # <<< REMOVED FROM HERE
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp # For backend.h include
          ${BACKEND_INCLUDE_DIRS} # Add MKL include dirs if found by
                                  # find_package
)
# Add Conda include paths explicitly for MKL/IPP headers if MKL backend is
# selected
if(BACKEND STREQUAL "ONEMKL")
  if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    set(conda_library_include_dir "")
    set(conda_main_include_dir "")
    set(conda_ipp_include_dir_specific "") # Path with /ipp subdir
    set(conda_ipp_include_dir_parent "") # Path without /ipp subdir
    set(found_ipp_include FALSE)

    # --- CORRECTED PATHS FOR WINDOWS ---
    # On Windows Conda, MKL/IPP headers are typically under
    # $CONDA_PREFIX/include
    set(conda_main_include_dir "${CONDA_PREFIX_PATH}/include")
    set(conda_library_include_dir "${CONDA_PREFIX_PATH}/Library/include"
    )# Keep check for Library/include just in case
    set(conda_ipp_include_dir_specific "${conda_main_include_dir}/ipp"
    )# Check under $CONDA_PREFIX/include/ipp
    set(conda_ipp_include_dir_parent "${conda_main_include_dir}"
    )# Check under $CONDA_PREFIX/include

    # Add MKL library include dir if exists (check both locations)
    set(found_mkl_include FALSE)
    if(EXISTS "${conda_library_include_dir}")
      target_include_directories(omnidsp PRIVATE "${conda_library_include_dir}")
      message(
        STATUS
          "Explicitly adding Conda Library include path from CONDA_PREFIX: ${conda_library_include_dir}"
      )
      if(EXISTS "${conda_library_include_dir}/mkl.h")
        set(found_mkl_include TRUE)
      endif()
    else()
      message(
        WARNING
          "Guessed Conda Library include path does not exist: ${conda_library_include_dir}"
      )
    endif()
    # Add main include dir only if different and exists
    if(EXISTS "${conda_main_include_dir}"
       AND NOT conda_main_include_dir STREQUAL conda_library_include_dir)
      target_include_directories(omnidsp PRIVATE "${conda_main_include_dir}")
      message(
        STATUS
          "Explicitly adding Conda main include path from CONDA_PREFIX: ${conda_main_include_dir}"
      )
      if(EXISTS "${conda_main_include_dir}/mkl.h")
        set(found_mkl_include TRUE)
      endif()
    elseif(EXISTS "${conda_main_include_dir}"
           AND conda_main_include_dir STREQUAL conda_library_include_dir)
      # Already added, just check for mkl.h
      if(EXISTS "${conda_main_include_dir}/mkl.h")
        set(found_mkl_include TRUE)
      endif()
    else()
      message(
        WARNING
          "Guessed Conda main include path does not exist: ${conda_main_include_dir}"
      )
    endif()

    if(NOT found_mkl_include)
      message(
        WARNING
          "Could not find mkl.h in Conda include paths. MKL compilation might fail."
      )
    endif()

    # --- Check for IPP includes (Corrected Paths) ---
    # 1. Check specific ipp subdirectory under $CONDA_PREFIX/include
    if(EXISTS "${conda_ipp_include_dir_specific}")
      target_include_directories(omnidsp
                                 PRIVATE "${conda_ipp_include_dir_specific}")
      message(
        STATUS
          "Explicitly adding Conda IPP include path (specific): ${conda_ipp_include_dir_specific}"
      )
      set(found_ipp_include TRUE)
      # 1. If not found, check parent directory ($CONDA_PREFIX/include) for a key
      #   header (ipps.h)
    elseif(EXISTS "${conda_ipp_include_dir_parent}/ipps.h")
      target_include_directories(omnidsp
                                 PRIVATE "${conda_ipp_include_dir_parent}")
      message(
        STATUS
          "Explicitly adding Conda IPP include path (parent): ${conda_ipp_include_dir_parent}"
      )
      set(found_ipp_include TRUE)
    endif()

    # 1. Error if neither location worked
    if(NOT found_ipp_include)
      message(
        FATAL_ERROR
          "Required Conda IPP include path not found in specific ('${conda_ipp_include_dir_specific}') or parent ('${conda_ipp_include_dir_parent}') directory. Check IPP installation in Conda env."
      )
    endif()
    # --- End IPP check ---

  else()
    message(
      FATAL_ERROR
        "CONDA_PREFIX environment variable not set! Cannot find Conda includes for MKL/IPP. Activate Conda environment."
    )
  endif()
endif()

# --- Add Compile Definitions ---
if(NOT BACKEND_DEFINITION STREQUAL "")
  target_compile_definitions(omnidsp PRIVATE ${BACKEND_DEFINITION})
endif()
# Add definitions based on logging options
if(OMNIDSP_ENABLE_MKL_DEBUG_LOGS)
  target_compile_definitions(omnidsp PRIVATE OMNIDSP_ENABLE_MKL_DEBUG_LOGS)
  message(STATUS "Enabling MKL debug logs for omnidsp target.")
endif()
if(OMNIDSP_ENABLE_CQT_WARNINGS)
  target_compile_definitions(omnidsp PRIVATE OMNIDSP_ENABLE_CQT_WARNINGS)
  message(STATUS "Enabling CQT Nk clamping warnings for omnidsp target.")
endif()

# --- Add Linker Directories ---
if(NOT BACKEND_LINK_DIRS STREQUAL "")
  target_link_directories(
    omnidsp PUBLIC ${BACKEND_LINK_DIRS}) # Use PUBLIC if downstream targets need
                                         # it (e.g., MKL)
  message(
    STATUS "Adding PUBLIC link directories for omnidsp: ${BACKEND_LINK_DIRS}")
endif()

# --- Link Libraries ---
target_link_libraries(
  omnidsp
  PUBLIC # Propagate backend libs if needed (e.g., MKL interface)
         ${BACKEND_LINK_LIBS} Boost::headers # <<< ADDED Boost::headers HERE
                                             # (PUBLIC scope)
  PRIVATE # Link Boost component libraries privately if needed, e.g. Boost::math
          # Boost::boost # This links all found components, might be overkill.
          # Link specific components if known.
)

# =============================================================================
# Conditionally Add Subdirectories based on Build Type
# =============================================================================
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
  # Generate and install CMake config files
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in")
    # Create a basic Config.cmake.in if it doesn't exist This allows downstream
    # projects to find the library even without complex config
    file(
      WRITE "${CMAKE_CURRENT_BINARY_DIR}/Config.cmake.in"
      "@PACKAGE_INIT@\n"
      "include(CMakeFindDependencyMacro)\n"
      "# Find required dependencies here if any, e.g.:\n"
      "find_dependency(Boost REQUIRED)\n" # Add Boost dependency here
      "include(\"${CMAKE_CURRENT_LIST_DIR}/OmniDSPTargets.cmake\")\n"
      "check_required_components(OmniDSP)\n")
    message(STATUS "Config.cmake.in not found. Generated a basic version.")
    set(CONFIG_CMAKE_IN_PATH "${CMAKE_CURRENT_BINARY_DIR}/Config.cmake.in")
  else()
    set(CONFIG_CMAKE_IN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in")
    message(STATUS "Using existing Config.cmake.in.")
  endif()

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

  configure_package_config_file(
    "${CONFIG_CMAKE_IN_PATH}" # Use variable for path
    "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
    INSTALL_DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR # Add vars needed by
                                                            # Config.cmake.in
  )

  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/OmniDSPConfigVersion.cmake"
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    COMPONENT Devel)

  # Install library targets and export set
  install(
    TARGETS omnidsp
    EXPORT OmniDSPTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime
    INCLUDES # This is problematic, includes should be handled by
             # target_include_directories
    # DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # Remove INCLUDES DESTINATION
  )
  install(
    EXPORT OmniDSPTargets
    FILE OmniDSPTargets.cmake
    NAMESPACE OmniDSP::
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"
    COMPONENT Devel)
endif()

message(STATUS "Finished processing root CMakeLists.txt")
