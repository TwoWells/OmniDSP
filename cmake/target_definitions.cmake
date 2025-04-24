# cmake/target_definitions.cmake
# ==============================
# Defines and configures the main 'omnidsp' library target using properties
# aggregated from backend detection and project options.
#
# Variables Read: - CMAKE_CURRENT_SOURCE_DIR, CMAKE_CURRENT_BINARY_DIR (CMake
# Built-in) - OMNIDSP_ALL_BACKEND_SOURCES (from backend.cmake) -
# OMNIDSP_ALL_BACKEND_INCLUDE_DIRS (from backend.cmake) -
# OMNIDSP_ALL_BACKEND_COMPILE_DEFS (from backend.cmake) -
# OMNIDSP_ALL_BACKEND_LINK_LIBS (from backend.cmake) -
# OMNIDSP_ALL_BACKEND_LINK_DIRS (from backend.cmake) -
# OMNIDSP_ENABLE_MKL_DEBUG_LOGS (from project_options.cmake) -
# OMNIDSP_ENABLE_CQT_WARNINGS (from project_options.cmake) - OMNIDSP_HAS_ONEMKL
# (from backend.cmake)
#
# Targets Defined: - omnidsp (Library target)
# ==============================

message(STATUS "Defining and configuring 'omnidsp' library target...")

# --- Library Target Definition ---
# Defines the library target using the public API sources and the aggregated
# backend sources collected by cmake/backend.cmake.
add_library(
  omnidsp # Target name
  # Public API Implementation Files (Forwarders)
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/fft.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/cqt.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/convolution.cpp"
  # "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/window.cpp" # WindowSpec is
  # header-only
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/resample.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/omnidsp.cpp" # The main OmniDSP class
                                                        # impl
  # Backend Implementation Files (Aggregated by backend.cmake)
  ${OMNIDSP_ALL_BACKEND_SOURCES})
message(STATUS "  Added library target 'omnidsp'.")

# --- Export Header Generation ---
# Generates OmniDSP/omnidsp_export.h for handling symbol visibility across
# shared library boundaries (DLL import/export on Windows, visibility on
# GCC/Clang).
generate_export_header(
  omnidsp BASE_NAME OMNIDSP # Used to create macro names like OMNIDSP_EXPORT
  EXPORT_FILE_NAME OmniDSP/omnidsp_export.h # Relative path in build dir/include
                                            # install dir
)
message(STATUS "  Configured export header generation.")

# C++ standard is set globally in cmake/compiler_settings.cmake

# --- Include Directories ---
# Defines public and private include paths for the target.
target_include_directories(
  omnidsp
  PUBLIC # Interface includes (needed by users of the library)
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # Public headers
                                                                # in source dir
                                                                # during build
         $<INSTALL_INTERFACE:include> # Public headers in install destination
                                      # (e.g., /usr/local/include)
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # For generated headers
                                                        # (like export.h)
  PRIVATE # Private implementation includes (only needed to build the library
          # itself)
          ${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp # Private API implementation
                                                  # headers
          # Add private include dirs for all enabled backends (collected by
          # backend.cmake)
          ${OMNIDSP_ALL_BACKEND_INCLUDE_DIRS})
message(STATUS "  Configured target include directories.")
message(
  STATUS "    Public includes (build): ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "    Public includes (install): include")
message(STATUS "    Public includes (generated): ${CMAKE_CURRENT_BINARY_DIR}")
message(
  STATUS
    "    Private includes: ${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp ${OMNIDSP_ALL_BACKEND_INCLUDE_DIRS}"
)

# --- Compile Definitions ---
# Adds compile definitions based on enabled backends and build options.
target_compile_definitions(
  omnidsp
  PRIVATE
    # Add backend-specific definitions (e.g., USE_ACCELERATE, USE_ONEMKL) These
    # were collected by backend.cmake into OMNIDSP_ALL_BACKEND_COMPILE_DEFS
    ${OMNIDSP_ALL_BACKEND_COMPILE_DEFS}
    # Add option-based definitions using generator expressions
    # OMNIDSP_HAS_ONEMKL is set by backend.cmake
    $<$<AND:$<BOOL:${OMNIDSP_ENABLE_MKL_DEBUG_LOGS}>,$<BOOL:${OMNIDSP_HAS_ONEMKL}>>:OMNIDSP_ENABLE_MKL_DEBUG_LOGS>
    $<$<BOOL:${OMNIDSP_ENABLE_CQT_WARNINGS}>:OMNIDSP_ENABLE_CQT_WARNINGS>)
message(STATUS "  Configured target compile definitions.")
message(STATUS "    Backend definitions: ${OMNIDSP_ALL_BACKEND_COMPILE_DEFS}")
if(OMNIDSP_ENABLE_MKL_DEBUG_LOGS AND OMNIDSP_HAS_ONEMKL)
  message(STATUS "    Conditional definition: OMNIDSP_ENABLE_MKL_DEBUG_LOGS")
endif()
if(OMNIDSP_ENABLE_CQT_WARNINGS)
  message(STATUS "    Conditional definition: OMNIDSP_ENABLE_CQT_WARNINGS")
endif()
# Note: _USE_MATH_DEFINES is handled globally in cmake/compiler_settings.cmake
# for MSVC

# --- Linker Directories ---
# Adds link directories required by backends (e.g., Conda lib path for MKL).
# These were collected by backend.cmake into OMNIDSP_ALL_BACKEND_LINK_DIRS.
if(OMNIDSP_ALL_BACKEND_LINK_DIRS)
  target_link_directories(omnidsp PRIVATE ${OMNIDSP_ALL_BACKEND_LINK_DIRS})
  message(
    STATUS
      "  Configured target link directories (PRIVATE): ${OMNIDSP_ALL_BACKEND_LINK_DIRS}"
  )
else()
  message(STATUS "  No additional target link directories required.")
endif()

# --- Link Libraries ---
# Links the target against its dependencies (Boost, Highway, backend libs).
target_link_libraries(
  omnidsp
  PUBLIC # Libraries whose headers/symbols are needed by users of omnidsp
         # (propagated)
         Boost::headers # Boost headers are often needed transitively
         hwy::hwy # Highway target provides includes and link libs
  PRIVATE # Libraries only needed to build omnidsp itself Add backend libs
          # (e.g., MKL::MKL, Accelerate, ippcore, ipps) These were collected by
          # backend.cmake into OMNIDSP_ALL_BACKEND_LINK_LIBS
          ${OMNIDSP_ALL_BACKEND_LINK_LIBS})
message(STATUS "  Configured target link libraries.")
message(STATUS "    Public link libraries: Boost::headers hwy::hwy")
message(STATUS "    Private link libraries: ${OMNIDSP_ALL_BACKEND_LINK_LIBS}")

# --- Highway Dynamic Dispatch Configuration ---
# Configures Google Highway for dynamic dispatch based on CPU features. This
# must be called after the target is defined and sources are added.
hwy_dynamic_dispatch(omnidsp)
message(STATUS "  Configured Highway dynamic dispatch for 'omnidsp'.")

message(STATUS "Finished defining and configuring 'omnidsp' target.")
