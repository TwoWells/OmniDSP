# ==============================================================================
# cmake/target_definitions.cmake
# ==============================================================================
#
# Brief Description:
#   Defines the core 'omnidsp' library target. This includes adding core
#   source files, incorporating object files from selected backend OBJECT targets,
#   setting include directories, linking against backend targets and common
#   dependencies, defining core compile definitions, and generating the export
#   header for DLL symbol visibility.
#
# Variables Read:
#   - PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR (CMake Built-in)
#   - BUILD_SHARED_LIBS (from cmake/project_options.cmake or environment) # Used for compile defs
#   - OMNIDSP_LIBRARY_TYPE (from cmake/project_options.cmake) # <-- Now read from options
#   - OMNIDSP_SELECTED_BACKEND_TARGETS (List, from cmake/backend.cmake)
#   - Common dependency targets (e.g., hwy::hwy, hwy::dispatch, Boost::headers, expected from cmake/depend.cmake)
#   - OMNIDSP_COMPILE_DEFINITIONS (from cmake/compiler_settings.cmake)
#
# Targets Defined/Modified:
#   - omnidsp (Library Target): Defined and configured in this file.
#
# ==============================================================================

message(STATUS "Defining 'omnidsp' library target...")

#--------------------------------------------------------------------------
# Determine Library Type -- REMOVED (Now handled in project_options.cmake)
#--------------------------------------------------------------------------
# if(BUILD_SHARED_LIBS)
#     set(OMNIDSP_LIBRARY_TYPE SHARED)
#     message(STATUS "  Building omnidsp as SHARED library.")
# else()
#     set(OMNIDSP_LIBRARY_TYPE STATIC)
#     message(STATUS "  Building omnidsp as STATIC library.")
# endif()

#--------------------------------------------------------------------------
# Define Core Library Sources
#--------------------------------------------------------------------------
# These source files form the core of the library and are always included.
set(OMNIDSP_CORE_SOURCES
    # --- Public API Implementation ---
    ${PROJECT_SOURCE_DIR}/src/omnidsp/omnidsp.cpp # Impl for OmniDSP class (factory, dispatch)

    # --- Public Plan Wrappers (Interface Layer) ---
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/fft.cpp         # Impl for FFTPlan, RFFTPlan
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/cqt.cpp         # Impl for CQTPlan
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/convolution.cpp # Impl for ConvolutionPlan, CorrelationPlan
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/resample.cpp    # Impl for ResamplePlan
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/filter.cpp      # Impl for FIRFilterPlan, IIRFilterPlan
)

# --- Create the Library Target ---
# Defines the main 'omnidsp' library target initially with core sources.
# Uses OMNIDSP_LIBRARY_TYPE variable set in project_options.cmake
add_library(omnidsp ${OMNIDSP_LIBRARY_TYPE} "") # Start with no sources, add below

# --- Add Core Sources ---
target_sources(omnidsp PRIVATE ${OMNIDSP_CORE_SOURCES})

# --- Add Backend Sources (from OBJECT libraries) ---
# Iterate through the selected backend targets. If a target is an OBJECT library,
# add its compiled object files as sources to the main omnidsp library.
message(STATUS "  Adding sources/objects from selected backends:")
foreach(backend_target IN LISTS OMNIDSP_SELECTED_BACKEND_TARGETS)
    get_target_property(target_type ${backend_target} TYPE)
    if(target_type STREQUAL "OBJECT_LIBRARY")
          message(STATUS "    Adding object files from OBJECT backend: ${backend_target}")
          target_sources(omnidsp PRIVATE $<TARGET_OBJECTS:${backend_target}>)
    elseif(target_type STREQUAL "INTERFACE_LIBRARY")
          message(STATUS "    Backend ${backend_target} is INTERFACE, linking only.")
          # No object files to add directly for INTERFACE libraries.
    else()
          message(WARNING "    Backend target ${backend_target} is not an OBJECT or INTERFACE library. Type: ${target_type}. Linking only.")
          # Handle STATIC/SHARED backend libraries if necessary (link them below)
    endif()
endforeach()

# Set target properties (e.g., C++ standard) - Handled by cmake/compiler_settings.cmake

#--------------------------------------------------------------------------
# Target Include Directories
#--------------------------------------------------------------------------
# Specifies directories the compiler should search for included header files.
target_include_directories(omnidsp
    PUBLIC
        # Directories needed by code *using* the omnidsp library.
        $<INSTALL_INTERFACE:include>                    # For installed headers (e.g., /usr/local/include)
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>   # For finding <OmniDSP/omnidsp.hpp> etc. from source tree
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>   # For finding generated <OmniDSP/omnidsp_export.hpp>

    PRIVATE
        # Directories needed only for compiling omnidsp sources themselves.
        ${PROJECT_SOURCE_DIR}/src/omnidsp # Allows includes like "interface/backend.hpp"
)

#--------------------------------------------------------------------------
# Target Linking Libraries
#--------------------------------------------------------------------------
target_link_libraries(omnidsp
    PUBLIC
        # Public link dependencies (libraries users of omnidsp also need to link against)
        # Boost::headers # If using header-only boost parts publicly

    PRIVATE
        # Link selected backend targets as PRIVATE implementation details.
        # INTERFACE properties (includes, defines, link libs like MKL/Accelerate/hwy::dispatch)
        # are propagated from these targets.
        ${OMNIDSP_SELECTED_BACKEND_TARGETS}

        # Link common dependencies found via cmake/depend.cmake
        hwy::hwy # Core highway library

        # Example: Boost::math
        # Add other common dependency targets here...
)

#--------------------------------------------------------------------------
# Target Compile Definitions
#--------------------------------------------------------------------------
target_compile_definitions(omnidsp
    PUBLIC
        # Public definitions needed by users of the library
        # Example: $<$<PLATFORM_ID:Windows>:NOMINMAX>

    PRIVATE
        # Add core compile definitions (set in compiler_settings.cmake or elsewhere)
        ${OMNIDSP_COMPILE_DEFINITIONS}

        # Define OMNIDSP_SHARED_DEFINE if building shared libs, used by omnidsp_export.hpp
        $<$<BOOL:${BUILD_SHARED_LIBS}>:OMNIDSP_SHARED_DEFINE>

        # OMNIDSP_EXPORTS is defined automatically by generate_export_header
        # when building a SHARED library on Windows. No need to add manually.
)

#--------------------------------------------------------------------------
# Generate Export Header
#--------------------------------------------------------------------------
# Creates OmniDSP/omnidsp_export.hpp to manage symbol visibility (e.g., __declspec(dllexport/dllimport) on Windows).
# This uses OMNIDSP_SHARED_DEFINE and OMNIDSP_EXPORTS (auto-defined)
include(GenerateExportHeader) # Ensure module is included
generate_export_header(omnidsp
    BASE_NAME OMNIDSP # Used in macro names (e.g., OMNIDSP_EXPORT)
    EXPORT_FILE_NAME include/OmniDSP/omnidsp_export.hpp # Output location relative to CMAKE_BINARY_DIR
)

message(STATUS "Defined 'omnidsp' target with sources and dependencies.")
