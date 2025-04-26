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
#   - OMNIDSP_LIBRARY_TYPE (from cmake/project_options.cmake)
#   - OMNIDSP_SELECTED_BACKEND_TARGETS (List, from cmake/backend.cmake)
#   - Common dependency targets (e.g., HWY::hwy, Boost::math, expected from cmake/depend.cmake)
#   - OMNIDSP_COMPILE_DEFINITIONS (from cmake/compiler_settings.cmake)
#
# Targets Defined/Modified:
#   - omnidsp (Library Target): Defined and configured in this file.
#
# ==============================================================================

message(STATUS "Defining 'omnidsp' library target...")

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
# Can be SHARED (.dll/.so) or STATIC (.lib/.a) based on OMNIDSP_LIBRARY_TYPE option.
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
# Example: set_target_properties(omnidsp PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD})

#--------------------------------------------------------------------------
# Target Include Directories
#--------------------------------------------------------------------------
# Specifies directories the compiler should search for included header files.
# Include directories from backend targets and common dependencies (like Highway, MKL, IPP)
# will be inherited via target_link_libraries INTERFACE scope.
target_include_directories(omnidsp
    PUBLIC
        # Directories needed by code *using* the omnidsp library.
        # $<INSTALL_INTERFACE:...> specifies the include path after installation.
        # $<BUILD_INTERFACE:...> specifies the include path when building dependents locally.
        $<INSTALL_INTERFACE:include>                        # For installed headers (e.g., /usr/local/include)
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>   # For finding <OmniDSP/omnidsp.hpp> etc. from source tree
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>   # For finding generated <OmniDSP/omnidsp_export.hpp>

    PRIVATE
        # Directories needed only for compiling omnidsp sources themselves.
        ${PROJECT_SOURCE_DIR}/src/omnidsp # Allows includes like "interface/backend.hpp"
        # Common dependency includes (e.g., Boost headers) are ideally inherited via
        # target_link_libraries below. Add explicitly only if necessary.
        # Example: ${Boost_INCLUDE_DIRS} if not linking Boost::headers target.
)

#--------------------------------------------------------------------------
# Target Linking Libraries
#--------------------------------------------------------------------------
# Specifies libraries that the 'omnidsp' target depends on.
# Linking against backend targets and dependency targets propagates their
# INTERFACE properties (includes, definitions, link libraries).
target_link_libraries(omnidsp
    PUBLIC
        # Public link dependencies (libraries users of omnidsp also need to link against)
        # Example: If omnidsp's public API uses Boost types extensively.
        # Boost::headers # If using header-only boost parts publicly

    PRIVATE
        # Link selected backend targets as PRIVATE implementation details.
        # This prevents CMake's install(EXPORT ...) from complaining that the
        # OBJECT library backends are not in an export set, as their code
        # is already compiled into 'omnidsp'. Any necessary INTERFACE properties
        # (includes, defines, link libs like MKL/Accelerate) are still propagated.
        ${OMNIDSP_SELECTED_BACKEND_TARGETS}

        # Link common dependencies found via cmake/depend.cmake
        # Ensure these are imported targets (e.g., HWY::hwy, Boost::math)
        # Use PUBLIC or PRIVATE depending on whether users of omnidsp need them.
        # Example:
        # HWY::hwy       # Often PRIVATE if Highway usage is internal
        # Boost::math    # Often PRIVATE unless Boost types are in public API
        # Add other common dependency targets here...
)

#--------------------------------------------------------------------------
# Target Compile Definitions
#--------------------------------------------------------------------------
# Defines preprocessor macros used during compilation.
# Backend-specific definitions (e.g., OMNIDSP_USE_ONEMKL) are inherited from
# the INTERFACE properties of the linked backend targets.
target_compile_definitions(omnidsp
    PUBLIC
        # Public definitions needed by users of the library
        # Example: $<$<PLATFORM_ID:Windows>:NOMINMAX>

    PRIVATE
        # Add core compile definitions (set in compiler_settings.cmake or elsewhere)
        ${OMNIDSP_COMPILE_DEFINITIONS}

        # Define for DLL export/import handling (used by omnidsp_export.hpp)
        # Handled automatically by generate_export_header for SHARED builds on Windows.
        # Add other private definitions if needed.
        # Example: $<$<CONFIG:Debug>:OMNIDSP_INTERNAL_DEBUG>
)

#--------------------------------------------------------------------------
# Generate Export Header
#--------------------------------------------------------------------------
# Creates OmniDSP/omnidsp_export.hpp to manage symbol visibility (e.g., __declspec(dllexport/dllimport) on Windows).
# This makes the library usable as a shared library (.dll/.so).
include(GenerateExportHeader) # Ensure module is included
generate_export_header(omnidsp
    BASE_NAME OMNIDSP # Used in macro names (e.g., OMNIDSP_EXPORT)
    EXPORT_FILE_NAME include/OmniDSP/omnidsp_export.hpp # Output location relative to CMAKE_BINARY_DIR
)
# Note: ${PROJECT_BINARY_DIR}/include was added to PUBLIC includes above.

message(STATUS "Defined 'omnidsp' target with sources and dependencies.")
