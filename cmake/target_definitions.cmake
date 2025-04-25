# ==============================================================================
# cmake/target_definitions.cmake
# ==============================================================================
#
# Brief Description:
#   Defines the core 'omnidsp' library target. This includes aggregating
#   source files (core, default backend, and conditionally included optional
#   backends), setting include directories, linking dependencies, defining
#   compile definitions, and generating the export header for DLL symbol
#   visibility.
#
# Variables Read:
#   - PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR (CMake Built-in)
#   - OMNIDSP_LIBRARY_TYPE (from cmake/project_options.cmake)
#   - OMNIDSP_HAS_ACCELERATE (from cmake/backend/accelerate.cmake)
#   - OMNIDSP_HAS_ONEMKL (from cmake/backend/onemkl.cmake)
#   - OMNIDSP_INCLUDE_DIRS (from cmake/depend.cmake)
#   - OMNIDSP_ACCELERATE_INCLUDE_DIRS (from cmake/backend/accelerate.cmake)
#   - OMNIDSP_ONEMKL_INCLUDE_DIRS (from cmake/backend/onemkl.cmake)
#   - OMNIDSP_LINK_LIBS (from cmake/depend.cmake)
#   - OMNIDSP_ACCELERATE_LINK_LIBS (from cmake/backend/accelerate.cmake)
#   - OMNIDSP_ONEMKL_LINK_LIBS (from cmake/backend/onemkl.cmake)
#   - OMNIDSP_BACKEND_DEFINITIONS (from cmake/backend.cmake)
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
# They include the main OmniDSP class implementation, the public Plan class
# implementations (which act as wrappers/Pimpl forwarders), and the
# complete implementation of the Default backend.
set(OMNIDSP_CORE_SOURCES
    # --- Public API Implementation ---
    ${PROJECT_SOURCE_DIR}/src/omnidsp/omnidsp.cpp # Impl for OmniDSP class (factory, dispatch)

    # --- Public Plan Wrappers (Interface Layer) ---
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/fft.cpp         # Impl for FFTPlan, RFFTPlan
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/cqt.cpp         # Impl for CQTPlan
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/convolution.cpp # Impl for ConvolutionPlan, CorrelationPlan
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/resample.cpp    # Impl for ResamplePlan
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/filter.cpp      # Impl for FIRFilterPlan, IIRFilterPlan

    # --- Default Backend Implementation (Always Included) ---
    ${PROJECT_SOURCE_DIR}/src/omnidsp/default/backend.cpp      # Impl for DefaultBackend class
    ${PROJECT_SOURCE_DIR}/src/omnidsp/default/fft.cpp         # Impl for Default*FFTPlanImpl
    ${PROJECT_SOURCE_DIR}/src/omnidsp/default/cqt.cpp         # Impl for DefaultCQTPlanImpl
    ${PROJECT_SOURCE_DIR}/src/omnidsp/default/convolution.cpp # Impl for Default*ConvolutionPlanImpl
    ${PROJECT_SOURCE_DIR}/src/omnidsp/default/resample.cpp    # Impl for DefaultResamplePlanImpl
    ${PROJECT_SOURCE_DIR}/src/omnidsp/default/filter.cpp      # Impl for Default*FilterPlanImpl
    ${PROJECT_SOURCE_DIR}/src/omnidsp/default/window.cpp      # Impl for DefaultBackend window functions
    # Add other default backend source files (e.g., default/filter_design.cpp) if they exist
)

#--------------------------------------------------------------------------
# Conditionally Add Optional Backend Sources
#--------------------------------------------------------------------------
# These source files are only added if the corresponding backend was found
# and enabled during the CMake configuration process (via OMNIDSP_ENABLE_* options
# and checks in cmake/backend/*.cmake modules).
# The OMNIDSP_HAS_* variables are set by those backend modules.

# --- Accelerate Backend Sources (macOS/iOS only) ---
set(OMNIDSP_ACCELERATE_SOURCES
    # Generator expression: Adds files only if OMNIDSP_HAS_ACCELERATE is TRUE
    $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/backend.cpp>
    $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/fft.cpp>
    $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/convolution.cpp>
    $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/resample.cpp>
    $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/window.cpp>
    # Add $<$<BOOL...>:accelerate/filter.cpp> when implemented
)

# --- oneMKL Backend Sources ---
set(OMNIDSP_ONEMKL_SOURCES
    # Generator expression: Adds files only if OMNIDSP_HAS_ONEMKL is TRUE
    $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/backend.cpp>
    $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/fft.cpp>
    $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/convolution.cpp>
    $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/resample.cpp> # Uses IPP
    $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/window.cpp> # Uses IPP/VML
    # Add $<$<BOOL...>:onemkl/filter.cpp> when implemented
)

# --- Create the Library Target ---
# Defines the main 'omnidsp' library target.
# Can be SHARED (.dll/.so) or STATIC (.lib/.a) based on OMNIDSP_LIBRARY_TYPE option.
add_library(omnidsp ${OMNIDSP_LIBRARY_TYPE}
            ${OMNIDSP_CORE_SOURCES}      # Always include core and default backend
            ${OMNIDSP_ACCELERATE_SOURCES} # Conditionally include Accelerate sources
            ${OMNIDSP_ONEMKL_SOURCES}     # Conditionally include oneMKL sources
            # Add other conditional backend sources here... $<$<BOOL:${OMNIDSP_HAS_YOURBACKEND}>:...>
)

# Set target properties (e.g., C++ standard) - Handled by cmake/compiler_settings.cmake
# Ensure properties set there apply to the 'omnidsp' target.

#--------------------------------------------------------------------------
# Target Include Directories
#--------------------------------------------------------------------------
# Specifies directories the compiler should search for included header files.
target_include_directories(omnidsp
    PUBLIC
        # Directories needed by code *using* the omnidsp library.
        # $<INSTALL_INTERFACE:...> specifies the include path after installation.
        # $<BUILD_INTERFACE:...> specifies the include path when building dependents locally.
        $<INSTALL_INTERFACE:include>                     # For installed headers (e.g., /usr/local/include)
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> # For finding <OmniDSP/omnidsp.h> etc. from source tree
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> # For finding generated <OmniDSP/omnidsp_export.h>

    PRIVATE
        # Directories needed only for compiling omnidsp sources themselves.
        ${PROJECT_SOURCE_DIR}/src/omnidsp # Allows includes like "interface/backend.h", "default/backend.h" etc.

        # Include directories from dependencies (set in cmake/depend.cmake and cmake/backend/*.cmake)
        # These variables aggregate includes from Boost, Highway, MKL, IPP, Accelerate framework, etc.
        ${OMNIDSP_INCLUDE_DIRS}            # Core dependencies (Boost, Highway)
        ${OMNIDSP_ACCELERATE_INCLUDE_DIRS} # Specific includes for Accelerate (usually none needed)
        ${OMNIDSP_ONEMKL_INCLUDE_DIRS}     # Specific includes for MKL/IPP
)

#--------------------------------------------------------------------------
# Target Linking Libraries
#--------------------------------------------------------------------------
# Specifies libraries that the 'omnidsp' target depends on.
target_link_libraries(omnidsp
    PUBLIC
        # Public link dependencies (libraries users of omnidsp also need to link against)
        # Typically empty unless omnidsp exposes types/functions from a dependency in its public headers.
    PRIVATE
        # Private link dependencies (implementation details hidden from users)
        ${OMNIDSP_LINK_LIBS} # Core dependencies (Boost::math, Highway)

        # Conditionally link backend-specific libraries
        # Variables set in cmake/backend/*.cmake
        ${OMNIDSP_ACCELERATE_LINK_LIBS} # e.g., Accelerate.framework
        ${OMNIDSP_ONEMKL_LINK_LIBS}     # e.g., mkl_core, mkl_intel_thread, ippcore, ipps, etc.
)

#--------------------------------------------------------------------------
# Target Compile Definitions
#--------------------------------------------------------------------------
# Defines preprocessor macros used during compilation.
target_compile_definitions(omnidsp
    PRIVATE
        # Conditionally add definitions to enable/disable backend code paths via #ifdef
        # OMNIDSP_BACKEND_DEFINITIONS is populated in cmake/backend.cmake based on OMNIDSP_HAS_* vars.
        # Example content of OMNIDSP_BACKEND_DEFINITIONS might be:
        # $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:OMNIDSP_USE_ACCELERATE>
        # $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:OMNIDSP_USE_ONEMKL>
        ${OMNIDSP_BACKEND_DEFINITIONS}

        # Add other necessary compile definitions (set in compiler_settings.cmake or elsewhere)
        ${OMNIDSP_COMPILE_DEFINITIONS}

        # Define for DLL export/import handling (used by omnidsp_export.h)
        # Handled automatically when building SHARED library on Windows if using generate_export_header
        # $<$<CONFIG:Debug>:OMNIDSP_DEBUG> # Example: Define OMNIDSP_DEBUG in Debug builds
)

#--------------------------------------------------------------------------
# Generate Export Header
#--------------------------------------------------------------------------
# Creates OmniDSP/omnidsp_export.h to manage symbol visibility (e.g., __declspec(dllexport/dllimport) on Windows).
# This makes the library usable as a shared library (.dll/.so).
generate_export_header(omnidsp
    BASE_NAME OMNIDSP # Used in macro names (e.g., OMNIDSP_EXPORT)
    EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/OmniDSP/omnidsp_export.h # Output location
)
# Note: The output directory ${PROJECT_BINARY_DIR}/include is already added to PRIVATE includes above.

message(STATUS "Defined 'omnidsp' target with sources and dependencies.")
