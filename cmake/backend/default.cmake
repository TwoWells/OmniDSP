# ==============================================================================
# cmake/backend/default.cmake
# ==============================================================================
#
# Brief Description:
#   Configures the 'omnidsp_backend_default' OBJECT library target.
#   This backend provides portable C++ implementations (potentially SIMD
#   accelerated via Highway if enabled) and is typically always included.
#
# Variables Read:
#   - PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR (CMake Built-in)
#   - Boost::headers, hwy::hwy, hwy::dispatch targets (Expected from cmake/depend.cmake)
#   - BUILD_SHARED_LIBS (from cmake/project_options.cmake or environment)
#
# Variables Set (in Caller's Scope via include()):
#   - OMNIDSP_SELECTED_BACKEND_TARGETS (Appends 'omnidsp_backend_default')
#
# Targets Defined:
#   - omnidsp_backend_default (OBJECT Library)
#
# ==============================================================================

message(STATUS "   Configuring Default backend target...")

# --- Define Backend Target ---
add_library(omnidsp_backend_default OBJECT)

# --- Define Backend Directory (for convenience) ---
set(OMNIDSP_DEFAULT_BACKEND_SRC_DIR ${PROJECT_SOURCE_DIR}/src/omnidsp/default)

# --- Backend Sources ---
target_sources(omnidsp_backend_default
    PRIVATE
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/backend.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/fft.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/window.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/convolution.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/resample.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/cqt.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/filter.cpp"
)

# --- Backend Include Directories ---
target_include_directories(omnidsp_backend_default
    PRIVATE # Private headers needed only to compile this backend's sources
        ${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}
        ${PROJECT_BINARY_DIR}/include # For generated omnidsp_export.hpp if needed internally
        ${PROJECT_SOURCE_DIR}/include # For finding <OmniDSP/core_types.hpp> etc.
)

# --- Backend Dependencies ---
target_link_libraries(omnidsp_backend_default
    PRIVATE
        Boost::headers
        hwy::hwy
)

# --- Backend Compile Definitions ---

# Add definitions needed by the main 'omnidsp' library when it consumes
# the object files from this backend (e.g., to know this backend is active).
target_compile_definitions(omnidsp_backend_default
    INTERFACE
        OMNIDSP_BACKEND_DEFAULT=1
)

# Add definitions needed *only* when compiling this backend's sources.
# This includes setting up the custom export macro OMNIDSP_DEFAULT_BACKEND_EXPORT
# used within the HWY_NAMESPACE block in fft.cpp.
if(BUILD_SHARED_LIBS)
    target_compile_definitions(omnidsp_backend_default
        PRIVATE
            # Signal that a shared library build is happening (for the macro logic)
            OMNIDSP_SHARED_DEFINE
            # Signal that *this specific target* is being built (triggers dllexport)
            OMNIDSP_DEFAULT_BACKEND_EXPORTS
    )
    message(STATUS "     Defining export macros for SHARED build of default backend.")
else()
    message(STATUS "     Export macros disabled for STATIC build of default backend.")
endif()


# --- Add to Caller's Scope List ---
# Make the main library aware of this backend target.
list(APPEND OMNIDSP_SELECTED_BACKEND_TARGETS "omnidsp_backend_default")

message(STATUS "     Configured Default backend target: omnidsp_backend_default")
