# ==============================================================================
# cmake/target_definitions.cmake
# ==============================================================================
# Defines the core 'omnidsp' library target using variables set by backend configs.

message(STATUS "Defining 'omnidsp' library target...")

#--------------------------------------------------------------------------
# Define Core Library Sources
#--------------------------------------------------------------------------
set(OMNIDSP_CORE_HEADERS
    ${PROJECT_SOURCE_DIR}/include/OmniDSP/omnidsp.hpp
    ${PROJECT_SOURCE_DIR}/include/OmniDSP/core_types.hpp
    ${PROJECT_SOURCE_DIR}/include/OmniDSP/fft.hpp
    ${PROJECT_SOURCE_DIR}/include/OmniDSP/cqt.hpp
    ${PROJECT_SOURCE_DIR}/include/OmniDSP/convolution.hpp
    ${PROJECT_SOURCE_DIR}/include/OmniDSP/resample.hpp
    ${PROJECT_SOURCE_DIR}/include/OmniDSP/filter.hpp
    ${PROJECT_SOURCE_DIR}/include/OmniDSP/window.hpp
    ${PROJECT_BINARY_DIR}/include/OmniDSP/omnidsp_export.hpp # Generated export header
)
set(OMNIDSP_CORE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/omnidsp/omnidsp.cpp
    # Interface files implement the public Plan classes
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/fft.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/cqt.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/convolution.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/resample.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/filter.cpp
    # Add interface/backend.cpp if it contains implementations
    # ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/backend.cpp
)


# --- Create the Library Target ---
add_library(omnidsp ${OMNIDSP_LIBRARY_TYPE} "") # Start with no sources

# --- Add Core Sources ---
target_sources(omnidsp PRIVATE ${OMNIDSP_CORE_SOURCES})

# --- Add Backend Sources Conditionally ---
# Include sources based on CMake variables set by backend config files
message(STATUS "  Adding sources from selected backends:")
if(OMNIDSP_HAS_DEFAULT)
    target_sources(omnidsp PRIVATE ${DEFAULT_BACKEND_SOURCES})
    message(STATUS "    Added Default backend sources.")
endif()
if(OMNIDSP_HAS_ACCELERATE) # Assumes ACCELERATE_BACKEND_SOURCES is set in accelerate.cmake
    target_sources(omnidsp PRIVATE ${ACCELERATE_BACKEND_SOURCES})
    message(STATUS "    Added Accelerate backend sources.")
endif()
if(OMNIDSP_HAS_ONEMKL)
    target_sources(omnidsp PRIVATE ${ONEMKL_BACKEND_SOURCES})
    message(STATUS "    Added oneMKL backend sources.")
endif()
# Add other backends similarly...


#--------------------------------------------------------------------------
# Target Include Directories
#--------------------------------------------------------------------------
target_include_directories(omnidsp
    PUBLIC
        # Public headers needed by users of the library
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> # For generated export header

    PRIVATE
        # Core internal include directory
        ${PROJECT_SOURCE_DIR}/src/omnidsp # Allows includes like "interface/backend.hpp"

        # Add backend-specific include directories conditionally using generator expressions
        $<$<BOOL:${OMNIDSP_HAS_DEFAULT}>:${DEFAULT_BACKEND_INCLUDE_DIRS}>
        $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${ACCELERATE_BACKEND_INCLUDE_DIRS}>
        $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${ONEMKL_BACKEND_INCLUDE_DIRS}>
        # Add other backend includes...
)

#--------------------------------------------------------------------------
# Target Linking Libraries
#--------------------------------------------------------------------------
target_link_libraries(omnidsp
    PUBLIC
        # Public link dependencies (e.g., Boost::headers if needed by public API)
        Boost::headers # Example: Assume Boost headers are needed publicly

    PRIVATE
        # Link common dependencies needed by core or multiple backends
        hwy::hwy # Example: Core highway library (if used by default or others)

        # Link backend-specific libraries conditionally using generator expressions
        $<$<BOOL:${OMNIDSP_HAS_DEFAULT}>:${DEFAULT_BACKEND_LINK_LIBS}>
        $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:${ACCELERATE_BACKEND_LINK_LIBS}>
        $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:${ONEMKL_BACKEND_LINK_LIBS}>
        # Add other backend links...
)

#--------------------------------------------------------------------------
# Target Compile Definitions
#--------------------------------------------------------------------------
target_compile_definitions(omnidsp
    PUBLIC
        # Public definitions needed by users of the library

    PRIVATE
        # Add core compile definitions (set in compiler_settings.cmake or elsewhere)
        ${OMNIDSP_COMPILE_DEFINITIONS}

        # Define OMNIDSP_SHARED_DEFINE if building shared libs, used by omnidsp_export.hpp
        $<$<BOOL:${BUILD_SHARED_LIBS}>:OMNIDSP_SHARED_DEFINE>

        # Add backend-specific compile definitions conditionally using generator expressions
        $<$<BOOL:${OMNIDSP_HAS_DEFAULT}>:OMNIDSP_BACKEND_DEFAULT=1>
        $<$<BOOL:${OMNIDSP_HAS_ACCELERATE}>:OMNIDSP_BACKEND_ACCELERATE=1>
        $<$<BOOL:${OMNIDSP_HAS_ONEMKL}>:OMNIDSP_BACKEND_ONEMKL=1>
        # Add other backend defines...
)

#--------------------------------------------------------------------------
# Generate Export Header
#--------------------------------------------------------------------------
include(GenerateExportHeader)
generate_export_header(omnidsp
    BASE_NAME OMNIDSP # Used in macro names (e.g., OMNIDSP_EXPORT)
    EXPORT_FILE_NAME include/OmniDSP/omnidsp_export.hpp # Output location relative to CMAKE_BINARY_DIR
)

message(STATUS "Defined 'omnidsp' target with sources and dependencies.")
