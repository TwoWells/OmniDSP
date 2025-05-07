# ==============================================================================
# cmake/target_definitions.cmake (Refactored v4 - Simplify GenEx)
# ==============================================================================
# Defines the core 'omnidsp' library target using backend interface targets
# and OMNIDSP_ENABLED_* variables.

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
)
set(OMNIDSP_CORE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/omnidsp/omnidsp.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/utils/resample.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/utils/filter_design.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/fft.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/cqt.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/convolution.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/resample.cpp
    ${PROJECT_SOURCE_DIR}/src/omnidsp/interface/filter.cpp
)

#--------------------------------------------------------------------------
# Define Backend Source Files (Variables for clarity)
#--------------------------------------------------------------------------
# Default Backend Sources (Assuming DEFAULT_BACKEND_SOURCES is set in default.cmake)
# set(DEFAULT_BACKEND_SOURCES ...)

# Accelerate Backend Sources (Assuming ACCELERATE_BACKEND_SOURCES is set in accelerate.cmake)
# set(ACCELERATE_BACKEND_SOURCES ...)

# OneMKL Backend Sources
set(ONEMKL_BACKEND_SOURCES
    "${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/backend.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/fft.cpp"
    # Add onemkl/utils.cpp if it exists and is needed
)

# IntelIPP Backend Sources
set(INTELIPP_BACKEND_SOURCES
    "${PROJECT_SOURCE_DIR}/src/omnidsp/intelipp/backend.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/intelipp/fft.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/intelipp/fir_filter.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/intelipp/iir_filter.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/intelipp/resample.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/intelipp/window.cpp"
    "${PROJECT_SOURCE_DIR}/src/omnidsp/intelipp/utils.cpp"
    # Add intelipp/utils.cpp if it exists and is needed
)

# --- Create the Library Target ---
add_library(omnidsp ${OMNIDSP_LIBRARY_TYPE} "") # Start with no sources

# --- Add Core Sources ---
target_sources(omnidsp PRIVATE ${OMNIDSP_CORE_SOURCES})

# --- Add Backend Sources Conditionally ---
message(STATUS "  Adding sources from enabled backends:")
if(OMNIDSP_ENABLED_DEFAULT) # Use ENABLED flag
    target_sources(omnidsp PRIVATE ${DEFAULT_BACKEND_SOURCES})
    message(STATUS "    Added Default backend sources.")
endif()
if(OMNIDSP_ENABLED_ACCELERATE) # Use ENABLED flag
    target_sources(omnidsp PRIVATE ${ACCELERATE_BACKEND_SOURCES})
    message(STATUS "    Added Accelerate backend sources.")
endif()
if(OMNIDSP_ENABLED_ONEMKL) # Use ENABLED flag
    target_sources(omnidsp PRIVATE ${ONEMKL_BACKEND_SOURCES})
    message(STATUS "    Added oneMKL backend sources.")
endif()
if(OMNIDSP_ENABLED_INTELIPP) # Use ENABLED flag
    target_sources(omnidsp PRIVATE ${INTELIPP_BACKEND_SOURCES})
    message(STATUS "    Added IntelIPP backend sources.")
endif()
# Add other backends similarly...


#--------------------------------------------------------------------------
# Target Include Directories
#--------------------------------------------------------------------------
target_include_directories(omnidsp
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> # For generated export header

    PRIVATE
        ${PROJECT_SOURCE_DIR}/src/omnidsp
        ${PROJECT_BINARY_DIR}/include # For generated config header

        # Backend includes are handled by linking OmniDSP::* targets
)

#--------------------------------------------------------------------------
# Target Linking Libraries
#--------------------------------------------------------------------------
target_link_libraries(omnidsp
    PUBLIC
        # Public link dependencies

    PRIVATE
        # Common dependencies
        # $<$<TARGET_EXISTS:hwy::hwy>:hwy::hwy>

        # Link backend interface targets conditionally based on ENABLED flags
        # Simplified generator expressions: $<$<BOOL:VAR>:TARGET>
        # $<$<BOOL:${OMNIDSP_ENABLED_DEFAULT}>:${DEFAULT_BACKEND_LINK_LIBS}> # Keep if Default needs specific libs
        $<$<BOOL:${OMNIDSP_ENABLED_ACCELERATE}>:OmniDSP::accelerate> # Assuming OmniDSP::accelerate exists
        $<$<BOOL:${OMNIDSP_ENABLED_ONEMKL}>:OmniDSP::onemkl>
        $<$<BOOL:${OMNIDSP_ENABLED_INTELIPP}>:OmniDSP::intelipp>
        # Add other backend interface targets...
)

#--------------------------------------------------------------------------
# Target Compile Definitions
#--------------------------------------------------------------------------
target_compile_definitions(omnidsp
    PUBLIC
        # Public definitions

    PRIVATE
        ${OMNIDSP_COMPILE_DEFINITIONS} # Core definitions
        $<$<BOOL:${BUILD_SHARED_LIBS}>:OMNIDSP_SHARED_DEFINE>

        # Add definitions indicating which backends are ENABLED in the final build
        # These are intended for use in C++ via omnidsp_config.h
        $<$<BOOL:${OMNIDSP_ENABLED_DEFAULT}>:OMNIDSP_BACKEND_DEFAULT_ENABLED=1>
        $<$<BOOL:${OMNIDSP_ENABLED_ACCELERATE}>:OMNIDSP_BACKEND_ACCELERATE_ENABLED=1>
        $<$<BOOL:${OMNIDSP_ENABLED_ONEMKL}>:OMNIDSP_BACKEND_ONEMKL_ENABLED=1>
        $<$<BOOL:${OMNIDSP_ENABLED_INTELIPP}>:OMNIDSP_BACKEND_INTELIPP_ENABLED=1>
        # Add other backend defines...
)

#--------------------------------------------------------------------------
# Generate Export Header
#--------------------------------------------------------------------------
include(GenerateExportHeader)
generate_export_header(omnidsp
    BASE_NAME OMNIDSP
    EXPORT_FILE_NAME include/OmniDSP/omnidsp_export.hpp
)

message(STATUS "Defined 'omnidsp' target with sources and dependencies.")
