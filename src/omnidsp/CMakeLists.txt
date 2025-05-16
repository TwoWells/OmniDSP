# src/omnidsp/CMakeLists.txt
# Defines the 'OmniDSP' library, manages backend selection options,
# includes enabled backend configurations, and generates omnidsp_config.h and omnidsp_export.h.

message(STATUS "Configuring 'OmniDSP' library and backends...")

# --- Include CMake Modules ---
include(CMakeDependentOption) # For backend enabling options
include(GenerateExportHeader) # For omnidsp_export.h

# --- Backend Enabling Options ---
# Option for Accelerate Backend
cmake_dependent_option(
    OPTION_OMNIDSP_ENABLE_ACCELERATE
    "Enable Accelerate Backend (macOS only)"
    ON      # Default value IF dependency met (ON if APPLE)
    "APPLE" # Dependency variable(s)
    OFF     # Default value IF dependency NOT met (OFF if NOT APPLE)
)

# Option for oneMKL Backend
cmake_dependent_option(
    OPTION_OMNIDSP_ENABLE_ONEMKL
    "Enable oneMKL Backend (requires non-Apple OS and MKL library)"
    ON          # Default value IF dependency met (ON if NOT APPLE)
    "NOT APPLE" # Dependency variable(s)
    OFF         # Default value IF dependency NOT met (OFF if APPLE)
)

# Option for Intel IPP Backend
cmake_dependent_option(
    OPTION_OMNIDSP_ENABLE_INTELIPP
    "Enable Intel IPP Backend (requires non-Apple OS and IPP library)"
    ON          # Default value IF dependency met (ON if NOT APPLE)
    "NOT APPLE" # Dependency variable(s)
    OFF         # Default value IF dependency NOT met (OFF if APPLE)
)

# --- Initialize Backend Status Variables ---
# These are set with PARENT_SCOPE by the individual backend CMakeLists.txt files
# and checked here after they've been processed.
set(OMNIDSP_BACKEND_DEFAULT_ENABLED 0)
set(OMNIDSP_BACKEND_ACCELERATE_ENABLED 0)
set(OMNIDSP_BACKEND_ONEMKL_ENABLED 0)
set(OMNIDSP_BACKEND_INTELIPP_ENABLED 0)

# --- Define Core Library Sources ---

set(OMNIDSP_CORE_HEADERS
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/core_types.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/design.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/omnidsp.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/window.hpp"
    # include/OmniDSP/plan
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/plan/fft.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/plan/convolution.hpp"
    # include/OmniDSP/processor
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/processor/cqt.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/processor/fir_filter.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/processor/iir_filter.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/processor/resample.hpp"
    # include/OmniDSP/params
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/params/convolution.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/params/cqt.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/params/fft.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/params/fir_filter.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/params/iir_filter.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/params/resample.hpp"
    # include/OmniDSP/types
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/types/convolution.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/types/filter.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/types/window.hpp"
    # include/OmniDSP/coefs
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/coefs/fir_filter.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/coefs/iir_filter.hpp"
    # include/OmniDSP/specs
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/specs/window.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/specs/transform.hpp"
    # include/OmniDSP/design
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/design/fir_filter.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/design/resample.hpp"
    # ... other public headers ...
)

set(OMNIDSP_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/omnidsp.cpp" # Main class implementation
    # Params implementations (.cpp)
    "${CMAKE_CURRENT_SOURCE_DIR}/params/convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/params/cqt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/params/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/params/fir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/params/iir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/params/resample.cpp"
    # Specs implementations (.cpp)
    "${CMAKE_CURRENT_SOURCE_DIR}/specs/window.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/specs/transform.cpp"
    # Interface implementations (Pimpl wrappers for public API classes)
    #   plan
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/plan/convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/plan/fft.cpp"
    #   processor
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/processor/cqt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/processor/fir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/processor/iir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/processor/resample.cpp"
    # Design algorithm implementations (.cpp)
    "${CMAKE_CURRENT_SOURCE_DIR}/design/cqt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/design/fir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/design/iir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/design/resample.cpp"
    # Utils implementations (if any at this level)
    # "${CMAKE_CURRENT_SOURCE_DIR}/utils/some_util.cpp"
)

# --- Create the Main 'OmniDSP' Library Target ---
add_library(OmniDSP ${OMNIDSP_LIBRARY_TYPE} "") # OMNIDSP_LIBRARY_TYPE is set in the root CMakeLists.txt
target_sources(OmniDSP PRIVATE ${OMNIDSP_CORE_SOURCES})

# --- Public and Private Include Directories for 'OmniDSP' library ---
target_include_directories(OmniDSP
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Common Dependencies ---
if(TARGET hwy::hwy)
    target_link_libraries(OmniDSP PRIVATE hwy::hwy)
    message(STATUS "Linking OmniDSP with hwy::hwy")
else()
   message(WARNING "hwy::hwy target not found after find_package. Linking might fail.")
endif()

if(TARGET spdlog::spdlog)
   target_link_libraries(OmniDSP PRIVATE spdlog::spdlog)
   message(STATUS "Linking OmniDSP with spdlog::spdlog")
else()
   message(WARNING "spdlog::spdlog target not found after find_package. Linking might fail.")
endif()

# --- Include Backend Configurations ---
# This single call will process OmniDSP/src/omnidsp/backends/CMakeLists.txt,
# which in turn includes the individual backend configurations.
add_subdirectory(backend)


# --- Post Backend Configuration: Linking Interface Targets ---
# This section is now REMOVED as linking is handled by individual backend CMakeLists.txt files.
# if(OMNIDSP_BACKEND_ACCELERATE_ENABLED AND TARGET OmniDSP::Accelerate)
#     target_link_libraries(OmniDSP PRIVATE OmniDSP::Accelerate)
#     message(STATUS "Linking OmniDSP with OmniDSP::Accelerate")
# endif()
# if(OMNIDSP_BACKEND_ONEMKL_ENABLED AND TARGET OmniDSP::OneMKL)
#     target_link_libraries(OmniDSP PRIVATE OmniDSP::OneMKL)
#     message(STATUS "Linking OmniDSP with OmniDSP::OneMKL")
# endif()
# if(OMNIDSP_BACKEND_INTELIPP_ENABLED AND TARGET OmniDSP::IntelIPP)
#     target_link_libraries(OmniDSP PRIVATE OmniDSP::IntelIPP)
#     message(STATUS "Linking OmniDSP with OmniDSP::IntelIPP")
# endif()

# --- Generate Export Header for 'OmniDSP' library ---
set(EXPORT_HEADER_ABSOLUTE_PATH "${PROJECT_BINARY_DIR}/include/OmniDSP/omnidsp_export.hpp")
generate_export_header(OmniDSP
    BASE_NAME OMNIDSP
    EXPORT_FILE_NAME "${EXPORT_HEADER_ABSOLUTE_PATH}"
)

# --- Configure omnidsp_config.h ---
set(OMNIDSP_CONFIG_HPP_IN "${CMAKE_SOURCE_DIR}/include/OmniDSP/omnidsp_config.hpp.in")
set(OMNIDSP_CONFIG_HPP_OUT "${PROJECT_BINARY_DIR}/include/OmniDSP/omnidsp_config.hpp")

if(EXISTS "${OMNIDSP_CONFIG_HPP_IN}")
    message(STATUS "Generating ${OMNIDSP_CONFIG_HPP_OUT} from ${OMNIDSP_CONFIG_HPP_IN}")
    configure_file(
        "${OMNIDSP_CONFIG_HPP_IN}"
        "${OMNIDSP_CONFIG_HPP_OUT}"
    )
else()
    message(WARNING "omnidsp_config.hpp.in not found at ${OMNIDSP_CONFIG_HPP_IN}. Please ensure this template file exists.")
endif()

# --- Target Compile Definitions for 'OmniDSP' library ---
# These definitions are for omnidsp_config.hpp and allow C++ code to check
# which backends were reported as enabled by their respective CMakeLists.txt files.
# The OMNIDSP_INTERNAL_USE_* definitions are now set by individual backend CMake files.
target_compile_definitions(OmniDSP
    PRIVATE
        ${OMNIDSP_COMPILE_DEFINITIONS}

        $<$<STREQUAL:${OMNIDSP_BACKEND_DEFAULT_ENABLED},1>:OMNIDSP_BACKEND_DEFAULT_ENABLED=1>
        $<$<STREQUAL:${OMNIDSP_BACKEND_ACCELERATE_ENABLED},1>:OMNIDSP_BACKEND_ACCELERATE_ENABLED=1>
        $<$<STREQUAL:${OMNIDSP_BACKEND_ONEMKL_ENABLED},1>:OMNIDSP_BACKEND_ONEMKL_ENABLED=1>
        $<$<STREQUAL:${OMNIDSP_BACKEND_INTELIPP_ENABLED},1>:OMNIDSP_BACKEND_INTELIPP_ENABLED=1>
)

# --- Report Enabled Backends ---
set(OMNIDSP_ENABLED_BACKEND_NAMES_LIST "")
if(OMNIDSP_BACKEND_DEFAULT_ENABLED)
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES_LIST "Default")
endif()
if(OMNIDSP_BACKEND_ACCELERATE_ENABLED)
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES_LIST "Accelerate")
endif()
if(OMNIDSP_BACKEND_ONEMKL_ENABLED)
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES_LIST "oneMKL")
endif()
if(OMNIDSP_BACKEND_INTELIPP_ENABLED)
    list(APPEND OMNIDSP_ENABLED_BACKEND_NAMES_LIST "IntelIPP")
endif()

if(NOT OMNIDSP_ENABLED_BACKEND_NAMES_LIST)
    message(FATAL_ERROR "No OmniDSP backends were configured successfully. Default backend must be available.")
elseif(NOT OMNIDSP_BACKEND_DEFAULT_ENABLED)
     message(FATAL_ERROR "The Default OmniDSP backend failed to configure. This backend is mandatory.")
endif()

list(REMOVE_DUPLICATES OMNIDSP_ENABLED_BACKEND_NAMES_LIST)
message(STATUS "Successfully configured and enabled OmniDSP Backends: ${OMNIDSP_ENABLED_BACKEND_NAMES_LIST}")

message(STATUS "Finished configuring 'OmniDSP' library and backends.")
