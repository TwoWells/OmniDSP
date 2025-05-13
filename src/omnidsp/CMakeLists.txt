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
set(OMNIDSP_BACKEND_DEFAULT_ENABLED 0)
set(OMNIDSP_BACKEND_ACCELERATE_ENABLED 0)
set(OMNIDSP_BACKEND_ONEMKL_ENABLED 0)
set(OMNIDSP_BACKEND_INTELIPP_ENABLED 0)

# --- Define Core Library Sources ---
set(OMNIDSP_CORE_HEADERS
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/convolution.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/core_types.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/cqt.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/fft.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/filter.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/omnidsp.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/resample.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/utils.hpp"
    "${CMAKE_SOURCE_DIR}/include/OmniDSP/window.hpp"
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
)
set(OMNIDSP_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/omnidsp.cpp" # Main class implementation
    # Params implementations (constructors)
    "${CMAKE_CURRENT_SOURCE_DIR}/params/fir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/params/resample.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/params/cqt.cpp"
    # Interface implementations (Pimpl wrappers)
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/cqt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/fir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/iir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/resample.cpp"
    # Utils implementations
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/cqt_design.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/filter_design.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/iir_filter_design.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/resample_design.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/resample.cpp"
    # Dispatcher
    "${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/backend.cpp"
)

# --- Create the Main 'OmniDSP' Library Target ---
add_library(OmniDSP ${OMNIDSP_LIBRARY_TYPE} "")
target_sources(OmniDSP PRIVATE ${OMNIDSP_CORE_SOURCES})

# --- DIAGNOSTICS: Check actual target type ---
if(TARGET OmniDSP)
    get_target_property(OMNIDSP_ACTUAL_TARGET_TYPE OmniDSP TYPE)
    message(STATUS "DEBUG: After add_library(OmniDSP): Actual target type = ${OMNIDSP_ACTUAL_TARGET_TYPE}")
else()
    message(FATAL_ERROR "DEBUG: Target OmniDSP was NOT created by add_library!")
endif()

# --- Public and Private Include Directories for 'OmniDSP' library ---
target_include_directories(OmniDSP
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        # This ensures the compiler can find headers in PROJECT_BINARY_DIR/include
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} # For internal includes like "interface/backend.hpp" from "omnidsp.cpp"
                                    # or "utils/internal_helper.hpp" from "utils/some_util.cpp"
)

# --- Common Dependencies ---
# Link against Google Highway (found in cmake/depend.cmake)
if(TARGET hwy::hwy)
    target_link_libraries(OmniDSP PRIVATE hwy::hwy)
    message(STATUS "Linking OmniDSP with hwy::hwy")
else()
   message(WARNING "hwy::hwy target not found after find_package. Linking might fail.")
endif()

# Link against spdlog (found in cmake/depend.cmake)
if(TARGET spdlog::spdlog)
   target_link_libraries(OmniDSP PRIVATE spdlog::spdlog)
   message(STATUS "Linking OmniDSP with spdlog::spdlog")
else()
   message(WARNING "spdlog::spdlog target not found after find_package. Linking might fail.")
endif()

# --- Include Backend Configurations ---
add_subdirectory(default)
add_subdirectory(accelerate)

# oneMKL Backend
if(OPTION_OMNIDSP_ENABLE_ONEMKL)
    message(STATUS "Attempting to configure oneMKL backend (OPTION_OMNIDSP_ENABLE_ONEMKL=ON)...")
    add_subdirectory(onemkl)
else()
    message(STATUS "oneMKL backend disabled by option.")
endif()

# Intel IPP Backend
if(OPTION_OMNIDSP_ENABLE_INTELIPP)
    message(STATUS "Attempting to configure Intel IPP backend (OPTION_OMNIDSP_ENABLE_INTELIPP=ON)...")
    add_subdirectory(intelipp)
else()
    message(STATUS "Intel IPP backend disabled by option.")
endif()

# --- Post Backend Configuration: Linking Interface Targets ---
if(OMNIDSP_BACKEND_ACCELERATE_ENABLED AND TARGET OmniDSP::Accelerate)
    target_link_libraries(OmniDSP PRIVATE OmniDSP::Accelerate)
    message(STATUS "Linking OmniDSP with OmniDSP::Accelerate")
endif()
if(OMNIDSP_BACKEND_ONEMKL_ENABLED AND TARGET OmniDSP::OneMKL)
    target_link_libraries(OmniDSP PRIVATE OmniDSP::OneMKL)
    message(STATUS "Linking OmniDSP with OmniDSP::OneMKL")
endif()
if(OMNIDSP_BACKEND_INTELIPP_ENABLED AND TARGET OmniDSP::IntelIPP)
    target_link_libraries(OmniDSP PRIVATE OmniDSP::IntelIPP)
    message(STATUS "Linking OmniDSP with OmniDSP::IntelIPP")
endif()

# --- Generate Export Header for 'OmniDSP' library ---
set(EXPORT_HEADER_ABSOLUTE_PATH "${PROJECT_BINARY_DIR}/include/OmniDSP/omnidsp_export.hpp")
generate_export_header(OmniDSP
    BASE_NAME OMNIDSP
    EXPORT_FILE_NAME "${EXPORT_HEADER_ABSOLUTE_PATH}" # Use absolute path
)

# --- Configure omnidsp_config.h ---
# Reflecting .hpp extension as seen in user's CMake log
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
    # If you intend to use .h extension, revert the names above and ensure your template is .h.in
endif()

# --- Target Compile Definitions for 'OmniDSP' library ---
target_compile_definitions(OmniDSP
    PRIVATE
        ${OMNIDSP_COMPILE_DEFINITIONS} # Common definitions

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
