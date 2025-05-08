# OmniDSP/src/omnidsp/intelipp/CMakeLists.txt
# Configures the Intel IPP backend for the 'OmniDSP' library.
# This version assumes you have moved a custom FindIPP.cmake
# from a project-level cmake/ directory to this directory (src/omnidsp/intelipp/).

# This file is included if OPTION_OMNIDSP_ENABLE_INTELIPP is ON.

message(STATUS "  Configuring Intel IPP backend...")

# --- Platform Check (Optional) ---
# if(APPLE)
#     message(STATUS "    Intel IPP backend is typically not used on Apple platforms. Skipping.")
#     return()
# endif()

# --- Find Intel IPP Package ---
# Temporarily add the current directory to CMAKE_MODULE_PATH
# so that find_package(IPP) can find a FindIPP.cmake located here.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "    Temporarily added ${CMAKE_CURRENT_SOURCE_DIR} to CMAKE_MODULE_PATH for FindIPP.cmake")

# Now, find_package will look in this directory first for FindIPP.cmake
find_package(IPP COMPONENTS s QUIET) # 's' for signal processing.

# It's good practice to remove the path after it's no longer needed,
# to avoid unintended side effects on other find_package calls elsewhere
# if this file were to be included in a different context.
list(REMOVE_ITEM CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "    Removed ${CMAKE_CURRENT_SOURCE_DIR} from CMAKE_MODULE_PATH")

# Standard handling for package finding results
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IPP
    FOUND_VAR IPP_FOUND
    REQUIRED_VARS IPP_FOUND # Requires IPP_FOUND to be true
    VERSION_VAR IPP_VERSION # Instruct FPHSA to use IPP_VERSION for messages
    FAIL_MESSAGE "Intel IPP (Signal Processing component 's', IPP::IPP target) not found. Ensure IPP is installed and IPPROOT or CMAKE_PREFIX_PATH is set, or FindIPP.cmake is correctly located and working."
)

if(NOT IPP_FOUND OR NOT TARGET IPP::IPP)
    message(WARNING "    Intel IPP package found (IPP_FOUND=${IPP_FOUND}) but target IPP::IPP is missing or invalid. Intel IPP backend will be disabled.")
    return() # Exit if IPP not properly found or target missing
endif()

message(STATUS "    Found Intel IPP (IPP::IPP target). Version: ${IPP_VERSION}")

# --- Define Intel IPP Backend Source Files ---
set(INTELIPP_BACKEND_SOURCES_INTERNAL
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/iir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/resample.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/window.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp"
)

# --- Add Sources to the 'OmniDSP' library target ---
target_sources(OmniDSP PRIVATE ${INTELIPP_BACKEND_SOURCES_INTERNAL})

# --- Include Directories for Intel IPP Backend ---
target_include_directories(OmniDSP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Create Interface Target for Intel IPP ---
if(NOT TARGET OmniDSP::IntelIPP)
    add_library(OmniDSP::IntelIPP INTERFACE IMPORTED GLOBAL)
    target_link_libraries(OmniDSP::IntelIPP INTERFACE IPP::IPP)
    message(STATUS "    Created OmniDSP::IntelIPP interface target.")
endif()

# --- Compile Definition for Intel IPP Backend ---
target_compile_definitions(OmniDSP PRIVATE OMNIDSP_INTERNAL_USE_INTELIPP)

# --- Signal Success and Set Configuration Variable ---
set(OMNIDSP_BACKEND_INTELIPP_ENABLED 1 PARENT_SCOPE)

message(STATUS "    Intel IPP backend configured successfully.")
