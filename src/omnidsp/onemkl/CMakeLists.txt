# OmniDSP/src/omnidsp/onemkl/CMakeLists.txt
# Configures the oneMKL backend for the 'OmniDSP' library.

# This file is included if OPTION_OMNIDSP_ENABLE_ONEMKL is ON.

message(STATUS "  Configuring oneMKL backend...")

# --- Platform Check (Optional, if oneMKL has platform restrictions not covered by option) ---
# if(APPLE)
#     message(STATUS "    oneMKL backend is typically not used on Apple platforms with Accelerate. Skipping.")
#     return()
# endif()

# --- Find oneMKL Package ---
# Assuming MKL_ROOT or CMAKE_PREFIX_PATH is set for CMake to find MKL.
# The original onemkl.cmake used find_package(MKL QUIET) and then FPHSA.
include(FindPackageHandleStandardArgs)
find_package(MKL QUIET) # Sets MKL_FOUND, MKL_VERSION and potentially MKL::MKL target

find_package_handle_standard_args(MKL
    FOUND_VAR MKL_FOUND
    REQUIRED_VARS MKL_FOUND # Requires MKL_FOUND to be true
    VERSION_VAR MKL_VERSION # Instruct FPHSA to use MKL_VERSION for messages
    FAIL_MESSAGE "oneMKL (MKL::MKL target) not found. Please ensure MKL is installed and MKLROOT or CMAKE_PREFIX_PATH is set."
)

if(NOT MKL_FOUND OR NOT TARGET MKL::MKL)
    message(WARNING "    oneMKL package found (MKL_FOUND=${MKL_FOUND}) but target MKL::MKL is missing or invalid. oneMKL backend will be disabled.")
    # MKL_VERSION might not be set if MKL_FOUND is false.
    return() # Exit if MKL not properly found or target missing
endif()

# If MKL_FOUND is true, MKL_VERSION should be set by find_package(MKL)
message(STATUS "    Found oneMKL (MKL::MKL target). Version: ${MKL_VERSION}")

# --- Define oneMKL Backend Source Files ---
set(ONEMKL_BACKEND_SOURCES_INTERNAL
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp" # As listed in original target_definitions
)

# --- Add Sources to the 'OmniDSP' library target ---
target_sources(OmniDSP PRIVATE ${ONEMKL_BACKEND_SOURCES_INTERNAL})

# --- Include Directories for oneMKL Backend ---
target_include_directories(OmniDSP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Create Interface Target for oneMKL ---
# This mirrors the existing OmniDSP::OneMKL target creation.
if(NOT TARGET OmniDSP::OneMKL)
    add_library(OmniDSP::OneMKL INTERFACE IMPORTED GLOBAL)
    # Link the actual MKL target (MKL::MKL) to this interface target
    target_link_libraries(OmniDSP::OneMKL INTERFACE MKL::MKL)
    # Propagate usage requirements from MKL::MKL if necessary
    # target_include_directories(OmniDSP::OneMKL INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    # target_compile_definitions(OmniDSP::OneMKL INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_DEFINITIONS>)
    message(STATUS "    Created OmniDSP::OneMKL interface target.")
endif()

# --- Compile Definition for oneMKL Backend ---
target_compile_definitions(OmniDSP PRIVATE OMNIDSP_INTERNAL_USE_ONEMKL)

# --- Signal Success and Set Configuration Variable ---
set(OMNIDSP_BACKEND_ONEMKL_ENABLED 1 PARENT_SCOPE)

message(STATUS "    oneMKL backend configured successfully.")
