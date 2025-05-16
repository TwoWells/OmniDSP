# OmniDSP/src/omnidsp/backends/intelipp/CMakeLists.txt
# Configures the Intel IPP backend for the 'OmniDSP' library using a custom FindIPP.cmake.

# This script is processed if OPTION_OMNIDSP_ENABLE_INTELIPP is ON.
message(STATUS "    Configuring Intel IPP backend (using custom FindIPP.cmake)...")

# Initialize a local variable to track if this specific backend configures successfully.
# This will be propagated to the parent scope's OMNIDSP_BACKEND_INTELIPP_ENABLED.
set(OMNIDSP_BACKEND_INTELIPP_ENABLED_CONFIG_SUCCESS 0)

# --- Use Custom FindIPP.cmake ---
# Temporarily add the current directory to CMAKE_MODULE_PATH
# so that find_package(IPP) can find the FindIPP.cmake located here.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "      Temporarily added ${CMAKE_CURRENT_SOURCE_DIR} to CMAKE_MODULE_PATH for FindIPP.cmake")

# Call find_package for IPP. Your custom FindIPP.cmake will be invoked.
# Request components your backend implementation actually uses.
# The FindIPP.cmake you provided handles component dependency resolution.
find_package(IPP COMPONENTS s core QUIET) # 's' for signal, 'core' is fundamental.

list(REMOVE_ITEM CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "      Removed ${CMAKE_CURRENT_SOURCE_DIR} from CMAKE_MODULE_PATH")

# Your FindIPP.cmake module sets IPP_FOUND and creates the IPP::IPP target.
# We rely on those outputs.

if(NOT IPP_FOUND OR NOT TARGET IPP::IPP)
    message(WARNING "      Intel IPP not found or IPP::IPP target not created by custom FindIPP.cmake. Intel IPP backend will be disabled. (IPP_FOUND=${IPP_FOUND}, Target IPP::IPP Exists: $<TARGET_EXISTS:IPP::IPP>)")
    # Ensure the parent scope knows this backend is not enabled.
    set(OMNIDSP_BACKEND_INTELIPP_ENABLED ${OMNIDSP_BACKEND_INTELIPP_ENABLED_CONFIG_SUCCESS} PARENT_SCOPE)
    return() # Do not proceed with configuring this backend.
endif()

# If we reach here, IPP_FOUND is TRUE and the IPP::IPP target was created by FindIPP.cmake.
message(STATUS "      Found Intel IPP using custom FindIPP.cmake. Version: ${IPP_VERSION}")
# You can add more detailed messages about what IPP::IPP provides if needed for debugging:
# get_target_property(IPP_INTERFACE_INCLUDES IPP::IPP INTERFACE_INCLUDE_DIRECTORIES)
# get_target_property(IPP_INTERFACE_LIBS IPP::IPP INTERFACE_LINK_LIBRARIES)
# message(STATUS "        IPP::IPP Interface Includes: ${IPP_INTERFACE_INCLUDES}")
# message(STATUS "        IPP::IPP Interface Link Libraries: ${IPP_INTERFACE_LIBS}")

# --- Define Intel IPP Backend Source Files ---
set(INTELIPP_BACKEND_SOURCES_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/details.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/iir_filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/resample.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/window.cpp"
)

# --- Add Sources to the 'OmniDSP' library target ---
target_sources(OmniDSP PRIVATE ${INTELIPP_BACKEND_SOURCES_LIST})
message(STATUS "        Added Intel IPP backend sources to OmniDSP target.")

# --- Include Directories for Intel IPP Backend ---
# The IPP::IPP target should propagate necessary include directories via INTERFACE_INCLUDE_DIRECTORIES.
# We still need to add the current directory for this backend's own internal headers.
target_include_directories(OmniDSP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" # For its own headers like details.hpp
)

# --- Link Intel IPP libraries to OmniDSP target ---
# Link against the aggregate IPP::IPP interface target. This target, as defined
# in your FindIPP.cmake, should correctly bundle all necessary IPP component libraries,
# threading libraries (if enabled), and Intel runtime libraries (on Linux).
target_link_libraries(OmniDSP PRIVATE IPP::IPP)
message(STATUS "        Linked OmniDSP target with IPP::IPP interface target.")

# --- Compile Definition for Intel IPP Backend internal C++ code ---
# This definition is used in C++ to conditionally compile IPP-specific code paths.
target_compile_definitions(OmniDSP PRIVATE OMNIDSP_INTERNAL_USE_INTELIPP)

# --- Signal Success and Set Configuration Variable for omnidsp_config.hpp ---
set(OMNIDSP_BACKEND_INTELIPP_ENABLED_CONFIG_SUCCESS 1)
set(OMNIDSP_BACKEND_INTELIPP_ENABLED ${OMNIDSP_BACKEND_INTELIPP_ENABLED_CONFIG_SUCCESS} PARENT_SCOPE) # Propagate 1

message(STATUS "      Intel IPP backend configured and enabled successfully using custom FindIPP.cmake.")
message(STATUS "    Finished configuring Intel IPP backend.")
