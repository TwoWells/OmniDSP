# - Find Intel Integrated Performance Primitives (IPP) Signal Processing Component
#
# This module finds if the Intel IPP signal processing library is available
# using CMake's find_package mechanism.
#
# Adds the following variables:
#  IPP_FOUND                - TRUE if IPP library/target is found by find_package.
#  OMNIDSP_ENABLED_INTELIPP - TRUE if IPP backend is enabled (requested and found). (CACHE BOOL)
#  INTELIPP_INCLUDE_DIRS    - Include directories provided by IPP::IPP. (Informational)
#  INTELIPP_LIBRARIES       - Link libraries/targets provided by IPP::IPP. (Informational, use target)
#
# Adds the following imported targets:
#  OmniDSP::intelipp        - INTERFACE target encapsulating IPP includes and libraries.
#
# Depends on:
#  OMNIDSP_ENABLE_INTELIPP - CMake option to request this backend.
#
# Hints:
# Ensure CMake can find the IPP installation. Set CMAKE_PREFIX_PATH,
# IPPROOT environment variable, or ONEAPI_ROOT environment variable.

# Use standard find package handling
include(FindPackageHandleStandardArgs)

# Initialize output variable
set(OMNIDSP_ENABLED_INTELIPP FALSE CACHE BOOL "Flag indicating Intel IPP backend is enabled")

# Define the variable for omnidsp_config.h
set(OMNIDSP_ENABLED_BACKEND_INTELIPP 0)

if(OMNIDSP_ENABLE_INTELIPP)
    message(STATUS "  Attempting to configure IntelIPP backend (enabled by option)...")

    # --- Find IPP Signal Processing Component ---
    # Use find_package, requesting the signal processing component ('s')
    # This should set IPP_FOUND.
    find_package(IPP COMPONENTS s QUIET) # 's' for signal processing

    # --- Handle Find Results ---
    # Use the IPP_FOUND variable set by find_package(IPP).
    # Explicitly require IPP_FOUND for FPHSA.
    find_package_handle_standard_args(IPP # Note: Using package name "IPP" here
        FOUND_VAR IPP_FOUND       # Trust the variable set by find_package(IPP)
        REQUIRED_VARS IPP_FOUND   # Explicitly require the FOUND variable itself
        # VERSION_VAR IPP_VERSION # Add if version checking is needed
        FAIL_MESSAGE "Intel IPP (Signal Processing component) not found (IPP_FOUND is FALSE). Please ensure IPP is installed and CMAKE_PREFIX_PATH or IPPROOT/ONEAPI_ROOT is set correctly."
    )

    # --- Set OmniDSP Specific Variable and Create Target ---
    # Proceed only if IPP was actually found (IPP_FOUND is TRUE)
    if(IPP_FOUND)
        # Double-check that the expected target exists before trying to use it
        if(TARGET IPP::IPP)
            # Set the final ENABLED flag to TRUE only if found and requested
            set(OMNIDSP_ENABLED_INTELIPP TRUE CACHE BOOL "Flag indicating Intel IPP backend is enabled" FORCE)
            # Define the variable for omnidsp_config.h
            set(OMNIDSP_ENABLED_BACKEND_INTELIPP 1)
            message(STATUS "    Found Intel IPP (IPP::IPP target): TRUE")

            # Retrieve properties from the found target (for information/verification if needed)
            get_target_property(TEMP_IPP_INCLUDES IPP::IPP INTERFACE_INCLUDE_DIRECTORIES)
            get_target_property(TEMP_IPP_LINK_LIBS IPP::IPP INTERFACE_LINK_LIBRARIES) # Might be empty if it just links the target
            set(INTELIPP_INCLUDE_DIRS "${TEMP_IPP_INCLUDES}")
            set(INTELIPP_LIBRARIES "${TEMP_IPP_LINK_LIBS}") # Primarily informational; linking IPP::IPP is preferred

            message(STATUS "      IPP Includes (from IPP::IPP): ${INTELIPP_INCLUDE_DIRS}")

            # Create INTERFACE target OmniDSP::intelipp if it doesn't exist
            if(NOT TARGET OmniDSP::intelipp)
                add_library(OmniDSP::intelipp INTERFACE IMPORTED GLOBAL)
                # Set properties to mirror the found IPP::IPP target
                target_include_directories(OmniDSP::intelipp INTERFACE
                    $<TARGET_PROPERTY:IPP::IPP,INTERFACE_INCLUDE_DIRECTORIES>
                )
                target_link_libraries(OmniDSP::intelipp INTERFACE IPP::IPP)
                target_compile_definitions(OmniDSP::intelipp INTERFACE
                     $<TARGET_PROPERTY:IPP::IPP,INTERFACE_COMPILE_DEFINITIONS>
                )
                 target_compile_options(OmniDSP::intelipp INTERFACE
                     $<TARGET_PROPERTY:IPP::IPP,INTERFACE_COMPILE_OPTIONS>
                )
                 target_link_options(OmniDSP::intelipp INTERFACE
                     $<TARGET_PROPERTY:IPP::IPP,INTERFACE_LINK_OPTIONS>
                )
                # Add compile definition to indicate IPP is used (can be checked in C++ code)
                target_compile_definitions(OmniDSP::intelipp INTERFACE OMNIDSP_USE_INTELIPP_IMPL=1)
            endif()
            message(STATUS "    IntelIPP backend configured successfully.")

        else()
            # This case indicates an inconsistency (IPP_FOUND true, but target missing)
            message(WARNING "Intel IPP found (IPP_FOUND=TRUE), but the required target 'IPP::IPP' does not exist. Cannot configure IntelIPP backend.")
            set(OMNIDSP_ENABLED_INTELIPP FALSE CACHE BOOL "Flag indicating Intel IPP backend is enabled" FORCE)
        endif()

    else()
        # IPP not found (IPP_FOUND is FALSE), ensure the flag is FALSE
        # FPHSA already printed the failure message.
        set(OMNIDSP_ENABLED_INTELIPP FALSE CACHE BOOL "Flag indicating Intel IPP backend is enabled" FORCE)
        # Clear informational variables
        set(INTELIPP_INCLUDE_DIRS "")
        set(INTELIPP_LIBRARIES "")
    endif()

else()
    # Backend explicitly disabled by option or platform dependency
    set(OMNIDSP_ENABLED_INTELIPP FALSE CACHE BOOL "Flag indicating Intel IPP backend is enabled" FORCE)
    message(STATUS "  IntelIPP backend disabled (OMNIDSP_ENABLE_INTELIPP=OFF or platform mismatch).")
endif()

# Mark internal variables as advanced
# mark_as_advanced(IPP_FOUND) # IPP_FOUND is standard
mark_as_advanced(INTELIPP_INCLUDE_DIRS INTELIPP_LIBRARIES)
