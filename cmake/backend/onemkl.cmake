# ==============================================================================
# cmake/backend/onemkl.cmake (Refactored v4 - Fix FPHSA Call v2)
# ==============================================================================
# Finds Intel oneMKL dependency and sets up an interface target for the MKL backend.
#
# Adds the following variables:
#  MKL_FOUND              - TRUE if MKL library/target is found by find_package.
#  OMNIDSP_ENABLED_ONEMKL - TRUE if MKL backend is enabled (requested and found). (CACHE BOOL)
#
# Adds the following imported targets:
#  OmniDSP::onemkl        - INTERFACE target encapsulating MKL includes and libraries.
#
# Depends on:
#  OMNIDSP_ENABLE_ONEMKL  - CMake option to request this backend.

include(CMakeDependentOption)
include(FindPackageHandleStandardArgs) # Include for find_package_handle_standard_args

# Initialize output variable
set(OMNIDSP_ENABLED_ONEMKL FALSE CACHE BOOL "Flag indicating Intel MKL backend is enabled")

# Define the variable for omnidsp_config.h
set(OMNIDSP_ENABLED_BACKEND_ONEMKL 0)

if(OMNIDSP_ENABLE_ONEMKL)
    message(STATUS "  Attempting to configure oneMKL backend (enabled by option)...")

    # --- Find MKL ---
    # Use find_package to locate MKL. This should set MKL_FOUND.
    find_package(MKL QUIET) # Find the main MKL configuration

    # --- Handle Find Results ---
    # Use the MKL_FOUND variable set by find_package(MKL).
    # Explicitly require MKL_FOUND for FPHSA.
    find_package_handle_standard_args(MKL
        FOUND_VAR MKL_FOUND       # Trust the variable set by find_package(MKL)
        REQUIRED_VARS MKL_FOUND   # Explicitly require the FOUND variable itself
        FAIL_MESSAGE "Intel MKL not found (MKL_FOUND is FALSE). Please ensure MKL is installed and CMAKE_PREFIX_PATH or MKLROOT/ONEAPI_ROOT is set correctly."
    )

    # --- Set OmniDSP Specific Variable and Create Target ---
    # Proceed only if MKL was actually found (MKL_FOUND is TRUE)
    if(MKL_FOUND)
        # Double-check that the expected target exists before trying to use it
        if(TARGET MKL::MKL)
            # Set the final ENABLED flag to TRUE only if found and requested
            set(OMNIDSP_ENABLED_ONEMKL TRUE CACHE BOOL "Flag indicating Intel MKL backend is enabled" FORCE)
            # Define the variable for omnidsp_config.h
            set(OMNIDSP_ENABLED_BACKEND_ONEMKL 1)
            message(STATUS "    Found Intel MKL (MKL::MKL target): TRUE")

            # Retrieve properties from the found target (for information/verification if needed)
            get_target_property(TEMP_MKL_INCLUDES MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "      MKL Includes (from MKL::MKL): ${TEMP_MKL_INCLUDES}")

            # Create INTERFACE target OmniDSP::onemkl if it doesn't exist
            if(NOT TARGET OmniDSP::onemkl)
                add_library(OmniDSP::onemkl INTERFACE IMPORTED GLOBAL)

                # Set properties to mirror the found MKL::MKL target
                target_include_directories(OmniDSP::onemkl INTERFACE
                    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
                )
                target_link_libraries(OmniDSP::onemkl INTERFACE MKL::MKL)
                target_compile_definitions(OmniDSP::onemkl INTERFACE
                     $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_DEFINITIONS>
                )
                 target_compile_options(OmniDSP::onemkl INTERFACE
                     $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
                )
                 target_link_options(OmniDSP::onemkl INTERFACE
                     $<TARGET_PROPERTY:MKL::MKL,INTERFACE_LINK_OPTIONS>
                )
                # Add compile definition to indicate MKL is used (can be checked in C++ code)
                target_compile_definitions(OmniDSP::onemkl INTERFACE OMNIDSP_USE_ONEMKL_IMPL=1)
            endif()
            message(STATUS "    oneMKL backend configured successfully.")

        else()
            # This case indicates an inconsistency (MKL_FOUND true, but target missing)
            message(WARNING "Intel MKL found (MKL_FOUND=TRUE), but the required target 'MKL::MKL' does not exist. Cannot configure oneMKL backend.")
            set(OMNIDSP_ENABLED_ONEMKL FALSE CACHE BOOL "Flag indicating Intel MKL backend is enabled" FORCE)
        endif()

    else()
        # MKL not found (MKL_FOUND is FALSE), ensure the flag is FALSE
        # FPHSA already printed the failure message.
        set(OMNIDSP_ENABLED_ONEMKL FALSE CACHE BOOL "Flag indicating Intel MKL backend is enabled" FORCE)
    endif()

else()
    # Backend explicitly disabled by option or platform dependency
    set(OMNIDSP_ENABLED_ONEMKL FALSE CACHE BOOL "Flag indicating Intel MKL backend is enabled" FORCE)
    message(STATUS "  oneMKL backend disabled (OMNIDSP_ENABLE_ONEMKL=OFF or platform mismatch).")
endif()

# Mark internal variable as advanced
# mark_as_advanced(MKL_FOUND) # MKL_FOUND is standard, maybe don't mark as advanced
