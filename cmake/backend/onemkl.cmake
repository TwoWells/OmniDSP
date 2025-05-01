# ==============================================================================
# cmake/backend/onemkl.cmake
# ==============================================================================
# Finds dependencies and sets variables for the oneMKL backend.

include(CMakeDependentOption)

cmake_dependent_option(
    OMNIDSP_ENABLE_ONEMKL
    "Enable Intel oneMKL backend (requires non-Apple OS and MKL/IPP libs)"
    ON "NOT APPLE" OFF
)

# Initialize output variables in the current scope (which is the includer's scope)
set(OMNIDSP_HAS_ONEMKL FALSE)
set(ONEMKL_BACKEND_SOURCES "")
set(ONEMKL_BACKEND_INCLUDE_DIRS "")
set(ONEMKL_BACKEND_LINK_LIBS "")

if(OMNIDSP_ENABLE_ONEMKL)
    message(STATUS "  Attempting to configure oneMKL backend...")

    # Find MKL and IPP (IPP::IPP is the key target we need from FindIPP)
    find_package(MKL HINTS ENV MKLROOT QUIET)
    # Requesting 's' component ensures FindIPP checks for core and signal libs
    # and defines IPP::IPP if successful.
    find_package(IPP COMPONENTS s QUIET)

    # Check if the required dependency targets exist
    if(TARGET MKL::MKL AND TARGET IPP::IPP)
        message(STATUS "      Found oneMKL (MKL::MKL) and IPP (IPP::IPP). Enabling backend.")

        # --- Set Backend Source Files ---
        set(ONEMKL_BACKEND_SOURCES
            "${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/backend.cpp"
            "${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/fft.cpp"
            "${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/window.cpp"
            "${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/convolution.cpp"
            "${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/resample.cpp"
            "${PROJECT_SOURCE_DIR}/src/omnidsp/onemkl/filter.cpp"
            # PARENT_SCOPE removed
        )

        # --- Set Backend Include Directories ---
        set(TEMP_ONEMKL_INCLUDES "")
        # Get MKL Includes
        if(TARGET MKL::MKL)
            get_target_property(MKL_INCLUDES MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
            if(MKL_INCLUDES)
                list(APPEND TEMP_ONEMKL_INCLUDES ${MKL_INCLUDES})
            endif()
        endif()
        # Get IPP Includes (from aggregate target)
        if(TARGET IPP::IPP)
             get_target_property(IPP_INCLUDES IPP::IPP INTERFACE_INCLUDE_DIRECTORIES)
             if(IPP_INCLUDES)
                 list(APPEND TEMP_ONEMKL_INCLUDES ${IPP_INCLUDES})
             endif()
        endif()
        list(REMOVE_DUPLICATES TEMP_ONEMKL_INCLUDES)
        set(ONEMKL_BACKEND_INCLUDE_DIRS ${TEMP_ONEMKL_INCLUDES}) # PARENT_SCOPE removed

        # --- Set Backend Link Libraries/Targets ---
        set(ONEMKL_BACKEND_LINK_LIBS MKL::MKL IPP::IPP) # PARENT_SCOPE removed

        # --- Signal Success ---
        set(OMNIDSP_HAS_ONEMKL TRUE) # PARENT_SCOPE removed
        message(STATUS "      oneMKL backend configured.")

    else()
        message(WARNING "oneMKL backend enabled (OMNIDSP_ENABLE_ONEMKL=ON), but required libraries (MKL::MKL, IPP::IPP) were not found. Disabling this backend.")
        # OMNIDSP_HAS_ONEMKL remains FALSE
    endif()
else()
    message(STATUS "  oneMKL backend disabled (OMNIDSP_ENABLE_ONEMKL=OFF or platform is Apple).")
    # OMNIDSP_HAS_ONEMKL remains FALSE
endif()
