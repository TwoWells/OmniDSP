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

# Initialize output variables
set(OMNIDSP_HAS_ONEMKL FALSE)
set(OMNIDSP_HAS_BACKEND_ONEMKL_VALUE 0) # Default to 0
set(ONEMKL_BACKEND_SOURCES "")
set(ONEMKL_BACKEND_INCLUDE_DIRS "")
set(ONEMKL_BACKEND_LINK_LIBS "")

if(OMNIDSP_ENABLE_ONEMKL)
    message(STATUS "  Attempting to configure oneMKL backend...")

    find_package(MKL HINTS ENV MKLROOT QUIET)
    find_package(IPP COMPONENTS s QUIET)

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
        )

        # --- Set Backend Include Directories ---
        set(TEMP_ONEMKL_INCLUDES "")
        if(TARGET MKL::MKL)
            get_target_property(MKL_INCLUDES MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
            if(MKL_INCLUDES)
                list(APPEND TEMP_ONEMKL_INCLUDES ${MKL_INCLUDES})
            endif()
        endif()
        if(TARGET IPP::IPP)
             get_target_property(IPP_INCLUDES IPP::IPP INTERFACE_INCLUDE_DIRECTORIES)
             if(IPP_INCLUDES)
                 list(APPEND TEMP_ONEMKL_INCLUDES ${IPP_INCLUDES})
             endif()
        endif()
        list(REMOVE_DUPLICATES TEMP_ONEMKL_INCLUDES)
        set(ONEMKL_BACKEND_INCLUDE_DIRS ${TEMP_ONEMKL_INCLUDES})

        # --- Set Backend Link Libraries/Targets ---
        set(ONEMKL_BACKEND_LINK_LIBS MKL::MKL IPP::IPP)

        # --- Signal Success and Set Config Variables ---
        set(OMNIDSP_HAS_ONEMKL TRUE)
        set(OMNIDSP_HAS_BACKEND_ONEMKL_VALUE 1) # Set value to 1 on success
        message(STATUS "      oneMKL backend configured.")

    else()
        message(WARNING "oneMKL backend enabled (OMNIDSP_ENABLE_ONEMKL=ON), but required libraries (MKL::MKL, IPP::IPP) were not found. Disabling this backend.")
        # OMNIDSP_HAS_ONEMKL remains FALSE
        # OMNIDSP_HAS_BACKEND_ONEMKL_VALUE remains 0
    endif()
else()
    message(STATUS "  oneMKL backend disabled (OMNIDSP_ENABLE_ONEMKL=OFF or platform is Apple).")
    # OMNIDSP_HAS_ONEMKL remains FALSE
    # OMNIDSP_HAS_BACKEND_ONEMKL_VALUE remains 0
endif()
