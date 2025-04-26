# ==============================================================================
# cmake/backend/onemkl.cmake
# ==============================================================================
#
# Brief Description:
#   Configures the 'omnidsp_backend_onemkl' OBJECT library target if the
#   Intel oneMKL and IPP libraries are found and the backend is enabled.
#
# Variables Read:
#   - CMAKE_SOURCE_DIR, PROJECT_BINARY_DIR (CMake Built-in)
#   - APPLE (CMake Built-in)
#   - OMNIDSP_ENABLE_ONEMKL (Option, dependent on NOT APPLE)
#   - ENV{MKLROOT} (Environment Variable)
#   - CMAKE_CURRENT_LIST_DIR (CMake Built-in)
#   - MKL::MKL, IPP::core, IPP::s targets (Expected from find_package)
#   - IPP_INCLUDE_DIRS (Variable potentially set by FindIPP.cmake)
#
# Variables Set (in Caller's Scope via include()):
#   - OMNIDSP_SELECTED_BACKEND_TARGETS (Appends 'omnidsp_backend_onemkl' if configured)
#
# Targets Defined:
#   - omnidsp_backend_onemkl (OBJECT Library, if configured successfully)
#
# Modules Included:
#   - CMakeDependentOption, FindMKL, FindIPP
#
# ==============================================================================

include(CMakeDependentOption)

cmake_dependent_option(
    OMNIDSP_ENABLE_ONEMKL
    "Enable Intel oneMKL backend (requires non-Apple OS and MKL/IPP libs)"
    ON "NOT APPLE" OFF
)

if(OMNIDSP_ENABLE_ONEMKL)
    message(STATUS "  Attempting to configure oneMKL backend...")

    find_package(MKL HINTS ENV MKLROOT QUIET)
    find_package(IPP COMPONENTS s QUIET) # Assumes FindIPP.cmake provides IPP::core, IPP::s

    if(TARGET MKL::MKL AND TARGET IPP::core AND TARGET IPP::s)
        message(STATUS "    Found oneMKL (MKL::MKL) and IPP (IPP::core, IPP::s). Configuring target.")

        add_library(omnidsp_backend_onemkl OBJECT)

        set(OMNIDSP_ONEMKL_BACKEND_SRC_DIR ${CMAKE_SOURCE_DIR}/src/omnidsp/onemkl)
        message(STATUS "      oneMKL backend source directory: ${OMNIDSP_ONEMKL_BACKEND_SRC_DIR}")

        target_sources(omnidsp_backend_onemkl
            PRIVATE
                "${OMNIDSP_ONEMKL_BACKEND_SRC_DIR}/backend.cpp"
                "${OMNIDSP_ONEMKL_BACKEND_SRC_DIR}/fft.cpp"
                "${OMNIDSP_ONEMKL_BACKEND_SRC_DIR}/window.cpp"
                "${OMNIDSP_ONEMKL_BACKEND_SRC_DIR}/convolution.cpp"
                "${OMNIDSP_ONEMKL_BACKEND_SRC_DIR}/resample.cpp"
        )

        # --- Get Include Directories from Dependencies ---
        set(MKL_INCLUDES "")
        set(IPP_INCLUDES "") # Consolidate IPP includes

        # MKL Includes
        if(TARGET MKL::MKL)
            get_target_property(MKL_INCLUDES MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
            if(NOT MKL_INCLUDES) # Check if property was empty or not found
                 message(WARNING "DEBUG oneMKL.cmake: MKL::MKL INTERFACE_INCLUDE_DIRECTORIES not set/empty. Check FindMKL.")
                 set(MKL_INCLUDES "") # Ensure it's empty if property failed
            else()
                 message(STATUS "DEBUG oneMKL.cmake: Retrieved MKL_INCLUDES: ${MKL_INCLUDES}")
            endif()
        else()
            message(WARNING "DEBUG oneMKL.cmake: MKL::MKL target not found.")
        endif()

        # IPP Includes (Check target properties first, then fallback to IPP_INCLUDE_DIRS)
        if(TARGET IPP::core)
             get_target_property(TEMP_IPP_CORE_INCLUDES IPP::core INTERFACE_INCLUDE_DIRECTORIES)
             if(TEMP_IPP_CORE_INCLUDES)
                 list(APPEND IPP_INCLUDES ${TEMP_IPP_CORE_INCLUDES})
                 message(STATUS "DEBUG oneMKL.cmake: Retrieved IPP_CORE_INCLUDES from property: ${TEMP_IPP_CORE_INCLUDES}")
             endif()
        else()
             message(WARNING "DEBUG oneMKL.cmake: IPP::core target not found.")
        endif()
        if(TARGET IPP::s)
             get_target_property(TEMP_IPP_S_INCLUDES IPP::s INTERFACE_INCLUDE_DIRECTORIES)
              if(TEMP_IPP_S_INCLUDES)
                 list(APPEND IPP_INCLUDES ${TEMP_IPP_S_INCLUDES})
                 message(STATUS "DEBUG oneMKL.cmake: Retrieved IPP_S_INCLUDES from property: ${TEMP_IPP_S_INCLUDES}")
             endif()
        else()
              message(WARNING "DEBUG oneMKL.cmake: IPP::s target not found.")
        endif()

        # Fallback/Primary check for IPP_INCLUDE_DIRS variable if properties were empty
        if(NOT IPP_INCLUDES)
            if(DEFINED IPP_INCLUDE_DIRS AND IPP_INCLUDE_DIRS)
                 message(STATUS "DEBUG oneMKL.cmake: IPP target properties did not provide includes. Falling back to IPP_INCLUDE_DIRS: ${IPP_INCLUDE_DIRS}")
                 set(IPP_INCLUDES ${IPP_INCLUDE_DIRS})
            else()
                 message(WARNING "DEBUG oneMKL.cmake: IPP target properties did not provide includes, AND IPP_INCLUDE_DIRS is not defined/empty.")
            endif()
        endif()
        # Remove duplicates if properties and variable both added paths
        if(IPP_INCLUDES)
            list(REMOVE_DUPLICATES IPP_INCLUDES)
            message(STATUS "DEBUG oneMKL.cmake: Final IPP_INCLUDES: ${IPP_INCLUDES}")
        endif()


        # --- Backend Include Directories ---
        message(STATUS "DEBUG oneMKL.cmake: Using MKL_INCLUDES='${MKL_INCLUDES}'")
        message(STATUS "DEBUG oneMKL.cmake: Using IPP_INCLUDES='${IPP_INCLUDES}'")

        target_include_directories(omnidsp_backend_onemkl
            PUBLIC # Public headers needed by consumers (main include dir)
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>

            PRIVATE # Private headers needed only to compile this backend's sources
                ${OMNIDSP_ONEMKL_BACKEND_SRC_DIR}
                # Add path to generated export header
                ${PROJECT_BINARY_DIR}/include # Or CMAKE_BINARY_DIR

                # Explicitly add include directories from MKL and IPP targets/variables
                ${MKL_INCLUDES}
                ${IPP_INCLUDES} # Use the consolidated list
        )

        target_compile_definitions(omnidsp_backend_onemkl
            INTERFACE # Propagated to the main library target
                OMNIDSP_USE_ONEMKL=1 # Indicate this backend is active
        )

        # Link MKL and IPP targets. INTERFACE ensures the main 'omnidsp'
        # library also links against them (important for shared libs).
        # The include paths are now handled explicitly above for OBJECT lib compilation.
        target_link_libraries(omnidsp_backend_onemkl
            INTERFACE
                MKL::MKL
                IPP::core
                IPP::s
        )

        list(APPEND OMNIDSP_SELECTED_BACKEND_TARGETS "omnidsp_backend_onemkl")
        message(STATUS "    Configured oneMKL backend target: omnidsp_backend_onemkl")

    else()
        message(WARNING "oneMKL backend enabled (OMNIDSP_ENABLE_ONEMKL=ON), but required libraries (MKL::MKL, IPP::core, IPP::s) were not found. Disabling this backend.")
    endif()
else()
    message(STATUS "  oneMKL backend disabled (OMNIDSP_ENABLE_ONEMKL=OFF or platform is Apple).")
endif()
