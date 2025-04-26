# ==============================================================================
# cmake/backend/accelerate.cmake
# ==============================================================================
#
# Brief Description:
#   Configures the 'omnidsp_backend_accelerate' OBJECT library target if the
#   Apple Accelerate framework is found and the backend is enabled via the
#   OMNIDSP_ENABLE_BACKEND_ACCELERATE option.
#
# Variables Read:
#   - CMAKE_SOURCE_DIR, PROJECT_BINARY_DIR (CMake Built-in)
#   - APPLE (CMake Built-in)
#   - OMNIDSP_ENABLE_BACKEND_ACCELERATE (Option, should depend on APPLE)
#   - CMAKE_CURRENT_LIST_DIR (CMake Built-in)
#
# Variables Set (in Caller's Scope via include()):
#   - OMNIDSP_SELECTED_BACKEND_TARGETS (Appends 'omnidsp_backend_accelerate' if configured)
#
# Targets Defined:
#   - omnidsp_backend_accelerate (OBJECT Library, if configured successfully)
#
# Modules Included:
#   - FindAccelerate (implicitly via find_framework)
#
# ==============================================================================

if(NOT APPLE)
    message(STATUS "  Accelerate backend is only available on Apple platforms. Skipping configuration.")
    if(OMNIDSP_ENABLE_BACKEND_ACCELERATE)
         message(WARNING "OMNIDSP_ENABLE_BACKEND_ACCELERATE was ON but platform is not Apple. Forcing OFF.")
         set(OMNIDSP_ENABLE_BACKEND_ACCELERATE OFF CACHE BOOL "Accelerate backend disabled because platform is not Apple" FORCE)
    endif()
    return()
endif()

if(OMNIDSP_ENABLE_BACKEND_ACCELERATE)
    message(STATUS "  Attempting to configure Accelerate backend...")

    find_framework(Accelerate REQUIRED)
    message(STATUS "    Found Accelerate framework.")

    add_library(omnidsp_backend_accelerate OBJECT)

    set(OMNIDSP_ACCELERATE_BACKEND_SRC_DIR ${CMAKE_SOURCE_DIR}/src/omnidsp/backend/accelerate)
    message(STATUS "      Accelerate backend source directory: ${OMNIDSP_ACCELERATE_BACKEND_SRC_DIR}")

    target_sources(omnidsp_backend_accelerate
        PRIVATE
            "${OMNIDSP_ACCELERATE_BACKEND_SRC_DIR}/fft.cpp"
            "${OMNIDSP_ACCELERATE_BACKEND_SRC_DIR}/window.cpp"
            "${OMNIDSP_ACCELERATE_BACKEND_SRC_DIR}/convolution.cpp"
            "${OMNIDSP_ACCELERATE_BACKEND_SRC_DIR}/resample.cpp"
            "${OMNIDSP_ACCELERATE_BACKEND_SRC_DIR}/backend.cpp"
    )

    # --- Backend Include Directories ---
    target_include_directories(omnidsp_backend_accelerate
        PUBLIC # Public headers needed by consumers (main include dir)
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>

        PRIVATE # Private headers needed only to compile this backend's sources
            ${OMNIDSP_ACCELERATE_BACKEND_SRC_DIR}
            # Add path to generated export header
            ${PROJECT_BINARY_DIR}/include # Or CMAKE_BINARY_DIR
    )

    target_compile_definitions(omnidsp_backend_accelerate
        INTERFACE
            OMNIDSP_USE_ACCELERATE=1
    )

    # Link the Accelerate framework. INTERFACE ensures the main 'omnidsp'
    # library also links against it. Include paths for frameworks are
    # typically handled differently and don't need explicit addition here.
    target_link_libraries(omnidsp_backend_accelerate
        INTERFACE
            "-framework Accelerate"
    )

    list(APPEND OMNIDSP_SELECTED_BACKEND_TARGETS "omnidsp_backend_accelerate")
    message(STATUS "    Configured Accelerate backend target: omnidsp_backend_accelerate")

else()
    message(STATUS "  Accelerate backend disabled (OMNIDSP_ENABLE_BACKEND_ACCELERATE=OFF).")
endif()
