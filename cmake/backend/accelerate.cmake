# cmake/backend/accelerate.cmake
# ==============================
# Detects and configures the Apple Accelerate framework backend.
# This script checks if the build is happening on an Apple platform and
# attempts to find the Accelerate framework.
#
# Variables Set (Current Scope, if Accelerate found):
# - OMNIDSP_HAS_ACCELERATE (BOOL): TRUE if Accelerate is found.
# - ACCELERATE_COMPILE_DEFINITION (STRING): Compile definition ("USE_ACCELERATE").
# - ACCELERATE_LINK_LIB (STRING): Link library/framework name ("Accelerate" or path).
# - ACCELERATE_BACKEND_SOURCES (LIST): List of source files for the Accelerate backend.
# - ACCELERATE_BACKEND_INCLUDE_DIR (STRING): Path to the Accelerate backend's private include directory.
# ==============================

message(STATUS "Checking for Apple Accelerate backend...")

# Default to not found
set(OMNIDSP_HAS_ACCELERATE FALSE)

# Accelerate is only available on Apple platforms
if(APPLE)
  find_package(Accelerate QUIET) # Find the Accelerate framework using CMake's built-in module

  # Check if found OR if the system is Darwin (macOS/iOS) as a fallback
  # Sometimes find_package might fail but the framework is implicitly available
  if(Accelerate_FOUND OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "  Apple Accelerate backend available.")
    set(OMNIDSP_HAS_ACCELERATE TRUE)

    # Define the compile definition to enable Accelerate code paths
    set(ACCELERATE_COMPILE_DEFINITION "USE_ACCELERATE")
    message(STATUS "    Compile definition: ${ACCELERATE_COMPILE_DEFINITION}")

    # Define the link library (framework name)
    # find_library is good practice but often linking "-framework Accelerate" works directly
    find_library(ACCELERATE_FRAMEWORK_PATH Accelerate)
    if(ACCELERATE_FRAMEWORK_PATH)
        set(ACCELERATE_LINK_LIB ${ACCELERATE_FRAMEWORK_PATH})
        message(STATUS "    Found Accelerate framework at: ${ACCELERATE_FRAMEWORK_PATH}")
    else()
        # Fallback to using the framework name directly, common practice
        set(ACCELERATE_LINK_LIB Accelerate)
        message(WARNING "    Could not find Accelerate framework path via find_library, will link by name 'Accelerate'.")
    endif()
    message(STATUS "    Link library: ${ACCELERATE_LINK_LIB}")


    # Define source files for the Accelerate backend
    set(ACCELERATE_BACKEND_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/fft.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/window.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/convolution.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/resample.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate/backend.cpp"
    )
    # Print sources list using the helper function
    print_prefixed_list(STATUS "    " "Accelerate backend sources:" ${ACCELERATE_BACKEND_SOURCES})

    # Define include directory for the Accelerate backend
    set(ACCELERATE_BACKEND_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/accelerate"
    )
     message(STATUS "    Accelerate backend include dir: ${ACCELERATE_BACKEND_INCLUDE_DIR}")

  else()
    message(STATUS "  Apple Accelerate framework not found.")
  endif()
else()
    message(STATUS "  Skipping Accelerate check (not on Apple platform).")
endif()
