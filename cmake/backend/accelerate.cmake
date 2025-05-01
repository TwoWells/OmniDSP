# ==============================================================================
# cmake/backend/accelerate.cmake
# ==============================================================================
# Finds the Accelerate framework and sets variables for the Accelerate backend.

# Initialize output variables in parent scope
set(OMNIDSP_HAS_ACCELERATE FALSE PARENT_SCOPE)
set(ACCELERATE_BACKEND_SOURCES "" PARENT_SCOPE)
set(ACCELERATE_BACKEND_INCLUDE_DIRS "" PARENT_SCOPE)
set(ACCELERATE_BACKEND_LINK_LIBS "" PARENT_SCOPE)

# Check if the backend is enabled via the option set in cmake/backend.cmake
if(NOT OMNIDSP_ENABLE_ACCELERATE)
    message(STATUS "  Accelerate backend disabled (OMNIDSP_ENABLE_ACCELERATE=OFF).")
    return() # Exit early if not enabled
endif()

# Double-check platform (redundant if option dependency is correct, but safe)
if(NOT APPLE)
    message(WARNING "Accelerate backend enabled but platform is not Apple. Disabling.")
    return() # Exit if not on Apple
endif()

message(STATUS "  Attempting to configure Accelerate backend...")

# Find the Accelerate framework
find_framework(Accelerate REQUIRED) # REQUIRED ensures CMake errors if not found

if(Accelerate_FOUND)
    message(STATUS "      Found Accelerate framework. Enabling backend.")

    # --- Set Backend Source Files ---
    set(ACCELERATE_BACKEND_SOURCES
        "${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/backend.cpp"
        "${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/fft.cpp"
        "${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/window.cpp"
        "${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/convolution.cpp"
        "${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/resample.cpp"
        # Add filter.cpp if/when it exists
        # "${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/filter.cpp"
        PARENT_SCOPE
    )

    # --- Set Backend Include Directories ---
    # Accelerate framework headers are usually found automatically via the framework path.
    # No specific include directories typically needed here unless there are other dependencies.
    set(ACCELERATE_BACKEND_INCLUDE_DIRS "" PARENT_SCOPE) # Explicitly empty for clarity

    # --- Set Backend Link Libraries/Targets ---
    # Specify the framework link flag needed by the main target
    set(ACCELERATE_BACKEND_LINK_LIBS "-framework Accelerate" PARENT_SCOPE)

    # --- Signal Success ---
    set(OMNIDSP_HAS_ACCELERATE TRUE PARENT_SCOPE)
    message(STATUS "      Accelerate backend configured.")

else()
    # This part should ideally not be reached if find_framework has REQUIRED
    message(WARNING "Accelerate backend enabled, but Accelerate framework was not found. Disabling this backend.")
    # OMNIDSP_HAS_ACCELERATE remains FALSE
endif()
