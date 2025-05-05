# ==============================================================================
# cmake/backend/accelerate.cmake
# ==============================================================================
# Finds the Accelerate framework and sets variables for the Accelerate backend.

# Initialize output variables
set(OMNIDSP_HAS_ACCELERATE FALSE)
set(OMNIDSP_HAS_BACKEND_ACCELERATE_VALUE 0) # Default to 0
set(ACCELERATE_BACKEND_SOURCES "")
set(ACCELERATE_BACKEND_INCLUDE_DIRS "")
set(ACCELERATE_BACKEND_LINK_LIBS "")
# Define the variable for omnidsp_config.h
set(OMNIDSP_ENABLED_BACKEND_ACCELERATE 0)

# Check if the backend is enabled via the option set in cmake/backend.cmake
if(NOT OMNIDSP_ENABLE_ACCELERATE)
    message(STATUS "  Accelerate backend disabled (OMNIDSP_ENABLE_ACCELERATE=OFF).")
    return() # Exit early if not enabled
endif()

# Double-check platform
if(NOT APPLE)
    message(WARNING "Accelerate backend enabled but platform is not Apple. Disabling.")
    return() # Exit if not on Apple
endif()

message(STATUS "  Attempting to configure Accelerate backend...")

# Find the Accelerate framework
find_framework(Accelerate REQUIRED)

if(Accelerate_FOUND)
    # Define the variable for omnidsp_config.h
    set(OMNIDSP_ENABLED_BACKEND_ACCELERATE 1)
    message(STATUS "      Found Accelerate framework. Enabling backend.")

    # --- Set Backend Source Files ---
    set(ACCELERATE_BACKEND_SOURCES
        "${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/backend.cpp"
        "${PROJECT_SOURCE_DIR}/src/omnidsp/accelerate/fft.cpp"
    )

    # --- Set Backend Include Directories ---
    set(ACCELERATE_BACKEND_INCLUDE_DIRS "") # Typically empty

    # --- Set Backend Link Libraries/Targets ---
    set(ACCELERATE_BACKEND_LINK_LIBS "-framework Accelerate")

    # --- Signal Success and Set Config Variables ---
    set(OMNIDSP_HAS_ACCELERATE TRUE)
    set(OMNIDSP_HAS_BACKEND_ACCELERATE_VALUE 1) # Set value to 1 on success
    message(STATUS "      Accelerate backend configured.")

else()
    message(WARNING "Accelerate backend enabled, but Accelerate framework was not found. Disabling this backend.")
    # OMNIDSP_HAS_ACCELERATE remains FALSE
    # OMNIDSP_HAS_BACKEND_ACCELERATE_VALUE remains 0
endif()
