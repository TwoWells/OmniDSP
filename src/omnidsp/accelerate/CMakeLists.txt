# OmniDSP/src/omnidsp/accelerate/CMakeLists.txt
# Configures the Accelerate backend for the 'OmniDSP' library.

# This file is included if OPTION_OMNIDSP_ENABLE_ACCELERATE is ON.

message(STATUS "  Configuring Accelerate backend...")

# --- Platform Check ---
if(NOT APPLE)
    message(STATUS "    Accelerate backend is only supported on Apple platforms. Skipping.")
    return() # Exit early if not on Apple
endif()

# --- Find Accelerate Framework ---
find_framework(Accelerate) # Don't make it REQUIRED here, handle Accelerate_FOUND

if(NOT Accelerate_FOUND)
    message(WARNING "    Accelerate framework not found, even though on Apple. Accelerate backend will be disabled.")
    return() # Exit if framework not found
endif()

message(STATUS "    Found Accelerate framework.")

# --- Define Accelerate Backend Source Files ---
set(ACCELERATE_BACKEND_SOURCES_INTERNAL
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp"
    # Add any other .cpp files specific to the Accelerate backend
)

# --- Add Sources to the 'OmniDSP' library target ---
target_sources(OmniDSP PRIVATE ${ACCELERATE_BACKEND_SOURCES_INTERNAL})

# --- Include Directories for Accelerate Backend ---
target_include_directories(OmniDSP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Create Interface Target for Accelerate Framework ---
if(NOT TARGET OmniDSP::Accelerate)
    add_library(OmniDSP::Accelerate INTERFACE IMPORTED GLOBAL)
    # Link the Accelerate framework to this interface target
    target_link_libraries(OmniDSP::Accelerate INTERFACE "-framework Accelerate")
    # Add any necessary include directories or compile definitions for Accelerate usage
    # target_include_directories(OmniDSP::Accelerate INTERFACE ...)
    message(STATUS "    Created OmniDSP::Accelerate interface target.")
endif()

# --- Compile Definition for Accelerate Backend ---
# This definition can be used in C++ code to enable Accelerate-specific paths.
# e.g., #ifdef OMNIDSP_INTERNAL_USE_ACCELERATE
target_compile_definitions(OmniDSP PRIVATE OMNIDSP_INTERNAL_USE_ACCELERATE)

# --- Signal Success and Set Configuration Variable ---
set(OMNIDSP_BACKEND_ACCELERATE_ENABLED 1 PARENT_SCOPE)

message(STATUS "    Accelerate backend configured successfully.")
