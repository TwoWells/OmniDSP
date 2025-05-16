# OmniDSP/src/omnidsp/backends/accelerate/CMakeLists.txt
# Configures the Accelerate backend for the 'OmniDSP' library.

# This script assumes OPTION_OMNIDSP_ENABLE_ACCELERATE is ON if it's processed.
# The parent backends/CMakeLists.txt handles the conditional inclusion.

message(STATUS "    Configuring Accelerate backend sources and dependencies...")

# Default to backend not functionally enabled. This will be updated if successful.
# The PARENT_SCOPE ensures this variable is visible to src/omnidsp/CMakeLists.txt
set(OMNIDSP_BACKEND_ACCELERATE_ENABLED 0 PARENT_SCOPE)

# Flags to control stub setup and source addition for IntelliSense.
# These are determined based on whether the real backend can be built or if stubs are needed.
set(SETUP_INTELLISENSE_STUBS FALSE)
set(ADD_SOURCES_FOR_INTELLISENSE FALSE)

# Define the source files that constitute the Accelerate backend.
set(ACCELERATE_BACKEND_SOURCES_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp"
    # Add any other .cpp files specific to the Accelerate backend
)

if(APPLE)
    # On Apple platforms, try to find and use the real Accelerate framework.
    message(STATUS "      Platform is Apple. Looking for system Accelerate framework.")
    find_framework(Accelerate REQUIRED) # Make it required here; if not found, it's an error for this backend configuration.

    if(NOT Accelerate_FOUND) # Should not happen if REQUIRED is used and find_framework fails.
        message(WARNING "      System Accelerate framework not found on Apple platform. Real Accelerate backend will be disabled.")
        # OMNIDSP_BACKEND_ACCELERATE_ENABLED remains 0.
        # Stubs are NOT set up in this specific case (OPTION ON, APPLE, Framework Missing)
        # as the primary intent was to use the real framework.
    else()
        message(STATUS "      Found system Accelerate framework.")

        # --- Add Sources to the 'OmniDSP' library target for actual compilation ---
        target_sources(OmniDSP PRIVATE ${ACCELERATE_BACKEND_SOURCES_LIST})
        message(STATUS "        Added Accelerate backend sources for real compilation on Apple.")

        # --- Include Directories for Accelerate Backend (for its own headers) ---
        target_include_directories(OmniDSP PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" # For headers local to the backend.cpp/fft.cpp
        )

        # --- Create or use existing interface target for Accelerate Framework ---
        # This provides a clean way to link the framework.
        if(NOT TARGET OmniDSP::Accelerate)
            add_library(OmniDSP::Accelerate INTERFACE IMPORTED GLOBAL)
            target_link_libraries(OmniDSP::Accelerate INTERFACE "-framework Accelerate")
            message(STATUS "        Created OmniDSP::Accelerate interface target.")
        endif()
        # Link the main OmniDSP library with the Accelerate framework via the interface target
        target_link_libraries(OmniDSP PRIVATE OmniDSP::Accelerate)
        message(STATUS "        Linked OmniDSP target with OmniDSP::Accelerate.")

        # --- Compile Definition for Accelerate Backend ---
        # This definition (OMNIDSP_INTERNAL_USE_ACCELERATE) is used in C++
        # to conditionally compile Accelerate-specific code paths within the backend implementation.
        target_compile_definitions(OmniDSP PRIVATE OMNIDSP_INTERNAL_USE_ACCELERATE)

        # --- Signal Success and Set Configuration Variable ---
        set(OMNIDSP_BACKEND_ACCELERATE_ENABLED 1 PARENT_SCOPE) # Real backend is functionally enabled.
        message(STATUS "      Real Accelerate backend configured and enabled successfully.")
        # If real backend is built, we don't need the fake SDK stubs or to add sources just for IntelliSense.
        set(SETUP_INTELLISENSE_STUBS FALSE)
        set(ADD_SOURCES_FOR_INTELLISENSE FALSE) # Sources are already added for real compilation.
    endif()
else() # Not Apple (and OPTION_OMNIDSP_ENABLE_ACCELERATE is ON, because this file is processed)
    message(STATUS "      Platform is not Apple. Real Accelerate backend cannot be built.")
    message(STATUS "      Will set up stubs and add sources for IntelliSense as OPTION_OMNIDSP_ENABLE_ACCELERATE is ON.")
    # Since OPTION_OMNIDSP_ENABLE_ACCELERATE is ON but it's not Apple,
    # the intention is to provide IntelliSense support using stubs.
    set(SETUP_INTELLISENSE_STUBS TRUE)
    set(ADD_SOURCES_FOR_INTELLISENSE TRUE)
    # OMNIDSP_BACKEND_ACCELERATE_ENABLED remains 0 (not functionally enabled).
endif()

# --- Add Accelerate backend sources if flagged for IntelliSense ---
# This block adds the sources to the OmniDSP target if ADD_SOURCES_FOR_INTELLISENSE is true.
# This happens if the real backend isn't compiled (e.g., on non-Apple when option is ON,
# or if the option is OFF but we still want IntelliSense).
if(ADD_SOURCES_FOR_INTELLISENSE)
    target_sources(OmniDSP PRIVATE ${ACCELERATE_BACKEND_SOURCES_LIST})
    message(STATUS "      Added Accelerate backend sources to target for IntelliSense purposes.")
    # Also add the local include directory for these files, so backend.cpp can find backend.h etc.
    target_include_directories(OmniDSP PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# --- Setup IntelliSense Stubs (fake SDK include path) if flagged ---
# This block adds the include path to the fake SDK if SETUP_INTELLISENSE_STUBS is true.
if(SETUP_INTELLISENSE_STUBS)
    message(STATUS "      Setting up Accelerate stubs (fake SDK includes) for IntelliSense.")
    # Path to the fake Accelerate SDK.
    # Assumes 'fake_sdk' directory is at the root of the OmniDSP project (where PROJECT_SOURCE_DIR points).
    set(FAKE_ACCELERATE_SDK_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/fake_sdk/macOS")

    if(EXISTS "${FAKE_ACCELERATE_SDK_INCLUDE_DIR}/Accelerate/Accelerate.h")
        message(STATUS "        Adding fake Accelerate headers include path: ${FAKE_ACCELERATE_SDK_INCLUDE_DIR}")
        target_include_directories(OmniDSP PRIVATE ${FAKE_ACCELERATE_SDK_INCLUDE_DIR})
    else()
        message(WARNING "        Fake Accelerate SDK directory not found at '${FAKE_ACCELERATE_SDK_INCLUDE_DIR}'. IntelliSense for Accelerate may not work correctly.")
    endif()
    # OMNIDSP_BACKEND_ACCELERATE_ENABLED would be 0 in all cases where stubs are set up.
    message(STATUS "      Accelerate stubs configured. Backend is NOT functionally enabled.")
endif()

message(STATUS "    Finished configuring Accelerate backend.")
