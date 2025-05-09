# OmniDSP/src/omnidsp/accelerate/CMakeLists.txt
# Configures the Accelerate backend for the 'OmniDSP' library.

# This script assumes it is always processed by the parent CMakeLists.txt.
# It uses OPTION_OMNIDSP_ENABLE_ACCELERATE to determine its behavior.

# Default to backend not functionally enabled. This will be updated if the real backend is successfully configured.
set(OMNIDSP_BACKEND_ACCELERATE_ENABLED 0 PARENT_SCOPE)
set(SETUP_INTELLISENSE_STUBS FALSE) # Flag to control stub setup (for fake SDK include path)
set(ADD_SOURCES_FOR_INTELLISENSE FALSE) # Flag to control adding backend sources for IntelliSense

# Define the source files that constitute the Accelerate backend.
# These are only *compiled* if the real Accelerate framework is used on Apple,
# but we might add them to the target for IntelliSense purposes on other platforms.
set(ACCELERATE_BACKEND_SOURCES_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp"
    # Add any other .cpp files specific to the Accelerate backend
)

if(OPTION_OMNIDSP_ENABLE_ACCELERATE)
    # The Accelerate feature is requested to be ON.
    message(STATUS "  OPTION_OMNIDSP_ENABLE_ACCELERATE is ON. Attempting to configure Accelerate backend...")

    if(APPLE)
        # On Apple platforms, try to find and use the real Accelerate framework.
        message(STATUS "    Platform is Apple. Looking for system Accelerate framework.")
        find_framework(Accelerate) # find_framework sets Accelerate_FOUND

        if(NOT Accelerate_FOUND)
            message(WARNING "    System Accelerate framework not found on Apple platform. Real Accelerate backend will be disabled.")
            # OMNIDSP_BACKEND_ACCELERATE_ENABLED remains 0.
            # Stubs are NOT set up in this specific case (OPTION ON, APPLE, Framework Missing)
            # as the primary intent was to use the real framework.
            # Sources are not added for compilation or IntelliSense if the framework is missing.
        else()
            message(STATUS "    Found system Accelerate framework.")

            # --- Add Sources to the 'OmniDSP' library target for actual compilation ---
            target_sources(OmniDSP PRIVATE ${ACCELERATE_BACKEND_SOURCES_LIST})
            message(STATUS "      Added Accelerate backend sources for real compilation on Apple.")

            # --- Include Directories for Accelerate Backend (for its own headers) ---
            target_include_directories(OmniDSP PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}" # For headers local to the backend.cpp/fft.cpp
            )

            # --- Create Interface Target for Accelerate Framework ---
            if(NOT TARGET OmniDSP::Accelerate)
                add_library(OmniDSP::Accelerate INTERFACE IMPORTED GLOBAL)
                target_link_libraries(OmniDSP::Accelerate INTERFACE "-framework Accelerate")
                message(STATUS "    Created OmniDSP::Accelerate interface target and linked system Accelerate framework.")
            endif()

            # --- Compile Definition for Accelerate Backend ---
            target_compile_definitions(OmniDSP PRIVATE OMNIDSP_INTERNAL_USE_ACCELERATE)

            # --- Signal Success and Set Configuration Variable ---
            set(OMNIDSP_BACKEND_ACCELERATE_ENABLED 1 PARENT_SCOPE) # Real backend is functionally enabled.
            message(STATUS "    Real Accelerate backend configured and enabled successfully.")
            # If real backend is built, we don't need the fake SDK stubs.
            set(SETUP_INTELLISENSE_STUBS FALSE)
            set(ADD_SOURCES_FOR_INTELLISENSE FALSE) # Sources are already added for real compilation.
        endif()
    else() # Not Apple (and OPTION_OMNIDSP_ENABLE_ACCELERATE is ON)
        message(STATUS "    Platform is not Apple. Real Accelerate backend cannot be built.")
        message(STATUS "    Will set up stubs and add sources for IntelliSense as OPTION_OMNIDSP_ENABLE_ACCELERATE is ON.")
        set(SETUP_INTELLISENSE_STUBS TRUE)
        set(ADD_SOURCES_FOR_INTELLISENSE TRUE)
        # OMNIDSP_BACKEND_ACCELERATE_ENABLED remains 0.
    endif()
else() # OPTION_OMNIDSP_ENABLE_ACCELERATE is OFF
    message(STATUS "  OPTION_OMNIDSP_ENABLE_ACCELERATE is OFF.")
    message(STATUS "    Accelerate backend will NOT be functionally enabled.")
    message(STATUS "    Will set up stubs and add sources for IntelliSense for development purposes.")
    set(SETUP_INTELLISENSE_STUBS TRUE)
    set(ADD_SOURCES_FOR_INTELLISENSE TRUE)
    # OMNIDSP_BACKEND_ACCELERATE_ENABLED remains 0.
endif()

# --- Add Accelerate backend sources if flagged for IntelliSense (and not already added for real compilation) ---
if(ADD_SOURCES_FOR_INTELLISENSE)
    target_sources(OmniDSP PRIVATE ${ACCELERATE_BACKEND_SOURCES_LIST})
    message(STATUS "    Added Accelerate backend sources to target for IntelliSense purposes.")
    # Also add the local include directory for these files, so backend.cpp can find backend.h etc.
    target_include_directories(OmniDSP PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# --- Setup IntelliSense Stubs (fake SDK include path) if flagged ---
if(SETUP_INTELLISENSE_STUBS)
    message(STATUS "    Setting up Accelerate stubs (fake SDK includes) for IntelliSense.")
    # Path to the fake Accelerate SDK.
    # Assumes 'fake_sdk' directory is at the root of the OmniDSP project (where PROJECT_SOURCE_DIR points).
    set(FAKE_ACCELERATE_SDK_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/fake_sdk/macOS")

    if(EXISTS "${FAKE_ACCELERATE_SDK_INCLUDE_DIR}/Accelerate/Accelerate.h")
        message(STATUS "      Adding fake Accelerate headers include path: ${FAKE_ACCELERATE_SDK_INCLUDE_DIR}")
        target_include_directories(OmniDSP PRIVATE ${FAKE_ACCELERATE_SDK_INCLUDE_DIR})
    else()
        message(WARNING "      Fake Accelerate SDK directory not found at '${FAKE_ACCELERATE_SDK_INCLUDE_DIR}'. IntelliSense for Accelerate may not work correctly.")
    endif()

    # OMNIDSP_BACKEND_ACCELERATE_ENABLED would be 0 in all cases where stubs are set up.
    message(STATUS "    Accelerate stubs configured. Backend is NOT functionally enabled.")
endif()
