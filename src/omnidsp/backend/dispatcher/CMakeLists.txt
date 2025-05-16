# OmniDSP/src/omnidsp/backends/dispatcher/CMakeLists.txt
# Configures the Dispatcher backend for the 'OmniDSP' library.

message(STATUS "    Configuring Dispatcher backend sources...")

# --- Define Dispatcher Backend Source Files ---
# The Dispatcher::Backend is a core component for routing to other backends.
set(DISPATCHER_BACKEND_SOURCES_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
    # Add any other .cpp files specific to the Dispatcher backend if they arise
)

# --- Add Sources to the 'OmniDSP' library target ---
# The 'OmniDSP' target is defined in the parent CMakeLists.txt (src/omnidsp/CMakeLists.txt).
target_sources(OmniDSP PRIVATE ${DISPATCHER_BACKEND_SOURCES_LIST})
message(STATUS "      Added Dispatcher backend sources to OmniDSP target.")

# --- Include Directories for Dispatcher Backend ---
# Add current directory for backend-specific headers like "backend.hpp".
# This allows dispatcher/backend.cpp to #include "backend.hpp"
target_include_directories(OmniDSP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# The Dispatcher backend itself doesn't have an OMNIDSP_BACKEND_DISPATCHER_ENABLED
# flag that's set to PARENT_SCOPE for omnidsp_config.h in the same way
# optional optimized backends do. It's considered a core part of the backend
# management system if multiple backends are in play.
# It also doesn't typically need an OMNIDSP_INTERNAL_USE_DISPATCHER definition,
# as its code is compiled directly into the OmniDSP library.

message(STATUS "    Finished configuring Dispatcher backend.")
