# cmake/backend/default.cmake
# ===========================
# Configures the default backend sources and include directory. This backend
# is always included.
#
# Variables Set (Current Scope):
# - DEFAULT_BACKEND_SOURCES (LIST): List of source files for the default backend.
# - DEFAULT_BACKEND_INCLUDE_DIR (STRING): Path to the default backend's private include directory.
# ===========================

message(STATUS "Configuring Default backend...")

# Define source files for the default backend
set(DEFAULT_BACKEND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/window.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/resample.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/cqt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/backend.cpp"
)
# Print sources list using the helper function
print_prefixed_list(STATUS "  " "Default backend sources:" ${DEFAULT_BACKEND_SOURCES})

# Define include directory for the default backend (used for private headers)
set(DEFAULT_BACKEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default"
)
message(STATUS "  Default backend include dir: ${DEFAULT_BACKEND_INCLUDE_DIR}")
