# cmake/backend/default.cmake
# ===========================
# Configures the default backend sources and include directory. This backend is
# always included.
#
# Variables Set (PARENT_SCOPE): - DEFAULT_BACKEND_SOURCES (LIST): List of source
# files for the default backend. - DEFAULT_BACKEND_INCLUDE_DIR (STRING): Path to
# the default backend's private include directory.
# ===========================

message(STATUS "Configuring Default backend...")

# Define source files for the default backend
set(DEFAULT_BACKEND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/fft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/window.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/resample.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/cqt.cpp"
    # Note: filter.cpp seems specific to default, include if needed by the
    # backend interface
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/filter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default/backend.cpp"
    PARENT_SCOPE # Make this variable available in the including scope
                 # (cmake/backend.cmake)
)
message(STATUS "  Default backend sources: ${DEFAULT_BACKEND_SOURCES}")

# Define include directory for the default backend (used for private headers)
set(DEFAULT_BACKEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/default"
    PARENT_SCOPE)
message(STATUS "  Default backend include dir: ${DEFAULT_BACKEND_INCLUDE_DIR}")
