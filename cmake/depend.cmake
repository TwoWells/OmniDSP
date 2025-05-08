# cmake/depend.cmake
# ==================
# Finds project dependencies: Boost and Google Highway.
#
# Note: Python bindings and GoogleTest dependencies are handled
#       in their respective consumer CMakeLists.txt files.
# ==================

message(STATUS "Processing project dependencies...")

# --- Define Required Dependency Versions ---
set(OMNIDSP_REQUIRED_BOOST_VERSION "1.85.0")
set(OMNIDSP_REQUIRED_HIGHWAY_VERSION "1.2.0")

# ==============================================================================
# Boost Dependency
# ==============================================================================
message(STATUS "  Finding Boost dependency (Required version: ${OMNIDSP_REQUIRED_BOOST_VERSION})...")

# Find Boost using find_package. CONFIG mode is preferred for modern Boost.
# REQUIRED will cause CMake to error if the specified version is not found.
find_package(Boost ${OMNIDSP_REQUIRED_BOOST_VERSION} CONFIG REQUIRED)

if(Boost_FOUND)
    message(STATUS "    Found Boost version ${Boost_VERSION}")
    # Modern Boost (with CONFIG mode) provides imported targets (e.g., Boost::boost).
else()
    # This block is unlikely to be reached due to 'REQUIRED' in find_package.
    message(FATAL_ERROR "Boost version ${OMNIDSP_REQUIRED_BOOST_VERSION} or newer is required but was not found.")
endif()

# ==============================================================================
# Google Highway Dependency
# ==============================================================================
message(STATUS "  Configuring Google Highway dependency (Required version: ${OMNIDSP_REQUIRED_HIGHWAY_VERSION})...")

# --- Optional Debugging ---
# Uncomment to print CMAKE_PREFIX_PATH to verify Conda environment paths.
# message(STATUS "    CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
# --- End Debugging ---

# Find the pre-installed Highway package.
# The package name is 'hwy' (derived from hwy-config.cmake).
# REQUIRED will cause CMake to error if not found.
# CONFIG mode tells CMake to look for hwy-config.cmake or hwyConfig.cmake.
find_package(hwy ${OMNIDSP_REQUIRED_HIGHWAY_VERSION} REQUIRED CONFIG)

# Check if the package was found (somewhat redundant due to REQUIRED).
if(hwy_FOUND)
    message(STATUS "    Found pre-installed Google Highway version ${hwy_VERSION}")
    # Imported targets like 'hwy::hwy' or 'hwy::headers' should be available.
else()
    # This block is unlikely to be reached due to 'REQUIRED' in find_package.
    message(FATAL_ERROR "  Could not find required Google Highway package (hwy >= ${OMNIDSP_REQUIRED_HIGHWAY_VERSION}). Ensure it is installed and CMake can find it (e.g., via CMAKE_PREFIX_PATH from an active Conda environment).")
endif()

message(STATUS "Finished processing project dependencies.")
