# cmake/depend.cmake
# ==================
# Finds project dependencies: Boost, Google Highway, and spdlog.
# Dependencies are found quietly, with custom messages upon success.
#
# Note: Python bindings and GoogleTest dependencies are handled
#       in their respective consumer CMakeLists.txt files.
# ==================

message(STATUS "Processing project dependencies...")

# --- Define Required Dependency Versions ---
set(OMNIDSP_REQUIRED_BOOST_VERSION "1.85.0")
set(OMNIDSP_REQUIRED_HIGHWAY_VERSION "1.2.0")
set(OMNIDSP_REQUIRED_SPDLOG_VERSION "1.15.2")

# ==============================================================================
# Boost Dependency
# ==============================================================================
# Find Boost using find_package. CONFIG mode is preferred for modern Boost.
# REQUIRED will cause CMake to error if the specified version is not found.
# QUIET suppresses default find_package messages.
find_package(Boost ${OMNIDSP_REQUIRED_BOOST_VERSION} CONFIG REQUIRED QUIET)

if(Boost_FOUND)
    # Print custom found message with version.
    message(STATUS "Found dependency: Boost ${Boost_VERSION}")
    # Modern Boost (with CONFIG mode) provides imported targets (e.g., Boost::boost).
else()
    # This block is unlikely to be reached due to 'REQUIRED' in find_package.
    message(FATAL_ERROR "Could not find required Boost version ${OMNIDSP_REQUIRED_BOOST_VERSION} (or compatible).")
endif()

# ==============================================================================
# Google Highway Dependency
# ==============================================================================
# Find the pre-installed Highway package.
# The package name is 'hwy' (derived from hwy-config.cmake).
# REQUIRED will cause CMake to error if not found.
# CONFIG mode tells CMake to look for hwy-config.cmake or hwyConfig.cmake.
# QUIET suppresses default find_package messages.
find_package(hwy ${OMNIDSP_REQUIRED_HIGHWAY_VERSION} REQUIRED CONFIG QUIET)

# Check if the package was found (somewhat redundant due to REQUIRED).
if(hwy_FOUND)
    # Print custom found message with version.
    message(STATUS "Found dependency: Google Highway ${hwy_VERSION}")
    # Imported targets like 'hwy::hwy' or 'hwy::headers' should be available.
else()
    # This block is unlikely to be reached due to 'REQUIRED' in find_package.
    message(FATAL_ERROR "Could not find required Google Highway package version ${OMNIDSP_REQUIRED_HIGHWAY_VERSION} (or compatible). Ensure it is installed and CMake can find it (e.g., via CMAKE_PREFIX_PATH from an active Conda environment).")
endif()

# ==============================================================================
# spdlog Dependency
# ==============================================================================
# Find spdlog using find_package. It provides CMake config files.
# REQUIRED will cause CMake to error if not found.
# QUIET suppresses the standard "Found spdlog..." message from find_package itself.
find_package(spdlog ${OMNIDSP_REQUIRED_SPDLOG_VERSION} REQUIRED CONFIG QUIET)

if(spdlog_FOUND)
    # spdlog's config file sets spdlog_VERSION.
    # Print custom found message with version.
    message(STATUS "Found dependency: spdlog ${spdlog_VERSION}")
    # Provides imported target spdlog::spdlog
else()
    # This block is unlikely to be reached due to 'REQUIRED' in find_package.
    message(FATAL_ERROR "Could not find required spdlog package version ${OMNIDSP_REQUIRED_SPDLOG_VERSION} (or compatible). Ensure it is installed (e.g., via Conda) and CMake can find it.")
endif()

message(STATUS "Finished processing project dependencies.")
