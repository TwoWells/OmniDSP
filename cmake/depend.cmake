# cmake/depend.cmake
# ==================
# Main orchestrator for finding/fetching project dependencies.
# Includes individual dependency modules from the cmake/depend/ directory.
# Also includes and configures FetchContent.
#
# Includes:
# - cmake/depend/python_bindings.cmake
# - cmake/depend/boost.cmake
# - cmake/depend/googletest.cmake
# - cmake/depend/highway.cmake
# ==================

message(STATUS "Processing project dependencies...")

# --- Include FetchContent module ---
# Needs to be included before setting FetchContent variables or calling FetchContent functions.
# include(FetchContent)

# --- Configure FetchContent Cache Location ---
# Set the base directory for FetchContent outside the build folder
# This allows downloaded sources to persist between clean builds.
# Needs to be set *before* the first FetchContent_Declare call.
# Using FORCE ensures that this path overwrites any previously cached default value.
# set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/cmake/.cache CACHE PATH "Root directory for FetchContent downloads" FORCE)
# file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
# message(STATUS "  FetchContent base directory set to: ${FETCHCONTENT_BASE_DIR}")


# Create the depend directory if it doesn't exist (for the modules below)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/depend)

# Find/fetch individual dependencies
include(cmake/depend/python_bindings.cmake)
include(cmake/depend/boost.cmake)
include(cmake/depend/googletest.cmake)
include(cmake/depend/highway.cmake)

message(STATUS "Finished processing project dependencies.")
