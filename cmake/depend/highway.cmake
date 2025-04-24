# cmake/depend/highway.cmake
# ===========================
# Fetches and configures Google Highway using FetchContent.
# ===========================

message(STATUS "  Configuring Google Highway dependency...")

# Adjust indentation for fetching message
message(STATUS "    Fetching Google Highway library using FetchContent...")
FetchContent_Declare(
  highway # Name for FetchContent internal tracking
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.2.0 # Use latest known tag 1.2.0
  GIT_SHALLOW    TRUE  # Only download necessary history
)
# Set Highway build options *before* FetchContent_MakeAvailable
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE) # Don't build Highway's own tests
# Fetch, configure, build (if necessary), and make Highway targets available
FetchContent_MakeAvailable(highway)
# Mark Highway includes as SYSTEM to suppress warnings from its headers
# FetchContent_MakeAvailable doesn't directly support the SYSTEM keyword like add_subdirectory,
# so we need to set the property on the directory afterwards if desired.
# Check if the source directory variable is defined before setting the property
if(DEFINED highway_SOURCE_DIR AND EXISTS ${highway_SOURCE_DIR})
    set_property(DIRECTORY ${highway_SOURCE_DIR} PROPERTY SYSTEM TRUE)
endif()
# Adjust indentation for made available message
message(STATUS "      Made highway targets available (e.g., hwy::hwy).")
# Highway should now provide the hwy::hwy target for linking.
