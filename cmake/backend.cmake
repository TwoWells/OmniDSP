# cmake/backend.cmake
# ===================
# Orchestrates the detection and configuration of available compute backends
# (Default, Accelerate, oneMKL). It includes specific backend configuration
# files from the cmake/backend/ directory. Each included file is expected
# to set backend-specific variables (sources, includes, libs, etc.) in the
# current scope. This script aggregates these settings into common variables.
#
# Includes:
# - cmake/backend/default.cmake
# - cmake/backend/accelerate.cmake
# - cmake/backend/onemkl.cmake
#
# Variables Set (Aggregated):
# - OMNIDSP_ALL_BACKEND_SOURCES (LIST): All source files for enabled backends.
# - OMNIDSP_ALL_BACKEND_INCLUDE_DIRS (LIST): Private include dirs for enabled backends.
# - OMNIDSP_ALL_BACKEND_COMPILE_DEFS (LIST): Compile definitions for enabled backends.
# - OMNIDSP_ALL_BACKEND_LINK_LIBS (LIST): Libraries to link for enabled backends.
# - OMNIDSP_ALL_BACKEND_LINK_DIRS (LIST): Link directories for enabled backends.
# - OMNIDSP_HAS_ACCELERATE (BOOL): True if Accelerate backend is available and configured.
# - OMNIDSP_HAS_ONEMKL (BOOL): True if oneMKL backend is available and configured.
# ===================

message(STATUS "Configuring OmniDSP Backends...")

# --- Initialize Aggregation Variables ---
# These variables will collect settings from all enabled backends.
set(OMNIDSP_ALL_BACKEND_SOURCES "")       # List of all source files for enabled backends
set(OMNIDSP_ALL_BACKEND_INCLUDE_DIRS "")  # List of private include dirs for enabled backends
set(OMNIDSP_ALL_BACKEND_COMPILE_DEFS "") # List of compile definitions for enabled backends
set(OMNIDSP_ALL_BACKEND_LINK_LIBS "")    # List of libraries to link for enabled backends
set(OMNIDSP_ALL_BACKEND_LINK_DIRS "")    # List of link directories for enabled backends

# --- Include Default Backend (Always Enabled) ---
include(cmake/backend/default.cmake)
# Add default backend settings to the aggregate lists
list(APPEND OMNIDSP_ALL_BACKEND_SOURCES ${DEFAULT_BACKEND_SOURCES})
list(APPEND OMNIDSP_ALL_BACKEND_INCLUDE_DIRS ${DEFAULT_BACKEND_INCLUDE_DIR})

# --- Include Accelerate Backend (Conditional) ---
include(cmake/backend/accelerate.cmake)
if(OMNIDSP_HAS_ACCELERATE)
  message(STATUS "  Enabling Accelerate backend configuration.")
  list(APPEND OMNIDSP_ALL_BACKEND_SOURCES ${ACCELERATE_BACKEND_SOURCES})
  list(APPEND OMNIDSP_ALL_BACKEND_INCLUDE_DIRS ${ACCELERATE_BACKEND_INCLUDE_DIR})
  list(APPEND OMNIDSP_ALL_BACKEND_COMPILE_DEFS ${ACCELERATE_COMPILE_DEFINITION})
  list(APPEND OMNIDSP_ALL_BACKEND_LINK_LIBS ${ACCELERATE_LINK_LIB})
  # Accelerate doesn't typically require special link directories
endif()

# --- Include oneMKL Backend (Conditional) ---
include(cmake/backend/onemkl.cmake)
if(OMNIDSP_HAS_ONEMKL)
   message(STATUS "  Enabling oneMKL backend configuration.")
   list(APPEND OMNIDSP_ALL_BACKEND_SOURCES ${ONEMKL_BACKEND_SOURCES})
   list(APPEND OMNIDSP_ALL_BACKEND_INCLUDE_DIRS ${ONEMKL_BACKEND_INCLUDE_DIR})
   list(APPEND OMNIDSP_ALL_BACKEND_COMPILE_DEFS ${ONEMKL_COMPILE_DEFINITION})
   list(APPEND OMNIDSP_ALL_BACKEND_LINK_LIBS ${ONEMKL_LINK_LIBS})
   list(APPEND OMNIDSP_ALL_BACKEND_INCLUDE_DIRS ${ONEMKL_INCLUDE_DIRS}) # Add MKL/IPP includes found by find_package (if any)
   list(APPEND OMNIDSP_ALL_BACKEND_LINK_DIRS ${ONEMKL_LINK_DIRS})       # Add link dirs (likely empty now)
endif()

# --- Finalize and Report ---
# Remove duplicates just in case paths were added multiple times
list(REMOVE_DUPLICATES OMNIDSP_ALL_BACKEND_INCLUDE_DIRS)
list(REMOVE_DUPLICATES OMNIDSP_ALL_BACKEND_LINK_DIRS)

message(STATUS "Backend configuration complete.")
# Use the helper function for final reporting (handles empty lists internally)
print_prefixed_list(STATUS "  " "Aggregate Backend Sources:" ${OMNIDSP_ALL_BACKEND_SOURCES})
print_prefixed_list(STATUS "  " "Aggregate Backend Include Dirs:" ${OMNIDSP_ALL_BACKEND_INCLUDE_DIRS})
print_prefixed_list(STATUS "  " "Aggregate Backend Compile Defs:" ${OMNIDSP_ALL_BACKEND_COMPILE_DEFS})
print_prefixed_list(STATUS "  " "Aggregate Backend Link Libs:" ${OMNIDSP_ALL_BACKEND_LINK_LIBS})
print_prefixed_list(STATUS "  " "Aggregate Backend Link Dirs:" ${OMNIDSP_ALL_BACKEND_LINK_DIRS})
# Print boolean status
message(STATUS "  Has Accelerate: ${OMNIDSP_HAS_ACCELERATE}")
message(STATUS "  Has oneMKL: ${OMNIDSP_HAS_ONEMKL}")
