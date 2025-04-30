# cmake/depend/highway.cmake
# ===========================
# Finds and configures the pre-installed Google Highway library using find_package.
# Assumes Highway (e.g., libhwy 1.2.0) is installed via Conda or system package manager.
# ===========================

message(STATUS "   Configuring Google Highway dependency...")

# --- Debugging ---
# Print the CMAKE_PREFIX_PATH to verify Conda environment path is included
# This should ideally contain C:/Users/mwell/anaconda3/envs/omnidsp-dev or similar
# Ensure the Conda environment is activated *before* running CMake.
message(STATUS "     CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
# --- End Debugging ---

# Find the pre-installed Highway package provided by Conda
# REQUIRED will cause CMake to error out if the package is not found.
# Trying both 'HWY' and 'hwy' as package names, as case/naming can vary.

# Try uppercase first quietly to avoid error messages if lowercase is the correct one.
find_package(HWY 1.2.0 QUIET)

# If uppercase wasn't found, try lowercase. Use REQUIRED here to fail if neither works.
if(NOT HWY_FOUND)
    message(STATUS "     Package 'HWY' not found. Trying lowercase 'hwy'...")
    find_package(hwy 1.2.0 REQUIRED) # Try lowercase, REQUIRED will error if not found

    # If lowercase is found, copy its variables to the uppercase equivalents for consistency
    # in the rest of the script (e.g., checking HWY_FOUND).
    if(hwy_FOUND)
      set(HWY_FOUND TRUE)
      set(HWY_VERSION ${hwy_VERSION}) # Copy version info
      # Add any other variables needed from hwy_* to HWY_* if necessary
    endif()
endif()


# Check if the package was found (either as HWY or hwy)
if(HWY_FOUND)
    message(STATUS "     Found pre-installed Google Highway version ${HWY_VERSION}")
    # The necessary targets (like 'hwy') should now be available.
else()
    # This message is now reachable only if REQUIRED failed on the second try (hwy).
    message(FATAL_ERROR "   Could not find required Google Highway package (libhwy >= 1.2.0) using name 'HWY' or 'hwy'. Please ensure it is installed in your active Conda environment ('${CMAKE_PREFIX_PATH}') and that the environment is activated before running CMake.")
endif()

# --- Workarounds removed ---
# Assuming Conda package handles DLL exports correctly.
