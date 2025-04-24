# cmake/omnidsp_py.cmake
# ======================
# Includes the Python bindings source subdirectory (src/omnidsp_py) if the
# OMNIDSP_BUILD_PYTHON_BINDINGS option is enabled.
#
# Variables Read: - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake) -
# CMAKE_CURRENT_SOURCE_DIR, CMAKE_CURRENT_BINARY_DIR (CMake Built-in)
# ======================

message(STATUS "Configuring Python bindings source directory...")

# Conditionally add the Python bindings source subdirectory Assumes
# OMNIDSP_BUILD_PYTHON_BINDINGS option is already defined.

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  message(
    STATUS
      "  Checking for Python bindings source directory (OMNIDSP_BUILD_PYTHON_BINDINGS=ON)."
  )
  # Define the path to the subdirectory's CMakeLists.txt
  set(OMNIDSP_PY_CMAKE_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp_py/CMakeLists.txt")

  if(EXISTS "${OMNIDSP_PY_CMAKE_PATH}")
    message(STATUS "    Found ${OMNIDSP_PY_CMAKE_PATH}, adding subdirectory.")
    # Add the subdirectory, specifying source and binary directories
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp_py
                     ${CMAKE_CURRENT_BINARY_DIR}/omnidsp_py)
    # Note: The src/omnidsp_py/CMakeLists.txt is responsible for: - Finding
    # Python (if not already found - though dependencies.cmake should handle it)
    # - Finding pybind11 (if not already found - though dependencies.cmake
    # should handle it) - Defining the pybind11 module target (e.g., using
    # pybind11_add_module) - Linking the module target against the main omnidsp
    # library (OmniDSP::omnidsp) - Setting installation rules for the Python
    # module (.pyd/.so)
  else()
    message(
      WARNING
        # Keep as warning, as building bindings was requested but source is
        # missing
        "    Python bindings subdirectory 'src/omnidsp_py' or its CMakeLists.txt not found at '${OMNIDSP_PY_CMAKE_PATH}'. Skipping Python module configuration despite OMNIDSP_BUILD_PYTHON_BINDINGS=ON."
    )
  endif()
else()
  message(
    STATUS
      "  Skipping Python bindings source directory (OMNIDSP_BUILD_PYTHON_BINDINGS=OFF)."
  )
endif()

message(STATUS "Finished configuring Python bindings source directory.")
