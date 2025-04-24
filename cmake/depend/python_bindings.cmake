# cmake/depend/python_bindings.cmake
# ==================================
# Finds Python and pybind11 dependencies if Python bindings are enabled.
#
# Variables Read:
# - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake)
# ==================================

message(STATUS "  Configuring Python binding dependencies...")

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  message(STATUS "    Python bindings enabled, finding Python and pybind11...")

  # --- Find Python ---
  find_package(
    Python
    COMPONENTS Interpreter Development # Find both interpreter and headers/libs
    REQUIRED)
  message(STATUS "      Found Python Interpreter: ${Python_EXECUTABLE}")
  message(STATUS "      Found Python Development Libraries: ${Python_LIBRARIES}")
  message(STATUS "      Found Python Include Directories: ${Python_INCLUDE_DIRS}")

  # --- Find pybind11 ---
  find_package(pybind11 CONFIG REQUIRED)
  message(
    STATUS
      "      Found pybind11 version ${pybind11_VERSION} via find_package at ${pybind11_DIR}"
  )

else()
  message(STATUS "    Skipping Python and pybind11 find_package (Python bindings disabled).")
endif()
