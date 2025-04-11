# src/omnidsp_py/CMakeLists.txt
# Updated for Conda-managed dependencies

# Ensure pybind11 is available (found via find_package in parent scope)
if(NOT TARGET pybind11::module)
    message(FATAL_ERROR "pybind11 target not found. Ensure it's configured in the parent CMakeLists.txt before adding this subdirectory.")
endif()

# Define the Python module target using pybind11's helper function
# This creates the C++ extension module (e.g., omnidsp_py.cp313-win_amd64.pyd)
# The first argument ('omnidsp_py') must match the PYBIND11_MODULE name in bindings.cpp
pybind11_add_module(omnidsp_py # Module target name
    SHARED # Build as shared library (.so/.pyd)
    bindings.cpp # Source file relative to this directory
)

# Link the Python module against the core C++ omnidsp library target
# (which should be defined in the parent CMakeLists.txt)
target_link_libraries(omnidsp_py PRIVATE omnidsp)

# Add include directories needed for compiling bindings.cpp
# - pybind11 headers are handled automatically by pybind11_add_module when using find_package
# - Need the project's public include directory for OmniDSP headers
target_include_directories(omnidsp_py PRIVATE
    ${PROJECT_SOURCE_DIR}/include # Path to the top-level 'include' directory
    # pybind11::headers # Usually not needed explicitly here with find_package
)

# Define the installation destination directory relative to site-packages
# This name should match the Python package name
set(PYTHON_PACKAGE_DESTINATION omnidsp_py)

# --- MKL DLL Handling Removed ---
# When using Conda to manage dependencies, Conda environment activation handles
# making the necessary runtime libraries (like MKL DLLs) available.
# Bundling them into the wheel via CMake/delvewheel is not required
# for running within the Conda environment.

# Install __init__.py (Optional, uncomment if you have package logic there)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
#     DESTINATION ${PYTHON_PACKAGE_DESTINATION}
# )

# Install the Python stub file for type hinting (PEP 561)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/omnidsp_py.pyi
    DESTINATION ${PYTHON_PACKAGE_DESTINATION}
)

# Install Python Module Target artifacts (.pyd, .so)
# ARCHIVE (.lib) is usually not needed for Python module installation
install(TARGETS omnidsp_py
    RUNTIME DESTINATION ${PYTHON_PACKAGE_DESTINATION} # Installs .dll/.so
    LIBRARY DESTINATION ${PYTHON_PACKAGE_DESTINATION} # Installs .pyd
    # ARCHIVE DESTINATION ${PYTHON_PACKAGE_DESTINATION} # Typically not needed
)

message(VERBOSE "Configured Python bindings target 'omnidsp_py' with installation into '${PYTHON_PACKAGE_DESTINATION}/' relative destination.")