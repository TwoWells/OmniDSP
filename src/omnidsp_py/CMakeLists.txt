# src/omnidsp_py/CMakeLists.txt
# Configures and builds the omnidsp_py Python bindings.
# This script is now responsible for finding Python and pybind11.

# This CMakeLists.txt is included via add_subdirectory if OMNIDSP_BUILD_PYTHON_BINDINGS is ON.
# A redundant check here is fine for self-containment.
if(NOT OMNIDSP_BUILD_PYTHON_BINDINGS)
    message(STATUS "Sanity check: OMNIDSP_BUILD_PYTHON_BINDINGS is OFF. src/omnidsp_py should not have been included.")
    return()
endif()

message(STATUS "Configuring omnidsp_py (Python Bindings)...")

# --- Find Dependencies: Python and pybind11 ---
# This section incorporates the logic previously in cmake/depend/python_bindings.cmake
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED CONFIG) # Use CONFIG mode for modern pybind11

message(STATUS "  Found Python: Version ${Python_VERSION_STRING}, Interpreter: ${Python_EXECUTABLE}")
message(STATUS "  Found pybind11: Version ${pybind11_VERSION}")


# --- Define Python Module ---
# pybind11_add_module is the modern way to create pybind11 modules.
# It handles linking against pybind11 and Python libraries.
pybind11_add_module(omnidsp_py SHARED
    bindings.cpp
    # Add other .cpp files for bindings if any
)

# --- Target Include Directories ---
target_include_directories(omnidsp_py PRIVATE
    # For #include <OmniDSP/omnidsp.hpp> etc. in bindings.cpp
    # The OmniDSP target (defined in src/omnidsp/CMakeLists.txt) should make its
    # public headers available via its INTERFACE_INCLUDE_DIRECTORIES.
    # Linking against OmniDSP should propagate these.
    # If direct access is needed: "${PROJECT_SOURCE_DIR}/include"
    # pybind11 and Python includes are handled by pybind11_add_module and linking their targets.
)

# --- Target Link Libraries ---
target_link_libraries(omnidsp_py PRIVATE
    OmniDSP         # Link against the main C++ OmniDSP library
    pybind11::module # Main pybind11 CMake target for creating modules
    # pybind11::embed # Only if embedding Python
    # Python::Module  # Often handled by pybind11::module, but can be explicit if needed
)

# --- Target Compile Definitions (if any specific to bindings) ---
# target_compile_definitions(omnidsp_py PRIVATE BINDING_SPECIFIC_DEF)

# --- Installation of the Python Module ---
# This installs the compiled .pyd/.so file and any Python files like __init__.py

# Set the correct extension suffix for the Python module
# pybind11_add_module usually sets this, but being explicit can be good.
if(Python_EXTENSION_SUFFIX)
    set_target_properties(omnidsp_py PROPERTIES SUFFIX ${Python_EXTENSION_SUFFIX})
    message(STATUS "  Python module suffix set to: ${Python_EXTENSION_SUFFIX}")
else()
    message(WARNING "  Python_EXTENSION_SUFFIX not found. Module suffix might be incorrect.")
endif()

# Installation destination:
# This typically installs into a path relative to the CMAKE_INSTALL_PREFIX,
# structured to be importable by Python, e.g., <prefix>/python/omnidsp_py
# Or for direct site-packages installation (more complex, often handled by setup.py):
# get_python_lib(PYTHON_SITE_PACKAGES)
# install(TARGETS omnidsp_py LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/omnidsp_py")

# A common approach for CMake-centric builds is to install into a subdirectory
# that mirrors the package structure.
set(OMNIDSP_PY_INSTALL_DESTINATION "python/omnidsp_py") # e.g. <prefix>/python/omnidsp_py
message(STATUS "  Python module will be installed to: ${CMAKE_INSTALL_PREFIX}/${OMNIDSP_PY_INSTALL_DESTINATION}")

install(TARGETS omnidsp_py
    LIBRARY DESTINATION ${OMNIDSP_PY_INSTALL_DESTINATION}
    RUNTIME DESTINATION ${OMNIDSP_PY_INSTALL_DESTINATION} # For any DLLs bundled by the module on Windows
)

# Install Python source files like __init__.py, api.py
install(FILES
    __init__.py
    api.py
    DESTINATION ${OMNIDSP_PY_INSTALL_DESTINATION}
)

message(STATUS "omnidsp_py (Python Bindings) configured successfully.")
