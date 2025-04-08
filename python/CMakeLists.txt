# CMakeLists.txt for OmniDSP Python Bindings

# Ensure pybind11 is available (it should be from the parent scope)
if(NOT TARGET pybind11::module)
    message(FATAL_ERROR "pybind11 target not found. Ensure it's configured in the parent CMakeLists.txt before adding this subdirectory.")
endif()

# Define the Python module target using pybind11's helper function
# pybind11_add_module(<target_name> [SHARED|MODULE] [NO_EXTRAS] source1 [source2 ...])
# <target_name> is the name Python will import (e.g., omnidsp)
pybind11_add_module(omnidsp_py # Module target name
    SHARED # Build as shared library (.so/.pyd)
    bindings.cpp # Source file relative to this directory
)

# Link the Python module against the core omnidsp library
# omnidsp target should be available from the parent scope
target_link_libraries(omnidsp_py PRIVATE omnidsp)

# Add include directories needed for bindings.cpp
# pybind11_add_module usually handles pybind11 includes automatically.
# Need the public include dir for OmniDSP headers from the project root.
target_include_directories(omnidsp_py PRIVATE
    ${PROJECT_SOURCE_DIR}/include # Use PROJECT_SOURCE_DIR for top-level include
)

# --- MKL DLL Handling for Windows ---
# If the selected backend is MKL (check MKL_FOUND from parent scope),
# find and install the necessary runtime DLLs alongside the Python module.
if(WIN32 AND MKL_FOUND) # MKL_FOUND is manually set in root CMakeLists now
    message(STATUS "Python Bindings: MKL backend detected, configuring DLL installation for omnidsp.")

    # Construct potential paths based on environment variables
    set(MKL_BIN_HINTS "")
    # Define potential relative paths within MKL/oneAPI root
    set(MKL_RELATIVE_BIN_PATHS "latest/bin/intel64" "latest/bin" "bin/intel64" "bin") # Add common variations

    # Check MKLROOT environment variable
    if(DEFINED ENV{MKLROOT})
        foreach(REL_PATH ${MKL_RELATIVE_BIN_PATHS})
            list(APPEND MKL_BIN_HINTS "$ENV{MKLROOT}/${REL_PATH}")
        endforeach()
        message(STATUS "Python Bindings: Added hints based on MKLROOT: $ENV{MKLROOT}")
    endif()

    # Check ONEAPI_ROOT environment variable (MKL might be under here)
    if(DEFINED ENV{ONEAPI_ROOT})
        foreach(REL_PATH ${MKL_RELATIVE_BIN_PATHS})
            list(APPEND MKL_BIN_HINTS "$ENV{ONEAPI_ROOT}/mkl/${REL_PATH}") # MKL is often in an 'mkl' subdir
        endforeach()
        message(STATUS "Python Bindings: Added hints based on ONEAPI_ROOT: $ENV{ONEAPI_ROOT}")
    endif()

    # Use MKL_RUNTIME_LIBRARY_DIRS if it was somehow set by FindMKL (unlikely based on warnings)
    set(SEARCH_HINTS ${MKL_BIN_HINTS})
    if(MKL_RUNTIME_LIBRARY_DIRS)
        list(APPEND SEARCH_HINTS ${MKL_RUNTIME_LIBRARY_DIRS})
        message(STATUS "Python Bindings: Also using MKL_RUNTIME_LIBRARY_DIRS hint: ${MKL_RUNTIME_LIBRARY_DIRS}")
    endif()
    # Always include the system PATH as a fallback hint
    list(APPEND SEARCH_HINTS "$ENV{PATH}")

    message(STATUS "Python Bindings: Searching for MKL DLLs with hints: ${SEARCH_HINTS}")

    # Find the MKL runtime DLL (mkl_rt.dll or versioned)
    # List common names in likely order
    find_file(MKL_RT_DLL_PATH NAMES mkl_rt.2.dll mkl_rt.1.dll mkl_rt.dll
        HINTS ${SEARCH_HINTS}
        NO_DEFAULT_PATH) # Avoid searching default system paths implicitly if hints cover it

    # Find the OpenMP runtime DLL (libiomp5md.dll)
    find_file(IOMP5_DLL_PATH NAMES libiomp5md.dll
        HINTS ${SEARCH_HINTS}
        NO_DEFAULT_PATH)

    # Fallback search for libiomp5md only in PATH if not found yet
    if(NOT IOMP5_DLL_PATH)
        find_file(IOMP5_DLL_PATH NAMES libiomp5md.dll HINTS "$ENV{PATH}")
        if(IOMP5_DLL_PATH)
            message(STATUS "Python Bindings: Found OpenMP runtime DLL via fallback PATH search.")
        endif()
    endif()

    # Collect found DLLs
    set(MKL_DLLS_TO_INSTALL "")
    if(MKL_RT_DLL_PATH)
        list(APPEND MKL_DLLS_TO_INSTALL "${MKL_RT_DLL_PATH}")
        message(STATUS "Python Bindings: Found MKL runtime DLL: ${MKL_RT_DLL_PATH}")
    else()
        message(WARNING "Python Bindings: Could not find mkl_rt DLL using hints ${SEARCH_HINTS}. Python module might fail at runtime if MKL DLL is not in PATH.")
    endif()

    if(IOMP5_DLL_PATH)
        list(APPEND MKL_DLLS_TO_INSTALL "${IOMP5_DLL_PATH}")
        message(STATUS "Python Bindings: Found OpenMP runtime DLL: ${IOMP5_DLL_PATH}")
    else()
        message(WARNING "Python Bindings: Could not find libiomp5md.dll using hints ${SEARCH_HINTS}. Python module might fail at runtime if OpenMP is needed and DLL is not in PATH.")
    endif()

    # Install the found DLLs into the package directory created by scikit-build-core
    if(MKL_DLLS_TO_INSTALL)
        install(FILES ${MKL_DLLS_TO_INSTALL}
            DESTINATION "omnidsp_py" # Installs to <prefix>/omnidsp_py/
            COMPONENT Runtime # Match the component of the python module
        )
        message(STATUS "Configured installation of MKL/OpenMP DLLs: ${MKL_DLLS_TO_INSTALL} to omnidsp_py/ directory.")
        # message(STATUS "Skipping installation of MKL/OpenMP DLLs into package directory (relying on environment PATH).")
    else()
        message(WARNING "Python Bindings: No MKL/OpenMP runtime DLLs found to install. Ensure they are available in the system PATH for the Python module to load correctly.")
    endif()

elseif(WIN32)
    message(STATUS "Python Bindings: MKL backend not selected or MKL not found, skipping MKL DLL installation.")
endif()

# --- Install Python Module ---
# Install the Python module target itself. scikit-build-core requires this.
# The destination directory should match the package name specified in pyproject.toml's
# wheel.packages setting.
install(TARGETS omnidsp_py
    LIBRARY DESTINATION "omnidsp_py" # Installs to <prefix>/omnidsp_py/
    RUNTIME DESTINATION "omnidsp_py" # For Windows .pyd files
    COMPONENT Runtime # Typically 'Runtime' for installable artifacts
)

message(STATUS "Configured Python bindings target 'omnidsp_py' with installation rules.")
