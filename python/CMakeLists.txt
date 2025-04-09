# python/CMakeLists.txt for OmniDSP Python Bindings

# Ensure pybind11 is available (it should be from the parent scope)
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
# - pybind11 headers are usually handled automatically by pybind11_add_module
# - Need the project's public include directory for OmniDSP headers
target_include_directories(omnidsp_py PRIVATE
    ${PROJECT_SOURCE_DIR}/include # Path to the top-level 'include' directory
)

# # --- MKL DLL Handling for Windows ---
# # If the MKL backend is selected and found (MKL_FOUND set in parent scope),
# # find the necessary MKL runtime DLLs and configure them to be installed (bundled)
# # alongside the Python .pyd module. This makes the Python package self-contained.
# if(WIN32 AND MKL_FOUND)
#     message(STATUS "Python Bindings: MKL backend detected, configuring DLL installation for omnidsp_py.")
# 
#     # --- Setup Search Hints for DLLs ---
#     set(MKL_BIN_HINTS "") # Hints specifically for MKL library paths
#     set(CMPLR_BIN_HINTS "") # Hints specifically for Intel Compiler library paths (for libiomp5md)
#     set(TBB_BIN_HINTS "") # Hints specifically for Intel TBB library paths
# 
#     # Define common relative paths within oneAPI component directories
#     set(MKL_RELATIVE_BIN_PATHS "latest/bin/intel64" "latest/bin" "bin/intel64" "bin")
#     set(CMPLR_RELATIVE_BIN_PATHS "latest/bin/intel64" "latest/bin" "latest/windows/bin")
#     set(TBB_RELATIVE_BIN_PATHS "latest/bin/intel64" "latest/bin") # Assuming TBB follows similar pattern
# 
#     # Add hints based on MKLROOT environment variable (if defined)
#     if(DEFINED ENV{MKLROOT})
#         foreach(REL_PATH ${MKL_RELATIVE_BIN_PATHS})
#             list(APPEND MKL_BIN_HINTS "$ENV{MKLROOT}/${REL_PATH}")
#         endforeach()
#         message(DEBUG "Python Bindings: Added MKL hints based on MKLROOT: $ENV{MKLROOT}")
#     endif()
# 
#     # Add hints based on TBBROOT environment variable (if defined)
#     if(DEFINED ENV{TBBROOT})
#         foreach(REL_PATH ${TBB_RELATIVE_BIN_PATHS})
#             list(APPEND TBB_BIN_HINTS "$ENV{TBBROOT}/${REL_PATH}")
#         endforeach()
#         message(DEBUG "Python Bindings: Added TBB hints based on TBBROOT: $ENV{TBBROOT}")
#     endif()
# 
#     # Add hints based on ONEAPI_ROOT environment variable (if defined)
#     if(DEFINED ENV{ONEAPI_ROOT})
#         # Hints for MKL relative to ONEAPI_ROOT
#         foreach(REL_PATH ${MKL_RELATIVE_BIN_PATHS})
#             list(APPEND MKL_BIN_HINTS "$ENV{ONEAPI_ROOT}/mkl/${REL_PATH}")
#         endforeach()
#         # Hints for Compiler relative to ONEAPI_ROOT
#         foreach(REL_PATH ${CMPLR_RELATIVE_BIN_PATHS})
#             list(APPEND CMPLR_BIN_HINTS "$ENV{ONEAPI_ROOT}/compiler/${REL_PATH}")
#         endforeach()
#         # Hints for TBB relative to ONEAPI_ROOT
#         foreach(REL_PATH ${TBB_RELATIVE_BIN_PATHS})
#             list(APPEND TBB_BIN_HINTS "$ENV{ONEAPI_ROOT}/tbb/${REL_PATH}")
#         endforeach()
#         message(DEBUG "Python Bindings: Added hints based on ONEAPI_ROOT: $ENV{ONEAPI_ROOT}")
#     endif()
# 
#     # Combine hints and add system PATH as fallback
#     set(SEARCH_HINTS_MKL ${MKL_BIN_HINTS})
#     list(APPEND SEARCH_HINTS_MKL "$ENV{PATH}") # Add system PATH last for MKL
# 
#     set(SEARCH_HINTS_OMP ${CMPLR_BIN_HINTS}) # Prioritize compiler path for OpenMP
#     list(APPEND SEARCH_HINTS_OMP ${MKL_BIN_HINTS}) # Also check MKL path for OpenMP
#     list(APPEND SEARCH_HINTS_OMP "$ENV{PATH}") # Add system PATH last for OpenMP
# 
#     set(SEARCH_HINTS_TBB ${TBB_BIN_HINTS}) # Use TBB hints
#     list(APPEND SEARCH_HINTS_TBB "$ENV{PATH}") # Add system PATH last for TBB
# 
#     # --- Find Required DLLs ---
#     message(DEBUG "Python Bindings: Searching for MKL RT DLLs with hints: ${SEARCH_HINTS_MKL}")
#     # Find the MKL runtime dispatcher DLL (e.g., mkl_rt.2.dll)
#     find_file(MKL_RT_DLL_PATH NAMES mkl_rt.2.dll mkl_rt.1.dll mkl_rt.dll
#         HINTS ${SEARCH_HINTS_MKL}
#         NO_DEFAULT_PATH) # Prefer hints over default system paths
# 
#     message(DEBUG "Python Bindings: Searching for MKL Threading DLLs with hints: ${SEARCH_HINTS_MKL}")
#     # Find the MKL threading DLL (e.g., mkl_intel_thread.2.dll)
#     find_file(MKL_THREAD_DLL_PATH NAMES mkl_intel_thread.2.dll mkl_intel_thread.1.dll mkl_intel_thread.dll
#         HINTS ${SEARCH_HINTS_MKL}
#         NO_DEFAULT_PATH)
# 
#     message(DEBUG "Python Bindings: Searching for MKL Core DLLs with hints: ${SEARCH_HINTS_MKL}")
#     # Find the MKL core DLL (e.g., mkl_core.2.dll)
#     find_file(MKL_CORE_DLL_PATH NAMES mkl_core.2.dll mkl_core.1.dll mkl_core.dll
#         HINTS ${SEARCH_HINTS_MKL}
#         NO_DEFAULT_PATH)
# 
#     message(DEBUG "Python Bindings: Searching for OpenMP DLL with hints: ${SEARCH_HINTS_OMP}")
#     # Find the Intel OpenMP runtime DLL (e.g., libiomp5md.dll)
#     find_file(IOMP5_DLL_PATH NAMES libiomp5md.dll
#         HINTS ${SEARCH_HINTS_OMP} # Use combined OMP hints
#         NO_DEFAULT_PATH)
# 
#     # (Optional but recommended) Find the Intel TBB runtime DLL (e.g., tbb12.dll)
#     # message(STATUS "Python Bindings: Searching for TBB DLL with hints: ${SEARCH_HINTS_TBB}")
#     # find_file(TBB_DLL_PATH NAMES tbb12.dll tbb.dll
#     #           HINTS ${SEARCH_HINTS_TBB}
#     #           NO_DEFAULT_PATH)
# 
#     # --- Collect Found DLLs into a List ---
#     set(MKL_DLLS_TO_INSTALL "") # Initialize the list
# 
#     # Add MKL Runtime Dispatcher DLL if found
#     if(MKL_RT_DLL_PATH)
#         list(APPEND MKL_DLLS_TO_INSTALL "${MKL_RT_DLL_PATH}")
#         message(VERBOSE "Python Bindings: Found MKL runtime DLL: ${MKL_RT_DLL_PATH}")
#     else()
#         message(WARNING "Python Bindings: Could not find mkl_rt DLL.")
#     endif()
# 
#     # Add MKL Threading DLL if found
#     if(MKL_THREAD_DLL_PATH)
#         list(APPEND MKL_DLLS_TO_INSTALL "${MKL_THREAD_DLL_PATH}")
#         message(VERBOSE "Python Bindings: Found MKL Threading DLL: ${MKL_THREAD_DLL_PATH}")
#     else()
#         message(WARNING "Python Bindings: Could not find MKL Threading DLL.")
#     endif()
# 
#     # Add MKL Core DLL if found
#     if(MKL_CORE_DLL_PATH)
#         list(APPEND MKL_DLLS_TO_INSTALL "${MKL_CORE_DLL_PATH}")
#         message(VERBOSE "Python Bindings: Found MKL Core DLL: ${MKL_CORE_DLL_PATH}")
#     else()
#         message(WARNING "Python Bindings: Could not find MKL Core DLL.")
#     endif()
# 
#     # Add OpenMP DLL if found
#     if(IOMP5_DLL_PATH)
#         list(APPEND MKL_DLLS_TO_INSTALL "${IOMP5_DLL_PATH}")
#         message(VERBOSE "Python Bindings: Found OpenMP runtime DLL: ${IOMP5_DLL_PATH}")
#     else()
#         message(WARNING "Python Bindings: Could not find libiomp5md.dll.")
#     endif()
# 
#     # (Optional) Add TBB DLL if found
#     # if(TBB_DLL_PATH)
#     #     list(APPEND MKL_DLLS_TO_INSTALL "${TBB_DLL_PATH}")
#     #     message(STATUS "Python Bindings: Found TBB runtime DLL: ${TBB_DLL_PATH}")
#     # else()
#     #     message(WARNING "Python Bindings: Could not find TBB runtime DLL (tbb12.dll/tbb.dll).")
#     # endif()
# 
#     # --- Install Collected DLLs ---
#     if(MKL_DLLS_TO_INSTALL)
#         # Install DLLs to the same destination component as the python module itself.
#         install(FILES ${MKL_DLLS_TO_INSTALL}
#             DESTINATION "${SKBUILD_PLATLIB_DIR}/omnidsp_py"
#             COMPONENT Runtime
#         )
#         message(VERBOSE "Configured installation of MKL/OpenMP DLLs: ${MKL_DLLS_TO_INSTALL} into '${SKBUILD_PLATLIB_DIR}/omnidsp_py/'.")
#     else()
#         message(WARNING "Python Bindings: No MKL/OpenMP runtime DLLs found to install. Package may fail at runtime if dependencies are not in PATH.")
#     endif()
# 
# elseif(WIN32) # Case where MKL is not used/found on Windows
#     message(STATUS "Python Bindings: MKL backend not selected or MKL not found, skipping MKL DLL installation.")
# endif() # End of WIN32 AND MKL_FOUND block


# --- Install __init__.py ---
# Copies the __init__.py file (which should contain the os.add_dll_directory logic)
# into the package directory. Assumes __init__.py exists in OmniDSP/python/
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    DESTINATION "${SKBUILD_PLATLIB_DIR}/omnidsp_py" # Same destination as the .pyd and DLLs
    COMPONENT Runtime # Match the component used for the target below
)

# --- Install Python Module Target ---

# --- Install Python Module Target ---

# 1. Install only the main target artifacts (.pyd, .so, .lib) first
install(TARGETS omnidsp_py
    RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/omnidsp_py" # Install .pyd here
    LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/omnidsp_py" # Install .so here
    ARCHIVE DESTINATION "${SKBUILD_PLATLIB_DIR}/omnidsp_py" # Install import .lib here (Windows)
    COMPONENT Runtime
)

# 2. Install runtime dependencies using the generator expression
# Requires CMake 3.21+
# $<TARGET_RUNTIME_DLLS:omnidsp_py> expands to the list of required runtime files
install(FILES $<TARGET_RUNTIME_DLLS:omnidsp_py>
    DESTINATION "${SKBUILD_PLATLIB_DIR}/omnidsp_py" # Install DLLs/shared libs here too
    COMPONENT Runtime
    OPTIONAL # Add OPTIONAL in case the list is empty (e.g., static linking or no external runtime deps)
)

# Keep the __init__.py installation
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    DESTINATION "${SKBUILD_PLATLIB_DIR}/omnidsp_py"
    COMPONENT Runtime
)

message(VERBOSE "Configured Python bindings target 'omnidsp_py' with installation into '${SKBUILD_PLATLIB_DIR}/omnidsp_py/'.")