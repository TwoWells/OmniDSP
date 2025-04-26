# FindIPP.cmake
#
# Finds the Intel(R) Integrated Performance Primitives (IPP) library.
#
# This module tries to find Intel IPP, assuming it might be installed
# within a Conda environment (typically dynamic/shared libraries). It handles
# differences between Windows and Linux installations found in Conda.
# It considers specific library naming conventions for threading layers,
# resolves inter-domain dependencies, and attempts to find required
# Intel runtime libraries on Linux. It also adds the 'ipp' subdirectory
# to the include path if component headers are located there.
#
# Input Variables:
#   IPPROOT             - Manually specify the root directory of the IPP installation.
#                         Overrides environment variable IPPROOT.
#   CONDA_PREFIX        - Environment variable usually set by Conda, pointing to the active environment.
#                         This module will try to use this if set.
#   ENV{IPPROOT}        - Environment variable specifying the IPP root directory. Used if IPPROOT is not set.
#
# Output Variables:
#   IPP_FOUND           - System has IPP and all REQUIRED components (and their dependencies).
#   IPP_INCLUDE_DIRS    - The IPP include directories (may include base and 'ipp' subdirectory).
#   IPP_LIBRARIES       - The core IPP libraries and dependencies (requested components,
#                         threading libs, Intel runtime libs on Linux) to link against (list of full paths).
#   IPP_DEFINITIONS     - Compiler definitions required (e.g., for threading layer).
#   IPP_VERSION         - The version of IPP found.
#   IPP::<component>    - Imported targets for each found component library (e.g., IPP::s, IPP::i).
#                         Requires CMake 3.1+
#   IPP::IPP            - Interface target linking all found components and includes/definitions/dependencies.
#                         Recommended way to link: target_link_libraries(my_target PRIVATE IPP::IPP)
#
# Options:
#   IPP_USE_THREADING_LAYER - ON/OFF (Default: OFF) - Link against the IPP Threading Layer libraries
#                             and enable necessary definitions.
#                             Requires the selected threading runtime (OpenMP or TBB) to be found.
#   IPP_THREADING_TYPE      - STRING (Default: "OpenMP") - Specify the threading layer to use.
#                             Currently supports "OpenMP" or "TBB". Case-insensitive.
#                             Influences which libraries are searched for (_tl_omp vs _tl_tbb)
#                             and which dependency package is found (FindOpenMP vs FindTBB).
#
# Components:
#   Specify components (e.g., s, i, vm, cv) using find_package(IPP COMPONENTS s i ...).
#   Component names are the short suffixes (e.g., 's' for ipps).
#   This module AUTOMATICALLY resolves inter-domain dependencies. For example,
#   requesting 'i' (Image Processing) will automatically cause 's' (Signal),
#   'vm' (Vector Math), and 'core' to be searched for and linked if found,
#   as per Intel IPP documentation. 'core' is always implicitly required.
#
# Example Usage:
#   find_package(IPP REQUIRED COMPONENTS s i) # Find core, signal, image processing (and their deps vm)
#   if(IPP_FOUND)
#     target_link_libraries(my_target PRIVATE IPP::IPP)
#     # Now you can use #include <ipps.h> or #include <ippi.h> directly
#   endif()
#
#   # Or with OpenMP threading:
#   set(IPP_USE_THREADING_LAYER ON)
#   set(IPP_THREADING_TYPE "OpenMP") # Default, but explicit
#   find_package(IPP REQUIRED COMPONENTS cv) # Finds cv, i, s, vm, core
#   if(IPP_FOUND)
#     target_link_libraries(my_target PRIVATE IPP::IPP)
#   endif()
#
# Notes on Conda & Linking:
#   - Assumes dynamic libraries (.dll/.so) as commonly packaged by Conda.
#   - Does not explicitly handle static linking complexities (mt suffixes, .a/.lib differences).
#   - Conda typically installs to $CONDA_PREFIX/lib (Linux) or $CONDA_PREFIX/Library/lib (Windows).
#   - On Linux, IPP may depend on Intel runtime libs (libirc, libsvml, libimf).
#     This script attempts to find them in the Conda prefix. Ensure the corresponding
#     Intel runtime packages are installed in your Conda environment.
#
# Runtime Performance Notes (OpenMP Affinity):
#   For optimal performance with certain threaded signal processing functions
#   (e.g., FFT, Div, Sqrt) on specific multi-core processors (especially those
#   sharing L2 cache across cores, like older Intel Core 2 or potentially relevant
#   modern architectures), Intel recommends setting the OpenMP environment variable:
#
#     KMP_AFFINITY=compact
#
#   This should be set in the *runtime environment* before executing the application,
#   not typically within the CMake build process itself. Refer to Intel IPP and
#   OpenMP documentation for details on thread affinity.
#

cmake_minimum_required(VERSION 3.9...3.25) # Need 3.9 for OpenMP::OpenMP target preference

# --- Options ---
option(IPP_USE_THREADING_LAYER "Enable IPP Threading Layer" OFF)
set(IPP_THREADING_TYPE "OpenMP" CACHE STRING "Threading layer type (OpenMP or TBB)")
set_property(CACHE IPP_THREADING_TYPE PROPERTY STRINGS OpenMP TBB)

# --- Standard FindPackage Handling ---
include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations) # Not actively used for config selection, but good practice

# --- Debugging ---
# set(IPP_DEBUG TRUE) # Uncomment to enable verbose debugging messages
macro(IPP_DEBUG_MSG _MSG)
    if(IPP_DEBUG)
        message(STATUS "[FindIPP DEBUG] ${_MSG}")
    endif()
endmacro()

# --- Helper Function for Path Escaping for Messages ---
# Escapes backslashes in a given path string for safe printing in messages.
function(IPP_GET_PRINTABLE_PATH _path_in _out_var)
    if(WIN32)
        string(REPLACE "\\" "\\\\" _printable_path "${_path_in}")
    else()
        set(_printable_path "${_path_in}")
    endif()
    set(${_out_var} "${_printable_path}" PARENT_SCOPE)
endfunction()

# --- Helper Function to Format Path List for Messages ---
# Takes a list of paths, escapes them, and joins them into a single string.
function(IPP_FORMAT_PATH_LIST_FOR_MESSAGE _path_list _out_var)
    set(_escaped_paths "")
    foreach(_path IN LISTS _path_list)
        IPP_GET_PRINTABLE_PATH("${_path}" _escaped_path)
        list(APPEND _escaped_paths "${_escaped_path}")
    endforeach()
    # Join with a comma and space for readability in the message
    string(REPLACE ";" ", " _formatted_list "${_escaped_paths}")
    set(${_out_var} "${_formatted_list}" PARENT_SCOPE)
endfunction()


# --- Define IPP Domain Dependencies ---
# Based on Intel Docs table (add others if known/needed)
# Format: set(_IPP_DEP_<component_code> "dep1;dep2;...")
set(_IPP_DEP_cc "core;vm;s;i")   # Color Conversion
set(_IPP_DEP_ch "core;vm;s")   # String Operations
set(_IPP_DEP_cv "core;vm;s;i")   # Computer Vision
set(_IPP_DEP_dc "core;vm;s")   # Data Compression
set(_IPP_DEP_i  "core;vm;s")   # Image Processing
set(_IPP_DEP_s  "core;vm")   # Signal Processing
set(_IPP_DEP_vm "core")       # Vector Math
# Components without explicitly listed dependencies (assume they only depend on core)
# These might need verification against specific IPP versions/docs if used.
set(_IPP_DEP_cp "core")       # Cryptography Primitives (Guessing core only)
set(_IPP_DEP_j  "core;i")     # JPEG (Guessing core+i)
set(_IPP_DEP_m  "core;vm")    # Matrix (Guessing core+vm)
set(_IPP_DEP_r  "core")       # Realistic Rendering (Guessing core only)
set(_IPP_DEP_sc "core;s")     # Speech Coding (Guessing core+s)
set(_IPP_DEP_sr "core;s")     # Speech Recognition (Guessing core+s)
set(_IPP_DEP_gen "core;s")    # Generic Signal Processing (Guessing core+s)
# Core depends on nothing else within IPP
set(_IPP_DEP_core "")

# Function to resolve all transitive dependencies
function(IPP_RESOLVE_DEPENDENCIES _requested_components _out_resolved_list_var)
    set(work_list ${_requested_components})
    set(resolved_list "")
    set(processed_marker "") # Track processed to avoid infinite loops

    while(work_list)
        list(GET work_list 0 current_comp)
        list(REMOVE_AT work_list 0)

        string(TOLOWER "${current_comp}" current_comp_lower)

        # Avoid processing the same component multiple times
        if(";${processed_marker};" MATCHES ";${current_comp_lower};")
            continue()
        endif()
        list(APPEND resolved_list ${current_comp_lower})
        set(processed_marker "${processed_marker};${current_comp_lower};") # Mark as processed

        # Check if this component has known dependencies
        if(DEFINED _IPP_DEP_${current_comp_lower})
            set(deps ${_IPP_DEP_${current_comp_lower}})
            IPP_DEBUG_MSG("Dependencies for '${current_comp_lower}': ${deps}")
            # Add dependencies to the work list if they aren't already resolved/processed
            foreach(dep ${deps})
                 string(TOLOWER "${dep}" dep_lower)
                 if(NOT ";${processed_marker};" MATCHES ";${dep_lower};")
                     list(APPEND work_list ${dep_lower})
                     IPP_DEBUG_MSG("Adding dependency '${dep_lower}' to work list.")
                 endif()
            endforeach()
        else()
             IPP_DEBUG_MSG("No defined dependencies for component '${current_comp_lower}'. Assuming only 'core'.")
             # Ensure core is processed if not already done
             if(NOT ";${processed_marker};" MATCHES ";core;")
                 list(APPEND work_list "core")
             endif()
        endif()
    endwhile()

    list(REMOVE_DUPLICATES resolved_list)
    set(${_out_resolved_list_var} ${resolved_list} PARENT_SCOPE)
endfunction()


# --- Find Paths Strategy ---
set(_IPP_SEARCH_PATHS "")
# 1. User-provided CMake variable hint
if(DEFINED IPPROOT)
    IPP_DEBUG_MSG("User provided IPPROOT variable: ${IPPROOT}")
    list(APPEND _IPP_SEARCH_PATHS "${IPPROOT}")
# 2. User-provided Environment variable hint
elseif(DEFINED ENV{IPPROOT})
    IPP_DEBUG_MSG("Using environment variable IPPROOT: $ENV{IPPROOT}")
    list(APPEND _IPP_SEARCH_PATHS "$ENV{IPPROOT}")
endif()
# 3. Conda Environment Prefix
set(_CONDA_PREFIX_ENV "$ENV{CONDA_PREFIX}")
if(_CONDA_PREFIX_ENV)
    IPP_DEBUG_MSG("Found CONDA_PREFIX environment variable: ${_CONDA_PREFIX_ENV}")
    list(APPEND _IPP_SEARCH_PATHS "${_CONDA_PREFIX_ENV}")
else()
    # Attempt to guess Conda prefix from CMake's location (less reliable)
    get_filename_component(_CMAKE_COMMAND_DIR "${CMAKE_COMMAND}" DIRECTORY)
    if(_CMAKE_COMMAND_DIR MATCHES "(.+)/bin$")
        set(_GUESS_CONDA_PREFIX "${CMAKE_MATCH_1}")
        if(EXISTS "${_GUESS_CONDA_PREFIX}/conda-meta")
            IPP_DEBUG_MSG("Guessed CONDA_PREFIX from CMake location: ${_GUESS_CONDA_PREFIX}")
            list(APPEND _IPP_SEARCH_PATHS "${_GUESS_CONDA_PREFIX}")
        else()
            IPP_DEBUG_MSG("CMake location ${_CMAKE_COMMAND_DIR} doesn't look like a standard Conda bin dir parent.")
        endif()
    else()
       IPP_DEBUG_MSG("CONDA_PREFIX environment variable not set, and couldn't guess from CMake location.")
    endif()
endif()
# 4. Standard system paths (will be searched by find_path/find_library anyway)

# Define platform-specific subdirectories within the search paths
set(_IPP_INCLUDE_SUBDIRS "include") # Default for Linux/macOS
set(_IPP_LIB_SUBDIRS "lib")         # Default for Linux/macOS
if(WIN32)
    # On Windows, Conda might use Library/include or just include
    # Prefer Library/include but check both.
    set(_IPP_INCLUDE_SUBDIRS "Library/include" "include")
    # Similarly for libraries, prefer Library/lib but check lib and bin (for DLLs)
    set(_IPP_LIB_SUBDIRS "Library/lib" "lib" "Library/bin" "bin")
endif()
IPP_DEBUG_MSG("Searching include suffixes: ${_IPP_INCLUDE_SUBDIRS}")
IPP_DEBUG_MSG("Searching library suffixes: ${_IPP_LIB_SUBDIRS}")


# --- Find Include Directory ---
find_path(IPP_INCLUDE_DIR
    NAMES ipp.h # Look for the main header first
    HINTS ${_IPP_SEARCH_PATHS}
    PATH_SUFFIXES ${_IPP_INCLUDE_SUBDIRS} # Use the potentially expanded list for Windows
    DOC "Intel IPP include directory"
)
mark_as_advanced(IPP_INCLUDE_DIR)

# Initialize main output variables
set(IPP_INCLUDE_DIRS "") # Will contain base include dir and potentially ./ipp subdir
set(IPP_LIBRARIES "")
set(IPP_DEFINITIONS "")

if(IPP_INCLUDE_DIR)
    # Use the helper to get a printable path for debug messages
    IPP_GET_PRINTABLE_PATH("${IPP_INCLUDE_DIR}" _PRINTABLE_IPP_INCLUDE_DIR)
    IPP_DEBUG_MSG("Found IPP include directory: ${_PRINTABLE_IPP_INCLUDE_DIR}")
    # Always add the base include directory
    list(APPEND IPP_INCLUDE_DIRS "${IPP_INCLUDE_DIR}")

    # Check if the 'ipp' subdirectory exists and add it if it does
    set(_IPP_SUBDIR_INCLUDE "${IPP_INCLUDE_DIR}/ipp")
    if(IS_DIRECTORY "${_IPP_SUBDIR_INCLUDE}")
        IPP_GET_PRINTABLE_PATH("${_IPP_SUBDIR_INCLUDE}" _PRINTABLE_IPP_SUBDIR_INCLUDE)
        IPP_DEBUG_MSG("Found IPP component include subdirectory: ${_PRINTABLE_IPP_SUBDIR_INCLUDE}")
        list(APPEND IPP_INCLUDE_DIRS "${_IPP_SUBDIR_INCLUDE}")
    else()
        IPP_DEBUG_MSG("IPP component include subdirectory (${_IPP_SUBDIR_INCLUDE}) not found.")
    endif()
    list(REMOVE_DUPLICATES IPP_INCLUDE_DIRS) # Ensure uniqueness
else()
    IPP_DEBUG_MSG("IPP include directory (ipp.h) not found in HINTS combined with PATH_SUFFIXES.")
    # FPHSA will handle failure later if headers are required
endif()


# --- Process Threading Options ---
set(_IPP_THREADING_SFX "")
set(_IPP_THREADING_PACKAGE "")
set(_IPP_THREADING_FOUND FALSE)
set(_IPP_THREADING_TARGET "")
set(_IPP_THREADING_LIBS "")
set(_IPP_THREADING_COMPILE_OPTIONS "")

if(IPP_USE_THREADING_LAYER)
    string(TOLOWER "${IPP_THREADING_TYPE}" _IPP_THREADING_TYPE_LOWER)
    if(_IPP_THREADING_TYPE_LOWER STREQUAL "openmp")
        set(_IPP_THREADING_SFX "omp")
        set(_IPP_THREADING_PACKAGE "OpenMP")
        find_package(OpenMP QUIET)
        if(OpenMP_FOUND)
           set(_IPP_THREADING_FOUND TRUE)
           if(TARGET OpenMP::OpenMP) # Prefer target (CMake >= 3.9)
               set(_IPP_THREADING_TARGET OpenMP::OpenMP)
               IPP_DEBUG_MSG("Found OpenMP (using target OpenMP::OpenMP).")
           else() # Fallback to variables
               set(_IPP_THREADING_TARGET "") # No target
               set(_IPP_THREADING_LIBS ${OpenMP_CXX_LIBRARIES})
               set(_IPP_THREADING_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
               IPP_DEBUG_MSG("Found OpenMP (using variables OpenMP_CXX_LIBRARIES). Flags: ${OpenMP_CXX_FLAGS}")
           endif()
        else()
            IPP_DEBUG_MSG("OpenMP requested for threading layer, but FindOpenMP failed.")
        endif()
    elseif(_IPP_THREADING_TYPE_LOWER STREQUAL "tbb")
        set(_IPP_THREADING_SFX "tbb")
        set(_IPP_THREADING_PACKAGE "TBB")
        # Requires FindTBB.cmake to be available in CMAKE_MODULE_PATH
        find_package(TBB QUIET)
        if(TBB_FOUND)
            set(_IPP_THREADING_FOUND TRUE)
            # TBB target names can vary (TBB::tbb, TBB::tbb_release, etc.)
            if(TARGET TBB::tbb)
                 set(_IPP_THREADING_TARGET TBB::tbb)
                 IPP_DEBUG_MSG("Found TBB (using target TBB::tbb).")
            elseif(TARGET TBB::tbb_release) # Example alternative
                 set(_IPP_THREADING_TARGET TBB::tbb_release)
                 IPP_DEBUG_MSG("Found TBB (using target TBB::tbb_release).")
            else() # Fallback to variables
                 set(_IPP_THREADING_TARGET "")
                 set(_IPP_THREADING_LIBS ${TBB_LIBRARIES}) # Variable contains library list
                 IPP_DEBUG_MSG("Found TBB (using variable TBB_LIBRARIES: ${TBB_LIBRARIES}).")
            endif()
        else()
             IPP_DEBUG_MSG("TBB requested for threading layer, but FindTBB failed.")
        endif()
    else()
        message(WARNING "[FindIPP] Invalid IPP_THREADING_TYPE: '${IPP_THREADING_TYPE}'. Supported: OpenMP, TBB.")
    endif()

    if(NOT _IPP_THREADING_FOUND)
        message(WARNING "[FindIPP] IPP_USE_THREADING_LAYER is ON, but the required threading library (${IPP_THREADING_PACKAGE}) was NOT found. Threading layer will not be enabled.")
        set(IPP_USE_THREADING_LAYER OFF) # Force disable if backend not found
    else()
         IPP_DEBUG_MSG("Threading layer enabled with ${_IPP_THREADING_PACKAGE} (suffix: ${_IPP_THREADING_SFX})")
         list(APPEND IPP_DEFINITIONS "IPP_ENABLED_THREADING_LAYER_REDEFINITIONS")
         # Add compile flags from threading package if they exist (primarily for older OpenMP variable approach)
         if(_IPP_THREADING_COMPILE_OPTIONS)
             list(APPEND IPP_DEFINITIONS ${_IPP_THREADING_COMPILE_OPTIONS})
         endif()
    endif()
endif()

# --- Determine Required Components (with Dependency Resolution) ---
set(_IPP_USER_REQUESTED_COMPONENTS "")
if(IPP_FIND_COMPONENTS)
    set(_IPP_USER_REQUESTED_COMPONENTS ${IPP_FIND_COMPONENTS})
else()
    IPP_DEBUG_MSG("No specific components requested by user.")
endif()

# Always start with 'core' plus what the user requested
set(_IPP_INITIAL_COMPONENT_LIST "core")
list(APPEND _IPP_INITIAL_COMPONENT_LIST ${_IPP_USER_REQUESTED_COMPONENTS})
list(REMOVE_DUPLICATES _IPP_INITIAL_COMPONENT_LIST)
IPP_DEBUG_MSG("Initial component list (user requested + core): ${_IPP_INITIAL_COMPONENT_LIST}")

# Resolve all dependencies
IPP_RESOLVE_DEPENDENCIES("${_IPP_INITIAL_COMPONENT_LIST}" _IPP_ALL_REQUIRED_COMPONENTS)
IPP_DEBUG_MSG("Resolved component list (including dependencies): ${_IPP_ALL_REQUIRED_COMPONENTS}")


# --- Find Component Libraries (Using Resolved List) ---
set(_IPP_FOUND_LIB_TARGETS "") # List of found component targets for IPP::IPP
set(_IPP_FOUND_LIB_PATHS "") # List of found library file paths
set(_IPP_FOUND_COMPONENTS "") # List of component names successfully found
set(_IPP_MISSING_COMPONENTS "") # List of component names required but not found

# Use the fully resolved list of components to search for libraries
foreach(_COMPONENT ${_IPP_ALL_REQUIRED_COMPONENTS})
    # Note: _COMPONENT is already lowercase from IPP_RESOLVE_DEPENDENCIES
    set(_LIB_BASENAME "ipp${_COMPONENT}")
    set(_LIB_NAMES "") # List of names to search for (e.g., threaded and non-threaded)

    # Determine library names based on threading choice
    if(IPP_USE_THREADING_LAYER AND _IPP_THREADING_FOUND)
        # Prefer threaded name (dynamic convention: _tl_), fall back to non-threaded
        list(APPEND _LIB_NAMES "${_LIB_BASENAME}_tl_${_IPP_THREADING_SFX}")
        list(APPEND _LIB_NAMES "${_LIB_BASENAME}") # Fallback
        IPP_DEBUG_MSG("Component ${_COMPONENT}: Searching for threaded names: ${_LIB_BASENAME}_tl_${_IPP_THREADING_SFX}; ${_LIB_BASENAME}")
    else()
        # Only search for non-threaded name
        list(APPEND _LIB_NAMES "${_LIB_BASENAME}")
        IPP_DEBUG_MSG("Component ${_COMPONENT}: Searching for standard name: ${_LIB_BASENAME}")
    endif()

    # Find the library for this component
    find_library(IPP_${_COMPONENT}_LIBRARY # Sets this variable with the full path
        NAMES ${_LIB_NAMES}
        HINTS ${_IPP_SEARCH_PATHS}
        PATH_SUFFIXES ${_IPP_LIB_SUBDIRS} # Use the potentially expanded list for Windows
        DOC "Intel IPP ${_COMPONENT} component library"
    )
    mark_as_advanced(IPP_${_COMPONENT}_LIBRARY)

    if(IPP_${_COMPONENT}_LIBRARY)
        IPP_GET_PRINTABLE_PATH("${IPP_${_COMPONENT}_LIBRARY}" _PRINTABLE_LIB_PATH) # Use helper for debug msg
        IPP_DEBUG_MSG("Found component library '${_COMPONENT}': ${_PRINTABLE_LIB_PATH}")

        list(APPEND _IPP_FOUND_LIB_PATHS "${IPP_${_COMPONENT}_LIBRARY}") # Add path to list
        list(APPEND _IPP_FOUND_COMPONENTS ${_COMPONENT}) # Mark component as found

        # Create IMPORTED target for the component library if possible
        if(NOT TARGET IPP::${_COMPONENT})
             # Check if the variable contains a valid library file path before creating target
            if(EXISTS "${IPP_${_COMPONENT}_LIBRARY}")
                add_library(IPP::${_COMPONENT} UNKNOWN IMPORTED) # UNKNOWN handles .so, .dll, .lib
                set_target_properties(IPP::${_COMPONENT} PROPERTIES IMPORTED_LOCATION "${IPP_${_COMPONENT}_LIBRARY}")
                # Interface includes/defines will be handled by the aggregate IPP::IPP target
                IPP_DEBUG_MSG("Created imported target IPP::${_COMPONENT}")
                list(APPEND _IPP_FOUND_LIB_TARGETS IPP::${_COMPONENT}) # Add target to list
            else()
                 IPP_DEBUG_MSG("Variable IPP_${_COMPONENT}_LIBRARY ('${IPP_${_COMPONENT}_LIBRARY}') does not point to an existing file. Cannot create target IPP::${_COMPONENT}.")
                 set(IPP_${_COMPONENT}_LIBRARY "") # Clear the invalid variable
            endif()
        else()
            IPP_DEBUG_MSG("Target IPP::${_COMPONENT} already exists.")
            # Ensure existing target is added to our list if not already there
            if(NOT IPP::${_COMPONENT} IN_LIST _IPP_FOUND_LIB_TARGETS)
                 list(APPEND _IPP_FOUND_LIB_TARGETS IPP::${_COMPONENT})
            endif()
        endif()
    else()
        IPP_DEBUG_MSG("Could NOT find component library '${_COMPONENT}' (Searched for: ${_LIB_NAMES}).")
    endif()
endforeach()

# Populate the list of missing components based on the search results
foreach(_COMPONENT ${_IPP_ALL_REQUIRED_COMPONENTS})
    if(NOT _COMPONENT IN_LIST _IPP_FOUND_COMPONENTS)
        list(APPEND _IPP_MISSING_COMPONENTS ${_COMPONENT})
    endif()
endforeach()


# --- Linux: Find Intel Runtime Dependencies ---
set(_IPP_INTEL_RUNTIME_LIBS "") # List of library paths
set(_IPP_INTEL_RUNTIME_TARGETS "") # List of targets
# Only search on Linux-like systems (excluding macOS, which doesn't typically use these)
if(IPP_INCLUDE_DIR AND UNIX AND NOT APPLE AND NOT WIN32)
    IPP_DEBUG_MSG("Checking for Intel runtime libraries on Linux...")
    # Base names for libirc.a/so, libsvml.a/so, libimf.a/so
    set(_INTEL_RUNTIME_NAMES irc svml imf)
    foreach(_RUNTIME_NAME ${_INTEL_RUNTIME_NAMES})
        find_library(IPP_${_RUNTIME_NAME}_LIBRARY
            NAMES ${_RUNTIME_NAME} # CMake prepends 'lib' and adds '.so'/'.a'
            HINTS ${_IPP_SEARCH_PATHS} # Search in conda prefix first
            PATH_SUFFIXES ${_IPP_LIB_SUBDIRS} "/opt/intel/lib" "/opt/intel/compiler/lib/intel64" # Add typical Intel install locations too
            DOC "Intel runtime library ${_RUNTIME_NAME} dependency for IPP"
        )
        mark_as_advanced(IPP_${_RUNTIME_NAME}_LIBRARY)

        if(IPP_${_RUNTIME_NAME}_LIBRARY)
            IPP_GET_PRINTABLE_PATH("${IPP_${_RUNTIME_NAME}_LIBRARY}" _PRINTABLE_LIB_PATH) # Use helper
            IPP_DEBUG_MSG("Found Intel runtime library: ${_PRINTABLE_LIB_PATH}")
            list(APPEND _IPP_INTEL_RUNTIME_LIBS "${IPP_${_RUNTIME_NAME}_LIBRARY}")

            # Create target if it doesn't exist
            if(NOT TARGET IPP::${_RUNTIME_NAME})
                 if(EXISTS "${IPP_${_RUNTIME_NAME}_LIBRARY}")
                    add_library(IPP::${_RUNTIME_NAME} UNKNOWN IMPORTED)
                    set_target_properties(IPP::${_RUNTIME_NAME} PROPERTIES IMPORTED_LOCATION "${IPP_${_RUNTIME_NAME}_LIBRARY}")
                    IPP_DEBUG_MSG("Created imported target IPP::${_RUNTIME_NAME}")
                    list(APPEND _IPP_INTEL_RUNTIME_TARGETS IPP::${_RUNTIME_NAME})
                 else()
                     IPP_DEBUG_MSG("Variable IPP_${_RUNTIME_NAME}_LIBRARY invalid, cannot create target.")
                     set(IPP_${_RUNTIME_NAME}_LIBRARY "") # Clear invalid var
                 endif()
            else()
                 # Ensure existing target is added to our list
                 if(NOT IPP::${_RUNTIME_NAME} IN_LIST _IPP_INTEL_RUNTIME_TARGETS)
                     list(APPEND _IPP_INTEL_RUNTIME_TARGETS IPP::${_RUNTIME_NAME})
                 endif()
            endif()
        else()
            IPP_DEBUG_MSG("Could NOT find Intel runtime library: lib${_RUNTIME_NAME}. Linker errors may occur if IPP requires it.")
        endif()
    endforeach()
endif()


# --- Parse Version ---
set(IPP_VERSION "0.0.0") # Default version
set(_IPP_VERSION_FILE "") # Initialize version file path

if(IPP_INCLUDE_DIR)
    # Check for ippversion.h directly in the include directory
    set(_VERSION_PATH_DIRECT "${IPP_INCLUDE_DIR}/ippversion.h")
    # Check for ippversion.h in the 'ipp' subdirectory
    set(_VERSION_PATH_SUBDIR "${IPP_INCLUDE_DIR}/ipp/ippversion.h")

    if(EXISTS "${_VERSION_PATH_DIRECT}")
        set(_IPP_VERSION_FILE "${_VERSION_PATH_DIRECT}")
        IPP_GET_PRINTABLE_PATH("${_IPP_VERSION_FILE}" _PRINTABLE_VERSION_FILE)
        IPP_DEBUG_MSG("Found version file at: ${_PRINTABLE_VERSION_FILE}")
    elseif(EXISTS "${_VERSION_PATH_SUBDIR}")
        set(_IPP_VERSION_FILE "${_VERSION_PATH_SUBDIR}")
        IPP_GET_PRINTABLE_PATH("${_IPP_VERSION_FILE}" _PRINTABLE_VERSION_FILE)
        IPP_DEBUG_MSG("Found version file at: ${_PRINTABLE_VERSION_FILE}")
    else()
        IPP_GET_PRINTABLE_PATH("${IPP_INCLUDE_DIR}" _PRINTABLE_INCLUDE_DIR)
        IPP_DEBUG_MSG("Could not find ippversion.h in ${_PRINTABLE_INCLUDE_DIR} or ${_PRINTABLE_INCLUDE_DIR}/ipp")
    endif()
endif()

if(_IPP_VERSION_FILE) # Proceed only if the version file was found
    IPP_DEBUG_MSG("Parsing version from: ${_IPP_VERSION_FILE}")
    # Try parsing MAJOR, MINOR, UPDATE first
    file(STRINGS "${_IPP_VERSION_FILE}" _IPP_VERSION_MAJOR_LINE REGEX "^#define +IPP_VERSION_MAJOR +[0-9]+")
    file(STRINGS "${_IPP_VERSION_FILE}" _IPP_VERSION_MINOR_LINE REGEX "^#define +IPP_VERSION_MINOR +[0-9]+")
    file(STRINGS "${_IPP_VERSION_FILE}" _IPP_VERSION_UPDATE_LINE REGEX "^#define +IPP_VERSION_UPDATE +[0-9]+") # Or IPP_VERSION_BUILD

    if(_IPP_VERSION_MAJOR_LINE AND _IPP_VERSION_MINOR_LINE AND _IPP_VERSION_UPDATE_LINE)
        string(REGEX MATCH "[0-9]+$" IPP_VERSION_MAJOR "${_IPP_VERSION_MAJOR_LINE}")
        string(REGEX MATCH "[0-9]+$" IPP_VERSION_MINOR "${_IPP_VERSION_MINOR_LINE}")
        string(REGEX MATCH "[0-9]+$" IPP_VERSION_UPDATE "${_IPP_VERSION_UPDATE_LINE}")
        set(IPP_VERSION "${IPP_VERSION_MAJOR}.${IPP_VERSION_MINOR}.${IPP_VERSION_UPDATE}")
        IPP_DEBUG_MSG("Found IPP Version: ${IPP_VERSION}")
    else()
        # Fallback to parsing IPP_VERSION_STR
        file(STRINGS "${_IPP_VERSION_FILE}" _IPP_VERSION_STR_LINE REGEX "^#define +IPP_VERSION_STR +\"[^\"]+\"")
        if (_IPP_VERSION_STR_LINE)
            string(REGEX MATCH "\"[^\"]+\"" _IPP_VERSION_MATCH "${_IPP_VERSION_STR_LINE}")
            string(REPLACE "\"" "" IPP_VERSION "${_IPP_VERSION_MATCH}") # Remove quotes
            IPP_DEBUG_MSG("Found IPP Version String: ${IPP_VERSION}")
        else()
            IPP_DEBUG_MSG("Could not parse version components or string from ${_IPP_VERSION_FILE}")
        endif()
    endif()
endif()


# --- Final Check and Set Variables ---

# Check if all required components (user requested + dependencies) were found.
set(_IPP_ALL_REQUIRED_COMPONENTS_FOUND TRUE)
if(_IPP_MISSING_COMPONENTS)
    set(_IPP_ALL_REQUIRED_COMPONENTS_FOUND FALSE)
endif()

# Determine required variables for FPHSA.
# At minimum, need the include directory.
# Also require the library variable for the 'core' component if it was found.
set(_FPHSA_REQUIRED_VARS IPP_INCLUDE_DIR) # IPP_INCLUDE_DIR must be set for FPHSA to pass
if("core" IN_LIST _IPP_FOUND_COMPONENTS)
    list(APPEND _FPHSA_REQUIRED_VARS IPP_core_LIBRARY)
endif()

# Format the search path list for potential use in the failure message
IPP_FORMAT_PATH_LIST_FOR_MESSAGE("${_IPP_SEARCH_PATHS}" _IPP_PRINTABLE_SEARCH_PATHS)

# Use standard handling to set IPP_FOUND based on headers/core.
# This variable might be set back to FALSE later if components are missing.
find_package_handle_standard_args(IPP
    FOUND_VAR IPP_FOUND # Use the standard variable name
    REQUIRED_VARS ${_FPHSA_REQUIRED_VARS}
    VERSION_VAR IPP_VERSION
    # Don't use HANDLE_COMPONENTS here, we do custom component logic
    FAIL_MESSAGE "Could NOT find critical IPP base components (Headers or Core Library). Searched hints: ${_IPP_PRINTABLE_SEARCH_PATHS}. Check IPP installation, IPPROOT, and CONDA_PREFIX."
)

# Now, refine IPP_FOUND based on whether all required components were found.
if(IPP_FOUND AND NOT _IPP_ALL_REQUIRED_COMPONENTS_FOUND)
    # FPHSA found headers/core, but we are missing other required components.
    set(IPP_FOUND FALSE) # Override FPHSA result
    IPP_DEBUG_MSG("FPHSA found base, but setting IPP_FOUND=FALSE due to missing required components.")
endif()


# --- Create Interface Target and Set Final Variables ---
if(IPP_FOUND) # Use the final, refined FOUND status
    # Combine all library paths and targets needed for linking
    set(IPP_LIBRARIES ${_IPP_FOUND_LIB_PATHS}) # Start with component library paths
    set(_IPP_ALL_TARGETS ${_IPP_FOUND_LIB_TARGETS}) # Start with component targets

    # Add threading library if enabled and found
    if(IPP_USE_THREADING_LAYER AND _IPP_THREADING_FOUND)
        if(_IPP_THREADING_TARGET) # Prefer linking the target
            list(APPEND _IPP_ALL_TARGETS ${_IPP_THREADING_TARGET})
        else() # Fallback to library path if no target
            list(APPEND IPP_LIBRARIES ${_IPP_THREADING_LIBS})
        endif()
    endif()

    # Add Intel runtime libs on Linux
    if(_IPP_INTEL_RUNTIME_LIBS)
        list(APPEND IPP_LIBRARIES ${_IPP_INTEL_RUNTIME_LIBS})
        list(APPEND _IPP_ALL_TARGETS ${_IPP_INTEL_RUNTIME_TARGETS}) # Corrected: Append targets
    endif()

    # Clean up lists before using them
    # IPP_INCLUDE_DIRS already cleaned up where it was populated
    list(REMOVE_DUPLICATES IPP_DEFINITIONS)
    list(REMOVE_DUPLICATES IPP_LIBRARIES)
    list(REMOVE_DUPLICATES _IPP_ALL_TARGETS)

    # Create the aggregate IPP::IPP interface target
    if(NOT TARGET IPP::IPP)
        add_library(IPP::IPP INTERFACE IMPORTED)
        set_target_properties(IPP::IPP PROPERTIES
            # Use the potentially multi-path IPP_INCLUDE_DIRS
            INTERFACE_INCLUDE_DIRECTORIES "${IPP_INCLUDE_DIRS}"
            INTERFACE_COMPILE_DEFINITIONS "${IPP_DEFINITIONS}"
            # Link the aggregated list of targets. Consumers linking IPP::IPP
            # will transitively link all component, threading, and runtime targets.
            INTERFACE_LINK_LIBRARIES "${_IPP_ALL_TARGETS}"
        )

        IPP_DEBUG_MSG("Created IPP::IPP interface target linking targets: ${_IPP_ALL_TARGETS}")
        IPP_DEBUG_MSG("Full library path list (IPP_LIBRARIES variable): ${IPP_LIBRARIES}")
    endif()

    # Set final output variables for potential non-target usage (less common)
    set(IPP_LIBRARIES ${IPP_LIBRARIES}) # Final list of paths
    set(IPP_INCLUDE_DIRS ${IPP_INCLUDE_DIRS}) # Final list of include paths
    set(IPP_DEFINITIONS ${IPP_DEFINITIONS})

    # Final status messages
    if(NOT IPP_FIND_QUIETLY)
        # Format lists nicely for printing
        string(REPLACE ";" "\n    " _LIBS_STR "${IPP_LIBRARIES}")
        # Format include dirs, potentially multiple paths now
        IPP_FORMAT_PATH_LIST_FOR_MESSAGE("${IPP_INCLUDE_DIRS}" _INCS_STR)
        string(REPLACE ";" ", " _DEFS_STR "${IPP_DEFINITIONS}")
        # Report components that were actually found
        string(REPLACE ";" ", " _FOUND_COMPONENTS_STR "${_IPP_FOUND_COMPONENTS}")

        message(STATUS "Found IPP: ${IPP_VERSION} (Found Components: ${_FOUND_COMPONENTS_STR})")
        message(STATUS "  Includes: ${_INCS_STR}") # Now shows potentially multiple paths
        message(STATUS "  Libraries (IPP_LIBRARIES):\n    ${_LIBS_STR}") # Print list line by line
        if(IPP_DEFINITIONS)
             message(STATUS "  Definitions: ${_DEFS_STR}")
        endif()
        if(IPP_USE_THREADING_LAYER) # Report status based on initial request vs final outcome
            if(_IPP_THREADING_FOUND)
                message(STATUS "  Threading Layer: Enabled (${IPP_THREADING_PACKAGE})")
            else() # Should not happen if IPP_FOUND is true, but for completeness
                message(STATUS "  Threading Layer: Requested but ${_IPP_THREADING_PACKAGE} NOT FOUND (Warning issued earlier)")
            endif()
        else()
            message(STATUS "  Threading Layer: Disabled")
        endif()
         if(UNIX AND NOT APPLE AND NOT WIN32) # Linux runtime check report
             if(NOT _IPP_INTEL_RUNTIME_LIBS) # Check if the list is empty
                  message(STATUS "  Intel Runtime Libs (Linux): Some or all NOT found (irc, svml, imf). Linking may fail.")
             else()
                  # Check if all three were found
                  list(LENGTH _IPP_INTEL_RUNTIME_LIBS _num_runtime_found)
                  if(_num_runtime_found LESS 3)
                       message(STATUS "  Intel Runtime Libs (Linux): Found some, but not all (irc, svml, imf). Linking may fail.")
                  else()
                       message(STATUS "  Intel Runtime Libs (Linux): Found and included.")
                  endif()
             endif()
         endif()
         message(STATUS "  Link via target: target_link_libraries(... IPP::IPP)")
    endif()

else()
    # IPP was not found OR a required component/dependency was missing
    if(NOT IPP_FIND_QUIETLY)
        # Only report missing components if that was the reason for failure AND IPP was required
        if(NOT _IPP_ALL_REQUIRED_COMPONENTS_FOUND AND IPP_FIND_REQUIRED)
             string(REPLACE ";" ", " _MISSING_COMPONENTS_STR "${_IPP_MISSING_COMPONENTS}")
             # Note: FPHSA might have already issued a FATAL_ERROR if headers/core were missing and required.
             # This message adds detail about missing components if that was the *specific* cause after finding headers/core.
             message(WARNING "[FindIPP] Could NOT find the following REQUIRED IPP components or their dependencies: ${_MISSING_COMPONENTS_STR}")
        # If components were optional and missing, no extra message needed beyond FPHSA's standard output (if it failed)
        # If FPHSA already failed (e.g., missing headers), its message is usually sufficient.
        endif()
    endif()
endif()

# Clean up internal variables
unset(_IPP_SEARCH_PATHS)
unset(_IPP_PRINTABLE_SEARCH_PATHS) # Clean up the formatted path string
unset(_IPP_INCLUDE_SUBDIRS)
unset(_IPP_LIB_SUBDIRS)
unset(_IPP_DEP_cc)
unset(_IPP_DEP_ch)
unset(_IPP_DEP_cv)
unset(_IPP_DEP_dc)
unset(_IPP_DEP_i)
unset(_IPP_DEP_s)
unset(_IPP_DEP_vm)
unset(_IPP_DEP_cp)
unset(_IPP_DEP_j)
unset(_IPP_DEP_m)
unset(_IPP_DEP_r)
unset(_IPP_DEP_sc)
unset(_IPP_DEP_sr)
unset(_IPP_DEP_gen)
unset(_IPP_DEP_core)
unset(_IPP_USER_REQUESTED_COMPONENTS)
unset(_IPP_INITIAL_COMPONENT_LIST)
unset(_IPP_ALL_REQUIRED_COMPONENTS)
unset(_IPP_FOUND_LIB_TARGETS)
unset(_IPP_FOUND_LIB_PATHS)
unset(_IPP_FOUND_COMPONENTS)
unset(_IPP_MISSING_COMPONENTS)
unset(_IPP_ALL_REQUIRED_COMPONENTS_FOUND)
unset(_IPP_THREADING_SFX)
unset(_IPP_THREADING_PACKAGE)
unset(_IPP_THREADING_FOUND)
unset(_IPP_THREADING_TARGET)
unset(_IPP_THREADING_LIBS)
unset(_IPP_THREADING_COMPILE_OPTIONS)
unset(_IPP_THREADING_TYPE_LOWER)
unset(_IPP_INTEL_RUNTIME_LIBS)
unset(_IPP_INTEL_RUNTIME_TARGETS)
unset(_INTEL_RUNTIME_NAMES)
unset(_num_runtime_found)
unset(_IPP_VERSION_FILE)
unset(_IPP_VERSION_MAJOR_LINE)
unset(_IPP_VERSION_MINOR_LINE)
unset(_IPP_VERSION_UPDATE_LINE)
unset(IPP_VERSION_MAJOR)
unset(IPP_VERSION_MINOR)
unset(IPP_VERSION_UPDATE)
unset(_IPP_VERSION_STR_LINE)
unset(_IPP_VERSION_MATCH)
unset(_FPHSA_REQUIRED_VARS)
unset(_PRINTABLE_IPP_INCLUDE_DIR)
unset(_PRINTABLE_LIB_PATH)
unset(_PRINTABLE_TARGETS)
unset(_LIBS_STR)
unset(_INCS_STR)
unset(_DEFS_STR)
unset(_FOUND_COMPONENTS_STR)
unset(_MISSING_COMPONENTS_STR)
unset(_CONDA_PREFIX_ENV)
unset(_CMAKE_COMMAND_DIR)
unset(_GUESS_CONDA_PREFIX)
unset(_IPP_ALL_TARGETS)
unset(_VERSION_PATH_DIRECT) # Clean up version check paths
unset(_VERSION_PATH_SUBDIR) # Clean up version check paths
unset(_IPP_SUBDIR_INCLUDE) # Clean up include subdir check path
unset(_PRINTABLE_IPP_SUBDIR_INCLUDE) # Clean up include subdir check path
unset(_PRINTABLE_VERSION_FILE) # Clean up version file path
