# cmake/backend/onemkl.cmake
# ==========================
# Detects and configures the Intel oneMKL / IPP backend.
# Relies on find_package(MKL) respecting CMAKE_PREFIX_PATH (set by Conda env activation).
# Also adds IPP libraries by name, assuming they are findable in the link path.
#
# Variables Set (Current Scope, if MKL found):
# - OMNIDSP_HAS_ONEMKL (BOOL): TRUE if MKL is found.
# - ONEMKL_COMPILE_DEFINITION (STRING): Compile definition ("USE_ONEMKL").
# - ONEMKL_LINK_LIBS (LIST): Link libraries (MKL::MKL or MKL_LIBRARIES, ippcore, ipps).
# - ONEMKL_INCLUDE_DIRS (LIST): Additional include directories from MKL_INCLUDE_DIRS (if set and no MKL::MKL target).
# - ONEMKL_LINK_DIRS (LIST): Additional link directories (likely empty, relying on linker paths).
# - ONEMKL_BACKEND_SOURCES (LIST): List of source files for the oneMKL backend.
# - ONEMKL_BACKEND_INCLUDE_DIR (STRING): Path to the oneMKL backend's private include directory.
# ==========================

message(STATUS "Checking for Intel oneMKL backend...")

# Default to not found
set(OMNIDSP_HAS_ONEMKL FALSE)

# --- Attempt to find MKL ---
# Use find_package. If the Conda environment is activated, CMAKE_PREFIX_PATH
# should be set, allowing find_package to locate the MKL config files or FindMKL.cmake module.
# Using QUIET to avoid errors if not found; we handle the messaging.
find_package(MKL QUIET)

if(MKL_FOUND)
  message(STATUS "  Intel oneMKL backend available (version ${MKL_VERSION}).")
  set(OMNIDSP_HAS_ONEMKL TRUE)

  # --- Compile Definition ---
  set(ONEMKL_COMPILE_DEFINITION "USE_ONEMKL")
  message(STATUS "    Compile definition: ${ONEMKL_COMPILE_DEFINITION}") # Indentation: 4 spaces

  # --- Link Libraries ---
  set(ONEMKL_LINK_LIBS "") # Initialize list in current scope
  # Prefer using the imported target MKL::MKL if available (modern CMake)
  if(TARGET MKL::MKL)
    list(APPEND ONEMKL_LINK_LIBS MKL::MKL)
    message(STATUS "    Adding MKL link target: MKL::MKL") # Indentation: 4 spaces
    # Check if the target provides include directories (modern approach)
    get_target_property(MKL_INTERFACE_INCLUDE_DIRS MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
    if(MKL_INTERFACE_INCLUDE_DIRS)
        # Use helper for list formatting
        print_prefixed_list(STATUS "    " "MKL::MKL target provides include directories:" ${MKL_INTERFACE_INCLUDE_DIRS})
        # These will be automatically used when linking against MKL::MKL
        # We don't need to add them manually to ONEMKL_INCLUDE_DIRS below if linking the target.
    endif()
  elseif(MKL_LIBRARIES) # Fallback to using the MKL_LIBRARIES variable
    list(APPEND ONEMKL_LINK_LIBS ${MKL_LIBRARIES})
    # Use helper for list formatting
    print_prefixed_list(STATUS "    " "Adding MKL link libraries from MKL_LIBRARIES:" ${MKL_LIBRARIES})
  else()
    message(WARNING "    FindMKL found MKL but did not define MKL::MKL target or MKL_LIBRARIES variable. Linking might fail.") # Indentation: 4 spaces
  endif()

  # Add IPP libraries by name. Assume they are findable via standard link paths
  # (including those added to CMAKE_PREFIX_PATH by Conda).
  list(APPEND ONEMKL_LINK_LIBS ippcore ipps)
  message(STATUS "    Adding IPP link libraries by name: ippcore ipps") # Indentation: 4 spaces
  # Set the final list in the current scope
  set(ONEMKL_LINK_LIBS ${ONEMKL_LINK_LIBS})
  # Use helper for list formatting
  print_prefixed_list(STATUS "    " "Final oneMKL link libraries:" ${ONEMKL_LINK_LIBS})

  # --- Include Directories ---
  set(ONEMKL_INCLUDE_DIRS "") # Initialize list
  # Add include dirs provided by FindMKL (MKL_INCLUDE_DIRS variable), if any,
  # *unless* we are using the MKL::MKL target which handles its own includes.
  if(DEFINED MKL_INCLUDE_DIRS AND NOT MKL_INCLUDE_DIRS STREQUAL "" AND NOT TARGET MKL::MKL)
    list(APPEND ONEMKL_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    # Use helper for list formatting
    print_prefixed_list(STATUS "    " "Adding MKL include directories from FindMKL variable:" ${MKL_INCLUDE_DIRS})
  elseif(NOT TARGET MKL::MKL)
     message(STATUS "    No MKL_INCLUDE_DIRS variable set by FindMKL (and no MKL::MKL target found).") # Indentation: 4 spaces
  endif()
  # We no longer manually add Conda include paths here. find_package or the compiler's
  # default search paths (influenced by CMAKE_PREFIX_PATH) should handle finding MKL/IPP headers.
  # If headers (like ipps.h) are not found during compilation, it indicates the
  # environment or CMAKE_PREFIX_PATH is not set up correctly.
  set(ONEMKL_INCLUDE_DIRS ${ONEMKL_INCLUDE_DIRS})
  if(ONEMKL_INCLUDE_DIRS)
      # Use helper for list formatting
      print_prefixed_list(STATUS "    " "Additional oneMKL include directories (from MKL_INCLUDE_DIRS if set and no target):" ${ONEMKL_INCLUDE_DIRS})
  else()
       message(STATUS "    Additional oneMKL include directories: (None)") # Indentation: 4 spaces
  endif()


  # --- Link Directories ---
  # We no longer manually add Conda link paths. find_package (for MKL::MKL or MKL_LIBRARIES)
  # and the linker's default search paths (influenced by CMAKE_PREFIX_PATH / rpath settings)
  # should handle finding the libraries.
  set(ONEMKL_LINK_DIRS "") # Initialize empty list (likely not needed)
  message(STATUS "    Relying on CMAKE_PREFIX_PATH and standard linker search paths for MKL/IPP libraries.") # Indentation: 4 spaces


  # --- Source Files ---
  set(ONEMKL_BACKEND_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/fft.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/window.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/convolution.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/resample.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl/backend.cpp"
  )
   # Use helper for list formatting
   print_prefixed_list(STATUS "    " "oneMKL backend sources:" ${ONEMKL_BACKEND_SOURCES})


  # --- Backend Include Directory (for private backend headers) ---
  set(ONEMKL_BACKEND_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/src/omnidsp/backend/onemkl"
  )
  message(STATUS "    oneMKL backend include dir: ${ONEMKL_BACKEND_INCLUDE_DIR}") # Indentation: 4 spaces


else()
  # MKL not found
  message(STATUS "  Intel oneMKL not found.")
  message(WARNING "  MKL backend cannot be configured. Ensure the Conda environment is activated and contains the necessary MKL/IPP development packages (e.g., 'intel-mkl-devel', 'intel-ipp-devel'). Verify CMAKE_PREFIX_PATH is set correctly if using a custom installation.")
endif()
