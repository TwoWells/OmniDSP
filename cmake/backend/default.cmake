# ==============================================================================
# cmake/backend/default.cmake
# ==============================================================================
#
# Brief Description:
#   Configures the 'omnidsp_backend_default' OBJECT library target.
#   This backend provides portable C++ implementations (potentially SIMD
#   accelerated via Highway if enabled) and is typically always included.
#
# Variables Read:
#   - CMAKE_SOURCE_DIR, PROJECT_BINARY_DIR (CMake Built-in)
#   - CMAKE_CURRENT_SOURCE_DIR (CMake Built-in)
#   - Boost::boost, hwy, hwy_contrib targets (Expected from cmake/depend.cmake - VERIFY NAMES)
#   - Boost_INCLUDE_DIRS (Variable set by find_package(Boost))
#   - highway_SOURCE_DIR (Variable set by FetchContent_MakeAvailable(highway))
#
# Variables Set (in Caller's Scope via include()):
#   - OMNIDSP_SELECTED_BACKEND_TARGETS (Appends 'omnidsp_backend_default')
#
# Targets Defined:
#   - omnidsp_backend_default (OBJECT Library)
#
# ==============================================================================

message(STATUS "  Configuring Default backend target...")

# --- Define Backend Target ---
add_library(omnidsp_backend_default OBJECT)

# --- Define Backend Directory (for convenience) ---
set(OMNIDSP_DEFAULT_BACKEND_SRC_DIR ${CMAKE_SOURCE_DIR}/src/omnidsp/default)
message(STATUS "    Default backend source directory: ${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}")

# --- Backend Sources ---
target_sources(omnidsp_backend_default
    PRIVATE
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/backend.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/fft.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/window.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/convolution.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/resample.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/cqt.cpp"
        "${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}/filter.cpp"
)

# --- Get Include Directories from Dependencies ---
# Retrieve INTERFACE_INCLUDE_DIRECTORIES directly from targets,
# or use fallback variables like Boost_INCLUDE_DIRS.

# Boost Includes
set(BOOST_INCLUDES "") # Initialize to empty
if(TARGET Boost::boost)
    message(STATUS "DEBUG Default.cmake: Found TARGET Boost::boost.")
    # Try getting the modern property first
    get_target_property(TEMP_BOOST_INCLUDES Boost::boost INTERFACE_INCLUDE_DIRECTORIES)
    if(TEMP_BOOST_INCLUDES) # Check if the retrieved value is non-empty and not NOTFOUND
         set(BOOST_INCLUDES ${TEMP_BOOST_INCLUDES})
         message(STATUS "DEBUG Default.cmake: Successfully retrieved BOOST_INCLUDES from INTERFACE property: ${BOOST_INCLUDES}")
    else()
        # Fallback: Check if the standard Boost_INCLUDE_DIRS variable is set
        if(DEFINED Boost_INCLUDE_DIRS AND Boost_INCLUDE_DIRS)
             message(STATUS "DEBUG Default.cmake: INTERFACE_INCLUDE_DIRECTORIES not set/empty on Boost::boost. Falling back to Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
             set(BOOST_INCLUDES ${Boost_INCLUDE_DIRS})
        else()
             message(WARNING "DEBUG Default.cmake: Target Boost::boost exists, but INTERFACE_INCLUDE_DIRECTORIES is not set/empty, AND Boost_INCLUDE_DIRS is not defined/empty.")
        endif()
    endif()
else()
    message(WARNING "DEBUG Default.cmake: Boost target (e.g., Boost::boost) not found. Checking Boost_INCLUDE_DIRS anyway...")
    # Fallback even if target doesn't exist, as find_package might just set the variable
    if(DEFINED Boost_INCLUDE_DIRS AND Boost_INCLUDE_DIRS)
         message(STATUS "DEBUG Default.cmake: Falling back to Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
         set(BOOST_INCLUDES ${Boost_INCLUDE_DIRS})
    else()
         message(WARNING "DEBUG Default.cmake: Boost_INCLUDE_DIRS is also not defined/empty.")
    endif()
endif()

# Highway Includes - We will add the source directory directly below
# Keep the checks for logging purposes
if(NOT TARGET hwy)
     message(WARNING "Highway target 'hwy' not found.")
endif()
if(NOT TARGET hwy_contrib)
     message(WARNING "Highway target 'hwy_contrib' not found.")
endif()


# --- Backend Include Directories ---
message(STATUS "DEBUG Default.cmake: Using BOOST_INCLUDES='${BOOST_INCLUDES}' in target_include_directories.")
# Check if highway_SOURCE_DIR is defined (it should be if FetchContent worked)
if(DEFINED highway_SOURCE_DIR)
    message(STATUS "DEBUG Default.cmake: Using highway_SOURCE_DIR='${highway_SOURCE_DIR}' for Highway includes.")
else()
    message(WARNING "DEBUG Default.cmake: highway_SOURCE_DIR is NOT DEFINED. Highway includes might be missing.")
endif()

target_include_directories(omnidsp_backend_default
    PUBLIC # Public headers needed by consumers (main include dir)
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>

    PRIVATE # Private headers needed only to compile this backend's sources
        ${OMNIDSP_DEFAULT_BACKEND_SRC_DIR}
        # Add path to generated export header (needed if sources include public headers)
        ${PROJECT_BINARY_DIR}/include # Or CMAKE_BINARY_DIR

        # Add include directories retrieved from targets/variables above
        ${BOOST_INCLUDES} # Use the variable populated above

        # Add Highway source directory directly. This directory contains the 'hwy' subdirectory.
        ${highway_SOURCE_DIR}
)

# --- Backend Dependencies ---
# Link dependencies required specifically by this backend's implementation.
# These are PRIVATE because the Default backend is an OBJECT library,
# and the main 'omnidsp' library will link the final versions.
# However, linking them here helps ensure properties like include paths are available.
# Note: The main omnidsp target also links these in target_definitions.cmake.
target_link_libraries(omnidsp_backend_default
    PRIVATE
        Boost::boost # Use actual Boost target name found!
        hwy
        hwy_contrib
)


# --- Backend Compile Definitions ---
target_compile_definitions(omnidsp_backend_default
    INTERFACE
        OMNIDSP_BACKEND_DEFAULT=1
)

# --- Add to Caller's Scope List ---
list(APPEND OMNIDSP_SELECTED_BACKEND_TARGETS "omnidsp_backend_default")

message(STATUS "    Configured Default backend target: omnidsp_backend_default")
