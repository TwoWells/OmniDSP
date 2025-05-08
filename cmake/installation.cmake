# cmake/installation.cmake
# ========================
# Defines installation rules for the OmniDSP project.
# This includes installing:
#   - Public C++ headers
#   - The main C++ library target (OmniDSP)
#   - CMake package configuration files for find_package() support.
#   - Python example notebooks (if Python bindings are enabled and examples are built).
#
# Assumes standard CMake modules GNUInstallDirs and CMakePackageConfigHelpers have been included.
# Assumes the 'OmniDSP' target exists.
#
# Variables Read:
#   - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake)
#   - OMNIDSP_BUILD_EXAMPLES (from project_options.cmake, assumed for Python examples)
#   - PROJECT_NAME, PROJECT_VERSION (from project() command)
#   - CMAKE_INSTALL_*, CMAKE_SOURCE_DIR, PROJECT_BINARY_DIR (CMake Built-in)
#   - Config.cmake.in (Template file, path needs to be correct)
# ========================

message(STATUS "Configuring installation rules...")

# Ensure GNUInstallDirs has been included for CMAKE_INSTALL_* paths.
if(NOT CMAKE_INSTALL_LIBDIR)
    message(FATAL_ERROR "GNUInstallDirs must be included before installation.cmake to define CMAKE_INSTALL_* paths.")
endif()

# Check if the main library target exists
if(NOT TARGET OmniDSP)
    message(FATAL_ERROR "The 'OmniDSP' target must be defined before including installation.cmake.")
endif()

# --- C++ Library Installation (Headers, CMake Package, Library Files) ---
# This section should generally always run if you intend to install the C++ dev package.
# You might wrap this in an `if(OMNIDSP_INSTALL_CPP_DEVEL_FILES)` option if needed.

message(STATUS "  Configuring C++ library installation rules...")

# Define the installation path for CMake package files
set(OMNIDSP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}") # e.g., lib/cmake/OmniDSP
message(STATUS "    CMake package install directory: ${OMNIDSP_CMAKE_INSTALL_DIR}")

# --- Install Public Headers ---
# Install the public headers from the include/OmniDSP directory
install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/include/OmniDSP/" # Source directory (note trailing slash)
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/OmniDSP"  # Destination: <prefix>/include/OmniDSP
    COMPONENT Devel                                  # Component for development files
    FILES_MATCHING
    PATTERN "*.hpp"                                  # Pattern for header files
)
# Install the generated export header and config header
install(
    FILES
        "${PROJECT_BINARY_DIR}/include/OmniDSP/omnidsp_export.hpp"
        "${PROJECT_BINARY_DIR}/include/OmniDSP/omnidsp_config.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/OmniDSP" # Place it inside the installed OmniDSP header dir
    COMPONENT Devel
)
message(STATUS "    Installation rules added for public headers (including generated ones).")

# --- Generate and Install CMake Package Configuration Files ---
message(STATUS "    Configuring CMake package files...")

# Define the path to Config.cmake.in. Assuming it's in the 'cmake' directory.
set(CONFIG_CMAKE_IN_PATH "${CMAKE_SOURCE_DIR}/cmake/Config.cmake.in") # Adjusted path
if(NOT EXISTS "${CONFIG_CMAKE_IN_PATH}")
    message(FATAL_ERROR "Config.cmake.in not found at '${CONFIG_CMAKE_IN_PATH}'. This file is required for C++ installation.")
else()
    message(STATUS "      Using Config.cmake.in template: ${CONFIG_CMAKE_IN_PATH}")
    # Reminder: Developers must ensure Config.cmake.in includes necessary find_dependency calls:
    # find_dependency(Boost ${OMNIDSP_REQUIRED_BOOST_VERSION} CONFIG REQUIRED)
    # find_dependency(hwy ${OMNIDSP_REQUIRED_HIGHWAY_VERSION} CONFIG REQUIRED)
    # These should match the PUBLIC dependencies in target_link_libraries of OmniDSP.
endif()

# Generate the Version file (<Package>ConfigVersion.cmake)
include(CMakePackageConfigHelpers) # Ensure it's included before write_basic_package_version_file
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" # Output path in build dir
    VERSION ${PROJECT_VERSION}                                 # Project version from project() command
    COMPATIBILITY AnyNewerVersion                              # Compatibility mode (adjust as needed)
)
message(STATUS "      Generated ${PROJECT_NAME}ConfigVersion.cmake")

# Configure the main package file (<Package>Config.cmake) from the template
configure_package_config_file(
    "${CONFIG_CMAKE_IN_PATH}"                                # Input template (Config.cmake.in)
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"      # Output file in build dir
    INSTALL_DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"       # Final install destination for this file
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR  # Variables to make absolute
                                                             # Add CMAKE_INSTALL_BINDIR if needed by your config
)
message(STATUS "      Generated ${PROJECT_NAME}Config.cmake")

# Install the generated Config and Version files
install(
    FILES
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}" # Install to lib/cmake/OmniDSP
    COMPONENT Devel                            # Development component
)
message(STATUS "      Installation rules added for CMake package config files.")

# --- Install Library Target and Export Set ---
message(STATUS "    Configuring library target installation...")
# Install the library target (OmniDSP) itself
install(
    TARGETS OmniDSP                            # The target to install
    EXPORT OmniDSPTargets                      # The name of the export set to create/add to
    # Define destinations for different artifact types based on GNUInstallDirs
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib     # Static libs (.a)
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib     # Shared libs (.so/.dylib) or DLL import libs (.lib)
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Runtime # Executables or DLLs (.dll on Windows)
)
message(STATUS "      Installation rules added for the 'OmniDSP' library files.")

# Install the export set file (<TargetsName>.cmake)
install(
    EXPORT OmniDSPTargets                        # The export set defined in the install(TARGETS) command
    FILE ${PROJECT_NAME}Targets.cmake           # The filename for the targets file (e.g., OmniDSPTargets.cmake)
    NAMESPACE ${PROJECT_NAME}::                 # The namespace for imported targets (e.g., OmniDSP::OmniDSP)
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}"   # Install alongside Config.cmake
    COMPONENT Devel                              # Development component
)
message(STATUS "      Installation rules added for the CMake targets file (${PROJECT_NAME}Targets.cmake).")


# --- Python Examples Installation (Notebooks) ---
# Conditional on building Python bindings AND examples being enabled
if(OMNIDSP_BUILD_PYTHON_BINDINGS AND OMNIDSP_BUILD_EXAMPLES)
    message(STATUS "  Configuring Python examples installation (Python bindings ON, Examples ON)")

    if(EXISTS "${CMAKE_SOURCE_DIR}/examples/python") # Path relative to project root
        install(
            DIRECTORY "${CMAKE_SOURCE_DIR}/examples/python/" # Source directory
            DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/examples_python" # Use GNUInstallDirs standard path
            COMPONENT Examples                                  # Optional component name
            FILES_MATCHING
            PATTERN "*.ipynb"
        )
        message(STATUS "    Installation rule added for Python examples (notebooks).")
    else()
        message(STATUS "    Python examples directory 'examples/python' not found, skipping their installation.")
    endif()
else()
    if(OMNIDSP_BUILD_EXAMPLES) # Only message if examples were generally requested
        message(STATUS "  Skipping Python examples installation (Python bindings OFF or Examples OFF).")
    endif()
endif()

message(STATUS "Finished configuring installation rules.")
