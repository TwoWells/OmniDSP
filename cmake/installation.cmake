# cmake/installation.cmake
# ========================
# Defines installation rules for the OmniDSP project. This includes installing:
# - Public headers - The main library target (omnidsp) - CMake package
# configuration files (Config.cmake, Targets.cmake, Version.cmake) for
# find_package() support. - Python example notebooks (if Python bindings are
# enabled).
#
# Assumes standard CMake modules GNUInstallDirs and CMakePackageConfigHelpers
# have been included previously. Assumes the 'omnidsp' target exists.
#
# Variables Read: - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake) -
# PROJECT_NAME, PROJECT_VERSION (from project() command) - CMAKE_INSTALL_*,
# CMAKE_CURRENT_SOURCE_DIR, CMAKE_CURRENT_BINARY_DIR (CMake Built-in) -
# Config.cmake.in (Template file expected in source directory)
# ========================

message(STATUS "Configuring installation rules...")

# Ensure GNUInstallDirs has been included in the main CMakeLists.txt to have
# standard variables like CMAKE_INSTALL_LIBDIR, CMAKE_INSTALL_INCLUDEDIR, etc.
if(NOT CMAKE_INSTALL_LIBDIR)
  message(
    FATAL_ERROR
      "GNUInstallDirs must be included before installation.cmake to define CMAKE_INSTALL_* paths."
  )
endif()

# Check if the main library target exists
if(NOT TARGET omnidsp)
  message(
    FATAL_ERROR
      "The 'omnidsp' target must be defined before including installation.cmake."
  )
endif()

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  # --- Python Examples Installation ---
  message(
    STATUS "  Configuring Python examples installation (Python bindings ON)")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/notebooks")
    install(
      DIRECTORY examples/notebooks/
      DESTINATION
        "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/examples_python" # Use
                                                                   # GNUInstallDirs
                                                                   # standard
                                                                   # path
      COMPONENT Examples # Optional component name
      FILES_MATCHING
      PATTERN "*.ipynb")
    message(
      STATUS "    Installation rule added for Python examples (notebooks).")
  else()
    message(
      STATUS
        "    Python examples directory 'examples/notebooks' not found, skipping installation."
    )
  endif()

else()
  # --- C++ Library Installation (Headers, CMake Package, Library Files) ---
  message(
    STATUS "  Configuring C++ library installation rules (Python bindings OFF)")

  # Define the installation path for CMake package files
  set(OMNIDSP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  )# e.g., lib/cmake/OmniDSP
  message(
    STATUS "    CMake package install directory: ${OMNIDSP_CMAKE_INSTALL_DIR}")

  # --- Install Public Headers ---
  # Install the public headers from the include/OmniDSP directory
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/OmniDSP/" # Source directory
                                                             # (note trailing
                                                             # slash)
    DESTINATION
      "${CMAKE_INSTALL_INCLUDEDIR}/OmniDSP" # Destination directory (relative to
                                            # prefix/include)
    COMPONENT Devel # Component for development files
    FILES_MATCHING # Only install files matching the pattern
    PATTERN "*.h") # Pattern for header files
  # Install the generated export header Assumes generate_export_header was
  # called with EXPORT_FILE_NAME OmniDSP/omnidsp_export.h
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/OmniDSP/omnidsp_export.h" # Source file
                                                                 # in build
                                                                 # directory
    DESTINATION
      "${CMAKE_INSTALL_INCLUDEDIR}/OmniDSP" # Place it inside the installed
                                            # OmniDSP header dir
    COMPONENT Devel)
  message(STATUS "    Installation rules added for public headers.")

  # --- Generate and Install CMake Package Configuration Files ---
  message(STATUS "    Configuring CMake package files...")
  # Ensure the template file (Config.cmake.in) exists in the source tree
  set(CONFIG_CMAKE_IN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in")
  if(NOT EXISTS "${CONFIG_CMAKE_IN_PATH}")
    message(
      FATAL_ERROR
        "Config.cmake.in not found at '${CONFIG_CMAKE_IN_PATH}'. This file is required for C++ installation."
    )
  else()
    message(
      STATUS "      Using Config.cmake.in template: ${CONFIG_CMAKE_IN_PATH}")
    # Reminder: Developers must ensure Config.cmake.in includes necessary
    # find_dependency calls: find_dependency(Boost REQUIRED)
    # find_dependency(highway CONFIG REQUIRED) These should match the PUBLIC
    # dependencies in target_link_libraries.
  endif()

  # Generate the Version file (<Package>ConfigVersion.cmake) This helps
  # find_package determine version compatibility.
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" # Output
                                                                     # path in
                                                                     # build dir
    VERSION ${PROJECT_VERSION} # Project version from project() command
    COMPATIBILITY AnyNewerVersion # Compatibility mode (adjust as needed)
  )
  message(STATUS "      Generated ${PROJECT_NAME}ConfigVersion.cmake")

  # Configure the main package file (<Package>Config.cmake) from the template
  # This substitutes variables like @PACKAGE_INIT@ and configured PATH_VARS.
  configure_package_config_file(
    "${CONFIG_CMAKE_IN_PATH}" # Input template (Config.cmake.in)
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" # Output file in
                                                              # build dir
    INSTALL_DESTINATION
      "${OMNIDSP_CMAKE_INSTALL_DIR}" # Final install destination for this file
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
              CMAKE_INSTALL_LIBDIR # Variables to make absolute in installed
                                   # config file Add other paths if needed
                                   # (e.g., CMAKE_INSTALL_BINDIR)
  )
  message(STATUS "      Generated ${PROJECT_NAME}Config.cmake")

  # Install the generated Config and Version files
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}" # Install to lib/cmake/OmniDSP
    COMPONENT Devel # Development component
  )
  message(
    STATUS "      Installation rules added for CMake package config files.")

  # --- Install Library Target and Export Set ---
  message(STATUS "    Configuring library target installation...")
  # Install the library target (omnidsp) itself This installs the actual library
  # files (.a, .lib, .so, .dylib, .dll) It also defines an "export set" named
  # OmniDSPTargets for the targets file.
  install(
    TARGETS omnidsp # The target to install
    EXPORT OmniDSPTargets # The name of the export set to create/add to
    # Define destinations for different artifact types based on GNUInstallDirs
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Lib # Static libs
                                                                # (.a)
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT Lib # Shared libs (.so/.dylib) or DLL import libs (.lib)
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT Runtime # Executables or DLLs (.dll)
  )
  message(
    STATUS "      Installation rules added for the 'omnidsp' library files.")

  # Install the export set file (<TargetsName>.cmake) This file contains
  # information about the installed targets (location, properties) allowing
  # other CMake projects to use them via find_package.
  install(
    EXPORT OmniDSPTargets # The export set defined in the install(TARGETS)
                          # command
    FILE ${PROJECT_NAME}Targets.cmake # The filename for the targets file (e.g.,
                                      # OmniDSPTargets.cmake)
    NAMESPACE ${PROJECT_NAME}:: # The namespace for imported targets (e.g.,
                                # OmniDSP::omnidsp)
    DESTINATION "${OMNIDSP_CMAKE_INSTALL_DIR}" # Install alongside Config.cmake
    COMPONENT Devel # Development component
  )
  message(
    STATUS
      "      Installation rules added for the CMake targets file (${PROJECT_NAME}Targets.cmake)."
  )

endif() # End C++/Python installation conditional

message(STATUS "Finished configuring installation rules.")
