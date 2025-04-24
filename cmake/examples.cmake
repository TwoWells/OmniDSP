# cmake/examples.cmake
# ====================
# Includes example subdirectories based on the build configuration. Currently,
# it only adds the C++ examples subdirectory (examples/cpp) if Python bindings
# are *not* being built. Python examples (notebooks) are handled by installation
# rules.
#
# Variables Read: - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake) -
# CMAKE_CURRENT_SOURCE_DIR, CMAKE_CURRENT_BINARY_DIR (CMake Built-in)
# ====================

message(STATUS "Configuring examples...")

# Conditionally add example subdirectories based on the build type Assumes
# OMNIDSP_BUILD_PYTHON_BINDINGS option is already defined.

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  # Python examples (notebooks) are handled during installation
  # (installation.cmake) No subdirectory needs to be added here for them.
  message(
    STATUS "  Python examples (notebooks) are handled by installation rules.")
else()
  # If not building Python bindings, include C++ examples if they exist
  message(STATUS "  Checking for C++ examples directory...")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/CMakeLists.txt")
    message(
      STATUS "    Found examples/cpp/CMakeLists.txt, adding subdirectory.")
    # Add the subdirectory, specifying source and binary directories
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp
                     ${CMAKE_CURRENT_BINARY_DIR}/examples/cpp)
    # Note: The examples/cpp/CMakeLists.txt will need to link its executables
    # against the main omnidsp library target (e.g.,
    # target_link_libraries(my_example PRIVATE OmniDSP::omnidsp))
  else()
    message(
      STATUS
        # Changed from WARNING to STATUS as it might be intentional
        "    C++ examples subdirectory 'examples/cpp' or its CMakeLists.txt not found, skipping C++ example configuration."
    )
  endif()
endif()

message(STATUS "Finished configuring examples.")
