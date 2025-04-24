# CMakeLists.txt for OmniDSP C++ Examples Defines executables for each example .cpp file.

# Find the main library target (defined in parent scope) Use the plain target name 'omnidsp' within
# the same build tree
if(NOT TARGET omnidsp) # Changed from OmniDSP::omnidsp
  message(FATAL_ERROR "OmniDSP library target (omnidsp) not found. Build the main library first.")
endif()

# List of example source files
set(EXAMPLE_SOURCES fft.cpp rfft.cpp cqt.cpp window.cpp # Renamed from windows.cpp
                    # Add other example cpp files here
)

# Create an executable for each example source file
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  # Get the base name without extension to use as target name
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

  # Add the executable target
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

  # Link the example against the OmniDSP library Use the plain target name 'omnidsp' within the same
  # build tree
  target_link_libraries(${EXAMPLE_NAME} PRIVATE omnidsp) # Changed from OmniDSP::omnidsp

  # Installation (optional, install examples to bin dir)
  install(TARGETS ${EXAMPLE_NAME} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/examples")

  message(STATUS "Configured C++ example: ${EXAMPLE_NAME}")
endforeach()
