# examples/cpp/CMakeLists.txt
# CMakeLists.txt for OmniDSP C++ Examples.
# Defines executables for each example .cpp file.
# This script is processed if OMNIDSP_BUILD_EXAMPLES is ON,
# and the parent examples/CMakeLists.txt decides to include C++ examples
# (i.e., OMNIDSP_BUILD_PYTHON_BINDINGS is OFF).

# OMNIDSP_BUILD_EXAMPLES is assumed to be ON if this script is reached.
# OMNIDSP_BUILD_PYTHON_BINDINGS is assumed to be OFF if this script is reached.

message(STATUS "  C++ Examples: Configuring C++ examples (OMNIDSP_BUILD_EXAMPLES=ON, OMNIDSP_BUILD_PYTHON_BINDINGS=OFF).")

# Find the main library target. Assuming the target name is 'OmniDSP'.
if(NOT TARGET OmniDSP)
    message(FATAL_ERROR "OmniDSP library target (OmniDSP) not found. Ensure the main library is configured before examples.")
endif()

# List of example source files
set(CPP_EXAMPLE_SOURCES
    fft.cpp
    rfft.cpp
    cqt.cpp
    window.cpp
    # Add other example .cpp files here, e.g., filter_example.cpp, resample_example.cpp
)

# Create an executable for each example source file
foreach(EXAMPLE_SOURCE_FILE ${CPP_EXAMPLE_SOURCES})
    # Get the base name without extension to use as target name
    get_filename_component(EXAMPLE_TARGET_NAME ${EXAMPLE_SOURCE_FILE} NAME_WE)

    # Add the executable target
    add_executable(${EXAMPLE_TARGET_NAME}_example ${EXAMPLE_SOURCE_FILE}) # Added _example suffix for clarity

    # Link the example against the OmniDSP library
    target_link_libraries(${EXAMPLE_TARGET_NAME}_example PRIVATE OmniDSP)

    # Installation (optional, install examples to bin dir)
    # CMAKE_INSTALL_BINDIR is from GNUInstallDirs (included in root)
    install(TARGETS ${EXAMPLE_TARGET_NAME}_example
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/omnidsp_examples" # Group examples in a subfolder
    )

    message(STATUS "    Configured C++ example: ${EXAMPLE_TARGET_NAME}_example")
endforeach()

message(STATUS "  Finished configuring C++ examples.")
