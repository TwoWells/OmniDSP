# examples/cpp/CMakeLists.txt

# Find all .cpp source files in this directory (examples/cpp/)
file(GLOB CPP_EXAMPLE_SOURCES CONFIGURE_DEPENDS "*.cpp")

if(NOT CPP_EXAMPLE_SOURCES)
    message(STATUS "No C++ example source files (.cpp) found in examples/cpp/ directory.")
    return() # Exit early if no sources
endif()

message(STATUS "Found C++ example sources: ${CPP_EXAMPLE_SOURCES}")

# Loop through each found source file
foreach(EXAMPLE_SOURCE ${CPP_EXAMPLE_SOURCES})
    # Get the filename without the path and extension to use as the target name
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

    # Add an executable target for this example
    # Example: Creates targets like 'cqt', 'fft', 'rfft', 'windows'
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

    # Link the example executable against the omnidsp library target
    # 'omnidsp' target is defined in the parent scope (root CMakeLists.txt)
    # PRIVATE linking is appropriate as the examples use omnidsp but don't export its interface.
    target_link_libraries(${EXAMPLE_NAME} PRIVATE omnidsp)

    message(STATUS "Added C++ example target: ${EXAMPLE_NAME}")

    # Optional: Install the C++ example executables
    # install(TARGETS ${EXAMPLE_NAME}
    #         RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/examples" # e.g., <prefix>/bin/examples
    #         COMPONENT Examples)

endforeach()