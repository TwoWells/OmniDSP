# examples/CMakeLists.txt
# This file is included if the examples directory exists and potentially if OMNIDSP_BUILD_EXAMPLES is ON.
# It adds subdirectories for specific types of examples (C++ or Python)
# based on build options.

# Options OMNIDSP_BUILD_EXAMPLES and OMNIDSP_BUILD_PYTHON_BINDINGS
# should be defined by project_options.cmake and inherited.

# Primary Guard: Only proceed if examples are globally enabled.
if(NOT OMNIDSP_BUILD_EXAMPLES)
    message(STATUS "  Examples: Build examples disabled (OMNIDSP_BUILD_EXAMPLES=OFF). Skipping examples configuration.")
    return()
endif()

message(STATUS "  Processing examples/CMakeLists.txt (OMNIDSP_BUILD_EXAMPLES=ON)...")

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
    # If Python bindings are enabled, handle Python examples (if any CMake configuration is needed).
    # Currently, assuming Python examples (notebooks) don't need CMake configuration here.
    message(STATUS "    Python bindings enabled. Checking for Python examples subdirectory (python)...")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt")
        # If you add CMake logic for Python examples later (e.g., installing data), add it here.
        add_subdirectory(python ${CMAKE_CURRENT_BINARY_DIR}/python)
    else()
        message(STATUS "      Directory examples/python or its CMakeLists.txt not found.")
    endif()
    message(STATUS "    (C++ examples will be skipped as Python bindings are enabled).")
    message(STATUS "    (Python example notebooks are typically handled by installation rules or run directly).")

else()
    # If Python bindings are NOT enabled, attempt to add C++ examples.
    message(STATUS "    Python bindings not enabled. Attempting to add C++ examples subdirectory (cpp)...")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")
        # Source directory is "cpp", binary directory will be <parent_binary_dir>/cpp
        add_subdirectory(cpp ${CMAKE_CURRENT_BINARY_DIR}/cpp)
    else()
        message(STATUS "      Directory examples/cpp or its CMakeLists.txt not found. Skipping C++ examples configuration.")
    endif()
endif()

message(STATUS "  Finished processing examples/CMakeLists.txt.")
