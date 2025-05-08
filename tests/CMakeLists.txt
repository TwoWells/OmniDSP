# tests/CMakeLists.txt
# This file is included if OMNIDSP_BUILD_TESTS is ON.
# It adds subdirectories for specific types of tests (C++ or Python)
# based on whether Python bindings are being built.

# The OMNIDSP_BUILD_TESTS option has already been checked by the parent (root) CMakeLists.txt,
# and enable_testing() has been called.
# OMNIDSP_BUILD_PYTHON_BINDINGS should be defined by project_options.cmake and inherited.

message(STATUS "  Processing tests/CMakeLists.txt...")

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
    # If Python bindings are enabled, prioritize Python tests.
    message(STATUS "    Python bindings enabled. Attempting to add Python tests subdirectory (python)...")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt")
        # Source directory is "python", binary directory will be <parent_binary_dir>/python
        add_subdirectory(python ${CMAKE_CURRENT_BINARY_DIR}/python)
    else()
        message(STATUS "      Directory tests/python or its CMakeLists.txt not found. Skipping Python tests configuration.")
    endif()
    message(STATUS "    (C++ tests will be skipped as Python bindings are enabled).")
else()
    # If Python bindings are NOT enabled, attempt to add C++ tests.
    message(STATUS "    Python bindings not enabled. Attempting to add C++ tests subdirectory (cpp)...")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")
        # Source directory is "cpp", binary directory will be <parent_binary_dir>/cpp
        add_subdirectory(cpp ${CMAKE_CURRENT_BINARY_DIR}/cpp)
    else()
        message(STATUS "      Directory tests/cpp or its CMakeLists.txt not found. Skipping C++ tests configuration.")
    endif()
endif()

message(STATUS "  Finished processing tests/CMakeLists.txt.")
