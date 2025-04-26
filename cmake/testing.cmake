# cmake/testing.cmake
# ===================
# Enables CTest support for the project and includes test subdirectories (tests/python or tests/cpp)
# based on the OMNIDSP_BUILD_PYTHON_BINDINGS option.
#
# Variables Read:
#   - OMNIDSP_BUILD_PYTHON_BINDINGS (from project_options.cmake)
#   - CMAKE_CURRENT_SOURCE_DIR, CMAKE_CURRENT_BINARY_DIR (CMake Built-in)
# ===================

message(STATUS "Configuring testing...")

# Enable CTest support for the project
enable_testing()
message(STATUS "  CTest support enabled.")

# Conditionally add test subdirectories based on the build type Assumes
# OMNIDSP_BUILD_PYTHON_BINDINGS option is already defined.

if(OMNIDSP_BUILD_PYTHON_BINDINGS)
  # If building Python bindings, include Python tests
  message(STATUS "  Checking for Python tests directory...")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/CMakeLists.txt")
    message(STATUS "    Found tests/python/CMakeLists.txt, adding subdirectory.")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/python
                     ${CMAKE_CURRENT_BINARY_DIR}/tests/python)
  else()
    message(
      STATUS
        # Changed from WARNING to STATUS as it might be intentional
        "    Python tests subdirectory 'tests/python' or its CMakeLists.txt not found, skipping Python test configuration."
    )
  endif()
else()
  # If not building Python bindings, include C++ tests
  message(STATUS "  Checking for C++ tests directory...")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
    message(STATUS "    Found tests/cpp/CMakeLists.txt, adding subdirectory.")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp ${CMAKE_CURRENT_BINARY_DIR}/tests/cpp)
    # Note: The C++ tests CMakeLists.txt will need to find GTest itself using find_package(GTest
    # REQUIRED) and link targets appropriately. The find_package(GTest) call was moved to
    # dependencies.cmake, but the targets (GTest::gtest, GTest::gtest_main) need linking in
    # tests/cpp.
  else()
    message(
      STATUS
        # Changed from WARNING to STATUS
        "    C++ tests subdirectory 'tests/cpp' or its CMakeLists.txt not found, skipping C++ test configuration."
    )
  endif()
endif()

message(STATUS "Finished configuring testing.")
