# tests/python/CMakeLists.txt
# Integrates pytest tests with CTest

# Find Python components required by pybind11 targets (if not already found)
# This ensures ${Python_EXECUTABLE} is set.
# Use the same components as pybind11 likely uses internally.
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find pytest using the located Python interpreter
message(STATUS "Checking for pytest using interpreter: ${Python_EXECUTABLE}")
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pip show pytest
    RESULT_VARIABLE PYTEST_FOUND_RESULT
    OUTPUT_QUIET # Don't print pip output during configure
    ERROR_QUIET # Don't print pip error during configure
)

if(NOT PYTEST_FOUND_RESULT EQUAL 0)
    message(FATAL_ERROR "pytest is required to run python tests. Please install it for interpreter ${Python_EXECUTABLE} (e.g., 'pip install pytest' or 'pip install .[test]')")
else()
    message(STATUS "Found pytest.")
endif()

# Ensure the Python module target exists before referencing its directory
# Replace 'omnidsp' with your actual pybind11_add_module target name if different
if(TARGET omnidsp)
    # Add test executable for pytest
    # Runs pytest on the test files located in this source directory
    add_test(
        NAME python_tests # Name of the test in CTest
        # Set PYTHONPATH to find the built module in the build tree
        # Use CMAKE_COMMAND -E env for portability
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:omnidsp>"
                ${Python_EXECUTABLE} -m pytest "${CMAKE_CURRENT_SOURCE_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} # Optional: Set working dir explicitly if tests need files relative to source
    )
    message(STATUS "Added CTest entry 'python_tests' to run pytest.")
else()
    message(WARNING "CMake target 'omnidsp' (Python module) not found. Cannot add python_tests via CTest.")
endif()