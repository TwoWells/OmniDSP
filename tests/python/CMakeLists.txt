# tests/python/CMakeLists.txt
# Integrates pytest tests with CTest

# --- NO find_package(Python ...) here ---
# It should be found implicitly by pybind11 in the root CMakeLists, guided by VS Code's selected interpreter.

# Ensure the Python module target exists before adding the test
# Replace 'omnidsp_py' with your actual pybind11_add_module target name
if(TARGET omnidsp_py) #<-- Make sure this target name is correct!

    # Check if the variable holding the interpreter path is set
    if(NOT DEFINED PYTHON_EXECUTABLE)
         message(FATAL_ERROR "PYTHON_EXECUTABLE not set by CMake's FindPython. Cannot configure python tests.")
    endif()

    # Add test executable for pytest
    add_test(
        NAME python_tests # Name of the test in CTest
        # Set PYTHONPATH to find the built module in the build tree
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:omnidsp_py>" #<-- Target name check
                # Use the variable found by CMake (ALL CAPS)
                "${PYTHON_EXECUTABLE}"
                -m pytest "${CMAKE_CURRENT_SOURCE_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    message(STATUS "Added CTest entry 'python_tests' to run pytest using ${PYTHON_EXECUTABLE}.")
else()
    message(WARNING "CMake target 'omnidsp_py' (Python module) not found. Cannot add python_tests via CTest.") #<-- Target name check
endif()