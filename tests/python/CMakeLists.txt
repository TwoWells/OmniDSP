# tests/python/CMakeLists.txt
# CMakeLists.txt for OmniDSP Python Tests using pytest.
# This script is processed if OMNIDSP_BUILD_TESTS is ON,
# and the parent tests/CMakeLists.txt decides to include Python tests
# (which means OMNIDSP_BUILD_PYTHON_BINDINGS is also ON).

# OMNIDSP_BUILD_TESTS and OMNIDSP_BUILD_PYTHON_BINDINGS are assumed to be ON
# if this script is reached, due to checks in parent CMakeLists.txt files.

message(STATUS "  Python Tests: Configuring Python tests (OMNIDSP_BUILD_TESTS=ON, OMNIDSP_BUILD_PYTHON_BINDINGS=ON).")

# --- Find Dependencies ---
# Find Python interpreter
find_package(Python COMPONENTS Interpreter REQUIRED)
message(STATUS "    Found Python Interpreter: ${Python_EXECUTABLE}")

# Find pytest executable
find_program(PYTEST_EXECUTABLE
    NAMES pytest pytest.exe
    DOC "Path to pytest executable"
)
if(NOT PYTEST_EXECUTABLE)
    message(FATAL_ERROR "pytest not found. Please install pytest (e.g., pip install pytest) or ensure it's in your PATH.")
else()
    message(STATUS "    Found pytest: ${PYTEST_EXECUTABLE}")
endif()

# --- Discover Test Files ---
# Assuming Python test files are in the current directory (tests/python)
# and follow a naming convention like test_*.py or *_test.py.
file(GLOB PYTHON_TEST_FILES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/*_test.py"
)

if(NOT PYTHON_TEST_FILES)
    message(WARNING "  Python Tests: No Python test files (test_*.py or *_test.py) found in ${CMAKE_CURRENT_SOURCE_DIR}.")
    return() # Still useful to return if no test files are found
endif()

message(STATUS "    Found Python test files: ${PYTHON_TEST_FILES}")

# --- Determine Path to the Built Python Module (omnidsp_py) ---
# The omnidsp_py module is built by src/omnidsp_py/CMakeLists.txt.
# Its output directory depends on the generator and configuration.
# We need to add the directory containing the 'omnidsp_py' package to PYTHONPATH.

set(OMNIDSP_PY_MODULE_BUILD_DIR "${PROJECT_BINARY_DIR}/src/omnidsp_py")
# This path assumes that the omnidsp_py module (.pyd/.so and __init__.py if it's a package)
# is directly in the build directory of src/omnidsp_py.
# If pybind11_add_module creates an additional subdirectory for the package,
# this path might need to be adjusted (e.g., to just PROJECT_BINARY_DIR/src if that's what needs to be in PYTHONPATH).

if(NOT EXISTS "${OMNIDSP_PY_MODULE_BUILD_DIR}")
    # Fallback or more robust way to find where omnidsp_py target outputs its library
    # This is tricky without knowing the exact output of pybind11_add_module.
    # For now, we'll stick to the assumption and warn if it doesn't exist.
    message(WARNING "  Python Tests: Assumed omnidsp_py module build directory ${OMNIDSP_PY_MODULE_BUILD_DIR} does not exist. PYTHONPATH might be incorrect.")
else()
    message(STATUS "    Using omnidsp_py module build/output directory for PYTHONPATH: ${OMNIDSP_PY_MODULE_BUILD_DIR}")
endif()


# --- Add Tests to CTest ---
foreach(PYTHON_TEST_FILE ${PYTHON_TEST_FILES})
    get_filename_component(TEST_NAME ${PYTHON_TEST_FILE} NAME_WE)

    add_test(
        NAME Python_${TEST_NAME}
        COMMAND ${Python_EXECUTABLE} -m pytest -v "${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_TEST_FILE}" # Quoted path
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" # Run tests from where the test files are
    )

    # Set environment for the test: PYTHONPATH
    # This ensures that 'import omnidsp_py' works.
    # We prepend the path to our built module.
    # If PYTHONPATH is already set, CMake prepends with the platform's path separator.
    set_tests_properties(Python_${TEST_NAME}
        PROPERTIES ENVIRONMENT "PYTHONPATH=${OMNIDSP_PY_MODULE_BUILD_DIR}"
    )
    message(STATUS "    Added Python test: Python_${TEST_NAME}")
endforeach()

message(STATUS "  Finished configuring Python tests.")
