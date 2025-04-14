# Contributing to OmniDSP

Thank you for your interest in contributing to OmniDSP! We welcome contributions from everyone. This document provides guidelines for contributing to the project.

## Table of Contents

- [Contributing to OmniDSP](#contributing-to-omnidsp)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Setting up the Conda Environment](#setting-up-the-conda-environment)
  - [Building for Development](#building-for-development)
    - [Python Package (Recommended)](#python-package-recommended)
    - [C++ Library Only](#c-library-only)
  - [Project Structure](#project-structure)
  - [Understanding Backends](#understanding-backends)
  - [Running Tests](#running-tests)
    - [C++ Tests (GoogleTest)](#c-tests-googletest)
    - [Python Tests (pytest)](#python-tests-pytest)
    - [Test Reference Data](#test-reference-data)
  - [Coding Style](#coding-style)
  - [Submitting Contributions](#submitting-contributions)
    - [Pull Request Workflow](#pull-request-workflow)
  - [Reporting Bugs and Suggesting Features](#reporting-bugs-and-suggesting-features)
  - [Where to Contribute](#where-to-contribute)

## Getting Started

### Prerequisites

* **Conda:** Used for managing the development environment and dependencies. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
* **Git:** For version control.
* **(Optional) Apple Developer Tools:** Required for the Accelerate backend on macOS (Xcode or Command Line Tools).

### Setting up the Conda Environment

The project uses a Conda environment to manage all dependencies, including the C++ compiler, CMake, Python, MKL, IPP, and Python packages.

1.  **Create Environment:** From the project root directory, create the environment using the provided file:
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate Environment:** Activate the newly created environment (the name is defined inside `environment.yml`, likely `omnidsp_env`):
    ```bash
    conda activate omnidsp_env
    ```
    **Important:** Always ensure this environment is activated before building, running tests, or using the library.

## Building for Development

### Python Package (Recommended)

This is the standard way to build if you intend to use or test the Python bindings.

1.  **Activate Conda Environment:** (See above)
2.  **Editable Install:** From the project root directory (containing `pyproject.toml`), run:
    ```bash
    pip install -e . -v
    ```
    * The `-e` flag installs the package in "editable" mode, meaning changes to the Python source code (`src/omnidsp_py`) are reflected immediately without reinstalling. Changes to C++ code still require rebuilding (which `pip install -e .` might trigger automatically, or you can run it again).
    * The `-v` flag provides verbose output, helpful for debugging build issues.
    * This uses `scikit-build-core` and CMake to compile the C++ library and Python bindings, installing them into your active Conda environment.

### C++ Library Only

If you only need the C++ library for use in another C++ project:

1.  **Activate Conda Environment:** (See above)
2.  **Configure with CMake:**
    ```bash
    mkdir build && cd build
    # Configure without Python bindings (default)
    cmake .. -DCMAKE_INSTALL_PREFIX=../install # Or another install location
    ```
3.  **Build:**
    ```bash
    cmake --build . --config Release --parallel
    ```
4.  **(Optional) Install:**
    ```bash
    cmake --install . --config Release
    ```

## Project Structure

* `include/OmniDSP/`: Public C++ headers.
* `src/omnidsp/`: Core C++ library implementation (platform-independent parts).
* `src/omnidsp/backend/`: Backend-specific C++ implementations (MKL, Accelerate, Stub).
* `src/omnidsp_py/`: Python bindings source code (pybind11, wrappers).
* `tests/cpp/`: C++ unit tests (GoogleTest).
* `tests/python/`: Python unit tests (pytest).
* `examples/`: Usage examples (C++ and Python notebooks).
* `environment.yml`: Conda environment definition.
* `CMakeLists.txt`: Main CMake build script.
* `pyproject.toml`: Python packaging configuration.

## Understanding Backends

OmniDSP uses different backends for performance:
* **oneMKL:** Uses Intel MKL and IPP. Preferred on non-Apple platforms if found by Conda. Requires `USE_ONEMKL` preprocessor definition (set by CMake).
* **Accelerate:** Uses Apple's Accelerate framework. Preferred on macOS unless MKL is explicitly preferred. Requires `USE_ACCELERATE` definition.
* **Stub:** A fallback implementation that throws runtime errors if no optimized backend is found/selected.

CMake automatically detects and selects the backend based on the platform and libraries available within the **active Conda environment** during configuration.

## Running Tests

### C++ Tests (GoogleTest)

1.  **Build:** Ensure the project is built (either via `pip install -e .` or CMake directly). The tests are built as part of the default CMake build when `OMNIDSP_BUILD_PYTHON_BINDINGS` is OFF, or potentially alongside the Python build depending on configuration.
2.  **Run:** Execute tests using CTest from the `build` directory:
    ```bash
    cd build
    ctest -C Release --verbose
    # Or run the executable directly if built:
    # ./tests/cpp/Release/omnidsp_tests.exe (Windows)
    # ./tests/cpp/omnidsp_tests (Linux/macOS)
    ```
    * Note: The C++ tests require the `test_references.txt` file, which CMake copies to the build directory.

### Python Tests (pytest)

1.  **Build/Install:** Ensure the Python package is installed (e.g., `pip install -e .`).
2.  **Run:** Execute pytest from the project root directory:
    ```bash
    pytest tests/python
    ```

### Test Reference Data

Some C++ tests rely on reference data generated by `tests/cpp/generate_references.py` (using NumPy, SciPy, and Librosa) and stored in `tests/cpp/test_references.txt`. If you modify inputs or expected outputs for these tests, re-run the Python script to update the reference file.

## Coding Style

* **C++:** Please try to follow the existing code style. Consider using `clang-format` with a standard style (e.g., Google, LLVM). Use Doxygen-style comments for documenting headers (`/** ... */`) and implementation details where appropriate.
* **Python:** Please use `black` for code formatting and `flake8` for linting. Follow PEP 8 guidelines. Use Google-style docstrings.
* **CMake:** Follow standard CMake practices for readability.

## Submitting Contributions

We appreciate contributions! Please follow this general workflow:

1.  **Fork the Repository:** Create your own fork of the OmniDSP repository on GitHub.
2.  **Create a Branch:** Create a new branch in your fork for your feature or bug fix (e.g., `git checkout -b feature/add-convolution-modes`).
3.  **Make Changes:** Implement your feature or fix the bug.
4.  **Add Tests:** Add appropriate unit tests (C++ and/or Python) to cover your changes. Ensure all tests pass.
5.  **Update Documentation:** If you add new features or change existing ones, update the README, docstrings, and potentially other documentation files.
6.  **Commit Changes:** Commit your changes with clear and concise commit messages.
7.  **Push to Your Fork:** Push your branch to your GitHub fork.
8.  **Open a Pull Request:** Create a Pull Request (PR) from your branch to the main OmniDSP repository's `main` (or appropriate development) branch. Provide a clear description of your changes in the PR.

### Pull Request Workflow

* Ensure your code builds successfully on relevant platforms (if possible).
* Ensure all tests (C++ and Python) pass.
* Address any feedback from reviewers.
* Once approved, your PR will be merged.

## Reporting Bugs and Suggesting Features

Please use the GitHub Issues tracker for the OmniDSP repository to:
* Report bugs (include steps to reproduce, environment details, expected vs. actual results).
* Suggest new features or enhancements.

## Where to Contribute

Check the `TODO.md` file for a list of known tasks and desired features. Some key areas include:
* Tuning the CQT scaling factor.
* Implementing `double` precision support for resampling in the MKL backend.
* Adding 'same'/'full' convolution modes.
* Setting up CI.
* Adding more tests and examples.
* Improving documentation.

Thank you for contributing!
