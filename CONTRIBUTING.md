# Contributing to OmniDSP

Thank you for your interest in contributing to OmniDSP! We welcome contributions from everyone. This document provides guidelines for contributing to the project.

## Table of Contents

- [Contributing to OmniDSP](#contributing-to-omnidsp)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Setting up the Development Environment](#setting-up-the-development-environment)
    - [Dependency Rationale: Boost](#dependency-rationale-boost)
  - [Dependency Management with Conda Lock](#dependency-management-with-conda-lock)
    - [Contributor Workflow for Runtime Dependencies](#contributor-workflow-for-runtime-dependencies)
  - [Developer Tools and Workflow](#developer-tools-and-workflow)
    - [Using `pre-commit` for Code Quality](#using-pre-commit-for-code-quality)
    - [Updating Development Tools](#updating-development-tools)
  - [Building for Development](#building-for-development)
    - [Python Package (Recommended)](#python-package-recommended)
    - [C++ Library Only](#c-library-only)
  - [Project Structure](#project-structure)
  - [Understanding the Architecture](#understanding-the-architecture)
    - [Core `OmniDSP` Class and Factories](#core-omnidsp-class-and-factories)
    - [`Plan::[Type]` Objects (Stateless)](#plantype-objects-stateless)
    - [`Processor::[Type]` Objects (Stateful)](#processortype-objects-stateful)
    - [Configuration: `Params`, `Window::[SpecType]`, `Coefs`, `Design`](#configuration-params-windowspectype-coefs-design)
    - [Backend Abstraction (PIMPL)](#backend-abstraction-pimpl)
    - [Internal Namespace and PIMPL Naming](#internal-namespace-and-pimpl-naming)
    - [Design Rationale](#design-rationale)
  - [Running Tests](#running-tests)
    - [Python Tests](#python-tests)
    - [C++ Tests](#c-tests)
      - [Selecting Backend for Tests](#selecting-backend-for-tests)
      - [C++ Testing Framework Overview](#c-testing-framework-overview)
        - [Directory Structure](#directory-structure)
        - [Reference Data Files](#reference-data-files)
        - [Generating/Regenerating Reference Data](#generatingregenerating-reference-data)
        - [Running C++ Tests](#running-c-tests)
  - [Coding Style](#coding-style)
  - [Commit Message Format (Conventional Commits)](#commit-message-format-conventional-commits)
  - [Submitting Contributions](#submitting-contributions)
    - [Pull Request Workflow](#pull-request-workflow)
  - [Reporting Bugs and Suggesting Features](#reporting-bugs-and-suggesting-features)
  - [Where to Contribute](#where-to-contribute)

## Getting Started

### Prerequisites

- **Conda:** Used for managing the development environment and dependencies. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
- **Git:** For version control.
- **(Optional) Apple Developer Tools:** Required for the Accelerate backend on macOS (Xcode or Command Line Tools).

### Setting up the Development Environment

The project uses Conda to manage dependencies. For contributors, we provide a specific development environment definition (`environment-dev.yml`) that includes not only the runtime dependencies needed to run OmniDSP but also essential tools for development, testing, and managing dependencies. Code formatting/linting tools are primarily managed via `pre-commit`.

1.  **Create Development Environment:** From the project root directory, create the development environment using the `environment-dev.yml` file:

    ```
    conda env create -f environment-dev.yml
    ```

    This command creates a Conda environment named `omnidsp-dev` (defined within the file).

2.  **Activate Environment:** Activate the newly created environment:

    ```
    conda activate omnidsp-dev
    ```

    **Important:** Always ensure this environment is activated before building, running tests, using the library, or running developer tools like `pre-commit` or `conda-lock`.

3.  **Install Git Hooks:** After activating the environment for the first time in a new clone, install the `pre-commit` Git hooks:
    ```
    pre-commit install --hook-type commit-msg --hook-type pre-commit
    ```
    This sets up automatic code quality checks (formatting, linting) and commit message validation that run before each commit. See the "Developer Tools and Workflow" section for more details.

### Dependency Rationale: Boost

The Boost C++ libraries were added as a dependency primarily for `Boost.Math`. Specifically, the library requires the 0th order modified Bessel function of the first kind (`I₀`) to generate Kaiser windows coefficients accurately and consistently (used by `OmniDSP::Internal::WindowFormulas` for the `Default::Backend`'s `Plan::Window` for Kaiser windows). While C++17 includes mathematical special functions like `std::cyl_bessel_i`, its implementation is missing in certain standard library versions (notably libc++ on macOS). Using `boost::math::cyl_bessel_i` ensures a reliable implementation across all supported platforms. Boost is automatically installed via the Conda environment file.

## Dependency Management with Conda Lock

(This section remains largely unchanged but is important for context)

To ensure consistent and reproducible _runtime_ environments across different operating systems and in our Continuous Integration (CI) workflows, this project uses `conda-lock` with **explicit, platform-specific lock files**.

- **Source File:** The `environment.yml` file defines the primary _runtime_ dependencies (for the `omnidsp` environment), their constraints, and the target platforms for locking.
- **Lock Files:** Explicit, platform-specific lock files (e.g., `conda-osx-arm64.lock`, `conda-linux-64.lock`, `conda-win-64.lock`) are generated from `environment.yml`. These lock files contain the exact package URLs and hashes needed for _each specific platform_ and are committed to the repository. End-users and CI should ideally use these lock files to create the `omnidsp` environment.
- **Development Environment:** The `environment-dev.yml` file includes all runtime dependencies _plus_ developer tools like `pytest`, `ruff`, `pre-commit`, and `conda-lock`. It is typically used directly for creating the separate `omnidsp-dev` developer environment (`conda env create -f environment-dev.yml`).

### Contributor Workflow for Runtime Dependencies

**If you are NOT changing runtime dependencies in `environment.yml`:**
You generally don't need to interact directly with `conda-lock`. Simply use the `omnidsp-dev` environment created from `environment-dev.yml`.

**If you ARE changing runtime dependencies in `environment.yml`:**

1.  **Ensure `conda-lock` is Available:** The `conda-lock` tool is included in the `omnidsp-dev` environment. Ensure this environment is active.
2.  **Modify `environment.yml`:** Add, remove, or update _runtime_ packages. Also, make the same changes to `environment-dev.yml`.
3.  **Regenerate Lock Files:** Run `conda-lock lock -f environment.yml --kind explicit`.
4.  **Commit Changes:** Commit updated `environment.yml`, `environment-dev.yml`, and all regenerated `conda-*.lock` files.

## Developer Tools and Workflow

(This section remains largely unchanged but is important for context)

The `omnidsp-dev` Conda environment provides `pytest`, `ruff`, `pre-commit`, `conda-lock`. Other tools are managed by `pre-commit` itself:

- `clang-format` (for C++)
- `prettier` (for Markdown, YAML, TOML)
- `conventional-pre-commit` (for commit message validation)
- `nbstripout` (for Jupyter Notebooks)

### Using `pre-commit` for Code Quality

1.  **Initial Setup:** `pre-commit install --hook-type commit-msg --hook-type pre-commit` (once per clone).
2.  **Workflow:** Hooks run automatically on `git commit`. If they fail (e.g., formatting changes), `git add` the modified files and commit again.

### Updating Development Tools

- **Conda Environment Tools:** Modify `environment-dev.yml`, commit, and developers update via `conda env update --name omnidsp-dev --file environment-dev.yml --prune`.
- **`pre-commit` Hooks:** Modify `.pre-commit-config.yaml`, commit. `pre-commit` auto-updates on the next run.

## Building for Development

_(This section assumes the `omnidsp-dev` environment is active)_

### Python Package (Recommended)

This is the standard way to build if you intend to use or test the Python bindings.

1.  **Activate Conda Environment:** `conda activate omnidsp-dev`
2.  **Editable Install:** From the project root directory (containing `pyproject.toml`), run:
    ```
    pip install -e . -v
    ```

### C++ Library Only

If you only need the C++ library:

1.  **Activate Conda Environment:** `conda activate omnidsp-dev`
2.  **Configure with CMake:**
    ```
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install -DOMNIDSP_BUILD_PYTHON_BINDINGS=OFF # Add other options
    ```
3.  **Build:** `cmake --build . --config Release --parallel`
4.  **(Optional) Install:** `cmake --install . --config Release`

## Project Structure

- `include/OmniDSP/`: Public C++ headers.
  - `omnidsp.hpp`: Main `OmniDSP` class definition.
  - `fft.hpp`, `convolution.hpp`, `correlation.hpp`, `window.hpp`: Headers for `Plan` types.
  - `fir_filter.hpp`, `iir_filter.hpp`, `cqt.hpp`, `resample.hpp`: Headers for `Processor` types.
  - `window/specs.hpp`: Definitions for `Window::[SpecType]` structs (e.g., `Window::Hann`, `Window::Kaiser`).
  - `params/`: `Params::[Type]` structs for operation configuration.
  - `design/`: `Design::[Type]` structs (internal representations, e.g., `Design::FIRFilter`).
  - `coefs/`: `Coefs::[Type]` structs for pre-calculated coefficients.
  - `core_types.hpp`, `status.hpp`, `expected.hpp`: Core utilities.
- `src/omnidsp/`: Core C++ library implementation.
  - `interface/`: Contains abstract PIMPL interfaces.
    - `plan/`: Directory for abstract plan interfaces (e.g., `plan/fft_impl.hpp` for `Abstract::Plan::FFTImpl`).
    - `processor/`: Directory for abstract processor interfaces (e.g., `processor/fir_filter_impl.hpp` for `Abstract::Processor::FIRFilterImpl`).
  - `default/`: `Default::Backend` and its concrete PIMPL implementations.
    - `plan/`: Concrete default plan PIMPLs (e.g., `plan/fft_impl.cpp` for `Default::Plan::FFTImpl`, `plan/window/hann_impl.cpp` for `Default::Plan::Window::HannImpl`).
    - `processor/`: Concrete default processor PIMPLs.
  - `dispatcher/`: `Dispatcher::Backend` implementation.
  - `accelerate/`, `onemkl/`, `intelipp/`: Optimized backend implementations, structured similarly with `plan/` and `processor/` subdirectories for their PIMPLs.
  - `params/`: Source files for `Params::[Type]` constructors and validation.
  - `design/`: Source files for `OmniDSP::Design::create` functions.
  - `window/`: Internal window formula utilities (`OmniDSP::Internal::WindowFormulas`).
  - `omnidsp.cpp`: `OmniDSP` class implementation (factory methods).
- `src/omnidsp_py/`: Python bindings source code.
- `tests/`: Unit tests (`cpp/`, `python/`).
- `examples/`: Usage examples (`cpp/`, `notebooks/`).
- `environment.yml`, `environment-dev.yml`, `conda-*.lock`: Conda environment files.
- `.pre-commit-config.yaml`: Configuration for pre-commit Git hooks.
- `CMakeLists.txt`: Main CMake build script.
- `pyproject.toml`: Python packaging configuration.

## Understanding the Architecture

OmniDSP employs a specific architecture to provide a consistent API while leveraging different high-performance backends.

### Core `OmniDSP` Class and Factories

- The primary entry point for users is the `OmniDSP` class (defined in `omnidsp.hpp`).
- An instance of this class is created via `OmniDSP::create(...)`, which selects and initializes the backend.
- This `OmniDSP` instance acts as a **factory** for creating `Plan` and `Processor` objects:
  - `dsp.create_plan(...)` for stateless operations (FFT, Convolution, Correlation).
  - `dsp.create_window_plan<T_Data>(const Window::[SpecType]& spec, size_t length)` for stateless windowing operations.
  - `dsp.create_processor(...)` for stateful operations (Filters, CQT, Resampling).
- It also provides convenience wrappers like `dsp.generate_window_coefficients<T_Data>(...)`.

### `Plan::[Type]` Objects (Stateless)

- For operations that do not maintain state across calls (e.g., FFT, Convolution, Correlation, Windowing).
- Created by `OmniDSP::create_plan(...)` or `OmniDSP::create_window_plan(...)`.
- Examples: `Plan::FFT`, `Plan::RFFT`, `Plan::Convolution`, `Plan::Correlation`, `Plan::Window`.
- Encapsulate configuration and optimized context. `execute(...)` method performs the operation.
- Generally thread-safe for concurrent `execute` if the backend implementation is reentrant.

### `Processor::[Type]` Objects (Stateful)

- For operations requiring state to be maintained between calls (e.g., FIR/IIR filters, CQT, Resamplers).
- Created by `OmniDSP::create_processor(...)`.
- Examples: `Processor::FIRFilter`, `Processor::IIRFilter`, `Processor::CQT`, `Processor::Resample`.
- Encapsulate configuration, optimized context, and internal state.
- `execute(...)` method performs the operation and updates internal state.
- `reset()` method re-initializes the internal state.
- **Not thread-safe** for concurrent `execute` on the same instance.

### Configuration: `Params`, `Window::[SpecType]`, `Coefs`, `Design`

- **`Params::[Type]`**: User-facing structs for configuring operations (e.g., `Params::FFT`, `Params::FIRFilter`).
- **`Window::[SpecType]`**: Lightweight, user-facing structs for specifying window types and parameters (e.g., `Window::Hann{}`, `Window::Kaiser{beta}`). Used with `create_window_plan` and `generate_window_coefficients`, and internally by some `Params` like `Params::FIRFilter`.
- **`Coefs::[Type]`**: For providing pre-calculated coefficients, bypassing design steps.
- **`Design::[Type]`**: Internal representation (e.g., `Design::FIRFilter`) created from `Params::[Type]` via `OmniDSP::Design::create(...)` for complex design-based operations like filters. This `Design` object is then passed to the backend processor implementation.

### Backend Abstraction (PIMPL)

- The library heavily uses the Pointer to Implementation (PIMPL) idiom for `OmniDSP`, `Plan::[Type]`, and `Processor::[Type]` classes.
- Public headers do not expose backend-specific details, ensuring ABI stability and decoupling.

### Internal Namespace and PIMPL Naming

- Backend-specific implementation code resides in `OmniDSP::[BackendName]` namespaces (e.g., `OmniDSP::Default`, `OmniDSP::IntelIPP`). `[BackendName]` is the conceptual name of the backend.
- Abstract PIMPL interfaces are defined within the `OmniDSP::Abstract` namespace, further organized into `Plan` and `Processor` sub-namespaces.
  - Example for a plan: `OmniDSP::Abstract::Plan::FFTImpl`
  - Example for a processor: `OmniDSP::Abstract::Processor::FIRFilterImpl`
  - Example for a window plan: `OmniDSP::Abstract::Plan::WindowImpl<T_Data>`
- Concrete backend PIMPLs implement these interfaces and follow a convention like `OmniDSP::[BackendName]::Plan::[Type]Impl` or `OmniDSP::[BackendName]::Processor::[Type]Impl`.
  - Example: `OmniDSP::Default::Plan::Window::HannImpl<T_Data>` implements `OmniDSP::Abstract::Plan::WindowImpl<T_Data>`.

### Design Rationale

The architecture (central `OmniDSP` factory, `Plan`/`Processor` distinction, specific window configuration) aims for:

- **Clarity & Consistency:** Explicit creation of backend-managed objects.
- **Control & Flexibility:** Clear control over backend and operation configuration.
- **Maintainability & Extensibility:** Well-defined roles and interfaces.
- **Clear Concurrency Model:** `Plan` objects are generally shareable; `Processor` objects are not for concurrent execution on the same instance.

## Running Tests

### Python Tests

Run Python tests using pytest from the root directory:

```
pytest tests/python
```

These tests check Python bindings and API, ensuring `OmniDSP` instances correctly create and use `Plan` and `Processor` objects, including window plans.

### C++ Tests

C++ tests use GoogleTest and reference data.

#### Selecting Backend for Tests

Tests compile against the backend selected during CMake configuration (`-DBACKEND=...`).

#### C++ Testing Framework Overview

(This sub-section remains largely unchanged)

- Test implementations in `tests/cpp/tests/`.
- Reference data in `tests/cpp/data/`.
- Generation scripts in `tests/cpp/scripts/` (master script `tests/cpp/data.py`).
- `TestDataLoader.h/.cpp` for loading data.

##### Generating/Regenerating Reference Data

Run `python tests/cpp/data.py` (or with `--force` or specific suites). Regenerate if generation logic or parameters change, or new tests are added. Commit updated `.txt` files.

##### Running C++ Tests

1. Navigate to build directory: `cd build`
2. Run CTest: `ctest -C Debug` (or Release). Add `-V` for verbose output.
3. Or run executable directly: `./tests/cpp/omnidsp_tests` (Linux/macOS), `.\tests\cpp\Debug\omnidsp_tests.exe` (Windows). Filter with `--gtest_filter=SuiteName.*`.

**Note:** C++ tests must instantiate `OmniDSP::OmniDSP` and use its factory methods (`create_plan`, `create_window_plan`, `create_processor`) to obtain `Plan` or `Processor` objects for testing.

## Coding Style

- **C++:** `clang-format` via `pre-commit`. Doxygen-style comments.
- **Python:** `ruff` via `pre-commit`. PEP 8, Google-style docstrings.
- **CMake, YAML/TOML/Markdown:** `prettier` via `pre-commit`.

## Commit Message Format (Conventional Commits)

(This section remains unchanged)
Format: `<type>[optional scope]: <description>`. Hook `conventional-pre-commit` validates. See [conventionalcommits.org](https://www.conventionalcommits.org/).

## Submitting Contributions

(This section remains largely unchanged, but steps should reflect new architecture if adding features/tests)

1. Fork, clone, branch.
2. Make changes.
3. **Add Tests:** Cover new `Plan`/`Processor` logic, windowing features, backend fallbacks, etc.
4. Update dependencies if needed (regenerate `conda-*.lock` files).
5. **Update Documentation:** Reflect API changes for `Plan`/`Processor`, windowing, etc.
6. Commit (Conventional Commits).
7. Push to fork, open PR.

## Reporting Bugs and Suggesting Features

Use GitHub Issues for bug reports and feature suggestions.

## Where to Contribute

Check `TODO.md`. Key areas:

- Implementing remaining features (STFT, DCT/DST, `filtfilt`).
- Expanding test coverage for the new `Plan`/`Processor`/`Window` architecture, including backend fallbacks for windowing and parameter validation in window specs.
- Improving documentation and examples for the new API, especially window creation and usage.
- CI for different platforms.
- Performance benchmarking.

Thank you for contributing!
