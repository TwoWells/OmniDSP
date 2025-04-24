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
    - [Core `OmniDSP` Class](#core-omnidsp-class)
    - [`Plan` Objects](#plan-objects)
    - [Backend Abstraction (PIMPL)](#backend-abstraction-pimpl)
    - [Internal Namespace](#internal-namespace)
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

The Boost C++ libraries were added as a dependency primarily for `Boost.Math`. Specifically, the library requires the 0th order modified Bessel function of the first kind (`I₀`) to generate Kaiser windows coefficients accurately and consistently. While C++17 includes mathematical special functions like `std::cyl_bessel_i`, its implementation is missing in certain standard library versions (notably libc++ on macOS). Using `boost::math::cyl_bessel_i` ensures a reliable implementation across all supported platforms. Boost is automatically installed via the Conda environment file.

## Dependency Management with Conda Lock

To ensure consistent and reproducible _runtime_ environments across different operating systems and in our Continuous Integration (CI) workflows, this project uses `conda-lock` with **explicit, platform-specific lock files**.

- **Source File:** The `environment.yml` file defines the primary _runtime_ dependencies (for the `omnidsp` environment), their constraints, and the target platforms for locking.
- **Lock Files:** Explicit, platform-specific lock files (e.g., `conda-osx-arm64.lock`, `conda-linux-64.lock`, `conda-win-64.lock`) are generated from `environment.yml`. These lock files contain the exact package URLs and hashes needed for _each specific platform_ and are committed to the repository. End-users and CI should ideally use these lock files to create the `omnidsp` environment.
- **Development Environment:** The `environment-dev.yml` file includes all runtime dependencies _plus_ developer tools like `pytest`, `ruff`, `pre-commit`, and `conda-lock`. It is typically used directly for creating the separate `omnidsp-dev` developer environment (`conda env create -f environment-dev.yml`).

### Contributor Workflow for Runtime Dependencies

**If you are NOT changing runtime dependencies in `environment.yml`:**

You generally don't need to interact directly with `conda-lock`. Simply use the `omnidsp-dev` environment created from `environment-dev.yml`.

**If you ARE changing runtime dependencies in `environment.yml`:**

1.  **Ensure `conda-lock` is Available:** The `conda-lock` tool is included in the `omnidsp-dev` environment defined by `environment-dev.yml`. Ensure this environment is active (`conda activate omnidsp-dev`).
2.  **Modify `environment.yml`:** Add, remove, or update _runtime_ packages or target platforms in `environment.yml` as needed. Remember to use platform selectors (e.g., `# [win]`, `# [linux]`) if a dependency is platform-specific. **Crucially, also make the same additions/removals/updates to the corresponding runtime dependency section in `environment-dev.yml` to keep the development environment consistent.**
3.  **Regenerate Lock Files:** After modifying `environment.yml`, you **MUST** regenerate the explicit lock files for all supported platforms. From the project root directory (with the `omnidsp-dev` environment activated), run:

    ```
    # Reads platforms from environment.yml, generates explicit lock files for the 'omnidsp' env
    conda-lock lock -f environment.yml --kind explicit
    ```

    _(This command generates files like `conda-osx-arm64.lock`, `conda-linux-64.lock`, etc.)_

4.  **Commit Changes:** Commit **both** the updated `environment.yml` file AND all the regenerated `conda-*.lock` files in the same commit. This keeps the source definition and the resolved dependencies in sync. Remember to also commit the changes made to `environment-dev.yml`.

## Developer Tools and Workflow

The `omnidsp-dev` Conda environment (created from `environment-dev.yml`) provides essential tools like `pytest`, `ruff`, `pre-commit`, and `conda-lock`. Other tools used for code quality are managed directly by `pre-commit`:

- `pytest`: For running the Python test suite (provided by Conda env).
- `ruff`: A fast Python linter and formatter (provided by Conda env, used via `pre-commit`).
- `clang-format`: A C++ code formatter (installed automatically by the `pre-commit` hook `mirrors-clang-format`).
- `prettier`: An opinionated code formatter (installed automatically by the `pre-commit` hook `mirrors-prettier`) for Markdown (`.md`), YAML (`.yaml`/`.yml`), and TOML (`.toml`) files.
- `pre-commit`: A framework for managing Git hooks (provided by Conda env).
- `conventional-pre-commit`: A hook (used via `pre-commit`) to validate commit messages against the Conventional Commits standard.
- `conda-lock`: For regenerating runtime dependency lock files (provided by Conda env).
- `nbstripout`: Removes output cells from Jupyter Notebooks (installed automatically by its `pre-commit` hook).

### Using `pre-commit` for Code Quality

To maintain consistent code style, catch potential issues early, and ensure informative commit messages, this project uses `pre-commit` hooks defined in `.pre-commit-config.yaml`. These hooks automatically run tools like `ruff`, `clang-format`, and `prettier` (using versions managed internally by `pre-commit` itself) on changed files or the commit message itself before you make a commit.

1.  **Initial Setup:** After creating and activating the `omnidsp-dev` environment, run `pre-commit install --hook-type commit-msg --hook-type pre-commit` once per clone (as described in "Setting up the Development Environment"). This installs hooks for both pre-commit (formatting/linting) and commit-msg (message validation) stages.
2.  **Workflow:**
    - Stage your changes (`git add ...`).
    - Run `git commit`.
    - **Commit Message Validation:** The `commit-msg` hook (`conventional-pre-commit`) runs first. If your commit message doesn't follow the Conventional Commits format, the commit will be aborted with an error message. Edit your commit message and try again.
    - **Formatting/Linting:** If the message is valid, the `pre-commit` hooks (formatters, linters) run on the staged files.
    - **If hooks pass:** Your commit proceeds as usual.
    - **If hooks fail (e.g., a formatter modified files):** The commit will be aborted. `pre-commit` will output messages indicating which files were changed. Note that `pre-commit` modifies your files in place but **does not automatically stage** these changes. You must manually run `git add` on the files listed in the output before attempting the commit again. The second attempt should pass if the only issues were formatting changes fixed by the hooks.

### Updating Development Tools

The tools used for development might occasionally need updates or additions.

- **Adding/Updating a Tool in the Conda Environment:**

  1.  Modify the `dependencies:` list within `environment-dev.yml` to add, remove, or change the version constraint of a Conda package (e.g., updating `pytest` or `ruff`). Remember this file contains both runtime and dev dependencies, so modify the appropriate section.
  2.  Commit the changes to `environment-dev.yml`.
  3.  Existing developers will need to update their local `omnidsp-dev` environment by running:

      ```
      # Ensure no environment is active or activate a different one first
      # conda deactivate
      conda env update --name omnidsp-dev --file environment-dev.yml --prune
      conda activate omnidsp-dev
      ```

      The `--prune` option removes packages that are no longer listed in the file.

- **Adding/Updating a `pre-commit` Hook:**
  1.  Modify the `.pre-commit-config.yaml` file to add a new hook, update the `rev:` of an existing hook repository (e.g., updating the version of `clang-format` used by `mirrors-clang-format`), or change hook arguments.
  2.  Commit the changes to `.pre-commit-config.yaml`.
  3.  **No extra steps are usually required by developers.** The `pre-commit` framework automatically detects changes to `.pre-commit-config.yaml`. The next time you run `git commit` (or manually run `pre-commit run ...`), it will download and install any new tools or updated versions specified in the configuration into its internal cache. You generally **do not** need to re-run `pre-commit install` or run commands like `pre-commit clean`.
  4.  If a hook fails unexpectedly after an update (e.g., after pulling changes that modified `.pre-commit-config.yaml`), you can try running `pre-commit run <hook_id> --all-files` (replace `<hook_id>` with the specific hook ID) to see more detailed output or potentially force a refresh of that hook's environment. In rare cases, `pre-commit clean` followed by a hook run might be needed for troubleshooting, but avoid it unless necessary.

**Always commit changes to configuration files (`environment.yml`, `environment-dev.yml`, `conda-*.lock`, `.pre-commit-config.yaml`) so that all contributors stay synchronized.**

## Building for Development

_(This section assumes the `omnidsp-dev` environment is active)_

### Python Package (Recommended)

This is the standard way to build if you intend to use or test the Python bindings.

1.  **Activate Conda Environment:** `conda activate omnidsp-dev`
2.  **Editable Install:** From the project root directory (containing `pyproject.toml`), run:

    ```
    pip install -e . -v
    ```

    The `-e` flag installs the package in "editable" mode. Changes to C++ code require rebuilding (which `pip install -e .` triggers if needed). The `-v` flag provides verbose output.

### C++ Library Only

If you only need the C++ library for use in another C++ project:

1.  **Activate Conda Environment:** `conda activate omnidsp-dev`
2.  **Configure with CMake:**

    ```
    mkdir build && cd build
    # Configure without Python bindings (set OFF explicitly)
    cmake .. -DCMAKE_INSTALL_PREFIX=../install -DOMNIDSP_BUILD_PYTHON_BINDINGS=OFF # Add other options
    ```

3.  **Build:** `cmake --build . --config Release --parallel`
4.  **(Optional) Install:** `cmake --install . --config Release`

## Project Structure

- `include/OmniDSP/`: Public C++ headers defining the API (`omnidsp.h`, `fft.h`, `cqt.h`, `resample.h`, `core_types.h`).
- `src/omnidsp/`: Core C++ library implementation files (`omnidsp.cpp`, `fft.cpp`, `cqt.cpp`, `resample.cpp`). These implement the public API classes and forward calls to the backend implementations via PIMPL.
- `src/omnidsp/backend/`: Contains backend-specific implementations and the internal interface header.
  - `backend.h`: Internal header defining implementation class interfaces/signatures (within `OmniDSP::backend` namespace). **Not part of the public API.**
  - `onemkl/`: Implementations using Intel oneMKL/IPP (namespace `OmniDSP::backend::oneMKL`).
  - `accelerate/`: Implementations using Apple Accelerate/vDSP (namespace `OmniDSP::backend::Accelerate`).
  - `stub/`: Stub implementations that throw runtime errors (namespace `OmniDSP::backend::Stub`).
- `src/omnidsp_py/`: Python bindings source code (`bindings.cpp`, `api.py`, `__init__.py`).
- `tests/`: Unit tests.
  - `cpp/`: C++ tests using GoogleTest.
  - `python/`: Python tests using pytest.
- `examples/`: Usage examples.
  - `cpp/`: C++ examples.
  - `notebooks/`: Python examples using Jupyter notebooks.
- `environment.yml`, `environment-dev.yml`, `conda-*.lock`: Conda environment files.
- `.pre-commit-config.yaml`: Configuration for pre-commit Git hooks.
- `CMakeLists.txt`: Main CMake build script.
- `pyproject.toml`: Python packaging configuration.
- `TODO.md`: Current development tasks.
- `CONTRIBUTING.md`: Guidelines for contributors (this file).

## Understanding the Architecture

OmniDSP employs a specific architecture to provide a consistent API while leveraging different high-performance backends (Intel MKL/IPP or Apple Accelerate).

### Core `OmniDSP` Class

- The primary entry point for users is the `OmniDSP` class (defined in `omnidsp.h`).
- You **must** create an instance of this class first (e.g., `OmniDSP::OmniDSP dsp;` in C++ or `dsp = omnidsp_py.OmniDSP()` in Python).
- The `OmniDSP` constructor automatically selects and initializes the appropriate backend based on the build configuration (MKL or Accelerate).
- This instance manages the backend context and any associated global state (e.g., potential MKL thread settings).
- It provides methods for **stateless** DSP operations (e.g., `dsp.convolve1d(...)`, `dsp.get_hann_coeffs(...)`).
- It acts as a **factory** for creating stateful `Plan` objects.

### `Plan` Objects

- For operations requiring significant setup or maintaining state (like FFTs, CQT, Resampling), OmniDSP uses dedicated `Plan` classes (`FFTPlan`, `RFFTPlan`, `CQTPlan`, `ResamplePlan`).
- These `Plan` objects are **created** using factory methods on an existing `OmniDSP` instance (e.g., `auto fft_plan = dsp.create_fft_plan<float>(...);`). You do not construct `Plan` objects directly.
- Each `Plan` object encapsulates the specific configuration and optimized state for its operation, consistent with the backend context managed by the `OmniDSP` instance that created it.
- Once created, you use the `Plan` object's `execute(...)` method to perform the DSP operation efficiently multiple times.

### Backend Abstraction (PIMPL)

- The library heavily uses the **Pointer to Implementation (PIMPL)** idiom.
- Both the main `OmniDSP` class and the individual `Plan` classes use PIMPL internally.
- This means the public headers (`include/OmniDSP/`) do not expose any backend-specific details (like MKL or Accelerate headers).
- The actual backend logic resides in internal implementation classes (e.g., `OmniDSP::backend::OmniDSPImpl`, `OmniDSP::backend::FFTPlanImpl`).
- This approach provides strong **ABI stability** (allowing library updates without forcing users to recompile) and **decoupling** (users don't need backend SDKs to compile against OmniDSP).

### Internal Namespace

- Backend-specific implementation code resides in the `OmniDSP::backend` C++ namespace and its sub-namespaces (`OmniDSP::backend::oneMKL`, `OmniDSP::backend::Accelerate`, `OmniDSP::backend::Stub`).
- Files and classes within this namespace are considered internal implementation details and are subject to change without notice. Users should only interact with the public API defined in the `include/OmniDSP/` headers and the main `OmniDSP` namespace.

### Design Rationale

The decision to center the API around an `OmniDSP` instance (which manages the backend and acts as a factory) rather than using standalone functions was made to:

- **Increase Clarity:** Explicitly creating an `OmniDSP` object makes the user aware they are interacting with a specific backend instance and its configuration. Standalone functions can obscure this dependency.
- **Ensure Consistency:** Centralizing backend setup within the `OmniDSP` constructor ensures that all operations (stateless methods and created Plans) execute within a consistent backend context (e.g., regarding threading settings).
- **Improve Control:** Users gain clearer control over the backend configuration by managing the `OmniDSP` instance. This could potentially allow for multiple instances with different configurations in the future.
- **Enhance Flexibility:** This architecture provides a natural point for future extensions, such as runtime backend switching, without requiring major changes to the user-facing API of the Plans or stateless methods.

While this approach requires users to instantiate an `OmniDSP` object first, the benefits in terms of transparency, consistency, control, and future maintainability were deemed to outweigh the simplicity of standalone functions.

## Running Tests

Testing is crucial. Please ensure all tests pass before submitting changes.

### Python Tests

Run Python tests using pytest from the root directory:

```
pytest tests/python
```

These tests check the Python bindings and API. Tests should now instantiate an `omnidsp_py.OmniDSP` object and use its methods to create plans and perform operations.

### C++ Tests

The C++ tests use GoogleTest and rely on reference data files.

#### Selecting Backend for Tests

The C++ tests are compiled against the backend selected during the CMake configuration step (`-DBACKEND=...`). When you run the C++ tests, they will exercise the implementation specific to that chosen backend (oneMKL or Accelerate).

To test a different backend, you need to re-configure CMake with the desired `-DBACKEND` option and rebuild.

#### C++ Testing Framework Overview

The C++ tests compare the output of OmniDSP functions against pre-generated reference data stored in plain text files. This ensures consistency across different backends and platforms.

##### Directory Structure

The relevant directories within `tests/cpp/` are:

- `tests/`: Contains the C++ test implementation files (e.g., `fft.cpp`, `window.cpp`), organized by module/suite.
- `data/`: Contains the reference data text files, organized in subdirectories named after the test suite (e.g., `data/fft/`, `data/window/`).
- `scripts/`: Contains Python scripts used to generate the reference data files found in `data/`. Includes a shared utility `_generate_utils.py`.
- `TestDataLoader.h/.cpp`: C++ utility code responsible for loading the reference data from the text files during test execution.
- `CMakeLists.txt`: Configures the C++ test build.

##### Reference Data Files

- **Location:** `tests/cpp/data/<SuiteName>/`
- **Naming Convention:** `<TestCaseName>_<Purpose>_<TypeSuffix>.txt`

  - `<SuiteName>`: Matches the test suite (e.g., `fft`, `window`).
  - `<TestCaseName>`: Matches the specific GoogleTest name (e.g., `Plan_FFT_Forward_Double`).
  - `<Purpose>`: Indicates the data's role, usually `input` or `expected`.
  - `<TypeSuffix>`: Indicates the data type: `d` (double), `f` (float), `cd` (complex double), `cf` (complex float).

  Example: `tests/cpp/data/fft/Plan_FFT_Forward_Double_expected_cd.txt`

- **Format:**
  1.  **Header Line:** The first line indicates dimensions, starting with '#'.
      - Vectors (1D): `# <rows>` (e.g., `# 1024`)
      - Matrices (2D): `# <rows>x<columns>` (e.g., `# 84x12`)
  2.  **Data Lines:** Subsequent lines contain whitespace-separated numerical data.
      - Real types: One number per element.
      - Complex types: `real imag` pairs per element.
- **Important:** These generated data files **should be committed to the repository**.

##### Generating/Regenerating Reference Data

Reference data is generated using Python scripts located in `tests/cpp/scripts/`. A master script (`tests/cpp/data.py`) is provided for convenience:

- Run `python tests/cpp/data.py --help` to see options.
- To regenerate **all** reference data (after prompting for confirmation):
  `python tests/cpp/data.py`
- To regenerate **all** reference data **without prompting**:
  `python tests/cpp/data.py --force` (or `-f`)
- To regenerate data only for **specific suites** (e.g., fft and window):
  `python tests/cpp/data.py fft window`

**When to Regenerate:** You should run the appropriate regeneration command if you:

- Modify the data generation logic in any `tests/cpp/scripts/*.py` file.
- Change parameters used for generation (e.g., window size, FFT length).
- Add new tests that require new reference data.

Remember to commit any changes to the generated `.txt` files in the `tests/cpp/data/` directories.

##### Running C++ Tests

After building the project:

1. Navigate to the build directory: `cd build`
2. Run CTest (which executes the test executable):
   `ctest -C Debug` (or Release)
   Add `-V` for verbose output: `ctest -C Debug -V`
3. Alternatively, run the executable directly (useful for debugging specific tests with filters):
   `./tests/cpp/omnidsp_tests` (Linux/macOS)
   `.\tests\cpp\Debug\omnidsp_tests.exe` (Windows, path might vary)
   Example with filter: `./tests/cpp/omnidsp_tests --gtest_filter=FFT_Test.*`

**Note:** C++ tests need to be updated to instantiate `OmniDSP::OmniDSP` and use its factory methods (`create_*_plan`) to obtain `Plan` objects for testing stateful operations. Stateless operations should be tested via methods on the `OmniDSP` instance.

## Coding Style

- **C++:** Please try to follow the existing code style. We use `clang-format` via `pre-commit` to enforce consistency (check `.pre-commit-config.yaml` and potentially a `.clang-format` file for style details). The `clang-format` executable itself is installed and managed automatically by the relevant `pre-commit` hook (`mirrors-clang-format`) and **does not** need to be installed separately in the Conda environment. Use Doxygen-style comments for documenting headers (`/** ... */`) and implementation details where appropriate.
- **Python:** We use `ruff` via `pre-commit` for code formatting and linting. Please follow PEP 8 guidelines. Use Google-style docstrings.
- **CMake:** Follow standard CMake practices for readability.
- **YAML/TOML/Markdown:** We use `prettier` via `pre-commit` to ensure consistent formatting for configuration and documentation files (`.yaml`, `.yml`, `.toml`, `.md`).

## Commit Message Format (Conventional Commits)

To ensure a clear and informative Git history, and to enable automated tooling (like skipping CI runs for documentation-only changes), this project adheres to the **Conventional Commits** specification.

- **Format:** Commit messages should follow the structure:

  ```
  <type>[optional scope]: <description>

  [optional body]

  [optional footer(s)]
  ```

- **Types:** Use standard types like `feat` (new feature), `fix` (bug fix), `docs` (documentation changes), `style` (formatting, code style), `refactor`, `test`, `build`, `ci`, `chore`, `perf`.
- **Pre-commit Check:** A `pre-commit` hook (`conventional-pre-commit`) is configured to validate your commit message format _before_ the commit is finalized. If your message is invalid, the commit will be aborted, and you will need to amend the message.
- **Purpose:** This helps maintain a semantic history, makes it easier to automatically generate changelogs, and allows CI workflows to intelligently skip runs for commits that don't affect production code (e.g., those starting with `docs:`, `style:`, `chore:`, `test:`).
- **Learn More:** [https://www.conventionalcommits.org/](https://www.conventionalcommits.org/)

## Submitting Contributions

We appreciate contributions! Please follow this general workflow:

1.  **Fork the Repository:** Create your own fork of the OmniDSP repository on GitHub.
2.  **Clone your fork locally.**
3.  **Create a Branch:** Create a new branch in your fork for your feature or bug fix (e.g., `git checkout -b feature/add-stft` or `bugfix/fix-cqt-scaling`).
4.  **Make Changes:** Implement your feature or fix the bug.
5.  **Add Tests:** Add appropriate unit tests (C++ and/or Python) to cover your changes. Ensure all tests pass (`pytest tests/python` and `ctest` in build dir).
6.  **Update Dependencies (if needed):**
    - If you changed _runtime_ dependencies in `environment.yml`, regenerate and commit the explicit `conda-*.lock` files **and ensure `environment-dev.yml` is also updated** (see Dependency Management section).
    - If you changed _developer_ tools in `environment-dev.yml` or hooks in `.pre-commit-config.yaml`, commit those files.
7.  **Update Documentation:** If you add new features or change existing ones, update the README, docstrings, and potentially other documentation files (like this one!).
8.  **Commit Changes:** Commit your changes with clear and concise commit messages adhering to the **Conventional Commits** format. `pre-commit` hooks will run automatically to validate formatting and the commit message.
9.  **Push to Your Fork:** Push your branch to your GitHub fork.
10. **Open a Pull Request:** Create a Pull Request (PR) from your branch to the main OmniDSP repository's `main` (or appropriate development) branch. Provide a clear description of your changes in the PR. Link to any relevant issues.

### Pull Request Workflow

- Ensure your code builds successfully on relevant platforms (if possible).
- Ensure all tests (C++ and Python) pass.
- Ensure commit messages follow the Conventional Commits standard.
- Address any feedback from reviewers.
- Once approved, your PR will be merged.

## Reporting Bugs and Suggesting Features

Please use the GitHub Issues tracker for the OmniDSP repository to:

- Report bugs (include steps to reproduce, environment details, expected vs. actual results).
- Suggest new features or enhancements.

## Where to Contribute

Check the `TODO.md` file (or the TODO list canvas artifact if using this tool) for a list of known tasks and desired features. Key areas include:

- Implementing remaining features outlined in the `TODO.md` (e.g., convolution modes, filter design).
- Expanding test coverage for the new architecture.
- Improving documentation and examples for the new API.
- Adding CI for different platforms.
- Performance benchmarking.

Thank you for contributing!
