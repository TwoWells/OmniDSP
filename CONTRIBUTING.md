# Contributing to OmniDSP

Thank you for your interest in contributing to OmniDSP! We welcome contributions from everyone. This document provides guidelines for contributing to the project.

## Table of Contents

- [Contributing to OmniDSP](#contributing-to-omnidsp)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Setting up the Development Environment](#setting-up-the-development-environment)
  - [Dependency Management with Conda Lock](#dependency-management-with-conda-lock)
    - [Contributor Workflow for Runtime Dependencies](#contributor-workflow-for-runtime-dependencies)
  - [Developer Tools and Workflow](#developer-tools-and-workflow)
    - [Using `pre-commit` for Code Quality](#using-pre-commit-for-code-quality)
    - [Updating Development Tools](#updating-development-tools)
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

The project uses Conda to manage dependencies. For contributors, we provide a specific development environment definition (`environment-dev.yml`) that includes not only the runtime dependencies needed to run OmniDSP but also essential tools for development, testing, code formatting, and managing dependencies.

1.  **Create Development Environment:** From the project root directory, create the development environment using the `environment-dev.yml` file:

    ```bash
    conda env create -f environment-dev.yml
    ```

    This command creates a Conda environment named `omnidsp-dev` (defined within the file). This environment contains all packages from `environment.yml` plus developer-specific tools like `pytest`, `ruff`, `clang-format`, `pre-commit`, and `conda-lock`.

2.  **Activate Environment:** Activate the newly created environment:

    ```bash
    conda activate omnidsp-dev
    ```

    **Important:** Always ensure this environment is activated before building, running tests, using the library, or running developer tools like `pre-commit` or `conda-lock`. The `pre-commit` hooks rely on tools (`clang-format`, `ruff`, etc.) being available within this activated environment.

3.  **Install Git Hooks:** After activating the environment for the first time in a new clone, install the `pre-commit` Git hooks:
    ```bash
    pre-commit install --hook-type commit-msg --hook-type pre-commit
    ```
    This sets up automatic code quality checks (formatting, linting) and commit message validation that run before each commit. See the "Developer Tools and Workflow" section for more details.

_(Note: While end-users or CI might use the platform-specific `conda-_.lock`files derived from`environment.yml`for maximum reproducibility of the runtime environment (which will be named`omnidsp`), developers typically work with the slightly more flexible `environment-dev.yml`which creates the`omnidsp-dev` environment.)\*

## Dependency Management with Conda Lock

To ensure consistent and reproducible _runtime_ environments across different operating systems and in our Continuous Integration (CI) workflows, this project uses `conda-lock` with **explicit, platform-specific lock files**.

- **Source File:** The `environment.yml` file defines the primary _runtime_ dependencies (for the `omnidsp` environment), their constraints, and the target platforms for locking.
- **Lock Files:** Explicit, platform-specific lock files (e.g., `conda-osx-arm64.lock`, `conda-linux-64.lock`, `conda-win-64.lock`) are generated from `environment.yml`. These lock files contain the exact package URLs and hashes needed for _each specific platform_ and are committed to the repository. End-users and CI should ideally use these lock files to create the `omnidsp` environment.
- **Development Environment:** The `environment-dev.yml` file includes all runtime dependencies _plus_ developer tools. It is typically used directly for creating the separate `omnidsp-dev` developer environment (`conda env create -f environment-dev.yml`).

### Contributor Workflow for Runtime Dependencies

**If you are NOT changing runtime dependencies in `environment.yml`:**

You generally don't need to interact directly with `conda-lock`. Simply use the `omnidsp-dev` environment created from `environment-dev.yml`.

**If you ARE changing runtime dependencies in `environment.yml`:**

1.  **Ensure `conda-lock` is Available:** The `conda-lock` tool is included in the `omnidsp-dev` environment defined by `environment-dev.yml`. Ensure this environment is active (`conda activate omnidsp-dev`).

2.  **Modify `environment.yml`:** Add, remove, or update _runtime_ packages or target platforms in `environment.yml` as needed. Remember to use platform selectors (e.g., `# [win]`, `# [linux]`) if a dependency is platform-specific. **Crucially, also make the same additions/removals/updates to the corresponding runtime dependency section in `environment-dev.yml` to keep the development environment consistent.**

3.  **Regenerate Lock Files:** After modifying `environment.yml`, you **MUST** regenerate the explicit lock files for all supported platforms. From the project root directory (with the `omnidsp-dev` environment activated), run:

    ```bash
    # Reads platforms from environment.yml, generates explicit lock files for the 'omnidsp' env
    conda-lock lock -f environment.yml --kind explicit
    ```

    _(This command generates files like `conda-osx-arm64.lock`, `conda-linux-64.lock`, etc.)_

4.  **Commit Changes:** Commit **both** the updated `environment.yml` file AND all the regenerated `conda-*.lock` files in the same commit. This keeps the source definition and the resolved dependencies in sync. Remember to also commit the changes made to `environment-dev.yml`.

## Developer Tools and Workflow

The `omnidsp-dev` Conda environment (created from `environment-dev.yml`) provides several tools to aid development and ensure code quality:

- **`pytest`:** For running the Python test suite.
- **`ruff`:** A fast Python linter and formatter (used via `pre-commit`).
- **`clang-format`:** A C++ code formatter (provided by this environment and used via `pre-commit`).
- **`prettier`:** An opinionated code formatter used via `pre-commit` for Markdown (`.md`), YAML (`.yaml`/`.yml`), and TOML (`.toml`) files.
- **`pre-commit`:** A framework for managing Git hooks that run checks before commits.
- **`conventional-pre-commit`:** A hook (used via `pre-commit`) to validate commit messages against the Conventional Commits standard.
- **`conda-lock`:** For regenerating runtime dependency lock files (see previous section).
- **`nbstripout`:** Removes output cells from Jupyter Notebooks (used via `pre-commit`).

### Using `pre-commit` for Code Quality

To maintain consistent code style, catch potential issues early, and ensure informative commit messages, this project uses `pre-commit` hooks defined in `.pre-commit-config.yaml`. These hooks automatically run tools like `ruff`, `clang-format`, and `prettier` (using the versions installed in the `omnidsp-dev` environment or managed by the hook itself) on changed files or the commit message itself before you make a commit.

1.  **Initial Setup:** After creating and activating the `omnidsp-dev` environment, run `pre-commit install --hook-type commit-msg --hook-type pre-commit` once per clone (as described in "Setting up the Development Environment"). This installs hooks for both pre-commit (formatting/linting) and commit-msg (message validation) stages.
2.  **Workflow:**
    - Stage your changes (`git add ...`).
    - Run `git commit`.
    - **Commit Message Validation:** The `commit-msg` hook (`conventional-pre-commit`) runs first. If your commit message doesn't follow the Conventional Commits format, the commit will be aborted with an error message. Edit your commit message and try again.
    - **Formatting/Linting:** If the message is valid, the `pre-commit` hooks (formatters, linters) run on the staged files.
    - **If hooks pass:** Your commit proceeds as usual.
    - **If hooks fail (e.g., a formatter modified files):** The commit will be aborted. `pre-commit` will output messages indicating which files were changed. Simply `git add` the modified files again and re-run `git commit`. The second attempt should pass if the only issues were formatting changes fixed by the hooks.

### Updating Development Tools

The tools used for development might occasionally need updates or additions.

- **Adding/Updating a Tool in the Conda Environment:**

  1.  Modify the `dependencies:` list within `environment-dev.yml` to add, remove, or change the version constraint of a Conda package (e.g., updating `pytest` or `clang-format`). Remember this file contains both runtime and dev dependencies, so modify the appropriate section.
  2.  Commit the changes to `environment-dev.yml`.
  3.  Existing developers will need to update their local `omnidsp-dev` environment by running:
      ```bash
      # Ensure no environment is active or activate a different one first
      # conda deactivate
      conda env update --name omnidsp-dev --file environment-dev.yml --prune
      conda activate omnidsp-dev
      ```
      The `--prune` option removes packages that are no longer listed in the file.

- **Adding/Updating a `pre-commit` Hook:**
  1.  Modify the `.pre-commit-config.yaml` file to add a new hook, update the `rev:` of an existing hook repository, or change hook arguments.
  2.  Commit the changes to `.pre-commit-config.yaml`.
  3.  **No extra steps are usually required by developers.** The `pre-commit` framework automatically detects changes to `.pre-commit-config.yaml`. The next time you run `git commit` (or manually run `pre-commit run ...`), it will download and install any new tools or updated versions specified in the configuration into its internal cache _unless_ the hook is configured to use the `language: system` entry (which we try to avoid). You generally **do not** need to re-run `pre-commit install` or run commands like `pre-commit clean`.
  4.  If a hook fails unexpectedly after an update (e.g., after pulling changes that modified `.pre-commit-config.yaml`), you can try running `pre-commit run <hook_id> --all-files` (replace `<hook_id>` with the specific hook ID) to see more detailed output or potentially force a refresh of that hook's environment. In rare cases, `pre-commit clean` followed by a hook run might be needed for troubleshooting, but avoid it unless necessary.

**Always commit changes to configuration files (`environment.yml`, `environment-dev.yml`, `conda-*.lock`, `.pre-commit-config.yaml`) so that all contributors stay synchronized.**

## Building for Development

_(This section assumes the `omnidsp-dev` environment is active)_

### Python Package (Recommended)

This is the standard way to build if you intend to use or test the Python bindings.

1.  **Activate Conda Environment:** `conda activate omnidsp-dev`
2.  **Editable Install:** From the project root directory (containing `pyproject.toml`), run:
    ```bash
    pip install -e . -v
    ```
    - The `-e` flag installs the package in "editable" mode... _(rest of explanation unchanged)_

### C++ Library Only

If you only need the C++ library for use in another C++ project:

1.  **Activate Conda Environment:** `conda activate omnidsp-dev`
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

- `include/OmniDSP/`: Public C++ headers defining the API (`fft.h`, `cqt.h`, `convolution.h`, `window.h`, `resample.h`, `omnidsp.h`).
- `src/omnidsp/`: Core C++ library implementation files defining the public API wrappers (`fft.cpp`, `cqt.cpp`, `convolution.cpp`, `window.cpp`, `resample.cpp`).
- `src/omnidsp/backend/`: Contains backend-specific implementations.
  - `backend_impl.h`: Declares the interface required by all backends.
  - `onemkl/`: Implementations using Intel oneMKL/IPP.
  - `accelerate/`: Implementations using Apple Accelerate/vDSP.
  - `stub/`: Stub implementations that throw runtime errors.
- `src/omnidsp_py/`: Python bindings source code (pybind11 wrappers).
- `tests/`: Unit tests (primarily for developers).
  - `cpp/`: C++ tests using GoogleTest.
  - `python/`: Python tests using pytest.
- `examples/`: Usage examples.
  - `cpp/`: C++ examples.
  - `notebooks/`: Python examples using Jupyter notebooks.
- `environment.yml`: Conda environment definition source (for **runtime** env named `omnidsp`).
- `environment-dev.yml`: Conda environment definition source (for **developers** env named `omnidsp-dev`).
- `conda-*.lock`: Explicit, platform-specific Conda lock files (generated by `conda-lock` from `environment.yml`).
- `.pre-commit-config.yaml`: Configuration for pre-commit Git hooks.
- `CMakeLists.txt`: Main CMake build script.
- `pyproject.toml`: Python packaging configuration (uses `scikit-build-core`).
- `TODO.md`: Current development tasks.
- `CONTRIBUTING.md`: Guidelines for contributors.

## Understanding Backends

_(No changes needed here)_

## Running Tests

_(Assumes `omnidsp-dev` environment is active)_

### C++ Tests (GoogleTest)

_(No changes needed here)_

### Python Tests (pytest)

_(No changes needed here)_

### Test Reference Data

_(No changes needed here)_

## Coding Style

- **C++:** Please try to follow the existing code style. We use `clang-format` via `pre-commit` to enforce consistency (check `.pre-commit-config.yaml` and potentially a `.clang-format` file for style details). The `clang-format` executable itself is provided by the `omnidsp-dev` conda environment defined in `environment-dev.yml`. Use Doxygen-style comments for documenting headers (`/** ... */`) and implementation details where appropriate.
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
- **Learn More:** <https://www.conventionalcommits.org/>

## Submitting Contributions

We appreciate contributions! Please follow this general workflow:

1.  **Fork the Repository:** Create your own fork of the OmniDSP repository on GitHub.
2.  **Create a Branch:** Create a new branch in your fork for your feature or bug fix (e.g., `git checkout -b feat/add-convolution-modes`).
3.  **Make Changes:** Implement your feature or fix the bug.
4.  **Add Tests:** Add appropriate unit tests (C++ and/or Python) to cover your changes. Ensure all tests pass (`pytest tests/python`, `ctest` in build dir).
5.  **Update Dependencies (if needed):**
    - If you changed _runtime_ dependencies in `environment.yml`, regenerate and commit the explicit `conda-*.lock` files **and ensure `environment-dev.yml` is also updated** (see Dependency Management section).
    - If you changed _developer_ tools in `environment-dev.yml` or hooks in `.pre-commit-config.yaml`, commit those files.
6.  **Update Documentation:** If you add new features or change existing ones, update the README, docstrings, and potentially other documentation files (like this one!).
7.  **Commit Changes:** Commit your changes with clear and concise commit messages adhering to the **Conventional Commits** format. `pre-commit` hooks will run automatically to validate formatting and the commit message.
8.  **Push to Your Fork:** Push your branch to your GitHub fork.
9.  **Open a Pull Request:** Create a Pull Request (PR) from your branch to the main OmniDSP repository's `main` (or appropriate development) branch. Provide a clear description of your changes in the PR.

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

Check the `TODO.md` file for a list of known tasks and desired features. Some key areas include:

- Tuning the CQT scaling factor.
- Implementing `double` precision support for resampling in the MKL backend.
- Adding 'same'/'full' convolution modes.
- Expanding CI coverage.
- Adding more tests and examples.
- Improving documentation.

Thank you for contributing!
