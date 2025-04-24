# OmniDSP Project TODO List

**Goal:** Outline the development tasks for the OmniDSP library, including C++ core features, backend implementations, Python bindings, build system refactoring, testing, and documentation, ensuring alignment with the design philosophy.

## High Priority / Blocking Tasks

- **Refactor `ResamplePlan` to use `FilterPlan` Infrastructure:**

  - [ ] **Refactor `AccelerateResamplePlanImpl` (`src/omnidsp/backend/accelerate/resample.cpp`) (If Applicable):**
    - Update constructor to take `const OmniDSPImpl* owner` and `const ResampleSpec& spec`.
    - Constructor calls `calculate_factors` and `design_prototype_filter` to initialize members (`upsample_factor_L_`, `downsample_factor_M_`, `prototype_filter_coeffs_`).
    - Initialize `filter_state_` based on `prototype_filter_coeffs_` size.
    - Implement `execute` using Accelerate vDSP primitives for polyphase FIR filtering, using the designed `prototype_filter_coeffs_`.
    - Implement `reset()`.
  - [ ] **Add/Update Tests:**
    - Add C++ tests (`tests/cpp`) for `design_fir_filter`.
    - Add C++ and Python tests for the refactored `ResamplePlan`, covering various rate combinations, quality settings, state handling (`reset`), and edge cases.
    - Ensure CQT tests still pass after the CQT plan's dependency on the refactored resampler.
  - [ ] **Update Python Bindings:**
    - Modify Python `create_resample_plan` wrapper (`src/omnidsp_py/api.py`) to accept new parameters (e.g., quality) or a spec-like object/dictionary.
    - Update `src/omnidsp_py/bindings.cpp` to bind the new `create_resample_plan` C++ signature.

- **Implement Filter Module:**
  - [ ] Implement `FIRFilterPlanImpl` for all backends (`default`, `accelerate`, `onemkl`). _(Partially done for default)_
  - [ ] Implement `IIRFilterPlanImpl` for all backends (`default`, `accelerate`, `onemkl`). _(Partially done for default)_
  - [ ] Implement `OmniDSPImpl::create_fir_filter_plan` factory methods in all backends. _(Done for default)_
  - [ ] Implement `OmniDSPImpl::create_iir_filter_plan` factory methods in all backends. _(Done for default)_
  - [ ] Implement `OmniDSP::design_iir_filter` design method (likely in `DefaultOmniDSPImpl` initially).
  - [ ] Add Python bindings for Filter Plans and design methods.
  - [ ] Add comprehensive C++ and Python tests for the Filter module.
- **Review Core Implementations:**
  - [ ] Review C++ implementations (`.cpp` files) across all modules and backends against the latest `design_philosophy.md` to ensure full compliance with Pimpl structure, `OmniExpected`, API contracts, and naming conventions.

## Core Library Features

### CQT Module

- [ ] Review CQT Plan implementation (`cqt.cpp` / `backend/*/cqt.cpp`) - Ensure recursive CQT uses `ResamplePlan` correctly and Pimpl is correctly applied.
- [ ] Optimize CQT kernel calculation (`default/cqt.cpp`).
- [ ] Refine hop length handling in recursive CQT implementation.
- [ ] Tune CQT scaling factor.

### Convolution / Correlation Module

- [ ] Implement 'same' and 'full' modes for `OmniDSP::convolve` / `correlate` member functions and Plan execution. _(Partially implemented in Plan execute, needs one-off methods)_
- [ ] Add C++ and Python tests covering all convolution/correlation modes.

### STFT Module

- [ ] Design `STFTPlan` interface (consider using `FFTPlan` or `RFFTPlan` internally).
- [ ] Implement `STFTPlanImpl` for all backends.
- [ ] Implement `OmniDSPImpl::create_stft_plan` factory methods in all backends.
- [ ] Include inverse STFT (ISTFT) capability.
- [ ] Add Python bindings for STFT/ISTFT.
- [ ] Add tests for STFT module.

## Python Bindings (`omnidsp_py`)

- [ ] Review/Add bindings for Convolution/Correlation modes ('same'/'full') in one-off functions.
- [ ] Ensure seamless NumPy integration (dtype handling, `std::span` usage).
- [ ] Investigate memory view / zero-copy support for NumPy arrays.
- [ ] Expose more Plan parameters via getters if useful (e.g., CQT frequencies).
- [ ] Allow pre-allocated output arrays in convenience functions (e.g., `api.py::convolve`).

## Build System & CI

- **CMake Refactoring:**
  - **Eliminate C++ `#ifdef`s:**
    - **Strategy:** Rely _solely_ on CMake for managing platform/backend variations.
    - **Conditional Source Files:** Ensure backend/platform-specific `.cpp` files are added conditionally via generator expressions in `cmake/target_definitions.cmake`. Refactor _any_ C++ code using `#ifdef` for conditional compilation into separate files managed by CMake.
    - **Conditional Compile Definitions:** Remove _all_ usage of `#ifdef USE_...`, `#ifdef _WIN32`, etc., for conditional compilation within C++ source. CMake definitions should only be used by CMake itself (generator expressions), except for OS/compiler requirements like `_USE_MATH_DEFINES`.
    - **Conditional Linking:** Verify libraries are linked conditionally based on logic in `cmake/backend/*.cmake`.
- **CI Configuration:**
  - [ ] Configure full CI via GitHub Actions (or similar) for Linux, macOS, and Windows.
  - [ ] Include builds for all available backends.
  - [ ] Integrate C++ and Python test execution.
  - [ ] Add code coverage checks (ensure CMake flags don't hide code).
- **Packaging:**
  - [ ] Investigate static linking options for dependencies.
  - [ ] Create Conan package for the C++ library.
  - [ ] Build Python wheels for distribution on PyPI (handle backend selection/inclusion robustly).

## Testing

- [ ] Add specific tests for Convolution/Correlation modes ('same'/'full') in one-off functions.
- [ ] Add tests for Filter module (once implemented).
- [ ] Add tests for STFT module (once implemented).
- [ ] Increase parameter coverage for existing tests (edge cases, different lengths, etc.).
- [ ] Add backend-specific tests to verify optimized implementations against the default/reference.
- [ ] Test inherited methods for optimized backends (ensure fallback works).
- [ ] Formalize and run a performance benchmarking suite across backends.

## Documentation & Examples

- [ ] Update/Generate Doxygen documentation for the latest C++ API (including Filter module, ResampleSpec).
- [ ] Update/Generate Sphinx documentation for the Python API (including Filter module, ResampleSpec).
- [ ] Add Doxygen comments to `.cpp` implementation files. _(Partially done)_
- [ ] Add C++ and Python usage examples for the Filter module.
- [ ] Add C++ and Python usage examples for the refactored Resample module.
- [ ] Update README and CONTRIBUTING.md if any procedures changed significantly.
- [ ] Add backend-specific performance notes or usage guidelines to documentation.

## Backend Enhancements / Future Work

- [ ] Refine backend selection logic (`OmniDSP::create`) and error handling/reporting.
- [ ] Add `OmniDSP::get_available_backends()` method.
- [ ] Implement Kaiser/Flattop window generation in `AccelerateOmniDSPImpl` (if feasible).
- [ ] Implement window generation methods in `OneMKLOmniDSPImpl` using VML.
- [ ] Add support for GPU acceleration (CUDA/OpenCL) via optional backends.
- [ ] Explore asynchronous execution model (e.g., `Submission` object pattern).
