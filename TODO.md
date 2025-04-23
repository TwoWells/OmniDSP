# OmniDSP TODO List (Updated & Reconciled)

_Based on design_philosophy.md (primary source), existing TODOs, and current code state._

## High Priority / Blocking Tasks

- **Refactor `ResamplePlan` to use `FilterPlan` Infrastructure:**

  - [x] **Define `ResampleSpec`:** Create `ResampleSpec` struct in `include/OmniDSP/resample.h` to hold `input_rate`, `output_rate`, and a `quality` factor (or specific filter design parameters like transition width/stopband attenuation).
  - [x] **Update `create_resample_plan` API:**
    - Modify signature in `src/omnidsp/backend/backend.h` (`OmniDSPImpl`) to accept `const ResampleSpec& spec`.
    - Modify signature in `include/OmniDSP/omnidsp.h` (`OmniDSP`) to accept `const ResampleSpec& spec`.
    - Update calls to this factory method (e.g., in `DefaultCQTPlanImpl` constructor, one-off methods if they use it, tests, examples).
  - [x] **Implement `design_fir_filter` Method:**
    - Add virtual `design_fir_filter(const FIRFilterSpec<T>& spec)` method to `OmniDSPImpl` base class (`src/omnidsp/backend/backend.h`).
    - Implement `design_fir_filter` in `DefaultOmniDSPImpl` (`src/omnidsp/backend/default/backend.cpp`) using a standard technique (e.g., windowed-sinc) to calculate FIR coefficients based on the spec. Handle different `FilterType`s if necessary (though resampler only needs lowpass).
    - Instantiate `design_fir_filter` templates in `default/backend.cpp`.
  - [x] **Refactor `ResamplePlanImpl` Base Class (`src/omnidsp/backend/backend.h`):**
    - Add protected helper method `calculate_factors(double in_rate, double out_rate, size_t& L, size_t& M)` (or similar).
    - Add protected helper method `design_prototype_filter(const OmniDSPImpl* owner, const ResampleSpec& spec, std::vector<T>& coeffs_out)` that determines `FIRFilterSpec` based on `ResampleSpec` and calls `owner->design_fir_filter`.
    - Add `mutable std::vector<T> filter_state_` and `mutable size_t current_phase_` members.
    - Add virtual `reset()` method declaration.
  - [x] **Refactor `DefaultResamplePlanImpl` (`src/omnidsp/backend/default/resample.cpp`):**
    - Update constructor to take `const OmniDSPImpl* owner` and `const ResampleSpec& spec`.
    - Constructor calls `calculate_factors` and `design_prototype_filter` to initialize members (`upsample_factor_L_`, `downsample_factor_M_`, `prototype_filter_coeffs_`).
    - Initialize `filter_state_` based on `prototype_filter_coeffs_` size.
    - Implement `execute` using polyphase FIR logic based on `prototype_filter_coeffs_`, `filter_state_`, `current_phase_`, `L`, and `M`. Use standard C++ loops/math.
    - Implement `reset()` to clear state and phase.
  - [ ] **Refactor `AccelerateResamplePlanImpl` (`src/omnidsp/backend/accelerate/resample.cpp`) (If Applicable):**
    - Update constructor as above.
    - Implement `execute` using Accelerate vDSP primitives for polyphase FIR filtering, using the designed `prototype_filter_coeffs_`.
    - Implement `reset()`.
  - [x] **Refactor `OneMKLResamplePlanImpl` (`src/omnidsp/backend/onemkl/resample.cpp`) (If Applicable):**
    - Update constructor as above.
    - Implement `execute` using oneMKL VML/VSL primitives for polyphase FIR filtering, using the designed `prototype_filter_coeffs_`.
    - Implement `reset()`.
  - [x] **Update `DefaultCQTPlanImpl` (`src/omnidsp/backend/default/cqt.cpp`):**
    - Ensure its constructor calls the _new_ `create_resample_plan` signature via `owner_impl_`, passing a `ResampleSpec` (e.g., with default quality).
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
- [ ] Implement backend-specific CQT logic if beneficial (`AccelerateCQTPlanImpl`, `OneMKLCQTPlanImpl`) or confirm default implementation is sufficient.
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

- [ ] Configure full CI via GitHub Actions (or similar) for Linux, macOS, and Windows.
  - Include builds for all available backends.
  - Integrate C++ and Python test execution.
  - Add code coverage checks (ensure CMake flags don't hide code).
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
