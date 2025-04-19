# OmniDSP TODO List (Restructured)

## High Priority / Blocking Tasks

- [ ] **Implement Core Resampling Refactor:** Execute the main steps outlined in the "Generic Internal CQT Resampling" plan. This is a **BLOCKER** for CQT functionality and potentially other resampling-dependent features. _(New - High Priority)_
  - [ ] **Remove Old Resample API:** Delete old `resample.h`, `resample.cpp` (core and backend), remove includes, and update/remove related bindings for `filter_and_downsample`.
  - [ ] **Define WindowSpec API (Header):**
    - [ ] Add `WindowType` enum to `core_types.h` or `window.h`.
    - [ ] Add `WindowSpec` struct (without defaults) to `window.h`.
  - [ ] **Implement Window Utilities (Source):**
    - [ ] Implement `get_window_coeffs` utility function in `window.cpp`.
    - [ ] Add necessary template instantiations for `get_window_coeffs`.
  - [ ] **Define ResamplePlan API & Convenience Funcs (Header):**
    - [ ] Create/Modify `resample.h`.
    - [ ] Declare concrete `ResamplePlan<T>` class (constructor taking factor, coefficients, precision; `resample` method).
    - [ ] Declare overloaded `resample<T>` convenience functions (taking coefficients or `WindowSpec`+length).
  - [ ] **Implement ResamplePlan & Convenience Funcs (Source):**
    - [ ] Create/Modify `resample.cpp`.
    - [ ] Define `ResamplePlan<T>` public methods.
    - [ ] Define overloaded `resample<T>` convenience functions.
    - [ ] Add explicit template instantiations for `ResamplePlan` and `resample`.
  - [ ] **Implement Backend ResamplePlanImpl:**
    - [ ] Declare backend-specific `ResamplePlanImpl` forward declarations in `backend.h`.
    - [ ] Implement `onemkl/ResamplePlanImpl<float/double>` using `ippsFIRMR` in `onemkl/resample.cpp`.
    - [ ] Implement `accelerate/ResamplePlanImpl<float/double>` using `vDSP_desamp` in `accelerate/resample.cpp`.
    - [ ] Implement `stub/ResamplePlanImpl<float/double>` (constructor stores coefficients, execute throws) in `stub/resample.cpp`.
- [ ] **Modify CQTPlan Implementation:** Update CQT to use the new `ResamplePlan` and `WindowSpec`. _(Modified - High Priority, depends on Core Resampling Refactor)_
  - [ ] Update `cqt.h`: Constructors accept optional `WindowSpec(s)`, add `FFTPlan`/`ResamplePlan` members, rename `execute` to `cqt`.
  - [ ] Update `cqt.cpp`: Implement constructors (apply default `WindowSpec` if needed), update `initializePlan` to generate coeffs and create plans, rename method, modify internal downsampling call.
  - [ ] Update `cqt` convenience function (header/source) to handle optional/default `WindowSpec`.
- [ ] **Update Python Bindings & API:** Reflect resampling and CQT changes in Python layer. _(Modified - High Priority, depends on Core Resampling Refactor & CQTPlan Modification)_
  - [ ] Update `bindings.cpp`: Bind `WindowType`, `WindowSpec`; update `CQTPlan` bindings (constructor, rename method); add `ResamplePlan`/`resample` bindings; remove old symbols.
  - [ ] Update `api.py`: Update `create_cqt_plan` (handle default `WindowSpec`), `cqt` wrapper; add `create_resample_plan`, `resample` wrappers; remove old wrappers.
- [ ] **Update & Add Unit Tests:** Adapt existing tests and add new ones for the refactored components. _(Modified - High Priority, depends on implementation tasks)_
  - [ ] Modify C++ CQT tests (`tests/cpp/tests/cqt.cpp`) for new constructor, test default/explicit `WindowSpec`, update method calls (`plan->cqt()`).
  - [ ] Add C++ tests for `ResamplePlan` and `resample` convenience functions (coefficient and `WindowSpec` based).
  - [ ] Verify MKL double CQT behavior with the new resampling.
  - [ ] Adjust/regenerate CQT reference data if necessary due to filter changes.
  - [ ] Modify Python tests (`tests/python`) for CQT and add tests for new resampling API. _(Implied by binding changes)_
- [ ] **Debug C++ Tests (Post-Refactor & Resampling Changes):** Fix all remaining test failures observed after the refactoring _and_ resampling/CQT changes are complete. _(Modified - High Priority)_
- [ ] **Tune CQT Scaling Factor:** Adjust CQT coefficient scaling in `cqt.cpp` (float path) to potentially match Librosa or achieve desired normalization. _(Original - High Priority - depends on passing CQT tests)_
- [ ] **Verify Accelerate Backend via CI:** Confirm Accelerate build passes _and C++ tests pass_ on macOS CI runner. Verify `vDSP_desamp` behavior via `ResamplePlanImpl` tests. _(Modified - High Priority)_

## Core Implementation & Backend Tasks

- **CQT (`cqt.cpp`)**
  - [ ] **CQT Implementation Optimization:** Investigate `Nk` precalculation, analyze memory usage. _(Original - Low Priority)_
- **Convolution (`convolution.cpp`, `backend/...`)**
  - [ ] **Implement 'same' Mode:** Add `'same'` mode support for `convolve1d`/`correlate1d`. _(Original - Medium Priority)_
  - [ ] **Implement 'full' Mode:** Add `'full'` mode support for `convolve1d`/`correlate1d`. _(Original - Medium Priority)_
  - [ ] **Investigate MKL IPP Convolution Alternative:** Explore other IPP convolution/correlation functions if `ippsConvolve` has limitations. _(Original - Low Priority)_
- **Windowing (`accelerate/window.cpp`)**
  - [ ] **Implement Accelerate Kaiser Window:** Add `kaiser_window_impl` using Accelerate/vDSP if possible. _(Original - Low Priority - may allow removing Boost dep)_
  - [ ] **Implement Accelerate Flattop Window:** Add `flattop_window_impl` using Accelerate/vDSP if possible. _(Original - Low Priority)_
- **Filter Design (`filter.h`, `filter.cpp`)**
  - [ ] **Implement FIR Filter Design:** e.g., `designLowpassFIR` using window method. _(Original - Low Priority)_
  - [ ] **Implement IIR Filter Design:** e.g., `designButterworth`. Create `src/omnidsp/filter.cpp`. _(Original - Low Priority)_
- **Error Handling & API**
  - [ ] **Improve Error Propagation:** Enhance error reporting from C++ backends to Python. _(Original - Medium Priority)_
  - [ ] **Backend Query API:** Add function to query the active backend at runtime. _(Original - Low Priority)_
- **New Algorithms**
  - [ ] **STFT Implementation:** Consider adding Short-Time Fourier Transform (STFT) functionality. _(Original - Low Priority)_

## Testing Tasks (`tests/...`)

- **C++ Tests (`tests/cpp`)**
  - [ ] **Add Conv/Corr Mode Tests:** Add C++ tests for 'same'/'full' modes once implemented. _(Becomes Medium Priority once modes implemented)_
  - [ ] **Increase Parameter Coverage:** Add tests for FFT/CQT/Window using different parameters (sizes, betas, bins, etc.). _(Original - Medium Priority)_
  - [ ] **Backend-Specific Tests:** Add tests targeting edge cases or known differences between MKL and Accelerate backends (beyond resampling handled by `ResamplePlanImpl` tests). _(Original - Medium Priority)_
  - [ ] **Add More CQT Signals:** Test CQT with different input signals (e.g., noise, multiple sinusoids) after refactor. _(Original - Low Priority)_
- **Python Tests (`tests/python`)**
  - [ ] **Add Python CQT Tests:** Add tests comparing `omnidsp_py.create_cqt_plan().cqt()` against Librosa reference more rigorously. _(Modified naming, still Medium Priority post-refactor)_
  - [ ] **Add Python FFT Tests:** Add tests comparing `omnidsp_py.fft`, `rfft`, etc. against `scipy.fft`. _(Original - Medium Priority)_
  - [ ] **Add Python Window Tests:** Add tests comparing `omnidsp_py.Window` methods against `scipy.signal.get_window`. _(Original - Medium Priority)_
- **Benchmarking**
  - [ ] **Formalize Benchmarking Suite:** Develop and integrate performance benchmarks comparing backends for key operations (FFT, CQT, Resample). _(Modified - Medium Priority)_

## Python Bindings (`src/omnidsp_py`)

- [ ] **Add Conv/Corr Wrappers:** Create Python wrappers in `api.py` for `convolve1d`/`correlate1d`, handling modes. _(Original - Medium Priority)_
- [ ] **Memory View Support:** Investigate using `py::buffer_protocol` to allow zero-copy operations on NumPy arrays where feasible. _(Original - Low Priority)_
- [ ] **Expose Plan Execution API:** Decide if Python API should expose `plan.fft()`, `plan.ifft()`, `plan.resample()`, etc., directly in addition to factory/convenience functions. _(Modified - Low Priority)_
- [ ] **Expose More Plan Parameters:** Add getters for more internal plan parameters (`CQTPlan`, `FFTPlan`, `ResamplePlan`) if useful for debugging/introspection. _(Modified - Low Priority)_
- [ ] **Pre-allocated Output Option:** Allow passing pre-allocated NumPy arrays to functions like `fft`, `resample` to avoid internal allocation. _(Modified - Low Priority)_

## Build System & CI (`CMakeLists.txt`, `.github/workflows`)

- **Continuous Integration (CI):**
  - [ ] **Add Linux Runner:** Add Ubuntu CI runner once macOS build & tests pass. _(Original - Medium Priority)_
  - [ ] **Add Windows Runner:** Add Windows CI runner once macOS build & tests pass. _(Original - Medium Priority)_
- **Build System (`CMakeLists.txt`)**
  - [ ] **Update Build System for Refactor:** Ensure new/modified files (resample.h/cpp, backend resample.cpp, etc.) are correctly included in CMakeLists.txt files. _(New - Medium Priority, part of refactor)_
  - [ ] **Static Linking Investigation:** Explore options and add CMake flag for building OmniDSP as a static library. _(Original - Low Priority)_
  - [ ] **Add More CMake Options:** Consider options for disabling CQT, forcing MKL linking mode, etc. _(Original - Low Priority)_

## Documentation & Examples

- [ ] **API Documentation:** Generate Doxygen (C++) & Sphinx (Python) documentation, updating for new APIs (`WindowSpec`, `ResamplePlan`, modified `CQTPlan`, etc.). _(Modified - Medium Priority)_
- [ ] **Tutorials/Examples:** Add/update comprehensive examples (C++/Python) for common use cases, including new resampling and CQT configuration. _(Modified - Medium Priority)_
- [ ] **Backend Documentation:** Improve README/docs with details on backend differences (e.g., `ippsFIRMR` vs `vDSP_desamp` notes), performance notes, and limitations. _(Modified - Medium Priority)_
- [ ] **Resampling/WindowSpec Documentation:** Specifically document the `WindowSpec` usage, available window types, the `ResamplePlan` API, and the `resample` convenience functions. _(New - Medium Priority)_

## Summary of Next Steps (Prioritized):

1.  **IMPLEMENT CORE RESAMPLING REFACTOR:** Remove old API, define/implement `WindowSpec`, `get_window_coeffs`, `ResamplePlan`, backend `ResamplePlanImpl`.
2.  **MODIFY CQTPLAN IMPLEMENTATION:** Adapt CQT to use the new resampling mechanism and `WindowSpec`.
3.  **UPDATE PYTHON BINDINGS & API:** Expose the changes to Python.
4.  **UPDATE & ADD UNIT TESTS:** Ensure CQT and Resampling are tested correctly.
5.  **DEBUG C++ TESTS (Post-Refactor & Resampling Changes):** Fix any failures after the major changes.
6.  **VERIFY ACCELERATE VIA CI:** Confirm macOS CI build and tests pass with the refactored code.
7.  **TUNE CQT SCALING:** Adjust scaling factor now that CQT tests should be passing.
8.  Proceed with other medium/low priority tasks (Convolution modes, other backends, documentation, etc.).
