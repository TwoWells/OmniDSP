# OmniDSP TODO List

## Highest Priority / Next Up

- [ ] **Refactor C++ Test Reference Data Handling:** Replace the current \`TestDataUtils\` parser with a modular system using plain text files per test case. (Highest Priority - Blocker for backend verification & other test debugging)
  - [x] Create new directory structure: `tests/cpp/data/<Suite>/`, `tests/cpp/scripts/`, `tests/cpp/tests/`.
  - [x] Organize data files under `data/<Suite>/<TestCase>_<Purpose>_<Type>.txt` (where `<TestCase>` matches the GoogleTest name).
    - `<Purpose>`: Indicates if data is `input` or `expected` for the test.
    - `<Type>`: Specifies data type/precision (e.g., `d`\=double, `f`\=float, `cd`\=complex-double, `cf`\=complex-float).
  - [x] Place Python generation scripts under `scripts/<Suite>.py`.
  - [x] Place C++ test files under `tests/<Suite>.cpp`.
  - [x] Update Python scripts (`scripts/*.py`) to generate data in the new structure/format.
  - [x] Update `tests/cpp/CMakeLists.txt`: Define `OMNIDSP_TEST_DATA_DIR`, adjust sources for `tests/` subdir, ensure Python scripts run before tests.
  - [x] Implement new C++ `TestDataLoader.h/.cpp` utility.
  - [x] Update all C++ tests (`tests/*.cpp`) to use `TestDataLoader`.
  - [x] Remove old `TestDataUtils.h/.cpp` and `test_references.txt`.
  - [x] Update `CONTRIBUTING.md` to reflect new testing suite.
- [ ] **Tune Final Scaling Factor (CQT):** Adjust CQT coefficient scaling in `cqt.cpp` (float path) to match Librosa. (High Priority - after test refactor)

## I. Core Implementation Tasks (in `src/omnidsp/`)

- **`accelerate/resample.cpp`**:
  - (Blocked by Test Refactor & CI Verification) **Verify Behavior:** Use CI runs to confirm `vDSP_desamp` output size/behavior vs reference generation logic.
- **`onemkl/resample.cpp`**:
  - (Blocked by Accelerate Verification) **Decide Resampling Strategy:** Choose Option A (conform to IPP) or B (IPP alternative) based on verified Accelerate behavior.
  - (Blocked by Strategy Decision) **Implement MKL Resampling:** Code the chosen strategy.
  - (Blocked by Implementation) **Implement `double` Precision:** Add double-precision support.
  - (If Option A) **Tune IPP Parameters:** Tune `ippsResamplePolyphaseFixed` parameters if used.
- **`convolution.cpp` / `backend/...`**:
  - [ ] **Add 'same'/'full' Modes:** Implement for `convolve1d`/`correlate1d`. (Medium Priority)
- **Error Handling:**
  - [ ] **Improve Error Propagation:** Enhance error reporting from C++ backends to Python. (Medium Priority)
- **Filter Design Functions:**
  - [ ] Implement functions declared in `include/OmniDSP/filter.h`. Create `src/omnidsp/filter.cpp`. (Low Priority)

## II. Unit Testing Tasks (in `tests/cpp/`)

- **Investigate Test Failures (Blocked by Test Refactor):**
  - [ ] **`FFT_Test` Failures:** Debug FFT test failures after refactoring is complete.
  - [ ] **`FilterAndDownsample` Test:** Verify output _values_ against new reference data once resampling strategy (Task I) is finalized.
  - [ ] **`FullRecursiveCQT_Execute` Test:** Debug float execution errors or mismatches vs Librosa reference, especially after CQT scaling (Task I) is tuned.
  - [ ] **`WindowTest.Kaiser` Test:** Re-check after refactor. Determine cause (IPP vs manual calc). Adjust tolerance if needed.
  - [ ] **`WindowTest.EdgeCases` Test:** Re-check after refactor. Determine which edge case failed and debug.
- [ ] **Add Conv/Corr Mode Tests:** Add C++ tests for 'same'/'full' modes once implemented (Task I). (Medium Priority)

## III. Build System & Integration

- **Continuous Integration (CI):**
  - **Current Status:** CI build PASSES, but C++ tests FAIL (Accelerate backend). (Blocked by Test Refactor)
  - **Future:** Add Linux/Windows runners once macOS build passes _and tests pass_ post-refactor.

## IV. Tuning and Verification (Post-Implementation Fixes)

- [ ] **Parameter Tuning:** Revisit CQT `sparsity_threshold_` and scaling factor. (Medium Priority)
- [ ] **Reference Comparison:** Thoroughly compare CQT (float) against Librosa after scaling is corrected. (Medium Priority)
- [ ] **Formalize Benchmarking Suite:** Develop benchmarks. (Medium Priority)

## V. Python Bindings (`src/omnidsp_py`)

- [ ] **Add Conv/Corr Wrappers:** Create Python wrappers in `api.py`. (Medium Priority)
- [ ] **Expose More Plan Parameters:** (Low Priority)
- [ ] **Python CQT Tests:** Add Python tests comparing vs Librosa. (Low Priority)
- [ ] **Pre-allocated Output Option:** (Low Priority)

## VI. Additional Recommendations (Lower Priority)

- [ ] Handle Kaiser/Flattop Windows in Accelerate Backend. (Low Priority)
- [ ] Build System (`CMakeLists.txt`): Static linking investigation.
- [ ] CQT Implementation (`src/omnidsp/cqt.cpp`): Optimization (Nk precalc), Memory analysis.
- [ ] Backend (`src/omnidsp/backend/onemkl.cpp`): IPP Convolution alternative.
- [ ] Testing (`tests/cpp/cqt.cpp`): More CQT signals.
- [ ] API Query: Backend query function.
- **Documentation:**
  - [ ] Add backend details, CQT details, API docs.

## Summary of Next Steps (Prioritized):

1.  **REFACTOR C++ TEST DATA HANDLING:** Implement the new text-file based system.
2.  **DEBUG C++ TESTS (Post-Refactor):** Fix failures observed after the refactor is complete (FFT, CQT, Window, Resample tests).
3.  **VERIFY ACCELERATE VIA CI:** Confirm Accelerate build passes _and tests pass_ on macOS CI runner. Verify `vDSP_desamp` behavior.
4.  **DECIDE MKL RESAMPLING STRATEGY:** Based on verified Accelerate behavior, choose Option A or B for MKL.
5.  **IMPLEMENT MKL RESAMPLING:** Code the chosen strategy.
6.  **ADDRESS CQT SCALING:** Tune the scaling factor in `cqt.cpp`.
7.  Proceed with other medium/low priority tasks.
