# OmniDSP TODO List (Updated April 18, 2025)

Reflects successful C++ test refactoring to use a central data utility (`TestDataUtils`) and reference data (`test_references.txt`). Accelerate backend build errors are **RESOLVED**. CI setup for macOS is done. Focus is now on debugging **new FFT test failures** observed post-refactor on both backends, then verifying Accelerate behavior via CI, deciding the MKL resampling strategy, and addressing remaining test failures and features.

**I. Core Implementation Tasks (in `src/omnidsp/`)**

- **`accelerate/convolution.cpp`:**
  - **RESOLVED:** Fix Build Error: Resolved `use of undeclared identifier 'vDSP_fir_f'` and `vDSP_fir_d` errors.
- **`accelerate/resample.cpp`:**
  - **(Blocked by Test Fixes & CI)** **Verify Behavior:** Once tests pass, use CI runs to confirm the output size and behavior of `vDSP_desamp` (used in `filterAndDownsampleBy2`), particularly checking if it aligns with the 'valid' convolution + downsampling logic used in reference generation.
- **`onemkl/resample.cpp`:**
  - **(Blocked by Accelerate Verification)** **Decide Resampling Strategy:** Based on the verified behavior of `vDSP_desamp` from CI:
    - **If Accelerate matches reference (e.g., output size 974):** Prioritize implementing **Option B** for MKL - use alternative IPP functions (e.g., `ippsFIRSR` + manual downsampling) that accept external coefficients and aim for functional parity.
    - **If Accelerate behavior differs significantly:** Re-evaluate if **Option A** (conforming C++ code/tests to `ippsResamplePolyphaseFixed` behavior) is necessary, or if adapting both backends to a common behavior is feasible.
  - **(Blocked by Strategy Decision)** **Implement MKL Resampling:** Code the chosen strategy (Option A or B).
  - **(Blocked by Implementation)** **Implement `double` Precision:** Add double-precision support for the chosen MKL strategy.
  - **(If Option A)** **Tune IPP Parameters:** Tune `ippsResamplePolyphaseFixed` parameters if used.
- **`cqt.cpp` - `CQTPlan::calculateSingleOctaveCQT`:**
  - **Tune Final Scaling Factor:** Adjust the CQT coefficient scaling factor for `float` precision path to match Librosa conventions. **(High Priority - after test fixes)**
- **`convolution.cpp` / `backend/...`:**
  - **Add 'same'/'full' Modes:** Implement support for 'same' and 'full' output modes for `convolve1d` and `correlate1d`. **(Medium Priority)**
- **Error Handling:**
  - **Improve Error Propagation:** Enhance error reporting from C++ backends to Python. **(Medium Priority)**
- **Filter Design Functions:**
  - Implement filter design functions declared in `include/OmniDSP/filter.h`. Create `src/omnidsp/filter.cpp`. **(Low Priority)**

**II. Unit Testing Tasks (in `tests/cpp/`)**

- **RESOLVED:** Refactor C++ tests to use central `TestDataUtils` and load data from `test_references.txt`. (Includes updating `cqt.cpp`, `fft.cpp`, `window.cpp`, `test_data_utils.h/cpp`, `generate_references.py`, `CMakeLists.txt`).
- **RESOLVED:** Consolidate `ifft`, `rfft`, `irfft` tests into `fft.cpp`; remove redundant files.
- **RESOLVED:** Expand `test_references.txt` to include comprehensive FFT data (all types, norms, precisions) and float window data.
- **RESOLVED:** Add Window Backend Tests: The refactored `tests/cpp/window.cpp` now tests the abstracted API against reference data.
- **Investigate Test Failures (High Priority):**
  - **NEW:** **`FFT_Test` Failures:** Debug FFT test failures (`tests/cpp/fft.cpp`) observed after refactoring on both Intel (oneMKL) and Accelerate backends. Compare output against `test_references.txt` values. **(Highest Priority - Blocker for backend verification)**
  - **`FilterAndDownsample` Test:** Verify output _values_ against reference data once the resampling strategy (Task I) is finalized for both backends and the output size discrepancy is resolved or accepted. _(Size mismatch expected until MKL strategy decided based on Accelerate verification)_. Re-enable double precision checks if applicable after MKL implementation.
  - **`FullRecursiveCQT_Execute` Test:** Debug float execution errors or shape/value mismatches vs Librosa reference, especially after CQT scaling (Task I) is tuned. (Note: Test was renamed from `...vs_Librosa` as direct comparison only feasible for double currently).
  - **`WindowTest.Kaiser` Test:** Determine cause of failure (IPP vs manual calculation). Adjust tolerance if needed. _(May be resolved by test refactoring, needs re-checking)_.
  - **`WindowTest.EdgeCases` Test:** Determine which edge case failed and debug. _(May be resolved by test refactoring, needs re-checking)_.
- **Add Conv/Corr Mode Tests:** Add C++ tests specifically for 'same' and 'full' modes once implemented (Task I). **(Medium Priority)**

**III. Build System & Integration**

- **Continuous Integration (CI):**
  - **Setup:** Initial macOS CI workflow (`macos_test.yml`) using GitHub Actions is set up (using explicit lock files). Primary motivation was Accelerate backend verification. **(DONE)**
  - **Current Status:** CI build **PASSES**, but C++ tests (specifically FFT tests) are **FAILING** on macOS (Accelerate backend). FFT tests also reported failing locally (Intel backend). **(Debugging tests is Highest Priority)**
  - **Future:** Add Linux/Windows runners once basic macOS build passes _and tests pass_.
- **Dependency Management:** Using `conda-lock` with explicit, platform-specific lock files (`conda-*.lock`) generated via `conda-lock lock --kind explicit`. Contributors need to follow process in `CONTRIBUTING.md`. **(Setup DONE)**
- **Code Style / Linting:** Integrate formatters/linters via `pre-commit`. **(Setup DONE)**

**IV. Tuning and Verification (Post-Implementation Fixes)**

- **Parameter Tuning:** Revisit CQT `sparsity_threshold_` and scaling factor. **(Medium Priority)**
- **Reference Comparison:** Thoroughly compare CQT (float) against Librosa after scaling is corrected. **(Medium Priority)**
- **Formalize Benchmarking Suite:** Develop benchmarks. **(Medium Priority)**

**V. Python Bindings (`src/omnidsp_py`)**

- **Add Conv/Corr Wrappers:** Create Python wrappers in `api.py`. **(Medium Priority)**
- **Expose More Plan Parameters:** (Low Priority)
- **Python CQT Tests:** Add Python tests comparing vs Librosa. **(Low Priority)**
- **Pre-allocated Output Option:** (Low Priority)

**VI. Additional Recommendations (Lower Priority)**

- **Handle Kaiser/Flattop Windows in Accelerate Backend:** Verify manual calculations or consider alternatives. **(Low Priority)**
- **Build System (`CMakeLists.txt`):** Static linking investigation.
- **CQT Implementation (`src/omnidsp/cqt.cpp`):** Optimization (Nk precalc), Memory analysis.
- **Backend (`src/omnidsp/backend/onemkl.cpp`):** IPP Convolution alternative.
- **Testing (`tests/cpp/cqt.cpp`):** More CQT signals.
- **API Query:** Backend query function.
- **Documentation:**
  - `CONTRIBUTING.md` updated for testing backend. **(DONE)**
  - Add backend details, CQT details, API docs.

**Summary of Next Steps (Prioritized):**

1.  **DEBUG FFT TESTS:** Fix `FFT_Test` failures in `tests/cpp/fft.cpp` reported on both backends after the refactor. Compare output values against `test_references.txt`.
2.  **VERIFY ACCELERATE VIA CI:** Once tests pass, confirm Accelerate build passes _and tests pass_ on macOS CI runner. Verify `vDSP_desamp` behavior (especially output size).
3.  **DECIDE MKL RESAMPLING STRATEGY:** Based on verified Accelerate behavior, choose Option A or B for MKL resampling and document the decision.
4.  **IMPLEMENT MKL RESAMPLING:** Code the chosen strategy.
5.  **ADDRESS CQT SCALING:** Tune the scaling factor in `cqt.cpp`.
6.  **DEBUG OTHER TESTS:** Fix remaining C++ test failures (`FilterAndDownsample`, `FullRecursiveCQT_Execute`, `WindowTest.*`).
7.  Proceed with other medium/low priority tasks.
