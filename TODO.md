# OmniDSP TODO List (Updated April 16, 2025)

Reflects successful C++ test refactoring and CI setup for macOS. Focus is now on fixing the Accelerate backend build, verifying its behavior via CI, deciding the MKL resampling strategy, and then addressing remaining test failures and features.

**I. Core Implementation Tasks (in `src/omnidsp/`)**

- **`accelerate/convolution.cpp`:**

  - **Fix Build Error:** Resolve `use of undeclared identifier 'vDSP_fir_f'` and `vDSP_fir_d` errors. Investigate function names (`vDSP_fir_f`/`vDSP_fir_d` vs `vDSP_conv`/`vDSP_desamp`?) and ensure required headers (`<Accelerate/vDSP.h>`) are included. **(Highest Priority - Blocker)**

- **`accelerate/resample.cpp`:**

  - **(Blocked by Build Fix)** **Verify Behavior:** Once the Accelerate backend builds, use CI runs to confirm the output size and behavior of `vDSP_desamp` (used in `filterAndDownsampleBy2`), particularly checking if it aligns with the 'valid' convolution + downsampling logic used in reference generation.

- **`onemkl/resample.cpp`:**

  - **(Blocked by Accelerate Verification)** **Decide Resampling Strategy:** Based on the verified behavior of `vDSP_desamp` from CI:
    - **If Accelerate matches reference (e.g., output size 974):** Prioritize implementing **Option B** for MKL - use alternative IPP functions (e.g., `ippsFIRSR` + manual downsampling) that accept external coefficients and aim for functional parity.
    - **If Accelerate behavior differs significantly:** Re-evaluate if **Option A** (conforming C++ code/tests to `ippsResamplePolyphaseFixed` behavior) is necessary, or if adapting both backends to a common behavior is feasible.
  - **(Blocked by Strategy Decision)** **Implement MKL Resampling:** Code the chosen strategy (Option A or B).
  - **(Blocked by Implementation)** **Implement `double` Precision:** Add double-precision support for the chosen MKL strategy.
  - **(If Option A)** **Tune IPP Parameters:** Tune `ippsResamplePolyphaseFixed` parameters if used.

- **`cqt.cpp` - `CQTPlan::calculateSingleOctaveCQT`:**

  - **Tune Final Scaling Factor:** Adjust the CQT coefficient scaling factor for `float` precision path to match Librosa conventions. **(High Priority - after build fixes)**

- **`convolution.cpp` / `backend/...`:**

  - **Add 'same'/'full' Modes:** Implement support for 'same' and 'full' output modes for `convolve1d` and `correlate1d`. **(Medium Priority)**

- **Error Handling:**

  - **Improve Error Propagation:** Enhance error reporting from C++ backends to Python. **(Medium Priority)**

- **Filter Design Functions:**
  - Implement filter design functions declared in `include/OmniDSP/filter.h`. Create `src/omnidsp/filter.cpp`. **(Low Priority)**

**II. Unit Testing Tasks (in `tests/cpp/`)**

- **Investigate Test Failures (High Priority - after build fixed & backends aligned):**
  - **`FilterAndDownsample` Test:** Verify output _values_ against reference data once the resampling strategy (Task I) is finalized for both backends and the output size discrepancy is resolved or accepted. _(Size mismatch expected until MKL strategy decided based on Accelerate verification)_. Re-enable double precision checks.
  - **`FullRecursiveCQT_Execute_vs_Librosa` Test:** Debug float execution errors or shape/value mismatches vs Librosa reference, especially after CQT scaling (Task I) is tuned.
  - **`WindowTest.Kaiser` Test:** Determine cause of failure (IPP vs manual calculation). Adjust tolerance if needed.
  - **`WindowTest.EdgeCases` Test:** Determine which edge case failed and debug.
- **RESOLVED:** `StubBackend_PublicApiThrowsError` Test: Resolved by refactoring tests into `tests/cpp/backend/` and using conditional compilation in CMakeLists.txt.
- **Add Conv/Corr Mode Tests:** Add C++ tests specifically for 'same' and 'full' modes once implemented (Task I). **(Medium Priority)**
- **Add Window Backend Tests:** Ensure `tests/cpp/window.cpp` correctly tests the abstracted API. **(Medium Priority)**

**III. Build System & Integration**

- **Continuous Integration (CI):**
  - **Setup:** Initial macOS CI workflow (`macos_test.yml`) using GitHub Actions is set up (using explicit lock files). Primary motivation was Accelerate backend verification.
  - **Current Status:** CI build currently **FAILS** due to compilation errors in the Accelerate backend (Task I `accelerate/convolution.cpp`). **(Fixing this is Highest Priority)**
  - **Future:** Add Linux/Windows runners once basic macOS build passes.
- **Dependency Management:** Using `conda-lock` with explicit, platform-specific lock files (`conda-*.conda.lock`) generated via `conda-lock lock --kind explicit`. Contributors need to follow process in `CONTRIBUTING.md`.
- **Code Style / Linting:** Integrate formatters/linters. **(Low Priority)**

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
- **Documentation:** Backend details, CQT details, API docs.

**Summary of Next Steps (Prioritized):**

1.  **FIX BUILD:** Resolve Accelerate backend compile errors (`vDSP_fir_f`/`vDSP_fir_d` in `accelerate/convolution.cpp`).
2.  **VERIFY ACCELERATE VIA CI:** Confirm build passes and verify `vDSP_desamp` behavior (especially output size) on macOS CI runner.
3.  **DECIDE MKL RESAMPLING STRATEGY:** Based on verified Accelerate behavior, choose Option A or B for MKL resampling and document the decision.
4.  **IMPLEMENT MKL RESAMPLING:** Code the chosen strategy.
5.  **ADDRESS CQT SCALING:** Tune the scaling factor in `cqt.cpp`.
6.  **DEBUG TESTS:** Fix remaining C++ test failures (`FilterAndDownsample`, `FullRecursiveCQT_Execute_vs_Librosa`, `WindowTest.*`).
7.  Proceed with other medium/low priority tasks.
