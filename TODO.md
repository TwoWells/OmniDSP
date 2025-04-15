# OmniDSP TODO List (Updated April 15, 2025)

Reflects successful backend code structure refactoring. Focus is now on resolving remaining test failures, MKL resampling strategy, CQT scaling, and CI setup.

**I. Core Implementation Tasks (in `src/omnidsp/`)**

* **`cqt.cpp` - `CQTPlan::calculateSingleOctaveCQT`:**
    * **Tune Final Scaling Factor:** Adjust the CQT coefficient scaling factor. The current implementation (scaling=1.0) leads to magnitude differences compared to Librosa. Investigate Librosa's scaling conventions and adjust the factor in `calculateSingleOctaveCQT` for the `float` precision path to achieve closer results. **(High Priority)**

* **`onemkl.cpp` (or `backend/onemkl/resample.cpp` after refactor) - `filter_and_downsample_impl`:**
    * **Address Filter Coefficient Discrepancy:** **(High Priority)**
        * The current IPP function (`ippsResamplePolyphaseFixed`) does *not* use the explicitly passed `kernel` coefficients; it designs its own internal filter based on `filterLen`, `rolloff`, `alpha`.
        * Decide on the strategy:
            * **Option A (Conform to IPP):** Modify `filter_and_downsample_impl` and its usage (e.g., in `CQTPlan::filterAndDownsampleBy2`) to work with IPP's constraints, passing parameters like `filterLen` instead of explicit coefficients. This prioritizes getting the MKL backend working.
            * **Option B (Find Alternate IPP):** Investigate if other IPP functions (like `ippsFIRSR`) allow using external coefficients for filtering *before* manual downsampling, which would align better with the Accelerate backend (`vDSP_desamp`).
    * **(Blocked by above)** **Implement `double` Precision Resampling:** If the IPP design strategy is resolved, implement the `double` precision version. **(High Priority)**
    * **(Blocked by above)** **Tune IPP Resampling Parameters:** If `ippsResamplePolyphaseFixed...` is used (Strategy A), tune its parameters (`filterLen`, `rolloff`, `alpha`) if defaults are insufficient.

* **`accelerate.cpp` (or `backend/accelerate/resample.cpp` after refactor) - `filter_and_downsample_impl`:**
    * **(Depends on MKL Strategy)** Once the MKL/IPP approach is finalized, implement/verify the Accelerate version using `vDSP_desamp` or an alternative vDSP approach to maintain functional parity.

* **`convolution.cpp` / `backend/...` (or `backend/*/convolution.cpp` after refactor):**
    * **Add 'same'/'full' Modes:** Implement support for 'same' and 'full' output modes for `convolve1d` and `correlate1d`. **(Medium Priority)**

* **Error Handling:**
    * **Improve Error Propagation:** Enhance error reporting from C++ backends to Python. **(Medium Priority)**

* **Filter Design Functions:**
    * Implement filter design functions declared in `include/OmniDSP/filter.h` (e.g., `designLowpassFIR`, `designButterworth`). Create `src/omnidsp/filter.cpp` for implementations. **(Low Priority)**

**II. Unit Testing Tasks (in `tests/cpp/`)**

* **Investigate Test Failures (High Priority):**
    * **`FullRecursiveCQT_Execute_vs_Librosa` Test:** Debug why this test failed. Check for unexpected errors during float execution or shape mismatches vs Librosa reference.
    * **`StubBackend_PublicApiThrowsError` Test:** Determine why this test failed. If MKL/Accelerate was active, it shouldn't have run (check `#if` guard logic/CMake flags). If Stub *was* active, why didn't the public API calls throw the expected `std::runtime_error`?
    * **`WindowTest.Kaiser` Test:** Determine cause of failure. Compare IPP Kaiser output vs manual calculation more closely. Adjust test tolerance if appropriate.
    * **`WindowTest.EdgeCases` Test:** Determine which edge case failed (empty input throw, N=1 output check, negative beta throw). Debug the corresponding public API call and backend implementation.
* **`cqt.cpp` (Test Fixture `PrecomputedRecursiveCQTTest`):**
    * **`FilterAndDownsample` Test:**
        * **Address Size/Value Mismatch:** Investigate why the C++ backend (IPP) output size (1024) differs from the reference calculation ('valid' correlate + downsample = 974). Decide whether to adjust the backend, the reference calculation method, or the test assertion based on desired behavior (linked to Task I MKL Resampling Strategy). *(Failure Expected until resolved)* **(Medium Priority)**
        * **(Blocked by I.onemkl.cpp)** Re-enable checks for `double` precision once the backend implementation is available and working. Verify output values against reference.
    * **`FullRecursiveCQT_Execute_vs_Librosa` Test:**
        * **(Blocked by I.cqt.cpp)** Debug magnitude differences for `float` precision comparison against Librosa reference. Adjust test tolerance (`comparison_tolerance`) if needed *after* CQT scaling factor is corrected.
* **Add Conv/Corr Mode Tests:** Add C++ tests specifically for 'same' and 'full' modes once implemented (Task I.convolution.cpp). **(Medium Priority)**
* **Add Window Backend Tests:** Ensure `tests/cpp/window.cpp` correctly tests the abstracted API and potentially add backend-specific checks if needed. **(Medium Priority)**

**III. Build System & Integration**

* **Setup Continuous Integration (CI):** Implement CI workflows (GitHub Actions). **(High Priority)**
* **CMake IPP Linking:** Linking seems okay, but verify compilation with chosen IPP resampling approach. **(Partially Done)**
* **Verify IPP/Accelerate Runtime:** Once backend functions compile and link, ensure tests run correctly.
* **Code Style / Linting:** Integrate formatters/linters. **(Low Priority)**

**IV. Tuning and Verification (Post-Implementation Fixes)**

* **Parameter Tuning:** Revisit CQT `sparsity_threshold_` and scaling factor. **(Medium Priority)**
* **Reference Comparison:** Thoroughly compare CQT (float) against Librosa after scaling is corrected. **(Medium Priority)**
* **Formalize Benchmarking Suite:** Develop benchmarks. **(Medium Priority)**

**V. Python Bindings (`src/omnidsp_py`)**

* **Add Conv/Corr Wrappers:** Create Python wrappers in `api.py` for the `convolve1d`/`correlate1d` C++ functions. **(Medium Priority)**
* **Expose More Plan Parameters:** (Low Priority)
* **Python CQT Tests:** Add Python tests comparing vs Librosa. **(Low Priority)**
* **Pre-allocated Output Option:** (Low Priority)

**VI. Additional Recommendations (Lower Priority)**

* **Handle Kaiser/Flattop Windows in Accelerate Backend:** Accelerate/vDSP lacks native Kaiser/Flattop functions. The backend implementation (`backend/accelerate/window.cpp`) currently uses manual calculations. Verify these calculations or consider alternatives if needed for performance/accuracy parity with MKL/IPP's Kaiser. **(Low Priority)**
* **Build System (`CMakeLists.txt`):** Static linking investigation.
* **CQT Implementation (`src/omnidsp/cqt.cpp`):** Optimization (Nk precalc), Memory analysis.
* **Backend (`src/omnidsp/backend/onemkl.cpp`):** IPP Convolution alternative.
* **Testing (`tests/cpp/cqt.cpp`):** More CQT signals.
* **API Query:** Backend query function.
* **Documentation:** Backend details, CQT details, API docs.

**Summary of Next Steps (Prioritized):**

1.  **Focus on Task II:** Debug the 4 unexpected test failures.
2.  **Focus on Task I.onemkl.cpp:** Address filter coefficient strategy for resampling.
3.  **Focus on Task I.cqt.cpp:** Determine the correct `scale_factor` for CQT (float precision).
4.  Implement CI (Task III).
5.  Implement `double` precision resampling (Task I.onemkl.cpp) once strategy is decided.
6.  Address the expected `FilterAndDownsample` test failure (Task II) based on resampling strategy.
7.  Proceed with other tasks.