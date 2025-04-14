# OmniDSP TODO List (Updated April 14, 2025)

Here's an updated breakdown of the remaining tasks based on recent progress, including the integration of Intel IPP for resampling and the subsequent compilation issues.

**I. Core Implementation Tasks (in `src/omnidsp/`)**

* **`cqt.cpp` - `CQTPlan::calculateSingleOctaveCQT`:**
    * **Tune Final Scaling Factor:** Adjust the CQT coefficient scaling factor. The previous attempts (`1.0 / fft_len_oct` and `1.0`) resulted in incorrect magnitudes compared to Librosa. Further investigation is needed, potentially involving the per-bin filter length (`Nk`), window normalization details, and comparison with Librosa's scaling conventions. **(Top Priority)**

* **`onemkl.cpp` - `filter_and_downsample_impl`:**
    * **Resolve IPP `2022.1.0` API Compilation Errors:** Fix the persistent `C2065` (undeclared identifier for `IppsResamplingPolyphaseFixedSpec_...`) and `C2660` (argument mismatch for `ippsResamplePolyphaseFixedGetSize_32f`) errors when using the `ippsResamplePolyphaseFixed...` API.
        * **Investigate Header:** Manually inspect the installed `ipps.h` (and related headers like `ippdefs.h`, `ippcore.h`) in the Conda environment (`$CONDA_PREFIX/include/ipp/`) to verify the exact declarations and signatures for the `IppsResamplingPolyphaseFixedSpec_...` types and the `ippsResamplePolyphaseFixedGetSize_32f/64f`, `Init_32f/64f`, and `_32f/64f` functions in the `2022.1.0` version. Note the discrepancy between the compiler error regarding `GetSize` arguments and the official 2022.1 documentation.
        * **Consider Alternatives:** If the "Fixed" API proves unusable with the installed version (due to missing declarations or persistent signature issues), reconsider implementing this function using `ippsFIRSR...` followed by manual decimation (as explored in `onemkl_cpp_fix_3`).
        * **(Optional) Try Different IPP Version:** Consider creating a new Conda environment and installing a slightly newer or older `ipp-devel` version from the Intel channel to see if the API behavior changes.
    * **(Blocked by above)** **Tune IPP Resampling Parameters:** If the `ippsResamplePolyphaseFixed...` API is successfully used, tune its parameters (`filterLen`, `rolloff`) based on test results if the defaults are insufficient. If `ippsFIRSR` is used, ensure the FIR filter design is appropriate for anti-aliasing before downsampling.

* **`convolution.cpp` / `backend/...`:**
    * **Add 'same'/'full' Modes:** Implement support for 'same' and 'full' output modes for `convolve1d` and `correlate1d`, similar to SciPy/NumPy. This likely requires adjusting backend calls or adding padding/trimming logic. **(Medium Priority)**

* **Error Handling:**
    * **Improve Error Propagation:** Enhance error reporting from C++ backends to Python. Propagate more specific error messages (e.g., MKL/IPP status strings) instead of generic runtime errors where possible. **(Medium Priority)**

**II. Unit Testing Tasks (in `tests/cpp/`)**

* **`cqt.cpp` (Test Fixture `PrecomputedRecursiveCQTTest`):**
    * **(Blocked by I.onemkl.cpp)** **Re-evaluate/Debug `FilterAndDownsample` Test:** Once the IPP function call in `onemkl.cpp` is compiling and working correctly (whether using `ResamplePolyphaseFixed` or `FIRSR`), re-evaluate the assertions in this test. Adjust the expected output or tolerance based on the actual filter characteristics achieved by the chosen IPP method.
    * **Debug failing assertions for `FullRecursiveCQT_Execute_vs_Librosa`:** The comparison with the Librosa reference fails due to large magnitude differences. This is directly linked to the scaling factor issue in Task I.cqt.cpp. Resolve the scaling factor first. Adjust test tolerance (`comparison_tolerance`) if needed *after* scaling is corrected. **(High Priority - linked to Task I.cqt.cpp)**
* **Add Conv/Corr Mode Tests:** Add C++ tests specifically for 'same' and 'full' modes once implemented (Task I.convolution.cpp). **(Medium Priority)**

**III. Build System & Integration**

* **CMake IPP Linking:** The system now links IPP libraries (`ippcore`, `ipps`) by name when the MKL backend is selected, relying on `CMAKE_PREFIX_PATH`. `find_package(IPP)` was removed due to unreliability. This seems functional at the CMake configuration level, but C++ compilation using IPP is currently blocked (see Task I.onemkl.cpp). **(Partially Done - Linking Configured, Compilation Blocked)**
* **Verify IPP Runtime:** Once IPP functions compile and link, ensure tests requiring IPP (like the downsampling test) run correctly within the Conda environment (dynamic libraries should be found automatically).
* **Setup Continuous Integration (CI):** Implement CI workflows (e.g., GitHub Actions) to automatically build and run C++ and Python tests on Linux, macOS, and Windows, testing different backend selections (MKL, Accelerate, Stub). **(High Priority)**
* **Code Style / Linting:** Integrate tools like `clang-format` (C++) and `black`/`flake8` (Python) into the build or CI process to enforce consistent code style. **(Low Priority)**

**IV. Tuning and Verification (Post-Compilation Fixes)**

* **Parameter Tuning:** Revisit CQT `sparsity_threshold_` and the final scaling factor (Task I.cqt.cpp) after initial implementation and testing. Adjust based on performance measurements and comparison with reference CQT results. **(Medium Priority)**
* **Reference Comparison:** Thoroughly compare the output of `OmniDSP::CQTPlan::execute` against Librosa's CQT for various signals and parameter settings *after* the scaling factor is corrected and tests pass. Identify and resolve any remaining discrepancies. **(Medium Priority)**
* **Formalize Benchmarking Suite:** Develop a dedicated benchmarking suite (potentially using Google Benchmark for C++ and pytest-benchmark for Python) to measure performance of key functions (FFT, CQT, Conv) across different backends and input sizes. **(Medium Priority)**

**V. Python Bindings (`src/omnidsp_py`)**

* **Add Conv/Corr Wrappers:** Create Python wrappers in `api.py` for the `convolve1d` and `correlate1d` C++ functions, handling NumPy array conversion and potentially the different modes ('valid', 'same', 'full'). **(Medium Priority)**
* **Expose More Plan Parameters:** Consider exposing more underlying plan parameters (e.g., specific FFT plan settings, CQT kernel details if useful) via getters in the Python plan classes or factory functions. **(Low Priority)**
* **Python CQT Tests:** Add Python-level tests in `tests/python/` comparing `omnidsp_py.create_cqt_plan(...).execute(...)` directly against `librosa.cqt`. **(Low Priority)**
* **Pre-allocated Output Option:** Add optional arguments to Python wrappers (and corresponding C++ functions) to allow users to provide pre-allocated NumPy arrays for output, potentially reducing memory allocation overhead in tight loops. **(Low Priority)**

**VI. Additional Recommendations (Lower Priority)**

* **Build System (`CMakeLists.txt`):**
    * *Verify IPP Linking (Static):* If static linking is ever desired, investigate robust configuration methods.
* **CQT Implementation (`src/omnidsp/cqt.cpp`):**
    * *Optimization:* Investigate pre-calculating and storing the per-bin window length (`Nk`) during construction.
    * *Memory:* Analyze memory usage of `precomputed_sparse_kernels_`.
* **Backend (`src/omnidsp/backend/onemkl.cpp`):**
    * *IPP Convolution:* Consider implementing `convolve1d_impl` using IPP FIR functions (`ippsFIR...`) as an alternative to the VSL functions.
* **Testing (`tests/cpp/cqt.cpp`):**
    * *More Signals:* Add CQT tests comparing against Librosa using different input signals.
* **API Query:** Add functions (C++/Python) to query which backend (MKL, Accelerate, Stub) was selected at runtime/build time. **(Low Priority)**
* **Documentation:**
    * *IPP Backend:* Document the IPP functions used (once finalized).
    * *CQT Details:* Expand documentation (README, Doxygen) on implementation, scaling, hop length, etc.
    * *API Docs:* Improve Python API documentation (docstrings, potentially Sphinx).

**Summary of Next Steps:**

1.  **Focus on Task I.onemkl.cpp:** Resolve the IPP API compilation errors (`C2065`/`C2660`).
2.  **Focus on Task I.cqt.cpp:** Determine the correct `scale_factor` for CQT.
3.  Once compilation and scaling are fixed, re-run C++ tests and address failures (Task II).
4.  Implement CI (Task III).
5.  Proceed with implementing Conv/Corr modes, improving error handling, and adding Python wrappers (Tasks I, V).
6.  Address tuning, benchmarking, and lower-priority items.
