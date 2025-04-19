# OmniDSP TODO List

## High Priority / Blocking Tasks

- [x] **Refactor C++ Test Reference Data Handling:** Complete the transition to the new text-file based system (ensure all tests use `TestDataLoader` and old system is removed). (Highest Priority - Blocker for backend verification & other test debugging)
- [ ] **Debug C++ Tests (Post-Refactor):** Fix all remaining test failures observed after the refactoring is complete (FFT, CQT, Window tests). (High Priority)
- [ ] **Tune CQT Scaling Factor:** Adjust CQT coefficient scaling in `cqt.cpp` (float path) to potentially match Librosa or achieve desired normalization. (High Priority - depends on passing CQT tests)
- [ ] **Verify Accelerate Backend via CI:** Confirm Accelerate build passes _and C++ tests pass_ on macOS CI runner. Verify `vDSP_desamp` behavior vs reference data. (High Priority)

## Core Implementation & Backend Tasks

- **Resampling (`onemkl/resample.cpp`, `accelerate/resample.cpp`)**
  - [ ] **Decide MKL Resampling Strategy:** (Blocked by Accelerate Verification) Based on verified Accelerate behavior, choose Option A (conform to IPP) or B (IPP alternative) for MKL `float` resampling. (Medium Priority)
  - [ ] **Implement MKL Resampling:** (Blocked by Strategy Decision) Code the chosen strategy for `float`. (Medium Priority)
  - [ ] **Address MKL `filter_and_downsample<double>`:** Implement `double` precision support. (Low Priority)
    - Consider implementing a custom (non-IPP) `double` precision routine if direct IPP support remains unavailable or undesirable.
- **CQT (`cqt.cpp`)**
  - [ ] **Refactor CQT Downsampling:** Modify the recursive CQT implementation to use the public `OmniDSP::filter_and_downsample` function for internal downsampling steps. (Medium Priority)
    - Note: This links the CQT's `double` precision support directly to the availability of `filter_and_downsample<double>`.
  - [ ] **CQT Implementation Optimization:** Investigate `Nk` precalculation, analyze memory usage. (Low Priority)
- **Convolution (`convolution.cpp`, `backend/...`)**
  - [ ] **Implement 'same' Mode:** Add `'same'` mode support for `convolve1d`/`correlate1d`. (Medium Priority)
  - [ ] **Implement 'full' Mode:** Add `'full'` mode support for `convolve1d`/`correlate1d`. (Medium Priority)
  - [ ] **Investigate MKL IPP Convolution Alternative:** Explore other IPP convolution/correlation functions if `ippsConvolve` has limitations. (Low Priority)
- **Windowing (`accelerate/window.cpp`)**
  - [ ] **Implement Accelerate Kaiser Window:** Add `kaiser_window_impl` using Accelerate/vDSP if possible. (Low Priority - may allow removing Boost dep)
  - [ ] **Implement Accelerate Flattop Window:** Add `flattop_window_impl` using Accelerate/vDSP if possible. (Low Priority)
- **Filter Design (`filter.h`, `filter.cpp`)**
  - [ ] **Implement FIR Filter Design:** e.g., `designLowpassFIR` using window method. (Low Priority)
  - [ ] **Implement IIR Filter Design:** e.g., `designButterworth`. Create `src/omnidsp/filter.cpp`. (Low Priority)
- **Error Handling & API**
  - [ ] **Improve Error Propagation:** Enhance error reporting from C++ backends to Python. (Medium Priority)
  - [ ] **Backend Query API:** Add function to query the active backend at runtime. (Low Priority)
- **New Algorithms**
  - [ ] **STFT Implementation:** Consider adding Short-Time Fourier Transform (STFT) functionality. (Low Priority)

## Testing Tasks (`tests/...`)

- **C++ Tests (`tests/cpp`)**
  - [ ] **Add Conv/Corr Mode Tests:** Add C++ tests for 'same'/'full' modes once implemented. (Medium Priority)
  - [ ] **Increase Parameter Coverage:** Add tests for FFT/CQT/Window using different parameters (sizes, betas, bins, etc.). (Medium Priority)
  - [ ] **Backend-Specific Tests:** Add tests targeting edge cases or known differences between MKL and Accelerate backends. (Medium Priority)
  - [ ] **Resampling Value Verification:** Verify `FilterAndDownsample` output _values_ (not just shape/type) against reference data once resampling strategy is finalized. (Medium Priority)
  - [ ] **Add More CQT Signals:** Test CQT with different input signals (e.g., noise, multiple sinusoids). (Low Priority)
- **Python Tests (`tests/python`)**
  - [ ] **Add Python CQT Tests:** Add tests comparing `omnidsp_py.create_cqt_plan().execute()` against Librosa reference more rigorously. (Medium Priority)
  - [ ] **Add Python FFT Tests:** Add tests comparing `omnidsp_py.fft`, `rfft`, etc. against `scipy.fft`. (Medium Priority)
  - [ ] **Add Python Window Tests:** Add tests comparing `omnidsp_py.Window` methods against `scipy.signal.get_window`. (Medium Priority)
- **Benchmarking**
  - [ ] **Formalize Benchmarking Suite:** Develop and integrate performance benchmarks comparing backends for key operations (FFT, CQT). (Medium Priority)

## Python Bindings (`src/omnidsp_py`)

- [ ] **Add Conv/Corr Wrappers:** Create Python wrappers in `api.py` for `convolve1d`/`correlate1d`, handling modes. (Medium Priority)
- [ ] **Memory View Support:** Investigate using `py::buffer_protocol` to allow zero-copy operations on NumPy arrays where feasible. (Low Priority)
- [ ] **Expose Plan Execution API:** Decide if Python API should expose `plan.fft()`, `plan.ifft()`, etc., directly in addition to factory/convenience functions. (Low Priority)
- [ ] **Expose More Plan Parameters:** Add getters for more internal plan parameters if useful for debugging/introspection. (Low Priority)
- [ ] **Pre-allocated Output Option:** Allow passing pre-allocated NumPy arrays to functions like `fft` to avoid internal allocation. (Low Priority)

## Build System & CI (`CMakeLists.txt`, `.github/workflows`)

- **Continuous Integration (CI):**
  - [ ] **Add Linux Runner:** Add Ubuntu CI runner once macOS build & tests pass. (Medium Priority)
  - [ ] **Add Windows Runner:** Add Windows CI runner once macOS build & tests pass. (Medium Priority)
- **Build System (`CMakeLists.txt`)**
  - [ ] **Static Linking Investigation:** Explore options and add CMake flag for building OmniDSP as a static library. (Low Priority)
  - [ ] **Add More CMake Options:** Consider options for disabling CQT, forcing MKL linking mode, etc. (Low Priority)

## Documentation & Examples

- [ ] **API Documentation:** Generate Doxygen (C++) & Sphinx (Python) documentation. (Medium Priority)
- [ ] **Tutorials/Examples:** Add more comprehensive examples (C++/Python) for common use cases. (Medium Priority)
- [ ] **Backend Documentation:** Improve README/docs with details on backend differences, performance notes, and limitations (e.g., MKL double resampling). (Medium Priority)

## Summary of Next Steps (Prioritized):

1.  **REFACTOR C++ TEST DATA HANDLING:** Complete implementation and ensure all tests use it.
2.  **DEBUG C++ TESTS (Post-Refactor):** Fix remaining C++ test failures.
3.  **VERIFY ACCELERATE VIA CI:** Confirm macOS CI build and tests pass. Verify resampling behavior.
4.  **DECIDE MKL RESAMPLING STRATEGY:** Choose approach for `float`.
5.  **IMPLEMENT MKL RESAMPLING:** Code the chosen `float` strategy.
6.  **ADDRESS CQT SCALING:** Tune the scaling factor in `cqt.cpp`.
7.  Proceed with other medium/low priority tasks.
