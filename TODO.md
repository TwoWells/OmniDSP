# **OmniDSP Project TODO List (Cleaned)**

**Goal:** Outline the development tasks for the OmniDSP library, including C++ core features, backend implementations, Python bindings, build system refactoring, testing, and documentation, ensuring alignment with the design philosophy and current architecture.

**Directory Structure Reminder:**

- src/omnidsp/interface/: Abstract interfaces (`AbstractBackend`, `*PlanImpl`) in `OmniDSP::abstract` namespace and public Plan wrappers (fft.cpp, cqt.cpp, etc.).
- src/omnidsp/default/: Default backend implementation (`DefaultBackend`, `Default*PlanImpl`, etc.) in `OmniDSP::default` namespace.
- src/omnidsp/{accelerate, onemkl, intelipp, your_backend}/: Specific backend implementations inheriting from `DefaultBackend` in their respective namespaces (e.g., `OmniDSP::accelerate`).
- src/omnidsp/omnidsp.cpp: Public `OmniDSP` class implementation (factory, dispatch).
- src/omnidsp/utils/: Common utilities (resampling factors, filter design helpers).

## **High Priority / Blocking Tasks**

- **Implement Filter Module:** _(Crucial core feature)_
  - \[ \] Implement `IntelIPPFIRFilterPlanImpl` (src/omnidsp/intelipp/filter.cpp) - _(Structure exists, needs IPP implementation)_.
  - \[ \] Implement `IntelIPPIIRFilterPlanImpl` (src/omnidsp/intelipp/filter.cpp) - _(Structure exists, needs IPP implementation)_.
  - \[ \] Implement `create_fir_filter_plan_*` factory methods in `IntelIPPBackend`.
  - \[ \] Implement `create_iir_filter_plan_*` factory methods in `IntelIPPBackend`.
  - \[ \] Implement `design_iir_filter_*` methods in `DefaultBackend` (src/omnidsp/default/filter_design.cpp or similar).
  - \[ \] Add Python bindings for Filter Plans and design methods.
  - \[ \] Add comprehensive C++ and Python tests for the Filter module across all backends _(See Testing section)_.
- **Complete ResamplePlan Infrastructure:**
  - \[ \] **Add/Update Tests:** _(See Testing section)_.
  - \[ \] **Update Python Bindings:**
    - Ensure Python `create_resample_plan` wrapper (src/omnidsp_py/api.py) correctly passes `ResampleSpec` information.
    - Ensure `src/omnidsp_py/bindings.cpp` binds the `create_resample_plan` C++ signature taking `ResampleSpec`.
- **Review Core Implementations:**
  - \[ \] Review C++ implementations (.cpp files) across `interface/`, `default/`, `accelerate/`, `onemkl/`, `intelipp/` against the latest `design_philosophy.md` and `backend_developer_guide.md` to ensure compliance with Pimpl, inheritance (`DefaultBackend`), API contracts, and naming conventions (`abstract::AbstractBackend`, `default::DefaultBackend`, etc.).
  - \[ \] Review error propagation from backends (ensure sufficient detail is captured/logged via `OmniException` or return status).

## **Core Library Features**

### **Default Backend Highway Refactor**

- \[x] **Build System:** Ensure Highway dependency is correctly found/configured in CMake (`cmake/depend/highway.cmake`) and linked to the `DefaultBackend`. _(Completed)_
- \[ \] **Incremental Refactoring:** _(Approach incrementally, benchmark each step)_
  - \[ \] Refactor simpler functions first (e.g., window generation helpers in `default/window.cpp`, basic math within filters).
  - \[ \] Refactor core filtering loops (FIR/IIR in `default/filter.cpp`, polyphase in `default/resample.cpp`).
  - \[ \] Refactor convolution/correlation (`default/convolution.cpp`) if direct methods are used or FFT steps can be optimized.
  - \[ \] Refactor FFT (`default/fft.cpp`) last (potentially complex).
  - \[ \] Review CQT (`default/cqt.cpp`) for optimizable loops.
- \[ \] **Utilities:** Review/update `src/omnidsp/utils/hwy.hpp` with any necessary helper functions for the refactoring.
- \[ \] **Testing:** Add specific tests comparing Highway implementations against non-SIMD results within the Default backend.
- \[ \] **Benchmarking:** Benchmark Highway-enabled Default backend against the pure C++ version and other backends.

### **Intel IPP Backend Enhancements** _(Medium-to-High Priority)_

- \[ \] **Arbitrary Length FFT/DFT:** Modify `IntelIPPFFTPlanImpl` and `IntelIPPRFFTPlanImpl` (`intelipp/fft.cpp`) to use `ippsDFT*` functions when the length is not a power of two, allowing `FFTPlan`/`RFFTPlan` to support arbitrary lengths with this backend.
- \[ \] **DCT Implementation:** Implement `IntelIPPDCTPlanImpl` using IPP DCT routines (`src/omnidsp/intelipp/dct.cpp` - New files). _(See DCT/DST Module)_
- \[ \] **DST Implementation:** Implement `IntelIPPDSTPlanImpl` using IPP RFFT routines (`src/omnidsp/intelipp/dst.cpp` - New files). _(See DCT/DST Module)_
- \[ \] **DWT Implementation:** Implement `IntelIPPDWTPlanImpl` using IPP Wavelet routines (`src/omnidsp/intelipp/dwt.cpp` - New files). _(See DWT/DHT Module)_
- \[ \] **DHT Implementation:** Implement `IntelIPPDHTPlanImpl` using IPP Hilbert routines (`src/omnidsp/intelipp/dht.cpp` - New files). _(See DWT/DHT Module)_
- \[ \] **Convolution/Correlation:** Implement `IntelIPPConvolutionPlanImpl` / `IntelIPPCorrelationPlanImpl` (Optional optimization, possibly using IPP's own Conv/Corr).

### **CQT Module**

- \[ \] Review CQT Plan implementation (`interface/cqt.cpp`, `default/cqt.cpp`) - Ensure `DefaultCQTPlanImpl` correctly uses the owner's `create_fft_plan_*` and `create_resample_plan_*` methods.
- \[ \] Optimize CQT kernel calculation (`default/cqt.cpp`).
- \[ \] Refine hop length handling in recursive CQT implementation.
- \[ \] Tune CQT scaling factor.

### **Convolution / Correlation Module**

- \[ \] Implement 'same' and 'full' modes correctly in one-off `convolve_*`/`correlate_*` methods in `DefaultBackend` (`default/backend.cpp`). _(Plan implementations handle this, but one-off functions need review)_.
- \[ \] **Implement oneMKL Convolution/Correlation Plans:** _(Optional optimization)_
  - \[ \] Create `src/omnidsp/onemkl/convolution.hpp` and `src/omnidsp/onemkl/convolution.cpp`.
  - \[ \] Implement `OneMKLConvolutionPlanImpl` using `vslConvNewTask1D`, `vslsConvExec1D`, `vslConvDeleteTask` (and `d` variants).
  - \[ \] Implement `OneMKLCorrelationPlanImpl` using `vslCorrNewTask1D`, `vslsCorrExec1D`, `vslCorrDeleteTask` (and `d` variants).
  - \[ \] Override `create_convolution_plan_*` / `create_correlation_plan_*` in `OneMKLBackend` to return these plans.
  - \[ \] Add tests specifically for the oneMKL convolution/correlation implementations.
- \[ \] Add C++ and Python tests covering all convolution/correlation modes for one-off functions and plans.

### **STFT Module**

- \[ \] Design `STFTPlan` interface (consider using `FFTPlan` or `RFFTPlan` internally).
- \[ \] Implement `DefaultSTFTPlanImpl` (`default/stft.cpp` - New file).
- \[x] ~~Implement `OneMKLSTFTPlanImpl`~~ _(Inherited from Default)_.
- \[x] ~~Implement `IntelIPPSTFTPlanImpl`~~ _(Inherited from Default)_.
- \[ \] Implement `abstract::AbstractBackend::create_stft_plan_*` factory methods (pure virtual).
- \[ \] Implement factory methods in `DefaultBackend` and other relevant backends.
- \[ \] Include inverse STFT (ISTFT) capability.
- \[ \] Add Python bindings for STFT/ISTFT.
- \[ \] Add tests for STFT module.

### **DCT/DST Module** _(Medium Priority)_

- \[ \] Design public `DCTPlan` / `DSTPlan` interfaces (consider different types I-IV).
- \[ \] Define `abstract::DCTPlanImpl` / `abstract::DSTPlanImpl` interfaces.
- \[ \] Implement `DefaultDCTPlanImpl` / `DefaultDSTPlanImpl` (likely using FFT-based algorithms).
- \[ \] Implement `OneMKLDCTPlanImpl` / `OneMKLDSTPlanImpl` using oneMKL Trigonometric Transform routines (`src/omnidsp/onemkl/dct_dst.cpp` - New files).
- \[ \] Implement `IntelIPPDCTPlanImpl` _(See IPP Enhancements)_.
- \[ \] Implement `IntelIPPDSTPlanImpl` _(See IPP Enhancements)_.
- \[ \] Add `create_dct_plan_*` / `create_dst_plan_*` factory methods to `AbstractBackend` (pure virtual).
- \[ \] Implement factory methods in `DefaultBackend`, `OneMKLBackend`, and `IntelIPPBackend`.
- \[ \] Add Python bindings for DCT/DST.
- \[ \] Add tests for DCT/DST module, comparing backends.

### **DWT/DHT Module** _(Low-to-Medium Priority)_

- \[ \] Design `DWTPlan` / `DHTPlan` interfaces.
- \[ \] Define `abstract::DWTPlanImpl` / `abstract::DHTPlanImpl` interfaces.
- \[ \] Implement Default versions (likely complex or using FFT).
- \[ \] Implement `IntelIPPDWTPlanImpl` _(See IPP Enhancements)_.
- \[ \] Implement `IntelIPPDHTPlanImpl` _(See IPP Enhancements)_.
- \[ \] Add factory methods to `AbstractBackend` and relevant backends.
- \[ \] Add Python bindings and tests.

## **Python Bindings (omnidsp_py)**

- \[ \] Review/Add bindings for Convolution/Correlation modes ('same'/'full') in one-off functions (`bindings.cpp`).
- \[ \] Ensure seamless NumPy integration (dtype handling, `std::span` usage in Plan wrappers).
- \[ \] Investigate memory view / zero-copy support for NumPy arrays passed to Plan `execute`.
- \[ \] Expose more Plan parameters via getters if useful (e.g., CQT frequencies).
- \[ \] Allow pre-allocated output arrays in convenience functions (`api.py`).
- \[ \] Review overall Python API for "Pythonic" feel and ease of use.

## **Build System & CI**

- **CI Configuration:** _(Set up early to catch issues)_
  - \[ \] Configure full CI via GitHub Actions (or similar) for Linux, macOS, and Windows.
  - \[ \] Include builds for all available backends.
  - \[ \] Integrate C++ and Python test execution.
  - \[ \] Add code coverage checks (ensure CMake flags don't hide code).
- **Packaging:** _(Plan early)_
  - \[ \] Investigate static linking options for dependencies.
  - \[ \] Create Conan package for the C++ library.
  - \[ \] Build Python wheels for distribution on PyPI (handle backend selection/inclusion robustly).

## **Testing** _(High Priority)_

- **Coverage:**
  - \[ \] Add tests for Filter module (once implemented).
  - \[ \] Add tests for STFT module (once implemented).
  - \[ \] Add specific tests for Convolution/Correlation modes ('same'/'full') in one-off functions.
  - \[ \] Add tests specifically for the IntelIPP backend implementations (FFT/DFT, Filter, Resample, Window, **DCT**, **DWT**, **DHT**).
  - \[ \] Increase parameter coverage for existing tests (edge cases, different lengths, zero-length, etc.).
  - \[ \] Test arbitrary FFT lengths for IPP backend.
- **Correctness:**
  - \[ \] Add backend-specific tests to verify optimized implementations (Accelerate FFT, IntelIPP\*, OneMKL FFT, **OneMKL Conv/Corr**, **OneMKL DCT/DST**) against the default/reference results.
  - \[ \] Test inherited methods for optimized backends (ensure fallback to `DefaultBackend` works correctly and uses correct sub-plans).
  - \[ \] Test backend fallback logic (e.g., unsupported FFT lengths for Accelerate/oneMKL).
- **Performance:**
  - \[ \] Formalize and run a performance benchmarking suite across backends.

## **Documentation & Examples** _(Update incrementally)_

- \[ \] Update/Generate Doxygen documentation for the latest C++ API (including Filter module, ResampleSpec, updated namespaces, IntelIPP backend).
- \[ \] Update/Generate Sphinx documentation for the Python API (including Filter module, ResampleSpec).
- \[ \] Add C++ and Python usage examples for the Filter module.
- \[ \] Add C++ and Python usage examples for the Resample module.
- \[ \] Update README and CONTRIBUTING.md if any procedures changed significantly (e.g., build steps, contribution guide reflecting new structure).
- \[ \] Add backend-specific performance notes or usage guidelines to documentation.

## **Backend Enhancements / Future Work**

- \[ \] Refine backend selection logic (`OmniDSP::create`) and error handling/reporting.
- \[ \] Add `OmniDSP::get_available_backends()` method.
- \[ \] Add support for GPU acceleration (CUDA/OpenCL) via optional backends (`src/omnidsp/cuda/`...).
- \[ \] Explore asynchronous execution model (e.g., Submission object pattern).
