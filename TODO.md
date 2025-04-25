# **OmniDSP Project TODO List**

**Goal:** Outline the development tasks for the OmniDSP library, including C++ core features, backend implementations, Python bindings, build system refactoring, testing, and documentation, ensuring alignment with the design philosophy and current architecture.

**Directory Structure Reminder:**

- src/omnidsp/interface/: Abstract interfaces (backend.h) and public Plan wrappers (fft.cpp, cqt.cpp, etc.).
- src/omnidsp/default/: Default backend implementation (backend.h/cpp, fft.cpp, window.cpp, etc.).
- src/omnidsp/{accelerate, onemkl, your_backend}/: Specific backend implementations inheriting from DefaultBackend.
- src/omnidsp/omnidsp.cpp: Public OmniDSP class implementation.

## **High Priority / Blocking Tasks**

- **Refactor ResamplePlan Infrastructure (If needed after initial impl):**
  - _(Review if ResamplePlan needs further refactoring based on Filter module integration. The current DefaultResamplePlanImpl uses the owner's design_fir_filter.)_
  - \[ \] **Review AccelerateResamplePlanImpl (src/omnidsp/accelerate/resample.cpp):**
    - Ensure constructor takes const AbstractBackend\* owner and const ResampleSpec& spec.
    - Ensure constructor calls owner-\>design_fir_filter_f32/f64 (or equivalent helper) to get coefficients.
    - Verify execute uses Accelerate vDSP primitives correctly for polyphase FIR filtering.
    - Verify reset() implementation.
  - \[ \] **Review OneMKLResamplePlanImpl (src/omnidsp/onemkl/resample.cpp):**
    - Ensure constructor takes const AbstractBackend\* owner and const ResampleSpec& spec.
    - Ensure constructor calls owner-\>design_fir_filter_f32/f64 (or equivalent helper) to get coefficients for IPP FIRMR setup.
    - Verify execute using IPP FIRMR is correct.
    - Verify reset() implementation.
  - \[ \] **Add/Update Tests:**
    - Add C++ tests (tests/cpp) for design_fir_filter.
    - Add C++ and Python tests for ResamplePlan, covering various rate combinations, quality settings, state handling (reset), and edge cases across all implemented backends.
    - Ensure CQT tests still pass after the CQT plan's dependency on the refactored resampler.
  - \[ \] **Update Python Bindings:**
    - Ensure Python create_resample_plan wrapper (src/omnidsp_py/api.py) correctly passes ResampleSpec information.
    - Ensure src/omnidsp_py/bindings.cpp binds the create_resample_plan C++ signature taking ResampleSpec.
- **Implement Filter Module:**
  - \[ \] Implement AccelerateFIRFilterPlanImpl (src/omnidsp/accelerate/filter.cpp \- New file).
  - \[ \] Implement OneMKLFIRFilterPlanImpl (src/omnidsp/onemkl/filter.cpp \- New file).
  - \[ \] Implement AccelerateIIRFilterPlanImpl (src/omnidsp/accelerate/filter.cpp).
  - \[ \] Implement OneMKLIIRFilterPlanImpl (src/omnidsp/onemkl/filter.cpp).
  - \[ \] Implement create_fir_filter_plan\_\* factory methods in AccelerateBackend and OneMKLBackend.
  - \[ \] Implement create_iir_filter_plan\_\* factory methods in AccelerateBackend and OneMKLBackend.
  - \[ \] Implement design_iir_filter\_\* methods in DefaultBackend (src/omnidsp/default/backend.cpp or filter_design.cpp).
  - \[ \] Implement design_fir_filter\_\* methods in DefaultBackend. _(Partially done)_
  - \[ \] Add Python bindings for Filter Plans and design methods.
  - \[ \] Add comprehensive C++ and Python tests for the Filter module across all backends.
- **Review Core Implementations:**
  - \[ \] Review C++ implementations (.cpp files) across interface/, default/, accelerate/, onemkl/ against the latest design_philosophy.md and backend_developer_guide.md to ensure compliance with Pimpl, inheritance (DefaultBackend), API contracts, and naming conventions (AbstractBackend, DefaultBackend, etc.).

## **Core Library Features**

### **CQT Module**

- \[ \] Review CQT Plan implementation (interface/cqt.cpp, default/cqt.cpp) \- Ensure DefaultCQTPlanImpl correctly uses the owner's create_fft_plan\_\* and create_resample_plan\_\* methods.
- \[ \] Optimize CQT kernel calculation (default/cqt.cpp).
- \[ \] Refine hop length handling in recursive CQT implementation.
- \[ \] Tune CQT scaling factor.

### **Convolution / Correlation Module**

- \[ \] Implement 'same' and 'full' modes correctly in one-off convolve\_\*/correlate\_\* methods in DefaultBackend (default/backend.cpp). _(Plan implementations likely handle this already)_
- \[ \] Add C++ and Python tests covering all convolution/correlation modes for one-off functions and plans.

### **STFT Module**

- \[ \] Design STFTPlan interface (consider using FFTPlan or RFFTPlan internally).
- \[ \] Implement DefaultSTFTPlanImpl (default/stft.cpp \- New file).
- \[ \] Implement AccelerateSTFTPlanImpl, OneMKLSTFTPlanImpl (Optional).
- \[ \] Implement AbstractBackend::create_stft_plan\_\* factory methods (pure virtual).
- \[ \] Implement factory methods in DefaultBackend, AccelerateBackend, OneMKLBackend.
- \[ \] Include inverse STFT (ISTFT) capability.
- \[ \] Add Python bindings for STFT/ISTFT.
- \[ \] Add tests for STFT module.

## **Python Bindings (omnidsp_py)**

- \[ \] Review/Add bindings for Convolution/Correlation modes ('same'/'full') in one-off functions (bindings.cpp).
- \[ \] Ensure seamless NumPy integration (dtype handling, std::span usage in Plan wrappers).
- \[ \] Investigate memory view / zero-copy support for NumPy arrays passed to Plan execute.
- \[ \] Expose more Plan parameters via getters if useful (e.g., CQT frequencies).
- \[ \] Allow pre-allocated output arrays in convenience functions (api.py). _(Partially done for windows)_

## **Build System & CI**

- **CMake Refactoring:**
  - **File Paths:** Update all CMake target definitions (target_sources, target_include_directories, etc. in cmake/target_definitions.cmake and elsewhere) to reflect the new interface/, default/, accelerate/, onemkl/ directory structure.
  - **Eliminate C++ \#ifdefs:**
    - **Strategy:** Rely _solely_ on CMake for managing platform/backend variations.
    - **Conditional Source Files:** Ensure backend-specific .cpp files (accelerate/\*.cpp, onemkl/\*.cpp) are added conditionally via generator expressions in cmake/target_definitions.cmake based on OMNIDSP_HAS_ACCELERATE/OMNIDSP_HAS_ONEMKL. Refactor _any_ remaining C++ code using \#ifdef for conditional compilation into separate files managed by CMake.
    - **Conditional Compile Definitions:** Remove _all_ usage of \#ifdef OMNIDSP_USE\_..., \#ifdef \_WIN32, etc., for conditional compilation within C++ source. CMake definitions should only be used by CMake itself (generator expressions), except for OS/compiler requirements like \_USE_MATH_DEFINES.
    - **Conditional Linking:** Verify libraries are linked conditionally based on logic in cmake/backend/\*.cmake.
- **CI Configuration:**
  - \[ \] Configure full CI via GitHub Actions (or similar) for Linux, macOS, and Windows.
  - \[ \] Include builds for all available backends.
  - \[ \] Integrate C++ and Python test execution.
  - \[ \] Add code coverage checks (ensure CMake flags don't hide code).
- **Packaging:**
  - \[ \] Investigate static linking options for dependencies.
  - \[ \] Create Conan package for the C++ library.
  - \[ \] Build Python wheels for distribution on PyPI (handle backend selection/inclusion robustly).

## **Testing**

- \[ \] Add specific tests for Convolution/Correlation modes ('same'/'full') in one-off functions.
- \[ \] Add tests for Filter module (once implemented).
- \[ \] Add tests for STFT module (once implemented).
- \[ \] Increase parameter coverage for existing tests (edge cases, different lengths, etc.).
- \[ \] Add backend-specific tests to verify optimized implementations (Accelerate\*, OneMKL\*) against the default/reference.
- \[ \] Test inherited methods for optimized backends (ensure fallback to DefaultBackend works correctly).
- \[ \] Formalize and run a performance benchmarking suite across backends.

## **Documentation & Examples**

- \[ \] Update/Generate Doxygen documentation for the latest C++ API (including Filter module, ResampleSpec, updated window signatures).
- \[ \] Update/Generate Sphinx documentation for the Python API (including Filter module, ResampleSpec).
- \[ \] Add Doxygen comments to .cpp implementation files (in interface/, default/, accelerate/, onemkl/). _(Partially done)_
- \[ \] Add C++ and Python usage examples for the Filter module.
- \[ \] Add C++ and Python usage examples for the Resample module.
- \[ \] Update README and CONTRIBUTING.md if any procedures changed significantly (e.g., build steps, contribution guide reflecting new structure).
- \[ \] Update backend_developer_guide.md to ensure it fully matches the final implemented structure.
- \[ \] Add backend-specific performance notes or usage guidelines to documentation.

## **Backend Enhancements / Future Work**

- \[ \] Refine backend selection logic (OmniDSP::create) and error handling/reporting.
- \[ \] Add OmniDSP::get_available_backends() method.
- \[ \] Implement Kaiser/Flattop/Gaussian window generation in AccelerateBackend (accelerate/window.cpp \- New file).
- \[ \] Implement window generation methods in OneMKLBackend using VML (onemkl/window.cpp). _(Partially done)_
- \[ \] Add support for GPU acceleration (CUDA/OpenCL) via optional backends (src/omnidsp/cuda/...).
- \[ \] Explore asynchronous execution model (e.g., Submission object pattern).
