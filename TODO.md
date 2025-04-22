# OmniDSP TODO List

## Core Library (`omnidsp`)

- [x] **Refactor Core Types & API Structure:**

  - [x] Replace `Real`/`Complex` typedefs with template aliases (`RealT`, `ComplexT`).
  - [x] Use `std::vector` directly instead of `VectorReal`/`VectorComplex`.
  - [x] Introduce core enums (`Status`, `DataType`, `Domain`, `Window`, `ConvolutionMode`, `Backend`).
  - [x] Add Doxygen comments to core types and API headers.
  - [x] Implement Pimpl idiom for `OmniDSP` class (central API entry point).
  - [x] Define Plan-based interfaces (`FFTPlan`, `RFFTPlan`, `CQTPlan`, `ConvolutionPlan`, `CorrelationPlan`) using Pimpl.
  - [ ] Update C++ implementations (.cpp files) across all modules and backends to use new types, `OmniExpected`, and Pimpl structure.
  - [ ] Define and implement `OmniDSPImpl` class in `omnidsp.cpp`.
  - [ ] Define and implement Plan `Impl` classes in respective `.cpp` files.
  - [ ] Implement `OmniDSP` member functions (DSP ops, Plan factories) in `omnidsp.cpp` (delegating to `pimpl_`).

- [ ] **Implement FFT Module:**

  - [x] Define `FFTPlan` and `RFFTPlan` interfaces (`fft.h`).
  - [ ] Add `create_fft_plan` and `create_rfft_plan` factory methods to `OmniDSP` class.
  - [ ] Implement `FFTPlanImpl`/`RFFTPlanImpl` for all backends (`fft.cpp`, backend sources).
  - [ ] Implement factory methods in `omnidsp.cpp`.

- [ ] **Implement CQT Module:**

  - [x] Define `CQTPlan` interface (`cqt.h`).
  - [ ] Add `create_cqt_plan` factory method to `OmniDSP` class.
  - [ ] Implement `CQTPlanImpl` for all backends (`cqt.cpp`, backend sources).
  - [ ] Implement factory method in `omnidsp.cpp`.
  - [ ] Optimize kernel calculation.
  - [ ] Add option for different window functions during plan creation.

- [ ] **Implement Convolution/Correlation Module:**

  - [x] Define `ConvolutionPlan` and `CorrelationPlan` interfaces (`convolution.h`).
  - [ ] Add `create_convolution_plan` and `create_correlation_plan` factory methods to `OmniDSP` class.

  * [ ] Implement direct `convolve1d`/`correlate1d` member functions in `OmniDSP` class (delegating to `pimpl_`).

  - [ ] Implement `ConvolutionPlanImpl`/`CorrelationPlanImpl` for all backends (`convolution.cpp`, backend sources).
  - [ ] Implement factory methods in `omnidsp.cpp`.

- [ ] **Implement Resample Module:**

  - [ ] Refactor `resample.h` to define `ResamplePlan` interface (using Pimpl).
  - [ ] Add `create_resample_plan` factory method to `OmniDSP` class.
  - [ ] Implement `ResamplePlanImpl` for all backends (`resample.cpp`, backend sources).
  - [ ] Implement factory method in `omnidsp.cpp`.

- [ ] **Implement Filter Module:**

  - [ ] Add basic FIR/IIR filter design functions (potentially static methods in `OmniDSP` or standalone in `filter.h`?).
  - [ ] Implement filtering operations (consider using `ConvolutionPlan` internally).
  - [ ] Define `FilterPlan` interface? Or handle via direct convolution?

- [ ] **Add STFT Module:**

  - [ ] Implement Short-Time Fourier Transform (likely using `FFTPlan` or `RFFTPlan` internally).
  - [ ] Define `STFTPlan` interface? Or provide helper functions/class?
  - [ ] Include inverse STFT (ISTFT).

- [ ] **Enhance Backend System:**
  - [ ] Add support for GPU acceleration (CUDA/OpenCL) via optional backends.
  - [ ] Refine backend selection logic and error handling in `OmniDSP::create`.

## Python Bindings (`omnidsp_py`)

- [ ] **Refactor Bindings for `OmniDSP` Class:**
  - [ ] Bind the `OmniDSP` class (creation, methods).
  - [ ] Bind Plan classes (`FFTPlan`, `RFFTPlan`, `CQTPlan`, `ConvolutionPlan`, `CorrelationPlan`, `ResamplePlan`).
  - [ ] Ensure `OmniExpected` return values are handled correctly (e.g., raise Python exceptions on error).
- [ ] **Expose Convolution/Correlation Module:**
  - [ ] Add bindings for `OmniDSP::convolve1d/correlate1d` methods.
  - [ ] Add bindings for `ConvolutionPlan/CorrelationPlan` creation and execution.
- [ ] **Expose Filter Module:**
  - [ ] Add Python bindings for the new filter functions/classes.
- [ ] **Expose CQT Module:**
  - [ ] Ensure all CQT options are available via `create_cqt_plan` binding.
- [ ] **Expose STFT Module:**
  - [ ] Add Python bindings for STFT/ISTFT.
- [ ] **Improve NumPy Integration:**
  - [ ] Ensure seamless conversion between NumPy arrays and `std::span` arguments in C++ bindings.
  - [ ] Handle different NumPy dtypes correctly (map to float/double templates).

## Build System & CI

- [ ] **Add Unit Tests:**
  - [ ] Implement comprehensive unit tests for all modules (core C++ and Python bindings).
  - [ ] Test different backends.
  - [ ] Integrate testing into the CI pipeline.
- [ ] **Configure CI:**
  - [ ] Set up GitHub Actions for Linux, macOS, and Windows builds.
  - [ ] Include testing and code coverage checks.
- [ ] **Packaging:**
  - [ ] Create Conan package for the C++ library.
  - [ ] Build Python wheels for distribution on PyPI (handle backend selection/inclusion).

## Documentation

- [x] **Write API Documentation:**
  - [x] Generate Doxygen documentation for the C++ API headers (`core_types.h`, `omnidsp.h`, `fft.h`, `cqt.h`, `convolution.h` done).
  - [ ] Add Doxygen comments to remaining headers (`resample.h`, `filter.h`, `stft.h`).
  - [ ] Add Doxygen comments to `.cpp` implementation files.
  - [ ] Use Sphinx for Python API documentation.
- [ ] **Add Examples:**
  - [ ] Provide usage examples for both C++ (using `OmniDSP` class and Plans) and Python.

# OmniDSP TODO List (Refactored for New Architecture)

_High-level goals: Flatten public namespaces, introduce OmniDSP class (using PIMPL) as the main interface and factory for Plans, use PIMPL for Plan classes, add ResamplePlan. Python API will mirror C++, requiring an OmniDSP instance._

## Core Refactoring Tasks (High Priority / Blocking)

- [x] **Define Core API Headers (New Structure):**
  - [x] **`core_types.h`:** Review and finalize core enums (`Precision`, `Direction`, `Domain`, `FFTNorm`, `ConvMode`, `WindowType`). Ensure no nested namespaces.
  - [x] **`omnidsp.h`:**
    - Define the main `OmniDSP` class (forward-declare `OmniDSPImpl` in `OmniDSP::backend` namespace).
    - Declare `OmniDSP` constructor and destructor (defined in `.cpp`).
    - Declare move constructor/assignment (defaulted or defined in `.cpp`). Delete copy constructor/assignment.
    - Declare methods for stateless operations: `convolve1d`, `correlate1d`, `get_hann_coeffs`, `get_hamming_coeffs`, `get_kaiser_coeffs`, `get_flattop_coeffs`.
    - Declare factory methods for creating Plans: `create_fft_plan`, `create_rfft_plan`, `create_cqt_plan`, `create_resample_plan`. These should return `std::unique_ptr<FFTPlan<T>>`, etc., and take the necessary configuration parameters (length, precision, sample_rate, etc.).
    - Remove old standalone function declarations.
    - Include necessary Plan headers (`fft.h`, `cqt.h`, `resample.h`).
  - [x] **`fft.h`:**
    - Define `FFTPlan` and `RFFTPlan` classes within the `OmniDSP` namespace.
    - Forward-declare `FFTPlanImpl` and `RFFTPlanImpl` (in `OmniDSP::backend`).
    - Make constructors **private** or **protected**. Add `OmniDSP` as a friend class OR provide an internal constructor taking backend context from the factory.
    - Declare destructor, move operations (potentially delete copy ops), and execution methods (`fft`, `ifft`, `rfft`, `irfft`).
    - Remove old standalone convenience function declarations.
  - [x] **`cqt.h`:**
    - Define `CQTPlan` class within the `OmniDSP` namespace.
    - Forward-declare `CQTPlanImpl` (in `OmniDSP::backend`).
    - Make constructor **private** or **protected**. Add `OmniDSP` as a friend class OR provide internal constructor. Update signature (e.g., accept `WindowSpec`).
    - Declare destructor, move operations, `execute` method.
    - Remove old standalone `cqt` convenience function declaration.
  - [x] **`convolution.h`:**
    - Remove standalone `convolve1d`/`correlate1d` declarations.
    - Keep `ConvMode` enum if defined here (or move to `core_types.h`).
  - [x] **`window.h`:**
    - Remove `Window` class declaration.
    - Define `WindowType` enum and `WindowSpec` struct.
  - [x] **`resample.h`:**
    - Define `ResamplePlan` class (within `OmniDSP` namespace).
    - Forward-declare `ResamplePlanImpl` (in `OmniDSP::backend`).
    - Make constructor **private** or **protected**. Add `OmniDSP` as a friend class OR provide internal constructor.
    - Declare destructor, move operations, and `execute` method.
    - Remove old standalone `filter_and_downsample` declaration.
- [ ] **Implement Core API Source Files:**
  - [x] **`omnidsp.cpp`:**
    - Implement `OmniDSP` constructor: Create the correct `backend::OmniDSPImpl` based on CMake flags (`USE_ONEMKL`, `USE_ACCELERATE`). Perform any one-time backend setup (e.g., MKL threading).
    - Implement `OmniDSP` destructor.
    - Implement `OmniDSP` move operations.
    - Implement `OmniDSP` stateless methods (`convolve1d`, etc.) by forwarding calls to `pimpl_->method(...)`.
    - Implement `OmniDSP` factory methods (`create_*_plan`): Use `pimpl_` to access backend context/factory logic, create the appropriate `backend::*PlanImpl`, and construct/return the `std::unique_ptr<PlanType>`.
    - Add necessary explicit template instantiations for `OmniDSP` methods.
  - [x] **`fft.cpp`:**
    - Implement internal `FFTPlan`/`RFFTPlan` constructors (taking backend context).
    - Implement `FFTPlan`/`RFFTPlan` destructor, move operations, execute methods (forward to `pimpl_`).
    - Add explicit template instantiations.
    - Remove old convenience function implementations.
  - [x] **`cqt.cpp`:**
    - Implement internal `CQTPlan` constructor (taking backend context). Update internal logic to use `ResamplePlan`.
    - Implement `CQTPlan` destructor, move operations, execute method (forward to `pimpl_`).
    - Add explicit template instantiations.
    - Remove old convenience function implementation.
  - [x] **`window.cpp`:**
    - Implement `get_window_coeffs` utility function (potentially internal to `backend::OmniDSPImpl`).
    - Remove old `Window` class implementations.
    - Add necessary template instantiations for any remaining helpers.
  - [x] **`resample.cpp`:**
    - Implement internal `ResamplePlan` constructor (taking backend context).
    - Implement `ResamplePlan` destructor, move operations, execute method (forward to `pimpl_`).
    - Add explicit template instantiations.
    - Remove old `filter_and_downsample` implementation.
  - [x] **`convolution.cpp`:**
    - Remove old standalone function implementations.
- [ ] **Refactor Backend Interface and Implementations:**
  - [ ] **`backend.h` (Internal Header):**
    - Define `OmniDSP::backend` namespace.
    - Declare backend implementation classes (`OmniDSPImpl`, `FFTPlanImpl`, `CQTPlanImpl`, `ResamplePlanImpl`). These might inherit from internal base classes or be defined directly.
    - Define any necessary internal base classes/interfaces within `OmniDSP::backend`.
    - Remove forward declarations for old standalone backend functions.
  - [ ] **`backend.cpp`:** Remove this file.
  - [ ] **Backend Files (`src/omnidsp/backend/[name]/*.cpp`):**
    - Update namespaces to `OmniDSP::backend::[name]`.
    - Implement the backend-specific `OmniDSPImpl`, `FFTPlanImpl`, `CQTPlanImpl`, `ResamplePlanImpl` classes.
    - `OmniDSPImpl` constructor handles backend setup.
    - `*PlanImpl` constructors likely take context/handles from the `OmniDSPImpl` that created them.
    - Ensure these classes contain the actual backend API calls (MKL, Accelerate, or stub logic).
    - Handle PIMPL implementation details (constructors, destructors managing backend resources).
    - Remove implementations for old standalone backend functions.
- [ ] **Update Build System (`CMakeLists.txt`):**
  - Update source file lists for `omnidsp` library target.
  - Update backend source file lists (`OMNIDSP_BACKEND_SOURCES`).
  - Ensure correct include paths are set (public API vs. internal backend).
  - Adjust installation rules if header structure changes significantly.
- [ ] **Update Python Bindings & API:**
  - [ ] **`bindings.cpp`:**
    - Bind the `OmniDSP` class, its constructor, and its methods (stateless ops + factory methods).
    - Bind the `FFTPlan`, `CQTPlan`, `ResamplePlan` classes (only execution methods and getters should be public; constructors are effectively hidden).
    - Remove bindings for old standalone functions/`Window` class.
  - [ ] **`api.py`:**
    - **Strategy:** Python users will instantiate `OmniDSP` (e.g., `dsp = ods.OmniDSP()`).
    - Implement wrappers/functions that take an `OmniDSP` instance as an argument OR require users to call methods directly (e.g., `dsp.convolve1d(...)`, `plan = dsp.create_fft_plan(...)`). Define clear usage patterns.
    - Remove wrappers for old standalone functions/`Window` class.
  - [ ] **`__init__.py`:** Update `__all__` to export `OmniDSP` and the Plan classes (and potentially helper wrappers from `api.py`).
- [ ] **Update & Add Unit Tests:**
  - [ ] Adapt existing C++ tests (`tests/cpp`) for new class structure (create `OmniDSP` object, use factory methods to get Plans).
  - [ ] Add C++ tests for `ResamplePlan`.
  - [ ] Adapt existing Python tests (`tests/python`) for new API (create `OmniDSP` instance, use factory methods).
  - [ ] Add Python tests for `OmniDSP` methods and `ResamplePlan`.
- [ ] **Debug & Verify:** Fix any compilation errors and test failures introduced by the refactor.

## Dependent / Follow-Up Tasks (Medium/Low Priority - Re-evaluate after core refactor)

- **Tasks from Previous TODO (Integrate/Adapt):**
  - [ ] **CQT Resampling:** Ensure the CQT implementation now correctly uses the new `ResamplePlan`. _(Part of core refactor)_
  - [ ] **CQT Scaling Factor:** Tune CQT scaling factor. _(Depends on passing CQT tests)_
  - [ ] **Convolution Modes:** Implement 'same'/'full' modes for `OmniDSP::convolve1d`/`correlate1d`.
  - [ ] **Filter Design:** Implement FIR/IIR filter design functions (likely methods on `OmniDSP`).
  - [ ] **Error Handling:** Improve error propagation from `backend::*Impl` classes through `OmniDSP`/`Plan` classes to Python.
  - [ ] **Backend Query:** Add `OmniDSP::getActiveBackend()` method.
  - [ ] **STFT Implementation:** Consider adding `OmniDSP::create_stft_plan()`.
  - [ ] **Accelerate Window Implementations:** Implement Kaiser/Flattop in `backend::Accelerate::OmniDSPImpl` if possible.
- **Testing:**
  - [ ] Add Conv/Corr Mode Tests (C++/Python).
  - [ ] Increase parameter coverage.
  - [ ] Add backend-specific tests.
  - [ ] Formalize benchmarking suite.
- **Python Bindings:**
  - ~~Define Python API strategy (convenience funcs vs explicit `OmniDSP` instance).~~ _(Decision made: Explicit OmniDSP instance)_
  - [ ] Investigate memory view / zero-copy support.
  - [ ] Expose more Plan parameters via getters.
  - [ ] Allow pre-allocated output arrays.
- **Build System & CI:**
  - [ ] Add Linux/Windows CI runners.
  - [ ] Investigate static linking.
- **Documentation & Examples:**
  - [ ] Update/Generate Doxygen/Sphinx docs for the new API (`OmniDSP` class, factory methods, Plans).
  - [ ] Update C++/Python examples for the new API (showing `OmniDSP` instantiation and usage).
  - [ ] Update README and backend documentation.

## Summary of Next Steps (Prioritized):

1.  **DEFINE CORE API HEADERS:** Update `.h` files for `OmniDSP`, Plans, `core_types`. Adjust constructors/factories. Remove old declarations.
2.  **REFACTOR BACKEND INTERFACE:** Define internal `backend::*Impl` classes/interfaces in `backend.h`.
3.  **IMPLEMENT CORE API SOURCE:** Implement `.cpp` files for `OmniDSP` (including factory methods) and Plans (internal constructors, forwarding methods).
4.  **IMPLEMENT BACKEND DETAILS:** Implement the `backend::*Impl` classes in `backend/[name]/*.cpp`.
5.  **UPDATE BUILD SYSTEM:** Adjust `CMakeLists.txt`.
6.  **UPDATE PYTHON BINDINGS/API:** Modify `bindings.cpp`, `api.py`, `__init__.py` to reflect the explicit `OmniDSP` instance strategy.
7.  **UPDATE TESTS:** Adapt C++ and Python tests. Add tests for `ResamplePlan`.
8.  **DEBUG & VERIFY:** Build, run tests, fix issues.
9.  Proceed with dependent tasks.
