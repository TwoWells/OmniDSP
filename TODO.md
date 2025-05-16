# OmniDSP TODO List (Reorganized - May 16, 2025)

This document outlines the development tasks for the OmniDSP project.

## **I. Active Development Focus / High Priority**

### **A. Windowing System Refactor (Window as Plan - Overloaded Backend Factory)**

- **`[ ] Define lightweight window specification classes/structs in `OmniDSP::Window` namespace (non-hierarchical).`**
  - **Details:** E.g., `Window::Hann {}`, `Window::Kaiser { double beta; }`. These act as type tags and parameter holders.
  - **Enhancements:**
    - Each specification class/struct should overload `operator<<` for easy logging (e.g., with `std::ostream` and compatible with `spdlog`).
    - Constructors for specifications with parameters (e.g., `Kaiser`) **must validate** these parameters upon construction (e.g., throw `std::invalid_argument` for invalid values).
  - **Location:** `OmniDSP/include/OmniDSP/window/specs.hpp` (or individual files like `kaiser_spec.hpp`).
- **`[ ] Create internal C++ window formula utilities (e.g., `namespace OmniDSP::Internal::WindowFormulas`).`**
- **`[ ] Define `template<typename T_Data> OmniDSP::Abstract::Plan::WindowImpl` as the PIMPL interface for window plans.`**
  - **Interface:** `virtual OmniStatus execute(std::span<const T_Data> src, std::span<T_Data> dest) const = 0;`, `virtual OmniStatus execute_inplace(std::span<T_Data> src_dest) const = 0;`, `virtual size_t get_length() const = 0;`. (Corrected from `forward/inverse` to `execute/execute_inplace` for windows).
- **`[ ] Define `template<typename T_Data> OmniDSP::Plan::Window` as the user-facing window plan class.`**
- **`[ ] Modify `OmniDSP::Abstract::Backend`to include overloaded pure virtual`create_window_plan_impl` methods for each window specification type.`**
- **`[ ] Implement overloaded `template<typename T_Data> OmniDSP::create_plan(const Window::[Type]& spec, size_t length) const`methods in the`OmniDSP` facade class for windows.`** (Note: Renamed from `create_window_plan` for consistency with other plans).
- **`[ ] Implement `template<typename T_Data> OmniDSP::generate_window_coefficients(const Window::[Type]& spec, size_t length) const` as overloaded convenience wrappers.`**
- **`[ ] Implement all `template<typename T_Data> create_window_plan_impl(const Window::[Type]& spec, ...)`overrides in`OmniDSP::Default::Backend`.`**
- **`[ ] Implement/Update `template<typename T_Data> create_window_plan_impl(const Window::[Type]& spec, ...)`overrides in optimized backends.`**
- **`[ ] Update/Add tests for `create_plan`(window overloads),`Plan::Window::execute`, `Plan::Window::execute_inplace`, and `generate_window_coefficients`.`**
- **`[ ] Update Python bindings for `create_plan`(window overloads) and`Plan::Window`methods. Expose `Window::[SpecType]` structs.`**
- **`[ ] Update documentation for "Window as Plan" architecture.`**

### **B. Transform API Refactor (Transform as Plan - Overloaded Backend Factory)**

- **`[ ] Define lightweight transform specification structs in `OmniDSP::Transform`namespace (e.g.,`Transform::FFT{size_t length}`, `Transform::RFFT{size_t real_length}`, `Transform::DFT{size_t length}`).`**
  - **Details:** Include constructors with validation and `operator<<` for logging.
  - **Location:** `OmniDSP/include/OmniDSP/transform/specs.hpp` (or individual files).
- **`[ ] Define `template<typename T_In, typename T_Out> OmniDSP::Abstract::Plan::TransformImpl` as the PIMPL interface for transform plans.`**
  - **Interface:** `virtual OmniStatus forward(...) const = 0;`, `virtual OmniStatus inverse(...) const = 0;`, `virtual size_t get_input_length() const = 0;`, `virtual size_t get_output_length() const = 0;`.
  - **Location:** `OmniDSP/src/omnidsp/interface/plan/transform_impl.hpp`.
- **`[ ] Define `template<typename T_In, typename T_Out> OmniDSP::Plan::Transform` as the user-facing transform plan class.`**
  - **Details:** Wraps `std::unique_ptr<Abstract::Plan::TransformImpl<T_In, T_Out>>`. Exposes `forward()`, `inverse()`, `get_input_length()`, `get_output_length()`.
  - **Location:** `OmniDSP/include/OmniDSP/plan/transform.hpp`.
- **`[ ] Modify `OmniDSP::Abstract::Backend`to include overloaded pure virtual`create_transform_plan_impl`methods for each`Transform::[SpecType]`.`**
  - **Example:** `template<typename T_Complex> virtual OmniExpected<std::unique_ptr<Abstract::Plan::TransformImpl<T_Complex, T_Complex>>> create_transform_plan_impl(const Transform::FFT& spec) const = 0;`
  - **Example:** `template<typename T_Real> virtual OmniExpected<std::unique_ptr<Abstract::Plan::TransformImpl<T_Real, Utils::GetComplexType<T_Real>>>> create_transform_plan_impl(const Transform::RFFT& spec) const = 0;`
- **`[ ] Implement overloaded `OmniDSP::create_plan(const Transform::[SpecType]& spec) const`methods in the`OmniDSP` facade class for transforms.`**
  - **Details:** Each overload calls the corresponding `pimpl_->create_transform_plan_impl(...)` and wraps the result in a `Plan::Transform<T_In, T_Out>`.
- **`[ ] Implement all `create_transform_plan_impl(const Transform::[SpecType]& spec, ...)`overrides in`OmniDSP::Default::Backend`.`**
  - **Details:** Create and return backend-specific PIMPLs (e.g., `Default::Plan::Transform::FFTC2CImpl<C32>`). Adapt existing `Default::FFTPlanImpl` and `Default::RFFTPlanImpl` logic.
- **`[ ] Implement/Update `create_transform_plan_impl(const Transform::[SpecType]& spec, ...)` overrides in optimized backends (Accelerate, OneMKL, IntelIPP) for FFT and RFFT.`**
- **`[ ] Update/Add tests for `create_plan`(transform overloads) and `Plan::Transform::forward`/`inverse` methods. Cover different transform types, lengths, data types, and backend overrides.`**
- **`[ ] Update Python bindings for `create_plan`(transform overloads) and `Plan::Transform`methods. Expose`Transform::[SpecType]` structs.`**
- **`[ ] Deprecate old `OmniDSP::Plan::FFT`, `OmniDSP::Plan::RFFT`, `OmniDSP::Params::FFT`, `OmniDSP::Params::RFFT`and their associated factory methods in`OmniDSP`and`Abstract::Backend`.`**
- **`[ ] Update all documentation (Design Philosophy, Backend Guide, README, Contributing, Examples) to reflect the new "Transform as Plan" architecture.`**

### **C. Core Infrastructure & Flow**

- **`[ ] Implement internal `Params::[Type]`to`Design::[Type]`conversion (via`OmniDSP::Design::create`) for all design-based operations (Filters, CQT, Resample).`**
- **`[ ] Standardize backend Plan/Processor PIMPL naming and organization to mirror frontend structure.`**

### **D. Testing (High Priority)**

- **`[ ] Test OmniDSP::create_plan (for Convolution, Correlation using direct `Params`) and OmniDSP::create_processor (for Filters, CQT, Resample using `Params`->`Design` flow) paths.`**
- **`[ ] Test [Type]Plan::execute (Convolution, Correlation).`**
- **`[ ] Test IIR, Conv/Corr, STFT, DCT/DST, DWT/DHT modules comprehensively.`**
- **`[ ] Test specific backend implementations against Default::*.`**
- **`[ ] Add tests for Dispatcher::Backend configurations.`**

### **E. Python Bindings (High Priority for Core Functionality)**

- **`[ ] Bind all Params::[Type], Coefs::[Type], Design::[Type].`**
- **`[ ] Bind `Plan::Convolution`, `Plan::Correlation` and their execute methods.`**
- **`[ ] Bind OmniDSP::create_plan (for Convolution/Correlation), OmniDSP::create_processor overloads.`**
- **`[ ] Python bindings and tests for IIRFilterProcessor.`**
- **`[ ] Python bindings and tests for FIRFilterProcessor.`**
- **`[ ] Python bindings and tests for CQTProcessor.`**
- **`[ ] Python bindings and tests for ResampleProcessor.`**

## **II. New Features & Modules (Not Started / Planning)**

### **A. Zero-Phase Filtering (`filtfilt`)**

- **`[ ] Design `Params::ZeroPhaseFilter`(or determine if existing`Params::FIRFilter`/`Params::IIRFilter` are sufficient input along with data).`**
- **`[ ] Add `OmniDSP::zero_phase_filter<T_Data>(std::span<const T_Data> input, const Coefs::FIRFilter<T_Data>& fir_coeffs) const`and`OmniDSP::zero_phase_filter<T_Data>(std::span<const T_Data> input, const Coefs::IIRFilterSOS& iir_coeffs) const` for one-off operations.`**
  - These will likely need to internally create temporary forward and backward filter processors or use a direct implementation.
- **`[ ] Add `zero_phase_filter_fir_impl`and`zero_phase_filter_iir_impl`to`Abstract::Backend`(or a single`zero_phase_filter_impl` that dispatches based on coefficient type).`**
  - These would take input data and coefficients and return the filtered output.
- **`[ ] Implement in `Default::Backend`using standard forward-backward filtering approach (e.g., using`Processor::FIRFilter`and`Processor::IIRFilter` internally).`**
  - Careful state management or stateless processing of segments will be needed.
- **`[ ] Consider optimized implementations in other backends if their filter primitives can be leveraged efficiently for this pattern.`**
- **`[ ] Test `OmniDSP::zero_phase_filter` for both FIR and IIR inputs, various data types, and edge cases (e.g., short signals, different filter orders).`**
- **`[ ] Add Python bindings and corresponding tests for zero-phase filtering functionality.`**

### **B. STFT Module** (Likely `Plan::Transform` specialization or new `Plan::STFT`)

- **`[ ] Design `Transform::STFT`spec (or`Params::STFT`) including window, hop size, FFT length, padding mode.`**
- **`[ ] Implement `Plan::Transform`(or`Plan::STFT`) for STFT. `forward`would produce spectrogram,`inverse` (ISTFT) would reconstruct time-domain signal.`**
  - Will internally use `Plan::Window` and `Plan::Transform` (for RFFT/FFT).
- **`[ ] Add `create_transform_plan_impl`(or`create_stft_plan_impl`) to `Abstract::Backend`. This might be complex as STFT itself is a composite operation.`**
  - Alternatively, STFT could be a higher-level utility in `OmniDSP` that uses existing plans.
- **`[ ] Python bindings & tests.`**

### **C. DCT/DST Module** (Likely `Plan::Transform` specializations)

- **`[ ] Design `Transform::DCT`/`Transform::DST` specs, including transform type (I, II, III, IV) and normalization options.`**
- **`[ ] Implement `Plan::Transform`for DCT/DST.`forward`and`inverse` methods.`**
- **`[ ] Add `create_transform_plan_impl`overloads to`Abstract::Backend`for`Transform::DCT`and`Transform::DST`.`**
- **`[ ] Implement in `Default::Backend` (e.g., using FFT-based algorithms if direct DCT/DST not available).`**
- **`[ ] Implement in OneMKL, IntelIPP backends if they offer direct DCT/DST functions.`**
- **`[ ] Python bindings & tests.`**

### **D. DWT/DHT Module** (Likely `Plan::Transform` specializations)

- **`[ ] Design `Transform::DWT`/`Transform::DHT` specs. For DWT, include wavelet type, levels, mode.`**
- **`[ ] Implement `Plan::Transform`for DWT/DHT.`forward`and`inverse` methods.`**
- **`[ ] Add `create_transform_plan_impl`overloads to`Abstract::Backend`.`**
- **`[ ] Implement in `Default::Backend`.`**
- **`[ ] Implement in IntelIPP backend if DWT/DHT are supported.`**
- **`[ ] Python bindings & tests.`**

### **E. Intel IPP Backend Enhancements (`IntelIPP::Backend`)**

- **`[ ] Arbitrary Length FFT/DFT in IntelIPP `TransformImpl` (e.g., using Bluestein's algorithm if IPP doesn't support it directly, or checking newer IPP versions for direct support).`**

## **III. Ongoing / Maintenance Tasks**

### **A. Build System & CI**

- **`[ ] Establish Full CI Pipeline:** Configure GitHub Actions (or chosen CI service) for Linux, macOS (x86_64 and arm64), and Windows.
  - Include steps for: environment setup (Conda), CMake configuration for various backends, build, C++ tests (GoogleTest), Python tests (pytest).
- **`[ ] Code Coverage:** Integrate code coverage tools (e.g., gcov/lcov for C++, coverage.py for Python) into CI and aim for high coverage. Report coverage (e.g., to Codecov).
- **`[ ] Packaging:**
  - **Conan:** Create Conan recipes for the C++ library to facilitate C++ dependency management for users.
  - **Python Wheels:** Set up CI to build and publish Python wheels for different platforms and Python versions to PyPI.
- **`[ ] Dependency Updates:** Regularly review and update dependencies (Conda environments, pre-commit hooks, CMake modules).
- **`[ ] Build Time Optimization:** Investigate and implement strategies to reduce library build times (e.g., precompiled headers, unity builds if appropriate, ccache).

### **B. Documentation & Examples** _(Update Incrementally)_

- **`[ ] Update Design Philosophy.md (reflect Plan/Processor, factories, Params/Specs, new Windowing & Transform architecture, backend PIMPL organization).`**
- **`[ ] Update Backend Developer Guide.md (reflect Plan/Processor, factories, *Impl state, Params/Specs, updated Abstract::Backend signatures for Windowing & Transforms, backend PIMPL organization).`**
- **`[ ] Update README.md (reflect Plan/Processor, factories, Params/Specs, new Windowing & Transform architecture, backend PIMPL organization).`**
- **`[ ] Doxygen for C++ API: Ensure all public classes, methods, enums, and structs are fully documented. Generate HTML documentation.`**
- **`[ ] Sphinx for Python API: Generate Python API documentation from docstrings. Integrate with Doxygen output if possible (e.g., via Breathe).`**
- **`[ ] Add/Update examples demonstrating `Plan::Transform`, `Plan::Window`, and `Processor` usage for all core operations in both C++ and Python.`**
  - Include examples for backend selection and dispatcher usage.
  - Provide practical use-case examples (e.g., audio filtering, spectral analysis).

### **C. Backend Enhancements / Future Work (General)**

- **`[ ] Refine `OmniDSP::create` logic and error handling for backend instantiation, especially for the Dispatcher.`**
- **`[ ] Implement and test `OmniDSP::get_available_backends()`and`OmniDSP::get_preferred_backend()`.`**
- **`[ ] GPU Acceleration:**
  - **`[ ] Design `CUDA::Backend` (or similar for other GPU platforms like SYCL/OpenCL).`**
  - **`[ ] Implement `CUDA::Plan::Transform::FFTC2CImpl` (using cuFFT).`**
  - **`[ ] Investigate GPU acceleration for other operations (Convolution, Filtering).`**
- **`[ ] Asynchronous Execution:**
  - **`[ ] Explore API for asynchronous operations (e.g., returning `std::future` or using callbacks).`**
  - **`[ ] This would be a significant architectural addition, especially for stateful `Processor` objects. May require careful consideration of threading models and state synchronization.`**
- **`[ ] Logging Improvements: Standardize logging levels and messages across the library. Allow users to configure log levels and outputs.`**
- **`[ ] Performance Benchmarking Suite: Develop a comprehensive suite to compare performance across different backends for various operations and data sizes.`**

## **IV. Completed Tasks**

- **`[x] Implement OmniDSP::create overloads supporting overrides.`**
- **`[x] Implement OmniDSP::create_processor overloads (taking `Params::[Type]`leading to`Design::[Type]`, or `Coefs::[Type]`) for stateful operations (FIR, IIR, Resample, CQT).`**
- **`[x] Define public [Type]Processor classes (FIRFilterProcessor, IIRFilterProcessor, ResampleProcessor, CQTProcessor) with execute and reset.`** (Now `Processor::FIRFilter`, etc.)
- **`[x] Add/Update create_correlation_plan_impl in Abstract::Backend to accept `Params::Correlation` (or max_input_length) and kernel.`** (Superseded for FFT/RFFT by Transform API, but relevant for Convolution/Correlation `Params` usage).
- **`[x] Rename and finalize create_*_processor_impl virtual methods in Abstract::Backend (for FIR, IIR, Resample, CQT).`**
- **`[x] Implement Default::*ProcessorImpl classes (stateful, internal state, reset logic for FIR, IIR, Resample, CQT).`**
- **`[x] Update #includes of "OmniDSP/filter.hpp" to "OmniDSP/fir_filter.hpp" or "OmniDSP/iir_filter.hpp" throughout the codebase.`**
- **`[x] Refactor WindowSpec to WindowSetup.`** (Note: `WindowSetup` is to be superseded by the new `OmniDSP::Window::[Type]` specification structs used with `create_window_plan`).
- **`[x] Establish new directory structure.`**
- **`[x] Define FilterType, FIRFilterDesignMethod, IIRFilterFormat.`**
- **`[x] Establish "Params::[Type] -> Design::[Type] -> Processor" (for design-based ops) and "Params::[Type] -> Plan" (for non-design ops like Conv/Corr) pipelines conceptually.`**
- **`[x] Update CMakeLists.txt & Includes for new structure.`**
- **`[x] Establish Optimized::Backend inherits Default::Backend pattern (conceptual).`**
- **`[x] Initial Plan/Processor class definitions and backend implementation structure (e.g., `Default::CQTPlanImpl`became`CQTProcessorImpl`).`**
- **`[x] Backend `create*\*\_plan_impl`methods renamed to `create*\*\_processor_impl` for stateful types; signatures for convolution, correlation, and FIR creation updated.`**
- **`[x] IIR Filter Module: Defined `Params::IIRFilter`, `Design::IIRFilter`, `Coefs::IIRFilterSOS`. Refactored to `Processor::IIRFilter`. Implemented `create_iir_filter_processor_impl`and`OmniDSP::Design::create` for IIR.`**
- **`[x] FIR Filter Module: Defined `Params::FIRFilter`, `Design::FIRFilter`, `Coefs::FIRFilter`. Refactored to `Processor::FIRFilter`. Implemented `create_fir_filter_processor_impl`and`OmniDSP::Design::create` for FIR.`**
- **`[x] Convolution / Correlation Module: Finalized `Params::Convolution`, `Params::Correlation`. Ensured factory usage for `Plan::Convolution`and`Plan::Correlation`(taking`Params`).`**
- **`[x] CQT Module: Defined `Params::CQT`, `Design::CQT`. Refactored to `Processor::CQT`. Implemented `create_cqt_processor_impl`and`OmniDSP::Design::create`for CQT. Reviewed`Default::CQTProcessorImpl`.`**
- **`[x] Resampling Module: Defined `Params::Resample`, `Design::Resample`. Refactored to `Processor::Resample`. Implemented `create_resample_processor_impl`and`OmniDSP::Design::create` for Resample.`**
- **`[x] Intel IPP Backend Enhancements: Renamed `IntelIPP::*PlanImpl`to`IntelIPP::*ProcessorImpl` where appropriate (FIR, IIR, Resample).`**
- **`[x] Build System: Updated `src/omnidsp/CMakeLists.txt` for filter file split.`**
- **`[x] Documentation & Examples: Updated `examples/cpp/cqt.cpp`to use`CQTProcessor`.`**
- **`[x] Testing: Partially updated CQT tests to `CQTProcessor`. FIR/IIR/Resample tests reviewed. `CQTProcessor` reset method added.`**
- **`[x] Python Bindings: Partially updated CQT and Resample bindings to `Processor` classes. FIR/IIR bindings reviewed.`**

### **(Superseded by Transform API Refactor)**

- **`[x] Implement OmniDSP::create_plan overloads (taking `Params::[Type]`for FFT, Convolution, Correlation; or`Coefs` where applicable) for stateless operations.`**
- **`[x] Define public [Type]Plan classes (e.g., FFTPlan, ConvolutionPlan, CorrelationPlan) with execute.`**
- **`[x] Add create_*_plan_impl virtual methods to Abstract::Backend (for FFT, RFFT, Convolution, Correlation).`**
- **`[x] Implement Default::*PlanImpl classes (stateless for FFT, Convolution, Correlation).`**

## **V. Removed / Obsolete**

- `~~Standardize State Management (Implement Plan::initialize_state(), Define StateType, Update Plan::execute to accept StateType&)~~ - Replaced by internal state in Processor.`
- `~~General "Implement these factory methods in Default::Backend and optimized backends"~~ - Covered by module-specific tasks.`
- `~~Original task related to defining filter classes in the old filter.hpp~~ - Superseded by split into fir_filter.hpp and iir_filter.hpp.`
- `~~Concept of separate "Setup" objects (e.g., STFTSetup, WindowSetup)~~ - Superseded by self-validating Params objects or dedicated configuration class hierarchies (like Window::[SpecType], Transform::[SpecType]).`
- `~~Abstract::Backend::generate_window_coeffs_impl~~ - Removed to simplify backend interface.`
- `~~WindowType enum and Abstract::Window::get_type()~~ - Removed in favor of RTTI (e.g. dynamic_cast) or patterns like Visitor for type-specific logic.` (Note: `WindowType` enum still exists as `OmniDSP::Type::Window` but is used with spec structs, not for RTTI on a base `Abstract::Window` class).
- `~~Static coefficient generation methods on concrete window classes~~ - Logic moved to internal utility functions.`
- `~~Visitor Pattern for Window Dispatch (Abstract::WindowVisitor, Abstract::Window::accept)~~ - Replaced by RTTI dispatch within optimized backend's apply_window_impl to call private helper methods.` (This itself was superseded by overloaded `create_window_plan_impl`).
- **`[NEWLY OBSOLETE] `OmniDSP::Abstract::Window`(as a base for specifications passed to a single backend`create_window_plan_impl` method that uses RTTI).`** - Replaced by distinct `OmniDSP::Window::[Type]` spec structs and overloaded `create_window_plan_impl` methods in `Abstract::Backend`.
- **`[NEWLY OBSOLETE] RTTI dispatch (e.g., `dynamic_cast`) within `Default::Backend`or`Optimized::Backend`'s *single* `create_window_plan_impl` method for window types.`** - Superseded by overloaded virtual `create_window_plan_impl` methods in `Abstract::Backend`.
- **`[NEWLY OBSOLETE] `OmniDSP::Plan::FFT`, `OmniDSP::Plan::RFFT`and their associated`Params::FFT`, `Params::RFFT`.`** - Replaced by `OmniDSP::Plan::Transform` and `OmniDSP::Transform::[SpecType]` structs.
- **`[NEWLY OBSOLETE] `Abstract::FFTPlanImpl`, `Abstract::RFFTPlanImpl`and their factory methods in`Abstract::Backend`.`** - Replaced by `Abstract::Plan::TransformImpl` and overloaded `create_transform_plan_impl` factory methods.

---
