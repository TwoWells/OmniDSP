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
  - **Details:** Contains static templated functions for generating coefficients for each window type (e.g., `static template<typename T> OmniExpected<std::vector<T>> generate_kaiser_coeffs(size_t length, double beta);`). Used by `Default::Backend` during plan creation.
- **`[ ] Define `template<typename T_Data> OmniDSP::Abstract::WindowPlanImpl` as the PIMPL interface for window plans.`**
  - **Details:** Backend-specific plan implementations will derive from this.
  - **Interface:**
    - `virtual ~WindowPlanImpl() = default;`
    - `virtual OmniExpected<void> execute(const T_Data* src, T_Data* dest, size_t length) = 0;`
    - `virtual OmniExpected<void> execute_inplace(T_Data* src_dest, size_t length) = 0;`
    - `virtual size_t get_length() const = 0;`
- **`[ ] Define `template<typename T_Data> OmniDSP::Plan::Window` as the user-facing window plan class.`**
  - **Details:** Wraps a `std::unique_ptr<Abstract::WindowPlanImpl<T_Data>>`. Its methods call the PIMPL.
- **`[ ] Modify `OmniDSP::Abstract::Backend`to include overloaded pure virtual`create_window_plan_impl` methods for each window specification type.`**
  - **Details:**
    - `template<typename T_Data> virtual OmniExpected<std::unique_ptr<Abstract::WindowPlanImpl<T_Data>>> create_window_plan_impl(const Window::Hann& spec, size_t length) const = 0;`
    - `template<typename T_Data> virtual OmniExpected<std::unique_ptr<Abstract::WindowPlanImpl<T_Data>>> create_window_plan_impl(const Window::Kaiser& spec, size_t length) const = 0;`
    - (One pure virtual overload for each `OmniDSP::Window::[Type]` specification).
- **`[ ] Implement overloaded `template<typename T_Data> OmniDSP::create_window_plan(const Window::[Type]& spec, size_t length) const`methods in the`OmniDSP` facade class.`**
  - **Details:** Each overload calls the corresponding `pimpl_->create_window_plan_impl<T_Data>(spec, length)` and wraps the result in a `Plan::Window<T_Data>`.
- **`[ ] Implement `template<typename T_Data> OmniDSP::generate_window_coefficients(const Window::[Type]& spec, size_t length) const` as overloaded convenience wrappers.`**
  - **Details:** Each overload will internally:
    1. Create a temporary `WindowPlan` using `this->create_window_plan<T_Data>(spec, length)`.
    2. If plan creation succeeds, create temporary `ones` and `result_coeffs` vectors.
    3. Call `plan->execute(ones.data(), result_coeffs.data(), length);`.
    4. Return the `result_coeffs` vector.
- **`[ ] Implement all `template<typename T_Data> create_window_plan_impl(const Window::[Type]& spec, ...)`overrides in`OmniDSP::Default::Backend`.`**
  - **Details:**
    1. Each override creates and returns a backend-specific plan PIMPL, e.g., `std::make_unique<OmniDSP::Default::Plan::Window::HannImpl<T_Data>>(...)`. These classes derive from `OmniDSP::Abstract::WindowPlanImpl<T_Data>`.
    2. The constructor of these `...Impl` classes (or the factory method) calls the appropriate function from `OmniDSP::Internal::WindowFormulas` (e.g., `generate_kaiser_coeffs<T_Data>(length, spec.beta)`).
    3. The `...Impl` class stores these pre-computed coefficients.
- **`[ ] Implement/Update `template<typename T_Data> create_window_plan_impl(const Window::[Type]& spec, ...)`overrides in optimized backends (e.g.,`IntelIPP::Backend`) *only for window types they specifically optimize*.`**
  - **Details:**
    1. Optimized backends (likely inheriting from `Default::Backend`) override only the specific `create_window_plan_impl` overloads they want to handle.
    2. For an overridden type (e.g., `Hann`), they return their specialized PIMPL, e.g., `std::make_unique<OmniDSP::IntelIPP::Plan::Window::HannImpl<T_Data>>(...)`.
    3. **Automatic Fallback:** For window types where the optimized backend does _not_ provide an override, calls will automatically resolve to the `Default::Backend`'s implementation via standard C++ virtual function inheritance.
- **`[ ] Update/Add tests for `create_window_plan`(all overloads),`Plan::Window::execute`, and the `generate_window_coefficients` wrappers. Cover different window types, lengths, data types (`T_Data`), inplace/out-of-place, and backend overrides including automatic fallback. Include tests for constructor validation of window specs.`**
- **`[ ] Update Python bindings for `create_window_plan`(all overloads) and`Plan::Window`methods. Expose concrete window specification classes/structs (e.g.,`Kaiser`, `Hann`) to Python. Ensure parameter validation in Python mirrors C++ behavior.`**
- **`[ ] Update documentation to reflect the new "Window as Plan" architecture (overloaded backend factory methods, automatic fallback, role of window specification structs including logging and validation, backend plan PIMPL naming convention).`**

### **B. Core Infrastructure & Flow**

- **`[ ] Implement internal `Params::[Type]`to`Design::[Type]`conversion (via`OmniDSP::Design::create`) for all design-based operations (Filters, CQT, Resample).`**
  - **Observation:** `OmniDSP::Design::create` (formerly `Utils::create_spec`) is the new path for converting `Params` to `Design` objects. This task focuses on ensuring all relevant `Params::[Type]` (for Filters, CQT, Resample) have this conversion fully implemented and consistently used.
  - **Likely Status:** In progress.
- **`[ ] Standardize backend Plan/Processor PIMPL naming and organization to mirror frontend structure.`**
  - **Details:** For a frontend API class like `OmniDSP::Plan::[Type]`, the corresponding backend PIMPL interface should be `OmniDSP::Abstract::[Type]PlanImpl` (or `ProcessorImpl`). Concrete backend implementations should follow a pattern like `OmniDSP::[BackendName]::Plan::[Type]::[ConcreteImplNameIfMultiple]` or `OmniDSP::[BackendName]::Plan::[Type]Impl`.
  - **Example:** `OmniDSP::Plan::FFT` (frontend) -> `OmniDSP::Abstract::FFTPlanImpl` (PIMPL interface) -> `OmniDSP::Default::Plan::FFTImpl` (default backend PIMPL).
  - **Action:** Review and refactor existing backend PIMPL classes to conform to this convention. This applies to FFT, Convolution, Correlation, FIRFilter, IIRFilter, Resample, CQT, and the new Window plans.

### **C. Testing (High Priority)**

- **`[ ] Test OmniDSP::create_plan (for FFT, Convolution, Correlation using direct `Params`) and OmniDSP::create_processor (for Filters, CQT, Resample using `Params`->`Design` flow) paths.`**
- **`[ ] Test [Type]Plan::execute (FFT, Convolution, Correlation).`** (Tests for Conv/Corr will need updates for new factory method signatures).
- **`[ ] Test IIR, Conv/Corr, STFT, DCT/DST, DWT/DHT modules comprehensively.`**
- **`[ ] Test specific backend implementations (e.g., IntelIPP::*, OneMKL::*) against Default::*.`** (Will require updates due to Abstract::Backend signature changes).
- **`[ ] Add tests for Dispatcher::Backend configurations.`**
- **`[ ] Test Dispatcher::Backend with various backend override combinations.`**

### **D. Python Bindings (High Priority for Core Functionality)**

- **`[ ] Bind all Params::[Type], Coefs::[Type], Design::[Type].`**
- **`[ ] Bind [Type]Plan classes and their execute.`**
- **`[ ] Bind OmniDSP::create_plan, OmniDSP::create_processor overloads reflecting updated flows.`** (Will require updates due to Abstract::Backend signature changes affecting underlying calls).
- **`[ ] Python bindings and tests for IIRFilterProcessor.`**
- **`[ ] Python bindings and tests for FIRFilterProcessor.`**
- **`[ ] Python bindings and tests for CQTProcessor.`**
- **`[ ] Python bindings and tests for ResampleProcessor.`**

## **II. New Features & Modules (Not Started / Planning)**

### **A. Zero-Phase Filtering (`filtfilt`)**

- **`[ ] Add OmniDSP::zero_phase_filter methods (likely stateless one-off).`**
- **`[ ] Add zero_phase_filter_impl to Abstract::Backend.`**
- **`[ ] Implement in Default::Backend.`**
- **`[ ] Test OmniDSP::zero_phase_filter.`**
- **`[ ] Python bindings and tests for Zero-Phase Filtering.`**

### **B. STFT Module** (Likely Plan - Stateless)

- **`[ ] Design Params::STFT.`**
- **`[ ] Implement STFTPlan class (with execute, taking `Params::STFT`).`**
- **`[ ] Add create_stft_plan_impl to Abstract::Backend. Python bindings & tests.`**
- **`[ ] Python bindings & tests for STFT Module.`**

### **C. DCT/DST Module** (Likely Plan - Stateless transform)

- **`[ ] Design Params::DCT/Params::DST.`**
- **`[ ] Implement DCTPlan/DSTPlan classes (with execute, taking `Params`).`**
- **`[ ] Implement OneMKL::DCTPlanImpl/OneMKL::DSTPlanImpl.`**
- **`[ ] Implement IntelIPP::DCTPlanImpl/IntelIPP::DSTPlanImpl.`**

### **D. DWT/DHT Module** (Likely Plan - Stateless transform)

- **`[ ] Design Params::DWT/Params::DHT.`**
- **`[ ] Implement DWTPlan/DHTPlan classes (with execute, taking `Params`).`**
- **`[ ] Implement IntelIPP::DWTPlanImpl/IntelIPP::DHTPlanImpl. Python bindings & tests.`**
- **`[ ] Python bindings & tests for DWT/DHT Module.`**

### **E. Intel IPP Backend Enhancements (`IntelIPP::Backend`)**

- **`[ ] Arbitrary Length FFT/DFT in IntelIPP::FFTPlanImpl.`**
  - _(Note: `Implement IntelIPP::FIRFilterProcessorImpl, IntelIPP::IIRFilterProcessorImpl, IntelIPP::ResampleProcessorImpl` is marked [x] but noted as "Implementation files will require updates")_

## **III. Ongoing / Maintenance Tasks**

### **A. Build System & CI**

- **`[ ] Full CI for Linux, macOS, Windows.`**
- **`[ ] Code coverage checks.`**
- **`[ ] Conan package, Python wheels.`**

### **B. Documentation & Examples** _(Update Incrementally)_

- **`[ ] Update Design Philosophy.md (reflect Plan/Processor, factories, Params::[Type], Design::[Type] flow, new Windowing architecture, backend PIMPL organization).`**
- **`[ ] Update Backend Developer Guide.md (reflect Plan/Processor, factories, *Impl state, Params::[Type], Design::[Type] flow, updated Abstract::Backend signatures, new Windowing architecture, backend PIMPL organization).`**
- **`[ ] Update README.md (reflect Plan/Processor, factories, Params::[Type], Design::[Type] flow, new Windowing architecture, backend PIMPL organization).`**
- **`[ ] Doxygen for C++ API, Sphinx for Python API.`**
- **`[ ] Add/Update examples demonstrating Plan vs Processor usage, including reset().`**
- **`[ ] Add/Update examples for new window plan creation and execution.`**

### **C. Backend Enhancements / Future Work (General)**

- **`[ ] Refine OmniDSP::create logic, error handling.`**
- **`[ ] OmniDSP::get_available_backends().`**
- **`[ ] GPU acceleration (e.g., CUDA::Backend).`**
- **`[ ] Asynchronous execution (May require rethinking internal state if applied to Processors).`**

## **IV. Completed Tasks**

### **A. Core Infrastructure & Factory Methods**

- **`[x] Implement OmniDSP::create overloads supporting overrides.`**
- **`[x] Implement OmniDSP::create_plan overloads (taking `Params::[Type]`for FFT, Convolution, Correlation; or`Coefs` where applicable) for stateless operations.`**
- **`[x] Implement OmniDSP::create_plan overload for CorrelationPlan (taking `Params::Correlation` and kernel).`**
- **`[x] Implement OmniDSP::create_processor overloads (taking `Params::[Type]`leading to`Design::[Type]`, or `Coefs::[Type]`) for stateful operations (FIR, IIR, Resample, CQT).`**
- **`[x] Define public [Type]Plan classes (e.g., FFTPlan, ConvolutionPlan, CorrelationPlan) with execute.`**
- **`[x] Define public [Type]Processor classes (FIRFilterProcessor, IIRFilterProcessor, ResampleProcessor, CQTProcessor) with execute and reset.`**
- **`[x] Add create_*_plan_impl virtual methods to Abstract::Backend (for FFT, RFFT, Convolution, Correlation).`**
- **`[x] Add/Update create_correlation_plan_impl in Abstract::Backend to accept `Params::Correlation` (or max_input_length) and kernel.`**
- **`[x] Rename and finalize create_*_processor_impl virtual methods in Abstract::Backend (for FIR, IIR, Resample, CQT).`**
- **`[x] Implement Default::*PlanImpl classes (stateless for FFT, Convolution, Correlation).`**
- **`[x] Implement Default::*ProcessorImpl classes (stateful, internal state, reset logic for FIR, IIR, Resample, CQT).`**
- **`[x] Update #includes of "OmniDSP/filter.hpp" to "OmniDSP/fir_filter.hpp" or "OmniDSP/iir_filter.hpp" throughout the codebase.`**
- **`[x] Refactor WindowSpec to WindowSetup.`** (Note: `WindowSetup` is to be superseded by the new `OmniDSP::Window::[Type]` specification structs used with `create_window_plan`).
- **`[x] Establish new directory structure.`**
- **`[x] Define FilterType, FIRFilterDesignMethod, IIRFilterFormat.`**
- **`[x] Establish "Params::[Type] -> Design::[Type] -> Processor" (for design-based ops) and "Params::[Type] -> Plan" (for non-design ops) pipelines conceptually.`**
- **`[x] Update CMakeLists.txt & Includes for new structure.`**
- **`[x] Establish Optimized::Backend inherits Default::Backend pattern (conceptual).`**
- **`[x] Initial Plan/Processor class definitions and backend implementation structure (e.g., `Default::CQTPlanImpl`exists but needs to become`CQTProcessorImpl` and finalize state management).` -> **Superseded by direct renames to Processor.\*\*
- **`[x] Backend `create*\*\_plan_impl`methods exist (e.g.,`create_cqt_plan_impl`) but need renaming to `create*\*\_processor_impl` for stateful types and signature review.` -> **Renaming to ProcessorImpl done.\*\* (Signatures for convolution, correlation, and FIR creation updated).

### **B. IIR Filter Module**

- **`[x] Define Params::IIRFilter, Design::IIRFilter, Coefs::SOSs.`** (Note: `Coefs::SOSs` is `Coefs::IIRFilterSOS`)
- **`[x] Refactor IIRFilterPlan into IIRFilterProcessor class (with execute, reset) in iir_filter.hpp.`**
- **`[x] Implement create_iir_filter_processor_impl in Abstract::Backend / Default::Backend.`**
- **`[x] Implement `OmniDSP::Design::create`(from`Params::IIRFilter`to`Design::IIRFilter`).`** (Formerly `Utils::create_spec for IIR`)

### **C. FIR Filter Module**

- **`[x] Define Params::FIRFilter, Design::FIRFilter, Coefs::FIRFilter.`**
- **`[x] Refactor FIRFilterPlan into FIRFilterProcessor class (with execute, reset) in fir_filter.hpp.`**
- **`[x] Implement create_fir_filter_processor_impl in Abstract::Backend / Default::Backend.`**
- **`[x] Define Params::FIRFilter, Params::Resample, Params::CQT.`**
- **`[x] Refine/Define Design::FIRFilter, Design::Resample, Design::CQT.`**
- **`[x] Implement `OmniDSP::Design::create`(from`Params::FIRFilter`to`Design::FIRFilter`).`** (Formerly `Utils::create_spec for FIR`)

### **D. Convolution / Correlation Module**

- **`[x] Finalize Params::Convolution, Params::Correlation definitions (self-validating).`**
- **`[x] Finalize public API and ensure consistent factory usage for ConvolutionPlan and CorrelationPlan (taking `Params`).`**

### **E. CQT Module**

- **`[x] Define Params::CQT, Design::CQT.`**
- **`[x] Refactor CQTPlan into CQTProcessor class (with execute, reset).`**
- **`[x] Implement create_cqt_processor_impl in Abstract::Backend / Default::Backend.`**
- **`[x] Review/Implement Default::CQTProcessorImpl constructor taking Design::CQT, managing internal state, and reset.`**
- **`[x] Implement `OmniDSP::Design::create`(from`Params::CQT`to`Design::CQT`).`** (Formerly `Utils::create_spec for CQT`)

### **F. Resampling Module**

- **`[x] Define Params::Resample, Design::Resample.`**
- **`[x] Refactor ResamplePlan into ResampleProcessor class (with execute, reset).`**
- **`[x] Implement create_resample_processor_impl in Abstract::Backend / Default::Backend.`**
- **`[x] Implement `OmniDSP::Design::create`(from`Params::Resample`to`Design::Resample`).`** (Formerly `Utils::create_spec for Resample`)

### **G. Intel IPP Backend Enhancements (`IntelIPP::Backend`)**

- **`[x] Implement IntelIPP::FIRFilterProcessorImpl, IntelIPP::IIRFilterProcessorImpl, IntelIPP::ResampleProcessorImpl (including internal state and reset).`**
  - **Observation:** Files in `src/omnidsp/intelipp/` reflect these renames. (Note: These will need updates to reflect `Abstract::Backend` signature changes for their factory methods).
  - **Status:** Renaming and structural updates complete. Specific feature implementations (like arbitrary length FFT) are separate. (Implementation files will require updates).

### **H. Build System**

- **`[x] Update src/omnidsp/CMakeLists.txt for filter file split.`**

### **I. Documentation & Examples**

- **`[x] Update examples/cpp/cqt.cpp to use CQTProcessor.`**

### **J. Testing**

- **`[x] Test [Type]Processor::execute and [Type]Processor::reset statefulness (CQT tests updated to CQTProcessor).`**
  - **Observation:** `tests/cpp/cqt.cpp` and `tests/cpp/tests/cqt.cpp` updated to use `CQTProcessor`. FIR/IIR/Resample tests need similar review/update. `CQTProcessor` now has a `reset` method to test. (Tests for FIR will need updates for new factory method signatures).
  - **Status:** Partially complete. _(Marked as [x] in original but noted as partially complete, keeping original status for now)_

### **K. Python Bindings**

- **`[x] Bind [Type]Processor classes and their execute, reset (CQT, Resample updated).`**
  - **Observation:** `src/omnidsp_py/bindings.cpp` shows `CQTProcessor` and `ResampleProcessor` being bound (previously `...Plan`). FIR/IIR bindings still need update. `CQTProcessor` bindings should now include `reset`. (Bindings for Plan/Processor creation will need updates due to factory signature changes).
  - **Status:** Partially complete. _(Marked as [x] in original but noted as partially complete, keeping original status for now)_

## **V. Removed / Obsolete**

- `~~Standardize State Management (Implement Plan::initialize_state(), Define StateType, Update Plan::execute to accept StateType&)~~ - Replaced by internal state in Processor.`
- `~~General "Implement these factory methods in Default::Backend and optimized backends"~~ - Covered by module-specific tasks.`
- `~~Original task related to defining filter classes in the old filter.hpp~~ - Superseded by split into fir_filter.hpp and iir_filter.hpp.`
- `~~Concept of separate "Setup" objects (e.g., STFTSetup, WindowSetup)~~ - Superseded by self-validating Params objects or dedicated configuration class hierarchies (like Abstract::Window).`
- `~~Abstract::Backend::generate_window_coeffs_impl~~ - Removed to simplify backend interface.`
- `~~WindowType enum and Abstract::Window::get_type()~~ - Removed in favor of RTTI (e.g. dynamic_cast) or patterns like Visitor for type-specific logic.`
- `~~Static coefficient generation methods on concrete window classes~~ - Logic moved to internal utility functions.`
- `~~Visitor Pattern for Window Dispatch (Abstract::WindowVisitor, Abstract::Window::accept)~~ - Replaced by RTTI dispatch within optimized backend's apply_window_impl to call private helper methods.`
- **`[NEWLY OBSOLETE] `OmniDSP::Abstract::Window`(as a base for specifications passed to a single backend`create_window_plan_impl` method that uses RTTI).`** - Replaced by distinct `OmniDSP::Window::[Type]` spec structs and overloaded `create_window_plan_impl` methods in `Abstract::Backend`.
- **`[NEWLY OBSOLETE] RTTI dispatch (e.g., `dynamic_cast`) within `Default::Backend`or`Optimized::Backend`'s *single* `create_window_plan_impl` method for window types.`** - Superseded by overloaded virtual `create_window_plan_impl` methods in `Abstract::Backend`.

---

_Note: The "Summary of Potential TODO Updates" section from the original has been removed as this reorganized list serves that purpose._
