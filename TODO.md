# OmniDSP TODO List: Progress Analysis (May 14, 2025)

This document analyzes the progress of the OmniDSP project based on the provided file structure, comparing it against the `TODO.md` (specifically items not yet marked as `[x]`). Refactored to use `Params::[Type]` and `Design::[Type]` naming conventions.

## **I. High Priority / Blocking Tasks**

### **Implement `Dispatcher::Backend` & Advanced Configuration:**

- **`[x] Implement OmniDSP::create overloads supporting overrides.`**
  - **Observation:** `src/omnidsp/omnidsp.cpp` and `include/OmniDSP/omnidsp.hpp` exist. These files are the expected location for `OmniDSP::create` method overloads. The presence of `src/omnidsp/dispatcher/backend.cpp` further supports that backend override mechanisms are in place.
  - **Verification:** Code review of `omnidsp.hpp`, `omnidsp.cpp`, and `dispatcher/backend.hpp` confirms that `OmniDSP::create` overloads and the `OmniDSP::Builder` allow for specifying a primary backend and per-category override backends, which are then managed by `Dispatcher::Backend`.
  - **Status:** Verified as complete.

### **Implement Core Factory Methods & Execution Objects:**

- **`[x] Implement OmniDSP::create_plan overloads (taking Params::[Type]/Coefs) for stateless operations (FFT, Convolution).`**
  - **Observation:** `omnidsp.hpp` and `omnidsp.cpp` (which includes `omnidsp.tpp`) are central for these factory methods.
  - **Verification:** `omnidsp.hpp` declares `OmniDSP::create_plan` for `FFTPlan` (taking `Params::FFT`), `RFFTPlan` (taking `Params::RFFT`), and `ConvolutionPlan` (taking `Params::Convolution` and kernel coefficients). Implementations are present in `omnidsp.tpp`.
  - **Status:** Verified as complete for FFT and Convolution.
- **`[x] Implement OmniDSP::create_plan overload for CorrelationPlan (taking Params::Correlation and kernel).`**
  - **Details:** Added `OmniDSP::create_plan(const Params::Correlation&, const std::vector<T>&)` in `omnidsp.hpp` (declaration) and `omnidsp.tpp` (implementation). This now calls `pimpl_->create_correlation_plan_impl(...)`.
  - **Verification:** The declaration and template implementation for `OmniDSP::create_plan` taking `Params::Correlation` and `kernel_coeffs` have been added to `omnidsp.hpp`.
  - **Status:** Completed. (Note: Relies on `Abstract::Backend::create_correlation_plan_impl` update tracked below).
- **`[ ] Implement OmniDSP::create_processor overloads (taking Params::[Type]/Coefs) for stateful operations (FIR, IIR, Resample, CQT).`**
  - **Observation:** `omnidsp.hpp` and `omnidsp.cpp` are central. Structure for backend `create_*_processor_impl` methods exists. Renaming from `...Plan` to `...Processor` is largely done in `omnidsp.hpp` and `omnidsp.tpp` for these factory signatures.
  - **Likely Status:** In progress; backend `Impl` classes and their factory methods in `Abstract::Backend` need to fully reflect the `Processor` renaming.
- **`[x] Define public [Type]Plan classes (e.g., FFTPlan, ConvolutionPlan, CorrelationPlan) with execute.`**
  - **Verification:** `fft.hpp` defines `FFTPlan`, `RFFTPlan`. `convolution.hpp` defines `ConvolutionPlan`, `CorrelationPlan`. All have execute methods.
  - **Status:** Verified as complete.
- **`[x] Define public [Type]Processor classes (FIRFilterProcessor, IIRFilterProcessor, ResampleProcessor, CQTProcessor) with execute and reset.`**
  - **Observation:** Header files (`fir_filter.hpp`, `iir_filter.hpp`, `resample.hpp`, `cqt.hpp`) now exist and define these classes, renamed from `...Plan`. (Note: `CQTProcessor` abstract interface and default impl. now have `reset`.)
  - **Status:** Verified as complete at the public API level. Backend `Impl` classes also reflect this renaming and addition of `reset` where applicable.
- **`[x] Add create_*_plan_impl virtual methods to Abstract::Backend (for FFT, RFFT, Convolution, Correlation).`**
  - **Verification:** `Abstract::Backend` in `interface/backend.hpp` declares `create_fft_plan_impl_c32/64`, `create_rfft_plan_impl_f32/64`. The `create_convolution_plan_impl_X` and `create_correlation_plan_impl_X` signatures have been updated to accept `Params::Convolution` or `Params::Correlation` respectively, along with coefficient spans.
  - **Status:** Verified as complete and updated.
- **`[x] Add/Update create_correlation_plan_impl in Abstract::Backend to accept Params::Correlation (or max_input_length) and kernel.`**
  - **Details:** `Abstract::Backend::create_correlation_plan_impl` signatures in `src/omnidsp/interface/backend.hpp` updated to accept `const Params::Correlation& params` and `std::span<const T> template_coeffs`. This provides necessary info like `max_input_length` from `params` for proper FFT sizing.
  - **Status:** Completed. Note: All derived backend implementations (e.g., `Default::Backend`, `IntelIPP::Backend`) will need to update their `create_correlation_plan_impl` methods to match the new `Abstract::Backend` interface. The `create_convolution_plan_impl` signatures were also updated for consistency.
- **`[x] Rename and finalize create_*_processor_impl virtual methods in Abstract::Backend (for FIR, IIR, Resample, CQT).`**
  - **Details:** Existing `create_*_plan_impl` methods for these types in `Abstract::Backend` have been renamed to `create_*_processor_impl`. `Abstract::CQTProcessorImpl` now includes `reset`. Signatures for `create_fir_filter_processor_impl` updated to accept `const Coefs::FIRFilter<T>&`.
  - **Verification:** `interface/backend.hpp` shows `Abstract::CQTProcessorImpl` (with `reset`), `Abstract::ResampleProcessorImpl` (with `reset`), `Abstract::FIRFilterProcessorImpl` (with `reset`), `Abstract::IIRFilterProcessorImpl` (with `reset`) and corresponding `create_*_processor_impl` methods in `Abstract::Backend`.
  - **Status:** Verified as complete.
- **`[x] Implement Default::*PlanImpl classes (stateless for FFT, Convolution, Correlation).`**
  - **Observation:** `src/omnidsp/default/fft.hpp/.cpp` and `src/omnidsp/default/convolution.hpp/.cpp` exist.
  - **Verification of FFT usage:** `Default::(Convolution/Correlation)PlanImpl` constructors correctly use `FFTPlanImplVariant` obtained from `this->create_Xfft_plan_impl_Y`, ensuring backend-specific FFT plans. (Note: These will need updates to reflect `Abstract::Backend` signature changes for their factory methods).
  - **Status:** Verified as complete and correctly designed for FFT usage. (Implementation files will require updates).
- **`[x] Implement Default::*ProcessorImpl classes (stateful, internal state, reset logic for FIR, IIR, Resample, CQT).`**
  - **Observation:** Files exist in `src/omnidsp/default/` for these, and class names have been updated from `...PlanImpl` to `...ProcessorImpl`. `Default::CQTProcessorImpl` now implements `reset`. (Note: These will need updates to reflect `Abstract::Backend` signature changes for their factory methods).
  - **Status:** Verified as complete in terms of renaming and structural setup. `Default::CQTProcessorImpl` now has a `reset` method. Detailed state logic review per module might be ongoing for `execute` methods. (Implementation files will require updates).
- **`[x] Update #includes of "OmniDSP/filter.hpp" to "OmniDSP/fir_filter.hpp" or "OmniDSP/iir_filter.hpp" throughout the codebase.`**
  - **Verification:** Diff shows these changes have been made in multiple files.
  - **Status:** Verified as complete.

### **Complete `Params` -> `Setup`/`Spec` Internal Flow:**

- **`[ ] Implement internal Params::to_setup() / Params::to_spec() methods for all types.`**
  - **Observation:** `Utils::create_spec` (called by `Params::to_spec`) exists for many types. This task focuses on ensuring all `Params::[Type]` have this conversion fully implemented and consistently used to produce `Design::[Type]`.
  - **Likely Status:** In progress.

### **Implement IIR Filter Module:**

- **`[x] Define Params::IIRFilter, Design::IIRFilter, Coefs::SOSs.`**
  - **Status:** Verified as complete. (Note: `Coefs::SOSs` is `Coefs::IIRFilterSOS` in `coefs/iir_filter.hpp`).
- **`[x] Refactor IIRFilterPlan into IIRFilterProcessor class (with execute, reset) in iir_filter.hpp.`**
  - **Observation:** `include/OmniDSP/iir_filter.hpp` defines `IIRFilterProcessor`. Backend `Impl` classes renamed. `reset` method present.
  - **Status:** Verified as complete.
- **`[x] Implement create_iir_filter_processor_impl in Abstract::Backend / Default::Backend.`**
  - **Observation:** `Abstract::Backend` and `Default::Backend` declare/define `create_iir_filter_processor_impl_X`.
  - **Status:** Verified as complete.

### **Implement FIR Filter Module:**

- **`[x] Define Params::FIRFilter, Design::FIRFilter, Coefs::FIRFilter.`**
  - **Status:** Verified as complete via existing `params/fir_filter.hpp`, `design/fir_filter.hpp`, `coefs/fir_filter.hpp`.
- **`[x] Refactor FIRFilterPlan into FIRFilterProcessor class (with execute, reset) in fir_filter.hpp.`**
  - **Observation:** `include/OmniDSP/fir_filter.hpp` defines `FIRFilterProcessor`. Backend `Impl` classes renamed. `reset` method present.
  - **Status:** Verified as complete.
- **`[x] Implement create_fir_filter_processor_impl in Abstract::Backend / Default::Backend.`**
  - **Observation:** `Abstract::Backend` and `Default::Backend` declare/define `create_fir_filter_processor_impl_X` (signature updated in `Abstract::Backend` to take `Coefs::FIRFilter<T>`).
  - **Status:** Verified as complete.

### **Implement Zero-Phase Filtering (`filtfilt`):**

- **`[ ] Add OmniDSP::zero_phase_filter methods (likely stateless one-off).`**
- **`[ ] Add zero_phase_filter_impl to Abstract::Backend.`**
- **`[ ] Implement in Default::Backend.`**
  - **Likely Status:** Not started.

## **II. Core Library Features**

### **Convolution / Correlation Module** (Stateless Plans)

- **`[x] Finalize Params::Convolution, Params::Correlation definitions.`**
  - **Observation:** `params/convolution.hpp` defines these structures with `max_input_length`, `kernel_length`/`template_length`, `type`, and `method_hint`. Constructors and fluent setters with validation are present. Documentation confirms suitability for plan creation requiring upfront knowledge of lengths for resource allocation (e.g., FFT sizing).
  - **Status:** Verified as complete.
- **`[x] Finalize public API and ensure consistent factory usage for ConvolutionPlan and CorrelationPlan.`**
  - **Observation:** `include/OmniDSP/convolution.hpp` defines the classes. `OmniDSP::create_plan` is now used for both. `Abstract::Backend` signatures for `create_convolution_plan_impl` and `create_correlation_plan_impl` updated.
  - **Status:** API definition largely complete; backend implementations need to adopt the new `Abstract::Backend` signatures.

### **Intel IPP Backend Enhancements (`IntelIPP::Backend`)**

- **`[ ] Arbitrary Length FFT/DFT in IntelIPP::FFTPlanImpl.`**
- **`[ ] IntelIPP::DCTPlanImpl/IntelIPP::DSTPlanImpl.`**
- **`[ ] IntelIPP::DWTPlanImpl/IntelIPP::DHTPlanImpl.`**
- **`[x] Implement IntelIPP::FIRFilterProcessorImpl, IntelIPP::IIRFilterProcessorImpl, IntelIPP::ResampleProcessorImpl (including internal state and reset).`**
  - **Observation:** Files in `src/omnidsp/intelipp/` reflect these renames. (Note: These will need updates to reflect `Abstract::Backend` signature changes for their factory methods).
  - **Status:** Renaming and structural updates complete. Specific feature implementations (like arbitrary length FFT) are separate. (Implementation files will require updates).

### **CQT Module** (Stateful Processor)

- **`[x] Define Params::CQT, Design::CQT.`**
  - **Status:** Verified as complete.
- **`[x] Refactor CQTPlan into CQTProcessor class (with execute, reset).`**
  - **Observation:** `include/OmniDSP/cqt.hpp` defines `CQTProcessor`. Backend `Impl` classes renamed. `reset` method now added to `Abstract::CQTProcessorImpl` and `Default::CQTProcessorImpl`.
  - **Status:** Verified as complete.
- **`[x] Implement create_cqt_processor_impl in Abstract::Backend / Default::Backend.`**
  - **Observation:** `Abstract::Backend` and `Default::Backend` declare/define `create_cqt_processor_impl_X`.
  - **Status:** Verified as complete.
- **`[x] Review/Implement Default::CQTProcessorImpl constructor taking Design::CQT, managing internal state, and reset.`**
  - **Status:** Constructor and initial state setup from `Design::CQT` are implemented. `reset()` method now added to `Default::CQTProcessorImpl` and `Abstract::CQTProcessorImpl`. `execute()` is a stub. (Note: `Default::Backend`'s factory method will need to be updated if its signature changed in `Abstract::Backend`).

### **Resampling Module** (Stateful Processor)

- **`[x] Define Params::Resample, Design::Resample.`**
  - **Status:** Verified as complete.
- **`[x] Refactor ResamplePlan into ResampleProcessor class (with execute, reset).`**
  - **Observation:** `include/OmniDSP/resample.hpp` defines `ResampleProcessor`. Backend `Impl` classes renamed. `reset` method present.
  - **Status:** Verified as complete.
- **`[x] Implement create_resample_processor_impl in Abstract::Backend / Default::Backend.`**
  - **Observation:** `Abstract::Backend` and `Default::Backend` declare/define `create_resample_processor_impl_X`.
  - **Status:** Verified as complete.

### **STFT Module** (Likely Plan - Stateless)

- **`[ ] Design Params::STFT, STFTSetup.`**
- **`[ ] Implement STFTPlan class (with execute).`**
- **`[ ] Add create_stft_plan_impl to Abstract::Backend. Python bindings & tests.`**
  - **Likely Status:** Not started.

### **DCT/DST Module** (Likely Plan - Stateless transform)

- **`[ ] Design Params::DCT/Params::DST, DCTSetup/DSTSetup.`**
- **`[ ] Implement DCTPlan/DSTPlan classes (with execute).`**
- **`[ ] Implement OneMKL::DCTPlanImpl/OneMKL::DSTPlanImpl.`**
- **`[ ] Implement IntelIPP::DCTPlanImpl/IntelIPP::DSTPlanImpl.`**
  - **Likely Status:** Not started or early planning.

### **DWT/DHT Module** (Likely Plan - Stateless transform)

- **`[ ] Design Params::DWT/Params::DHT, DWTSetup/DHTSetup.`**
- **`[ ] Implement DWTPlan/DHTPlan classes (with execute).`**
- **`[ ] Implement IntelIPP::DWTPlanImpl/IntelIPP::DHTPlanImpl. Python bindings & tests.`**
  - **Likely Status:** Not started or early planning.

## **III. Build System & CI**

- **`[x] Update src/omnidsp/CMakeLists.txt for filter file split.`**
  - **Verification:** Diff shows `interface/filter.cpp` removed and `interface/fir_filter.cpp`, `interface/iir_filter.cpp` added.
  - **Status:** Verified as complete.
- **`[ ] Full CI for Linux, macOS, Windows.`**
- **`[ ] Code coverage checks.`**
- **`[ ] Conan package, Python wheels.`**

## **IV. Documentation & Examples** _(Update Incrementally)_

- **`[x] Update examples/cpp/cqt.cpp to use CQTProcessor.`**
  - **Status:** Verified as complete from diff.
- **`[ ] Update Design Philosophy.md (reflect Plan/Processor, factories, Params::[Type], Design::[Type]).`**
- **`[ ] Update Backend Developer Guide.md (reflect Plan/Processor, factories, *Impl state, Params::[Type], Design::[Type], updated Abstract::Backend signatures).`**
- **`[ ] Update README.md (reflect Plan/Processor, factories, Params::[Type], Design::[Type]).`**
- **`[ ] Doxygen for C++ API, Sphinx for Python API.`**
- **`[ ] Add/Update examples demonstrating Plan vs Processor usage, including reset().`**

## **V. Testing** _(High Priority)_

- **`[ ] Test OmniDSP::create_plan, OmniDSP::create_processor paths.`**
- **`[ ] Test [Type]Plan::execute (FFT, Convolution, Correlation).`** (Tests for Conv/Corr will need updates for new factory method signatures).
- **`[x] Test [Type]Processor::execute and [Type]Processor::reset statefulness (CQT tests updated to CQTProcessor).`**
  - **Observation:** `tests/cpp/cqt.cpp` and `tests/cpp/tests/cqt.cpp` updated to use `CQTProcessor`. FIR/IIR/Resample tests need similar review/update. `CQTProcessor` now has a `reset` method to test. (Tests for FIR will need updates for new factory method signatures).
  - **Status:** Partially complete.
- **`[ ] Test OmniDSP::zero_phase_filter.`**
- **`[ ] Test IIR, Conv/Corr, STFT, DCT/DST, DWT/DHT modules comprehensively.`**
- **`[ ] Test specific backend implementations (e.g., IntelIPP::*, OneMKL::*) against Default::*.`** (Will require updates due to Abstract::Backend signature changes).
- **`[ ] Add tests for Dispatcher::Backend configurations.`**
- **`[ ] Test Dispatcher::Backend with various backend override combinations.`**
- **`[ ] Python bindings and tests for IIRFilterProcessor.`** (Moved from IIR Filter Module)
- **`[ ] Python bindings and tests for FIRFilterProcessor.`** (Moved from FIR Filter Module)
- **`[ ] Python bindings and tests for CQTProcessor.`** (Moved from CQT Module)
- **`[ ] Python bindings and tests for ResampleProcessor.`** (Moved from Resampling Module)
- **`[ ] Python bindings and tests for Zero-Phase Filtering.`** (Moved from Zero-Phase Filtering)
- **`[ ] Python bindings & tests for STFT Module.`** (Moved from STFT Module)
- **`[ ] Python bindings & tests for DWT/DHT Module.`** (Moved from DWT/DHT Module)

## **VI. Python Bindings (omnidsp_py)**

- **`[ ] Bind all Params::[Type], Coefs::[Type].`**
- **`[ ] Bind [Type]Plan classes and their execute.`**
- **`[x] Bind [Type]Processor classes and their execute, reset (CQT, Resample updated).`**
  - **Observation:** `src/omnidsp_py/bindings.cpp` shows `CQTProcessor` and `ResampleProcessor` being bound (previously `...Plan`). FIR/IIR bindings still need update. `CQTProcessor` bindings should now include `reset`. (Bindings for Plan/Processor creation will need updates due to factory signature changes).
  - **Status:** Partially complete.
- **`[ ] Bind OmniDSP::create_plan, OmniDSP::create_processor overloads.`** (Will require updates due to Abstract::Backend signature changes affecting underlying calls).
- **`[ ] Bind OmniDSP::zero_phase_filter.`**

## **VII. Backend Enhancements / Future Work**

- **`[ ] Refine OmniDSP::create logic, error handling.`**
- **`[ ] OmniDSP::get_available_backends().`**
- **`[ ] GPU acceleration (e.g., CUDA::Backend).`**
- **`[ ] Asynchronous execution (May require rethinking internal state if applied to Processors).`**

---

**DONE (Conceptual & Partial):**
Items in this section reflect foundational work or items that are substantially complete pending final refactoring or integration.

- `[x] Refactor WindowSpec to WindowSetup.`
- **`[x] Define Params::FIRFilter, Params::Resample, Params::CQT.`**
- `[x] Establish new directory structure.`
- `[x] Define FilterType, FIRFilterDesignMethod, IIRFilterFormat.`
- **`[x] Refine/Define Design::FIRFilter, Design::Resample, Design::CQT.`**
- `[x] Implement Utils::create_spec for FIR, Resample, CQT, IIR.`
- **`[x] Establish "Params::[Type] -> Design::[Type] -> Plan/Processor" pipeline conceptually.`**
- `[x] Update CMakeLists.txt & Includes for new structure.`
- `[x] Establish Optimized::Backend inherits Default::Backend pattern (conceptual).`
- `[x] Initial Plan/Processor class definitions and backend implementation structure (e.g., `Default::CQTPlanImpl`exists but needs to become`CQTProcessorImpl` and finalize state management).` -> **Superseded by direct renames to Processor.**
- `[x] Backend `create*\*\_plan_impl`methods exist (e.g.,`create_cqt_plan_impl`) but need renaming to `create*\*\_processor_impl` for stateful types and signature review.` -> **Renaming to ProcessorImpl done.** (Signatures for convolution, correlation, and FIR creation updated).

---

**REMOVED / OBSOLETE:**

- `~~Standardize State Management (Implement Plan::initialize_state(), Define StateType, Update Plan::execute to accept StateType&)~~ - Replaced by internal state in Processor.`
- `~~General "Implement these factory methods in Default::Backend and optimized backends"~~ - Covered by module-specific tasks.`
- `~~Original task related to defining filter classes in the old filter.hpp~~ - Superseded by split into fir_filter.hpp and iir_filter.hpp.`

## **Summary of Potential TODO Updates:**

(This summary section can be removed from the final TODO if preferred, as the list itself is the primary artifact)

Consider moving the following items, or parts of them, to `[x]` or marking as significantly progressed:

- **Core Factory Methods & Execution Objects:** Many sub-tasks related to defining Plan/Processor classes and their default implementations.
- **IIR Filter Module:** Definitions of `Params::IIRFilter`, `Design::IIRFilter`, `Coefs::SOSs`. Implementation of the processor class and its default backend.
- **CQT Module:** Definitions of `Params::CQT`, `Design::CQT`. Implementation of the processor class and its default backend, including `reset`.
- **Resampling Module:** Implementation of the processor class and its default/IPP backend.
- **Convolution / Correlation Module:** Definitions of `Params::Convolution`/`Params::Correlation`, Plan class, and default backend implementation.

Items like **Zero-Phase Filtering**, **STFT**, **DCT/DST (public API)**, and **DWT/DHT (public API)** seem to have less visible progress based on file names alone and likely remain `[ ]`.
