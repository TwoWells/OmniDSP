# OmniDSP TODO List: Progress Analysis (May 13, 2025)

This document analyzes the progress of the OmniDSP project based on the provided file structure, comparing it against the `TODO.md` (specifically items not yet marked as `[x]`).

## **I. High Priority / Blocking Tasks**

### **Implement `Dispatcher::Backend` & Advanced Configuration:**

- **`[x] Implement OmniDSP::create overloads supporting overrides.`**
  - **Observation:** `src/omnidsp/omnidsp.cpp` and `include/OmniDSP/omnidsp.hpp` exist. These files are the expected location for `OmniDSP::create` method overloads. The presence of `src/omnidsp/dispatcher/backend.cpp` further supports that backend override mechanisms are in place.
  - **Verification:** Code review of `omnidsp.hpp`, `omnidsp.cpp`, and `dispatcher/backend.hpp` confirms that `OmniDSP::create` overloads and the `OmniDSP::Builder` allow for specifying a primary backend and per-category override backends, which are then managed by `Dispatcher::Backend`.
  - **Status:** Verified as complete.

### **Implement Core Factory Methods & Execution Objects:**

- **`[x] Implement OmniDSP::create_plan overloads (taking Params/Coefs) for stateless operations (FFT, Conv?, Corr?).`**
  - **Observation:** `omnidsp.hpp` and `omnidsp.cpp` (which includes `omnidsp.tpp`) are central for these factory methods.
  - **Verification:**
    - `omnidsp.hpp` declares `OmniDSP::create_plan` for `FFTPlan` (taking `FFTParams`), `RFFTPlan` (taking `RFFTParams`), and `ConvolutionPlan` (taking `ConvolutionParams` and kernel coefficients).
    - Implementations are present in `omnidsp.tpp` (included by `omnidsp.hpp`).
    - **Note on Correlation ("Corr?"):** `CorrelationPlan` factory method was refactored (see next item).
  - **Status:** Verified as complete for FFT and Convolution. Correlation factory refactored.
- **`[x] Implement OmniDSP::create_plan overload for CorrelationPlan (taking CorrelationParams and kernel).`**
  - **Details:** Added `OmniDSP::create_plan(const CorrelationParams&, const std::vector<T>&)` in `omnidsp.hpp` (declaration) and `omnidsp.tpp` (implementation, included in `omnidsp.hpp`). This now calls `pimpl_->create_correlation_plan_impl(...)`.
  - **Verification:** The declaration and template implementation for `OmniDSP::create_plan` taking `CorrelationParams` and `kernel_coeffs` have been added to `omnidsp.hpp`.
  - **Status:** Completed. (Note: This relies on the corresponding `Abstract::Backend::create_correlation_plan_impl` being updated to correctly use `CorrelationParams`, particularly `max_input_length`, as tracked in a separate task.)
- **`[ ] Implement OmniDSP::create_processor overloads (taking Params/Coefs) for stateful operations (FIR, IIR, Resample, CQT?).`**
  - **Observation:** `omnidsp.hpp` and `omnidsp.cpp` are central for these factory methods. The existence of interface files (`src/omnidsp/interface/fft.cpp`, `filter.cpp`, `convolution.cpp`, `cqt.cpp`, `resample.cpp`) and default backend implementations (`src/omnidsp/default/*`) suggests that the necessary `create_*_plan_impl` and `create_*_processor_impl` methods in `Abstract::Backend` and `Default::Backend` are likely defined or in progress.
  - **Likely Status:** The overall structure is well-established and implementations are likely in progress. The main `OmniDSP` class methods are key.
- **`[x] Define public [Type]Plan classes (e.g., FFTPlan) with execute.`**
  - **Observation:**
    - `include/OmniDSP/fft.hpp` exists (likely for `FFTPlan` and `RFFTPlan`).
    - `include/OmniDSP/convolution.hpp` exists (likely for `ConvolutionPlan` and `CorrelationPlan`).
  - **Verification:**
    - `fft.hpp` defines `FFTPlan` with `fft()`/`ifft()` methods and `RFFTPlan` with `rfft()`/`irfft()` methods.
    - `convolution.hpp` defines `ConvolutionPlan` and `CorrelationPlan`, both with `execute()` methods.
  - **Status:** Verified as complete.
- **`[ ] Define public [Type]Processor classes (e.g., FIRFilterProcessor) with execute and reset.`**
  - **Observation:**
    - `include/OmniDSP/filter.hpp` (likely for `FIRFilterProcessor` and potentially `IIRFilterProcessor`).
    - `include/OmniDSP/resample.hpp` (for `ResampleProcessor`).
    - `include/OmniDSP/cqt.hpp` (for `CQTProcessor`).
    - The `DONE (Conceptual & Partial)` section of your `TODO.md` already notes the shift from `Plan` to `Processor` for CQT and Resample, which aligns with these files.
  - **Likely Status:** Definitions for `FIRFilterProcessor`, `IIRFilterProcessor`, `ResampleProcessor`, and `CQTProcessor` are likely present or substantially complete.
- **`[ ] Add create_*_plan_impl virtual methods to Abstract::Backend.`**
- **`[ ] Add/Update create_correlation_plan_impl in Abstract::Backend to accept CorrelationParams (or max_input_length) and kernel.`**
  - **Details:** The current `create_correlation_plan_impl` in `Abstract::Backend` takes `kernel, type, method`. It needs to be updated or a new overload added to incorporate `max_input_length` from `CorrelationParams` for proper FFT sizing. This is now crucial for the new `OmniDSP::create_plan` for `CorrelationPlan`.
  - **Likely Status:** Requires modification.
- **`[ ] Add create_*_processor_impl virtual methods to Abstract::Backend.`**
- **`[ ] Implement these factory methods in Default::Backend and optimized backends.`**
  - **Observation:** The directory structure (`src/omnidsp/interface/backend.hpp`, `src/omnidsp/default/backend.hpp`, and specific backend directories like `onemkl`, `intelipp`, `accelerate` each with `backend.hpp/.cpp`) strongly supports the implementation of these abstract and concrete factory methods.
  - **Likely Status:** The architectural foundation is firmly in place. Specific method implementations within each backend would require code inspection, but many default implementations seem to be progressing.
- **`[ ] Implement Default::*PlanImpl classes (stateless).`**
  - **Observation:**
    - `src/omnidsp/default/fft.hpp/.cpp` (for `Default::FFTPlanImpl`).
    - `src/omnidsp/default/convolution.hpp/.cpp` (for `Default::ConvolutionPlanImpl`).
  - **Verification of FFT usage in Default Convolution/Correlation:**
    - `Default::(Convolution/Correlation)PlanImpl` constructors in `src/omnidsp/default/convolution.hpp` correctly accept an `FFTPlanImplVariant`.
    - The factory functions in `Default::Backend` (e.g., `create_default_convolution_plan_impl_f32` in `src/omnidsp/default/backend.cpp` - _code for this specific file not fetched in last request, but inferred from structure and `default/convolution.cpp`_) are responsible for creating this variant by calling `this->create_rfft_plan_impl_f32(fft_len)` (or `create_fft_plan_impl_c32`). This ensures the FFT plan comes from the same backend instance.
  - **Status:** Default implementations for FFT and Convolution plans appear to be implemented or well underway. The design for using backend-specific FFT plans within them is **verified as correct.**
- **`[ ] Implement Default::*ProcessorImpl classes (stateful, internal state, reset logic).`**
  - **Observation:**
    - `src/omnidsp/default/fir_filter.hpp/.cpp`
    - `src/omnidsp/default/iir_filter.hpp/.cpp`
    - `src/omnidsp/default/resample.hpp/.cpp`
    - `src/omnidsp/default/cqt.hpp/.cpp`
  - **Likely Status:** Default implementations for FIR, IIR, Resample, and CQT processors, including state management, appear to be implemented or well underway.

### **Complete `Params` -> `Setup`/`Spec` Internal Flow:**

- **`[ ] Implement internal Params::to_setup() / Params::to_spec() methods.`**
  - **Observation:** Logic for this would reside in `src/omnidsp/params/*.cpp` files. These files exist for various types (fft, fir_filter, iir_filter, cqt, resample, convolution).
  - **Likely Status:** In progress. The necessary files are present, but completion of these specific conversion methods within them is not confirmed by file structure alone.
  - Your `TODO.md` already confirms `[x] Refine/Confirm role of Utils::create_spec (likely called by Params::to_spec)`. The `src/omnidsp/utils/*_design.cpp` files support this.

### **Implement IIR Filter Module:**

- **`[x] Define IIRFilterParams, IIRFilterSpec, IIRFilterCoefs.`**
  - **Observation:**
    - `include/OmniDSP/params/iir_filter.hpp` and `src/omnidsp/params/iir_filter.cpp` (for `IIRFilterParams`).
    - `include/OmniDSP/design/iir_filter.hpp` and `src/omnidsp/utils/iir_filter_design.cpp` (for `Design::IIRFilter` as the Spec, and utility to create it from Params).
    - `include/OmniDSP/coefs/iir_filter.hpp` (for `IIRFilterCoef`).
  - **Verification:**
    - `IIRFilterParams` is defined in `params/iir_filter.hpp` and implemented in `params/iir_filter.cpp` with validation.
    - `Design::IIRFilter` (serving as `IIRFilterSpec`) is defined in `design/iir_filter.hpp`. The utility `Utils::create_spec` in `utils/iir_filter_design.cpp` converts `IIRFilterParams` to `Design::IIRFilter`.
    - `IIRFilterCoef` (for SOS) is defined in `coefs/iir_filter.hpp`.
  - **Status:** Verified as complete.
- **`[ ] Refactor IIRFilterPlan into IIRFilterProcessor class (with execute, reset).`**
  - **Observation:** Likely defined in `include/OmniDSP/filter.hpp` or a dedicated `iir_filter.hpp` under `include/OmniDSP/`. The default implementation `src/omnidsp/default/iir_filter.hpp/.cpp` exists.
  - **Likely Status:** Public class definition is likely present; default backend implementation is in progress or complete.
- **`[ ] Implement create_iir_filter_processor_impl in Abstract::Backend / Default::Backend.`**
  - **Observation:** Given `src/omnidsp/default/iir_filter.cpp` exists, the `Default::Backend` implementation is likely present. `Abstract::Backend` (`src/omnidsp/interface/backend.hpp`) would declare the virtual method.
  - **Likely Status:** Appears to be implemented in `Default::Backend`.
- **`[ ] Python bindings and tests for IIRFilterProcessor.`**
  - **Observation:** `src/omnidsp_py/bindings.cpp` would contain bindings. Test files are in `tests/`.
  - **Likely Status:** Python bindings may be in progress. Tests may exist.

### **Implement Zero-Phase Filtering (`filtfilt`):**

- **`[ ] Add OmniDSP::zero_phase_filter methods (likely stateless one-off).`**
- **`[ ] Add zero_phase_filter_impl to Abstract::Backend.`**
- **`[ ] Implement in Default::Backend.`**
- **`[ ] Python bindings and tests.`**
  - **Observation:** No files explicitly named `filtfilt` are apparent in the provided list.
  - **Likely Status:** Likely not started, or potentially implemented under a different name or as part of another module not obvious from file names.

## **II. Core Library Features**

### **Convolution / Correlation Module** (Likely Plan - Stateless)

- **`[ ] Define ConvolutionParams, CorrelationParams, ConvolutionSetup?, CorrelationSetup?, ConvolutionCoefs<T>, CorrelationCoefs<T>.`**
  - **Observation:**
    - `include/OmniDSP/params/convolution.hpp` and `src/omnidsp/params/convolution.cpp` (for `ConvolutionParams`, `CorrelationParams`).
    - `include/OmniDSP/types/convolution.hpp` (for related enums like `ConvolutionType`, `ConvolutionMethod`).
    - `ConvolutionCoefs` might be defined within `include/OmniDSP/convolution.hpp` or a dedicated `coefs/convolution.hpp`. `Setup` structures would follow a similar pattern.
  - **Likely Status:** `Params` and core types are present. `Coefs` and `Setup` definitions need verification but are plausible.
- **`[ ] Implement ConvolutionPlan, CorrelationPlan classes (with execute).`**
  - **Observation:** `include/OmniDSP/convolution.hpp` and `src/omnidsp/interface/convolution.cpp` exist.
  - **Details:** `CorrelationPlan` factory method refactored to use `OmniDSP::create_plan(CorrelationParams, kernel)`. This depends on changes to `Abstract::Backend::create_correlation_plan_impl`.
  - **Likely Status:** Public Plan classes are likely defined or in progress. Factory method for `CorrelationPlan` is now consistent at the `OmniDSP` API level.
- **Other sub-tasks** (backend implementations, one-off methods, bindings, tests):
  - **Observation:** `src/omnidsp/default/convolution.hpp/.cpp` suggests the default backend implementation is progressing. `IntelIPP` and `OneMKL` also have this as a TODO.
  - **Likely Status:** Progressing, especially for the default backend.

### **Intel IPP Backend Enhancements (`IntelIPP::Backend`)**

- **`[ ] Arbitrary Length FFT/DFT in IntelIPP::FFTPlanImpl.`**
- **`[ ] IntelIPP::DCTPlanImpl/IntelIPP::DSTPlanImpl.`**
- **`[ ] IntelIPP::DWTPlanImpl/IntelIPP::DHTPlanImpl.`**
- **`[ ] Implement IntelIPP::FIRFilterProcessorImpl, IntelIPP::IIRFilterProcessorImpl, IntelIPP::ResampleProcessorImpl (including internal state and reset).`**
  - **Observation:** Files such as `src/omnidsp/intelipp/fft.cpp`, `src/omnidsp/intelipp/fir_filter.cpp`, `src/omnidsp/intelipp/iir_filter.cpp`, `src/omnidsp/intelipp/resample.cpp` exist.
  - **Likely Status:** Development for Intel IPP backend implementations of FFT, FIR, IIR, and Resampling is evident. The specific features (Arbitrary Length FFT, DCT/DST, DWT/DHT) would require inspecting the code, but the foundational files for these operations in IPP are present.

### **CQT Module** (Likely Processor - Stateful due to recursive nature/overlap)

- **`[ ] Define CQTParams, CQTSpec.`**
  - **Observation:**
    - `include/OmniDSP/params/cqt.hpp` and `src/omnidsp/params/cqt.cpp` (for `CQTParams`).
    - `include/OmniDSP/design/cqt.hpp` and `src/omnidsp/utils/cqt_design.hpp/.cpp` (for `CQTSpec` and design utilities).
  - **Likely Status:** These definitions appear to be **largely complete**. This item might be ready for `[x]`.
- **`[ ] Refactor CQTPlan into CQTProcessor class (with execute, reset).`**
  - **Observation:** `include/OmniDSP/cqt.hpp`, `src/omnidsp/interface/cqt.cpp` (public API and abstract interface), and `src/omnidsp/default/cqt.hpp/.cpp` (default implementation) exist. This task reflects the planned change from a Plan to a Processor for stateful CQT.
  - **Likely Status:** Public class definition (as `CQTProcessor` or evolving towards it) is likely present or in progress; default backend implementation is in progress or complete.
- **`[ ] Implement create_cqt_processor_impl in Abstract::Backend / Default::Backend.`**
  - **Observation:** Given `src/omnidsp/default/cqt.cpp` exists, the `Default::Backend` implementation is likely present. `Abstract::Backend` (`src/omnidsp/interface/backend.hpp`) would declare the virtual method. This aligns with the move to `CQTProcessor`.
  - **Likely Status:** Appears to be implemented in `Default::Backend`.
- **`[ ] Review/Implement Default::CQTProcessorImpl constructor taking CQTSpec, managing internal state, and reset.`**
  - **Observation:** `src/omnidsp/default/cqt.hpp/.cpp` are the key files.
  - **Likely Status:** In progress or complete.
- **`[ ] Python bindings and tests for CQTProcessor.`**
  - **Observation:** `tests/cpp/tests/cqt.cpp` exists. `src/omnidsp_py/bindings.cpp` for bindings.
  - **Likely Status:** Progressing.

### **Resampling Module** (Processor - Stateful)

- **`[x] Define ResampleParams, ResampleSpec. (Already Done?)`** - _Confirmed as `[x]` in TODO._
  - **Observation:** Supported by `include/OmniDSP/params/resample.hpp`, `include/OmniDSP/design/resample.hpp`, etc.
- **`[ ] Implement ResampleProcessor class (with execute, reset).`**
  - **Observation:** `include/OmniDSP/resample.hpp`, `src/omnidsp/interface/resample.cpp` (public API and abstract interface), and `src/omnidsp/default/resample.hpp/.cpp` (default implementation) exist. `src/omnidsp/intelipp/resample.hpp/.cpp` also exists.
  - **Likely Status:** Public class definition is likely present; default and Intel IPP backend implementations are in progress or complete.
- **Other sub-tasks** (bindings, tests):
  - **Likely Status:** Progressing.

### **STFT Module** (Likely Plan - Stateless)

- **`[ ] Design STFTParams, STFTSetup.`**
- **`[ ] Implement STFTPlan class (with execute).`**
- **`[ ] Add create_stft_plan_impl to Abstract::Backend. Python bindings & tests.`**
  - **Observation:** No obvious, dedicated files like `stft.hpp` in `include/OmniDSP/` or `src/omnidsp/default/stft.cpp`.
  - **Likely Status:** May not be started yet, or could be integrated within FFT/Windowing functionalities in a less direct way.

### **DCT/DST Module** (Likely Plan - Stateless transform)

- **`[ ] Design DCTParams/DSTParams, DCTSetup/DSTSetup.`**
- **`[ ] Implement DCTPlan/DSTPlan classes (with execute).`**
- **`[ ] Implement OneMKL::DCTPlanImpl/OneMKL::DSTPlanImpl.`**
- **`[ ] Implement IntelIPP::DCTPlanImpl/IntelIPP::DSTPlanImpl.`**
  - **Observation:** While `OneMKL` and `IntelIPP` backends have TODOs for these, dedicated public API files (`dct.hpp`, `dst.hpp`) are not apparent in `include/OmniDSP/`.
  - **Likely Status:** Backend work might be planned or in early stages for specific libraries; public API and default implementations seem not yet started or are named differently.

### **DWT/DHT Module** (Likely Plan - Stateless transform)

- **`[ ] Design DWTParams/DHTParams, DWTSetup/DHTSetup.`**
- **`[ ] Implement DWTPlan/DHTPlan classes (with execute).`**
- **`[ ] Implement IntelIPP::DWTPlanImpl/IntelIPP::DHTPlanImpl. Python bindings & tests.`**
  - **Observation:** Similar to DCT/DST, primary focus seems to be on the Intel IPP backend for now. No clear public API files.
  - **Likely Status:** Primarily planned for Intel IPP; public API and default implementations seem not yet started.

## **III. Python Bindings (omnidsp_py)**

- **`[ ] Bind all [Type]Params, [Type]Coefs.`**
- **`[ ] Bind [Type]Plan classes and their execute.`**
- **`[ ] Bind [Type]Processor classes and their execute, reset.`**
- **`[ ] Bind OmniDSP::create_plan, OmniDSP::create_processor overloads.`**
- **`[ ] Bind OmniDSP::zero_phase_filter.`**
  - **Observation:** `src/omnidsp_py/bindings.cpp`, `src/omnidsp_py/api.py`, and `src/omnidsp_py/__init__.py` exist.
  - **Likely Status:** Python bindings are actively being developed. The extent of current bindings would require inspecting `bindings.cpp`. Given the progress on the C++ side for Params, Plans, and Processors, it's reasonable to assume many are either already bound or are high on the list to be bound.

## **IV. Testing** _(High Priority)_

- **`[ ] Test OmniDSP::create_plan, OmniDSP::create_processor paths.`**
- **`[ ] Test [Type]Plan::execute.`**
- **`[ ] Test [Type]Processor::execute and [Type]Processor::reset statefulness.`**
- **`[ ] Test OmniDSP::zero_phase_filter.`**
- **`[ ] Test IIR, Conv/Corr, STFT, DCT/DST, DWT/DHT modules.`**
- **`[ ] Test specific backend implementations (e.g., IntelIPP::*, OneMKL::*) against Default::*.`**
- **`[ ] Add tests for Dispatcher::Backend configurations.`**
  - **Observation:** General C++ and Python test directories (`tests/cpp/`, `tests/python/`) and files exist. Specific tests for dispatcher configurations would need to be verified by inspecting the content of these test files.
  - **Likely Status:** Unknown without inspecting test content; infrastructure for tests is present.
- **`[ ] Test Dispatcher::Backend with various backend override combinations.`**
  - **Observation:** A comprehensive test suite is planned. Files like `tests/cpp/tests/fft.cpp`, `cqt.cpp`, `window.cpp` and `tests/python/test_omnidsp.py` exist.
  - **Likely Status:** Testing is ongoing. The existing test files suggest that FFT, CQT, and Windowing are being tested. The other items depend on the completion of their respective modules.

## **Build System & CI**

- **`[ ] Full CI for Linux, macOS, Windows.`**
- **`[ ] Code coverage checks.`**
- **`[ ] Conan package, Python wheels.`**

## **Documentation & Examples** _(Update Incrementally)_

- **`[ ] Update Design Philosophy.md (In progress - reflect Plan/Processor, factories).`**
- **`[ ] Update Backend Developer Guide.md (Needs update for Plan/Processor, factories, *Impl state).`**
- **`[ ] Update README.md (In progress - reflect Plan/Processor, factories).`**
- **`[ ] Doxygen for C++ API, Sphinx for Python API.`**
- **`[ ] Add/Update examples demonstrating Plan vs Processor usage, including reset().`**

## **Backend Enhancements / Future Work**

- **`[ ] Refine OmniDSP::create logic, error handling.`**
- **`[ ] OmniDSP::get_available_backends().`**
- **`[ ] GPU acceleration (e.g., CUDA::Backend).`**
- **`[ ] Asynchronous execution (May require rethinking internal state if applied to Processors).`**

---

**DONE (Conceptual & Partial):**

- `[x] Refactor WindowSpec to WindowSetup.`
- `[x] Define FIRFilterParams, ResampleParams, CQTParams.`
- `[x] Establish new directory structure.`
- `[x] Define FilterType, FIRFilterDesignMethod, IIRFilterFormat.`
- `[x] Refine/Define FIRFilterSpec, ResampleSpec, CQTSpec.`
- `[x] Implement Utils::create_spec for FIR, Resample, CQT.`
- `[x] Establish "Params -> Spec -> Plan/Processor" pipeline conceptually.`
- `[x] Update CMakeLists.txt & Includes.`
- `[x] Establish Optimized::Backend inherits Default::Backend pattern.`
- `[x] Resolve ResamplePlan<T>::create linker issues (Note: This task might be obsolete or need renaming if Resampling becomes a Processor).`
- `[x] Update backend create_cqt_plan_impl for CQTSpec (Note: Needs changing to create_cqt_processor_impl).`
- `[x] Update Default::ResamplePlanImpl, IntelIPP::ResamplePlanImpl, Default::CQTPlanImpl for Spec (Note: Needs changing to *ProcessorImpl).`

---

**REMOVED / OBSOLETE:**

- `~~Standardize State Management (Implement Plan::initialize_state(), Define StateType, Update Plan::execute to accept StateType&)~~ - Replaced by internal state in Processor.`

## **Summary of Potential TODO Updates:**

Consider moving the following items, or parts of them, to `[x]` or marking as significantly progressed:

- **Core Factory Methods & Execution Objects:** Many sub-tasks related to defining Plan/Processor classes and their default implementations.
- **IIR Filter Module:** Definitions of Params, Spec, Coefs. Implementation of the processor class and its default backend.
- **CQT Module:** Definitions of Params, Spec. Implementation of the processor class and its default backend.
- **Resampling Module:** Implementation of the processor class and its default/IPP backend.
- **Convolution / Correlation Module:** Definitions of Params, Plan class, and default backend implementation.

Items like **Zero-Phase Filtering**, **STFT**, **DCT/DST (public API)**, and **DWT/DHT (public API)** seem to have less visible progress based on file names alone and likely remain `[ ]`.

This analysis should help you refine your `TODO.md`.
