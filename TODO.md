# **OmniDSP Project TODO List (Post-Refactor Proposal)**

**Goal:** Outline development tasks reflecting the `Plan` (stateless) vs. `Processor` (stateful) distinction, `create_plan`/`create_processor` factories, and internal state management for Processors.

**Directory Structure Reminder:**

- `include/OmniDSP/`: Public API headers (`OmniDSP`, `FFTPlan`, `FIRFilterProcessor`, etc.).
- `include/OmniDSP/params/`: `[Type]Params` structs.
- `include/OmniDSP/types/`: Domain-specific enums (`OperationCategory`, etc.).
- `src/omnidsp/interface/`: `Abstract::Backend`, `Abstract::*PlanImpl`, `Abstract::*ProcessorImpl`, public `Plan`/`Processor` wrappers.
- `src/omnidsp/default/`: `Default::Backend`, `Default::*PlanImpl`, `Default::*ProcessorImpl`.
- `src/omnidsp/dispatcher/`: `Dispatcher::Backend`.
- `src/omnidsp/{accelerate, onemkl, intelipp}/`: Specific backends (e.g., `Accelerate::Backend`).
- `src/omnidsp/params/`: `Params` constructors/fluent methods.
- `src/omnidsp/utils/`: `Utils::create_spec` implementations.
- `src/omnidsp/omnidsp.cpp`: `OmniDSP` class implementation.
- `src/omnidsp_py/`: Python bindings.

---

## **High Priority / Blocking Tasks**

- **Implement `Dispatcher::Backend` & Advanced Configuration:**
  - \[ \] Define `OperationCategory` enum.
  - \[ \] Implement `Dispatcher::Backend` class.
  - \[ \] Implement `OmniDSP::create` overloads supporting overrides.
  - \[ \] Add tests for `Dispatcher::Backend` configurations.
- **Implement Core Factory Methods & Execution Objects:**
  - \[ \] Implement `OmniDSP::create_plan` overloads (taking `Params`/`Coefs`) for stateless operations (FFT, Conv?, Corr?).
  - \[ \] Implement `OmniDSP::create_processor` overloads (taking `Params`/`Coefs`) for stateful operations (FIR, IIR, Resample, CQT?).
  - \[ \] Define public `[Type]Plan` classes (e.g., `FFTPlan`) with `execute`.
  - \[ \] Define public `[Type]Processor` classes (e.g., `FIRFilterProcessor`) with `execute` and `reset`.
  - \[ \] Add `create_*_plan_impl` virtual methods to `Abstract::Backend`.
  - \[ \] Add `create_*_processor_impl` virtual methods to `Abstract::Backend`.
  - \[ \] Implement these factory methods in `Default::Backend` and optimized backends.
  - \[ \] Implement `Default::*PlanImpl` classes (stateless).
  - \[ \] Implement `Default::*ProcessorImpl` classes (stateful, internal state, `reset` logic).
- **Complete `Params` -> `Setup`/`Spec` Internal Flow:**
  - \[ \] Finalize `[Type]Params` fluent interfaces.
  - \[ \] Implement internal `Params::to_setup()` / `Params::to_spec()` methods.
  - \[ \] Ensure validation logic is called correctly during conversion.
  - \[ \] Refine/Confirm role of `Utils::create_spec` (likely called by `Params::to_spec`).
- **Implement IIR Filter Module:**
  - \[ \] Define `IIRFilterParams`, `IIRFilterSpec`, `IIRFilterCoefs`.
  - \[ \] Implement `IIRFilterProcessor` class (with `execute`, `reset`).
  - \[ \] Implement `create_iir_filter_processor_impl` in `Abstract::Backend` / `Default::Backend`.
  - \[ \] Python bindings and tests for `IIRFilterProcessor`.
- **Implement Zero-Phase Filtering (`filtfilt`):**
  - \[ \] Add `OmniDSP::zero_phase_filter` methods (likely stateless one-off).
  - \[ \] Add `zero_phase_filter_impl` to `Abstract::Backend`.
  - \[ \] Implement in `Default::Backend`.
  - \[ \] Python bindings and tests.
- **Define Core Enums:**
  - \[ \] Define `PaddingMode` enum.
  - \[ \] Move `ConvolutionType`, `ConvolutionMethod` to `types/convolution.hpp`.

## **Core Library Features**

### **Convolution / Correlation Module** (Likely Plan - Stateless)

- \[ \] Define `ConvolutionParams`, `CorrelationParams`, `ConvolutionSetup`?, `CorrelationSetup`?, `ConvolutionCoefs<T>`, `CorrelationCoefs<T>`.
- \[ \] Implement `ConvolutionPlan`, `CorrelationPlan` classes (with `execute`).
- \[ \] Implement `OmniDSP::create_plan` for Conv/Corr.
- \[ \] Implement `create_convolution_plan_impl`, `create_correlation_plan_impl` in `Abstract::Backend` / `Default::Backend`.
- \[ \] Implement `Default::ConvolutionPlanImpl`, `Default::CorrelationPlanImpl`.
- \[ \] Implement one-off `convolve_*`/`correlate_*` in `OmniDSP`.
- \[ \] (Opt) `OneMKL::ConvolutionPlanImpl` / `OneMKL::CorrelationPlanImpl`.
- \[ \] (Opt) `IntelIPP::ConvolutionPlanImpl` / `IntelIPP::CorrelationPlanImpl`.
- \[ \] Python bindings and tests.

### **Default Backend Highway Refactor** _(Ongoing)_

- \[ \] Continue applying Highway SIMD to `Default::Backend` implementations.

### **Intel IPP Backend Enhancements (`IntelIPP::Backend`)**

- \[ \] Arbitrary Length FFT/DFT in `IntelIPP::FFTPlanImpl`.
- \[ \] `IntelIPP::DCTPlanImpl`/`IntelIPP::DSTPlanImpl`.
- \[ \] `IntelIPP::DWTPlanImpl`/`IntelIPP::DHTPlanImpl`.
- \[ \] Implement `IntelIPP::FIRFilterProcessorImpl`, `IntelIPP::IIRFilterProcessorImpl`, `IntelIPP::ResampleProcessorImpl` (including internal state and `reset`).

### **CQT Module** (Likely Processor - Stateful due to recursive nature/overlap)

- \[ \] Define `CQTParams`, `CQTSpec`.
- \[ \] Implement `CQTProcessor` class (with `execute`, `reset`).
- \[ \] Implement `create_cqt_processor_impl` in `Abstract::Backend` / `Default::Backend`.
- \[ \] Review/Implement `Default::CQTProcessorImpl` constructor taking `CQTSpec`, managing internal state, and `reset`.
- \[ \] Python bindings and tests for `CQTProcessor`.

### **Resampling Module** (Processor - Stateful)

- \[ \] Define `ResampleParams`, `ResampleSpec`. (Already Done?)
- \[ \] Implement `ResampleProcessor` class (with `execute`, `reset`).
- \[ \] Implement `create_resample_processor_impl` in `Abstract::Backend` / `Default::Backend`.
- \[ \] Review/Implement `Default::ResampleProcessorImpl` managing internal state (e.g., filter state, phase) and `reset`.
- \[ \] Review/Implement `IntelIPP::ResampleProcessorImpl`.
- \[ \] Python bindings and tests for `ResampleProcessor`.

### **STFT Module** (Likely Plan - Stateless, assumes user handles framing/overlap)

- \[ \] Design `STFTParams`, `STFTSetup`.
- \[ \] Implement `STFTPlan` class (with `execute`).
- \[ \] Add `create_stft_plan_impl` to `Abstract::Backend`. Python bindings & tests.

### **DCT/DST Module** (Likely Plan - Stateless transform)

- \[ \] Design `DCTParams`/`DSTParams`, `DCTSetup`/`DSTSetup`.
- \[ \] Implement `DCTPlan`/`DSTPlan` classes (with `execute`).
- \[ \] Implement `OneMKL::DCTPlanImpl`/`OneMKL::DSTPlanImpl`.
- \[ \] Implement `IntelIPP::DCTPlanImpl`/`IntelIPP::DSTPlanImpl`.
- \[ \] Add `create_dct_plan_impl`/`create_dst_plan_impl` to `Abstract::Backend`. Python bindings & tests.

### **DWT/DHT Module** (Likely Plan - Stateless transform)

- \[ \] Design `DWTParams`/`DHTParams`, `DWTSetup`/`DHTSetup`.
- \[ \] Implement `DWTPlan`/`DHTPlan` classes (with `execute`).
- \[ \] Implement `IntelIPP::DWTPlanImpl`/`IntelIPP::DHTPlanImpl`. Python bindings & tests.

## **Python Bindings (omnidsp_py)**

- \[ \] Bind all `[Type]Params`, `[Type]Coefs`.
- \[ \] Bind `[Type]Plan` classes and their `execute`.
- \[ \] Bind `[Type]Processor` classes and their `execute`, `reset`.
- \[ \] Bind `OmniDSP::create_plan`, `OmniDSP::create_processor` overloads.
- \[ \] Bind `OmniDSP::zero_phase_filter`.

## **Build System & CI**

- \[ \] Full CI for Linux, macOS, Windows.
- \[ \] Code coverage checks.
- \[ \] Conan package, Python wheels.

## **Testing** _(High Priority)_

- \[ \] Test `OmniDSP::create_plan`, `OmniDSP::create_processor` paths.
- \[ \] Test `[Type]Plan::execute`.
- \[ \] Test `[Type]Processor::execute` and `[Type]Processor::reset` statefulness.
- \[ \] Test `OmniDSP::zero_phase_filter`.
- \[ \] Test IIR, Conv/Corr, STFT, DCT/DST, DWT/DHT modules.
- \[ \] Test specific backend implementations (e.g., `IntelIPP::*`, `OneMKL::*`) against `Default::*`.
- \[ \] Test `Dispatcher::Backend` with various backend override combinations.

## **Documentation & Examples** _(Update Incrementally)_

- \[ \] **Update `Design Philosophy.md`** (In progress - reflect `Plan`/`Processor`, factories).
- \[ \] **Update `Backend Developer Guide.md`** (Needs update for `Plan`/`Processor`, factories, `*Impl` state).
- \[ \] **Update `README.md`** (In progress - reflect `Plan`/`Processor`, factories).
- \[ \] Doxygen for C++ API, Sphinx for Python API.
- \[ \] Add/Update examples demonstrating `Plan` vs `Processor` usage, including `reset()`.

## **Backend Enhancements / Future Work**

- \[ \] Refine `OmniDSP::create` logic, error handling.
- \[ \] `OmniDSP::get_available_backends()`.
- \[ \] GPU acceleration (e.g., `CUDA::Backend`).
- \[ \] Asynchronous execution (May require rethinking internal state if applied to Processors).

---

**DONE (Conceptual & Partial):**

- \[x] Refactor `WindowSpec` to `WindowSetup`.
- \[x] Define `FIRFilterParams`, `ResampleParams`, `CQTParams`.
- \[x] Establish new directory structure.
- \[x] Define `FilterType`, `FIRFilterDesignMethod`, `IIRFilterFormat`.
- \[x] Refine/Define `FIRFilterSpec`, `ResampleSpec`, `CQTSpec`.
- \[x] Implement `Utils::create_spec` for FIR, Resample, CQT.
- \[x] Establish "Params -> Spec -> Plan/Processor" pipeline conceptually.
- \[x] Update CMakeLists.txt & Includes.
- \[x] Establish `Optimized::Backend` inherits `Default::Backend` pattern.
- \[x] Resolve `ResamplePlan<T>::create` linker issues (Note: This task might be obsolete or need renaming if Resampling becomes a Processor).
- \[x] Update backend `create_cqt_plan_impl` for `CQTSpec` (Note: Needs changing to `create_cqt_processor_impl`).
- \[x] Update `Default::ResamplePlanImpl`, `IntelIPP::ResamplePlanImpl`, `Default::CQTPlanImpl` for `Spec` (Note: Needs changing to `*ProcessorImpl`).

---

**REMOVED / OBSOLETE:**

- ~~Standardize State Management (Implement `Plan::initialize_state()`, Define `StateType`, Update `Plan::execute` to accept `StateType&`)~~ - Replaced by internal state in `Processor`.
