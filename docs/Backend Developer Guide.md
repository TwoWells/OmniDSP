## OmniDSP Backend Developer Guide (Post-Refactor)

This document provides guidelines for developers integrating new computation backends into the OmniDSP library, reflecting the architecture using distinct stateless `Plan` and stateful `Processor` objects, created via `create_plan`/`create_processor` factory methods.

Refer to the main `Design Philosophy.md` for overarching goals.

**1\. Overview: Directory Structure**

- **`interface/`**: Defines abstract contracts in `OmniDSP::abstract`.
  - `backend.hpp`: `Abstract::Backend` interface.
  - `plan_impl.hpp` (example): Abstract interfaces like `Abstract::FFTPlanImpl`, `Abstract::ConvolutionPlanImpl`.
  - `processor_impl.hpp` (example): Abstract interfaces like `Abstract::FIRFilterProcessorImpl`, `Abstract::ResampleProcessorImpl`.
  - `fft.cpp`, `filter.cpp`, etc.: Public `Plan`/`Processor` wrappers (`FFTPlan`, `FIRFilterProcessor`) using Pimpl.
- **`default/`**: Contains the default C++ backend in `OmniDSP::default`.
  - `backend.hpp/.cpp`: `Default::Backend` class (inherits `Abstract::Backend`).
  - `fft_plan.cpp`, `fir_filter_processor.cpp`, etc.: Implementations of `Default::*PlanImpl` and `Default::*ProcessorImpl`.
- **`dispatcher/`**: `Dispatcher::Backend` implementation (inherits `Abstract::Backend`).
- **`accelerate/`, `onemkl/`, `intelipp/`, `(your_namespace)/`**: Specific backend implementations (e.g., `Accelerate::Backend`).
  - **Must inherit `OmniDSP::Default::Backend`**.
  - Implementations of `YourNamespace::*PlanImpl` and `YourNamespace::*ProcessorImpl`.
- **`params/`**: `[Type]Params` source files.
- **`utils/`**: `Utils::create_spec` implementations.
- **`omnidsp.cpp`**: `OmniDSP` class implementation.

**2\. Overview: Backend Architecture**

- **`OmniDSP` (Public Class):** User-facing entry point. Holds `pimpl_` (`std::unique_ptr<Abstract::Backend>`). Created via `OmniDSP::create()`. Acts as factory via `create_plan`/`create_processor`.
- **`Abstract::Backend` (Interface):** Defines the pure virtual contract (`get_backend()`) and distinct virtual factory methods for creating plan/processor implementations:
  - `create_*_plan_impl(const SetupOrSpecOrCoefs&)` -> `OmniExpected<std::unique_ptr<Abstract::*PlanImpl>>`
  - `create_*_processor_impl(const SetupOrSpecOrCoefs&)` -> `OmniExpected<std::unique_ptr<Abstract::*ProcessorImpl>>`
    Also defines virtual methods for one-off operations (`convolve_impl`, etc.).
- **`Default::Backend` (Base Implementation):** Inherits `Abstract::Backend`. Provides standard C++ implementations for **all** virtual methods. **Base class for optimized backends.**
- **`YourNamespace::Backend` (Optimized Backend):** Inherits `OmniDSP::Default::Backend`. Overrides methods it optimizes (specific `create_*_plan_impl`, `create_*_processor_impl`, one-offs).
- **`Dispatcher::Backend` (Dispatching Strategy):** Inherits `Abstract::Backend`. Holds primary and override backend instances. Overrides all factory/one-off methods to route calls.
- **`Abstract::*PlanImpl` (Stateless Impl Interface):** Defines interface for backend-specific stateless plan implementations (e.g., `execute`).
- **`Abstract::*ProcessorImpl` (Stateful Impl Interface):** Defines interface for backend-specific stateful processor implementations (e.g., `execute`, `reset`, internal state management).
- **`YourNamespace::*PlanImpl` / `YourNamespace::*ProcessorImpl` (Concrete Impls):** Implement the respective interfaces using the backend's API.

**3\. Implementing the Core Backend Class (`YourNamespace::Backend`)**

- **Inheritance:** `namespace OmniDSP::your_namespace { class Backend final : public OmniDSP::Default::Backend { ... }; }`.
- **Override Virtual Functions:**
  - **`get_backend()`:** **Must** override.
  - **Plan Factories (`create_*_plan_impl`):** Override if your backend provides an optimized stateless `YourNamespace::*PlanImpl`. Implement overrides for relevant `Setup`, `Spec`, or `Coefs` inputs.
  - **Processor Factories (`create_*_processor_impl`):** Override if your backend provides an optimized stateful `YourNamespace::*ProcessorImpl`. Implement overrides for relevant `Setup`, `Spec`, or `Coefs` inputs.
  - **Coefficient/Resource Generation (`generate_*_coeffs`):** Override if optimized generation from `Spec` exists.
  - **One-Off Operations:** Override if specifically optimized.
- Use `override`. If a factory method is not overridden, the call falls through to `Default::Backend`.

**4\. Implementing Plan Implementation Classes (`YourNamespace::*PlanImpl`) - Stateless**

- **Inheritance:** Inherit from `Abstract::*PlanImpl<DataType>`.
- **Constructor:** Takes `Setup`, `Spec`, or `Coefs`. Performs backend setup, precomputation. Throws on failure.
- **Destructor:** Cleans up backend resources.
- **`execute(...)`:** Implements the core stateless DSP operation using your backend's API. Must be reentrant if intended for sharing across threads.

**5\. Implementing Processor Implementation Classes (`YourNamespace::*ProcessorImpl`) - Stateful**

- **Inheritance:** Inherit from `Abstract::*ProcessorImpl<DataType>`.
- **Internal State:** Define member variables to hold the necessary operational state (e.g., delay lines).
- **Constructor:** Takes `Setup`, `Spec`, or `Coefs`. Performs backend setup, precomputation, **and initializes internal state**. Throws on failure.
- **Destructor:** Cleans up backend resources and state.
- **`execute(...)`:** Implements the core stateful DSP operation, **reading and modifying internal state**.
- **`reset()`:** Implements the logic to reset the internal state variables to their initial conditions.

**6\. The `Dispatcher::Backend`**

- **Location:** `src/omnidsp/dispatcher/backend.hpp` and `.cpp`.
- **Inheritance:** `class DispatcherBackend final : public OmniDSP::Abstract::Backend { ... };`.
- **Constructor:** Takes primary `Abstract::Backend` and map of `OperationCategory` to override `Abstract::Backend` instances.
- **Override All Factory/One-Off Methods:** For every `create_*_plan_impl`, `create_*_processor_impl`, and one-off method in `Abstract::Backend`:
  1. Determine the `OperationCategory`.
  2. Check override map.
  3. Forward call to override backend instance or primary backend instance.

**7\. Integrating with the Factory (`OmniDSP::create`)**

- (Remains the same as previous version - `OmniDSP::create` handles instantiation of single backends or the `Dispatcher::Backend`).

**8\. Integrating with the Build System (CMake)**

- (Remains the same as previous version - `Dispatcher::Backend` is a core component).

**9\. Adhering to API Conventions**

- Use `std::span`, `OmniExpected`, error handling, exceptions.
- `*PlanImpl` / `*ProcessorImpl` constructors throw exceptions on failure. Factory methods (`create_*_plan_impl`, `create_*_processor_impl`) catch these and return `std::unexpected`.
- Ensure `*ProcessorImpl` correctly manages internal state and implements `reset()`.

This structure provides clear separation between stateless (`Plan`) and stateful (`Processor`) operations, reflected both in the user-facing API (`create_plan`/`create_processor`) and the backend implementation interfaces (`Abstract::*PlanImpl`/`Abstract::*ProcessorImpl`).
