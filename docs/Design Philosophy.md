# **OmniDSP Design Philosophy (Post-Refactor)**

The OmniDSP library is designed around core concepts and patterns to achieve portability, performance, consistency, and ease of use. This document reflects the May 10, 2025 Refactor Proposal, advanced backend configuration, a fluent parameter model, and a clear distinction between stateless `Plan` objects and stateful `Processor` objects, created via distinct factory methods.

## **Core Components: The `OmniDSP` Class, Configuration (`Params`, `Coefs`), Intermediate Representations (`Setup`, `Spec`), `Plan` Objects, and `Processor` Objects**

The primary user workflow involves:

1. Configuring an operation using a `[Type]Params` object (or providing pre-calculated `[Type]Coefs`).
2. Calling the appropriate factory method on the `OmniDSP` instance:
   - `OmniDSP::create_plan(...)` for stateless operations (e.g., FFT).
   - `OmniDSP::create_processor(...)` for stateful operations (e.g., Filtering).
     These methods accept either `[Type]Params` or `[Type]Coefs`.
   - Internally, if `Params` are provided, they are converted into an intermediate, validated `[Type]Setup` or `[Type]Spec` object.
3. The `OmniDSP` instance then uses this `Setup`, `Spec`, or `Coefs` object to create an optimized execution object (`[Type]Plan` or `[Type]Processor`) via the selected backend.
   - **`[Type]Processor` objects** initialize and manage their own internal state upon creation.
4. The user calls `execute(...)` on the returned `Plan` or `Processor` object to process data. For **`[Type]Processor` objects**, this method modifies the object's internal state.
5. The user can call `reset()` on a **`[Type]Processor` object** to reset its internal state, allowing it to be reused for a new, independent data stream.

### **1. The `OmniDSP` Class: Backend Encapsulation and Factory**

- **Purpose:** The primary user-facing entry point, encapsulating the selected backend implementation(s).
- **Backend Management & Instantiation (`OmniDSP::create`)**:
  - Users create an instance using static `OmniDSP::create` factory methods, which can select a single backend or configure a `Dispatcher::Backend` for per-operation-category overrides. The `pimpl_` of `OmniDSP` points to the active `Abstract::Backend` implementation.
- **Factory Role (`create_plan` / `create_processor`)**: The `OmniDSP` instance acts as the central factory for creating execution objects. It provides distinct methods:
  - **`create_plan(...)`**: Overloaded methods accepting `const [Type]Params&` or `const [Type]Coefs&` for operations resulting in a stateless `[Type]Plan`. Returns `OmniExpected<[Type]Plan>`.
  - **`create_processor(...)`**: Overloaded methods accepting `const [Type]Params&` or `const [Type]Coefs&` for operations resulting in a stateful `[Type]Processor`. Returns `OmniExpected<[Type]Processor>`.
    Internally, these methods handle the conversion from `Params` to `Setup`/`Spec` before dispatching to the appropriate backend implementation creation method.
- **One-Off Operations & Helpers:** Provides methods for simple operations (e.g., `convolve`) and helpers (e.g., `generate_window_coeffs`), routed through its `Abstract::Backend` Pimpl.

### **`OperationCategory` Enum (for Dispatcher)**

```cpp
// Example: Potentially in types/operation.hpp or core_types.hpp
enum class OperationCategory {
    FFT, Convolution, Correlation, FIRFilter, IIRFilter, Resampling, CQT, Windowing,
    GenericFallback
};
```

### **2. User Configuration & Internal Transformation**

- **`[Type]Params` (Struct/Class - Fluent Builder for Operation Parameters)**:

  - **Purpose:** The **primary user interface** for defining parameters for an operation. Supports a fluent interface.
  - **Location:** `include/OmniDSP/params/[type].hpp`.
  - **Validation:** Comprehensive validation performed during internal conversion to `Setup` or `Spec`.
  - **Internal Conversion Methods (Not typically called directly by user):**
    - `to_setup() -> OmniExpected<[Type]Setup>`: Converts `Params` to `[Type]Setup`.
    - `to_spec() -> OmniExpected<[Type]Spec>`: Converts `Params` to `[Type]Spec` (may invoke `Utils::create_spec`).
  - **User Workflow:** User creates/configures `[Type]Params`, then calls `dsp.create_plan(params_object)` or `dsp.create_processor(params_object)`.

- **`[Type]Coefs` (Class/Struct - Pre-Calculated Data for Plans/Processors)**:

  - **Purpose:** Encapsulates pre-calculated, validated coefficients/kernels, bypassing `Params`-based design.
  - **Location:** Relevant public headers (e.g., `OmniDSP/filter.hpp`).
  - **User Workflow:** User creates/obtains `[Type]Coefs`, then calls `dsp.create_plan(coeffs_object)` or `dsp.create_processor(coeffs_object)`.

- **`WindowSetup` (Struct - Embedded Configuration)**:
  - **Purpose:** Defines window characteristics. Used in `[Type]Params` or for direct window generation.
  - **Location:** `OmniDSP/window.hpp`.

### **3. Intermediate Representations (`Setup` and `Spec` - Internal Library Detail)**

Crucial for library architecture and backend contracts, though abstracted from the user at the factory method call site:

- **`[Type]Setup` (Struct - Internal Direct Configuration)**:

  - **Purpose (Internal):** Holds validated, simple configuration parameters for creating a `[Type]Plan` or `[Type]Processor` by a backend.
  - **Creation (Internal):** By `[Type]Params::to_setup()`. Contract for backend methods like `Abstract::Backend::create_fft_plan_impl(const FFTSetup& setup)`.

- **`[Type]Spec` (Struct - Internal Resolved Design Specification)**:
  - **Purpose (Internal):** Represents the fully resolved, detailed, validated specification for complex operations.
  - **Creation (Internal):** By `[Type]Params::to_spec()`. Contract for backend methods like `Abstract::Backend::create_fir_filter_processor_impl(const FIRFilterSpec& spec)`.

### **4. Utility Functions (`Utils::create_spec`)**

- **Purpose:** Encapsulates complex design algorithms.
- **Invocation:** Typically called internally by `[Type]Params::to_spec()` methods. Signature likely `Utils::create_spec(const [Type]Params& params_obj)`.
- **Location:** `include/OmniDSP/utils.hpp` (declarations), `src/omnidsp/utils/` (implementations).

### **5. Execution Objects: `[Type]Plan` (Stateless) and `[Type]Processor` (Stateful)**

- **`[Type]Plan` (Stateless Execution Object - e.g., `FFTPlan`, `ConvolutionPlan`)**

  - **Purpose:** Encapsulates the **immutable context** and pre-computed data for efficient, stateless execution. Created by `OmniDSP::create_plan`.
  - **Characteristics:** Stateless, immutable resources, factory-created, backend-dependent implementation (`Abstract::*PlanImpl`), optimized `execute` method, no state-related `reset()` method.
  - **Thread Safety:** Instances are inherently thread-safe for concurrent `execute` calls if the backend implementation of `execute` is reentrant. Can be safely shared among threads.

- **`[Type]Processor` (Stateful Execution Object - e.g., `FIRFilterProcessor`, `ResampleProcessor`)**
  - **Purpose:** Encapsulates the **immutable context**, pre-computed data, and the **mutable internal state** for efficient, stateful execution. Created by `OmniDSP::create_processor`.
  - **Characteristics:** Stateful (manages internal state modified by `execute`), immutable resources, factory-created, backend-dependent implementation (`Abstract::*ProcessorImpl` or similar), optimized `execute` method, provides `reset()` method to re-initialize internal state.
  - **Thread Safety:** Instances are **NOT thread-safe** for concurrent `execute()` calls on the _same instance_. Each thread processing an independent stream requires its own `[Type]Processor` instance.

### **6. State Objects (Mutable Execution State) - OBSOLETE**

The concept of a separate, user-managed `State` object is now obsolete. State, when required, is managed internally by **`[Type]Processor` objects**.

## **Key Design Patterns and Techniques**

- **Builder/Fluent Interface:** `[Type]Params` objects for user configuration.
- **Pimpl Idiom:** In `OmniDSP`, `[Type]Plan`, and `[Type]Processor` classes.
- **Backend Interface Contract (`Abstract::Backend`, `Abstract::*PlanImpl`, `Abstract::*ProcessorImpl`):** Defines clear contracts. `Abstract::Backend` will have distinct `create_*_plan_impl` and `create_*_processor_impl` methods. The implementation classes (`*PlanImpl`/`*ProcessorImpl`) encapsulate backend logic and state management (for Processors).
- **Factory Pattern:** `OmniDSP::create()` (for `OmniDSP`), `OmniDSP::create_plan()` (for `[Type]Plan`), `OmniDSP::create_processor()` (for `[Type]Processor`).
- **Strategy Pattern (via Backends):** `Dispatcher::Backend` as a composite strategy.
- **Implementation Inheritance (from `Default::Backend`):** For optimized backends.
- **RAII, `std::expected` / `Status` Enum.**

## **The `Dispatcher::Backend`**

- **Role:** A specialized implementation of `Abstract::Backend` that delegates calls to other concrete `Abstract::Backend` instances based on a runtime configuration.
- **Construction:** Instantiated by `OmniDSP::create` when backend overrides are provided.
- **Operation:** When a factory method like `create_fft_plan_impl` or `create_fir_filter_processor_impl` is called on the `Dispatcher::Backend`:
  1. It identifies the `OperationCategory` (e.g., `FFT`, `FIRFilter`).
  2. Checks its configuration for an override for this category.
  3. If an override exists, it forwards the call to the corresponding specialized `Abstract::Backend` instance.
  4. If no override exists, it forwards the call to its primary/default `Abstract::Backend` instance.
- **Inheritance:** Inherits directly from `Abstract::Backend`.

## **Performance Considerations**

- **Backend Strategy:** Primary mechanism, enhanced by `Dispatcher::Backend`.
- **Dispatcher Overhead:** Small, generally negligible.
- **Cost of Object Creation:** Setup cost for **`[Type]Processor` instances** is incurred per thread in parallel scenarios.
- **Plan/Processor Object Optimization:** Amortizes setup costs.
- **User-Managed Memory (`std::span`), Memory Alignment:** Crucial for performance.

## **Build System Philosophy (CMake)**

The `Dispatcher::Backend` integrates with the CMake-based build system as a core architectural element:

- **Core Library Component:** The source files for `Dispatcher::Backend` (e.g., `src/omnidsp/dispatcher/backend.cpp` and its header) are integral to the OmniDSP library. They are included in the main `omnidsp` library target's source list within `CMakeLists.txt`.
- **No Dedicated CMake Option for Dispatcher:** A specific CMake option like `OMNIDSP_ENABLE_DISPATCHER` is unnecessary. The `Dispatcher::Backend` serves as a mechanism for utilizing other available backends, rather than being an optional backend dependent on external libraries. It is compiled as part of the OmniDSP library itself.
- **Relies on Existing Backend Enablement:** The functionality of `Dispatcher::Backend` depends on its ability to dispatch to various concrete backend implementations (such as `OneMKL::Backend`, `IntelIPP::Backend`, `Accelerate::Backend`, or `Default::Backend`). The established CMake options (e.g., `OMNIDSP_ENABLE_ONEMKL`, `OMNIDSP_ENABLE_INTELIPP`) and their associated `find_package` logic in the `cmake/backend/` scripts are essential. The `Dispatcher::Backend` can only route operations to backends that are successfully detected, enabled, and compiled into the library.
- **Conditional Compilation of Dispatch Logic:** While the `Dispatcher::Backend` itself is compiled unconditionally, the `OmniDSP::create` factory method responsible for its instantiation may employ preprocessor checks. These checks, driven by CMake-set definitions (e.g., `OMNIDSP_ENABLED_ONEMKL`), ensure that it only attempts to create and configure dispatch targets for backends actually available in the current build configuration.
- **`OperationCategory` Enum:** The `OperationCategory` enum, along with any related type definitions necessary for configuring the dispatcher, are also part of the core library headers and are compiled unconditionally.
- **Dependency Management:** The `Dispatcher::Backend` itself does not introduce new external dependencies. Its dependencies are implicitly those of the concrete backends to which it might dispatch operations.
- **Target Definitions:** The CMake target for the `omnidsp` library simply includes the `dispatcher` source files. No special linking or interface properties are required for the dispatcher beyond those of the core library.

The build system's role is to ensure that all potential concrete backend targets for the dispatcher are available if their respective dependencies are met, and to compile the dispatcher mechanism as a fundamental part of the library. The runtime configuration provided to `OmniDSP::create` then dictates how these available components are interconnected and utilized.

## **Other Design Considerations**

- **Concurrency and Thread Safety (Revisited):**
  - **`[Type]Plan` objects (Stateless):** Shareable if `execute` is reentrant.
  - **`[Type]Processor` objects (Stateful):** **NOT thread-safe for concurrent `execute` on the same instance.** Requires separate instances per thread for parallel streams.
  - `OmniDSP` instance requires external synchronization if used concurrently.
- **Testing:** Strategy needs to cover `create_plan`, `create_processor`, and the behavior of both `[Type]Plan` and `[Type]Processor` objects (including `reset`). Dispatcher combinations also need testing.
- **Data Type Extensibility:** The library primarily targets `float` and `double` precision floating-point types via templates. Support for new data types should ideally be added to `Default::Backend` first. Optimized backends can then optionally override this if they offer specialized support for types like `float16` or fixed-point, implementing the necessary template specializations for their `*PlanImpl` or `*ProcessorImpl` classes.
- **Backend Discovery:** Static functions `OmniDSP::get_available_backends() -> std::vector<BackendType>` and `OmniDSP::get_preferred_backend() -> BackendType` should be provided. `get_available_backends` allows users to query at runtime which concrete backends (e.g., `BackendType::OneMKL`, `BackendType::IntelIPP`, `BackendType::Accelerate`, `BackendType::Default`) were successfully compiled and detected as available on the current system. `get_preferred_backend` indicates the default backend that `OmniDSP::create()` would select if no specific type is requested, based on a predefined preference order (e.g., platform-specific optimized backends preferred over `Default::Backend`).

## **Rationale Summary**

This design, using distinct **`create_plan`** and **`create_processor`** methods to produce stateless **`[Type]Plan`** and stateful **`[Type]Processor`** objects respectively, aims for a DSP library that is:

1.  **Highly Configurable & Flexible.**
2.  **Consistent & Predictable:** Unified configuration via `Params`; clear factory methods.
3.  **Performant.**
4.  **Portable.**
5.  **Easy to Use:** Fluent interface for parameters; explicit API for creating stateless vs. stateful objects.
6.  **Maintainable & Extensible:** Clear internal roles and backend contracts.
7.  **Robust.**
8.  **Clear (Re: Concurrency):** Explicit guidance on thread safety based on object type (`Plan` vs. `Processor`).
