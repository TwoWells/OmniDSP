# **OmniDSP Design Philosophy**

The OmniDSP library is designed around core concepts and patterns to achieve portability, performance, consistency, and ease of use. This document reflects the latest architectural decisions, including the "Window as Plan" model, advanced backend configuration, a fluent parameter model, and a clear distinction between stateless `Plan` objects and stateful `Processor` objects, created via distinct factory methods.

## **Core Components: The `OmniDSP` Class, Configuration (`Params`, `Window::[SpecType]`, `Coefs`), Intermediate Representations (`Design`), `Plan` Objects, and `Processor` Objects**

The primary user workflow involves:

1.  Creating an `OmniDSP` instance, which manages the selected backend.
2.  Configuring an operation:
    - Using a `Params::[Type]` object for most operations (e.g., `Params::FFT`, `Params::FIRFilter`).
    - Using a `Window::[SpecType]` object (e.g., `Window::Kaiser`, `Window::Hann`) along with a length for windowing operations.
    - Optionally, providing pre-calculated `Coefs::[Type]` to bypass design stages.
3.  Calling the appropriate factory method on the `OmniDSP` instance:
    - `OmniDSP::create_plan(...)` for stateless operations like FFT, Convolution, and Correlation. This returns an `OmniExpected<Plan::[Type]>`.
    - `OmniDSP::create_window_plan<T_Data>(...)` for stateless windowing operations. This returns an `OmniExpected<Plan::Window<T_Data>>`.
    - `OmniDSP::create_processor(...)` for stateful operations (e.g., Filtering, CQT, Resampling). This returns an `OmniExpected<Processor::[Type]>`.
4.  Internally, if `Params` are provided for design-based operations (like filters), they are converted into an intermediate, validated `Design::[Type]` object via `OmniDSP::Design::create`.
5.  The `OmniDSP` instance then uses the configuration (`Params`, `Window::[SpecType]`, `Coefs`, or `Design`) to create an optimized execution object (`Plan::[Type]` or `Processor::[Type]`) via the selected backend.
    - **`Processor::[Type]` objects** initialize and manage their own internal state upon creation.
6.  The user calls `execute(...)` on the returned `Plan` or `Processor` object to process data. For **`Processor::[Type]` objects**, this method modifies the object's internal state.
7.  The user can call `reset()` on a **`Processor::[Type]` object** to reset its internal state, allowing it to be reused for a new, independent data stream.

### **1. The `OmniDSP` Class: Backend Encapsulation and Factory**

- **Purpose:** The primary user-facing entry point, encapsulating the selected backend implementation(s).
- **Backend Management & Instantiation (`OmniDSP::create`)**:
  - Users create an instance using static `OmniDSP::create` factory methods, which can select a single backend or configure a `Dispatcher::Backend` for per-operation-category overrides. The `pimpl_` of `OmniDSP` points to the active `Abstract::Backend` implementation.
- **Factory Role (`create_plan` / `create_processor` / `create_window_plan`)**: The `OmniDSP` instance acts as the central factory for creating execution objects. It provides distinct methods:
  - **`create_plan(...)`**: Overloaded methods accepting `const Params::[Type]&` or `const Coefs::[Type]&` for operations like FFT, Convolution, and Correlation, resulting in a stateless `Plan::[Type]`. Returns `OmniExpected<Plan::[Type]>`.
  - **`create_window_plan<T_Data>(const Window::[SpecType]& spec, size_t length)`**: Overloaded methods (per window specification type) for creating stateless `Plan::Window<T_Data>`. Returns `OmniExpected<Plan::Window<T_Data>>`.
  - **`create_processor(...)`**: Overloaded methods accepting `const Params::[Type]&` or `const Coefs::[Type]&` for operations resulting in a stateful `Processor::[Type]` (e.g., FIR/IIR Filters, CQT, Resampling). Returns `OmniExpected<Processor::[Type]>`.
    Internally, these methods handle the conversion from `Params` to `Design` (for design-based operations) before dispatching to the appropriate backend implementation creation method.
- **One-Off Operations & Helpers:** Provides methods for simple operations (e.g., `convolve`) and helpers (e.g., `generate_window_coefficients`), routed through its `Abstract::Backend` Pimpl.
  - **`generate_window_coefficients<T_Data>(const Window::[SpecType]& spec, size_t length)`**: Convenience wrappers that internally create a temporary `Plan::Window` to generate and return coefficients.

### **`OperationCategory` Enum (for Dispatcher)**

```cpp
// Example: Potentially in types/operation.hpp or core_types.hpp
enum class OperationCategory {
    FFT, Convolution, Correlation, FIRFilter, IIRFilter, Resampling, CQT, Windowing,
    GenericFallback // etc.
};
```

### **2. User Configuration & Internal Transformation**

- **`Params::[Type]` (Struct/Class - Fluent Builder for Operation Parameters)**:

  - **Purpose:** The **primary user interface** for defining parameters for an operation (excluding windowing, which uses `Window::[SpecType]`). Supports a fluent interface.
  - **Location:** `include/OmniDSP/params/[type].hpp`.
  - **Validation:** Comprehensive validation performed during internal conversion to `Design` objects (for design-based operations) or directly by backend plan/processor creation.
  - **Internal Conversion to `Design` (Not typically called directly by user):**
    - For operations like Filters, CQT, Resampling, `Params::[Type]` are converted to `Design::[Type]` via `OmniDSP::Design::create(const Params::[Type]&)`.
  - **User Workflow:** User creates/configures `Params::[Type]`, then calls `dsp.create_plan(params_object)` or `dsp.create_processor(params_object)`.

- **`Window::[SpecType]` (Struct - Lightweight Window Specifications)**:

  - **Purpose:** User-facing, lightweight, non-hierarchical structs for defining window types and their specific parameters (e.g., `Window::Hann{}`, `Window::Kaiser{double beta}`).
  - **Location:** `include/OmniDSP/window/specs.hpp` (or individual headers like `kaiser_spec.hpp`).
  - **Validation:** Constructors of these specification structs (e.g., `Kaiser`) validate their parameters.
  - **User Workflow:** User creates a `Window::[SpecType]` object (e.g., `Window::Kaiser k_spec(5.0);`) and passes it along with the desired length to `dsp.create_window_plan<float>(k_spec, 1024)` or `dsp.generate_window_coefficients<float>(k_spec, 1024)`. Also used within `Params::FIRFilter` etc. where a window is part of a larger design.

- **`Coefs::[Type]` (Class/Struct - Pre-Calculated Data for Plans/Processors)**:
  - **Purpose:** Encapsulates pre-calculated, validated coefficients/kernels, bypassing `Params`-based design stages.
  - **Location:** Relevant public headers (e.g., `OmniDSP/fir_filter.hpp`).
  - **User Workflow:** User creates/obtains `Coefs::[Type]`, then calls `dsp.create_plan(coeffs_object)` or `dsp.create_processor(coeffs_object)`.

### **3. Intermediate Representations (`Design` - Internal Library Detail)**

Crucial for library architecture and backend contracts for design-based operations, though abstracted from the user at the factory method call site:

- **`Design::[Type]` (Struct - Internal Resolved Design Specification)**:
  - **Purpose (Internal):** Represents the fully resolved, detailed, validated specification for complex operations like filters, CQT, or resampling.
  - **Creation (Internal):** By `OmniDSP::Design::create(const Params::[Type]&)`. This is the contract for backend methods like `Abstract::Backend::create_fir_filter_processor_impl(const Design::FIRFilter& design)`.

### **4. Utility Functions (`OmniDSP::Design::create`)**

- **Purpose:** Encapsulates complex design algorithms (e.g., filter design from parameters).
- **Invocation:** Called internally by `OmniDSP::create_processor` when `Params::[Type]` are provided for a design-based operation. Signature example: `OmniDSP::OmniExpected<Design::FIRFilter> OmniDSP::Design::create(const Params::FIRFilter& params_obj);`.
- **Location:** Declarations in `include/OmniDSP/design/[type].hpp`, implementations in `src/omnidsp/design/`.

### **5. Execution Objects: `Plan::[Type]` (Stateless) and `Processor::[Type]` (Stateful)**

- **`Plan::[Type]` (Stateless Execution Object - e.g., `Plan::FFT`, `Plan::Convolution`, `Plan::Window`)**

  - **Purpose:** Encapsulates the **immutable context** and pre-computed data for efficient, stateless execution.
    - `Plan::FFT`, `Plan::Convolution`, `Plan::Correlation` are created by `OmniDSP::create_plan` taking `Params` or `Coefs`.
    - `Plan::Window` is created by `OmniDSP::create_window_plan` taking a `Window::[SpecType]` and length.
  - **Characteristics:** Stateless, immutable resources, factory-created, backend-dependent PIMPL (e.g., `OmniDSP::Default::Plan::FFTImpl` implementing `OmniDSP::Abstract::FFTPlanImpl`), optimized `execute` method, no state-related `reset()` method.
  - **Thread Safety:** Instances are inherently thread-safe for concurrent `execute` calls if the backend implementation of `execute` is reentrant. Can be safely shared among threads.

- **`Processor::[Type]` (Stateful Execution Object - e.g., `Processor::FIRFilter`, `Processor::Resample`)**
  - **Purpose:** Encapsulates the **immutable context**, pre-computed data, and the **mutable internal state** for efficient, stateful execution. Created by `OmniDSP::create_processor`.
  - **Characteristics:** Stateful (manages internal state modified by `execute`), immutable resources, factory-created, backend-dependent PIMPL (e.g., `OmniDSP::Default::Processor::FIRFilterImpl` implementing `OmniDSP::Abstract::FIRFilterProcessorImpl`), optimized `execute` method, provides `reset()` method to re-initialize internal state.
  - **Thread Safety:** Instances are **NOT thread-safe** for concurrent `execute()` calls on the _same instance_. Each thread processing an independent stream requires its own `Processor::[Type]` instance.

## **Key Design Patterns and Techniques**

- **Builder/Fluent Interface:** `Params::[Type]` objects for user configuration.
- **Lightweight Specification Structs:** `Window::[SpecType]` for window configuration.
- **Pimpl Idiom:** In `OmniDSP`, `Plan::[Type]`, and `Processor::[Type]` classes.
- **Backend Interface Contract (`Abstract::Backend`, `Abstract::[Type]PlanImpl`, `Abstract::[Type]ProcessorImpl`):** Defines clear contracts.
  - `Abstract::Backend` will have distinct `create_*_plan_impl`, `create_*_processor_impl` methods.
  - It will also have overloaded `create_window_plan_impl` methods for each `Window::[SpecType]`.
  - The concrete backend implementation classes (e.g., `OmniDSP::Default::Plan::FFTImpl`, `OmniDSP::IntelIPP::Processor::FIRFilterImpl`) implement these abstract interfaces and encapsulate backend-specific logic.
- **Factory Pattern:** `OmniDSP::create()` (for `OmniDSP`), `OmniDSP::create_plan()` (for `Plan::[Type]`), `OmniDSP::create_window_plan()` (for `Plan::Window`), `OmniDSP::create_processor()` (for `Processor::[Type]`).
- **Strategy Pattern (via Backends):** `Dispatcher::Backend` as a composite strategy.
- **Implementation Inheritance (from `Default::Backend`):** For optimized backends. Optimized backends override specific `create_*_plan_impl` / `create_*_processor_impl` / `create_window_plan_impl` methods for operations/types they optimize; others fall back to `Default::Backend`'s implementation.
- **RAII, `OmniExpected` / `Status` Enum.**

## **The `Dispatcher::Backend`**

- **Role:** A specialized implementation of `Abstract::Backend` that delegates calls to other concrete `Abstract::Backend` instances based on a runtime configuration.
- **Construction:** Instantiated by `OmniDSP::create` when backend overrides are provided.
- **Operation:** When a factory method like `create_fft_plan_impl`, `create_fir_filter_processor_impl`, or `create_window_plan_impl(const Window::Kaiser&, ...)` is called on the `Dispatcher::Backend`:
  1.  It identifies the `OperationCategory` (e.g., `FFT`, `FIRFilter`, `Windowing`).
  2.  Checks its configuration for an override for this category.
  3.  If an override exists, it forwards the call to the corresponding specialized `Abstract::Backend` instance.
  4.  If no override exists, it forwards the call to its primary/default `Abstract::Backend` instance.
- **Inheritance:** Inherits directly from `Abstract::Backend`.

## **Performance Considerations**

- **Backend Strategy:** Primary mechanism, enhanced by `Dispatcher::Backend`.
- **Dispatcher Overhead:** Small, generally negligible.
- **Cost of Object Creation:** Setup cost for **`Processor::[Type]` instances** is incurred per thread/stream in parallel scenarios.
- **Plan/Processor Object Optimization:** Amortizes setup costs.
- **User-Managed Memory (`std::span`), Memory Alignment:** Crucial for performance.

## **Build System Philosophy (CMake)**

The `Dispatcher::Backend` integrates with the CMake-based build system as a core architectural element:

- **Core Library Component:** The source files for `Dispatcher::Backend` are integral to the OmniDSP library and compiled as part of the main `omnidsp` target.
- **No Dedicated CMake Option for Dispatcher:** It's a fundamental mechanism, not an optional backend.
- **Relies on Existing Backend Enablement:** Its utility depends on which other backends (OneMKL, Accelerate, etc.) are enabled via their respective CMake options (`OMNIDSP_ENABLE_ONEMKL`, etc.).
- **Conditional Compilation of Dispatch Logic:** `OmniDSP::create` uses preprocessor checks (driven by CMake-set definitions like `OMNIDSP_ENABLED_ONEMKL`) to only configure dispatch targets for available backends.
- **`OperationCategory` Enum:** Part of core library headers.
- **Dependency Management:** Introduces no new external dependencies itself.

The build system ensures all potential concrete backend targets are available if their dependencies are met, and compiles the dispatcher as a fundamental part of the library.

## **Other Design Considerations**

- **Concurrency and Thread Safety (Revisited):**
  - **`Plan::[Type]` objects (Stateless):** Shareable if `execute` is reentrant.
  - **`Processor::[Type]` objects (Stateful):** **NOT thread-safe for concurrent `execute` on the same instance.** Requires separate instances per thread/stream.
  - `OmniDSP` instance requires external synchronization if its methods (like factory calls) are invoked concurrently.
- **Testing:** Strategy needs to cover `create_plan`, `create_window_plan`, `create_processor`, and the behavior of `Plan::[Type]` and `Processor::[Type]` objects (including `reset`). Dispatcher combinations and backend fallbacks (especially for windowing) also need testing.
- **Data Type Extensibility:** Primarily targets `float` and `double` via templates. New types ideally added to `Default::Backend` first.
- **Backend Discovery:** Static functions `OmniDSP::get_available_backends() -> std::vector<BackendType>` and `OmniDSP::get_preferred_backend() -> BackendType` are provided.

## **Rationale Summary**

This design, using distinct **`create_plan`** (including `create_window_plan`) and **`create_processor`** methods to produce stateless **`Plan::[Type]`** and stateful **`Processor::[Type]`** objects respectively, and introducing dedicated `Window::[SpecType]` structs for window configuration, aims for a DSP library that is:

1.  **Highly Configurable & Flexible.**
2.  **Consistent & Predictable:** Unified configuration via `Params` or `Window::[SpecType]`; clear factory methods.
3.  **Performant.**
4.  **Portable.**
5.  **Easy to Use:** Fluent interface for parameters; explicit API for creating stateless vs. stateful objects; clear window specification.
6.  **Maintainable & Extensible:** Clear internal roles and backend contracts, including overloaded backend factories for windowing.
7.  **Robust.**
8.  **Clear (Re: Concurrency):** Explicit guidance on thread safety based on object type (`Plan` vs. `Processor`).
