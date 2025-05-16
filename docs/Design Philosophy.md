# **OmniDSP Design Philosophy**

The OmniDSP library is designed around core concepts and patterns to achieve portability, performance, consistency, and ease of use. This document reflects the latest architectural decisions, including the "Window as Plan" and "Transform as Plan" models, advanced backend configuration, a fluent parameter model for some operations, dedicated specification structs for others, and a clear distinction between stateless `Plan` objects and stateful `Processor` objects, created via distinct factory methods.

## **Core Components: The `OmniDSP` Class, Configuration (`Params`, `Window::[SpecType]`, `Transform::[SpecType]`, `Coefs`), Intermediate Representations (`Design`), `Plan` Objects, and `Processor` Objects**

The primary user workflow involves:

1.  Creating an `OmniDSP` instance, which manages the selected backend.
2.  Configuring an operation:
    - Using a `Params::[Type]` object for operations like Convolution, Correlation, FIR/IIR Filters (design path), CQT (design path), Resampling (design path).
    - Using a `Window::[SpecType]` object (e.g., `Window::Kaiser`, `Window::Hann`) along with a length for windowing operations.
    - Using a `Transform::[SpecType]` object (e.g., `Transform::FFT`, `Transform::RFFT`) for transform operations.
    - Optionally, providing pre-calculated `Coefs::[Type]` (e.g., for filters) to bypass design stages.
3.  Calling the appropriate factory method on the `OmniDSP` instance:
    - `OmniDSP::create_plan(...)`:
      - For stateless operations like Convolution and Correlation, accepting `const Params::[Type]&` or `const Coefs::[Type]&`, resulting in `Plan::Convolution` or `Plan::Correlation`.
      - For stateless transform operations (FFT, RFFT, DFT, etc.), accepting `const Transform::[SpecType]&`, resulting in `Plan::Transform<T_In, T_Out>`.
      - For stateless windowing operations, accepting `const Window::[SpecType]&` and `length`, resulting in `Plan::Window<T_Data>`.
    - `OmniDSP::create_processor(...)` for stateful operations (e.g., Filtering, CQT, Resampling), accepting `const Params::[Type]&` or `const Coefs::[Type]&`, resulting in a `Processor::[Type]`.
4.  Internally, if `Params` are provided for design-based operations (like filters, CQT, Resampling), they are converted into an intermediate, validated `Design::[Type]` object via `OmniDSP::Design::create`.
5.  The `OmniDSP` instance then uses the configuration (`Params`, `Window::[SpecType]`, `Transform::[SpecType]`, `Coefs`, or `Design`) to create an optimized execution object (`Plan::[Type]` or `Processor::[Type]`) via the selected backend.
    - **`Processor::[Type]` objects** initialize and manage their own internal state upon creation.
6.  The user calls the appropriate execution method on the returned `Plan` or `Processor` object:
    - For `Plan::Transform`: `forward(...)` or `inverse(...)`.
    - For `Plan::Window`: `execute(...)` or `execute_inplace(...)`.
    - For `Plan::Convolution`/`Correlation`: `execute(...)`.
    - For `Processor::[Type]` objects: `execute(...)`, which modifies the object's internal state.
7.  The user can call `reset()` on a **`Processor::[Type]` object** to reset its internal state, allowing it to be reused for a new, independent data stream.

### **1. The `OmniDSP` Class: Backend Encapsulation and Factory**

- **Purpose:** The primary user-facing entry point, encapsulating the selected backend implementation(s).
- **Backend Management & Instantiation (`OmniDSP::create`)**:
  - Users create an instance using static `OmniDSP::create` factory methods, which can select a single backend or configure a `Dispatcher::Backend` for per-operation-category overrides. The `pimpl_` of `OmniDSP` points to the active `Abstract::Backend` implementation.
- **Factory Role (`create_plan` / `create_processor`)**: The `OmniDSP` instance acts as the central factory for creating execution objects.
  - **`create_plan(...)`**:
    - Overloaded methods accepting `const Params::[Type]&` or `const Coefs::[Type]&` for operations like Convolution and Correlation, resulting in a stateless `Plan::[Type]`. Returns `OmniExpected<Plan::[Type]>`.
    - Overloaded methods accepting `const Window::[SpecType]& spec, size_t length` for creating stateless `Plan::Window<T_Data>`. Returns `OmniExpected<Plan::Window<T_Data>>`.
    - Overloaded methods accepting `const Transform::[SpecType]& spec` for creating stateless `Plan::Transform<T_In, T_Out>`. Returns `OmniExpected<Plan::Transform<T_In, T_Out>>`.
  - **`create_processor(...)`**: Overloaded methods accepting `const Params::[Type]&` or `const Coefs::[Type]&` for operations resulting in a stateful `Processor::[Type]` (e.g., FIR/IIR Filters, CQT, Resampling). Returns `OmniExpected<Processor::[Type]>`.
    Internally, these methods handle the conversion from `Params` to `Design` (for design-based operations) before dispatching to the appropriate backend implementation creation method.
- **One-Off Operations & Helpers:** Provides methods for simple operations (e.g., `convolve`) and helpers (e.g., `generate_window_coefficients`), routed through its `Abstract::Backend` Pimpl.
  - **`generate_window_coefficients<T_Data>(const Window::[SpecType]& spec, size_t length)`**: Convenience wrappers that internally create a temporary `Plan::Window` to generate and return coefficients.
  - **`generate_transform_coefficients<T_Out>(const Transform::[SpecType]& spec, ...)`**: Potentially, for transforms that can be represented by coefficients (e.g., DFT matrix), though less common for FFTs.

### **`OperationCategory` Enum (for Dispatcher)**

```cpp
// Example: Potentially in types/operation.hpp or core_types.hpp
enum class OperationCategory {
    FFT, RFFT, // Specific transform categories for dispatch if needed
    DFT, // Example for other transforms
    Convolution, Correlation, FIRFilter, IIRFilter, Resampling, CQT, Windowing,
    Transform, // Generic category for transforms if not dispatching per specific type
    GenericFallback // etc.
};
```

### 2. User Configuration & Internal Transformation

- `Params::[Type]` **(`Struct`/`Class` - Fluent Builder for Operation Parameters)**:
  - **Purpose:** The primary user interface for defining parameters for specific operations (e.g., `Filter`s, `CQT`, `Resample`, `Convolution`, `Correlation`). Supports a fluent interface.
  - **Location:** `include/OmniDSP/params/[type].hpp`.
  - **Validation:** Comprehensive validation performed during internal conversion to `Design` objects (for design-based operations) or directly by backend plan/processor creation.
  - **User Workflow:** User creates/configures `Params::[Type]`, then calls `dsp.create_plan(params_object)` or `dsp.create_processor(params_object)`.
- **`Window::[SpecType]` (Struct - Lightweight `Window` Specifications)**:
  - **Purpose:** User-facing, lightweight, non-hierarchical structs for defining window types and their specific parameters (e.g., `Window::Hann{}`, `Window::Kaiser{double beta}`).
  - **Location:** `include/OmniDSP/window/specs.hpp` (or individual headers).
  - **Validation:** Constructors of these specification structs validate their parameters.
  - **User Workflow:** User creates a `Window::[SpecType]` object and passes it to `dsp.create_plan(...)` (renamed from create_window_plan) or `dsp.generate_window_coefficients(...)`. Also used within `Params::FIRFilter` etc.
- **Transform::[SpecType] (Struct - Lightweight `Transform` Specifications)**:
  - **Purpose:** User-facing, lightweight, non-hierarchical structs for defining transform types and their specific parameters (e.g., `Transform::FFT{size_t length}`, `Transform::RFFT{size_t real_length}`).
  - **Location:** `include/OmniDSP/transform/specs.hpp` (or individual headers).
  - **Validation:** Constructors of these specification structs validate their parameters.
  - **User Workflow:** User creates a `Transform::[SpecType]` object and passes it to `dsp.create_plan(...)` to obtain a `Plan::Transform<T_In, T_Out>`.
- **`Coefs::[Type]` (Class/Struct - Pre-Calculated Data for `Plan`s/`Processor`s)**:
  - **Purpose:** Encapsulates pre-calculated, validated coefficients/kernels, bypassing `Params`-based design stages.
  - **Location:** Relevant public headers (e.g., `include/OmniDSP/coefs/fir_filter.hpp`).
  - **User Workflow:** User creates/obtains `Coefs::[Type]`, then calls `dsp.create_processor(coeffs_object)`.

### 3. Intermediate Representations (`Design` - **Internal** Library Detail)

Crucial for library architecture and backend contracts for design-based operations, though abstracted from the user at the factory method call site:

- **`Design::[Type]` (Struct - Internal Resolved `Design` Specification)**:
  - **Purpose (Internal):** Represents the fully resolved, detailed, validated specification for complex operations like filters, CQT, or resampling.
  - **Creation (Internal):** By `OmniDSP::Design::create(const Params::[Type]&)`. This is the contract for backend methods like `Abstract::Backend::create_fir_filter_processor_impl(const Design::FIRFilter& design)`.

### 4. Utility Functions (`OmniDSP::Design::create`)

- **Purpose:** Encapsulates complex design algorithms (e.g., filter design from parameters).
- **Invocation:** Called internally by `OmniDSP::create_processor` when `Params::[Type]` are provided for a design-based operation. Signature example: `OmniDSP::OmniExpected<Design::FIRFilter> OmniDSP::Design::create(const Params::FIRFilter& params_obj);`.
- **Location:** Declarations in `include/OmniDSP/design.hpp`, implementations in `src/omnidsp/design/`.

### 5. Execution Objects: `Plan::[Type]` (Stateless) and `Processor::[Type]` (Stateful)

- **`Plan::[Type]` (Stateless Execution Object - e.g., `Plan::Transform`, `Plan::Convolution`, `Plan::Window`)**
  - **Purpose:** Encapsulates the **immutable context** and pre-computed data for efficient, stateless execution.
    - `Plan::Transform<T_In, T_Out>` is created by `OmniDSP::create_plan` taking `Transform::[SpecType]`. Has `forward(...)` and `inverse(...)` methods.
    - `Plan::Convolution`, `Plan::Correlation` are created by `OmniDSP::create_plan` taking `Params` or `Coefs`. Has `apply(...)` method.
    - `Plan::Window<T_Data>` is created by `OmniDSP::create_plan` taking a `Window::[SpecType]` and `length`. Has `apply(...)` method.
  - **Characteristics:** Stateless, immutable resources, factory-created, backend-dependent PIMPL (e.g., `OmniDSP::Default::Plan::Transform::FFTC2CImpl<C32>` implementing `OmniDSP::Abstract::Plan::TransformImpl<C32, C32>`), optimized execution methods, no state-related `reset()` method.
  - **Thread Safety:** Instances are inherently thread-safe for concurrent execution method calls if the backend implementation is reentrant. Can be safely shared among threads.
- **Processor::[Type] (Stateful Execution Object - e.g., Processor::FIRFilter, Processor::Resample)**
  - **Purpose:** Encapsulates the **immutable context**, pre-computed data, and the **mutable internal state** for efficient, stateful execution. Created by OmniDSP::create_processor.
  - **Characteristics:** Stateful (manages internal state modified by execute), immutable resources, factory-created, backend-dependent PIMPL (e.g., `OmniDSP::Default::Processor::FIRFilterImpl` implementing `OmniDSP::Abstract::Processor::FIRFilterImpl`), optimized execute method, provides `reset()` method to re-initialize internal state.
  - **Thread Safety:** Instances are **NOT thread-safe** for concurrent `execute()` calls on the _same instance_. Each thread processing an independent stream requires its own `Processor::[Type]` instance.

## Key Design Patterns and Techniques

- **Builder/Fluent Interface:** Used for `Params::[Type]` objects to provide a user-friendly way to set multiple operation parameters.
- **Lightweight Specification Structs:** `Window::[SpecType]` and `Transform::[SpecType]` are used for configuring windowing and transform operations, respectively. These structs are simple, focus on parameters, and their constructors handle validation.
- **Pimpl Idiom (Pointer to Implementation):** Extensively used in OmniDSP, `Plan::[Type]`, and `Processor::[Type]` classes. This decouples the public interface from the internal implementation details, improving ABI stability and reducing compilation dependencies.
- **Backend Interface Contract:**
  - `Abstract::Backend`: Defines the core interface that all backend implementations must adhere to. It includes pure virtual factory methods for creating plan and processor implementations.
  - `Abstract::Plan::[Type]Impl` (e.g., `Abstract::Plan::TransformImpl<T_In, T_Out>`, `Abstract::Plan::WindowImpl<T_Data>`): Abstract interfaces for the PIMPLs of stateless `Plan` objects.
  - `Abstract::Processor::[Type]Impl` (e.g., `Abstract::Processor::FIRFilterImpl<T_Data>`): Abstract interfaces for the PIMPLs of stateful `Processor` objects.
  - Concrete backend implementation classes (e.g., `OmniDSP::Default::Plan::Transform::FFTC2CImpl`, `OmniDSP::IntelIPP::Processor::FIRFilterImpl`) implement these abstract interfaces and encapsulate backend-specific logic and resource management.
- **Factory Pattern:**
  - OmniDSP::create(...): Static factory method for creating OmniDSP instances, allowing selection of primary and override backends.
  - OmniDSP::create_plan(...): Overloaded factory methods on the OmniDSP instance for creating various stateless Plan objects (Plan::Transform, Plan::Window, Plan::Convolution, etc.).
  - OmniDSP::create_processor(...): Overloaded factory methods on the OmniDSP instance for creating stateful Processor objects.
- **Strategy Pattern (via Backends):** The OmniDSP class uses an Abstract::Backend pointer (pimpl\_) to delegate operations. The Dispatcher::Backend acts as a composite strategy, allowing different operations to be routed to different concrete backend strategies at runtime.
- **Implementation Inheritance (from Default::Backend):** Optimized backends (like Accelerate::Backend, OneMKL::Backend, IntelIPP::Backend) inherit from OmniDSP::Default::Backend. This allows them to override only the specific create\_\*\_impl factory methods for operations they optimize. For operations not overridden, the call falls through to the Default::Backend's implementation, promoting code reuse and simplifying the development of new optimized backends.
- **RAII (Resource Acquisition Is Initialization):** Used for managing resources within Plan and Processor implementations, ensuring that resources (like backend-specific handles or allocated memory) are properly acquired in constructors and released in destructors. std::unique_ptr is commonly used for this.
- **OmniExpected / Status Enum:** Used for error handling. Functions that can fail return an OmniExpected<Value, Status> (or OmniExpected<void, Status>), which either contains the result or an error code from the OmniStatus enum. This promotes explicit error checking by the caller.

## The Dispatcher::Backend

- **Role:** A specialized implementation of Abstract::Backend that delegates calls to other concrete Abstract::Backend instances based on a runtime configuration. This allows for fine-grained control over which backend handles which category of DSP operations.
- **Construction:** Instantiated by OmniDSP::create when backend overrides are provided by the user (typically via OmniDSP::Builder). It holds a pointer to a primary (default) Abstract::Backend instance and a map of OperationCategory to overriding Abstract::Backend instances.
- **Operation:** When a factory method like create_transform_plan_impl(const Transform::FFT&, ...) (for FFTs), create_fir_filter_processor_impl(...), or create_window_plan_impl(const Window::Kaiser&, ...) is called on the Dispatcher::Backend:
  1. It identifies the OperationCategory relevant to the called method (e.g., FFT, FIRFilter, Windowing, Transform).
  2. It checks its internal configuration map for an override backend registered for this specific OperationCategory.
  3. If an override exists and the backend pointer is valid, it forwards the call (along with all its parameters) to the corresponding factory method of the registered override Abstract::Backend instance.
  4. If no override exists for that category, or if a generic fallback override is configured and applicable, it may use that. Otherwise, it forwards the call to its primary Abstract::Backend instance.
- **Inheritance:** Dispatcher::Backend inherits directly from Abstract::Backend and overrides all its pure virtual factory methods to implement this dispatching logic.

## Performance Considerations

- **Backend Strategy:** The choice of backend (Default, Accelerate, OneMKL, IntelIPP) is the primary mechanism for performance optimization. The Dispatcher::Backend allows mixing these for optimal performance across different operation types.
- **Dispatcher Overhead:** The overhead of the Dispatcher::Backend itself is minimal, typically involving a map lookup and a virtual function call, which is generally negligible compared to the DSP computation time.
- **Cost of Object Creation:**
  - Plan objects (including Plan::Transform and Plan::Window) are designed for stateless operations. Their creation cost (which might involve pre-computation like twiddle factors for FFTs or window coefficients) is amortized over many executions.
  - Processor objects are stateful. Their creation involves setup for their internal state. In scenarios with many parallel independent streams (e.g., per-thread processing), the cost of creating multiple Processor instances should be considered.
- **Plan/Processor Object Optimization:** The core idea of Plan and Processor objects is to perform setup and pre-computation once, allowing subsequent execution calls to be highly efficient.
- **User-Managed Memory (std::span):** The API uses std::span for passing data buffers. This avoids unnecessary data copies and gives the user control over memory allocation and management.
- **Memory Alignment:** While not explicitly enforced by the public API for user-provided spans, backend implementations (especially optimized ones like Intel IPP or oneMKL) may perform better with or require aligned memory. Users should be aware of this for performance-critical applications. Internal buffers within backend implementations should be aligned appropriately.

## Build System Philosophy (CMake)

The OmniDSP library uses CMake as its build system generator. The philosophy is to create a modular and configurable build process.

- **Modularity:** The core library, different backends, Python bindings, tests, and examples are organized into CMake subdirectories or targets.
- **Backend Selection:** CMake options (e.g., OPTION_OMNIDSP_ENABLE_ACCELERATE, OPTION_OMNIDSP_ENABLE_ONEMKL) control which backends are compiled and linked. This allows users to build a library tailored to their needs and available system dependencies.
- **Dependency Management:** CMake's find_package is used to locate external dependencies like Boost, and potentially MKL, IPP, etc. The conda-lock generated environments provide these dependencies.
- **omnidsp_config.hpp Generation:** A configuration header (omnidsp_config.hpp) is generated by CMake. This header contains preprocessor definitions (e.g., OMNIDSP_BACKEND_ACCELERATE_ENABLED) that allow C++ code to conditionally compile features or backend-specific paths based on the build configuration.
- **omnidsp_export.hpp Generation:** CMake's GenerateExportHeader module is used to create omnidsp_export.hpp. This header defines macros (e.g., OMNIDSP_EXPORT) for managing symbol visibility (import/export) when building shared libraries, which is crucial for ABI stability and cross-platform compatibility.
- **Python Bindings:** If enabled, CMake integrates with pybind11 (typically via pybind11_add_module) to build the Python extension module.
- **Testing Integration:** CTest is used for running C++ tests, and pytest for Python tests, both integrated into the CMake workflow.
- **Installation:** CMake install rules are defined for installing the library, headers, and Python module.
- **Dispatcher::Backend Integration:** The Dispatcher::Backend is a core library component, not an optional backend selected by a CMake option. Its functionality relies on which _other_ concrete backends are enabled via their respective CMake options. The OmniDSP::create factory uses preprocessor checks (driven by CMake-set definitions like OMNIDSP_HAS_ONEMKL_BACKEND) to determine which backends are available to be dispatched to.

## Other Design Considerations

- **Concurrency and Thread Safety (Revisited):**
  - **Plan::[Type] objects (Stateless):** Designed to be stateless. Their execution methods (forward, inverse, execute, execute_inplace) should be reentrant. If the underlying backend implementation of these methods is also reentrant, Plan objects can be safely shared among threads and their execution methods called concurrently on different data.
  - **Processor::[Type] objects (Stateful):** These objects manage internal state that is modified by their execute(...) method. Therefore, a single Processor instance is **NOT thread-safe** for concurrent calls to execute() on that _same instance_. If multiple threads need to process independent data streams using the same filter or CQT parameters, each thread should have its own Processor instance. The reset() method allows reusing a processor for a new stream on the same thread.
  - **OmniDSP instance:** The OmniDSP instance itself, particularly its factory methods (create_plan, create_processor), may not be inherently thread-safe if multiple threads attempt to create plans/processors simultaneously using the same OmniDSP instance, especially if backend initialization or resource allocation within the factory methods is not synchronized. External synchronization (e.g., a mutex) would be required if an OmniDSP instance is shared and its factory methods are called from multiple threads. One-off operations on the OmniDSP instance also fall under this consideration.
- **Testing:** A comprehensive testing strategy is crucial. This includes:
  - Unit tests for individual components (parameter validation, design algorithms, plan/processor execution logic for each backend).
  - Integration tests for the OmniDSP facade, ensuring correct plan/processor creation and delegation to backends.
  - Tests for the Dispatcher::Backend with various override configurations and fallbacks.
  - Specific tests for windowing (Plan::Window, generate_window_coefficients) and transforms (Plan::Transform, Transform::[SpecType]).
  - Python binding tests to ensure the Python API works as expected.
- **Data Type Extensibility:** The library primarily targets float (F32) and double (F64) for real data, and std::complex<float> (C32) and std::complex<double> (C64) for complex data. Adding support for new data types (e.g., fixed-point, or different floating-point precisions) would require:
  - Adding corresponding type aliases in core_types.hpp.
  - Templating relevant classes and functions for the new types.
  - Ensuring backend implementations (starting with Default::Backend) support the new type.
  - Updating type traits and helper utilities.
- **Backend Discovery and Selection:**
  - OmniDSP::get_available_backends() -> std::vector<BackendType>: A static utility function to query which backends are compiled into the library and available at runtime.
  - OmniDSP::get_preferred_backend() -> BackendType: A static utility function that could suggest a default "best" backend based on platform and availability (e.g., Accelerate on macOS, oneMKL/IntelIPP if available on other platforms, otherwise Default).
  - The OmniDSP::create() factory methods and the OmniDSP::Builder provide the primary mechanisms for users to select and configure the desired backend(s).

## Rationale Summary

This design, using distinct **create_plan** (for stateless Plan::Transform, Plan::Window, Plan::Convolution, etc.) and **create_processor** (for stateful Processor::[Type]) methods, and introducing dedicated Window::[SpecType] and Transform::[SpecType] structs for configuration, aims for a DSP library that is:

1. **Highly Configurable & Flexible:** Users can choose specific backends or let the dispatcher manage them. Parameter objects and specification structs offer detailed control.
2. **Consistent & Predictable:** A unified approach to creating execution objects (Plan vs. Processor). Clear distinction between stateless and stateful operations. Consistent use of OmniExpected for error handling. The Transform and Window APIs now follow a similar pattern using dedicated specification structs.
3. **Performant:** Enables leveraging optimized backend libraries (Accelerate, oneMKL, IntelIPP) where available, while providing a portable default. The Plan/Processor pattern amortizes setup costs.
4. **Portable:** The Default::Backend ensures functionality across all platforms. Abstract interfaces allow for new backends to be added.
5. **Easy to Use:** The OmniDSP facade simplifies interaction. Fluent interfaces for Params objects and clear SpecType structs for windows/transforms improve usability. Explicit forward/inverse methods for Plan::Transform clarify intent.
6. **Maintainable & Extensible:** Clear separation of concerns (interface, implementation, parameters, design). The backend abstraction and inheritance from Default::Backend simplify adding new optimized backends. The spec-based factory pattern for windows and transforms makes adding new types straightforward.
7. **Robust:** Parameter validation in specification structs and Params objects ensures that configurations are sound before plan/processor creation. Comprehensive error reporting via OmniExpected and OmniStatus allows callers to gracefully handle issues. The PIMPL idiom and clear interface contracts contribute to a stable API.
