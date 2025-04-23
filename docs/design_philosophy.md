# OmniDSP Design Philosophy

The OmniDSP library is designed around several core concepts and software design patterns to achieve portability, performance, and ease of use. Understanding these principles is key to using and extending the library effectively.

## Core Components: The `OmniDSP` Class, Types, Specs, Plans, and State

### The `OmniDSP` Class: Backend Encapsulation and Entry Point

- **Purpose:** The `OmniDSP` class serves as the primary user-facing entry point for interacting with the library's functionalities. Its most crucial role is to **encapsulate the selected backend implementation**.
- **Backend Management:** DSP algorithms often have multiple possible implementations leveraging different hardware acceleration libraries (e.g., Apple Accelerate, Intel oneMKL), parallel execution strategies (e.g., OpenMP, TBB, CUDA), specialized hardware (e.g., FPGAs), distributed systems (e.g., Clusters), or a portable default implementation (using C++ and Highway SIMD). The `OmniDSP` class manages which of these backends is active for all subsequent operations performed through that instance.
- **Instantiation (`OmniDSP::create`)**: Users **must** create an instance of `OmniDSP` using the static factory method `OmniDSP::create(Backend backend_type)`. This step is essential because it:
  - Selects the desired backend implementation (e.g., `Backend::Accelerate`, `Backend::OneMKL`, `Backend::Default`, or potential future backends like `Backend::CUDA`, `Backend::FPGA`, `Backend::Cluster`). The library compiles support for all available backends, and `create` chooses the appropriate one at runtime based on the request and availability checks.
  - Initializes any necessary resources or context required by that specific backend (e.g., connecting to cluster nodes, loading an FPGA bitstream).
  - Hides the backend-specific details (including its internal parallelism model, communication protocols, or hardware interactions) from the user using the **Pimpl (Pointer to Implementation) idiom** (see below).
- **Factory Role for Plans & State:** The `OmniDSP` instance acts as a factory for creating all `Plan` objects (like `FFTPlan`, `ConvolutionPlan`, etc.) and potentially associated `State` objects (e.g., `dsp.create_filter_state(plan)`) primarily for use with the synchronous API.
- **One-Off Operations:** The `OmniDSP` class also provides methods for performing "one-off" DSP operations (e.g., window generation like `dsp.hann_window(len)`, or future stateless operations like `dsp.convolve(input, kernel)`). These methods directly utilize the encapsulated backend for computation.

### Type Objects (Enums)

- **Purpose:** To represent simple, distinct categories, types, or modes for a specific operation or component. They act as identifiers or flags, selecting a particular variant of behavior.
- **Examples:** `WindowType`, `FilterType`, `ConvolutionType`, `Backend`, `Status`.
- **Characteristics:** Simple identifiers (`enum class`), stateless, used directly or within `Spec` objects.
- **Header Placement:** Usually defined within the header file most relevant to their function (e.g., `WindowType` in `window.h`). Core types used across many modules (like `Status`, `Backend`) reside in `core_types.h`.
- **Usage:** Passed directly to methods (like `OmniDSP::create(Backend::Default)`) or stored within `Spec` objects.
- **Thread Safety:** As stateless value types, `Type` enums are inherently **thread-safe** to copy and use across threads.

### Specification (`Spec`) Objects

- **Purpose:** To hold configuration data and _multiple_ parameters required to define a specific DSP operation or component design. They represent a complete and validated set of parameters for a design task.
- **Examples:** `WindowSpec`, `FIRFilterSpec`, `IIRFilterSpec`, `ResampleSpec`.
- **Characteristics:** Immutable classes, data aggregators, stateless post-construction, user-created & validated, backend-independent creation.
- **Usage:** Passed as arguments to `OmniDSP` methods for design/generation or `Plan` factory methods.
- **Thread Safety:** As immutable value objects, `Spec` instances are inherently **thread-safe** to copy and share across threads.

### Plan Objects

- **Purpose:** To encapsulate the **immutable context** and pre-computed data (e.g., FFT twiddles, filter coefficients, backend handles/descriptors, network connections, hardware configurations) required for efficient execution of a specific DSP operation based on a configuration defined by a `Spec` or parameters.
- **Examples:** `FFTPlan`, `CQTPlan`, `ConvolutionPlan`, `FIRFilterPlan`, `IIRFilterPlan`, `ResamplePlan`.
- **Characteristics:**
  - **Immutable (Post-Creation):** Once created via an `OmniDSP` factory, the Plan object itself and its core resources are typically immutable.
  - **Stateless (Ideally):** Plans themselves should ideally not hold mutable state related to signal processing between `execute` calls. State required for continuous processing (like filter delays) should be managed separately (see State Objects).
  - **Factory-Created:** Always created via factory methods on an `OmniDSP` instance (e.g., `dsp.create_fft_plan(len)`). This links the Plan to the specific backend and configuration.
  - **Backend-Dependent:** The internal implementation (`*PlanImpl`) depends on the backend selected when the parent `OmniDSP` object was created.
  - **Optimized for Execution:** Designed for low-overhead execution via their `execute` method after the initial setup cost during creation.
- **Usage:**
  - Created once via an `OmniDSP` factory method.
  - Used repeatedly via the synchronous `execute` method, requiring a separate `State` object for stateful operations.
  - Potentially used to create `Submission` objects for asynchronous operations.
- **Thread Safety:** See "Concurrency and Thread Safety" section. **Assume NOT thread-safe** for concurrent `execute` calls on the same instance without external locking.

### State Objects

- **Purpose:** To hold the **mutable state** required for continuous processing by certain `Plan` objects, specifically for the **synchronous `execute` API**. This decouples the changing state from the immutable Plan resources for blocking calls.
- **Examples:** `FilterState`, `ResampleState`. (Note: These are conceptual examples; the exact API needs implementation).
- **Characteristics:**
  - **Mutable:** Designed to be modified by the synchronous `Plan::execute` method.
  - **Plan-Associated:** Each State object is typically associated with a specific Plan instance (or Plan configuration).
  - **Stream-Specific:** A separate State object should be used for each independent stream of data being processed synchronously by the same Plan.
- **Usage:**
  - Created for a specific Plan configuration, likely via `OmniDSP`.
  - Passed as a mutable argument to the synchronous `Plan::execute` method (e.g., `plan->execute(input, output, state)`).
  - Can be reset via a dedicated method (e.g., `state.reset()`).
- **Thread Safety:** See "Concurrency and Thread Safety" section. **Assume NOT thread-safe**. Primarily intended for single-threaded synchronous use or per-thread use.

## Key Design Patterns and Techniques

Beyond the core components, OmniDSP employs several standard design patterns:

### Pimpl Idiom (Pointer to Implementation)

- **Application:** Used extensively in the public API classes (`OmniDSP`, `Plan` objects, potentially future `Submission` objects).
- **Benefits:**
  - **Decoupling:** Reduces compile-time dependencies between user code and backend implementations, speeding up builds.
  - **ABI Stability:** Helps maintain a stable Application Binary Interface for the library, as changes to the implementation details don't affect the public header or require users to recompile against minor internal changes.
  - **Encapsulation:** Cleanly separates the public interface from private implementation details.
  - **Backend Abstraction:** Enables the runtime selection and encapsulation of different backend implementations.

### Backend Interface Contract (`backend.h`)

- **Application:** Abstract base classes (`OmniDSPImpl`, `*PlanImpl`, potentially future `*SubmissionImpl`) define the contract for all backends.
- **Benefits:** Interchangeability, Consistency, Clear Extension Point. This contract is the key enabler for supporting diverse execution targets.

### Factory Pattern

- **Application:** `OmniDSP::create()` (Static Factory) and `OmniDSP` instance methods like `create_fft_plan()`, `create_filter_state()`, potentially `create_submission()` (Instance Factories).
- **Benefits:** Decouples client code from object creation, centralizes creation logic, ensures Plans/State/Submissions use the correct backend context.

### Strategy Pattern (via Backends)

- **Application:** Different backend implementations (`DefaultOmniDSPImpl`, etc.) act as interchangeable strategies selected by `OmniDSP`. The choice of backend determines the underlying execution strategy.
- **Benefits:** Allows algorithms and their execution characteristics (serial CPU, SIMD CPU, parallel CPU, GPU, FPGA, Cluster) to vary independently from clients using the standard API. Facilitates adding new backends with different performance profiles or targeting different hardware.
- **Example Flexibility:** This pattern enables diverse backends like `CudaOmniDSPImpl`, `OpenMPDefaultOmniDSPImpl`, `ClusterOmniDSPImpl`, `FpgaOmniDSPImpl`.

### Template Method Pattern (Default Backend Inheritance)

- **Application:** Optimized backends inherit from the SIMD-optimized `DefaultOmniDSPImpl` and override only necessary methods.
- **Benefits:** Code Reuse, Guaranteed Functionality (optimized baseline), Simplified Development, Selective Fallback.

### Composite Plans & Orchestration

- **Application:** Some Plans (e.g., `CQTPlan`) may orchestrate multiple sub-operations using other Plans.
- **Benefits:** Flexibility for complex algorithms, Encapsulation of orchestration logic.

### RAII (Resource Acquisition Is Initialization)

- **Application:** `std::unique_ptr` manages `pimpl_` lifetimes. Backends manage their own resources via RAII. State objects might also use RAII for internal buffers.
- **Benefits:** Prevents resource leaks, simplifies resource management, enhances exception safety.

### Error Handling (`std::expected` / `Status` Enum)

- **Application:** Methods that can fail return an `OmniExpected<T>`.
- **Benefits:** Clear, modern C++23 error handling. `Status` enum provides specific error codes.

## Performance Considerations

Achieving high performance is a primary goal of OmniDSP. Several design choices contribute to this:

- **Backend Strategy:** The core mechanism for performance is selecting the appropriate backend via `OmniDSP::create`. Users can choose hardware-accelerated backends (Accelerate, oneMKL, potential future CUDA/FPGA) when available for maximum speed on specific platforms.
- **Optimized Default Backend (Highway SIMD):** The `Default` backend is not merely a basic C++ implementation. It utilizes the Google Highway library for portable SIMD (Single Instruction, Multiple Data) acceleration with runtime dynamic dispatch. This ensures good performance even without platform-specific libraries, leveraging CPU vector units (SSE, AVX, NEON) effectively across different architectures.
- **Plan Objects:** The Plan pattern avoids redundant setup costs. Expensive operations like FFT planning, filter coefficient calculation, or backend resource allocation happen once during Plan creation, making subsequent `execute` calls lightweight and efficient for processing data streams or repeated operations.
- **Synchronous Core API:** The primary `Plan::execute` API is synchronous. While potentially limiting for hiding latency from specific backends (see Asynchronous Execution below), this model avoids the overhead associated with asynchronous task creation, scheduling, and synchronization (e.g., `std::future`, thread pools) in the common case, maximizing raw single-threaded throughput for CPU-bound operations.
- **User-Managed Memory (`std::span`):** The API uses `std::span` for input/output buffers, requiring the user to manage memory allocation. This avoids hidden memory copies or allocations within the `execute` methods, giving the user full control and enabling optimizations like buffer reuse or operating directly on memory-mapped regions.
- **Memory Alignment:** While the library aims to function correctly with unaligned data (especially the Highway-accelerated Default backend), performance on many backends (including SIMD) can be significantly improved if input/output buffers passed via `std::span` are aligned to appropriate boundaries (e.g., 32 or 64 bytes). Using standard containers like `std::vector` often provides suitable alignment, but users seeking maximum performance should consider using aligned allocators or memory allocation functions.

## Build System Philosophy (CMake) The build system plays a critical role in managing complexity, dependencies, and portability.

### Dependency Management

- **Standard `find_package`:** Dependencies (like Boost, Highway, MKL, GTest, pybind11) should primarily be located using CMake's standard `find_package` mechanism, leveraging the config-files or find-modules provided by the dependencies themselves (often installed via package managers like Conda).
- **Modularization:** Dependency finding and configuration logic should be encapsulated within dedicated CMake script files located in a `cmake/` directory (e.g., `cmake/boost.cmake`, `cmake/highway.cmake`). These files are responsible for:
  - Calling `find_package`.
  - Checking if the dependency was found successfully.
  - Setting any necessary cache or status variables (e.g., `OMNIDSP_HAS_ONEMKL`, using `PARENT_SCOPE` if needed by the root `CMakeLists.txt`).
  - Appending required include directories, link libraries, compile definitions, etc., to common list variables defined in the root scope (e.g., `OMNIDSP_LINK_LIBS`, `OMNIDSP_COMPILE_DEFINITIONS`).
  - Defining backend-specific source file lists (e.g., `ONEMKL_BACKEND_SOURCES`, set with `PARENT_SCOPE`).
    This approach keeps the root `CMakeLists.txt` cleaner (using `include(my_dependency)`), makes dependency logic reusable, and simplifies adding or modifying dependencies.
- **Conda Integration:** When dependencies are managed via Conda, the build system explicitly uses the `$ENV{CONDA_PREFIX}` environment variable to locate necessary headers and libraries within the active Conda environment, particularly for libraries like MKL/IPP that might not install standard CMake config files. This logic should also reside within the relevant `cmake/*.cmake` module.

### Handling Platform/Backend Specifics

- **Avoid Preprocessor Conditionals (`#ifdef`) in Source Code:** Platform-specific code (e.g., different API calls for Windows vs. Linux, or code specific to Accelerate vs. MKL) should be segregated into different implementation files whenever possible. Preprocessor directives (`#ifdef _WIN32`, `#ifdef USE_ACCELERATE`) within the main C++ logic should be **strongly avoided**.
  - **Rationale:** Extensive `#ifdef` blocks make code harder to read, maintain, and test. Crucially, they interfere with code coverage tools, which typically only analyze one branch of an `#ifdef` during a single build configuration, leading to inaccurate coverage metrics. Code clarity and testability are prioritized.
- **CMake-Managed Conditional Compilation:** The build system (CMake) is responsible for handling platform and backend differences:
  - **Conditional Source Files:** CMake logic (like the generator expressions used for backends: `$<$<BOOL:${VAR}>:file.cpp>`) determines which implementation files are compiled for a given target based on the detected platform and available backends. This is the preferred way to manage platform/backend-specific code.
  - **Conditional Compile Definitions:** CMake sets preprocessor definitions (e.g., `USE_ACCELERATE`, `USE_ONEMKL`) via `target_compile_definitions`. These definitions should ideally only be used for _minimal_ conditional inclusion of backend-specific _headers_ or very small, unavoidable code snippets where separate files are impractical, not for large blocks of differing logic.
  - **Conditional Linking:** CMake links the appropriate libraries (`Accelerate.framework`, MKL/IPP libraries) based on the selected or available backends, typically configured within the `cmake/*.cmake` modules.

## Other Design Considerations

### Data Type Extensibility

- **Core Support:** The library primarily targets `float` and `double` precision floating-point types via templates.
- **Backend Variations:** Specific backends might offer optimized support for additional types (e.g., `float16` in IPP).
- **Adding New Types:** The `Default` backend serves as the primary place to add support for other data types (e.g., fixed-point, integers). Users wishing to add support for a new type should ideally contribute an implementation to the `Default` backend first. Optimized backends can then optionally override this if they offer specialized support.

### Backend Discovery and Selection

- **Availability:** A static function `OmniDSP::get_available_backends() -> std::vector<Backend>` should be provided to allow users to query at runtime which backends were successfully compiled and detected as available on the current system.
- **Preference:** A static function `OmniDSP::get_preferred_backend() -> Backend` should indicate the default backend that `OmniDSP::create()` would select if none is explicitly requested. (The preference logic, e.g., MKL > Accelerate > Default, needs to be defined).
- **Runtime Choice:** Users select the desired backend via the `OmniDSP::create(Backend)` factory method.

### Execution Model (Synchronous Core API)

- **Core Model:** All `Plan::execute` methods in the core API operate **synchronously**. The call blocks until the computation is complete and the output buffer is populated (or an error occurs). For stateful operations (filtering, resampling), this requires passing a mutable `State` object: `plan->execute(input, output, state)`.
- **Rationale:** Prioritizes API simplicity, safety (especially regarding state management), and efficiency for CPU-bound backends.
- **Asynchronous Execution (Future Direction - `Submission` Object):**
  - **Concept:** To support asynchronous operations safely, particularly for high-latency backends (GPU, FPGA, Cluster) and stateful Plans, a future extension could introduce a `Submission` object pattern.
  - **Workflow:** User creates a `Plan`, then requests async operation via `dsp.create_submission(plan, input, output)`. The `Submission` object internally creates and encapsulates the necessary `State`. User calls `submission.launch()` (or similar) returning `std::future<Status>`. The `Submission` manages state safely during background execution.
  - **Benefits:** Safer async state handling, leverages backend strategy pattern. Lifecycle details (resetting/reusing submissions) need careful design.
  - **Status:** This asynchronous `Submission` pattern is **not part of the initial core API** but represents the planned direction for future asynchronous capabilities. The core API remains synchronous.

### Concurrency and Thread Safety

- **Core Principle:** OmniDSP prioritizes maximum **single-threaded performance** and implementation simplicity by **not providing internal thread safety guarantees for its objects**. Parallelism, if desired, is achieved by selecting a backend implementation specifically designed for parallel execution or by user-managed threading.
- **User Responsibility:** Users **must assume** that `OmniDSP` instances, `Plan::execute` calls on the same Plan instance, and `State` objects are **NOT thread-safe** unless explicitly documented otherwise for a specific backend or feature. External synchronization (e.g., `std::mutex`) is required for concurrent access.
- **Parallelism Strategy:**
  1.  **Select Parallel Backend:** Choose a backend designed for parallelism (e.g., `Backend::CUDA`, `Backend::OpenMP_Default`) via `OmniDSP::create`. Follow that backend's specific concurrency model.
  2.  **Manual Parallelization:** Create multiple `OmniDSP` instances (typically one per application thread) using a standard backend (like `Backend::Default`) and manage task distribution externally.
- **Object Thread Safety Summary:**
  - **`OmniDSP` Instances:** **NOT thread-safe.** Requires external locking for concurrent calls.
  - **`Plan` Objects:** Immutable post-creation, potentially sharable. Concurrent `execute()` on the _same instance_ is **assumed unsafe** without external locking or specific backend guarantees.
  - **`State` Objects:** Mutable and **NOT thread-safe.** Requires a separate instance per stream/thread for the synchronous API.
  - **`Spec` Objects & `Type` Enums:** Immutable/stateless value types. Inherently **thread-safe** to copy/share.
- **Recommended Usage for Multithreading (Synchronous API):**
  - **Safest & Simplest:** One `OmniDSP` instance per thread, creating thread-local Plans and State.
  - **Using Parallel Backends:** Select a parallel backend. Follow its concurrency model. Use separate `State` per stream.
  - **Plan Sharing (Advanced):** Discouraged unless backend `execute` re-entrancy is guaranteed and performance benefits outweigh complexity. Requires separate `State` per thread and likely external locking around `execute`.

### Testing Strategy Implications

- **Backend Inheritance:** The use of the Default Backend Inheritance pattern (where optimized backends inherit from `DefaultOmniDSPImpl`) has implications for testing:
  - **Testing Overrides:** Methods explicitly overridden by an optimized backend (e.g., `AccelerateOmniDSPImpl::fft`) must be thoroughly tested against known results or reference implementations to ensure correctness and performance gains.
  - **Testing Inherited Methods:** It is also crucial to test the _inherited_ methods for each optimized backend. This verifies that the optimized backend correctly uses the functional baseline provided by `DefaultOmniDSPImpl` for operations it doesn't override itself.
  - **Testing Fallback Logic:** If an optimized backend implements logic to selectively fall back to the default implementation (`DefaultOmniDSPImpl::method()`) under certain conditions, these fallback paths must also be specifically tested.
- **Comprehensive Testing:** A robust test suite should cover all public API functions across all compiled backends, exercising both optimized and inherited code paths.

## Rationale Summary

This combination of components and patterns aims to create a DSP library that is:

1.  **Performant:** Leverages optimized backends (including potential parallel/hardware-accelerated ones) and provides a SIMD-optimized default implementation, prioritizing single-threaded speed in the core API.
2.  **Portable:** Provides a default backend that works (and performs well) across different platforms.
3.  **Easy to Use:** Offers a clear API separating configuration (Types, Specs), state management (State), and execution (Plans) via a central entry point (`OmniDSP`).
4.  **Maintainable & Extensible:** Uses established patterns like Pimpl, Factory, and Strategy to facilitate code organization and the addition of new features or backends (including diverse execution strategies like parallel CPU, GPU, FPGA, Cluster). The build system philosophy supports this maintainability.
5.  **Robust:** Employs RAII and `std::expected` for resource management and clear error handling.
6.  **Clear (Re: Concurrency):** Explicitly states that the core objects are not internally thread-safe, placing responsibility on the user for synchronization or choosing appropriate parallel backends. Defines the roles of Plans and State in managing concurrency for stateful operations, and outlines a potential path (`Submission` objects) for future safe asynchronous APIs.
