## OmniDSP Backend Developer Guide

This document provides guidelines for developers integrating new computation backends into the OmniDSP library. It reflects the architecture using distinct stateless `Plan` and stateful `Processor` objects, created via `create_plan`/`create_processor` (and `create_window_plan`) factory methods on the `OmniDSP` facade, and the new "Window as Plan" model with overloaded backend factory methods. The naming convention for backend PIMPL classes is `OmniDSP::[BackendName]::Plan::[OperationType]Impl` or `OmniDSP::[BackendName]::Processor::[OperationType]Impl`. Abstract PIMPL interfaces follow `OmniDSP::Abstract::Plan::[OperationType]Impl` or `OmniDSP::Abstract::Processor::[OperationType]Impl`.

Refer to the main `Design Philosophy.md` for overarching goals.

**1. Overview: Directory Structure**

- **`include/OmniDSP/interface/`**: Defines abstract contracts in `OmniDSP::Abstract`.
  - `backend.hpp`: `Abstract::Backend` interface.
  - `plan/fft_impl.hpp`, `plan/window_impl.hpp` (example directory structure and filenames): Abstract interfaces like `Abstract::Plan::FFTImpl`, `Abstract::Plan::WindowImpl<T_Data>`.
  - `processor/fir_filter_impl.hpp` (example directory structure and filename): Abstract interfaces like `Abstract::Processor::FIRFilterImpl`.
- **Public API Headers** (e.g., `OmniDSP/fft.hpp`, `OmniDSP/fir_filter.hpp`, `OmniDSP/window.hpp`): Contain public `Plan`/`Processor` wrapper classes (`Plan::FFT`, `Processor::FIRFilter`, `Plan::Window`) which use Pimpl to point to `Abstract::Plan::[OperationType]Impl` or `Abstract::Processor::[OperationType]Impl`.
- **`src/omnidsp/default/`**: Contains the default C++ backend implementation in `OmniDSP::Default`.
  - `backend.hpp/.cpp`: `Default::Backend` class (inherits `Abstract::Backend`).
  - `plan/` (e.g., `fft_plan.cpp`, `window/hann_plan.cpp`): Implementations of `OmniDSP::Default::Plan::[OperationType]Impl` (e.g., `OmniDSP::Default::Plan::FFTImpl`, `OmniDSP::Default::Plan::Window::HannImpl<T_Data>`).
  - `processor/` (e.g., `fir_filter_processor.cpp`): Implementations of `OmniDSP::Default::Processor::[OperationType]Impl` (e.g., `OmniDSP::Default::Processor::FIRFilterImpl`).
- **`src/omnidsp/dispatcher/`**: `Dispatcher::Backend` implementation (inherits `Abstract::Backend`).
- **`src/omnidsp/[YourNamespace]/`** (e.g., `accelerate/`, `onemkl/`, `intelipp/`): Specific backend implementations. `[YourNamespace]` here refers to the actual C++ namespace chosen for the backend (e.g., `Accelerate`, `OneMKL`).
  - **Must inherit `OmniDSP::Default::Backend`**.
  - Contains implementations like `OmniDSP::[YourNamespace]::Plan::[OperationType]Impl` and `OmniDSP::[YourNamespace]::Processor::[OperationType]Impl`.
- **PIMPL Naming Convention (using `[BackendName]` as a placeholder for the backend's conceptual name, often matching `[YourNamespace]`):**
  - Frontend API Class: `OmniDSP::Plan::[OperationType]` or `OmniDSP::Processor::[OperationType]` (e.g., `OmniDSP::Plan::FFT`, `OmniDSP::Processor::FIRFilter`).
  - Abstract PIMPL Interface: `OmniDSP::Abstract::Plan::[OperationType]Impl` or `OmniDSP::Abstract::Processor::[OperationType]Impl` (e.g., `OmniDSP::Abstract::Plan::FFTImpl`, `OmniDSP::Abstract::Processor::FIRFilterImpl`).
  - Concrete Backend PIMPL: `OmniDSP::[BackendName]::Plan::[OperationType]Impl` or `OmniDSP::[BackendName]::Processor::[OperationType]Impl`. If multiple variants exist for an operation within a backend: `OmniDSP::[BackendName]::Plan::[OperationType]::[VariantName]Impl`.
  - Example: `OmniDSP::Plan::Window` (frontend) -> `OmniDSP::Abstract::Plan::WindowImpl<T_Data>` (PIMPL interface) -> `OmniDSP::Default::Plan::Window::HannImpl<T_Data>` (concrete default backend PIMPL for Hann window). Here, `[BackendName]` is `Default`, `[OperationType]` is `Window`, and `[VariantName]` (or specifier) is `Hann`.
- **Other source directories:** `params/`, `design/`, `window/` (for internal formulas), `omnidsp.cpp`.

**2. Overview: Backend Architecture**

- **`OmniDSP` (Public Class):** User-facing entry point. Holds `pimpl_` (`std::unique_ptr<Abstract::Backend>`). Created via `OmniDSP::create()`. Acts as factory.
- **`Abstract::Backend` (Interface):** Defines pure virtual contract (`get_backend_type()`) and virtual factory methods:
  - Example Plan factory: `virtual OmniExpected<std::unique_ptr<Abstract::Plan::FFTImpl<float>>> create_fft_plan_impl(const Params::FFT& params, FFTType fft_type, bool inverse) = 0;`
  - Example Processor factory: `virtual OmniExpected<std::unique_ptr<Abstract::Processor::FIRFilterImpl<float>>> create_fir_filter_processor_impl(const Design::FIRFilter& design) = 0;`
  - Overloaded `create_window_plan_impl<T_Data>(const Window::[SpecType]& spec, size_t length)` for each window specification struct (e.g., `Window::Hann`, `Window::Kaiser`).
    - Example: `virtual OmniExpected<std::unique_ptr<Abstract::Plan::WindowImpl<T_Data>>> create_window_plan_impl(const Window::Kaiser& spec, size_t length) = 0;`
- **`Default::Backend` (Base Implementation):** Inherits `Abstract::Backend`. Provides standard C++ implementations for all virtual methods. Base class for optimized backends.
- **`OmniDSP::[YourNamespace]::Backend` (Optimized Backend):** Inherits `OmniDSP::Default::Backend`. `[YourNamespace]` is the C++ namespace of the backend. Overrides methods it optimizes.
- **`Dispatcher::Backend`:** Inherits `Abstract::Backend`. Routes calls.
- **`OmniDSP::Abstract::Plan::[OperationType]Impl` (Stateless PIMPL Interface):** Defines interface for backend-specific stateless plan PIMPLs.
- **`OmniDSP::Abstract::Processor::[OperationType]Impl` (Stateful PIMPL Interface):** Defines interface for backend-specific stateful processor PIMPLs.
- **`OmniDSP::[BackendName]::Plan::[OperationType]Impl` / `OmniDSP::[BackendName]::Processor::[OperationType]Impl` (Concrete Backend PIMPLs):**
  These classes are the concrete implementations of the `Abstract::Plan::[OperationType]Impl` or `Abstract::Processor::[OperationType]Impl` interfaces, respectively. They reside within their backend's namespace structure (e.g., `OmniDSP::Default::Plan::FFTImpl` or `OmniDSP::IntelIPP::Processor::FIRFilterImpl`). `[BackendName]` is the conceptual name of the backend (e.g., `Default`, `IntelIPP`), and `[OperationType]` is the specific operation (e.g., `FFT`, `FIRFilter`). They use the specific backend's API to perform operations.

**3. Implementing the Core Backend Class (`OmniDSP::[YourNamespace]::Backend`)**

- **Inheritance:** `namespace OmniDSP { namespace [YourNamespace] { class Backend final : public OmniDSP::Default::Backend { /* ... */ }; } }`
  (Replace `[YourNamespace]` with your backend's actual C++ namespace, e.g., `Accelerate`, `OneMKL`).
- **Override Virtual Functions:**
  - `get_backend_type()`: **Must** override.
  - Plan Factories (e.g., `create_fft_plan_impl`, `create_convolution_plan_impl`): Override if your backend provides an optimized PIMPL for that operation.
  - Window Plan Factories (`create_window_plan_impl`): Override specific overloads (e.g., for `Window::Hann`, `Window::Kaiser`) for window types your backend optimizes.
  - Processor Factories (e.g., `create_fir_filter_processor_impl`, `create_cqt_processor_impl`): Override if your backend provides an optimized PIMPL for that operation.
- Use `override`. Un-overridden methods fall through to `Default::Backend`'s implementation.

**4. Implementing Plan PIMPL Classes (`OmniDSP::[BackendName]::Plan::[OperationType]Impl`) - Stateless**

- **Namespace and Class Name:** e.g., `OmniDSP::[BackendName]::Plan::FFTImpl` or `OmniDSP::[BackendName]::Plan::Window::KaiserImpl<T_Data>`.
  (Replace `[BackendName]` with your backend's name, e.g., `MyAccelerateBackend`, and `[OperationType]` or `Window::[SpecName]` accordingly).
- **Inheritance:** Inherit from `OmniDSP::Abstract::Plan::[OperationType]Impl<DataType>` (e.g., `OmniDSP::Abstract::Plan::FFTImpl<float>`, `OmniDSP::Abstract::Plan::WindowImpl<T_Data>`).
- **Constructor:** Takes configuration (`Params::[OperationType]`, `Window::[SpecType]` & length, or `Coefs::[OperationType]`). Performs backend setup.
- **Destructor:** Cleans up resources.
- **`execute(...)`:** Implements the core operation. For `Plan::WindowImpl`: `execute(const T_Data* src, T_Data* dest, size_t length)` and `execute_inplace(T_Data* src_dest, size_t length)`.

**5. Implementing Processor PIMPL Classes (`OmniDSP::[BackendName]::Processor::[OperationType]Impl`) - Stateful**

- **Namespace and Class Name:** e.g., `OmniDSP::[BackendName]::Processor::FIRFilterImpl`.
  (Replace `[BackendName]` and `[OperationType]` accordingly).
- **Inheritance:** Inherit from `OmniDSP::Abstract::Processor::[OperationType]Impl<DataType>` (e.g., `OmniDSP::Abstract::Processor::FIRFilterImpl<float>`).
- **Internal State:** Member variables for state (e.g., delay lines).
- **Constructor:** Takes configuration (`Design::[OperationType]` or `Coefs::[OperationType]`). Performs setup and **initializes internal state**.
- **Destructor:** Cleans up resources and state.
- **`execute(...)`:** Implements the operation, **modifying internal state**.
- **`reset()`:** Resets internal state.

**6. The `Dispatcher::Backend`**

- **Role:** The `Dispatcher::Backend` is a specialized concrete implementation of `OmniDSP::Abstract::Backend`. Its primary purpose is to allow runtime selection of different backend implementations for various categories of DSP operations (defined by the `OperationCategory` enum).
- **Construction:** It is typically instantiated by the `OmniDSP::create` factory method when the user provides a map of `OperationCategory` to specific `BackendType` overrides. It holds a pointer to a primary (default) `Abstract::Backend` instance and a map of `OperationCategory` to overriding `Abstract::Backend` instances.
- **Operation:**
  1.  When a factory method (e.g., `create_fft_plan_impl`, `create_fir_filter_processor_impl`, or any `create_window_plan_impl` overload) is called on the `Dispatcher::Backend` instance:
  2.  It first determines the `OperationCategory` relevant to the called method (e.g., `OperationCategory::FFT`, `OperationCategory::FIRFilter`, `OperationCategory::Windowing`).
  3.  It then consults its internal map to see if an override backend instance is registered for that specific `OperationCategory`.
  4.  If an override is found, the call (along with all its parameters) is forwarded to the corresponding factory method of the registered override `Abstract::Backend` instance.
  5.  If no override is found for that category, the call is forwarded to the primary `Abstract::Backend` instance.
- **Inheritance:** `Dispatcher::Backend` directly inherits from `OmniDSP::Abstract::Backend` and overrides all its pure virtual factory methods to implement this dispatching logic.

**7. Integrating with the Factory (`OmniDSP` facade)**

- **User Interaction:** Users interact with the `OmniDSP` facade class, not directly with `Abstract::Backend` or its concrete implementations (like `Default::Backend` or `YourNamespace::Backend`).
- **Facade's Role:** The public factory methods on the `OmniDSP` instance (e.g., `OmniDSP::create_plan(const Params::FFT& params)`, `OmniDSP::create_window_plan<float>(const Window::Kaiser& spec, size_t length)`, `OmniDSP::create_processor(const Params::FIRFilter& params)`) serve as the entry points.
- **Internal Dispatch:** Each of these facade methods internally calls the corresponding `create_*_impl` method on its `pimpl_` member, which is a `std::unique_ptr<Abstract::Backend>`. This `pimpl_` could point to a `Default::Backend`, an `Accelerate::Backend`, a `Dispatcher::Backend`, etc., depending on how the `OmniDSP` instance was created.
  - For example, `dsp.create_plan(fft_params)` might internally call `pimpl_->create_fft_plan_impl(params, type, inverse)`.
- **Backend Developer's Responsibility:** Backend developers focus on correctly implementing their `[YourNamespace]::Backend` class, ensuring it fully adheres to the `Abstract::Backend` interface (by overriding necessary methods from `Default::Backend`). The `OmniDSP` facade will then be able to use this backend implementation seamlessly.
- **Backend Registration (Conceptual):**
  - When adding a new backend (e.g., `MyNewBackend`), you'll need to add a corresponding entry to the `BackendType` enum (e.g., `BackendType::MyNewBackend`).
  - The main `OmniDSP::create(BackendType type, ...)` factory method will need to be updated to recognize this new `BackendType` and know how to instantiate `OmniDSP::MyNewBackend::Backend`.
  - The `OmniDSP::get_available_backends()` method should also be updated to report the availability of `MyNewBackend` if it's compiled in and its dependencies are met.

**8. Integrating with the Build System (CMake)**

- **Create Backend Directory:** Add a new subdirectory for your backend under `src/omnidsp/`, for example, `src/omnidsp/[your_namespace]/` (e.g., `src/omnidsp/my_new_backend/`).
- **Backend CMakeLists.txt:** Inside this new directory, create a `CMakeLists.txt` file. This file will list all source files (`.cpp`, `.h`) for your backend implementation (e.g., `my_new_backend_impl.cpp`, `my_new_fft_plan_impl.cpp`). It should define how these files are compiled, typically by adding them to a source group or directly to the parent library target.
- **Update Main CMakeLists.txt:** Add your backend's subdirectory to the main `src/omnidsp/CMakeLists.txt` using `add_subdirectory([your_namespace])`. This ensures CMake processes your backend's build script.
- **CMake Option:** Introduce a new CMake option to enable/disable your backend. This is typically done in a central CMake file (e.g., `cmake/OmniDSPBackendOptions.cmake` or the top-level `CMakeLists.txt`).
  ```cmake
  option(OMNIDSP_ENABLE_MY_NEW_BACKEND "Enable MyNewBackend support" OFF) # Or ON by default
  ```
- **Conditional Compilation:** Use this option to conditionally compile your backend's code and link its dependencies.

  ```cmake
  if(OMNIDSP_ENABLE_MY_NEW_BACKEND)
      # Add sources to the main omnidsp library target
      target_sources(omnidsp PRIVATE
          ${CMAKE_CURRENT_LIST_DIR}/[your_namespace]/my_new_backend_impl.cpp
          # ... other source files ...
      )
      # Find and link external dependencies if any
      # find_package(MyDependency REQUIRED)
      # target_link_libraries(omnidsp PRIVATE MyDependency::MyDependency)

      # Set a preprocessor definition to indicate the backend is available
      target_compile_definitions(omnidsp PRIVATE OMNIDSP_HAS_MY_NEW_BACKEND)
  endif()
  ```

- **External Dependencies:** If your backend relies on external libraries, use `find_package()` to locate them. Ensure you handle cases where dependencies are not found (e.g., by disabling your backend or issuing a warning). Consider creating `Find[YourDependency].cmake` modules if they don't exist.
- **Testing:** Ensure your backend's tests are also conditionally compiled and run based on the CMake option.

**9. Adhering to API Conventions**

- **`OmniExpected` and `Status` Enum:**
  - All factory methods in the `Abstract::Backend` interface (e.g., `create_fft_plan_impl`, `create_fir_filter_processor_impl`) and their overrides in concrete backend classes **must** return an `OmniExpected<std::unique_ptr<Abstract::Plan::[OperationType]ImplSubclass>, Status>` or `OmniExpected<std::unique_ptr<Abstract::Processor::[OperationType]ImplSubclass>, Status>`.
  - On successful creation of the PIMPL object, return `std::move(unique_ptr_to_pimpl)`.
  - If creation fails within the factory method itself (e.g., due to an issue caught before PIMPL construction, or an exception caught from PIMPL construction), return `std::unexpected(Status::ErrorCode)` where `ErrorCode` is an appropriate value from the `OmniDSP::Status` enum (e.g., `Status::BackendError`, `Status::ConfigurationError`).
- **PIMPL Constructors and Exceptions:**
  - Constructors of your concrete PIMPL classes (e.g., `OmniDSP::[BackendName]::Plan::FFTImpl(...)`) are responsible for all backend-specific setup, resource allocation, and validation of parameters passed to them.
  - If an unrecoverable error occurs during PIMPL construction (e.g., a backend library call fails, memory allocation fails, an invalid configuration is detected that wasn't caught earlier), the constructor **should throw an appropriate C++ exception** (e.g., `std::runtime_error`, `std::bad_alloc`, or a custom OmniDSP exception type).
  - The `create_*_impl` methods in your `OmniDSP::[YourNamespace]::Backend` class (which call these PIMPL constructors) **must** catch these exceptions and convert them into an `OmniExpected` with a relevant `Status` code. For example:
    ```cpp
    // In YourNamespace::Backend::create_fft_plan_impl
    try {
        auto pimpl = std::make_unique<YourNamespace::Plan::FFTImpl>(params); // Assuming YourNamespace::Plan::FFTImpl
        return std::move(pimpl);
    } catch (const std::runtime_error& e) {
        // Log e.what() if a logging mechanism exists, e.g., using spdlog
        // spdlog::error("Plan::FFTImpl construction failed: {}", e.what());
        return std::unexpected(Status::BackendError);
    } catch (const std::bad_alloc&) {
        // spdlog::error("Plan::FFTImpl construction failed: Out of memory");
        return std::unexpected(Status::OutOfMemory);
    } // ... other specific exceptions
    ```
- **`std::span` for Data Buffers:**
  - Methods within your PIMPL classes that process data (e.g., `execute(...)`) should accept data buffers as `std::span<T>` or `std::span<const T>`. This provides a safe, non-owning view of contiguous memory and clearly communicates whether the buffer is for input or output.
- **Return `Status` from `execute` methods:**
  - The `execute(...)` methods of PIMPL classes (and `reset()` for processors) should typically return an `OmniDSP::Status` enum value to indicate success (`Status::Success`) or a specific runtime error encountered during execution.
- **Resource Management (RAII):**
  - Your PIMPL classes must manage any backend-specific resources (e.g., library handles, allocated memory buffers not owned by the user) using RAII principles. This usually means acquiring resources in the constructor and releasing them in the destructor. `std::unique_ptr` with custom deleters can be useful here.
- **Templating for Data Types:**
  - Ensure PIMPL interfaces (like `Abstract::Plan::WindowImpl<T_Data>`) and their concrete implementations are correctly templated for data types (typically `float` and `double`) where the operation supports multiple precisions.
- **No Direct Console Output:**
  - Library code (including backend implementations) **must not** use `printf`, `std::cout`, `std::cerr`, etc., for error reporting or logging. Error conditions should be propagated using `OmniExpected` from factory methods and `Status` return codes from execution methods. For internal diagnostic messages or detailed error information not suitable for propagation via `Status` codes (e.g., during development or debugging), use the integrated `spdlog` library. Ensure that logging levels are appropriately managed (e.g., debug logs should not be active in release builds by default).

This structure, with clearly namespaced backend PIMPLs like `OmniDSP::[BackendName]::Plan::[OperationType]Impl`, ensures a consistent and maintainable architecture.
