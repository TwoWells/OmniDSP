## OmniDSP Backend Developer Guide

This document provides guidelines for developers integrating new computation backends into the OmniDSP library. It reflects the architecture using distinct stateless `Plan` and stateful `Processor` objects, created via `create_plan`/`create_processor` factory methods on the `OmniDSP` facade. It incorporates the "Window as Plan" and "Transform as Plan" models, which use overloaded backend factory methods based on specific configuration structs (`Window::[SpecType]`, `Transform::[SpecType]`). The naming convention for backend PIMPL classes is `OmniDSP::[BackendName]::Plan::[OperationType]Impl` (or `OmniDSP::[BackendName]::Plan::[DomainSpecificType]::[ConcreteImplName]Impl`) or `OmniDSP::[BackendName]::Processor::[OperationType]Impl`. Abstract PIMPL interfaces follow `OmniDSP::Abstract::Plan::[OperationType]Impl` or `OmniDSP::Abstract::Processor::[OperationType]Impl`.

Refer to the main `Design Philosophy.md` for overarching goals.

**1. Overview: Directory Structure**

- **`include/OmniDSP/interface/`**: Defines abstract contracts in `OmniDSP::Abstract`.
  - `backend.hpp`: `Abstract::Backend` interface.
  - `plan/transform_impl.hpp`, `plan/window_impl.hpp` (example directory structure and filenames): Abstract interfaces like `Abstract::Plan::TransformImpl<T_In, T_Out>`, `Abstract::Plan::WindowImpl<T_Data>`.
  - `processor/fir_filter_impl.hpp` (example directory structure and filename): Abstract interfaces like `Abstract::Processor::FIRFilterImpl`.
- **Public API Headers** (e.g., `OmniDSP/plan/transform.hpp`, `OmniDSP/processor/fir_filter.hpp`, `OmniDSP/window.hpp`): Contain public `Plan`/`Processor` wrapper classes (`Plan::Transform`, `Processor::FIRFilter`, `Plan::Window`) which use Pimpl to point to their respective `Abstract::Plan::[Type]Impl` or `Abstract::Processor::[Type]Impl` interfaces.
- **`src/omnidsp/default/`**: Contains the default C++ backend implementation in `OmniDSP::Default`.
  - `backend.hpp/.cpp`: `Default::Backend` class (inherits `Abstract::Backend`).
  - `plan/` (e.g., `plan/transform/fft_c2c_impl.cpp`, `plan/window/hann_impl.cpp`): Implementations of `OmniDSP::Default::Plan::[DomainSpecificType]::[ConcreteImplName]Impl` (e.g., `OmniDSP::Default::Plan::Transform::FFTC2CImpl<C32>`, `OmniDSP::Default::Plan::Window::HannImpl<T_Data>`).
  - `processor/` (e.g., `fir_filter_processor.cpp`): Implementations of `OmniDSP::Default::Processor::[OperationType]Impl` (e.g., `OmniDSP::Default::Processor::FIRFilterImpl`).
- **`src/omnidsp/dispatcher/`**: `Dispatcher::Backend` implementation (inherits `Abstract::Backend`).
- **`src/omnidsp/[YourNamespace]/`** (e.g., `accelerate/`, `onemkl/`, `intelipp/`): Specific backend implementations. `[YourNamespace]` here refers to the actual C++ namespace chosen for the backend (e.g., `Accelerate`, `OneMKL`).
  - **Must inherit `OmniDSP::Default::Backend`**.
  - Contains implementations like `OmniDSP::[YourNamespace]::Plan::[DomainSpecificType]::[ConcreteImplName]Impl` and `OmniDSP::[YourNamespace]::Processor::[OperationType]Impl`.
- **PIMPL Naming Convention (using `[BackendName]` as a placeholder for the backend's conceptual name, often matching `[YourNamespace]`):**
  - Frontend API Class: `OmniDSP::Plan::[OperationType]` or `OmniDSP::Processor::[OperationType]` (e.g., `OmniDSP::Plan::Transform`, `OmniDSP::Processor::FIRFilter`).
  - Abstract PIMPL Interface: `OmniDSP::Abstract::Plan::[OperationType]Impl` or `OmniDSP::Abstract::Processor::[OperationType]Impl` (e.g., `OmniDSP::Abstract::Plan::TransformImpl<T_In, T_Out>`, `OmniDSP::Abstract::Processor::FIRFilterImpl`).
  - Concrete Backend PIMPL: `OmniDSP::[BackendName]::Plan::[DomainSpecificType]::[ConcreteImplName]Impl` or `OmniDSP::[BackendName]::Processor::[OperationType]Impl`.
  - Example: `OmniDSP::Plan::Transform<C32, C32>` (frontend for FFT C2C) -> `OmniDSP::Abstract::Plan::TransformImpl<C32, C32>` (PIMPL interface) -> `OmniDSP::Default::Plan::Transform::FFTC2CImpl<C32>` (concrete default backend PIMPL for FFT C2C). Here, `[BackendName]` is `Default`, `[DomainSpecificType]` is `Transform`, and `[ConcreteImplName]` is `FFTC2C`.
- **Other source directories:** `params/`, `design/`, `window/` (for internal formulas), `transform/` (for internal transform helpers if any), `omnidsp.cpp`.

**2. Overview: Backend Architecture**

- **`OmniDSP` (Public Class):** User-facing entry point. Holds `pimpl_` (`std::unique_ptr<Abstract::Backend>`). Created via `OmniDSP::create()`. Acts as factory.
- **`Abstract::Backend` (Interface):** Defines pure virtual contract (`get_backend_type()`) and virtual factory methods:
  - Example Transform Plan factory: `template<typename T_Complex> virtual OmniExpected<std::unique_ptr<Abstract::Plan::TransformImpl<T_Complex, T_Complex>>> create_transform_plan_impl(const Transform::FFT& spec) const = 0;`
  - Example Processor factory: `virtual OmniExpected<std::unique_ptr<Abstract::Processor::FIRFilterImpl<float>>> create_fir_filter_processor_impl(const Design::FIRFilter& design) = 0;`
  - Overloaded `create_window_plan_impl<T_Data>(const Window::[SpecType]& spec, size_t length)` for each window specification struct.
  - Overloaded `create_transform_plan_impl<T_In, T_Out>(const Transform::[SpecType]& spec)` for each transform specification struct.
- **`Default::Backend` (Base Implementation):** Inherits `Abstract::Backend`. Provides standard C++ implementations for all virtual methods. Base class for optimized backends.
- **`OmniDSP::[YourNamespace]::Backend` (Optimized Backend):** Inherits `OmniDSP::Default::Backend`. `[YourNamespace]` is the C++ namespace of the backend. Overrides methods it optimizes.
- **`Dispatcher::Backend`:** Inherits `Abstract::Backend`. Routes calls.
- **`OmniDSP::Abstract::Plan::[OperationType]Impl` (Stateless PIMPL Interface):** Defines interface for backend-specific stateless plan PIMPLs (e.g., `Abstract::Plan::TransformImpl<T_In, T_Out>`, `Abstract::Plan::ConvolutionImpl<T>`).
- **`OmniDSP::Abstract::Processor::[OperationType]Impl` (Stateful PIMPL Interface):** Defines interface for backend-specific stateful processor PIMPLs.
- **`OmniDSP::[BackendName]::Plan::[DomainSpecificType]::[ConcreteImplName]Impl` / `OmniDSP::[BackendName]::Processor::[OperationType]Impl` (Concrete Backend PIMPLs):**
  These classes are the concrete implementations of the abstract PIMPL interfaces. They reside within their backend's namespace structure (e.g., `OmniDSP::Default::Plan::Transform::FFTC2CImpl<C32>` or `OmniDSP::IntelIPP::Processor::FIRFilterImpl`).

**3. Implementing the Core Backend Class (`OmniDSP::[YourNamespace]::Backend`)**

- **Inheritance:** `namespace OmniDSP { namespace [YourNamespace] { class Backend final : public OmniDSP::Default::Backend { /* ... */ }; } }`
- **Override Virtual Functions:**
  - `get_backend_type()`: **Must** override.
  - Plan Factories (e.g., `create_convolution_plan_impl`): Override if your backend provides an optimized PIMPL for that operation.
  - Window Plan Factories (`create_window_plan_impl`): Override specific overloads (e.g., for `Window::Hann`, `Window::Kaiser`) for window types your backend optimizes.
  - Transform Plan Factories (`create_transform_plan_impl`): Override specific overloads (e.g., for `Transform::FFT`, `Transform::RFFT`) for transform types your backend optimizes.
  - Processor Factories (e.g., `create_fir_filter_processor_impl`, `create_cqt_processor_impl`): Override if your backend provides an optimized PIMPL for that operation.
- Use `override`. Un-overridden methods fall through to `Default::Backend`'s implementation.

**4. Implementing Plan PIMPL Classes (Stateless)**

- **Namespace and Class Name:** e.g., `OmniDSP::[BackendName]::Plan::Transform::FFTC2CImpl<C32>`, `OmniDSP::[BackendName]::Plan::Window::KaiserImpl<T_Data>`.
- **Inheritance:** Inherit from the corresponding `OmniDSP::Abstract::Plan::[Type]Impl` interface (e.g., `OmniDSP::Abstract::Plan::TransformImpl<C32, C32>`, `OmniDSP::Abstract::Plan::WindowImpl<T_Data>`).
- **Constructor:** Takes configuration (`Params::[OperationType]`, `Window::[SpecType]` & length, `Transform::[SpecType]`, or `Coefs::[OperationType]`). Performs backend setup.
- **Destructor:** Cleans up resources.
- **Execution Methods:**
  - For `TransformImpl`: `forward(...)`, `inverse(...)`.
  - For `WindowImpl`: `execute(...)`, `execute_inplace(...)`.
  - For `ConvolutionImpl`/`CorrelationImpl`: `execute(...)`.

**5. Implementing Processor PIMPL Classes (Stateful)**

- **Namespace and Class Name:** e.g., `OmniDSP::[BackendName]::Processor::FIRFilterImpl`.
- **Inheritance:** Inherit from `OmniDSP::Abstract::Processor::[OperationType]Impl<DataType>`.
- **Internal State:** Member variables for state.
- **Constructor:** Takes configuration (`Design::[OperationType]` or `Coefs::[OperationType]`). Performs setup and **initializes internal state**.
- **Destructor:** Cleans up resources and state.
- **`execute(...)`:** Implements the operation, **modifying internal state**.
- **`reset()`:** Resets internal state.

**6. The `Dispatcher::Backend`**

- **Role:** A specialized concrete implementation of `OmniDSP::Abstract::Backend`. Allows runtime selection of different backend implementations for various `OperationCategory` enum values.
- **Operation:**
  1.  When a factory method (e.g., `create_transform_plan_impl(const Transform::FFT&, ...)` (formerly `create_fft_plan_impl`), `create_fir_filter_processor_impl`, or any `create_window_plan_impl` overload) is called:
  2.  It determines the `OperationCategory` (e.g., `FFT`, `FIRFilter`, `Windowing`, `Transform`).
  3.  Consults its internal map for an override for that `OperationCategory`.
  4.  If an override is found, forwards the call to the registered override `Abstract::Backend`.
  5.  Otherwise, forwards to the primary `Abstract::Backend`.

**7. Integrating with the Factory (`OmniDSP` facade)**

- **User Interaction:** Users interact with `OmniDSP` facade.
- **Facade's Role:** Public factory methods on `OmniDSP` (e.g., `OmniDSP::create_plan(const Transform::FFT& spec)`, `OmniDSP::create_plan(const Window::Kaiser& spec, size_t length)`, `OmniDSP::create_processor(const Params::FIRFilter& params)`) are entry points.
- **Internal Dispatch:** Facade methods call corresponding `create_*_impl` on its `pimpl_` (`std::unique_ptr<Abstract::Backend>`).
- **Backend Developer's Responsibility:** Implement `[YourNamespace]::Backend`, adhering to `Abstract::Backend` by overriding necessary methods from `Default::Backend`.

**8. Integrating with the Build System (CMake)**

- **Create Backend Directory:** Add a new subdirectory for your backend under `src/omnidsp/`, for example, `src/omnidsp/[your_namespace]/` (e.g., `src/omnidsp/my_new_backend/`). This directory will house all source (`.cpp`) and header (`.hpp`) files specific to your backend's implementation.
- **Backend CMakeLists.txt:** Inside this new directory, create a `CMakeLists.txt` file. This file will list all source files for your backend implementation (e.g., `my_new_backend_impl.cpp`, `my_new_transform_fft_impl.cpp`, `my_new_fir_filter_processor_impl.cpp`). It should define how these files are compiled, typically by adding them to the main `OmniDSP` library target.

  ```cmake
  # Example: src/omnidsp/my_new_backend/CMakeLists.txt
  set(MY_NEW_BACKEND_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
      # Add plan impl sources, e.g., for Transform plans
      "${CMAKE_CURRENT_SOURCE_DIR}/plan/transform/my_fft_impl.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/plan/window/my_hann_impl.cpp"
      # Add processor impl sources
      "${CMAKE_CURRENT_SOURCE_DIR}/processor/my_fir_impl.cpp"
      # ... other .cpp files ...
  )

  # Add these sources to the main OmniDSP library target
  # The 'OmniDSP' target is defined in src/omnidsp/CMakeLists.txt
  target_sources(OmniDSP PRIVATE ${MY_NEW_BACKEND_SOURCES})

  # Add include directory for this backend's headers, so its .cpp files can find them
  target_include_directories(OmniDSP PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  ```

- **Update Main CMakeLists.txt:** In `src/omnidsp/CMakeLists.txt`, add your backend's subdirectory using `add_subdirectory([your_namespace])`, guarded by its CMake option.
  ```cmake
  # In src/omnidsp/CMakeLists.txt
  if(OPTION_OMNIDSP_ENABLE_MY_NEW_BACKEND)
      add_subdirectory(my_new_backend)
  endif()
  ```
- **CMake Option:** Introduce a new CMake option (e.g., in `cmake/OmniDSPBackendOptions.cmake` or the top-level `CMakeLists.txt`) to enable/disable your backend.
  ```cmake
  option(OMNIDSP_ENABLE_MY_NEW_BACKEND "Enable MyNewBackend support" OFF)
  ```
- **Conditional Compilation (C++):** Use a preprocessor definition set by CMake to conditionally compile your backend's code within the OmniDSP library. This definition is typically set in your backend's `CMakeLists.txt` when the backend is enabled.
  ```cmake
  # In src/omnidsp/my_new_backend/CMakeLists.txt (inside if(OPTION_OMNIDSP_ENABLE_MY_NEW_BACKEND))
  target_compile_definitions(OmniDSP PRIVATE OMNIDSP_INTERNAL_USE_MY_NEW_BACKEND)
  # Also set the PARENT_SCOPE variable for omnidsp_config.h
  set(OMNIDSP_BACKEND_MY_NEW_BACKEND_ENABLED 1 PARENT_SCOPE)
  ```
  Then, in C++ code (e.g., in `OmniDSP::create` or `Dispatcher::Backend`):
  ```cpp
  #include <OmniDSP/omnidsp_config.hpp> // For generated config header
  // ...
  #if OMNIDSP_BACKEND_MY_NEW_BACKEND_ENABLED
  // Code related to MyNewBackend
  #endif
  ```
- **External Dependencies:** If your backend relies on external libraries (e.g., a specific hardware SDK), use `find_package()` in your backend's `CMakeLists.txt`. Ensure you handle cases where dependencies are not found (e.g., by disabling your backend or issuing a warning). Link these dependencies to the `OmniDSP` target (e.g., `target_link_libraries(OmniDSP PRIVATE MyDependency::MyDependency)`).
- **Testing:** Ensure your backend's tests are also conditionally compiled and run based on the CMake option. Test files should be structured similarly, perhaps under `tests/cpp/backends/[your_namespace]/`.

**9. Adhering to API Conventions**

- **`OmniExpected` and `OmniStatus` Enum:**
  - All factory methods in the `Abstract::Backend` interface (e.g., `create_transform_plan_impl`, `create_fir_filter_processor_impl`) and their overrides in concrete backend classes **must** return an `OmniExpected<std::unique_ptr<Abstract::Plan::[Type]ImplSubclass>, OmniStatus>` or `OmniExpected<std::unique_ptr<Abstract::Processor::[Type]ImplSubclass>, OmniStatus>`.
  - On successful creation of the PIMPL object, return `std::move(unique_ptr_to_pimpl)`.
  - If creation fails within the factory method itself (e.g., due to an issue caught before PIMPL construction, or an exception caught from PIMPL construction), return `std::unexpected(OmniStatus::ErrorCode)` where `ErrorCode` is an appropriate value from the `OmniDSP::OmniStatus` enum (e.g., `OmniStatus::BackendError`, `OmniStatus::ConfigurationError`).
- **PIMPL Constructors and Exceptions:**
  - Constructors of your concrete PIMPL classes (e.g., `OmniDSP::[BackendName]::Plan::Transform::FFTC2CImpl(...)`) are responsible for all backend-specific setup, resource allocation, and validation of parameters passed to them.
  - If an unrecoverable error occurs during PIMPL construction (e.g., a backend library call fails, memory allocation fails, an invalid configuration is detected that wasn't caught earlier), the constructor **should throw an appropriate C++ exception** (e.g., `std::runtime_error`, `std::bad_alloc`, or a custom `OmniException` type).
  - The `create_*_impl` methods in your `OmniDSP::[YourNamespace]::Backend` class (which call these PIMPL constructors) **must** catch these exceptions and convert them into an `OmniExpected` with a relevant `OmniStatus` code. For example:
    ```cpp
    // In YourNamespace::Backend::create_transform_plan_impl for FFT
    try {
        // Assuming YourNamespace::Plan::Transform::FFTC2CImpl exists
        auto pimpl = std::make_unique<YourNamespace::Plan::Transform::FFTC2CImpl<C32>>(spec.length);
        return std::move(pimpl);
    } catch (const OmniException& e) {
        // Log e.what() and e.get_status() if a logging mechanism exists
        return std::unexpected(e.get_status());
    } catch (const std::bad_alloc&) {
        return std::unexpected(OmniStatus::AllocationError);
    } catch (const std::exception& e) {
        // Log e.what()
        return std::unexpected(OmniStatus::BackendError);
    } // ... other specific exceptions
    ```
- **`std::span` for Data Buffers:**
  - Methods within your PIMPL classes that process data (e.g., `forward(...)`, `inverse(...)`, `execute(...)`) should accept data buffers as `std::span<T>` or `std::span<const T>`. This provides a safe, non-owning view of contiguous memory and clearly communicates whether the buffer is for input or output.
- **Return `OmniStatus` from execution methods:**
  - The execution methods of PIMPL classes (e.g., `forward`, `inverse`, `execute`) and `reset()` for processors should typically return an `OmniDSP::OmniStatus` enum value to indicate success (`OmniStatus::Success`) or a specific runtime error encountered during execution.
- **Resource Management (RAII):**
  - Your PIMPL classes must manage any backend-specific resources (e.g., library handles, allocated memory buffers not owned by the user) using RAII principles. This usually means acquiring resources in the constructor and releasing them in the destructor. `std::unique_ptr` with custom deleters can be useful here.
- **Templating for Data Types:**
  - Ensure PIMPL interfaces (like `Abstract::Plan::TransformImpl<T_In, T_Out>`, `Abstract::Plan::WindowImpl<T_Data>`) and their concrete implementations are correctly templated for data types (typically `float` and `double`, and their complex counterparts) where the operation supports multiple precisions.
- **No Direct Console Output:**
  - Library code (including backend implementations) **must not** use `printf`, `std::cout`, `std::cerr`, etc., for error reporting or logging. Error conditions should be propagated using `OmniExpected` from factory methods and `OmniStatus` return codes from execution methods. For internal diagnostic messages or detailed error information not suitable for propagation via `OmniStatus` codes (e.g., during development or debugging), use the integrated `spdlog` library. Ensure that logging levels are appropriately managed (e.g., debug logs should not be active in release builds by default).

This structure, with clearly namespaced backend PIMPLs like `OmniDSP::[BackendName]::Plan::Transform::FFTC2CImpl`, ensures a consistent and maintainable architecture.
