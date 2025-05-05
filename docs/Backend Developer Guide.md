## OmniDSP Backend Developer Guide

This document provides a roadmap and guidelines for developers looking to integrate a new computation backend (e.g., for specific hardware like GPUs/FPGAs, or a different acceleration library) into the OmniDSP library. It outlines the necessary steps and architectural patterns involved.

Refer to the main Design Philosophy.md for the overarching architectural goals.

**1\. Overview: Directory Structure**

The core C++ source code (src/omnidsp/) is organized as follows:

- **interface/**: Defines the abstract backend contract (`AbstractBackend`, `*PlanImpl` interfaces) in the `OmniDSP::abstract` namespace. Also implements the public Plan class wrappers (e.g., `FFTPlan`, `CQTPlan`) which forward calls to the backend-specific `*PlanImpl` via the Pimpl pattern.
- **default/**: Contains the default backend implementation using **standard C++** in the `OmniDSP::default` namespace.
  - backend.hpp/.cpp: Defines and implements `DefaultBackend`. **Crucially, `DefaultBackend` inherits from `abstract::AbstractBackend` and provides baseline implementations for all virtual methods.**
  - fft.cpp, cqt.cpp, etc.: Implementations of `DefaultFFTPlanImpl`, `DefaultCQTPlanImpl`, etc.
  - window.cpp: Implementation of default window generation helper functions.
  - filter_design.cpp (Example): Implementation of default filter design functions.
- **accelerate/**, **onemkl/**, **intelipp/**, **(your_backend)/**: Directories for specific backend implementations (e.g., in `OmniDSP::accelerate`, `OmniDSP::onemkl`, `OmniDSP::intelipp` namespaces), mirroring the structure of `default/`.
- **omnidsp.cpp**: Implementation of the main public `OmniDSP` class (factory, dispatch logic).
- **utils/**: Contains common utility functions (e.g., resampling factor calculation, filter design helpers) potentially used by multiple backends.

**2\. Overview: Backend Architecture**

OmniDSP uses a combination of the **Pimpl (Pointer to Implementation) idiom** and the **Strategy pattern** (with implementation inheritance) to support multiple backends:

- **OmniDSP (Public Class):** The user-facing class (`OmniDSP/include/OmniDSP/omnidsp.hpp`). Holds `pimpl_` (a `std::unique_ptr<OmniDSP::abstract::AbstractBackend>`). Uses templates for its public methods.
- **AbstractBackend (Backend Interface):** An abstract base class (`src/omnidsp/interface/backend.hpp`) in the `OmniDSP::abstract` namespace defining the virtual interface.
  - **Pure Virtual Functions (= 0):** Define methods that _every_ backend (including Default) absolutely must provide a unique implementation for (e.g., `get_backend()`).
  - **Virtual Functions (No \= 0):** Define methods where a reasonable default implementation is possible (e.g., one-off operations, plan factories, design functions). `DefaultBackend` provides these baseline implementations. Concrete backends can optionally override these for optimization.
- **DefaultBackend (Default Backend Class):** A concrete implementation (`src/omnidsp/default/`) in the `OmniDSP::default` namespace, derived from `abstract::AbstractBackend`. Provides baseline implementations (using standard C++) for all virtual methods defined in `AbstractBackend`. **Serves as the base class for other optimized backends.**
- **YourBackend (Concrete Backend Class):** The class you create (e.g., `CudaBackend`) in its own namespace (e.g., `OmniDSP::cuda`). **Should inherit from `OmniDSP::default::DefaultBackend`**. Overrides functions for optimization, relying on `DefaultBackend` for non-optimized operations.
- **\*PlanImpl (Plan Interfaces):** Abstract base classes (e.g., `abstract::FFTPlanImpl`) in `src/omnidsp/interface/backend.hpp` defining the interfaces for backend-specific plan implementations.
- **YourBackend\*PlanImpl (Concrete Plan Classes):** Classes implementing the `*PlanImpl` interfaces using your backend's API (e.g., `cuda::CudaFFTPlanImpl`). These are created by `YourBackend::create_*_plan_*` methods (which override the corresponding methods from `DefaultBackend`).
- **OmniDSP::create (Factory):** Static factory (`src/omnidsp/omnidsp.cpp`) instantiating the correct concrete `*Backend` class (e.g., `default::DefaultBackend`, `accelerate::AccelerateBackend`, `intelipp::IntelIPPBackend`, `onemkl::OneMKLBackend`, `your_backend::YourBackend`).

**3\. Implementing the Core Backend Class (YourBackend)**

- **Inheritance:** Your class (e.g., `namespace OmniDSP::your_backend { class YourBackend final : public OmniDSP::default::DefaultBackend { ... }; }`) **should inherit from `OmniDSP::default::DefaultBackend`**. This allows you to only override the methods you want to optimize.
- **Location:** Define in `src/omnidsp/your_backend/backend.hpp` and `.cpp`.
- **Override Virtual Functions:**
  - **`get_backend():`** You **must** override this pure virtual function to return your backend's unique `Backend` enum value.
  - **Focus on Optimizations:** Identify operations your backend accelerates better than the default implementation provided by `DefaultBackend`.
  - **Override Plan Factories (`create_*_plan_*`):** If your backend provides an optimized Plan implementation (e.g., `your_backend::YourFFTPlanImpl`), you **must** override the corresponding factory method (e.g., `create_fft_plan_c32`) from `DefaultBackend`. Your override should:
    - Attempt to create an instance of your backend-specific implementation (e.g., `std::make_unique<your_backend::YourFFTPlanImpl<C32>>(...)`). Catch any exceptions.
    - If creation fails, return `std::unexpected(Status::Failure)` or a more specific error code.
    - If creation succeeds, return the implementation pointer wrapped in the public Plan interface using the static factory: `return FFTPlan<C32>::create_from_impl(std::move(your_impl_ptr));`.
  - **Override Others (Optional):** Override one-off operations (`convolve_*`, `fft_*`), window generation, or filter design methods _only if_ your backend offers specific, optimized implementations superior to the default provided by `DefaultBackend`. For methods you don't override, the implementation from `DefaultBackend` will be used automatically via inheritance.
- **Use `override`:** Mark all overridden virtual functions with the `override` keyword.

**4\. Implementing Plan Classes (YourBackend\*PlanImpl)**

- **Inheritance:** Create classes (e.g., `your_backend::YourFFTPlanImpl`) inheriting from the corresponding base interface (`abstract::FFTPlanImpl<C32>`, etc.).
- **Location:** Define within your backend-specific directory (`src/omnidsp/your_backend/`).
- **Implementation:** Implement the constructor (setup backend resources, precompute data), destructor (cleanup backend resources), `execute`, `reset` (if stateful), and getter methods required by the `*PlanImpl` interface, using your backend's specific API calls. Ensure constructors throw appropriate exceptions on failure.

**5\. Example: Leveraging Inheritance and Overridden Plans**

- If `MyBackend` inherits `DefaultBackend` and only overrides `create_fft_plan_c32`, then:
  - A call to `myBackendInstance->create_fft_plan_c32(...)` will execute `MyBackend`'s overridden factory, returning a `MyFFTPlanImpl`.
  - A call to `myBackendInstance->create_convolution_plan_f32(...)` will execute `DefaultBackend`'s implementation (since `MyBackend` didn't override it).
  - If a complex plan implementation inherited from `DefaultBackend` (e.g., `default::DefaultCQTPlanImpl`) needs an FFT plan, it will call `owner->create_fft_plan_c32(...)`. Since the `owner` is actually `MyBackend`, this virtual call will correctly dispatch to `MyBackend::create_fft_plan_c32`, thus using the optimized FFT plan within the otherwise default CQT logic.

**6\. Integrating with the Factory (OmniDSP::create)**

- Add an enum value to `Backend` in `OmniDSP/include/OmniDSP/core_types.hpp`.
- Add a case `Backend::YourBackend:` block in the `switch` statement in `src/omnidsp/omnidsp.cpp`, guarded by an `#if OMNIDSP_ENABLED_BACKEND_YOURBACKEND` (or similar CMake variable defined in `omnidsp_config.h`), to instantiate `YourBackend`: `pimpl = abstract::create_your_backend();`.
- Conditionally include `your_backend/backend.hpp` in `src/omnidsp/omnidsp.cpp` using `#if`.
- Implement the factory function `std::unique_ptr<abstract::AbstractBackend> abstract::create_your_backend()` in `src/omnidsp/your_backend/backend.cpp` (outside the namespace) to return `std::make_unique<your_backend::YourBackend>()`.

**7\. Integrating with the Build System (CMake)**

- Add a CMake option (`OMNIDSP_ENABLE_YOURBACKEND`) in `cmake/project_options.cmake` (or `cmake/backend.cmake`).
- Create `cmake/backend/yourbackend.cmake`:
  - Check the `OMNIDSP_ENABLE_YOURBACKEND` option.
  - If enabled, check platform compatibility if necessary.
  - Use `find_package` to find your backend's external dependencies (libraries, headers).
  - If dependencies are found:
    - Set `OMNIDSP_ENABLED_YOURBACKEND` to `TRUE` (using `CACHE BOOL FORCE`).
    - Create an `INTERFACE IMPORTED GLOBAL` target named `OmniDSP::yourbackend`.
    - Set the `INTERFACE_INCLUDE_DIRECTORIES`, `INTERFACE_LINK_LIBRARIES`, `INTERFACE_COMPILE_DEFINITIONS`, etc., properties on `OmniDSP::yourbackend` based on the found dependencies.
    - Add any necessary compile definitions specifically for enabling your backend's code paths (e.g., `target_compile_definitions(OmniDSP::yourbackend INTERFACE OMNIDSP_USE_YOURBACKEND_IMPL=1)`).
  - If dependencies are not found or the option is disabled, ensure `OMNIDSP_ENABLED_YOURBACKEND` is set to `FALSE`.
- Update `cmake/backend.cmake` to unconditionally `include(cmake/backend/yourbackend.cmake)`.
- Update `cmake/target_definitions.cmake`:
  - Define a list variable `YOURBACKEND_SOURCES` containing all `.cpp` files for your backend implementation.
  - Conditionally add your backend's source files to the main `omnidsp` library target using a generator expression: `target_sources(omnidsp PRIVATE $<$<BOOL:${OMNIDSP_ENABLED_YOURBACKEND}>:${YOURBACKEND_SOURCES}>)`.
  - Conditionally link the main `omnidsp` target against your backend's interface target: `target_link_libraries(omnidsp PRIVATE $<$<BOOL:${OMNIDSP_ENABLED_YOURBACKEND}>:OmniDSP::yourbackend>)`.
- Update `cmake/omnidsp_config.h.in`:
  - Add `#cmakedefine OMNIDSP_ENABLED_BACKEND_YOURBACKEND @OMNIDSP_ENABLED_YOURBACKEND@`
  - Add `#ifndef OMNIDSP_ENABLED_BACKEND_YOURBACKEND` / `#define OMNIDSP_ENABLED_BACKEND_YOURBACKEND 0` / `#endif` block.

**8\. Adhering to API Conventions**

- Use `std::span<T>` for output parameters in functions like window generators and `Plan::execute`. Always check `output.size()` against required size.
- Use `double` for configuration parameters in Spec objects (sample rates, window shape params) for consistency. The backend implementation can cast to `float` internally if needed.
- Return `Status` or `OmniExpected<T>` from all fallible operations, including Plan factories and execute methods. Use the error checking macros (`OMNI_CHECK_*_STATUS_THROW`, `OMNI_CHECK_*_STATUS_RETURN`) provided in backend utility headers (e.g., `intelipp/utils.hpp`) or define your own.
- Ensure `*PlanImpl` constructors throw appropriate exceptions (`OmniException` with a relevant `Status` code, `std::invalid_argument`, `std::runtime_error`, `std::bad_alloc`) on setup failure. The corresponding `AbstractBackend::create_*_plan_*` method (whether in `DefaultBackend` or your override) should catch these and return `std::unexpected(Status::Failure)` or a more specific error code.
- Use the core type utilities (`Utils::IsComplex_v`, `Utils::GetRealType`, `Utils::GetComplexType`) from `core_types.hpp` for type checking and mapping.

By inheriting from `DefaultBackend` and selectively overriding methods, you can efficiently integrate new, partially or fully optimized backends.
