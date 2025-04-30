## OmniDSP Backend Developer Guide

This document provides a roadmap and guidelines for developers looking to integrate a new computation backend (e.g., for specific hardware like GPUs/FPGAs, or a different acceleration library) into the OmniDSP library. It outlines the necessary steps and architectural patterns involved.

Refer to the main Design Philosophy.md for the overarching architectural goals.

**1\. Overview: Directory Structure**

The core C++ source code (src/omnidsp/) is organized as follows:

- **interface/**: Defines the abstract backend contract and implements the public Plan class wrappers.
  - backend.hpp: Defines the AbstractBackend base class and the abstract \*PlanImpl interfaces. This is the core contract.
  - fft.cpp, cqt.cpp, convolution.cpp, resample.cpp, filter.cpp: Implementations of the public FFTPlan, CQTPlan, etc., classes. These act as wrappers, forwarding calls to the backend-specific \*PlanImpl via the Pimpl pattern.
- **default/**: Contains the default backend implementation using **standard C++**.
  - backend.hpp/.cpp: Defines and implements DefaultBackend. **Crucially, DefaultBackend inherits from AbstractBackend and provides baseline implementations for all virtual methods.**
  - fft.cpp, cqt.cpp, etc.: Implementations of DefaultFFTPlanImpl, DefaultCQTPlanImpl, etc.
  - window.cpp: Implementation of default window generation helper functions.
  - filter_design.cpp (Example): Implementation of default filter design functions.
- **accelerate/**, **onemkl/**, **(your_backend)/**: Directories for specific backend implementations, mirroring the structure of default/.
- **omnidsp.cpp**: Implementation of the main public OmniDSP class (factory, dispatch logic).

**2\. Overview: Backend Architecture**

OmniDSP uses a combination of the **Pimpl (Pointer to Implementation) idiom** and the **Strategy pattern** (with implementation inheritance) to support multiple backends:

- **OmniDSP (Public Class):** The user-facing class (OmniDSP/include/OmniDSP/omnidsp.hpp). Holds pimpl\_ (a std::unique_ptr\<backend::AbstractBackend\>). Uses templates for its public methods.
- **AbstractBackend (Backend Interface):** An abstract base class (src/omnidsp/interface/backend.hpp) defining the virtual interface.
  - **Pure Virtual Functions (= 0):** Define methods that _every_ backend (including Default) absolutely must provide a unique implementation for (e.g., get_backend()).
  - **Virtual Functions (No \= 0):** Define methods where a reasonable default implementation is possible (e.g., one-off operations, plan factories, design functions). DefaultBackend provides these baseline implementations. Concrete backends can optionally override these for optimization.
- **DefaultBackend (Default Backend Class):** A concrete implementation (src/omnidsp/default/) derived from AbstractBackend. Provides baseline implementations (using standard C++) for all virtual methods defined in AbstractBackend. **Serves as the base class for other optimized backends.**
- **YourBackend (Concrete Backend Class):** The class you create (e.g., CudaBackend). **Should inherit from DefaultBackend**. Overrides functions for optimization, relying on DefaultBackend for non-optimized operations.
- **\*PlanImpl (Plan Interfaces):** Abstract base classes (src/omnidsp/interface/backend.hpp) defining the interfaces for backend-specific plan implementations.
- **YourBackend\*PlanImpl (Concrete Plan Classes):** Classes implementing the \*PlanImpl interfaces using your backend's API (e.g., CudaFFTPlanImpl). These are created by YourBackend::create\_\*\_plan\_\* methods (which override the corresponding methods from DefaultBackend).
- **OmniDSP::create (Factory):** Static factory (src/omnidsp/omnidsp.cpp) instantiating the correct concrete \*Backend class (e.g., DefaultBackend, AccelerateBackend, YourBackend).

**3\. Implementing the Core Backend Class (YourBackend)**

- **Inheritance:** Your class (e.g., class CudaBackend final : public DefaultBackend) **should inherit from DefaultBackend**. This allows you to only override the methods you want to optimize.
- **Location:** Define in src/omnidsp/your_backend/backend.hpp and .cpp.
- **Override Virtual Functions:**
  - **get_backend():** You **must** override this pure virtual function to return your backend's unique Backend enum value.
  - **Focus on Optimizations:** Identify operations your backend accelerates better than the default implementation provided by DefaultBackend.
  - **Override Plan Factories (create\_\*\_plan\_\*):** If your backend provides an optimized Plan implementation (e.g., CudaFFTPlanImpl), you **must** override the corresponding factory method (e.g., create_fft_plan_c32) from DefaultBackend. Your override should:
    - Attempt to create an instance of your backend-specific implementation (e.g., std::make_unique\<CudaFFTPlanImpl\<C32\>\>(...)). Catch any exceptions.
    - If creation fails, return std::unexpected(Status::Failure) or a more specific error code.
    - If creation succeeds, return the implementation pointer wrapped in the public Plan interface using the static factory: return FFTPlan\<C32\>::create_from_impl(std::move(your_impl_ptr));.
  - **Override Others (Optional):** Override one-off operations (convolve\_\*, fft\_\*), window generation, or filter design methods _only if_ your backend offers specific, optimized implementations superior to the default provided by DefaultBackend. For methods you don't override, the implementation from DefaultBackend will be used automatically via inheritance.
- **Use override:** Mark all overridden virtual functions with the override keyword.

**4\. Implementing Plan Classes (YourBackend\*PlanImpl)**

- **Inheritance:** Create classes (e.g., CudaFFTPlanImpl) inheriting from the corresponding base interface (backend::FFTPlanImpl\<C32\>, etc.).
- **Location:** Define within your backend-specific directory (src/omnidsp/your_backend/).
- **Implementation:** Implement the constructor (setup backend resources, precompute data), destructor (cleanup backend resources), execute, reset (if stateful), and getter methods required by the \*PlanImpl interface, using your backend's specific API calls. Ensure constructors throw appropriate exceptions on failure.

**5\. Example: Leveraging Inheritance and Overridden Plans**

- If MyBackend inherits DefaultBackend and only overrides create_fft_plan_c32, then:
  - A call to myBackendInstance-\>create_fft_plan_c32(...) will execute MyBackend's overridden factory, returning a MyFFTPlanImpl.
  - A call to myBackendInstance-\>create_convolution_plan_f32(...) will execute DefaultBackend's implementation (since MyBackend didn't override it).
  - If a complex plan implementation inherited from DefaultBackend (e.g., DefaultCQTPlanImpl) needs an FFT plan, it will call owner-\>create_fft_plan_c32(...). Since the owner is actually MyBackend, this virtual call will correctly dispatch to MyBackend::create_fft_plan_c32, thus using the optimized FFT plan within the otherwise default CQT logic.

**6\. Integrating with the Factory (OmniDSP::create)**

- Add an enum value to Backend in OmniDSP/include/OmniDSP/core_types.hpp.
- Add a case Backend::YourBackend: block in the switch statement in src/omnidsp/omnidsp.cpp, guarded by an \#ifdef OMNIDSP_BACKEND_YOURBACKEND (or similar CMake variable), to instantiate YourBackend: pimpl \= std::make_unique\<backend::YourBackend\>();.
- Conditionally include your_backend/backend.hpp in src/omnidsp/omnidsp.cpp.

**7\. Integrating with the Build System (CMake)**

- Add a CMake option (OMNIDSP_ENABLE_YOURBACKEND) in cmake/project_options.cmake.
- Create cmake/depend/yourbackend_deps.cmake (or similar) to find your backend's external dependencies (libraries, headers) using find_package. Set variables indicating success/failure and necessary link/include information.
- Create cmake/backend/yourbackend.cmake:
  - Include cmake/depend/yourbackend_deps.cmake.
  - Check if dependencies were found.
  - If found, set a variable like OMNIDSP_HAS_YOURBACKEND to TRUE (using PARENT_SCOPE).
  - Define a list variable YOURBACKEND_SOURCES containing all .cpp files for your backend implementation (using PARENT_SCOPE).
  - Append necessary link libraries and include directories found by the dependency script to common variables like OMNIDSP_LINK_LIBS, OMNIDSP_INCLUDE_DIRS (using PARENT_SCOPE if needed).
- Update cmake/backend.cmake to conditionally include(cmake/backend/yourbackend.cmake) based on the OMNIDSP_ENABLE_YOURBACKEND option.
- Update cmake/target_definitions.cmake (or potentially cmake/backend.cmake):
  - Conditionally add your backend's source files to the main omnidsp library target using the YOURBACKEND_SOURCES list and a generator expression: target_sources(omnidsp PRIVATE $\<$\<BOOL:${OMNIDSP\_HAS\_YOURBACKEND}\>:${YOURBACKEND_SOURCES}\>).
  - Conditionally add the compile definition \-DOMNIDSP_BACKEND_YOURBACKEND=1 (or similar) based on OMNIDSP_HAS_YOURBACKEND: target_compile_definitions(omnidsp PRIVATE $\<$\<BOOL:${OMNIDSP_HAS_YOURBACKEND}\>:OMNIDSP_BACKEND_YOURBACKEND=1\>).

**8\. Adhering to API Conventions**

- Use std::span\<T\> for output parameters in functions like window generators and Plan::execute. Always check output.size() against required size.
- Use double for configuration parameters in Spec objects (sample rates, window shape params) for consistency. The backend implementation can cast to float internally if needed.
- Return Status or OmniExpected\<T\> from all fallible operations, including Plan factories and execute methods. Use the OMNI_PROPAGATE_ERROR macro for cleaner error checking within factory implementations.
- Ensure \*PlanImpl constructors throw appropriate exceptions (std::invalid_argument, std::runtime_error, std::bad_alloc) on setup failure. The corresponding AbstractBackend::create\_\*\_plan\_\* method (whether in DefaultBackend or your override) should catch these and return std::unexpected(Status::Failure) or a more specific error code.

By inheriting from DefaultBackend and selectively overriding methods, you can efficiently integrate new, partially or fully optimized backends.
