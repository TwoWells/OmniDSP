## **OmniDSP Backend Developer Guide**

This document provides a roadmap and guidelines for developers looking to integrate a new computation backend (e.g., for specific hardware like GPUs/FPGAs, or a different acceleration library) into the OmniDSP library. It outlines the necessary steps and architectural patterns involved.

Refer to the main docs/design_philosophy.md for the overarching architectural goals.

**1\. Overview: Directory Structure**

The core C++ source code (src/omnidsp/) is organized as follows:

- **interface/**: Defines the abstract backend contract and implements the public Plan class wrappers.
  - backend.h: Defines the AbstractBackend base class and the abstract \*PlanImpl interfaces. This is the core contract.
  - fft.cpp, cqt.cpp, convolution.cpp, resample.cpp, filter.cpp: Implementations of the public FFTPlan, CQTPlan, etc., classes. These act as wrappers, forwarding calls to the backend-specific \*PlanImpl via the Pimpl pattern.
- **default/**: Contains the default backend implementation using C++/Highway.
  - backend.h/.cpp: Defines and implements DefaultBackend.
  - fft.cpp, cqt.cpp, etc.: Implementations of DefaultFFTPlanImpl, DefaultCQTPlanImpl, etc.
  - window.cpp: Implementation of default window generation functions.
  - filter_design.cpp (Example): Implementation of default filter design functions.
- **accelerate/**, **onemkl/**, **(your_backend)/**: Directories for specific backend implementations, mirroring the structure of default/.
- **omnidsp.cpp**: Implementation of the main public OmniDSP class (factory, dispatch logic).

**2\. Overview: Backend Architecture**

OmniDSP uses a combination of the **Pimpl (Pointer to Implementation) idiom** and the **Strategy pattern** to support multiple backends:

- **OmniDSP (Public Class):** The user-facing class (OmniDSP/include/OmniDSP/omnidsp.h). Holds pimpl\_ (a std::unique_ptr\<backend::AbstractBackend\>). Uses templates for its public methods.
- **AbstractBackend (Backend Interface):** An abstract base class (src/omnidsp/interface/backend.h) defining the virtual interface.
  - **Pure Virtual Functions (= 0):** Define the **core contract** (Plan Factories, Filter Design, Window Generation, Internal Helpers, get_backend). _Must_ be implemented by DefaultBackend and potentially overridden by specific backends.
  - **Virtual Functions (No \= 0):** One-off operations (convolve\_\*, fft\_\*). Have default implementations in DefaultBackend. Specific backends override only if a more optimized direct implementation exists.
- **DefaultBackend (Default Backend Class):** The baseline implementation (src/omnidsp/default/). **Implements all virtual functions** from AbstractBackend. **New backends should inherit from this class.**
- **YourBackend (Concrete Backend Class):** The class you create (e.g., CudaBackend). **Must inherit from DefaultBackend**. Overrides functions for optimization.
- **\*PlanImpl (Plan Interfaces):** Abstract base classes (src/omnidsp/interface/backend.h) defining plan interfaces.
- **YourBackend\*PlanImpl (Concrete Plan Classes):** Classes implementing the \*PlanImpl interfaces using your backend's API (e.g., CudaFFTPlanImpl).
- **OmniDSP::create (Factory):** Static factory (src/omnidsp/omnidsp.cpp) instantiating the correct \*Backend class.

**3\. Implementing the Core Backend Class (YourBackend)**

- **Inheritance:** Your class (e.g., class CudaBackend final : public DefaultBackend) **must inherit from DefaultBackend**.
- **Location:** Define in src/omnidsp/your_backend/backend.h and .cpp.
- **Override Virtual Functions:**
  - **Focus on Optimizations:** Identify operations your backend accelerates better than the default.
  - **Override Plan Factories:** **Must** override relevant create\_\*\_plan\_\* functions to return instances of your YourBackend\*PlanImpl (wrapped in the public Plan unique_ptr). This is the primary way to inject optimized logic.
  - **Override Others (Optional):** Override window generation, filter design, internal helpers, or one-off operations _only if_ your backend offers specific, optimized implementations superior to the default.
  - Use the override keyword.
- **Implement get_backend():** Override to return your backend's unique Backend enum value.

**4\. Implementing Plan Classes (YourBackend\*PlanImpl)**

- **Inheritance:** Create classes (e.g., CudaFFTPlanImpl) inheriting from the corresponding base interface (backend::FFTPlanImpl\<C32\>, etc.).
- **Location:** Define within your backend-specific directory (src/omnidsp/your_backend/).
- **Implementation:** Implement constructor (setup), destructor (cleanup), execute, reset (if stateful), and getters using your backend's API.

**5\. Example: Leveraging Overridden Plans**

If MyBackend inherits DefaultBackend and only overrides create_fft_plan_c32, calls to create_cqt_plan\<F32\> on a MyBackend instance will correctly use the overridden FFT factory within the default CQT implementation logic (inherited from DefaultBackend).

**6\. Integrating with the Factory (OmniDSP::create)**

- Add an enum value to Backend in OmniDSP/include/OmniDSP/core_types.h.
- Add a case Backend::YourBackend: block in the switch statement in src/omnidsp/omnidsp.cpp, guarded by an \#ifdef OMNIDSP_USE_YOURBACKEND, to instantiate YourBackend.
- Conditionally include your_backend/backend.h in src/omnidsp/omnidsp.cpp.

**7\. Integrating with the Build System (CMake)**

- Add a CMake option (OMNIDSP_ENABLE_YOURBACKEND) in cmake/project_options.cmake.
- Create cmake/backend/yourbackend.cmake to find dependencies and set availability/link variables.
- Update cmake/backend.cmake to include your module conditionally.
- Update cmake/target_definitions.cmake to:
  - Conditionally add your backend's source files (src/omnidsp/your_backend/\*.cpp) to the omnidsp library target.
  - Conditionally add include directories and link libraries.
  - Conditionally add the \-DOMNIDSP_USE_YOURBACKEND compile definition.

**8\. Adhering to API Conventions**

- Use std::span\<T\> for output parameters in functions like window generators and execute. Check output.size().
- Use double for configuration parameters (sample rates, window shape params) for consistency.
- Return Status or OmniExpected\<T\> appropriately.

By inheriting from DefaultBackend and focusing overrides on plan factories and implementations, you can efficiently integrate new backends.
