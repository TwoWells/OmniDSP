# Architecture Decision Record: Why Rust?

## 1. Context
The OmniDSP project aims to create a modular, high-performance, cross-platform signal processing library. A previous iteration was built in C++, utilizing CMake for build management and Abstract Base Classes for modularity.

**Pain Points of the C++ Iteration:**
*   **"Revision Hell":** Refactoring was high-friction due to header/source separation and fragile include hierarchies.
*   **Build Complexity:** CMake configuration grew exponentially with each added backend (IPP, Accelerate, OneAPI), making cross-platform maintenance difficult.
*   **Safety:** Manual memory management and pointer arithmetic introduced risks of buffer overflows and segfaults.

## 2. The Decision: Rust
We have selected **Rust** as the implementation language for OmniDSP.

## 3. Rationale

### 3.1. Build System & Dependency Management (Cargo)
*   **Problem:** CMake scripts are fragile, imperative, and non-standardized. Managing dependencies (like GoogleTest or internal modules) is manual.
*   **Solution:** Cargo provides a declarative, standardized build system. It handles dependencies, compilation flags, testing, and documentation generation out of the box.
*   **Impact:** Drastically reduces the "infrastructure" code required to maintain the project.

### 3.2. Parity with Vendor Libraries (FFI)
*   **Observation:** Key vendor libraries (Apple Accelerate, Intel IPP) expose **C APIs**.
*   **Parity:** C++ interacts with these via `extern "C"`. Rust interacts with these via `extern "C"`.
*   **Benefit:** Rust is on equal footing with C++ for invoking these low-level primitives. There is no "bridging overhead" compared to C++.

### 3.3. Modularity (Traits vs. Inheritance)
*   **Problem:** C++ inheritance hierarchies (Virtual Base Classes) are rigid. Adding a feature often requires modifying the base class, forcing recompilation of all children.
*   **Solution:** Rust Traits + Composition.
    *   **Traits** define behavior (interfaces) without mandating memory layout.
    *   **Composition** (Hardware Backends *owning* an Omni Backend) avoids the "Fragile Base Class" problem.
    *   **Result:** A flatter, more flexible architecture.

### 3.4. Safety & Correctness
*   **Problem:** DSP involves heavy array manipulation. Off-by-one errors in C++ lead to silent corruption or segfaults.
*   **Solution:** Rust's Borrow Checker and Bounds Checking (which can be elided in hot loops via iterators) ensure memory safety at compile time.
*   **Concurrency:** Rust's `Send/Sync` traits prevent data races at compile time, enabling "Fearless Concurrency" for parallel DSP algorithms.

### 3.5. Interoperability
*   **Goal:** The library must be callable from Python, C++, and other languages.
*   **Capability:** Rust compiles to a standard C-ABI shared library (`.so`/`.dll`/`.dylib`).
*   **Python:** The `PyO3` ecosystem allows generating safe, high-performance Python bindings more easily than C++ equivalents (pybind11/Swig).

### 3.6. Parity of Consumption
*   **Observation:** C++ applications currently consume vendor libraries (IPP, Accelerate) via C headers and shared object linking.
*   **Strategy:** By exposing a C-ABI, `libomnidsp` presents itself to a C++ consumer exactly like these existing dependencies.
*   **Benefit:** C++ users do not need to "learn Rust" or use a Rust toolchain. They simply include `omnidsp.h` and link `libomnidsp.so`, preserving their existing build workflows.

## 4. Discarded Alternatives

*   **C++20:** Still suffers from the CMake/Build complexity and header-file legacy.
*   **Julia:** Excellent for analysis, but the runtime weight and JIT latency make it unsuitable for embedding as a low-level system library.
*   **Zig:** Promising "Better C", but the ecosystem and language stability are not yet mature enough for a foundational library.
