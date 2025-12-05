# OmniDSP Concepts & Terminology

This document defines the core vocabulary and architectural components of the OmniDSP library. Adherence to these definitions ensures consistency across all language bindings (Rust, Python, C++).

## 1. The Core Architecture
The library is built on a modular "Manager-Provider" architecture. The lifecycle is: **Config → Backend → Implementation → Spec → Operator**.

### 1.1. Config (`Config`)
*   **Definition:** Pure data describing the *desired* environment and performance tuning.
*   **Role:** **Tuning & Strategy**.
*   **Scope:** Settings that affect *performance* or *resource usage* but **MUST NOT** change the mathematical result.
    *   *Examples:* "Prefer IPP", "Thread Limit = 4", "Use AVX-512 hint".
*   **Lifecycle:** Transient. Used solely to build the `Backend`.

### 1.2. Backend (`Backend`)
*   **Definition:** The **Manager** (Concrete Struct).
*   **Role:** A lightweight **Router**. It holds the `Config` and the active **Implementations**.
*   **Responsibilities:**
    *   Lazy Initialization of Implementations.
    *   Routing requests (e.g., delegating `create_dft` to `self.dft`).
*   **Lifecycle:** Created from a `Config`.

### 1.3. Implementations (Modules)
*   **Definition:** Concrete structs that implement the Capability Traits (e.g., `Dft`, `Fir`).
*   **Role:** The "Factories". They hold global state (library handles) and know how to manufacture Plans.
*   **Pattern:** **Composition**. The Backend holds a collection of these trait objects (`Box<dyn Dft>`).
*   **Naming:** We use **Modules** instead of prefixes.
    *   `backend::omni::Dft` (Pure Rust)
    *   `backend::accelerate::Dft` (MacOS)
    *   `backend::ipp::Dft` (Intel)
*   **Benefit:** Separation of concerns and cleaner namespaces.

### 1.4. Spec (`Specification`)
*   **Definition:** Backend-agnostic data describing the **Mathematical Operation**.
*   **Role:** **Behavior Definition**.
*   **Scope:** Parameters that determine the numerical output.
    *   *Examples:* "FFT Length = 1024", "Normalization = Unitary", "Filter Cutoff = 0.5".
*   **Contract:** All backends MUST produce the same numerical output (within floating-point tolerance) for a given Spec.
*   **Role:** Passed to the Implementation (via the Backend) to create an Operator.

### 1.5. Execution Objects (The Workers)
The objects created by an Implementation using a Spec.

#### A. Plan (`StatelessOperator`)
*   **Definition:** Executes a **stateless** transformation (FFT, Windowing).
*   **Contract:** Immutable, Reentrant. Created by `Dft::create_plan(spec)`.
*   **Why separate from the Implementation?**
    *   **The Implementation (`OmniDft`)** is the "Bakery" (holds the ovens/configuration).
    *   **The Plan (`DftPlan`)** is the "Cake" (pre-calculated weights, specific size). You bake it once, eat (execute) it many times.

#### B. Processor (`StatefulOperator`)
*   **Definition:** Processes a stream with **internal state** (IIR, Resampling).
*   **Contract:** Mutable, Not Thread-Safe (per stream).

---

## 2. Implementation Details

### 2.1. The Factory Pattern
Users never instantiate a Plan or Processor directly. They ask the Backend.

```rust
// Conceptual Rust Signature
let backend = Backend::new(config);

// 1. Create Spec
let spec = DftSpec::new(1024, Direction::Forward);

// 2. Create Plan (Backend does the heavy lifting here)
let plan: Box<dyn DftPlan> = backend.create_dft(spec)?;

// 3. Execute
plan.process(input, output);
```

### 2.2. Resource Acquisition Is Initialization (RAII)
*   **Creation:** `backend.create_...()` allocates all necessary resources (memory, handles).
*   **Execution:** `process()` performs **zero allocations**. It strictly operates on provided buffers.
*   **Destruction:** When the Plan/Processor goes out of scope, all resources are freed automatically.

### 2.3. Data Types
*   **Real:** `f32` (default), `f64`.
*   **Complex:** `Complex<f32>` (Interleaved: Real, Imag).
*   **Buffers:** Input buffers are `&[T]`, Output buffers are `&mut [T]`.

### 2.4. Direct Access (Bypassing the Manager)
While `OmniBackend` is the primary entry point for portable code, the library exposes the concrete Implementations publicly. A user can instantiate `IppDft` directly.

*   **Use Case:** "Power Users" requiring backend-specific features (e.g., an IPP-specific flag, a CUDA stream ID) that are not representable in the generic `Spec`.
*   **Trade-off:** This code becomes **non-portable**. Using `IppDft` directly means the code will fail to compile on ARM/MacOS.
*   **Philosophy:** The "Omni" spirit provides a unified baseline, but we do not prevent users from reaching the metal if they accept the portability cost.
