# OmniDSP Concepts & Terminology

This document defines the core vocabulary and architectural components of the OmniDSP library. Adherence to these definitions ensures consistency across all language bindings (Rust, Python, C++).

## 1. The Core Architecture
The library is built on a modular "Manager-Provider" architecture. The lifecycle is: **Config → Backend → Provider → Spec → Operator**.

### 1.1. Config (`OmniConfig`)
*   **Definition:** Pure data describing the *desired* environment.
*   **Responsibilities:** User preferences, hardware constraints, and strategy overrides.
*   **Lifecycle:** Transient. Used solely to build the `Backend`.

### 1.2. Backend (`OmniBackend`)
*   **Definition:** The **Manager** (Concrete Struct).
*   **Role:** A lightweight **Router**. It holds the `Config` and the active **Providers**.
*   **Responsibilities:**
    *   Lazy Initialization of Providers.
    *   Routing requests (e.g., delegating `create_dft` to `self.dft_provider`).
*   **Lifecycle:** Created from a `Config`.

### 1.3. Implementations (`OmniDft`, `IppDft`)
*   **Definition:** Concrete structs that implement the Capability Traits (e.g., `Dft`, `Fir`).
*   **Role:** The "Factories". They hold global state (library handles) and know how to manufacture Plans.
*   **Pattern:** **Composition**. The Backend holds a collection of these trait objects (`Box<dyn Dft>`).
*   **Naming:** `OmniDft` (Pure Rust), `AccelerateDft` (MacOS), `IppDft` (Intel).
*   **Benefit:** Separation of concerns. `AccelerateDft` only implements what it supports; it doesn't need to stub out missing features.

### 1.4. Spec (`Specification`)
*   **Definition:** Backend-agnostic data describing *what* operation to perform.
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
let context = OmniContext::new(config);
let backend = context.backend();

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
