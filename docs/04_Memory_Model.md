# OmniDSP Memory Model

This document defines how OmniDSP handles memory, buffers, alignment, and complex numbers.

## 1. Data Types

### 1.1. Primitives
*   `f32` (Single Precision): The primary data type for Audio/DSP.
*   `f64` (Double Precision): Supported for scientific accuracy where needed.

### 1.2. Complex Numbers (`Complex<T>`)
We use the **Interleaved** format as the standard public API.
*   **Type:** `num_complex::Complex<f32>`
*   **Layout:** `[Re, Im, Re, Im, ...]` in contiguous memory.
*   **Rationale:** Standard in Rust, compatible with C++ `std::complex`.

#### Handling Split-Complex Backends (Accelerate/IPP)
Some hardware backends require **Split-Complex** (`[Re, Re...]`, `[Im, Im...]`).
*   **Responsibility:** The **Operator** (Plan/Processor) handles this conversion.
*   **Strategy:** 
    *   If the cost of conversion is negligible (e.g. small FFTs), the Operator allocates a temporary scratch buffer and converts on the fly.
    *   **Future Optimization:** We may introduce a `SignalBuffer` type that abstracts the layout, allowing the user to provide Split data directly if they prefer.

## 2. Buffer Management

### 2.1. Zero Allocation (Hot Path)
The `process()` and `execute()` methods MUST NOT allocate heap memory.
*   **Input:** Passed as `&[T]` (Immutable Slice).
*   **Output:** Passed as `&mut [T]` (Mutable Slice).
*   **Scratch Space:** If the algorithm needs temporary workspace, this memory is allocated **once** during the Plan/Processor creation and stored in the struct.

### 2.2. In-Place Execution
Backends should support In-Place execution where possible.
*   **API:** `process_inplace(&mut buffer)` or `process(io, io)`.
*   **Contract:** If `input` and `output` point to the same memory, the implementation must handle it correctly or return an error if strictly not supported.

### 2.3. Alignment
SIMD instructions (AVX, NEON) often require memory to be aligned (16-byte, 32-byte, or 64-byte boundaries).
*   **User Responsibility:** The user provides the slice. If the slice is unaligned, the backend *may* fail or fall back to a slower scalar loop.
*   **Helpers:** The library provides `omnidsp::alloc::aligned_vec(size)` to help users allocate SIMD-friendly buffers.

## 3. Thread Safety & Ownership
*   **Send + Sync:** All Plans and Processors implement `Send` and `Sync` where possible.
*   **Plans:** Are stateless and `Sync`. Can be shared across threads (Arc).
*   **Processors:** Are stateful. They are `Send` (can move between threads) but NOT `Sync` (cannot be used concurrently).
