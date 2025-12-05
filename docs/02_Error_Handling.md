# OmniDSP Error Handling Standards

This document defines the standard error types and handling strategies for the OmniDSP library. The goal is to provide precise, actionable feedback to the user while maintaining the zero-cost philosophy of the core loop.

## 1. Philosophy
1.  **Fail Early:** Validate inputs during the **Spec** creation or **Plan/Processor** factory phase.
2.  **No Panics:** The library must never panic across the FFI boundary. All errors must be returned as `Result`.
3.  **Zero-Cost Execution:** The hot loops (`process()`, `execute()`) should ideally *not* return Results if possible (to avoid branch prediction checks), OR return a lightweight status that is easy for the CPU to ignore.
    *   *Current Decision:* `process()` methods WILL panic on debug builds if buffer sizes mismatch, but we may rely on strict contract checking in the Factory phase to avoid runtime checks in the loop.

## 2. The `OmniError` Enum

All fallible operations return `Result<T, OmniError>`.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum OmniError {
    /// The operation parameters are invalid (e.g. length 0, mismatched dimensions).
    InvalidSpec(String),
    
    /// The requested backend is not available on this system.
    BackendUnavailable(String),
    
    /// The backend failed to initialize (e.g. IPP returned an error code).
    BackendInitializationFailed { backend: String, code: i32, message: String },
    
    /// Memory allocation failed.
    AllocationFailed,
    
    /// Buffer size mismatch during execution (Input len != Output len).
    BufferSizeMismatch { expected: usize, actual: usize },
    
    /// An internal logic error (bug in the library).
    InternalError(String),
}
```

## 3. Error Propagation Strategy

### 3.1. Phase 1: Spec Creation
*   **Action:** User creates a `DftSpec`.
*   **Validation:** `DftSpec::new()` performs logic checks.
*   **Returns:** `Result<DftSpec, OmniError::InvalidSpec>`.

### 3.2. Phase 2: Factory (Context/Backend)
*   **Action:** User calls `backend.create_dft(spec)`.
*   **Validation:** 
    *   Is the backend loaded? -> `OmniError::BackendUnavailable`
    *   Did the FFI call succeed? -> `OmniError::BackendInitializationFailed`
    *   Did we run out of RAM? -> `OmniError::AllocationFailed`
*   **Returns:** `Result<Box<dyn DftPlan>, OmniError>`.

### 3.3. Phase 3: Execution (Hot Loop)
*   **Action:** User calls `plan.process(input, output)`.
*   **Validation:**
    *   Are buffers the right size? 
    *   *Debug Mode:* Panic or Return Error.
    *   *Release Mode:* We strictly enforce this check. The cost is negligible (2 integer comparisons).
*   **Returns:** `Result<(), OmniError::BufferSizeMismatch>`.

## 4. FFI Mapping (For C/Python)
When exposing this to C or Python, `OmniError` is mapped to:

*   **C API:** An integer status code (`OMNI_STATUS_OK = 0`, `OMNI_STATUS_INVALID_SPEC = -1`, etc.).
*   **Python:** Native Exceptions (`ValueError` for InvalidSpec, `RuntimeError` for BackendFailure).
