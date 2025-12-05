# OmniDSP Dispatch Strategy

This document details the mechanism for selecting and routing DSP operations to the appropriate backend implementation.

## 1. The "Operation Category" Model
Not all backends are equal. Intel IPP is great for FFTs, but might not have the specific IIR filter topology we want. The OmniDSP library allows **Granular Dispatch**.

We classify operations into **Categories**:

1.  **DFT:** Discrete Fourier Transforms (Forward/Inverse, Real/Complex).
2.  **Filtering:** FIR, IIR.
3.  **Windowing:** Generation of window coefficients.
4.  **VectorMath:** Element-wise operations (add, mul, dot-product).
5.  **Resampling:** Sample rate conversion.

## 2. The `Config` Object
The user directs the dispatch logic via the `Config` struct.

```rust
pub struct Config {
    // The global default (usually "Best Available")
    pub default_backend_preference: BackendType,

    // Granular overrides
    pub dft_preference: Option<BackendType>,
    pub filter_preference: Option<BackendType>,
    
    // Hardware constraints
    pub allow_gpu: bool,
    pub thread_limit: usize,
}
```

## 3. The Dispatch Algorithm
When `DspBackend::new(config)` is called, it builds a **Composite Backend**.

### 3.1. Priority Resolution
For each category (e.g., DFT), the system resolves the active implementation using this logic:

1.  **User Override:** Is `config.dft_preference` set?
    *   YES: Attempt to load that backend. If unavailable -> Error.
    *   NO: Proceed to step 2.
2.  **Global Preference:** Look at `config.default_backend_preference`.
    *   Attempt to load.
3.  **System Default:** If preference is `Auto` or failed:
    *   **MacOS:** Try Accelerate.
    *   **Intel:** Try IPP/MKL.
    *   **Fallback:** Use Omni (Rust Pure Implementation).

### 3.2. The Composite Backend (`OmniBackend`)
The `OmniBackend` acts as the **Manager**. It holds specific **Implementations** for each capability.

```rust
pub struct OmniBackend {
    // These fields are populated (lazily or eagerly) based on Config
    dft: Box<dyn Dft>,     // Points to IppDft or OmniDft
    fir: Box<dyn Fir>,     // Points to OmniFir
    // ...
}
```

When the user calls `backend.create_dft()`, it simply delegates:
`self.dft.create_plan(...)`.

This ensures **zero runtime dispatch overhead** during the factory call. The decision was made once, at Backend creation.

## 4. Runtime "Upgrades"
While the `OmniBackend` is immutable, users can create a *new* backend if they need to switch strategies mid-execution.

*   *Example:* An app starts in "Low Power Mode" (Omni Backend). User toggles "High Performance". App creates new `OmniBackend` (preferring IPP) and regenerates plans.
