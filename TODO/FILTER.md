# OmniDSP Filtering Module - Design Proposal (FILTER.md)

## 1. Introduction

This document outlines a proposed design for incorporating digital filtering capabilities into the OmniDSP library. The design aims to align with the library's core philosophy:

- Centralized API access via the `OmniDSP` class instance.
- Management of backend computation context within the `OmniDSP` instance.
- Use of stateful `Plan` objects (created by `OmniDSP` factories) for optimized, repeated operations.
- Use of `const` member functions on the `OmniDSP` instance for stateless calculations or parameter generation.
- Consistent error handling using `OmniExpected<T, Status>`.
- Encapsulation using the Pimpl idiom where appropriate.

Filtering involves two main stages: **Filter Design** (calculating coefficients) and **Filter Application** (applying the filter to a signal).

## 2. Filter Design API

Filter design involves calculating the filter coefficients based on user specifications. These calculations are typically standard mathematical procedures and may not require backend acceleration, but placing them within the `OmniDSP` class provides a consistent, centralized API.

**Proposal:** Add `const` member functions to the `OmniDSP` class (declarations in `omnidsp.h`) for designing common filter types.

**Supporting Types (to be defined, e.g., in `filter_types.h` or `core_types.h`):**

```c++
namespace OmniDSP {
    enum class FilterType { Butterworth, ChebyshevI, ChebyshevII, Elliptic /*, ... */ };
    enum class BandType { Lowpass, Highpass, Bandpass, Bandstop };
}
```

**Proposed `OmniDSP` Member Functions (Declarations in `omnidsp.h`):**

```c++
// Within OmniDSP class definition
/**
 * @brief Designs FIR filter coefficients using the window method.
 * @param order The filter order (number of taps - 1).
 * @param cutoff_freq The cutoff frequency (or center frequency for bandpass/stop). Normalized (0.0 to 1.0, where 1.0 is Nyquist).
 * @param window_spec Specification for the window function to use.
 * @param band_type Type of band (Lowpass default, Highpass, Bandpass, Bandstop).
 * @return Coefficients vector on success, or Status error.
 */
template<typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>> design_fir_filter(
    size_t order, RealT<T> cutoff_freq,
    const WindowSpec<T>& window_spec,
    BandType band_type = BandType::Lowpass /* Add cutoff2 for bandpass/stop */ ) const;

/**
 * @brief Designs IIR filter coefficients (biquad sections or direct form).
 * @param filter_type The type of IIR filter (e.g., Butterworth).
 * @param band_type The desired band type (Lowpass, Highpass, etc.).
 * @param order The filter order.
 * @param cutoff1 The primary cutoff frequency (or lower edge for bandpass/stop). Normalized (0.0 to 1.0).
 * @param cutoff2 Optional second cutoff frequency (upper edge for bandpass/stop).
 * @param passband_ripple_db Optional passband ripple in dB (e.g., for ChebyshevI, Elliptic).
 * @param stopband_atten_db Optional stopband attenuation in dB (e.g., for ChebyshevII, Elliptic).
 * @return Pair of vectors (b coefficients, a coefficients) on success, or Status error.
 * (Alternatively return vector of biquad sections).
 */
template<typename T>
[[nodiscard]] OmniExpected<std::pair<std::vector<RealT<T>>, std::vector<RealT<T>>>> design_iir_filter(
    FilterType filter_type, BandType band_type, size_t order,
    RealT<T> cutoff1, std::optional<RealT<T>> cutoff2 = std::nullopt,
    std::optional<RealT<T>> passband_ripple_db = std::nullopt,
    std::optional<RealT<T>> stopband_atten_db = std::nullopt) const;
```

## 3. Filter Application API

Applying the designed filter to a signal.

### 3.1. FIR Filter Application

**Mechanism:** Applying an FIR filter is equivalent to convolution with the filter coefficients (taps).

**Proposal:** Reuse the existing Convolution module.

1.  **Design:** User calls `dsp.design_fir_filter(...)` to obtain the FIR coefficients (`std::vector<RealT<T>> fir_coeffs`).
2.  **Apply:** User chooses one of the following:
    - **Direct (Stateless):** Call `dsp.convolve1d(signal, fir_coeffs, ConvolutionMode::Same)` using the `OmniDSP` member function. Suitable for one-off filtering.
    - **Optimized (Stateful Plan):** Call `dsp.create_convolution_plan(fir_coeffs, ConvolutionMode::Same)` (requires adding this factory to `OmniDSP`) to get a `ConvolutionPlan`. Then call `plan->execute(signal_chunk, output_chunk)` repeatedly. Ideal for filtering long signals or applying the same filter many times.

**Conclusion:** No specific FIR filtering API is needed beyond the design function and the existing/planned convolution tools.

### 3.2. IIR Filter Application

**Mechanism:** Applying an IIR filter involves a recursive difference equation, requiring internal state (delay lines for previous inputs and outputs) to be maintained between calls or blocks.

**Proposal:** Introduce a stateful `FilterPlan` (or `IIRFilterPlan`) class.

1.  **Interface (`filter.h`):**
    - Define `FilterPlan<T>` class (template on `RealT<F>`).
    - Use Pimpl idiom (`FilterPlanImpl` forward-declared in `OmniDSP::backend`).
    - Make constructor private, add `friend class OmniDSP;`.
    - Non-copyable, movable.
    - Declare public methods:
      - `~FilterPlan()`
      - `FilterPlan(FilterPlan&&) noexcept;`
      - `FilterPlan& operator=(FilterPlan&&) noexcept;`
      - `[[nodiscard]] Status execute(std::span<const T> input, std::span<T> output) const;` (Applies filter, updates internal state).
      - `void reset();` (Resets internal filter state/delay lines).
      - `std::vector<T> get_b_coeffs() const;`
      - `std::vector<T> get_a_coeffs() const;`
      - `size_t get_order() const;`
2.  **Creation (`omnidsp.h`):**

    - Add factory method declaration to `OmniDSP` class:

      ```c++
      /** @brief Creates a plan for applying a stateful IIR filter. */
      template<typename T>
      [[nodiscard]] OmniExpected<std::unique_ptr<FilterPlan<T>>> create_iir_filter_plan(
          const std::vector<RealT<T>>& b_coeffs, // Numerator
          const std::vector<RealT<T>>& a_coeffs  // Denominator (a[0] usually 1)
      ) const;
      // Could also take vector<BiquadSection> if using SOS format.
      ```

3.  **Usage:**
    1.  Design IIR filter: `auto coeffs = dsp.design_iir_filter(...).value();`
    2.  Create Plan: `auto plan = dsp.create_iir_filter_plan(coeffs.first, coeffs.second).value();`
    3.  Execute: `plan->execute(input_chunk, output_chunk);` (Can be called repeatedly for streaming data).
    4.  Reset state if needed: `plan->reset();`

## 4. Integration Summary

- **Filter Design:** Provided by `const` member functions on the `OmniDSP` instance (declared in `omnidsp.h`). Supporting enums defined in `filter_types.h` (or similar).
- **FIR Application:** Uses `OmniDSP::convolve1d` or `ConvolutionPlan` (from `convolution.h`).
- **IIR Application:** Uses `FilterPlan` (defined in `filter.h`, created via `OmniDSP::create_iir_filter_plan`).
- **`filter.h`:** Primarily defines supporting enums (`FilterType`, `BandType`) and the `FilterPlan` class interface for IIR filtering.

This approach integrates filtering into the existing OmniDSP structure, leverages existing modules (convolution), provides optimized paths (Plans), and maintains API consistency.

## 5. Future Considerations

- Support for different filter design methods (e.g., frequency sampling for FIR).
- Filter analysis functions (e.g., frequency response, phase response, group delay).
- Support for Second-Order Sections
