// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `omnidsp-wasm` — the browser CQT visualiser engine (DEMO-00).
//!
//! A thin [`wasm-bindgen`] binding over the OmniDSP `RustBackend` multirate
//! Constant-Q Transform. It builds the CQT **once** at construction time (at the
//! real audio-context sample rate handed in from JavaScript) and then exposes a
//! per-frame [`process`](CqtEngine::process) entry point that performs **no
//! allocation and no copy across the wasm boundary**:
//!
//! 1. JS writes one PCM hop directly into wasm linear memory at the address
//!    returned by [`input_ptr`](CqtEngine::input_ptr) (capacity
//!    [`input_len`](CqtEngine::input_len) `f32` samples — the CQT's FFT length).
//! 2. JS calls [`process`](CqtEngine::process); the engine runs the
//!    octave-recursive multirate CQT over that buffer and writes the
//!    magnitude spectrum into an internal output buffer.
//! 3. JS reads [`output_len`](CqtEngine::output_len) `f32` magnitudes back from
//!    [`output_ptr`](CqtEngine::output_ptr) via a `Float32Array` view over
//!    `WebAssembly.Memory.buffer` — again with no copy.
//!
//! The transform runs on the pure-Rust floor (RustFFT / realfft + scalar
//! auto-vectorised ops); built with `RUSTFLAGS="-C target-feature=+simd128"`
//! the inner loops vectorise and the multirate decomposition fits a real-time
//! `requestAnimationFrame` budget. No vendor acceleration is involved — the
//! floor itself is the product.
//!
//! See `README.md` for the `wasm-pack build --target web` packaging step and a
//! minimal JS usage sketch.

use omnidsp::OmniDSP;
use omnidsp::RustBackend;
use omnidsp::create::CreatePlan;
use omnidsp_core::design::cqt::{self, CqtSpec};
use omnidsp_core::types::Window;
use wasm_bindgen::prelude::wasm_bindgen;

/// Upper edge of the demo CQT range (must stay below Nyquist).
const MAX_FREQ: f64 = 16_000.0;

/// Bins per octave for the visualiser (12 = one bin per semitone).
const BINS_PER_OCTAVE: u32 = 12;

/// The concrete `RustBackend` multirate CQT plan over `f32`.
///
/// Spelled via the backend's public `CreatePlan` associated type so the wasm
/// crate never has to name the routed decimator sub-plan by hand; it resolves
/// to `OmniCqtPlan` over the floor's r2c plan, `ScalarVecOps`, and the
/// `OmniResample(1, 2)` decimator.
type RustCqtPlan = <RustBackend as CreatePlan<CqtSpec<f32>>>::Plan;

/// Real-time Constant-Q Transform engine for the browser visualiser.
///
/// Constructed once from the audio context's sample rate; every audio hop is
/// processed through [`process`](Self::process) with no per-frame allocation.
#[wasm_bindgen]
pub struct CqtEngine {
    plan: RustCqtPlan,
    /// PCM input scratch (length = FFT length); JS writes here directly.
    input: Vec<f32>,
    /// Magnitude output scratch (length = bin count); JS reads here directly.
    output: Vec<f32>,
    /// Per-bin centre frequencies in Hz (low → high), for axis labelling.
    frequencies: Vec<f32>,
}

#[wasm_bindgen]
impl CqtEngine {
    /// Build the CQT engine for a given audio-context sample rate and low-edge
    /// frequency.
    ///
    /// The `min_freq … 16 kHz` spec is built at **`sample_rate`** — the actual
    /// rate JS reports (commonly 48 kHz, sometimes 44.1 kHz). Do not hardcode a
    /// rate here; feeding a 48 kHz context a 44.1 kHz spec mislabels every bin.
    ///
    /// `min_freq` sets the analysis-window length, and hence the visualiser's
    /// latency: the CQT FFT length is sized to resolve `min_freq`, so a lower
    /// edge means a longer window (20 Hz ≈ 1.4 s at 48 kHz). Raising it (e.g.
    /// 55 Hz ≈ 0.34 s) shortens the window and cuts latency at the cost of
    /// low-bass coverage. The kernels are causally anchored at the window start
    /// (`cqt` batch-v1 convention), so latency tracks the window length.
    ///
    /// # Errors
    ///
    /// Returns a JS string error if `min_freq` / `sample_rate` cannot admit the
    /// `min_freq … 16 kHz` range (max frequency must stay below Nyquist) or if
    /// plan construction otherwise fails.
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32, min_freq: f32) -> Result<CqtEngine, String> {
        let spec: CqtSpec<f32> = cqt::design(
            f64::from(sample_rate),
            f64::from(min_freq),
            MAX_FREQ,
            BINS_PER_OCTAVE,
            &Window::<f32>::Hann,
        )
        .map_err(|e| {
            format!("CQT design failed at {sample_rate} Hz (min {min_freq} Hz): {e}")
        })?;

        let dsp = OmniDSP::rust();
        let plan: RustCqtPlan = dsp
            .cqt(&spec)
            .map_err(|e| format!("CQT plan creation failed: {e}"))?;

        let input = vec![0.0_f32; plan.fft_length()];
        let output = vec![0.0_f32; plan.num_bins()];
        let frequencies: Vec<f32> = plan
            .bin_frequencies()
            .iter()
            .map(|&f| f as f32)
            .collect();

        Ok(Self {
            plan,
            input,
            output,
            frequencies,
        })
    }

    /// Compute the CQT magnitude spectrum of the current input buffer.
    ///
    /// Reads the [`input_len`](Self::input_len) samples JS has written at
    /// [`input_ptr`](Self::input_ptr) and writes [`output_len`](Self::output_len)
    /// magnitudes to the buffer at [`output_ptr`](Self::output_ptr). No
    /// allocation, no boundary copy. Call once per hop, then read the output
    /// view in JS.
    ///
    /// # Errors
    ///
    /// Returns a JS string error if the transform fails (buffer lengths are
    /// engine-owned and always correct, so this is effectively infallible in
    /// normal use).
    pub fn process(&mut self) -> Result<(), String> {
        self.plan
            .process_magnitude(&self.input, &mut self.output)
            .map_err(|e| format!("CQT process failed: {e}"))
    }

    /// Pointer to the PCM input buffer in wasm linear memory.
    ///
    /// JS builds a `Float32Array(memory.buffer, engine.input_ptr(),
    /// engine.input_len())` view once and writes each hop into it before calling
    /// [`process`](Self::process). The view must be rebuilt if wasm memory grows.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn input_ptr(&self) -> *const f32 {
        self.input.as_ptr()
    }

    /// Length (in `f32` samples) of the input buffer — the CQT FFT length.
    ///
    /// This is the exact hop size JS must supply per frame.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    /// Pointer to the magnitude output buffer in wasm linear memory.
    ///
    /// JS builds a `Float32Array(memory.buffer, engine.output_ptr(),
    /// engine.output_len())` view and reads it after each
    /// [`process`](Self::process). Rebuild the view if wasm memory grows.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn output_ptr(&self) -> *const f32 {
        self.output.as_ptr()
    }

    /// Number of CQT bins — the magnitude output length and spectrogram height.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    /// Per-bin centre frequencies in Hz, low → high.
    ///
    /// Copied out as a fresh `Float32Array` (called once at setup for axis
    /// labelling, not on the hot path).
    #[must_use]
    pub fn frequencies(&self) -> Vec<f32> {
        self.frequencies.clone()
    }

    /// Recommended advance between frames in samples (advisory hop length).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn hop_length(&self) -> usize {
        self.plan.hop_length()
    }

    /// Number of octave bands the multirate transform decomposes into.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn num_octaves(&self) -> usize {
        self.plan.num_octaves()
    }
}
