// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! `omnidsp-wasm` — the browser CQT visualiser engine (DEMO-00).
//!
//! A thin [`wasm-bindgen`] binding over the OmniDSP `RustBackend` **streaming,
//! newest-anchored** multirate Constant-Q Transform (ticket 22). It builds the
//! CQT **once** at construction (at the real audio-context sample rate handed in
//! from JavaScript) and then exposes a [`process`](CqtEngine::process) entry
//! point that consumes **newly-arrived** PCM and emits the hop-boundary feature
//! columns it crossed. Because the analysis is newest-anchored, each bin's
//! latency is its Gabor floor `Q/f` — treble near-instant, only deep bass
//! inherently slow — rather than the whole analysis-window length the batch CQT
//! paid on every bin. No allocation, no copy across the wasm boundary:
//!
//! 1. JS writes up to [`input_capacity`](CqtEngine::input_capacity) newly-arrived
//!    PCM samples into wasm linear memory at [`input_ptr`](CqtEngine::input_ptr).
//! 2. JS calls [`process(n)`](CqtEngine::process) with the count it wrote; the
//!    engine feeds them to the persistent streaming plan and writes the
//!    newest-anchored magnitude columns produced into an internal buffer,
//!    returning the **column count**.
//! 3. JS reads `count *` [`num_bins`](CqtEngine::num_bins) `f32` magnitudes back
//!    from [`output_ptr`](CqtEngine::output_ptr) via a `Float32Array` view over
//!    `WebAssembly.Memory.buffer` — again with no copy — and scrolls each column.
//!
//! The transform runs on the pure-Rust floor (RustFFT / realfft + scalar
//! auto-vectorised ops); built with `RUSTFLAGS="-C target-feature=+simd128"`
//! the inner loops vectorise and the multirate decomposition fits a real-time
//! `requestAnimationFrame` budget. No vendor acceleration is involved — the
//! floor itself is the product.
//!
//! See `README.md` for the `wasm-pack build --target web` packaging step.

use omnidsp::OmniDSP;
use omnidsp::RustBackend;
use omnidsp::create::CreatePlan;
use omnidsp_core::design::cqt::{self, CqtSpec};
use omnidsp_core::modules::cqt::CqtStreamSpec;
use omnidsp_core::types::Window;
use wasm_bindgen::prelude::wasm_bindgen;

/// Upper edge of the demo CQT range (must stay below Nyquist).
const MAX_FREQ: f64 = 16_000.0;

/// Bins per octave for the visualiser (12 = one bin per semitone).
const BINS_PER_OCTAVE: u32 = 12;

/// Maximum new PCM samples JS may feed in a single [`process`](CqtEngine::process)
/// call — the input-scratch capacity. One `requestAnimationFrame` at 48 kHz is
/// ~800 samples, so this is several frames of headroom; JS feeds whatever has
/// arrived since the previous frame, never more than this.
const INPUT_CAPACITY: usize = 8192;

/// The concrete `RustBackend` streaming multirate CQT plan over `f32`.
///
/// Spelled via the backend's public `CreatePlan` associated type so the wasm
/// crate never names the routed decimator sub-plan by hand; it resolves to
/// `OmniCqtStreamPlan` over the floor's r2c plan, `ScalarVecOps`, and the
/// continuous `OmniResample(1, 2)` decimator.
type RustCqtStreamPlan = <RustBackend as CreatePlan<CqtStreamSpec<f32>>>::Plan;

/// Real-time streaming Constant-Q Transform engine for the browser visualiser.
///
/// Constructed once from the audio context's sample rate; newly-arrived audio is
/// fed through [`process`](Self::process) with no per-frame allocation, and the
/// persistent plan emits newest-anchored magnitude columns at the hop rate.
#[wasm_bindgen]
pub struct CqtEngine {
    plan: RustCqtStreamPlan,
    /// PCM input scratch (capacity `INPUT_CAPACITY`); JS writes new samples here.
    input: Vec<f32>,
    /// Magnitude output scratch, sized for the most columns one full input chunk
    /// can emit (`max_output_columns(INPUT_CAPACITY) * num_bins`).
    output: Vec<f32>,
    /// Per-bin centre frequencies in Hz (low → high), for axis labelling.
    frequencies: Vec<f32>,
    /// Bin count — the per-column magnitude count and spectrogram height.
    num_bins: usize,
}

#[wasm_bindgen]
impl CqtEngine {
    /// Build the streaming CQT engine for a given audio-context sample rate and
    /// low-edge frequency.
    ///
    /// The `min_freq … 16 kHz` spec is built at **`sample_rate`** — the actual
    /// rate JS reports (commonly 48 kHz, sometimes 44.1 kHz). Do not hardcode a
    /// rate; feeding a 48 kHz context a 44.1 kHz spec mislabels every bin.
    ///
    /// `min_freq` sets the deepest octave (and hence the longest analysis
    /// window), but with newest-anchoring each bin responds at its own `Q/f`
    /// floor — so a lower edge no longer adds latency across the whole spectrum,
    /// only the deep bass bins it introduces are inherently slow. Treble stays
    /// near-instant regardless.
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
        .map_err(|e| format!("CQT design failed at {sample_rate} Hz (min {min_freq} Hz): {e}"))?;

        let dsp = OmniDSP::rust();
        let plan: RustCqtStreamPlan = dsp
            .cqt_stream(&CqtStreamSpec::new(spec))
            .map_err(|e| format!("CQT stream plan creation failed: {e}"))?;

        let num_bins = plan.num_bins();
        let max_columns = plan.max_output_columns(INPUT_CAPACITY);
        let input = vec![0.0_f32; INPUT_CAPACITY];
        let output = vec![0.0_f32; max_columns * num_bins];
        let frequencies: Vec<f32> = plan.bin_frequencies().iter().map(|&f| f as f32).collect();

        Ok(Self {
            plan,
            input,
            output,
            frequencies,
            num_bins,
        })
    }

    /// Feed `n` newly-arrived PCM samples and compute the newest-anchored
    /// magnitude columns they crossed.
    ///
    /// Reads the first `n` samples JS wrote at [`input_ptr`](Self::input_ptr)
    /// (clamped to [`input_capacity`](Self::input_capacity)), advances the
    /// persistent streaming plan, writes `count *` [`num_bins`](Self::num_bins)
    /// magnitudes (one column at a time, low → high) to
    /// [`output_ptr`](Self::output_ptr), and returns `count` — the number of
    /// hop-boundary columns produced (often 0 or 1 per frame, more after a
    /// stall). No allocation, no boundary copy.
    ///
    /// # Errors
    ///
    /// Returns a JS string error if the transform fails (buffer lengths are
    /// engine-owned and always correct, so this is effectively infallible).
    pub fn process(&mut self, n: usize) -> Result<usize, String> {
        let n = n.min(self.input.len());
        self.plan
            .process_magnitude(&self.input[..n], &mut self.output)
            .map_err(|e| format!("CQT stream process failed: {e}"))
    }

    /// Reset the streaming state (decimator delay lines, per-octave rings)
    /// without rebuilding the plan.
    pub fn reset(&mut self) {
        self.plan.reset();
    }

    /// Pointer to the PCM input buffer in wasm linear memory.
    ///
    /// JS builds a `Float32Array(memory.buffer, engine.input_ptr,
    /// engine.input_capacity)` view, writes each frame's new samples into its
    /// first `n` slots, then calls [`process(n)`](Self::process). Rebuild the
    /// view if wasm memory grows.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn input_ptr(&self) -> *const f32 {
        self.input.as_ptr()
    }

    /// Maximum new samples JS may write/feed per [`process`](Self::process) call.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn input_capacity(&self) -> usize {
        self.input.len()
    }

    /// Pointer to the magnitude output buffer in wasm linear memory.
    ///
    /// After [`process`](Self::process) returns `count`, JS reads `count *`
    /// [`num_bins`](Self::num_bins) `f32` magnitudes from here. Rebuild the view
    /// if wasm memory grows.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn output_ptr(&self) -> *const f32 {
        self.output.as_ptr()
    }

    /// Number of CQT bins — the per-column magnitude count and spectrogram height.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Bins per octave (12) — the octave-gridline spacing for the overlay.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn bins_per_octave(&self) -> usize {
        BINS_PER_OCTAVE as usize
    }

    /// Per-bin centre frequencies in Hz, low → high.
    ///
    /// Copied out as a fresh `Float32Array` (called once at setup for axis
    /// labelling, not on the hot path).
    #[must_use]
    pub fn frequencies(&self) -> Vec<f32> {
        self.frequencies.clone()
    }
}
