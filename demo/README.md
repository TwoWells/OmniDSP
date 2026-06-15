<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# OmniDSP CQT visualiser (DEMO-00)

A real-time **20 Hz–16 kHz Constant-Q Transform** visualiser, compiled to
`wasm32` and running entirely on the OmniDSP `RustBackend` floor (RustFFT /
realfft + auto-vectorised scalar ops). No vendor acceleration, no framework —
the floor as a product.

## Stack

- **Vite + TypeScript + vanilla DOM/Canvas** — a spectrogram is one canvas and a
  button.
- **Engine:** the [`omnidsp-wasm`](../omnidsp-wasm) crate (`CqtEngine`,
  `wasm-bindgen`), built with `wasm-pack --target web`. Imported directly from
  `../omnidsp-wasm/pkg`.
- **Pipeline (main thread, v1):** Web Audio → `ScriptProcessorNode` tap → PCM
  ring buffer → per-`requestAnimationFrame` hop → `CqtEngine.process()` →
  scrolling log-frequency Canvas 2D spectrogram (viridis). No AudioWorklet, no
  `SharedArrayBuffer` — stays on free static hosting.

## Prerequisites

- Rust toolchain + the `wasm32-unknown-unknown` target.
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/): `cargo binstall wasm-pack`
  (or `cargo install wasm-pack`). **Required** — the app imports the `pkg/` glue
  it emits.
- Node.js + npm.

## Run

From the repo root, the one-shot path:

```sh
make demo          # wasm-pack build → npm install → npm run dev
```

Or step by step:

```sh
make wasm-pack     # build omnidsp-wasm → ../omnidsp-wasm/pkg
cd demo
npm install
npm run dev        # open the printed localhost URL
```

Press **Start** for the bundled looping sweep clip (a 20 Hz→16 kHz exponential
sweep — watch the bright bin climb diagonally through the octaves), or **Use
microphone** for live input. The stats line shows per-frame `process()` time and
fps — the real-time headroom readout.

## Build (static deploy)

```sh
make wasm-pack
cd demo && npm install && npm run build   # → demo/dist (static, GitHub-Pages ready)
```

## Notes

- **Sample rate.** The CQT spec is built at the *actual* `AudioContext` rate
  (commonly 48 kHz), never hardcoded. At 48 kHz the 20 Hz bin pushes the CQT FFT
  length (`engine.input_len`) up to ~65 536 samples, so the analysis window —
  and the ring buffer — is ~1.4 s. That is inherent to constant-Q at 20 Hz
  (Gabor uncertainty); the multirate decomposition is what keeps the *compute*
  tractable. Bass bins respond slowly, treble bins fast — expected.
- **Deferred (not in v1):** WebGL rendering, AudioWorklet + worker + SAB, file
  upload, live parameter sliders, mobile polish.
