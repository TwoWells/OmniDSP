<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Two Wells <contact@twowells.dev>
-->

# omnidsp-wasm

The browser engine for the DEMO-00 real-time CQT visualiser. A thin
[`wasm-bindgen`] binding over the OmniDSP `RustBackend` multirate Constant-Q
Transform (20 Hz – 16 kHz). The transform runs entirely on the pure-Rust floor
(RustFFT / realfft + scalar auto-vectorised ops) — **no vendor acceleration**.
The floor itself is the product.

This crate is **deliberately not a member of the core lint workspace** (see the
root `Cargo.toml` `[workspace.exclude]`): `wasm-bindgen`'s generated code does
not meet the library's `pedantic` / `nursery` / `forbid(unsafe_code)` bar, and
excluding it keeps `make check` on the core workspace clean and fast.

## Building (simd128 is mandatory)

The floor's real-time headroom in the browser depends on auto-vectorisation,
which requires the `simd128` target feature. Build with:

```sh
# Plain target build (compile proof; what `make wasm-pack`'s check step runs):
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --manifest-path omnidsp-wasm/Cargo.toml \
  --release --target wasm32-unknown-unknown

# Packaged ESM + JS glue for the Vite/TS app (requires `wasm-pack`):
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build omnidsp-wasm --release --target web
```

From the repo root the same is available as a make target:

```sh
make wasm-pack   # builds omnidsp-wasm for wasm32-unknown-unknown (+simd128)
```

`wasm-pack build --target web` emits a `pkg/` directory (`omnidsp_wasm.js`,
`omnidsp_wasm_bg.wasm`, `.d.ts`) the demo app imports directly — no bundler
plugin required for the static GitHub Pages host (no `SharedArrayBuffer`, so no
COOP/COEP headers needed).

> If `wasm-pack` is not installed, the plain `cargo build --target
> wasm32-unknown-unknown` above is sufficient proof the crate compiles for the
> browser target; `wasm-pack` is only needed to emit the JS glue for the app.

## Wasm boundary (no per-frame copy)

The CQT is constructed **once**; each hop is processed with no allocation and no
copy across the wasm boundary. JS writes PCM directly into wasm linear memory
and reads magnitudes back out of it via typed-array views:

```js
import init, { CqtEngine } from "./pkg/omnidsp_wasm.js";

const wasm = await init();                 // wasm.memory is the linear memory
const engine = new CqtEngine(audioCtx.sampleRate);   // build spec at the REAL rate

// Views over wasm memory (rebuild if memory grows — these buffers don't here,
// but a defensive app re-derives them when `wasm.memory.buffer` changes):
const input  = new Float32Array(wasm.memory.buffer, engine.input_ptr,  engine.input_len);
const output = new Float32Array(wasm.memory.buffer, engine.output_ptr, engine.output_len);

const freqs = engine.frequencies();        // Hz per bin, low → high (axis labels)

function onHop(pcmHop /* Float32Array, length === engine.input_len */) {
  input.set(pcmHop);                       // write straight into wasm memory
  engine.process();                        // run the multirate CQT
  drawColumn(output);                       // read magnitudes straight out
}
```

### API

| Member | Kind | Meaning |
| --- | --- | --- |
| `new CqtEngine(sample_rate)` | constructor | Build the 20 Hz – 16 kHz CQT at the given context rate. |
| `engine.input_ptr` / `engine.input_len` | getters | Address + length of the PCM input buffer (length = FFT length = required hop size). |
| `engine.process()` | method | Compute the magnitude spectrum of the current input buffer in place. |
| `engine.output_ptr` / `engine.output_len` | getters | Address + length of the magnitude output buffer (length = bin count = spectrogram height). |
| `engine.frequencies()` | method | Per-bin centre frequencies in Hz, low → high (copied once for axis labelling). |
| `engine.hop_length` | getter | Advisory advance between frames, in samples. |
| `engine.num_octaves` | getter | Octave bands the multirate transform decomposes into. |

## Sample-rate note

The spec is built at the sample rate passed to the constructor. Pass
`AudioContext.sampleRate` (often 48 kHz) — do **not** hardcode 44.1 kHz, or
every bin frequency will be mislabelled.

## Next step

The Vite + TypeScript visualiser app (the `demo/` directory) consumes this
crate: bundled looping clip + "use microphone" button → Web Audio ring buffer →
per-`requestAnimationFrame` hop → `CqtEngine.process()` → scrolling
log-frequency Canvas spectrogram. That app is a separate deliverable.

[`wasm-bindgen`]: https://rustwasm.github.io/wasm-bindgen/
