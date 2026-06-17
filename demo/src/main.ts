// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Entry point: load the wasm streaming CQT engine, wire Web Audio → ring buffer
// → per requestAnimationFrame: drain newly-arrived PCM → CqtEngine.process(n) →
// scroll the newest-anchored magnitude columns it emits.
//
// The engine is imported straight from the wasm-pack `--target web` output in
// ../../omnidsp-wasm/pkg (run `make wasm-pack` first). The `?url` import hands
// Vite the wasm asset so `init()` can fetch it.
import init, { CqtEngine } from "../../omnidsp-wasm/pkg/omnidsp_wasm.js";
import wasmUrl from "../../omnidsp-wasm/pkg/omnidsp_wasm_bg.wasm?url";
import { AudioEngine } from "./audio";
import { Spectrogram, drawOverlay } from "./spectrogram";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

function showError(err: unknown): void {
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
  $("error").textContent = msg;
  $("stats").textContent = "error";
  // eslint-disable-next-line no-console
  console.error(err);
}

/** Everything tied to one `min_freq` choice; rebuilt when the low edge changes. */
interface Active {
  engine: CqtEngine;
  audio: AudioEngine;
  spectro: Spectrogram;
  inCap: number;
  bins: number;
}

async function bootstrap(): Promise<void> {
  const wasm = await init(wasmUrl);

  // One AudioContext for the page; the engine spec is built at its *actual*
  // sample rate (often 48 kHz), never a hardcoded 44.1 kHz.
  const ctx = new AudioContext();

  const spectroCanvas = $<HTMLCanvasElement>("spectrogram");
  const overlayCanvas = $<HTMLCanvasElement>("overlay");
  const startBtn = $<HTMLButtonElement>("start");
  const micBtn = $<HTMLButtonElement>("mic");
  const minFreqSel = $<HTMLSelectElement>("minfreq");
  const stats = $("stats");

  let active: Active | null = null;

  // (Re)build the engine + audio graph + canvases for a given low edge. The CQT
  // FFT length — and thus the analysis window and latency — is sized to
  // `minFreq`, so changing it means a fresh engine and ring buffer.
  const rebuild = (minFreq: number): void => {
    const prev = active?.audio.source ?? "none";
    active?.audio.dispose();
    active?.engine.free(); // release the old engine's wasm-side buffers
    active = null;

    let engine: CqtEngine;
    try {
      engine = new CqtEngine(ctx.sampleRate, minFreq);
    } catch (err) {
      showError(`CqtEngine init failed at ${ctx.sampleRate} Hz / min ${minFreq} Hz:\n${err}`);
      return;
    }

    const inCap = engine.input_capacity;
    const bins = engine.num_bins;
    const binsPerOctave = engine.bins_per_octave;

    const audio = new AudioEngine(ctx);
    const spectro = new Spectrogram(spectroCanvas, bins);
    drawOverlay(overlayCanvas, engine.frequencies(), bins, binsPerOctave);
    active = { engine, audio, spectro, inCap, bins };

    // Carry the previous source across a rebuild so changing the low edge
    // mid-stream doesn't stop the audio.
    if (prev === "clip") void audio.startClip().catch(showError);
    else if (prev === "mic") void audio.useMic().catch(showError);
    else {
      stats.textContent = `ready · ${bins} bins · streaming (per-bin Q/f) · ${ctx.sampleRate / 1000} kHz — press Start`;
    }
  };

  startBtn.addEventListener("click", () => {
    void active?.audio.startClip().catch(showError);
    micBtn.disabled = false;
    startBtn.dataset.active = "true";
    micBtn.dataset.active = "false";
  });
  micBtn.addEventListener("click", () => {
    void active?.audio
      .useMic()
      .then(() => {
        micBtn.dataset.active = "true";
        startBtn.dataset.active = "false";
      })
      .catch(showError);
  });
  minFreqSel.addEventListener("change", () => rebuild(Number(minFreqSel.value)));

  rebuild(Number(minFreqSel.value));

  // --- Render loop ---
  let procMsAvg = 0;
  let frames = 0;
  let colSum = 0;
  let lastStat = performance.now();

  const frame = (): void => {
    requestAnimationFrame(frame);
    if (!active || !active.audio.running) return;
    const { engine, audio, spectro, inCap, bins } = active;

    // Rebuild views each frame: cheap (no copy), and robust if wasm linear
    // memory ever grows and detaches the backing ArrayBuffer.
    const inView = new Float32Array(wasm.memory.buffer, engine.input_ptr, inCap);
    const n = audio.drainNewSamples(inView);

    let cols = 0;
    if (n > 0) {
      const t0 = performance.now();
      try {
        cols = engine.process(n);
      } catch (err) {
        showError(err);
        return;
      }
      procMsAvg += (performance.now() - t0 - procMsAvg) * 0.1;

      if (cols > 0) {
        // Read the output view after process() (wasm memory may have grown) and
        // slice it per column — the plan emits low → high, oldest hop-boundary
        // column first.
        const outView = new Float32Array(wasm.memory.buffer, engine.output_ptr, cols * bins);
        for (let c = 0; c < cols; c++) {
          spectro.pushColumn(outView.subarray(c * bins, (c + 1) * bins));
        }
      }
    }

    frames++;
    colSum += cols;
    const now = performance.now();
    if (now - lastStat > 250) {
      const fps = (frames * 1000) / (now - lastStat);
      const cps = (colSum * 1000) / (now - lastStat);
      stats.textContent =
        `process ${procMsAvg.toFixed(2)} ms · ${fps.toFixed(0)} fps · ` +
        `${cps.toFixed(0)} col/s · ${bins} bins · ${ctx.sampleRate / 1000} kHz`;
      frames = 0;
      colSum = 0;
      lastStat = now;
    }
  };

  requestAnimationFrame(frame);
}

bootstrap().catch(showError);
