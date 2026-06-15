// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Entry point: load the wasm CQT engine, wire Web Audio → ring buffer → per
// requestAnimationFrame hop → CqtEngine.process() → scrolling spectrogram.
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
  inLen: number;
  outLen: number;
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

    const inLen = engine.input_len;
    const outLen = engine.output_len;
    const binsPerOctave = Math.max(1, Math.round(outLen / engine.num_octaves));

    const audio = new AudioEngine(ctx, inLen);
    const spectro = new Spectrogram(spectroCanvas, outLen);
    drawOverlay(overlayCanvas, engine.frequencies(), outLen, binsPerOctave);
    active = { engine, audio, spectro, inLen, outLen };

    // Carry the previous source across a rebuild so changing the low edge
    // mid-stream doesn't stop the audio.
    if (prev === "clip") void audio.startClip().catch(showError);
    else if (prev === "mic") void audio.useMic().catch(showError);
    else {
      const winS = (inLen / ctx.sampleRate).toFixed(2);
      stats.textContent = `ready · ${outLen} bins · win ${winS} s · ${ctx.sampleRate / 1000} kHz — press Start`;
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
  let lastStat = performance.now();

  const frame = (): void => {
    requestAnimationFrame(frame);
    if (!active || !active.audio.running) return;
    const { engine, audio, spectro, inLen, outLen } = active;

    // Rebuild views each frame: cheap (no copy), and robust if wasm linear
    // memory ever grows and detaches the backing ArrayBuffer.
    const inView = new Float32Array(wasm.memory.buffer, engine.input_ptr, inLen);
    audio.readLatestWindow(inView);

    const t0 = performance.now();
    try {
      engine.process();
    } catch (err) {
      showError(err);
      return;
    }
    const dt = performance.now() - t0;

    const outView = new Float32Array(wasm.memory.buffer, engine.output_ptr, outLen);
    spectro.pushColumn(outView);

    procMsAvg += (dt - procMsAvg) * 0.1;
    frames++;
    const now = performance.now();
    if (now - lastStat > 250) {
      const fps = (frames * 1000) / (now - lastStat);
      const winS = (inLen / ctx.sampleRate).toFixed(2);
      stats.textContent =
        `process ${procMsAvg.toFixed(1)} ms · ${fps.toFixed(0)} fps · ` +
        `${outLen} bins · win ${winS} s · ${ctx.sampleRate / 1000} kHz`;
      frames = 0;
      lastStat = now;
    }
  };

  requestAnimationFrame(frame);
}

bootstrap().catch(showError);
