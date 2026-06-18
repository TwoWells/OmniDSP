// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Entry point: load the wasm streaming CQT engine, wire Web Audio → ring buffer
// → per requestAnimationFrame: drain newly-arrived PCM → CqtEngine.process(n) →
// max-pool the emitted columns to the chosen on-screen span → scroll. Plus a
// live-spectrum meter, a hover readout, and span / sensitivity / low-edge
// controls.
//
// The engine is imported straight from the wasm-pack `--target web` output in
// ../../omnidsp-wasm/pkg (run `make wasm-pack` first). The `?url` import hands
// Vite the wasm asset so `init()` can fetch it.
import init, { CqtEngine } from "../../omnidsp-wasm/pkg/omnidsp_wasm.js";
import wasmUrl from "../../omnidsp-wasm/pkg/omnidsp_wasm_bg.wasm?url";
import { AudioEngine } from "./audio";
import { Spectrogram, drawOverlay, DEFAULT_HISTORY } from "./spectrogram";
import { SpectrumMeter, drawLegend } from "./meter";
import { freqToNote, hzLabel } from "./notes";

const DB_MAX = 0;

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

/** Everything tied to one engine build; rebuilt when the low edge changes. */
interface Active {
  engine: CqtEngine;
  audio: AudioEngine;
  spectro: Spectrogram;
  meter: SpectrumMeter;
  inCap: number;
  bins: number;
  freqs: Float32Array;
  /** Per-bin newest-anchored latency in input samples (low → high). */
  latencies: Float32Array;
  /** Max-pool accumulator across emitted columns (one displayed column per K). */
  pool: Float32Array;
}

async function bootstrap(): Promise<void> {
  const wasm = await init(wasmUrl);

  // One AudioContext for the page; the engine spec is built at its *actual*
  // sample rate (often 48 kHz), never a hardcoded 44.1 kHz.
  const ctx = new AudioContext();

  const spectroCanvas = $<HTMLCanvasElement>("spectrogram");
  const overlayCanvas = $<HTMLCanvasElement>("overlay");
  const meterCanvas = $<HTMLCanvasElement>("meter");
  const legendCanvas = $<HTMLCanvasElement>("legend");
  const readout = $<HTMLDivElement>("readout");
  const stage = $<HTMLDivElement>("stage");
  const startBtn = $<HTMLButtonElement>("start");
  const micBtn = $<HTMLButtonElement>("mic");
  const minFreqSel = $<HTMLSelectElement>("minfreq");
  const spanSel = $<HTMLSelectElement>("span");
  const sensInput = $<HTMLInputElement>("sens");
  const stats = $("stats");

  let active: Active | null = null;
  // The slider reads as "sensitivity": higher (right) = deeper dynamic range =
  // lower dB floor = fainter detail shown. dbMin is the negated slider value.
  let dbMin = -Number(sensInput.value);
  let targetSec = Number(spanSel.value); // desired on-screen time span
  let poolCount = 0; // emitted columns accumulated into the current pooled one
  let poolK = 1; // emitted columns per displayed column (drives the span)
  let lastCps = 0; // most recent columns/second, for poolK + the hover time-ago

  // Convert each bin's newest-anchored latency (input samples) to displayed
  // columns and hand it to the spectrogram for per-row de-warping. One displayed
  // column is `hop · poolK = (sr/cps) · poolK` input samples, so
  // `cols[bin] = latency_samples[bin] · cps / (sr · poolK)`. Recomputed whenever
  // poolK / cps drift (below), since the column→time scale moves with them.
  const updateLatencyColumns = (cps: number): void => {
    if (!active || cps <= 0) return;
    const { latencies, spectro } = active;
    const scale = cps / (ctx.sampleRate * poolK);
    const cols = new Float32Array(latencies.length);
    for (let b = 0; b < latencies.length; b++) cols[b] = latencies[b] * scale;
    spectro.setLatencyColumns(cols);
  };

  // Recompute the pool factor so `DEFAULT_HISTORY` displayed columns cover
  // `targetSec`: K = emitted-col/s · span / width. The displayed-column scale
  // moves with poolK, so refresh the per-bin latency offsets here too.
  const updatePoolK = (cps: number): void => {
    if (cps > 0) poolK = Math.max(1, Math.round((cps * targetSec) / DEFAULT_HISTORY));
    updateLatencyColumns(cps);
  };

  // (Re)build the engine + audio graph + canvases. The CQT is sized to the low
  // edge, so changing it means a fresh engine + ring; the span knob does not.
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
    const freqs = engine.frequencies();
    const latencies = engine.bin_latencies();

    const audio = new AudioEngine(ctx);
    const spectro = new Spectrogram(spectroCanvas, bins, DEFAULT_HISTORY, dbMin);
    const meter = new SpectrumMeter(meterCanvas, bins, dbMin);
    drawOverlay(overlayCanvas, freqs, bins, engine.bins_per_octave);
    active = {
      engine,
      audio,
      spectro,
      meter,
      inCap,
      bins,
      freqs,
      latencies,
      pool: new Float32Array(bins),
    };
    poolCount = 0;
    // Apply the current column scale to the new engine's per-bin latencies
    // (a fresh engine resets poolK's relationship to the new bin set).
    updateLatencyColumns(lastCps);

    // Carry the previous source across a rebuild so changing a control
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
  spanSel.addEventListener("change", () => {
    targetSec = Number(spanSel.value);
    updatePoolK(lastCps);
  });
  sensInput.addEventListener("input", () => {
    dbMin = -Number(sensInput.value);
    active?.spectro.setFloor(dbMin);
    active?.meter.setFloor(dbMin);
    drawLegend(legendCanvas, dbMin, DB_MAX);
  });

  // Hover readout: cursor over the spectrogram → nearest note, frequency, and
  // how long ago that column was (displayed-columns-from-right · seconds each).
  stage.addEventListener("mousemove", (e) => {
    if (!active) return;
    const rect = stage.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) {
      readout.hidden = true;
      return;
    }
    const bin = Math.min(active.bins - 1, Math.max(0, Math.round((1 - fy) * (active.bins - 1))));
    const f = active.freqs[bin];
    const colsFromRight = Math.round((1 - fx) * (active.spectro.history - 1));
    const age = lastCps > 0 ? (colsFromRight * poolK) / lastCps : 0;
    readout.textContent = `${freqToNote(f)} · ${hzLabel(f)} Hz · −${age.toFixed(1)} s`;
    readout.style.left = `${e.clientX}px`;
    readout.style.top = `${e.clientY}px`;
    readout.hidden = false;
  });
  stage.addEventListener("mouseleave", () => {
    readout.hidden = true;
  });

  drawLegend(legendCanvas, dbMin, DB_MAX);
  rebuild(Number(minFreqSel.value));

  // --- Render loop ---
  let procMsAvg = 0;
  let frames = 0;
  let colSum = 0;
  let lastStat = performance.now();

  const frame = (): void => {
    requestAnimationFrame(frame);
    if (!active || !active.audio.running) return;
    const { engine, audio, spectro, meter, inCap, bins, pool } = active;

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
        // Read the output view after process() (wasm memory may have grown). The
        // plan emits low → high, oldest hop-boundary column first. Max-pool each
        // emitted column into `pool`; flush one displayed column every poolK.
        const outView = new Float32Array(wasm.memory.buffer, engine.output_ptr, cols * bins);
        for (let c = 0; c < cols; c++) {
          const base = c * bins;
          for (let b = 0; b < bins; b++) {
            const v = outView[base + b];
            if (v > pool[b]) pool[b] = v;
          }
          if (++poolCount >= poolK) {
            spectro.pushColumn(pool);
            pool.fill(0);
            poolCount = 0;
          }
        }
        // The meter shows the live instantaneous spectrum (newest raw column).
        meter.update(outView.subarray((cols - 1) * bins, cols * bins));
      }
    }

    frames++;
    colSum += cols;
    const now = performance.now();
    if (now - lastStat > 250) {
      const fps = (frames * 1000) / (now - lastStat);
      const cps = (colSum * 1000) / (now - lastStat);
      lastCps = cps;
      updatePoolK(cps);
      const span = cps > 0 ? (DEFAULT_HISTORY * poolK) / cps : targetSec;
      stats.textContent =
        `process ${procMsAvg.toFixed(2)} ms · ${fps.toFixed(0)} fps · ` +
        `${cps.toFixed(0)} col/s · ${span.toFixed(1)} s span · ${bins} bins`;
      frames = 0;
      colSum = 0;
      lastStat = now;
    }
  };

  requestAnimationFrame(frame);
}

bootstrap().catch(showError);
