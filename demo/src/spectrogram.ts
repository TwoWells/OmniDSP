// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

import { viridis } from "./colormap";
import { freqToNote, hzLabel } from "./notes";

export const DEFAULT_HISTORY = 1800; // columns of scroll history (canvas width)
const DB_MAX = 0; // magnitude ceiling (dB) → bright

const EPS = 1e-6;

/**
 * Scrolling log-frequency spectrogram on a Canvas 2D. The CQT's bins are
 * already log-spaced, so the y-axis is free: bin 0 (low freq) at the bottom,
 * the top bin at the top. Each pushed column shifts the image one pixel left
 * and draws the newest magnitude column on the right.
 *
 * `history` (the canvas internal width) sets the on-screen span: longer = more
 * time, columns get thinner (the stage smooths them). The wall-clock span is
 * `history / columns-per-second`, the latter driven at the streaming hop rate.
 * `dbMin` is the magnitude floor (sensitivity): raise it to suppress faint
 * detail, lower it to pull it up.
 */
export class Spectrogram {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly w: number;
  private readonly h: number;
  private readonly column: ImageData;
  private dbMin: number;

  constructor(canvas: HTMLCanvasElement, numBins: number, history: number, dbMin: number) {
    canvas.width = history;
    canvas.height = numBins;
    this.w = history;
    this.h = numBins;
    this.dbMin = dbMin;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.w, this.h);
    this.column = this.ctx.createImageData(1, numBins);
  }

  /** Columns of scroll history (canvas internal width). */
  get history(): number {
    return this.w;
  }

  /** Move the dB floor (sensitivity). Higher floor → only louder detail shows. */
  setFloor(dbMin: number): void {
    this.dbMin = dbMin;
  }

  /** Push one CQT magnitude frame as the newest column. */
  pushColumn(mags: Float32Array): void {
    // Scroll left by one pixel (self-copy is well-defined on canvas).
    this.ctx.drawImage(this.ctx.canvas, -1, 0);

    const data = this.column.data;
    const span = DB_MAX - this.dbMin;
    for (let b = 0; b < this.h; b++) {
      const db = 20 * Math.log10(mags[b] + EPS);
      let t = (db - this.dbMin) / span;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const [r, g, bl] = viridis(t);
      const row = this.h - 1 - b; // low freq at the bottom
      const o = row * 4;
      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = bl;
      data[o + 3] = 255;
    }
    this.ctx.putImageData(this.column, this.w - 1, 0);
  }
}

/**
 * Draw static octave gridlines + note/frequency labels on the overlay canvas,
 * in CSS pixels (DPR-aware), redrawing on resize. `binsPerOctave` marks the
 * octave boundaries; `freqs` supplies the per-bin centre frequencies. Because
 * the bins are constant-Q, each octave line lands on a note — labelled `note ·
 * Hz` (e.g. "A1 · 55 Hz").
 */
export function drawOverlay(
  canvas: HTMLCanvasElement,
  freqs: Float32Array,
  numBins: number,
  binsPerOctave: number,
): void {
  const denom = Math.max(1, numBins - 1);

  const render = (): void => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";

    for (let b = 0; b < numBins; b += binsPerOctave) {
      const y = rect.height * (1 - b / denom);
      ctx.strokeStyle = "rgba(255,255,255,0.16)";
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(rect.width, y);
      ctx.stroke();

      const f = freqs[b];
      const label = `${freqToNote(f)} · ${hzLabel(f)} Hz`;
      ctx.fillStyle = "rgba(255,255,255,0.78)";
      ctx.fillText(label, 5, Math.max(8, y - 7));
    }
  };

  render();
  window.addEventListener("resize", render);
}
