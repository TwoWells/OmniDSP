// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

import { viridis } from "./colormap";

const HISTORY = 900; // columns of scroll history
const DB_MIN = -70; // magnitude floor (dB) → black
const DB_MAX = 0; // magnitude ceiling (dB) → bright

/**
 * Scrolling log-frequency spectrogram on a Canvas 2D. The CQT's bins are
 * already log-spaced, so the y-axis is free: bin 0 (low freq) at the bottom,
 * the top bin at the top. Each frame shifts the image one pixel left and draws
 * the newest magnitude column on the right.
 */
export class Spectrogram {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly w = HISTORY;
  private readonly h: number;
  private readonly column: ImageData;

  constructor(canvas: HTMLCanvasElement, numBins: number) {
    canvas.width = this.w;
    canvas.height = numBins;
    this.h = numBins;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.w, this.h);
    this.column = this.ctx.createImageData(1, numBins);
  }

  /** Push one CQT magnitude frame as the newest column. */
  pushColumn(mags: Float32Array): void {
    // Scroll left by one pixel (self-copy is well-defined on canvas).
    this.ctx.drawImage(this.ctx.canvas, -1, 0);

    const data = this.column.data;
    const span = DB_MAX - DB_MIN;
    for (let b = 0; b < this.h; b++) {
      const db = 20 * Math.log10(mags[b] + 1e-6);
      let t = (db - DB_MIN) / span;
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
 * Draw static octave gridlines + frequency labels on the overlay canvas, in CSS
 * pixels (DPR-aware), redrawing on resize. `binsPerOctave` marks the octave
 * boundaries; `freqs` supplies the per-bin centre frequencies for labels.
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
      const label = f >= 1000 ? `${(f / 1000).toFixed(1)}k` : `${Math.round(f)}`;
      ctx.fillStyle = "rgba(255,255,255,0.72)";
      ctx.fillText(`${label} Hz`, 5, Math.max(8, y - 7));
    }
  };

  render();
  window.addEventListener("resize", render);
}
