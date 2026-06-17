// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Right-edge live-spectrum meter: the newest CQT column as horizontal bars
// (one per bin, low freq at the bottom — aligned with the spectrogram rows),
// in viridis, with a slowly-decaying peak-hold tick per bin.

import { viridis } from "./colormap";

const METER_W = 96; // internal width in px; bar length maps magnitude → [0, METER_W]
const PEAK_DECAY_DB = 0.4; // dB the peak-hold falls per update (~24 dB/s at 60 fps)
const EPS = 1e-6;

export class SpectrumMeter {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly h: number;
  private readonly peakDb: Float32Array;
  private dbMin: number;
  private readonly dbMax = 0;

  constructor(canvas: HTMLCanvasElement, numBins: number, dbMin: number) {
    canvas.width = METER_W;
    canvas.height = numBins;
    this.h = numBins;
    this.dbMin = dbMin;
    this.peakDb = new Float32Array(numBins).fill(dbMin);
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, METER_W, numBins);
  }

  /** Move the dB floor (sensitivity). Higher floor → only louder detail shows. */
  setFloor(dbMin: number): void {
    this.dbMin = dbMin;
  }

  /** Redraw the meter from the current magnitude column. */
  update(mags: Float32Array): void {
    const span = this.dbMax - this.dbMin;
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, METER_W, this.h);

    for (let b = 0; b < this.h; b++) {
      const db = 20 * Math.log10(mags[b] + EPS);
      let t = (db - this.dbMin) / span;
      t = t < 0 ? 0 : t > 1 ? 1 : t;

      const peak = Math.max(db, this.peakDb[b] - PEAK_DECAY_DB);
      this.peakDb[b] = peak;
      let pt = (peak - this.dbMin) / span;
      pt = pt < 0 ? 0 : pt > 1 ? 1 : pt;

      const row = this.h - 1 - b; // low freq at the bottom
      const [r, g, bl] = viridis(t);
      this.ctx.fillStyle = `rgb(${r},${g},${bl})`;
      this.ctx.fillRect(0, row, t * METER_W, 1);

      // Peak-hold tick (bright, 2 px).
      this.ctx.fillStyle = "rgba(255,255,255,0.85)";
      this.ctx.fillRect(Math.min(METER_W - 2, pt * METER_W), row, 2, 1);
    }
  }
}

/**
 * Draw a compact horizontal viridis legend (dB floor → 0 dB, left → right) with
 * end labels, into a small inline canvas. Redrawn whenever the floor changes.
 */
export function drawLegend(canvas: HTMLCanvasElement, dbMin: number, dbMax: number): void {
  const dpr = window.devicePixelRatio || 1;
  const cssW = 150;
  const cssH = 14;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);

  const barW = 96;
  for (let x = 0; x < barW; x++) {
    const [r, g, b] = viridis(x / (barW - 1));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, 0, 1, cssH);
  }
  ctx.font = "10px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  // Floor (dbMin) is the dark left end; 0 dB (dbMax) the bright right end.
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fillText(`${dbMin}`, 3, cssH / 2);
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.fillText(`${dbMax} dB`, barW + 6, cssH / 2);
}
