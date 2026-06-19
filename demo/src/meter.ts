// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Right-edge live-spectrum meter: the newest CQT column as horizontal bars
// (one per bin, low freq at the bottom — aligned with the spectrogram rows),
// in viridis, with a slowly-decaying peak-hold tick per bin.

import { viridis } from "./colormap";

const METER_W = 96; // internal width in px; bar length maps magnitude → [0, METER_W]
const PEAK_DECAY_DB = 0.4; // dB the peak-hold falls per update (~24 dB/s at 60 fps)
const EPS = 1e-6;
const MIN_TAU = 0.005; // s — residual-stepping ease floor; the treble is ≈ instant
const HIST_FRAMES = 64; // rolling history depth (~0.8 s at 75 fps) for the delay-line read
// Across-frequency Gaussian σ, in bins. One bin is the CQT's resolution limit:
// the design sets each filter's fractional bandwidth 1/Q = 2^(1/B) − 1 to exactly
// one bin step, so σ = 1 bin smooths the bin-grid jaggedness without erasing
// resolved detail (FWHM ≈ 2.35 bins covers the Kaiser main lobe). At 12 bins/oct
// this is one semitone; expressing it in bins keeps it correct at any B.
const SMOOTH_SIGMA_BINS = 1;

export class SpectrumMeter {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly h: number;
  private readonly peakDb: Float32Array;
  /** Per-bin eased dB (the drawn bar), to absorb residual decimation stepping. */
  private readonly smoothDb: Float32Array;
  /** Per-bin ease time constant (seconds), from latency: treble fast, bass slow. */
  private tau: Float32Array;
  /**
   * Per-bin read-back (seconds): `D(k) − latency(k)`, where `D` is the *smoothed*
   * latency line. Sampling each bin this far behind the live edge makes its
   * effective delay `now − D(k)` — smooth across frequency, so the octave-step
   * kink is splined away — while keeping the read-back small (treble ≈ newest,
   * bass ≈ its own latency). See `update`.
   */
  private readBack: Float32Array;
  /** Symmetric Gaussian weights for the across-frequency dB smoothing. */
  private freqKernel: Float32Array;
  private freqRadius = 0;
  /** Rolling history of recent magnitude columns (`HIST_FRAMES × h`) + stamps. */
  private readonly hist: Float32Array;
  private readonly histT: Float32Array;
  private histHead = 0;
  private histCount = 0;
  private lastUpdate = 0;
  private dbMin: number;
  private readonly dbMax = 0;

  constructor(canvas: HTMLCanvasElement, numBins: number, dbMin: number) {
    canvas.width = METER_W;
    canvas.height = numBins;
    this.h = numBins;
    this.dbMin = dbMin;
    this.peakDb = new Float32Array(numBins).fill(dbMin);
    this.smoothDb = new Float32Array(numBins).fill(dbMin);
    this.tau = new Float32Array(numBins).fill(MIN_TAU); // instant until latencies set
    this.readBack = new Float32Array(numBins); // 0 → reads newest until set
    this.freqKernel = new Float32Array([1]); // single tap until setLatencies builds it
    this.hist = new Float32Array(HIST_FRAMES * numBins);
    this.histT = new Float32Array(HIST_FRAMES);
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

  /**
   * Derive the smoothed-delay-line read-back from each bin's newest-anchored
   * latency (input samples). `D(k)` is a monotone cubic (PCHIP) through one knot
   * per octave at the octave's **highest** bin — its latency *floor* — plus the
   * global bottom bin to anchor the low end. Latency falls monotonically with
   * frequency (smooth `Q/2f` arcs with a downward decimation step at each octave
   * seam), so each top-bin knot connects up to the next octave's top a full step
   * lower: the spline descends *from above* and only touches the live edge at the
   * octave tops. That makes `D` a smooth upper envelope — `readBack = D − latency`
   * is `≥ 0` everywhere (no clamping), `0` at each octave top (treble ≈ newest →
   * snappy), and grows toward each octave's bottom (bass blooms) — smooth across
   * the seams (no kink). Bin index is log-frequency for the geometric bins, so
   * this is a spline in log f. A light per-bin ease (`tau ∝ latency`) then absorbs
   * the decimated bass's residual stepping. See `update`.
   */
  setLatencies(latencies: Float32Array, sampleRate: number, binsPerOctave: number): void {
    const n = latencies.length;
    const latS = new Float32Array(n);
    for (let b = 0; b < n; b++) latS[b] = latencies[b] / sampleRate;

    const readBack = new Float32Array(n);
    const tau = new Float32Array(n);
    for (let b = 0; b < n; b++) tau[b] = Math.max(MIN_TAU, latS[b]);

    // Knot indices: the highest bin of each octave (counting down from the top),
    // plus the global bottom bin. Ascending, deduped → a strictly-descending
    // latency knot set (higher bin → lower latency) for the monotone spline.
    const idx: number[] = [];
    for (let k = n - 1; k >= 0; k -= binsPerOctave) idx.push(k);
    if (idx.length === 0 || idx[idx.length - 1] !== 0) idx.push(0);
    idx.reverse();

    if (idx.length >= 2) {
      const xs = new Float64Array(idx.length);
      const ys = new Float64Array(idx.length);
      for (let j = 0; j < idx.length; j++) {
        xs[j] = idx[j];
        ys[j] = latS[idx[j]];
      }
      const spline = fitPchip(xs, ys);
      for (let b = 0; b < n; b++) {
        const d = evalPchip(spline, b); // smoothed delay line D(b)
        readBack[b] = Math.max(0, d - latS[b]);
      }
    }
    // (idx < 2 octaves → readBack stays 0: every bin reads newest.)

    this.readBack = readBack;
    this.tau = tau;

    // Frequency-smoothing kernel: a Gaussian of σ = `SMOOTH_SIGMA_BINS` bins.
    // Bins are log-spaced, so a constant bin width is constant fractional-octave
    // (constant-Q) smoothing of the dB — the same blur at every frequency — and
    // σ = 1 bin matches the transform's own resolution (see the const).
    const sigma = SMOOTH_SIGMA_BINS;
    const radius = Math.min(16, Math.round(3 * sigma));
    const kernel = new Float32Array(2 * radius + 1);
    for (let j = -radius; j <= radius; j++) {
      kernel[j + radius] = Math.exp(-(j * j) / (2 * sigma * sigma));
    }
    this.freqKernel = kernel;
    this.freqRadius = radius;
  }

  /** Redraw the meter, sampling each bin along its smoothed delay line. */
  update(mags: Float32Array): void {
    const now = performance.now() / 1000;
    // Frame Δt drives the residual-stepping ease; clamp so a stalled tab is fine.
    const dt = this.lastUpdate > 0 ? Math.min(now - this.lastUpdate, 0.1) : 1 / 60;
    this.lastUpdate = now;

    // Push the newest column into the rolling history.
    const base = this.histHead * this.h;
    for (let b = 0; b < this.h; b++) this.hist[base + b] = mags[b];
    this.histT[this.histHead] = now;
    this.histHead = (this.histHead + 1) % HIST_FRAMES;
    if (this.histCount < HIST_FRAMES) this.histCount += 1;

    // Pass 1 — sample each bin along its delay line and ease it in time. The
    // residual-stepping ease is per bin (treble instant, bass slow).
    for (let b = 0; b < this.h; b++) {
      const mag = this.sampleBin(b, now - this.readBack[b]);
      const targetDb = Math.max(this.dbMin, 20 * Math.log10(mag + EPS));
      const alpha = 1 - Math.exp(-dt / this.tau[b]);
      this.smoothDb[b] += alpha * (targetDb - this.smoothDb[b]);
    }

    // Pass 2 — smooth the eased dB across frequency (Gaussian, constant octave-
    // fraction width) and draw. The peak-hold tick follows the smoothed bar.
    const span = this.dbMax - this.dbMin;
    const kernel = this.freqKernel;
    const radius = this.freqRadius;
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, METER_W, this.h);

    for (let b = 0; b < this.h; b++) {
      let db = this.smoothDb[b];
      if (radius > 0) {
        let acc = 0;
        let wsum = 0;
        for (let j = -radius; j <= radius; j++) {
          const bb = b + j;
          if (bb < 0 || bb >= this.h) continue; // partial window at the edges
          const w = kernel[j + radius];
          acc += w * this.smoothDb[bb];
          wsum += w;
        }
        db = acc / wsum;
      }

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

      // Peak-hold tick (bright, 2 px) — follows the smoothed bar.
      this.ctx.fillStyle = "rgba(255,255,255,0.85)";
      this.ctx.fillRect(Math.min(METER_W - 2, pt * METER_W), row, 2, 1);
    }
  }

  /**
   * Linearly interpolate bin `b`'s magnitude from the rolling history at wall-
   * clock `targetT` (seconds). Clamps to the newest/oldest retained column.
   */
  private sampleBin(b: number, targetT: number): number {
    if (this.histCount === 0) return 0;
    const newest = (this.histHead - 1 + HIST_FRAMES) % HIST_FRAMES;
    if (targetT >= this.histT[newest]) return this.hist[newest * this.h + b];
    for (let step = 1; step < this.histCount; step += 1) {
      const older = (newest - step + HIST_FRAMES) % HIST_FRAMES;
      if (this.histT[older] <= targetT) {
        const newer = (older + 1) % HIST_FRAMES;
        const t0 = this.histT[older];
        const t1 = this.histT[newer];
        const f = t1 > t0 ? (targetT - t0) / (t1 - t0) : 0;
        const v0 = this.hist[older * this.h + b];
        const v1 = this.hist[newer * this.h + b];
        return v0 + f * (v1 - v0);
      }
    }
    const oldest = (this.histHead - this.histCount + HIST_FRAMES) % HIST_FRAMES;
    return this.hist[oldest * this.h + b];
  }
}

/** A monotone piecewise-cubic Hermite interpolant (PCHIP / Fritsch–Carlson). */
interface Pchip {
  xs: Float64Array;
  ys: Float64Array;
  ms: Float64Array; // shape-preserving tangents at the knots
}

/**
 * Fit a shape-preserving monotone cubic (PCHIP) through ascending knots
 * `(xs[i], ys[i])`. Each segment stays within its two knot values — no overshoot
 * — so through a strictly-monotone knot set the interpolant never crosses a knot
 * level, the property the delay line relies on (see `setLatencies`). `xs` must be
 * strictly ascending with ≥ 2 knots.
 */
function fitPchip(xs: Float64Array, ys: Float64Array): Pchip {
  const n = xs.length;
  const h = new Float64Array(n - 1);
  const delta = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = xs[i + 1] - xs[i];
    delta[i] = (ys[i + 1] - ys[i]) / h[i];
  }
  const ms = new Float64Array(n);
  ms[0] = delta[0];
  ms[n - 1] = delta[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] * delta[i] <= 0) {
      ms[i] = 0; // local extremum (or a flat) → zero tangent keeps it monotone
    } else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      ms[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]); // weighted harmonic mean
    }
  }
  return { xs, ys, ms };
}

/** Evaluate a PCHIP at `x`, clamping to the end knots outside the knot range. */
function evalPchip(s: Pchip, x: number): number {
  const { xs, ys, ms } = s;
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 0;
  while (i < n - 1 && x > xs[i + 1]) i++; // knot count is tiny (~octaves) → linear scan
  const hseg = xs[i + 1] - xs[i];
  const t = (x - xs[i]) / hseg;
  const t2 = t * t;
  const t3 = t2 * t;
  // Cubic Hermite basis.
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  return h00 * ys[i] + h10 * hseg * ms[i] + h01 * ys[i + 1] + h11 * hseg * ms[i + 1];
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
