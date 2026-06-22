// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Web Audio plumbing for the CQT visualiser.
//
// v1 is deliberately simple and main-thread (no AudioWorklet, no
// SharedArrayBuffer — keeps the demo on free static hosting).
// A ScriptProcessorNode taps the live audio graph and fills a circular PCM
// buffer; the render loop drains the newly-arrived samples each frame and feeds
// them to the streaming CQT (each sample consumed once).
// ScriptProcessorNode is deprecated but needs no SAB/worklet and matches the
// "audio callback fills a ring buffer" model — the AudioWorklet upgrade is
// explicitly deferred.

const SCRIPT_BUFFER = 4096;
const RING_CAPACITY = 1 << 15; // ~0.68 s at 48 kHz — ample inter-frame buffer
const CLIP_SECONDS = 6;

/** Source kind currently feeding the tap. */
export type Source = "none" | "clip" | "mic";

export class AudioEngine {
  readonly ctx: AudioContext;
  /** Capacity-bounded circular PCM buffer; the render loop drains new samples. */
  private readonly ring = new Float32Array(RING_CAPACITY);
  private writeIdx = 0;
  /** Samples written but not yet drained (0 … ring capacity). */
  private available = 0;

  private readonly tap: ScriptProcessorNode;
  private readonly master: GainNode;
  private readonly clip: AudioBuffer;

  private clipSource: AudioBufferSourceNode | null = null;
  private micStream: MediaStream | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;

  source: Source = "none";

  constructor(ctx: AudioContext) {
    this.ctx = ctx;

    this.master = ctx.createGain();
    this.master.gain.value = 1;
    this.master.connect(ctx.destination);

    this.tap = ctx.createScriptProcessor(SCRIPT_BUFFER, 1, 1);
    this.tap.onaudioprocess = (e: AudioProcessingEvent) => {
      const input = e.inputBuffer.getChannelData(0);
      // Pass audio straight through so the clip stays audible via `master`.
      e.outputBuffer.getChannelData(0).set(input);
      this.writeRing(input);
    };
    this.tap.connect(this.master);

    this.clip = makeSweepClip(ctx);
  }

  get sampleRate(): number {
    return this.ctx.sampleRate;
  }

  get running(): boolean {
    return this.source !== "none" && this.ctx.state === "running";
  }

  /** Start (or restart) the bundled looping sweep clip. Requires a gesture. */
  async startClip(): Promise<void> {
    await this.ctx.resume();
    this.stopSources();
    const src = this.ctx.createBufferSource();
    src.buffer = this.clip;
    src.loop = true;
    src.connect(this.tap);
    src.start();
    this.clipSource = src;
    this.master.gain.value = 1; // audible
    this.source = "clip";
  }

  /** Switch the source to live microphone input. Requires a gesture. */
  async useMic(): Promise<void> {
    await this.ctx.resume();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.stopSources();
    this.micStream = stream;
    this.micSource = this.ctx.createMediaStreamSource(stream);
    this.micSource.connect(this.tap);
    // Mute the monitor path to avoid acoustic feedback (tap still receives it).
    this.master.gain.value = 0;
    this.source = "mic";
  }

  /** Tear down all nodes and sources (called before rebuilding the engine). */
  dispose(): void {
    this.stopSources();
    this.tap.onaudioprocess = null;
    this.tap.disconnect();
    this.master.disconnect();
    this.source = "none";
  }

  /**
   * Drain up to `out.length` newly-arrived samples (oldest→newest) into `out`
   * and return the count written. Each sample is returned exactly once; if the
   * reader falls more than a ring behind, the oldest undrained samples are
   * dropped to stay current.
   */
  drainNewSamples(out: Float32Array): number {
    const cap = this.ring.length;
    const n = Math.min(this.available, out.length);
    const read = (this.writeIdx - this.available + cap) % cap;
    const head = Math.min(n, cap - read);
    out.set(this.ring.subarray(read, read + head), 0);
    if (n > head) out.set(this.ring.subarray(0, n - head), head);
    this.available -= n;
    return n;
  }

  private writeRing(chunk: Float32Array): void {
    const cap = this.ring.length;
    const n = chunk.length;
    if (n >= cap) {
      this.ring.set(chunk.subarray(n - cap));
      this.writeIdx = 0;
      this.available = cap;
      return;
    }
    const head = Math.min(n, cap - this.writeIdx);
    this.ring.set(chunk.subarray(0, head), this.writeIdx);
    if (n > head) this.ring.set(chunk.subarray(head), 0);
    this.writeIdx = (this.writeIdx + n) % cap;
    this.available = Math.min(this.available + n, cap);
  }

  private stopSources(): void {
    if (this.clipSource) {
      try {
        this.clipSource.stop();
      } catch {
        /* already stopped */
      }
      this.clipSource.disconnect();
      this.clipSource = null;
    }
    if (this.micSource) {
      this.micSource.disconnect();
      this.micSource = null;
    }
    if (this.micStream) {
      for (const t of this.micStream.getTracks()) t.stop();
      this.micStream = null;
    }
  }
}

/**
 * Synthesize a seamlessly-looping exponential sweep that ramps 20 Hz → 16 kHz
 * **and back down**. Reproducible, needs no asset file or permissions, and
 * exercises the full CQT range — the bright bin climbs the diagonal through
 * every octave, then descends.
 *
 * Sweeping up *and* down keeps the instantaneous frequency continuous at both
 * turning points (there is no 16 kHz → 20 Hz jump), so the loop has no broadband
 * seam click — there is no transient to fade away. The clip is then trimmed to a
 * whole number of carrier cycles so the wrap is phase-continuous too (both ends
 * land on a rising zero-crossing at 20 Hz).
 */
function makeSweepClip(ctx: AudioContext): AudioBuffer {
  const sr = ctx.sampleRate;
  const nMax = Math.floor(CLIP_SECONDS * sr);
  const f0 = 20;
  const f1 = 16000;
  const k = Math.log(f1 / f0);
  const half = Math.floor(nMax / 2);

  const all = new Float32Array(nMax);
  let phase = 0;
  let loopLen = 0; // last index after which the phase completed a whole cycle
  for (let i = 0; i < nMax; i++) {
    all[i] = 0.3 * Math.sin(phase);
    // Triangle in log-frequency: t ramps 0 → 1 (up) then 1 → 0 (down), so the
    // frequency reverses smoothly at 16 kHz and at 20 Hz — no jump.
    const t = i < half ? i / half : (nMax - 1 - i) / (nMax - 1 - half);
    const f = f0 * Math.exp(k * t);
    const before = phase;
    phase += (2 * Math.PI * f) / sr;
    // A rising zero-crossing (phase crossed a multiple of 2π): looping the clip
    // here makes the wrap phase-continuous, not just frequency-continuous.
    if (Math.floor(phase / (2 * Math.PI)) > Math.floor(before / (2 * Math.PI))) {
      loopLen = i + 1;
    }
  }

  const n = loopLen > 0 ? loopLen : nMax;
  const buf = ctx.createBuffer(1, n, sr);
  buf.getChannelData(0).set(all.subarray(0, n));
  return buf;
}
