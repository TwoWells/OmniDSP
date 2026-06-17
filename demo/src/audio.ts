// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Web Audio plumbing for the CQT visualiser.
//
// v1 is deliberately simple and main-thread (no AudioWorklet, no
// SharedArrayBuffer — keeps the demo on free static hosting, see DEMO-00).
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
 * Synthesize a looping exponential sweep 20 Hz → 16 kHz. Reproducible, needs no
 * asset file or permissions, and exercises the full CQT range — the bright bin
 * climbs diagonally through every octave, an immediate visual sanity check.
 */
function makeSweepClip(ctx: AudioContext): AudioBuffer {
  const sr = ctx.sampleRate;
  const n = Math.floor(CLIP_SECONDS * sr);
  const buf = ctx.createBuffer(1, n, sr);
  const d = buf.getChannelData(0);

  const f0 = 20;
  const f1 = 16000;
  const k = Math.log(f1 / f0);
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const f = f0 * Math.exp((k * i) / n);
    phase += (2 * Math.PI * f) / sr;
    d[i] = 0.3 * Math.sin(phase);
  }

  // Short fades to avoid a click at the loop seam.
  const fade = Math.floor(0.01 * sr);
  for (let i = 0; i < fade; i++) {
    const g = i / fade;
    d[i] *= g;
    d[n - 1 - i] *= g;
  }
  return buf;
}
