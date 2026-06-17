// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Frequency → musical note. The CQT bins are constant-Q (geometric, one bin per
// semitone at 12 b/oct), so a bin *is* a note — labels are exact, no interpolation.

const NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

/** Nearest note name + octave for a frequency, e.g. 55 Hz → "A1", 440 → "A4". */
export function freqToNote(freq: number): string {
  // MIDI number: 69 + 12·log2(f/440); 69 = A4, 60 = C4.
  const midi = Math.round(69 + 12 * Math.log2(freq / 440));
  const name = NAMES[((midi % 12) + 12) % 12];
  const octave = Math.floor(midi / 12) - 1;
  return `${name}${octave}`;
}

/** Short Hz label: "55" below 1 kHz, "1.2k" above. */
export function hzLabel(freq: number): string {
  return freq >= 1000 ? `${(freq / 1000).toFixed(1)}k` : `${Math.round(freq)}`;
}
