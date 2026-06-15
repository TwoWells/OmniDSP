// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

// Compact polynomial approximation of the viridis colormap (Matt Zucker's fit
// of matplotlib's viridis). Takes t in [0, 1], returns 8-bit [r, g, b]. Cheap
// enough to evaluate per spectrogram cell every frame, no lookup table.

const C0 = [0.2777273272234177, 0.005407344544966578, 0.3340998053353061];
const C1 = [0.1050930431085774, 1.404613529898575, 1.384590162594685];
const C2 = [-0.3308618287255563, 0.214847559468213, 0.09509516302823659];
const C3 = [-4.634230498983486, -5.799100973351585, -19.33244095627987];
const C4 = [6.228269936347081, 14.17993336680509, 56.69055260068105];
const C5 = [4.776384997670288, -13.74514537774601, -65.35303263337234];
const C6 = [-5.435455855934631, 4.645852612178535, 26.3124352495832];

function channel(i: number, t: number): number {
  const v =
    C0[i] +
    t * (C1[i] + t * (C2[i] + t * (C3[i] + t * (C4[i] + t * (C5[i] + t * C6[i])))));
  const c = Math.round(v * 255);
  return c < 0 ? 0 : c > 255 ? 255 : c;
}

export function viridis(t: number): [number, number, number] {
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  return [channel(0, x), channel(1, x), channel(2, x)];
}
