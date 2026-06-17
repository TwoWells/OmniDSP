// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Parks–McClellan / Remez exchange for linear-phase FIR design.
//!
//! Computes the minimax-optimal (equiripple) symmetric FIR impulse response for
//! a set of frequency bands, each with a desired amplitude and an error weight.
//! This is the engine behind [`FirMethod::Equiripple`](super::fir::FirMethod);
//! the unit/weight mapping from the public dB specs lives in
//! [`fir`](super::fir).
//!
//! # Algorithm
//!
//! For a length-`N` symmetric (linear-phase) filter the frequency response is
//! `H(ω) = e^{-jω(N-1)/2} · A(ω)`, with a real amplitude `A(ω)`.  For an
//! odd length (Type I) `A(ω) = Σ_{k=0}^{L} a[k] cos(kω)` with `L = (N-1)/2`.
//! For an even length (Type II) `A(ω) = cos(ω/2) · Σ_{k=0}^{L} c[k] cos(kω)`
//! with `L = N/2 − 1`; folding the `cos(ω/2)` factor into the desired response
//! and weight reduces both cases to the same cosine-basis minimax problem
//! (Oppenheim & Schafer, *Discrete-Time Signal Processing*, §7.7).
//!
//! The Remez exchange then iterates:
//!
//! 1. Evaluate the weighted error on a candidate **extremal set** of `L + 2`
//!    frequencies via barycentric Lagrange interpolation (the closed-form
//!    deviation `δ` plus the interpolated amplitude).
//! 2. Scan a dense frequency grid for the new local extrema of the error and
//!    replace the extremal set with them.
//! 3. Stop when the extremal error magnitudes have equalized (the alternation
//!    is achieved) to within a tolerance, or fail if the iteration cap is hit.
//!
//! All arithmetic is `f64`.  The routine is **total and bounded**: it performs
//! at most [`MAX_ITERATIONS`] exchanges and returns
//! [`Error::DidNotConverge`](crate::error::Error) rather than
//! looping forever, and contains no `unwrap`/`panic`/`unsafe`.

use std::f64::consts::PI;

use crate::error::{Error, Result};

/// Maximum number of Remez exchange iterations before declaring failure.
///
/// Convergence is typically reached in well under a dozen exchanges; this cap
/// only guards against a pathological spec that never stabilizes.
const MAX_ITERATIONS: usize = 64;

/// Density of the frequency grid per basis function (grid points ≈ `GRID_DENSITY · L`).
///
/// `16` is the conventional Parks–McClellan oversampling factor (scipy's
/// default) — dense enough that the discrete grid maxima track the true
/// extrema.
const GRID_DENSITY: usize = 16;

/// Relative tolerance on the equalized extremal-error magnitudes that signals
/// convergence (the classic `(max − min) / max < tol` alternation test).
const CONVERGENCE_TOL: f64 = 1e-8;

/// A frequency band for the Remez design.
///
/// Frequencies are normalized to the sample rate, in `[0, 0.5]`.  Within the
/// band the design targets amplitude `desired` with relative error weight
/// `weight`; the transition gaps between bands are unconstrained.
#[derive(Debug, Clone, Copy)]
pub struct Band {
    /// Lower band edge (normalized frequency, `0..=0.5`).
    pub low: f64,
    /// Upper band edge (normalized frequency, `0..=0.5`).
    pub high: f64,
    /// Desired amplitude in this band.
    pub desired: f64,
    /// Error weight in this band (larger ⇒ smaller realized ripple here).
    pub weight: f64,
}

/// Filter symmetry / length parity (Parks–McClellan filter types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symmetry {
    /// Even-symmetric, odd length (Type I) — `A(ω) = Σ a[k] cos(kω)`.
    Type1,
    /// Even-symmetric, even length (Type II) — `A(ω) = cos(ω/2) Σ c[k] cos(kω)`.
    Type2,
}

/// Design a symmetric (even-symmetric, linear-phase) equiripple FIR.
///
/// Returns `num_taps` impulse-response coefficients minimizing the maximum
/// weighted deviation `W(f)·(D(f) − A(f))` over the union of `bands`, where
/// `D`/`W` are the per-band `desired`/`weight`.  Only even-symmetric responses
/// (Type I for odd `num_taps`, Type II for even `num_taps`) are produced, which
/// covers lowpass, highpass (odd length), bandpass and the half-band path.
///
/// # Errors
///
/// Returns [`Error::InvalidSpec`] if `num_taps < 3`, `bands` is empty, any band
/// is degenerate or out of `[0, 0.5]`, or a band weight is non-positive.
/// Returns [`Error::DidNotConverge`] if the exchange does not stabilize within
/// [`MAX_ITERATIONS`] (a spec whose alternation is unreachable at this length).
pub fn design(num_taps: usize, bands: &[Band]) -> Result<Vec<f64>> {
    if num_taps < 3 {
        return Err(Error::InvalidSpec(
            "equiripple design requires at least 3 taps".into(),
        ));
    }
    if bands.is_empty() {
        return Err(Error::InvalidSpec(
            "equiripple design requires at least one band".into(),
        ));
    }
    for b in bands {
        if !(0.0..=0.5).contains(&b.low) || !(0.0..=0.5).contains(&b.high) || b.high <= b.low {
            return Err(Error::InvalidSpec(format!(
                "band edges must satisfy 0 <= low < high <= 0.5 (got {}..{})",
                b.low, b.high
            )));
        }
        if b.weight <= 0.0 {
            return Err(Error::InvalidSpec(
                "equiripple band weight must be positive".into(),
            ));
        }
    }

    // Symmetry / number of cosine basis functions L (response = Σ_{k=0}^{L-1+1}).
    // Type I: num_taps = 2L + 1 → L = (num_taps - 1) / 2; basis size L + 1.
    // Type II: num_taps = 2L + 2 → L = num_taps / 2 - 1; basis size L + 1.
    let (symmetry, num_basis) = if num_taps % 2 == 1 {
        (Symmetry::Type1, (num_taps - 1) / 2 + 1)
    } else {
        (Symmetry::Type2, num_taps / 2)
    };

    // The minimax problem has `num_basis + 1` extremal frequencies (alternations).
    let num_extrema = num_basis + 1;

    let grid = build_grid(bands, num_basis, symmetry)?;
    if grid.len() < num_extrema {
        return Err(Error::InvalidSpec(
            "frequency grid too small for the requested filter length".into(),
        ));
    }

    let amplitude = remez_exchange(&grid, num_extrema)?;

    Ok(synthesize(&amplitude, num_taps, symmetry))
}

// ─── Dense frequency grid ─────────────────────────────────────────────

/// One grid point: angular frequency, the (possibly folded) desired amplitude,
/// and the (possibly folded) error weight.
struct GridPoint {
    omega: f64,
    desired: f64,
    weight: f64,
}

/// Build the dense frequency grid over the union of `bands`.
///
/// For Type II the `cos(ω/2)` amplitude factor is folded into the desired
/// response and weight (`D ← D / cos(ω/2)`, `W ← W · cos(ω/2)`), so the Remez
/// core sees a plain cosine-basis problem.  The grid omits any point where the
/// fold would divide by ~0 (ω = π, the Type II forced zero).
fn build_grid(bands: &[Band], num_basis: usize, symmetry: Symmetry) -> Result<Vec<GridPoint>> {
    let target = (GRID_DENSITY * num_basis).max(num_basis + 1);

    let total_width: f64 = bands.iter().map(|b| b.high - b.low).sum();
    if total_width <= 0.0 {
        return Err(Error::InvalidSpec(
            "equiripple bands have zero total width".into(),
        ));
    }

    let mut grid = Vec::new();
    for b in bands {
        // Allocate grid points to this band proportionally to its width.
        let frac = (b.high - b.low) / total_width;
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "grid point count is a small positive integer"
        )]
        let points = ((target as f64 * frac).ceil() as usize).max(2);
        for i in 0..points {
            #[allow(clippy::cast_precision_loss, reason = "grid index is small")]
            let f = b.low + (b.high - b.low) * (i as f64) / ((points - 1) as f64);
            let omega = 2.0 * PI * f;
            let (desired, weight) = match symmetry {
                Symmetry::Type1 => (b.desired, b.weight),
                Symmetry::Type2 => {
                    let half = (omega / 2.0).cos();
                    if half.abs() < 1e-9 {
                        // ω ≈ π: Type II forces a zero here; skip the point.
                        continue;
                    }
                    (b.desired / half, b.weight * half)
                }
            };
            grid.push(GridPoint {
                omega,
                desired,
                weight,
            });
        }
    }

    // Strictly increasing in ω (deduplicate shared band edges).
    grid.sort_by(|a, b| {
        a.omega
            .partial_cmp(&b.omega)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    grid.dedup_by(|a, b| (a.omega - b.omega).abs() < 1e-12);

    Ok(grid)
}

// ─── The cosine amplitude on the extremal set ─────────────────────────

/// The converged cosine-amplitude solution: the extremal frequencies, the
/// amplitude values interpolated there, and the per-extremum sign weights used
/// by the barycentric interpolation.
struct Amplitude {
    /// Extremal angular frequencies.
    omega: Vec<f64>,
    /// Amplitude `A(ω)` at each extremal frequency.
    values: Vec<f64>,
    /// Barycentric interpolation weights at the extremal frequencies.
    bary: Vec<f64>,
}

impl Amplitude {
    /// Evaluate the interpolated amplitude `A(ω)` at an arbitrary angular `w`.
    fn eval(&self, w: f64) -> f64 {
        let x = w.cos();
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &oi) in self.omega.iter().enumerate() {
            let xi = oi.cos();
            let d = x - xi;
            if d.abs() < 1e-12 {
                // Landed exactly on a node.
                return self.values[i];
            }
            let t = self.bary[i] / d;
            num += t * self.values[i];
            den += t;
        }
        if den.abs() < 1e-300 {
            return 0.0;
        }
        num / den
    }
}

/// Run the Remez exchange to convergence on `grid`, returning the converged
/// cosine amplitude.  `num_extrema` is the alternation count (`L + 2`).
fn remez_exchange(grid: &[GridPoint], num_extrema: usize) -> Result<Amplitude> {
    // Initial extremal set: evenly spaced grid indices.
    let mut ext: Vec<usize> = (0..num_extrema)
        .map(|i| {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "extremal index is a small in-range integer"
            )]
            let idx = (i * (grid.len() - 1)) / (num_extrema - 1);
            idx
        })
        .collect();

    let mut last_deviation = f64::INFINITY;

    for _ in 0..MAX_ITERATIONS {
        let amplitude = solve_on_set(grid, &ext, num_extrema)?;

        // Re-scan the dense grid for the new extrema of the weighted error.
        let next = find_extrema(grid, &amplitude, num_extrema);
        if next.len() != num_extrema {
            return Err(Error::DidNotConverge(format!(
                "extremal set collapsed to {} of {num_extrema} frequencies",
                next.len()
            )));
        }

        // Convergence: the magnitudes of the weighted error at the candidate
        // extrema have equalized.
        let mut e_min = f64::INFINITY;
        let mut e_max: f64 = 0.0;
        for &idx in &next {
            let g = &grid[idx];
            let err = (g.weight * (g.desired - amplitude.eval(g.omega))).abs();
            e_min = e_min.min(err);
            e_max = e_max.max(err);
        }
        let converged = e_max > 0.0 && (e_max - e_min) / e_max < CONVERGENCE_TOL;

        let deviation = e_max;
        let stalled = (deviation - last_deviation).abs() <= CONVERGENCE_TOL * deviation.max(1e-12)
            && next == ext;
        last_deviation = deviation;
        ext = next;

        if converged || stalled {
            return Ok(amplitude);
        }
    }

    Err(Error::DidNotConverge(format!(
        "Remez exchange did not converge in {MAX_ITERATIONS} iterations"
    )))
}

/// Solve the interpolation problem on a fixed extremal set: compute the
/// deviation `δ` and the amplitude values at the extrema (the linear system at
/// the heart of one Remez iteration), returned as a barycentric interpolant.
fn solve_on_set(grid: &[GridPoint], ext: &[usize], num_extrema: usize) -> Result<Amplitude> {
    // Barycentric weights γ_k = 1 / Π_{i≠k} (x_k − x_i), x = cos(ω).
    let xs: Vec<f64> = ext.iter().map(|&i| grid[i].omega.cos()).collect();
    let mut gamma = vec![0.0_f64; num_extrema];
    for k in 0..num_extrema {
        let mut prod = 1.0;
        for (i, &xi) in xs.iter().enumerate() {
            if i != k {
                let d = xs[k] - xi;
                if d.abs() < 1e-15 {
                    return Err(Error::DidNotConverge(
                        "extremal frequencies coincided (singular interpolation)".into(),
                    ));
                }
                prod *= d;
            }
        }
        gamma[k] = 1.0 / prod;
    }

    // Closed-form deviation:  δ = Σ γ_k D_k / Σ (γ_k (−1)^k / W_k).
    let mut num = 0.0;
    let mut den = 0.0;
    for k in 0..num_extrema {
        let g = &grid[ext[k]];
        num += gamma[k] * g.desired;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        den += gamma[k] * sign / g.weight;
    }
    if den.abs() < 1e-300 {
        return Err(Error::DidNotConverge(
            "deviation denominator vanished".into(),
        ));
    }
    let deviation = num / den;

    // Amplitude at each extremum: A_k = D_k − (−1)^k δ / W_k.
    let mut values = vec![0.0_f64; num_extrema];
    for k in 0..num_extrema {
        let g = &grid[ext[k]];
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        values[k] = g.desired - sign * deviation / g.weight;
    }

    // The amplitude is interpolated through the FIRST `num_extrema − 1` nodes
    // (a degree-`L` cosine polynomial is fixed by `L + 1` points); the last
    // node only carried the deviation sign.  Use the leading nodes for the
    // barycentric interpolant.
    let bary_nodes = num_extrema - 1;
    let xs_b = &xs[..bary_nodes];
    let mut bary = vec![0.0_f64; bary_nodes];
    for k in 0..bary_nodes {
        let mut prod = 1.0;
        for (i, &xi) in xs_b.iter().enumerate() {
            if i != k {
                prod *= xs_b[k] - xi;
            }
        }
        if prod.abs() < 1e-300 {
            return Err(Error::DidNotConverge("barycentric weight overflow".into()));
        }
        bary[k] = 1.0 / prod;
    }

    Ok(Amplitude {
        omega: ext[..bary_nodes].iter().map(|&i| grid[i].omega).collect(),
        values: values[..bary_nodes].to_vec(),
        bary,
    })
}

/// Scan the grid for the local extrema of the weighted error, then reduce them
/// to exactly `num_extrema` sign-alternating extrema (the new extremal set).
///
/// Candidates are grid turning points (where the discrete error derivative
/// reverses) together with every **band boundary** — the first and last grid
/// point of each band, which are always potential extrema (the transition gaps
/// between bands are unconstrained and not scanned).  Adjacent candidates of
/// the same error sign are collapsed to the larger-magnitude one, and any
/// surplus is trimmed by dropping the smallest-magnitude extremum (and its
/// weaker neighbour, to preserve the alternation parity).
fn find_extrema(grid: &[GridPoint], amplitude: &Amplitude, num_extrema: usize) -> Vec<usize> {
    let err = |i: usize| -> f64 {
        let g = &grid[i];
        g.weight * (g.desired - amplitude.eval(g.omega))
    };

    let n = grid.len();

    // Band boundaries: a grid frequency jump marks a transition gap. The point
    // before the jump ends one band; the point after starts the next. Both
    // endpoints of the whole grid are boundaries too.
    let mut is_boundary = vec![false; n];
    is_boundary[0] = true;
    is_boundary[n - 1] = true;
    let mut max_step = 0.0_f64;
    for i in 1..n {
        max_step = max_step.max(grid[i].omega - grid[i - 1].omega);
    }
    let gap_threshold = max_step * 1.5;
    for i in 1..n {
        if grid[i].omega - grid[i - 1].omega > gap_threshold {
            is_boundary[i - 1] = true;
            is_boundary[i] = true;
        }
    }

    // Collect candidate extrema: boundaries plus interior turning points.
    let mut candidates: Vec<usize> = Vec::new();
    for (i, &boundary) in is_boundary.iter().enumerate() {
        if boundary {
            candidates.push(i);
            continue;
        }
        let e = err(i);
        let dl = e - err(i - 1);
        let dr = err(i + 1) - e;
        // A turning point: the error stops rising and starts falling (or vice
        // versa). Include flats conservatively.
        if (dl >= 0.0 && dr <= 0.0) || (dl <= 0.0 && dr >= 0.0) {
            candidates.push(i);
        }
    }
    candidates.dedup();

    // Collapse runs of same-sign candidates to the largest-magnitude member,
    // so the survivors strictly alternate in sign.
    let mut alternating: Vec<usize> = Vec::with_capacity(candidates.len());
    for &idx in &candidates {
        if let Some(&last) = alternating.last() {
            let same_sign = err(idx).signum() == err(last).signum();
            if same_sign {
                if err(idx).abs() > err(last).abs() {
                    let n_alt = alternating.len();
                    alternating[n_alt - 1] = idx;
                }
                continue;
            }
        }
        alternating.push(idx);
    }

    // Trim surplus extrema, removing the smallest first while keeping the count
    // parity correct (extrema must alternate, so we drop the weaker of an
    // adjacent pair).
    while alternating.len() > num_extrema {
        // Find the index of the globally smallest-|error| extremum.
        let mut min_pos = 0;
        let mut min_mag = f64::INFINITY;
        for (pos, &idx) in alternating.iter().enumerate() {
            let m = err(idx).abs();
            if m < min_mag {
                min_mag = m;
                min_pos = pos;
            }
        }
        // Remove it together with the weaker of its neighbours so the remaining
        // sequence still alternates; if it's an endpoint, drop just it.
        let left_mag = if min_pos > 0 {
            err(alternating[min_pos - 1]).abs()
        } else {
            f64::INFINITY
        };
        let right_mag = if min_pos + 1 < alternating.len() {
            err(alternating[min_pos + 1]).abs()
        } else {
            f64::INFINITY
        };
        if alternating.len() - 1 == num_extrema
            || (left_mag.is_infinite() && right_mag.is_infinite())
        {
            alternating.remove(min_pos);
        } else if left_mag <= right_mag {
            // Remove min_pos and its left neighbour (drop higher index first).
            alternating.remove(min_pos);
            alternating.remove(min_pos - 1);
        } else {
            alternating.remove(min_pos + 1);
            alternating.remove(min_pos);
        }
    }

    alternating
}

// ─── Impulse-response synthesis ───────────────────────────────────────

/// Sample the converged amplitude back into impulse-response coefficients.
///
/// Samples `A(ω)` at `L + 1` uniformly-spaced frequencies and inverts the
/// cosine series to recover the half coefficients, then mirrors them into the
/// symmetric impulse response (folding back the `cos(ω/2)` factor for Type II).
fn synthesize(amplitude: &Amplitude, num_taps: usize, symmetry: Symmetry) -> Vec<f64> {
    let l = match symmetry {
        Symmetry::Type1 => (num_taps - 1) / 2,
        Symmetry::Type2 => num_taps / 2 - 1,
    };

    // Sample the amplitude at m = 0..=L equally-spaced points in [0, π].
    #[allow(clippy::cast_precision_loss, reason = "sample index is small")]
    let sample = |m: usize| -> f64 {
        let omega = PI * (m as f64) / (l as f64).max(1.0);
        let a = amplitude.eval(omega);
        match symmetry {
            Symmetry::Type1 => a,
            // Undo the folded cos(ω/2) factor to recover the true amplitude.
            Symmetry::Type2 => a * (omega / 2.0).cos(),
        }
    };

    // Inverse DCT-I → cosine-series coefficients a[k] from the amplitude
    // samples A_m = A(πm/L), m = 0..=L:
    //   a_k = (c_k / L) · [ A_0/2 + (−1)^k A_L/2 + Σ_{m=1}^{L−1} A_m cos(πmk/L) ]
    // with c_k = 1 for k ∈ {0, L} and c_k = 2 otherwise.
    let count = l + 1;
    #[allow(
        clippy::cast_precision_loss,
        reason = "L is small enough to be exact in f64"
    )]
    let lf = (l as f64).max(1.0);
    let mut coeffs = vec![0.0_f64; count];
    let samples: Vec<f64> = (0..count).map(sample).collect();
    for (k, ck) in coeffs.iter_mut().enumerate() {
        // Endpoint-weighted accumulation over m.
        let mut acc = samples[0] / 2.0;
        for (m, &s) in samples.iter().enumerate().take(count).skip(1) {
            #[allow(
                clippy::cast_precision_loss,
                reason = "indices m, k are small enough to be exact in f64"
            )]
            let arg = PI * (m as f64) * (k as f64) / lf;
            let weight = if m == count - 1 { 0.5 } else { 1.0 };
            acc += weight * s * arg.cos();
        }
        let c_k = if k == 0 || k == l { 1.0 } else { 2.0 };
        *ck = c_k * acc / lf;
    }

    impulse_from_cosine(&coeffs, num_taps, symmetry)
}

/// Build the symmetric impulse response from the half cosine-series
/// coefficients, per the Type I / Type II amplitude relations.
fn impulse_from_cosine(coeffs: &[f64], num_taps: usize, symmetry: Symmetry) -> Vec<f64> {
    let mut h = vec![0.0_f64; num_taps];
    match symmetry {
        Symmetry::Type1 => {
            // A(ω) = Σ a[k] cos(kω);  h[center] = a[0], h[center ± k] = a[k]/2.
            let center = (num_taps - 1) / 2;
            h[center] = coeffs[0];
            for (k, &ak) in coeffs.iter().enumerate().skip(1) {
                let v = ak / 2.0;
                if center >= k {
                    h[center - k] += v;
                }
                if center + k < num_taps {
                    h[center + k] += v;
                }
            }
        }
        Symmetry::Type2 => {
            // A(ω) = Σ b[n] cos((n − 1/2)ω), n = 1..=N/2; the b[n] relate to the
            // folded cosine coefficients c[k] by  b[1] = c[0] + c[1]/2,
            // b[n] = (c[n-1] + c[n]) / 2 (2 ≤ n ≤ N/2 − 1), b[N/2] = c[N/2−1]/2,
            // and h[N/2 − n] = h[N/2 − 1 + n] = b[n] / 2.
            let half = num_taps / 2;
            let mut b = vec![0.0_f64; half + 1]; // 1-indexed usage
            b[1] = coeffs[0] + coeffs.get(1).copied().unwrap_or(0.0) / 2.0;
            for (n, bn) in b.iter_mut().enumerate().take(half).skip(2) {
                let cprev = coeffs.get(n - 1).copied().unwrap_or(0.0);
                let cn = coeffs.get(n).copied().unwrap_or(0.0);
                *bn = f64::midpoint(cprev, cn);
            }
            b[half] = coeffs.get(half - 1).copied().unwrap_or(0.0) / 2.0;
            for (n, &bn) in b.iter().enumerate().take(half + 1).skip(1) {
                let v = bn / 2.0;
                let lo = half - n;
                let hi = half - 1 + n;
                if lo < num_taps {
                    h[lo] += v;
                }
                if hi < num_taps {
                    h[hi] += v;
                }
            }
        }
    }
    h
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
mod tests {
    use super::{Band, design};

    /// Evaluate the FIR magnitude response at normalized frequency `f`.
    fn mag(taps: &[f64], f: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * f;
        let re: f64 = taps
            .iter()
            .enumerate()
            .map(|(n, &h)| {
                #[allow(clippy::cast_precision_loss, reason = "small index")]
                let nn = n as f64;
                h * (omega * nn).cos()
            })
            .sum();
        let im: f64 = taps
            .iter()
            .enumerate()
            .map(|(n, &h)| {
                #[allow(clippy::cast_precision_loss, reason = "small index")]
                let nn = n as f64;
                h * (omega * nn).sin()
            })
            .sum();
        re.hypot(im)
    }

    #[test]
    fn type1_lowpass_is_symmetric() {
        let bands = [
            Band {
                low: 0.0,
                high: 0.1,
                desired: 1.0,
                weight: 1.0,
            },
            Band {
                low: 0.15,
                high: 0.5,
                desired: 0.0,
                weight: 10.0,
            },
        ];
        let taps = design(31, &bands).expect("type 1 lowpass");
        assert_eq!(taps.len(), 31, "should have 31 taps");
        for i in 0..taps.len() / 2 {
            let j = taps.len() - 1 - i;
            assert!(
                (taps[i] - taps[j]).abs() < 1e-12,
                "type 1 taps should be symmetric at {i}: {} vs {}",
                taps[i],
                taps[j]
            );
        }
    }

    #[test]
    fn type2_lowpass_is_symmetric() {
        let bands = [
            Band {
                low: 0.0,
                high: 0.1,
                desired: 1.0,
                weight: 1.0,
            },
            Band {
                low: 0.15,
                high: 0.5,
                desired: 0.0,
                weight: 10.0,
            },
        ];
        let taps = design(32, &bands).expect("type 2 lowpass");
        assert_eq!(taps.len(), 32, "should have 32 taps");
        for i in 0..taps.len() / 2 {
            let j = taps.len() - 1 - i;
            assert!(
                (taps[i] - taps[j]).abs() < 1e-12,
                "type 2 taps should be symmetric at {i}: {} vs {}",
                taps[i],
                taps[j]
            );
        }
        // Type II is exercised only here (the scipy goldens are all odd-length
        // Type I), so verify it actually realizes the spec, not just symmetry.
        let g_dc = mag(&taps, 0.0);
        assert!(
            (g_dc - 1.0).abs() < 0.1,
            "type 2 passband gain should be near unity, got {g_dc}"
        );
        for &f in &[0.2, 0.3, 0.4] {
            let g = mag(&taps, f);
            assert!(
                g < 0.12,
                "type 2 stopband gain at {f} should be small, got {g}"
            );
        }
    }

    #[test]
    fn lowpass_passband_and_stopband_meet_spec() {
        let bands = [
            Band {
                low: 0.0,
                high: 0.1,
                desired: 1.0,
                weight: 1.0,
            },
            Band {
                low: 0.15,
                high: 0.5,
                desired: 0.0,
                weight: 100.0,
            },
        ];
        let taps = design(63, &bands).expect("lowpass");
        // Passband gain near unity.
        let g_dc = mag(&taps, 0.0);
        assert!(
            (g_dc - 1.0).abs() < 0.05,
            "passband gain should be near unity, got {g_dc}"
        );
        // Stopband strongly attenuated (weight 100 in stop → deep null).
        for &f in &[0.2, 0.3, 0.4, 0.49] {
            let g = mag(&taps, f);
            assert!(g < 0.05, "stopband gain at {f} should be small, got {g}");
        }
    }

    #[test]
    fn invalid_specs_are_rejected() {
        assert!(design(2, &[]).is_err(), "too few taps must be rejected");
        let bad_band = [Band {
            low: 0.3,
            high: 0.1,
            desired: 1.0,
            weight: 1.0,
        }];
        assert!(
            design(15, &bad_band).is_err(),
            "high <= low band must be rejected"
        );
        let bad_weight = [Band {
            low: 0.0,
            high: 0.2,
            desired: 1.0,
            weight: 0.0,
        }];
        assert!(
            design(15, &bad_weight).is_err(),
            "non-positive weight must be rejected"
        );
    }
}
