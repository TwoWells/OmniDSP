// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Dynamic IIR dispatch enum.

use std::fmt;

use omnidsp_core::error::Result;
use omnidsp_core::traits::iir::{Iir, IirPlan, IirSpec};
use omnidsp_rust::{RustIir, RustIirPlan};

/// Dispatch enum wrapping concrete IIR factories.
#[derive(Debug, Clone, Copy)]
pub enum DynIir {
    /// Pure Rust backend (scalar DF2T biquad cascade).
    Rust(RustIir),
}

/// Dispatch enum wrapping concrete IIR plans.
pub enum DynIirPlan<T> {
    /// Pure Rust backend plan.
    Rust(RustIirPlan<T>),
}

impl<T: fmt::Debug> fmt::Debug for DynIirPlan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rust(p) => f.debug_tuple("Rust").field(p).finish(),
        }
    }
}

macro_rules! impl_dyn_iir {
    ($t:ty) => {
        impl Iir<$t> for DynIir {
            type Plan = DynIirPlan<$t>;

            fn create_plan(&self, spec: &IirSpec<$t>) -> Result<DynIirPlan<$t>> {
                match self {
                    Self::Rust(d) => Ok(DynIirPlan::Rust(d.create_plan(spec)?)),
                }
            }
        }

        impl IirPlan<$t> for DynIirPlan<$t> {
            fn process(&mut self, input: &[$t], output: &mut [$t]) -> Result<()> {
                match self {
                    Self::Rust(p) => p.process(input, output),
                }
            }

            fn reset(&mut self) {
                match self {
                    Self::Rust(p) => p.reset(),
                }
            }
        }
    };
}

impl_dyn_iir!(f32);
impl_dyn_iir!(f64);

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect is the preferred idiom in tests")]
#[allow(
    clippy::float_cmp,
    reason = "dispatch output must match inner backend exactly"
)]
mod tests {
    use super::*;
    use omnidsp_core::types::BiquadSection;

    fn passthrough_section() -> BiquadSection<f64> {
        BiquadSection {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    #[test]
    fn dyn_iir_impulse_response() {
        let section = BiquadSection {
            b0: 0.5,
            b1: 0.3,
            b2: 0.1,
            a1: -0.5,
            a2: -0.1,
        };
        let spec = IirSpec::new(vec![section]);

        // Reference: RustIir directly.
        let rust_iir = RustIir::new();
        let mut rust_plan =
            Iir::<f64>::create_plan(&rust_iir, &spec).expect("RustIir plan creation");
        let input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut rust_output = [0.0; 8];
        rust_plan
            .process(&input, &mut rust_output)
            .expect("RustIir process");

        // DynIir dispatch.
        let dispatch = DynIir::Rust(RustIir::new());
        let mut dispatch_plan =
            Iir::<f64>::create_plan(&dispatch, &spec).expect("DynIir plan creation");
        let mut dispatch_output = [0.0; 8];
        dispatch_plan
            .process(&input, &mut dispatch_output)
            .expect("DynIir process");

        assert_eq!(
            dispatch_output, rust_output,
            "DynIir impulse response should match RustIir exactly"
        );
    }

    #[test]
    fn dyn_iir_reset() {
        let spec = IirSpec::new(vec![passthrough_section()]);
        let dyn_iir = DynIir::Rust(RustIir::new());
        let mut plan = Iir::<f64>::create_plan(&dyn_iir, &spec).expect("plan creation");

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output1 = [0.0; 5];
        let mut output2 = [0.0; 5];

        plan.process(&input, &mut output1).expect("first run");
        plan.reset();
        plan.process(&input, &mut output2).expect("second run");

        assert_eq!(
            output1, output2,
            "output after reset should match first run"
        );
    }
}
