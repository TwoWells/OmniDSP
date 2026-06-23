// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Build script: locate the Intel oneMKL Single Dynamic Library (`mkl_rt`)
//! and emit the linker directives needed to link against it.
//!
//! Detection order:
//!   1. `MKLROOT` environment variable — emit a native search path for the
//!      platform's MKL library subdirectory and link `mkl_rt` dynamically.
//!   2. `pkg-config` probe of `mkl-dynamic-lp64-seq` (oneMKL installed via
//!      system packages or oneAPI ships this `.pc` file).
//!   3. Fallback — link `mkl_rt` dynamically and let the linker search its
//!      default paths.

use std::env;
use std::path::Path;

fn main() {
    // Rebuild if the operator points us at a different MKL install.
    println!("cargo:rerun-if-env-changed=MKLROOT");

    if try_mklroot() {
        return;
    }

    if try_pkg_config() {
        return;
    }

    // Fallback: assume `mkl_rt` is reachable on the linker's default paths.
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
}

/// Step 1: honor an explicit `MKLROOT`.
///
/// On Linux the libraries live under both `$MKLROOT/lib/intel64` and
/// `$MKLROOT/lib`; the oneAPI CI image presets `MKLROOT` and ships libs in
/// both locations, so emit a search path for each that exists. On Windows the
/// libraries live in `%MKLROOT%\lib` (not `lib\intel64`); its DLL resolves at
/// runtime from `%MKLROOT%\bin` on the machine PATH.
fn try_mklroot() -> bool {
    let Ok(mklroot) = env::var("MKLROOT") else {
        return false;
    };
    if mklroot.is_empty() {
        return false;
    }

    let root = Path::new(&mklroot);

    // `CARGO_CFG_TARGET_OS` is the target we are building for, which is the
    // platform whose library layout we must match.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows" || cfg!(target_os = "windows");

    let candidates: &[&str] = if is_windows {
        &["lib"]
    } else {
        &["lib/intel64", "lib"]
    };

    for sub in candidates {
        let dir = root.join(sub);
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    // Even if none of the expected subdirectories exist, MKLROOT was set
    // explicitly: link `mkl_rt` and trust the operator's environment. Returning
    // true short-circuits the pkg-config / fallback paths whenever MKLROOT is
    // set, whether or not a search path was emitted.
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
    true
}

/// Step 2: probe `pkg-config` for the sequential LP64 dynamic MKL package.
fn try_pkg_config() -> bool {
    pkg_config::Config::new()
        .probe("mkl-dynamic-lp64-seq")
        .is_ok()
}
