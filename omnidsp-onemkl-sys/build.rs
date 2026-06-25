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
//!   3. Well-known install path (`/opt/intel/oneapi/mkl/latest`) — the default
//!      oneAPI layout, so a standard install links with no environment set. This
//!      mirrors the IPP `-sys` crate's well-known fallback (`ipp/latest`).
//!   4. Fallback — link `mkl_rt` dynamically and let the linker search its
//!      default paths.
//!
//! Note: detection here is **link-time** only (emitting `-L` so the linker
//! resolves `mkl_rt`). Loading `libmkl_rt.so` at **run time** is a separate
//! concern — the lib dir must be on the loader path via `LD_LIBRARY_PATH`, an
//! rpath, or `ldconfig`.

use std::env;
use std::path::Path;

fn main() {
    // Rebuild if the operator points us at a different MKL install.
    println!("cargo:rerun-if-env-changed=MKLROOT");

    if try_mklroot() || try_pkg_config() || try_well_known() {
        return;
    }

    // Fallback: assume `mkl_rt` is reachable on the linker's default paths.
    emit_link_lib();
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

    emit_search_paths(Path::new(&mklroot));
    emit_link_lib();
    // Even if none of the expected subdirectories exist, MKLROOT was set
    // explicitly: link `mkl_rt` and trust the operator's environment. Returning
    // true short-circuits the pkg-config / well-known / fallback paths whenever
    // MKLROOT is set, whether or not a search path was emitted.
    true
}

/// Step 2: probe `pkg-config` for the sequential LP64 dynamic MKL package.
fn try_pkg_config() -> bool {
    pkg_config::Config::new()
        .probe("mkl-dynamic-lp64-seq")
        .is_ok()
}

/// Step 3: probe the well-known oneAPI install path.
///
/// The default oneAPI layout is `/opt/intel/oneapi/mkl/latest`, so a standard
/// install links with no `MKLROOT` set — the parallel of the IPP `-sys` crate's
/// well-known fallback.
fn try_well_known() -> bool {
    let root = Path::new("/opt/intel/oneapi/mkl/latest");
    if !root.is_dir() {
        return false;
    }
    emit_search_paths(root);
    emit_link_lib();
    true
}

/// Emit `rustc-link-search` directives for each MKL library subdirectory of
/// `root` that exists (`lib/intel64` then `lib`).
fn emit_search_paths(root: &Path) {
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
}

/// Emit the `rustc-link-lib=dylib` directive for the MKL single dynamic library.
fn emit_link_lib() {
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
}
