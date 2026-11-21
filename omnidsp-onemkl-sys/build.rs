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
//! Run time: whichever directory linked `mkl_rt` is also baked into this
//! crate's own test binaries as a run-time search path (`RUNPATH`), so they
//! load `libmkl_rt.so` with no `LD_LIBRARY_PATH` or `ldconfig` entry; a set
//! `LD_LIBRARY_PATH` still takes precedence. Cargo does not pass link arguments
//! on to dependents, so the directories are also published as `links` metadata
//! (`DEP_MKL_LIB_DIRS`, a platform path list) for a direct dependent's build
//! script to bake into its own binaries. Applications further downstream
//! arrange their own run-time path. The fallback (linker default paths) and
//! Windows (DLLs resolve from `PATH`) emit none.

use std::env;
use std::path::{Path, PathBuf};

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

    emit_runtime_paths(&emit_search_paths(Path::new(&mklroot)));
    emit_link_lib();
    // Even if none of the expected subdirectories exist, MKLROOT was set
    // explicitly: link `mkl_rt` and trust the operator's environment. Returning
    // true short-circuits the pkg-config / well-known / fallback paths whenever
    // MKLROOT is set, whether or not a search path was emitted.
    true
}

/// Step 2: probe `pkg-config` for the sequential LP64 dynamic MKL package.
fn try_pkg_config() -> bool {
    let Ok(lib) = pkg_config::Config::new().probe("mkl-dynamic-lp64-seq") else {
        return false;
    };
    emit_runtime_paths(&lib.link_paths);
    true
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
    emit_runtime_paths(&emit_search_paths(root));
    emit_link_lib();
    true
}

/// Emit `rustc-link-search` directives for each MKL library subdirectory of
/// `root` that exists (`lib/intel64` then `lib`), and return those directories.
fn emit_search_paths(root: &Path) -> Vec<PathBuf> {
    // `CARGO_CFG_TARGET_OS` is the target we are building for, which is the
    // platform whose library layout we must match.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows" || cfg!(target_os = "windows");

    let candidates: &[&str] = if is_windows {
        &["lib"]
    } else {
        &["lib/intel64", "lib"]
    };

    let mut found = Vec::new();
    for sub in candidates {
        let dir = root.join(sub);
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            found.push(dir);
        }
    }
    found
}

/// Bake `dirs` into this crate's binaries as a run-time search path, and
/// publish them as `DEP_MKL_LIB_DIRS` for direct dependents (see the module
/// docs). Windows resolves DLLs from `PATH` and has no `RUNPATH`.
fn emit_runtime_paths(dirs: &[PathBuf]) {
    if dirs.is_empty() || env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows") {
        return;
    }
    for dir in dirs {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    if let Ok(joined) = env::join_paths(dirs) {
        println!("cargo:lib_dirs={}", joined.to_string_lossy());
    }
}

/// Emit the `rustc-link-lib=dylib` directive for the MKL single dynamic library.
fn emit_link_lib() {
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
}
