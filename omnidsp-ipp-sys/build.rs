// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Build script: locate the Intel IPP signal-processing libraries and emit the
//! linker directives needed to link against them.
//!
//! IPP's `ipps` (signal processing) depends on `ippvm` (vector math) and
//! `ippcore` (runtime + CPU dispatch) at the ABI level; all three are linked
//! **dynamically** (static linking would vendor Intel's binaries and create
//! redistribution / license complications).
//!
//! Detection order:
//!   1. `IPPROOT` environment variable (Intel's standard convention) — emit a
//!      native search path for the platform's IPP library subdirectory.
//!   2. `IPP_ROOT` environment variable (alternative spelling).
//!   3. `pkg-config` probe of `ipp` (oneAPI ships a `.pc` file).
//!   4. Well-known install paths (`/opt/intel/oneapi/ipp/latest`).
//!
//! If none locate IPP, the script still emits the link directives and trusts
//! the linker's default search paths; a missing library then surfaces as a
//! clear link error (the crate cannot be built without IPP, by design).
//!
//! Run time: whichever directory linked the libraries is also baked into this
//! crate's own test binaries as a run-time search path (`RUNPATH`), so they
//! load `libipps.so` and its siblings with no `LD_LIBRARY_PATH` or `ldconfig`
//! entry; a set `LD_LIBRARY_PATH` still takes precedence. Cargo does not pass
//! link arguments on to dependents, so the directories are also published as
//! `links` metadata (`DEP_IPPS_LIB_DIRS`, a platform path list) for a direct
//! dependent's build script to bake into its own binaries. Applications further
//! downstream arrange their own run-time path. The fallback (linker default
//! paths) and Windows (DLLs resolve from `PATH`) emit none.

use std::env;
use std::path::{Path, PathBuf};

/// IPP signal-processing libraries, in dependency order (`ipps` first).
const IPP_LIBS: [&str; 3] = ["ipps", "ippvm", "ippcore"];

fn main() {
    // Rebuild if the operator points us at a different IPP install.
    println!("cargo:rerun-if-env-changed=IPPROOT");
    println!("cargo:rerun-if-env-changed=IPP_ROOT");

    if try_ipproot() || try_pkg_config() || try_well_known() {
        return;
    }

    // Fallback: assume the IPP libraries are reachable on the linker's default
    // paths. A missing library surfaces as a link error with the library name.
    emit_link_libs();
}

/// Steps 1–2: honor an explicit `IPPROOT` / `IPP_ROOT`.
///
/// On Linux the libraries live under `$IPPROOT/lib/intel64` (oneAPI) or
/// `$IPPROOT/lib`; emit a search path for each that exists. On Windows the
/// import libraries live in `%IPPROOT%\lib` and the DLLs resolve at runtime
/// from `%IPPROOT%\bin` on the machine PATH.
fn try_ipproot() -> bool {
    let Some(root) = env::var("IPPROOT")
        .ok()
        .or_else(|| env::var("IPP_ROOT").ok())
        .filter(|s| !s.is_empty())
    else {
        return false;
    };

    emit_runtime_paths(&emit_search_paths(Path::new(&root)));
    emit_link_libs();
    // IPPROOT was set explicitly: trust the operator's environment and
    // short-circuit pkg-config / well-known probing whether or not a search
    // path was emitted.
    true
}

/// Step 3: probe `pkg-config` for the `ipp` package (oneAPI ships it).
fn try_pkg_config() -> bool {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for lib in IPP_LIBS {
        let Ok(found) = pkg_config::Config::new().probe(lib) else {
            return false;
        };
        for dir in found.link_paths {
            if !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    emit_runtime_paths(&dirs);
    true
}

/// Step 4: probe the well-known oneAPI install path.
fn try_well_known() -> bool {
    let root = Path::new("/opt/intel/oneapi/ipp/latest");
    if !root.is_dir() {
        return false;
    }
    emit_runtime_paths(&emit_search_paths(root));
    emit_link_libs();
    true
}

/// Emit `rustc-link-search` directives for each IPP library subdirectory of
/// `root` that exists (`lib/intel64` then `lib`), and return those directories.
fn emit_search_paths(root: &Path) -> Vec<PathBuf> {
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
/// publish them as `DEP_IPPS_LIB_DIRS` for direct dependents (see the module
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

/// Emit `rustc-link-lib=dylib` directives for `ipps`, `ippvm`, and `ippcore`.
fn emit_link_libs() {
    for lib in IPP_LIBS {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
