// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Build script: give this crate's test, bench and example binaries a run-time
//! search path (`RUNPATH`) to the oneMKL library directories that
//! `omnidsp-onemkl-sys` linked against, so they load the libraries with no
//! `LD_LIBRARY_PATH` or `ldconfig` entry. Cargo does not pass a dependency's
//! link arguments on to its dependents, so the `-sys` crate publishes the
//! directories as `links` metadata and this script bakes them into this crate's
//! own binaries. Without the metadata (the fallback found `mkl_rt` on the
//! linker's default paths, or Windows) nothing is emitted.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in ["DEP_MKL_LIB_DIRS"] {
        let Some(dirs) = env::var_os(var) else {
            continue;
        };
        for dir in env::split_paths(&dirs) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
}
