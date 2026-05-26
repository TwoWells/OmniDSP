// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Two Wells <contact@twowells.dev>

//! Build script for `highway-sys`: compiles Highway runtime and the `OmniDSP` shim.

use std::path::PathBuf;

fn main() {
    // Highway source lives in a git submodule at highway-sys/highway/.
    let highway_dir = PathBuf::from("highway");
    let shim_dir = PathBuf::from("shim");

    // Highway runtime support sources (needed for dynamic dispatch).
    let hwy_sources: Vec<PathBuf> = [
        "hwy/targets.cc",
        "hwy/per_target.cc",
        "hwy/abort.cc",
        "hwy/aligned_allocator.cc",
        "hwy/print.cc",
    ]
    .iter()
    .map(|s| highway_dir.join(s))
    .collect();

    // Build Highway runtime as a static library.
    //
    // HWY_COMPILE_ONLY_STATIC: only compile for the target selected by
    // compiler flags (i.e. -mcpu=native). This avoids conflicts between
    // -mcpu=native and foreach_target's per-ISA pragmas, and turns
    // HWY_DYNAMIC_DISPATCH into a direct call (no function pointer
    // indirection), letting the compiler inline dispatch wrappers.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .define("HWY_STATIC_DEFINE", None)
        .define("HWY_COMPILE_ONLY_STATIC", None)
        .include(&highway_dir)
        .files(&hwy_sources)
        .compile("hwy");

    // Build our shim (uses foreach_target, but with HWY_COMPILE_ONLY_STATIC
    // it compiles once for the native target only).
    let mut shim = cc::Build::new();
    shim.cpp(true)
        .std("c++17")
        .define("HWY_STATIC_DEFINE", None)
        .define("HWY_COMPILE_ONLY_STATIC", None)
        .include(&highway_dir)
        .include(&shim_dir)
        .file(shim_dir.join("omnidsp_hwy.cpp"));

    // Native CPU tuning — the compiler generates code specifically for this
    // CPU. Combined with HWY_COMPILE_ONLY_STATIC, Highway targets the best
    // ISA the compiler selects (AVX2/AVX-512 on x86, NEON on ARM).
    shim.flag_if_supported("-mcpu=native");
    shim.flag_if_supported("-mtune=native");
    shim.flag_if_supported("-funroll-loops");

    shim.compile("omnidsp_hwy");

    println!("cargo::rerun-if-changed=shim/");
    println!("cargo::rerun-if-changed=highway/hwy/");
}
