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
    // HWY_COMPILE_ONLY_STATIC: compile once for the target selected by
    // compiler flags (-mcpu=native). HWY_DYNAMIC_DISPATCH becomes a
    // direct call (no function pointer indirection).
    //
    // HWY_DISABLE_ATTR: prevent HWY_BEFORE_NAMESPACE from attaching
    // target-specific attributes (#pragma clang attribute) to functions.
    // On Apple clang, these attributes conflict with the loop unroller,
    // causing "unsupported transformation ordering" warnings and preventing
    // all loop unrolling. With -mcpu=native the compiler already knows the
    // target, so the attributes are redundant.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .define("HWY_STATIC_DEFINE", None)
        .define("HWY_COMPILE_ONLY_STATIC", None)
        .define("HWY_DISABLE_ATTR", None)
        .include(&highway_dir)
        .files(&hwy_sources)
        .compile("hwy");

    // Build our shim.
    let mut shim = cc::Build::new();
    shim.cpp(true)
        .std("c++17")
        .define("HWY_STATIC_DEFINE", None)
        .define("HWY_COMPILE_ONLY_STATIC", None)
        .define("HWY_DISABLE_ATTR", None)
        .include(&highway_dir)
        .include(&shim_dir)
        .file(shim_dir.join("omnidsp_hwy.cpp"));

    // Native CPU tuning — the compiler generates code specifically for this
    // CPU. Combined with HWY_COMPILE_ONLY_STATIC + HWY_DISABLE_ATTR, the
    // compiler has full optimization freedom (including loop unrolling)
    // without conflicting Highway target pragmas.
    shim.flag_if_supported("-mcpu=native");
    shim.flag_if_supported("-mtune=native");
    shim.flag_if_supported("-funroll-loops");

    shim.compile("omnidsp_hwy");

    println!("cargo::rerun-if-changed=shim/");
    println!("cargo::rerun-if-changed=highway/hwy/");
}
